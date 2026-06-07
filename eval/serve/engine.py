"""Inference engine with model-aware preload, keyed ready-queue batching, and metrics."""

import asyncio
import os
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch

from serve.schemas import ChatRequest
from serve.models import ModelBackend
from serve.media import _set_media_cache_max, cache_stats as media_cache_stats

# Set before any CUDA allocation to reduce fragmentation and improve memory reuse
# across variable-length inference requests.
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _timed_call(fn, *args):
    """Run a callable and return (result, elapsed_seconds)."""
    t0 = time.monotonic()
    return fn(*args), time.monotonic() - t0


@dataclass
class RequestItem:
    request: ChatRequest
    future: asyncio.Future
    batch_key: Any
    submitted: float = field(default_factory=time.monotonic)
    preload_started: float | None = None
    preload_finished: float | None = None
    ready_at: float | None = None
    prepare_started: float | None = None
    prepared_at: float | None = None


@dataclass
class StageAccumulator:
    occurrences: int = 0
    requests: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def add(self, elapsed_s: float, requests: int = 1):
        self.occurrences += 1
        self.requests += requests
        self.total_s += elapsed_s
        self.max_s = max(self.max_s, elapsed_s)

    def snapshot(self) -> dict[str, float | int]:
        avg_batch_ms = (self.total_s / self.occurrences * 1000) if self.occurrences else 0.0
        avg_req_ms = (self.total_s / self.requests * 1000) if self.requests else 0.0
        return {
            "occurrences": self.occurrences,
            "requests": self.requests,
            "total_s": round(self.total_s, 3),
            "avg_batch_ms": round(avg_batch_ms, 2),
            "avg_request_ms": round(avg_req_ms, 2),
            "max_ms": round(self.max_s * 1000, 2),
        }


class EngineMetrics:
    """Lightweight in-process metrics for throughput tuning."""

    def __init__(self, num_replicas: int):
        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.batches = 0
        self.batch_size_hist = Counter()
        self.stages = {
            "preload": StageAccumulator(),
            "queue_wait": StageAccumulator(),
            "prepare": StageAccumulator(),
            "generate": StageAccumulator(),
            "end_to_end": StageAccumulator(),
        }
        self.per_replica = [
            {
                "requests": 0,
                "batches": 0,
                "batch_size_hist": Counter(),
                "prepare": StageAccumulator(),
                "generate": StageAccumulator(),
                "queue_wait": StageAccumulator(),
            }
            for _ in range(num_replicas)
        ]

    def record_batch(self, replica_idx: int, batch_size: int):
        self.batches += 1
        self.batch_size_hist[batch_size] += 1
        self.per_replica[replica_idx]["batches"] += 1
        self.per_replica[replica_idx]["batch_size_hist"][batch_size] += 1

    def record_stage(self, stage: str, elapsed_s: float, requests: int = 1, replica_idx: int | None = None):
        self.stages[stage].add(elapsed_s, requests=requests)
        if replica_idx is not None and stage in self.per_replica[replica_idx]:
            self.per_replica[replica_idx][stage].add(elapsed_s, requests=requests)

    def snapshot(self, *, current: dict[str, int], cache: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        return {
            "requests": {
                "submitted": self.submitted,
                "completed": self.completed,
                "failed": self.failed,
            },
            "batches": {
                "total": self.batches,
                "sizes": dict(sorted(self.batch_size_hist.items())),
            },
            "stages": {name: acc.snapshot() for name, acc in self.stages.items()},
            "current": current,
            "cache": cache,
            "config": config,
            "replicas": [
                {
                    "idx": idx,
                    "requests": replica["requests"],
                    "batches": replica["batches"],
                    "batch_sizes": dict(sorted(replica["batch_size_hist"].items())),
                    "queue_wait": replica["queue_wait"].snapshot(),
                    "prepare": replica["prepare"].snapshot(),
                    "generate": replica["generate"].snapshot(),
                }
                for idx, replica in enumerate(self.per_replica)
            ],
        }


class InferenceEngine:
    """Inference engine with keyed ready queues and separate CPU/GPU pipeline stages."""

    def __init__(self, args):
        self.args = args
        num_gpus = torch.cuda.device_count()
        if num_gpus <= 0:
            raise RuntimeError("No CUDA devices available for transformers_serve")

        self.tp = max(1, int(args.tp))
        if args.replicas > 0:
            num_replicas = args.replicas
        else:
            num_replicas = max(1, num_gpus // self.tp)

        if self.tp > 1 and num_replicas > 1:
            raise ValueError(
                "transformers_serve currently supports tp>1 only with a single replica per process"
            )
        if num_replicas * self.tp > num_gpus:
            raise ValueError(
                f"Requested replicas={num_replicas}, tp={self.tp}, but only {num_gpus} visible GPU(s)"
            )

        self.batch_max_size = max(1, int(args.batch_max_size))
        self.batch_max_wait_ms = max(0, int(args.batch_max_wait_ms))
        self.batch_starvation_ms = max(self.batch_max_wait_ms, int(args.batch_starvation_ms))
        self.metrics_log_every = max(1, int(args.metrics_log_every))

        cache_size = args.media_cache_size if args.media_cache_size > 0 else 50 * num_replicas
        _set_media_cache_max(cache_size)

        self.replica_groups = [
            list(range(i * self.tp, (i + 1) * self.tp))
            for i in range(num_replicas)
        ]
        topology = "tp" if self.tp > 1 else "dp"
        print(
            f"InferenceEngine: replicas={num_replicas}, tp={self.tp}, topology={topology}, "
            f"device_groups={self.replica_groups}, media_cache={cache_size}",
            flush=True,
        )

        force_single = num_replicas > 1 and self.tp == 1
        self.replicas: list[ModelBackend] = [
            ModelBackend(args, device_ids=group, force_single_device=force_single)
            for group in self.replica_groups
        ]

        for replica in self.replicas:
            replica.warmup()

        self._gpu_executors: list[ThreadPoolExecutor] = []
        for replica in self.replicas:
            primary = replica.primary_device_id
            self._gpu_executors.append(ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"gpu{primary}",
                initializer=lambda d=primary: torch.cuda.set_device(d),
            ))

        cpu_workers = args.cpu_prep_workers if args.cpu_prep_workers > 0 else max(4, num_replicas * 4)
        media_workers = args.media_workers if args.media_workers > 0 else max(4, num_replicas * 4)
        self._cpu_prepare_pool = ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="cpu-prepare")
        self._media_pool = ThreadPoolExecutor(max_workers=media_workers, thread_name_prefix="media")

        self._ready_condition = asyncio.Condition()
        self._ready_buckets: dict[Any, deque[RequestItem]] = {}
        self._leased_batch_keys: set[Any] = set()
        self._preload_tasks: set[asyncio.Task] = set()
        self._worker_tasks: list[asyncio.Task] = []

        self._metrics = EngineMetrics(num_replicas)
        self._request_count = 0

    async def start_workers(self):
        for idx in range(len(self.replicas)):
            self._worker_tasks.append(asyncio.create_task(self._batching_worker(idx)))
        print(
            f"Started {len(self._worker_tasks)} batching worker(s) "
            f"(batch_size≤{self.batch_max_size}, wait≤{self.batch_max_wait_ms}ms)",
            flush=True,
        )

    async def stop_workers(self):
        for task in self._worker_tasks:
            task.cancel()
        for task in list(self._preload_tasks):
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        await asyncio.gather(*self._preload_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        self._preload_tasks.clear()

        self._cpu_prepare_pool.shutdown(wait=False, cancel_futures=True)
        self._media_pool.shutdown(wait=False, cancel_futures=True)
        for executor in self._gpu_executors:
            executor.shutdown(wait=False, cancel_futures=True)

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "replicas": len(self.replicas),
            "tp": self.tp,
            "gpus": [replica.primary_device_id for replica in self.replicas],
            "device_groups": self.replica_groups,
            "ready_items": self._ready_item_count(),
            "ready_keys": len(self._ready_buckets),
            "leased_keys": len(self._leased_batch_keys),
            "pending_preloads": len(self._preload_tasks),
        }

    def metrics_snapshot(self) -> dict[str, Any]:
        return self._metrics.snapshot(
            current={
                "ready_items": self._ready_item_count(),
                "ready_keys": len(self._ready_buckets),
                "leased_keys": len(self._leased_batch_keys),
                "pending_preloads": len(self._preload_tasks),
            },
            cache=media_cache_stats(),
            config={
                "tp": self.tp,
                "replicas": len(self.replicas),
                "device_groups": self.replica_groups,
                "batch_max_size": self.batch_max_size,
                "batch_max_wait_ms": self.batch_max_wait_ms,
                "batch_starvation_ms": self.batch_starvation_ms,
            },
        )

    def preload_path(self, path: str, media_type: str):
        """Expose model-aware single-path preload to the FastAPI layer."""
        self.replicas[0].preload_path(path, media_type)

    async def submit(self, request: ChatRequest) -> str:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = RequestItem(
            request=request,
            future=future,
            batch_key=self.replicas[0].batch_key(request),
        )
        self._metrics.submitted += 1

        preload_task = asyncio.create_task(self._preload_and_enqueue(item))
        self._preload_tasks.add(preload_task)
        preload_task.add_done_callback(self._on_preload_done)

        return await future

    def _on_preload_done(self, task: asyncio.Task):
        self._preload_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            print(f"[ERROR] preload task failed: {exc}", flush=True)

    async def _preload_and_enqueue(self, item: RequestItem):
        loop = asyncio.get_running_loop()
        item.preload_started = time.monotonic()
        try:
            await loop.run_in_executor(self._media_pool, self.replicas[0].preload_request, item.request)
        except Exception as e:
            print(f"  [PRELOAD] model preload failed: {e}", flush=True)
        finally:
            item.preload_finished = time.monotonic()
            self._metrics.record_stage(
                "preload",
                item.preload_finished - item.preload_started,
                requests=1,
            )

        item.ready_at = time.monotonic()
        async with self._ready_condition:
            bucket = self._ready_buckets.setdefault(item.batch_key, deque())
            bucket.append(item)
            self._ready_condition.notify_all()

    async def _batching_worker(self, replica_idx: int):
        loop = asyncio.get_running_loop()
        replica = self.replicas[replica_idx]
        gpu_executor = self._gpu_executors[replica_idx]

        try:
            while True:
                batch_key, batch = await self._acquire_batch()
                try:
                    batch_started = time.monotonic()
                    self._metrics.record_batch(replica_idx, len(batch))
                    for item in batch:
                        ready_at = item.ready_at or item.submitted
                        self._metrics.record_stage(
                            "queue_wait",
                            max(0.0, batch_started - ready_at),
                            requests=1,
                            replica_idx=replica_idx,
                        )

                    if len(batch) > 1 and replica.supports_batch_inference():
                        handled = await self._run_specialized_batch(
                            loop, replica_idx, replica, gpu_executor, batch
                        )
                        if handled:
                            continue

                    await self._run_pipelined_sequential(
                        loop, replica_idx, replica, gpu_executor, batch
                    )
                finally:
                    await self._release_batch_key(batch_key)
        except asyncio.CancelledError:
            raise

    async def _acquire_batch(self) -> tuple[Any, list[RequestItem]]:
        async with self._ready_condition:
            while True:
                batch_key = self._select_batch_key_locked()
                if batch_key is None:
                    await self._ready_condition.wait()
                    continue

                self._leased_batch_keys.add(batch_key)
                batch = [self._pop_bucket_item_locked(batch_key)]
                base_wait = self.batch_max_wait_ms / 1000
                wait_time = base_wait * 2 if self._ready_buckets.get(batch_key) else base_wait
                deadline = time.monotonic() + wait_time

                while len(batch) < self.batch_max_size:
                    bucket = self._ready_buckets.get(batch_key)
                    if bucket:
                        batch.append(self._pop_bucket_item_locked(batch_key))
                        continue

                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        await asyncio.wait_for(self._ready_condition.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break

                return batch_key, batch

    async def _release_batch_key(self, batch_key: Any):
        async with self._ready_condition:
            self._leased_batch_keys.discard(batch_key)
            self._ready_condition.notify_all()

    def _select_batch_key_locked(self) -> Any | None:
        now = time.monotonic()
        starvation_cutoff = now - (self.batch_starvation_ms / 1000)
        candidates = []
        starved = []

        for key, bucket in self._ready_buckets.items():
            if not bucket or key in self._leased_batch_keys:
                continue
            head_ready = bucket[0].ready_at or bucket[0].submitted
            record = (key, len(bucket), head_ready)
            if head_ready <= starvation_cutoff:
                starved.append(record)
            else:
                candidates.append(record)

        if starved:
            return min(starved, key=lambda row: row[2])[0]
        if not candidates:
            return None
        return max(candidates, key=lambda row: (row[1], -row[2]))[0]

    def _pop_bucket_item_locked(self, batch_key: Any) -> RequestItem:
        bucket = self._ready_buckets[batch_key]
        item = bucket.popleft()
        if not bucket:
            del self._ready_buckets[batch_key]
        return item

    async def _prepare_single(self, loop, replica: ModelBackend, item: RequestItem):
        item.prepare_started = time.monotonic()
        try:
            prepared, elapsed = await loop.run_in_executor(
                self._cpu_prepare_pool, _timed_call, replica.prepare, item.request
            )
            item.prepared_at = time.monotonic()
            return item, prepared, elapsed, None
        except Exception as e:
            return item, None, 0.0, e

    async def _generate_single(self, loop, gpu_executor, replica: ModelBackend, item: RequestItem, prepared):
        try:
            result, elapsed = await loop.run_in_executor(
                gpu_executor, _timed_call, replica.generate_prepared, prepared
            )
            return item, result, elapsed, None
        except Exception as e:
            return item, None, 0.0, e

    async def _run_specialized_batch(self, loop, replica_idx: int, replica: ModelBackend, gpu_executor, batch: list[RequestItem]) -> bool:
        requests = [item.request for item in batch]
        try:
            prepared_batch, prep_elapsed = await loop.run_in_executor(
                self._cpu_prepare_pool, _timed_call, replica.prepare_batch, requests
            )
        except Exception as e:
            print(f"  [BATCH] prepare_batch failed on replica {replica_idx}: {e}", flush=True)
            return False

        if isinstance(prepared_batch, list):
            return False

        self._metrics.record_stage("prepare", prep_elapsed, requests=len(batch), replica_idx=replica_idx)

        try:
            results, gen_elapsed = await loop.run_in_executor(
                gpu_executor, _timed_call, replica.generate_batch_prepared, prepared_batch
            )
        except Exception as e:
            print(f"  [BATCH] generate_batch failed on replica {replica_idx}: {e}", flush=True)
            return False

        if len(results) != len(batch):
            print(
                f"  [BATCH] replica {replica_idx} returned {len(results)} result(s) for {len(batch)} request(s)",
                flush=True,
            )
            return False

        self._metrics.record_stage("generate", gen_elapsed, requests=len(batch), replica_idx=replica_idx)
        for item, result in zip(batch, results):
            if isinstance(result, Exception):
                self._complete_failure(item, result)
            else:
                self._complete_success(item, result, replica_idx)
        return True

    async def _run_pipelined_sequential(self, loop, replica_idx: int, replica: ModelBackend, gpu_executor, batch: list[RequestItem]):
        prepare_tasks = [asyncio.create_task(self._prepare_single(loop, replica, item)) for item in batch]
        generate_tasks = []

        for task in asyncio.as_completed(prepare_tasks):
            item, prepared, prep_elapsed, prep_exc = await task
            if prep_exc is not None:
                self._complete_failure(item, prep_exc)
                continue
            self._metrics.record_stage("prepare", prep_elapsed, requests=1, replica_idx=replica_idx)
            generate_tasks.append(
                asyncio.create_task(self._generate_single(loop, gpu_executor, replica, item, prepared))
            )

        for task in asyncio.as_completed(generate_tasks):
            item, result, gen_elapsed, gen_exc = await task
            if gen_exc is not None:
                self._complete_failure(item, gen_exc)
                continue
            self._metrics.record_stage("generate", gen_elapsed, requests=1, replica_idx=replica_idx)
            self._complete_success(item, result, replica_idx)

    def _complete_success(self, item: RequestItem, result: str, replica_idx: int):
        if not item.future.done():
            item.future.set_result(result)
        self._metrics.completed += 1
        self._metrics.per_replica[replica_idx]["requests"] += 1
        self._metrics.record_stage("end_to_end", time.monotonic() - item.submitted, requests=1)
        self._request_count += 1
        if self._request_count % self.metrics_log_every == 0:
            self._log_metrics()

    def _complete_failure(self, item: RequestItem, exc: Exception):
        if not item.future.done():
            item.future.set_exception(exc)
        self._metrics.failed += 1

    def _ready_item_count(self) -> int:
        return sum(len(bucket) for bucket in self._ready_buckets.values())

    def _log_metrics(self):
        metrics = self.metrics_snapshot()
        batches = metrics["batches"]["sizes"]
        batch_total = sum(size * count for size, count in batches.items())
        batch_occurrences = sum(batches.values())
        avg_batch = (batch_total / batch_occurrences) if batch_occurrences else 0.0
        print(
            "  [METRICS] "
            f"completed={metrics['requests']['completed']} failed={metrics['requests']['failed']} "
            f"ready={metrics['current']['ready_items']} preloads={metrics['current']['pending_preloads']} "
            f"avg_batch={avg_batch:.2f} "
            f"prepare={metrics['stages']['prepare']['avg_request_ms']:.1f}ms/req "
            f"generate={metrics['stages']['generate']['avg_request_ms']:.1f}ms/req "
            f"cache_hit={metrics['cache']['hit_rate']:.1%}",
            flush=True,
        )
