"""Agent inference loop, extracted from eval.py.

`run_agent_inference` owns the per-agent worker pool, dispatch logic, and
output-file appending. eval.py just calls it; everything agent-specific
lives in this module so adding a new agent doesn't touch eval.py.

Uses dedicated worker threads — one per GPU replica — so each thread only
ever touches a single CUDA device.  Samples are round-robin partitioned
across workers to keep load balanced.
"""

import json
import threading
from typing import Any, Callable, Dict, List, Optional

from filelock import FileLock
from tqdm import tqdm

from agents.base import BaseAgent


def run_agent_inference(
    agent: BaseAgent,
    samples: List[Dict[str, Any]],
    video_path_map: Dict[str, str],
    output_file: str,
    *,
    extract_question: Callable[[Dict[str, Any]], str],
    inject_response: Callable[[Dict[str, Any], str, List[Any]], None],
    concurrency_override: Optional[int] = None,
    agent_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Run an agent across a sample list with one dedicated thread per replica.

    Resolution order for parallelism:
      1. `concurrency_override` (e.g. CLI --num_workers > 1)
      2. `agent.recommended_concurrency` (e.g. multi-GPU pipelines self-report)
      3. `agent_config["inference"]["agent_concurrency"]`
      4. 1
    """
    agent_config = agent_config or {}

    if concurrency_override and concurrency_override > 1:
        concurrency = concurrency_override
    else:
        recommended = getattr(agent, "recommended_concurrency", None)
        config_value = agent_config.get("inference", {}).get("agent_concurrency")
        concurrency = recommended or config_value or 1

    print(f"Agent concurrency: {concurrency}")

    pbar = tqdm(total=len(samples), desc=f"Agent ({agent.name})")
    pbar_lock = threading.Lock()

    def _run_one(sample: Dict[str, Any]) -> None:
        video_id = sample.get("video_id", "")
        video_path = video_path_map.get(video_id)
        if not video_path:
            print(f"\n[WARNING] Video not found: {video_id}")
            return
        try:
            question = extract_question(sample)
            response = agent.process_sample(video_path, question)
            inject_response(
                sample,
                agent.adapt_response_for_eval(response),
            )
        except Exception as e:
            print(f"\n[ERROR] Agent failed on {sample.get('sample_id')}: {e}")
            return
        with FileLock(f"{output_file}.lock"):
            with open(output_file, "a") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if concurrency <= 1:
        for s in samples:
            _run_one(s)
            with pbar_lock:
                pbar.update(1)
        pbar.close()
        return

    # Partition samples round-robin across workers so each worker's
    # thread always acquires the same replica (FIFO queue with 1 item
    # per worker means each worker pins to one GPU).
    partitions: List[List[Dict[str, Any]]] = [[] for _ in range(concurrency)]
    for i, s in enumerate(samples):
        partitions[i % concurrency].append(s)

    stop_event = threading.Event()

    def _worker(partition: List[Dict[str, Any]]) -> None:
        for s in partition:
            if stop_event.is_set():
                break
            _run_one(s)
            with pbar_lock:
                pbar.update(1)

    threads = []
    for p in partitions:
        t = threading.Thread(target=_worker, args=(p,), daemon=True)
        threads.append(t)
        t.start()

    try:
        for t in threads:
            while t.is_alive():
                t.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\n[runner] KeyboardInterrupt — stopping agent workers")
        stop_event.set()
        for t in threads:
            t.join(timeout=5.0)
        raise
    finally:
        pbar.close()
