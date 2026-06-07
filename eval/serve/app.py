"""FastAPI application, CLI, and endpoints for transformers_serve."""

import argparse
import asyncio
import time
import traceback
import uuid

import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Monkey-patch DynamicCache for models expecting older transformers API
# (e.g. MiniCPM-o-4.5 expects .seen_tokens, removed in transformers >=4.52)
from transformers import DynamicCache
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

from serve.schemas import ChatRequest, ChatResponse, PreloadRequest
from serve.engine import InferenceEngine
from serve.media import _media_cache, _media_cache_lock, cache_stats as media_cache_stats


def parse_args():
    p = argparse.ArgumentParser(description="Transformers OpenAI-compatible server")
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel (device_map auto)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-video-frames", type=int, default=0, help="Max frames to sample (0=let model decide)")
    p.add_argument("--compile", action="store_true", default=True, help="torch.compile the model (default: enabled)")
    p.add_argument("--no-compile", action="store_false", dest="compile", help="Disable torch.compile")
    p.add_argument("--omni", action="store_true", help="Enable omni mode (audio+video)")
    p.add_argument("--replicas", type=int, default=0,
                   help="Number of model replicas for data parallelism (0=auto: one per GPU when tp=1)")
    p.add_argument("--media-cache-size", type=int, default=0,
                   help="Max cached media items (0=auto: 50 per replica)")
    p.add_argument("--batch-max-size", type=int, default=8,
                   help="Max keyed batch size per replica")
    p.add_argument("--batch-max-wait-ms", type=int, default=50,
                   help="Max wait to accumulate same-key requests into a batch")
    p.add_argument("--batch-starvation-ms", type=int, default=250,
                   help="Max time a ready request can wait before it overrides size-based key selection")
    p.add_argument("--cpu-prep-workers", type=int, default=0,
                   help="CPU workers for tokenizer/processor preparation (0=auto)")
    p.add_argument("--media-workers", type=int, default=0,
                   help="CPU workers for media preload/decode (0=auto)")
    p.add_argument("--metrics-log-every", type=int, default=50,
                   help="Print a metrics line after every N successful requests")
    return p.parse_args()


def create_app(args) -> FastAPI:
    engine = InferenceEngine(args)

    @asynccontextmanager
    async def lifespan(_app):
        await engine.start_workers()
        yield
        await engine.stop_workers()

    app = FastAPI(title="Transformers Serve", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return engine.health_snapshot()

    @app.get("/debug/forward")
    async def debug_forward():
        """Raw forward pass test — bypasses generate() entirely."""
        replica = engine.replicas[0]
        loop = asyncio.get_running_loop()
        def _test():
            torch.cuda.set_device(replica.device_id)
            tok = replica.tokenizer
            model = replica.model
            text = "The capital of France is"
            ids = tok.encode(text, return_tensors="pt").to(model.device)
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            with torch.inference_mode():
                out_full = model(input_ids=ids)
                logits_full = out_full.logits[0, -1]
                top5_full = [(tok.decode([t.item()]), logits_full[t].item()) for t in logits_full.topk(5).indices]

                embeds = model.model.embed_tokens(ids)
                inner_out = model.model(inputs_embeds=embeds)
                logits_inner = model.lm_head(inner_out[0])[0, -1]
                top5_inner = [(tok.decode([t.item()]), logits_inner[t].item()) for t in logits_inner.topk(5).indices]

            return {
                "input": text,
                "input_ids": ids[0].tolist()[-5:],
                "full_model_top5": top5_full,
                "inner_llm_top5": top5_inner,
                "lm_head_norm": model.lm_head.weight.float().norm().item(),
                "embed_norm": model.model.embed_tokens.weight.float().norm().item(),
                "layer0_norm": model.model.layers[0].input_layernorm.weight.float().norm().item(),
            }
        return await loop.run_in_executor(engine._executors[0], _test)

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": args.model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "transformers",
            }],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        try:
            text = await engine.submit(request)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[ERROR] Inference failed: {e}\n{tb}", flush=True)
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "type": "server_error"}},
            )

        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model or args.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    @app.post("/v1/preload")
    async def preload(request: PreloadRequest):
        """Pre-warm model-aware media caches for a list of file paths."""
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()

        tasks = [
            loop.run_in_executor(engine._media_pool, engine.preload_path, p, request.media_type)
            for p in request.paths
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = sum(1 for r in results if isinstance(r, Exception))
        elapsed = time.monotonic() - t0
        return {
            "status": "ok",
            "cached": len(request.paths) - errors,
            "errors": errors,
            "elapsed_s": round(elapsed, 2),
        }

    @app.get("/v1/cache/stats")
    async def cache_stats_endpoint():
        stats = media_cache_stats()
        with _media_cache_lock:
            stats["keys"] = list(_media_cache.keys())[:50]
        return stats

    @app.get("/v1/metrics")
    async def metrics_endpoint():
        return engine.metrics_snapshot()

    return app


def main():
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
