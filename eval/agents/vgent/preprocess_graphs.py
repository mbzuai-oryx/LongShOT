#!/usr/bin/env python3
"""Pre-build Vgent semantic graphs for all videos in a directory.

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python agents/vgent/preprocess_graphs.py \
        --video_dir ./data/videos \
        --graph_dir agents/vgent/graphs \
        --gpus 0,1,2,3

One VLM per GPU. Bypasses HF processor entirely — patchify + normalize on GPU.
Cached prompt tokenization. Prefetches next video via ffmpeg while GPU generates.
Resumes automatically.
"""

import argparse
import json
import os
import pickle
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_PATH = str(Path(__file__).resolve().parent.parent / "repos" / "Vgent")
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)


def _load_gpu(gpu_id, vlm_name, embed_name):
    from vgent_models.qwenvl import load_model
    from transformers import AutoModel, AutoTokenizer

    device = f"cuda:{gpu_id}"
    _, vlm, processor, _ = load_model(vlm_name, device=device)
    vlm.eval()
    embed_tok = AutoTokenizer.from_pretrained(embed_name)
    embed_model = AutoModel.from_pretrained(embed_name).to(device)
    embed_model.eval()
    return vlm, processor, embed_model, embed_tok, device


def _load_video_cpu(video_path, fps, chunk_size, total_pixels):
    """Load video via ffmpeg with resize baked in. Returns CPU float tensor."""
    from vgent_models.utils import read_video_resized_ffmpeg
    return read_video_resized_ffmpeg(video_path, fps, chunk_size, total_pixels)


def _extract_chunks(split_video, vlm, processor, device, batch_size, pbar):
    """VLM entity extraction — batched generation on GPU, no HF processor."""
    from vgent_models.qwenvl import generate_fast_batch
    from vgent_utils.prompts import GRAPH_PROMPT

    total = len(split_video)

    # Subsample all chunks to 8 frames upfront
    chunks_8f = []
    for chunk in split_video:
        if len(chunk) > 8:
            idx = torch.linspace(0, len(chunk) - 1, 8).round().long()
            chunks_8f.append(chunk[idx])
        else:
            chunks_8f.append(chunk)

    chunk_data = []
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_frames = chunks_8f[batch_start:batch_end]

        try:
            responses = generate_fast_batch(vlm, processor, batch_frames,
                                             GRAPH_PROMPT, device, max_new_tokens=256)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # OOM — fall back to batch_size=1
            torch.cuda.empty_cache()
            from vgent_models.qwenvl import generate_fast
            responses = []
            for frames in batch_frames:
                try:
                    r = generate_fast(vlm, processor, frames, GRAPH_PROMPT,
                                       device, max_new_tokens=256)
                    responses.append(r)
                except Exception:
                    responses.append("")

        for i, resp in enumerate(responses):
            entities, actions, scenes = [], [], []
            try:
                info = json.loads(resp.replace("```json", "").replace("```", "").strip())
                entities = [f"{e['entity name']}, {e['description']}"
                            for e in info.get("entities", [])
                            if "entity name" in e and "description" in e]
                actions = [f"{e['entity name']}, {e['action description']}"
                           for e in info.get("actions", [])
                           if "entity name" in e and "action description" in e]
                scenes = [s["location"] for s in info.get("scenes", []) if "location" in s]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
            chunk_data.append((batch_start + i, entities, actions, scenes))

        pbar.update(batch_end - batch_start)

    return chunk_data


def _build_graph(chunk_data, embed_model, embed_tok, device):
    """Batch entity dedup on GPU via single embedding pass + greedy clustering."""
    import networkx as nx

    video_graph = nx.DiGraph()
    entity_graph = defaultdict(set)

    for idx, entities, actions, scenes in chunk_data:
        video_graph.add_node(idx, actions=actions, scenes=scenes,
                             entities=entities, subtitles=None)

    all_items = []
    for idx, entities, actions, scenes in chunk_data:
        for item in entities + actions + scenes:
            all_items.append((item, idx))

    if not all_items:
        return video_graph, entity_graph

    texts = [t for t, _ in all_items]
    encoded = embed_tok(texts, padding=True, truncation=True, return_tensors='pt')
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        emb = embed_model(**encoded)[0][:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)

    cluster_names = []
    cluster_emb_tensor = None

    for i, (item_text, chunk_idx) in enumerate(all_items):
        ename = item_text.split(",")[0].lower()

        if cluster_emb_tensor is None:
            cluster_names.append(ename)
            cluster_emb_tensor = emb[i:i+1]
            entity_graph[ename].add(chunk_idx)
            continue

        sims = (emb[i:i+1] @ cluster_emb_tensor.T).squeeze(0)
        best_j = sims.argmax().item()

        if sims[best_j] > 0.7:
            entity_graph[cluster_names[best_j]].add(chunk_idx)
        else:
            cluster_names.append(ename)
            cluster_emb_tensor = torch.cat([cluster_emb_tensor, emb[i:i+1]], dim=0)
            entity_graph[ename].add(chunk_idx)

    for key, chunk_set in entity_graph.items():
        if len(chunk_set) > 1:
            chunks = list(chunk_set)
            for ci in chunks:
                for cj in chunks:
                    if ci != cj:
                        video_graph.add_edge(ci, cj, label=key)

    return video_graph, entity_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--graph_dir", default="agents/vgent/graphs")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Chunks per batched VLM forward pass")
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--total_pixels", type=int, default=16384)
    parser.add_argument("--n_retrieval", type=int, default=20)
    parser.add_argument("--vlm", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--embedding", default="BAAI/bge-large-en-v1.5")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    os.makedirs(args.graph_dir, exist_ok=True)

    video_files = []
    for root, _, files in os.walk(args.video_dir):
        for f in sorted(files):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_files.append(os.path.join(root, f))

    pending = []
    for vf in video_files:
        if not os.path.exists(os.path.join(args.graph_dir, f"{Path(vf).stem}.pkl")):
            pending.append(vf)

    print(f"{'='*60}")
    print(f"  Vgent Graph Preprocessing")
    print(f"  Videos: {len(video_files)} total, {len(video_files)-len(pending)} done, "
          f"{len(pending)} pending")
    print(f"  GPUs: {gpu_ids} ({len(gpu_ids)} workers)")
    print(f"{'='*60}\n")

    if not pending:
        print("All done!")
        return

    print(f"Loading models on {len(gpu_ids)} GPUs...")
    workers = {}
    for gid in gpu_ids:
        t0 = time.time()
        workers[gid] = _load_gpu(gid, args.vlm, args.embedding)
        print(f"  GPU {gid}: ready ({time.time() - t0:.1f}s)")
    print()

    gpu_queues = {gid: [] for gid in gpu_ids}
    for i, vf in enumerate(pending):
        gpu_queues[gpu_ids[i % len(gpu_ids)]].append(vf)

    pbar_total = tqdm(total=len(pending), desc="Videos", position=0, unit="vid")
    chunk_pbars = {}
    for i, gid in enumerate(gpu_ids):
        chunk_pbars[gid] = tqdm(total=0, desc=f"GPU{gid}", position=1+i, unit="ch",
                                leave=True, dynamic_ncols=True)

    t_start = time.time()
    stats_lock = threading.Lock()
    stats = {"built": 0, "short": 0, "error": 0, "chunks": 0}

    # Shared pool for CPU video loading (ffmpeg) — many workers since it's I/O bound
    prefetch_pool = ThreadPoolExecutor(max_workers=len(gpu_ids) * 3)

    def gpu_worker(gid):
        vlm, processor, embed_model, embed_tok, device = workers[gid]
        my_videos = gpu_queues[gid]
        pbar = chunk_pbars[gid]
        min_frames = args.chunk_size * args.n_retrieval

        # Prefetch first 3 videos
        futures = {}
        for j in range(min(3, len(my_videos))):
            futures[j] = prefetch_pool.submit(
                _load_video_cpu, my_videos[j], args.fps, args.chunk_size, args.total_pixels)
        next_prefetch = min(3, len(my_videos))

        for vi, vpath in enumerate(my_videos):
            vname = Path(vpath).stem
            graph_file = os.path.join(args.graph_dir, f"{vname}.pkl")

            if vi in futures:
                future = futures.pop(vi)
            else:
                future = prefetch_pool.submit(
                    _load_video_cpu, vpath, args.fps, args.chunk_size, args.total_pixels)

            if next_prefetch < len(my_videos):
                futures[next_prefetch] = prefetch_pool.submit(
                    _load_video_cpu, my_videos[next_prefetch],
                    args.fps, args.chunk_size, args.total_pixels)
                next_prefetch += 1

            try:
                video, num_frames, _ = future.result(timeout=180)
            except Exception as e:
                pbar.set_description(f"GPU{gid} ERR {vname[:12]}")
                with stats_lock:
                    stats["error"] += 1
                pbar_total.update(1)
                continue

            if num_frames < min_frames:
                with stats_lock:
                    stats["short"] += 1
                pbar_total.update(1)
                continue

            split_video = torch.split(video, args.chunk_size, dim=0)
            n_chunks = len(split_video)

            pbar.reset(total=n_chunks)
            pbar.set_description(f"GPU{gid} {vname[:14]}")

            with torch.inference_mode():
                chunk_data = _extract_chunks(split_video, vlm, processor, device,
                                             args.batch_size, pbar)
                video_graph, entity_graph = _build_graph(
                    chunk_data, embed_model, embed_tok, device)

            del video, split_video, chunk_data

            pickle.dump({"video_graph": video_graph, "entity_graph": entity_graph},
                        open(graph_file, "wb"))

            with stats_lock:
                stats["built"] += 1
                stats["chunks"] += n_chunks
            pbar_total.update(1)
            pbar.set_description(f"GPU{gid} done {vname[:11]}")

    threads = []
    for gid in gpu_ids:
        t = threading.Thread(target=gpu_worker, args=(gid,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    prefetch_pool.shutdown(wait=False)
    for pb in chunk_pbars.values():
        pb.close()
    pbar_total.close()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Built: {stats['built']} ({stats['chunks']} chunks)")
    print(f"  Short: {stats['short']}, Errors: {stats['error']}")
    if stats['built'] > 0:
        print(f"  Throughput: {stats['chunks']/elapsed:.1f} chunks/s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
