#!/usr/bin/env python3
"""Pre-warm the Video-RAG ASR cache using faster-whisper (batched).

faster-whisper's `BatchedInferencePipeline` runs multiple audio chunks in
parallel through CTranslate2, typically 4-8x faster than the upstream
HuggingFace Whisper loop. The runtime adapter (`pipeline.py`) reads cached
transcripts as plain text per chunk, so the choice of ASR backend at
preprocess time is invisible to it.

Usage:
    python preprocess.py --video_dir /path/to/videos \\
        --gpus 0,1,2,3,4,5,6,7 --workers_per_gpu 1 --batch_size 16
"""

import argparse
import os
import time
from multiprocessing import Process, Queue
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}
DEFAULT_CACHE = Path(__file__).parent / "cache" / "audio"


def find_videos(video_dir: str) -> list:
    out = []
    for root, _, files in os.walk(video_dir):
        for f in files:
            if Path(f).suffix.lower() in VIDEO_EXTS:
                out.append(os.path.join(root, f))
    return sorted(out)


def worker(
    gpu_id: int,
    queue: Queue,
    cache_dir: Path,
    model_size: str,
    compute_type: str,
    batch_size: int,
    language: str,
) -> None:
    """One worker per GPU. Loads faster-whisper once, then drains the queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import ffmpeg
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    print(f"[gpu{gpu_id}] loading faster-whisper {model_size} ({compute_type})", flush=True)
    base = WhisperModel(model_size, device="cuda", compute_type=compute_type)
    pipeline = BatchedInferencePipeline(model=base)

    cache_dir.mkdir(parents=True, exist_ok=True)

    while True:
        video_path = queue.get()
        if video_path is None:
            break
        stem = Path(video_path).stem
        cache_txt = cache_dir / f"{stem}.txt"
        if cache_txt.exists():
            continue

        audio_path = cache_dir / f"{stem}.wav"
        try:
            if not audio_path.exists():
                (
                    ffmpeg.input(video_path)
                    .output(str(audio_path), acodec="pcm_s16le", ac=1, ar="16k")
                    .overwrite_output()
                    .run(quiet=True)
                )
        except Exception as e:
            print(f"[gpu{gpu_id}] ffmpeg failed for {video_path}: {e}", flush=True)
            continue

        try:
            t0 = time.time()
            segments, info = pipeline.transcribe(
                str(audio_path),
                batch_size=batch_size,
                language=language if language != "auto" else None,
                vad_filter=True,
            )
            transcripts = [seg.text.strip() for seg in segments if seg.text.strip()]
        except Exception as e:
            print(f"[gpu{gpu_id}] faster-whisper failed for {video_path}: {e}", flush=True)
            continue

        cache_txt.write_text("\n".join(transcripts))
        dur = time.time() - t0
        print(
            f"[gpu{gpu_id}] cached {stem} ({len(transcripts)} segs, "
            f"{info.duration:.0f}s audio in {dur:.1f}s, "
            f"{info.duration / max(dur, 1e-6):.1f}x rt)",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True, help="Directory of input videos")
    ap.add_argument("--cache_dir", default=str(DEFAULT_CACHE),
                    help="Output cache dir (default: agents/video_rag/cache/audio)")
    ap.add_argument("--gpus", default="0", help="Comma-separated physical GPU ids")
    ap.add_argument("--workers_per_gpu", type=int, default=1,
                    help="faster-whisper is already batched; usually 1 worker/GPU is enough.")
    ap.add_argument("--model_size", default="large-v3",
                    help="faster-whisper model. 'large-v3' / 'large-v3-turbo' / 'distil-large-v3'.")
    ap.add_argument("--compute_type", default="float16",
                    help="float16 (default), int8_float16, int8")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="Chunks processed in parallel by BatchedInferencePipeline")
    ap.add_argument("--language", default="en",
                    help="Force language ('en') or 'auto' to detect")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    videos = find_videos(args.video_dir)
    print(f"Found {len(videos)} videos under {args.video_dir}")

    pending = [v for v in videos if not (cache_dir / f"{Path(v).stem}.txt").exists()]
    print(f"{len(pending)} videos need ASR ({len(videos) - len(pending)} cached already)")
    if not pending:
        return

    queue: Queue = Queue()
    for v in pending:
        queue.put(v)

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    procs = []
    for gpu_id in gpu_ids:
        for _ in range(args.workers_per_gpu):
            queue.put(None)
            p = Process(
                target=worker,
                args=(
                    gpu_id, queue, cache_dir, args.model_size,
                    args.compute_type, args.batch_size, args.language,
                ),
            )
            p.start()
            procs.append(p)

    t0 = time.time()
    for p in procs:
        p.join()
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
