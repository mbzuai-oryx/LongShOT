"""Lightweight OpenAI-compatible server backed by HuggingFace Transformers.

Drop-in replacement for vLLM's ``/v1/chat/completions`` when models lack
native vLLM support (e.g. MiniCPM-o omni with audio).

Optimizations:
    - Data-parallel multi-GPU: one model replica per GPU for N× throughput
    - Async media pre-processing: video/audio decoding overlaps with GPU inference
    - Per-replica dedicated inference threads with CUDA device affinity
    - Media pre-warm endpoint for eliminating cold-start latency
    - LRU caching for all media types (video, audio, images)

Usage:
    python transformers_serve.py --model openbmb/MiniCPM-o-2_6 --port 8100
    python transformers_serve.py --model openbmb/MiniCPM-o-2_6 --port 8100 --replicas 4
"""

from serve.app import main

if __name__ == "__main__":
    main()
