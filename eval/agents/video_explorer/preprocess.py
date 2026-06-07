#!/usr/bin/env python3
"""Preprocessing script for VideoExplorer agent.

Prepares video data for efficient VideoExplorer inference:
1. Cut videos into clips (default: 10 seconds) using ffmpeg stream copy
2. Compute LanguageBind embeddings for clip retrieval

Usage:
    python preprocess.py --video_dir /path/to/videos --output_dir /path/to/output
    python preprocess.py --video_dir /path/to/videos --output_dir /path/to/output --num_workers 32
"""

import argparse
import os
import sys
import pickle
import subprocess
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add VideoDeepResearch repo to path
AGENTS_DIR = Path(__file__).parent.parent
REPO_PATH = AGENTS_DIR / "repos" / "VideoDeepResearch"
if REPO_PATH.exists():
    sys.path.insert(0, str(REPO_PATH))


def get_video_files(video_dir: str) -> list:
    """Find all video files in directory."""
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv'}
    videos = []
    for root, _, files in os.walk(video_dir):
        for f in files:
            if Path(f).suffix.lower() in video_extensions:
                videos.append(os.path.join(root, f))
    return sorted(videos)


def get_video_info(video_path: str) -> dict:
    """Get video duration and metadata using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_format', '-show_streams',
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info = json.loads(result.stdout)
        duration = float(info.get('format', {}).get('duration', 0))
        return {'duration': duration, 'valid': True}
    except Exception:
        return {'duration': 0, 'valid': False}


def cut_clip_ffmpeg(video_path: str, start: float, end: float, output_path: str,
                    max_size: int = 700) -> bool:
    """Cut a video clip using ffmpeg with stream copy (no re-encoding = fast).

    Falls back to re-encoding only if stream copy fails.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    duration = end - start

    # First try: stream copy (fastest, no quality loss)
    cmd_copy = [
        'ffmpeg', '-y',
        '-ss', str(start),
        '-i', video_path,
        '-t', str(duration),
        '-c', 'copy',  # Stream copy - no re-encoding
        '-avoid_negative_ts', 'make_zero',
        '-movflags', '+faststart',
        output_path
    ]

    try:
        result = subprocess.run(
            cmd_copy,
            capture_output=True,
            timeout=60,
            text=True
        )
        if result.returncode == 0 and os.path.exists(output_path):
            # Verify output is valid
            if os.path.getsize(output_path) > 1000:
                return True
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    # Fallback: re-encode with scaling (slower but more reliable)
    cmd_encode = [
        'ffmpeg', '-y',
        '-ss', str(start),
        '-i', video_path,
        '-t', str(duration),
        '-vf', f'scale=\'min({max_size},iw)\':-2',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',  # Speed over compression
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        output_path
    ]

    try:
        result = subprocess.run(
            cmd_encode,
            capture_output=True,
            timeout=120,
            text=True
        )
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        print(f"Failed to cut clip: {e}")
        return False


def cut_video_clips(video_path: str, clips_dir: str, clip_duration: int = 10) -> list:
    """Cut a video into clips of specified duration."""
    video_name = Path(video_path).stem
    video_clips_dir = os.path.join(clips_dir, video_name)

    info = get_video_info(video_path)
    duration = info['duration']
    if duration <= 0:
        return []

    os.makedirs(video_clips_dir, exist_ok=True)

    clip_paths = []
    num_clips = int(duration // clip_duration) + (1 if duration % clip_duration > 1 else 0)

    for i in range(num_clips):
        start_time = i * clip_duration
        end_time = min((i + 1) * clip_duration, duration)

        # Skip very short clips
        if end_time - start_time < 0.5:
            continue

        clip_filename = f"clip_{i:04d}.mp4"
        clip_path = os.path.join(video_clips_dir, clip_filename)

        # Skip if already exists and valid
        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
            clip_paths.append(clip_path)
            continue

        if cut_clip_ffmpeg(video_path, start_time, end_time, clip_path):
            clip_paths.append(clip_path)

    return clip_paths


def process_single_video(args) -> dict:
    """Process a single video: cut into clips."""
    video_path, clips_dir, clip_duration = args
    video_name = Path(video_path).stem

    try:
        clip_paths = cut_video_clips(video_path, clips_dir, clip_duration)
        return {
            'video': video_name,
            'path': video_path,
            'num_clips': len(clip_paths),
            'success': True
        }
    except Exception as e:
        return {
            'video': video_name,
            'path': video_path,
            'num_clips': 0,
            'success': False,
            'error': str(e)
        }


def _patch_torchvision_compat():
    """Patch torchvision/torchaudio for compatibility with older dependencies."""
    # Patch torchvision.transforms.functional_tensor (removed in newer versions)
    try:
        import torchvision.transforms
        if not hasattr(torchvision.transforms, 'functional_tensor'):
            from torchvision.transforms import _functional_tensor
            torchvision.transforms.functional_tensor = _functional_tensor
            sys.modules['torchvision.transforms.functional_tensor'] = _functional_tensor
    except Exception:
        pass

    # Patch torchaudio.set_audio_backend (removed in newer versions)
    try:
        import torchaudio
        if not hasattr(torchaudio, 'set_audio_backend'):
            torchaudio.set_audio_backend = lambda x: None  # No-op
    except Exception:
        pass


def compute_embeddings_batch(
    videos: list,
    clips_dir: str,
    embeddings_dir: str,
    clip_duration: int,
    batch_size: int = 64,
    device: str = "cuda",
    num_workers: int = 16
):
    """Compute LanguageBind embeddings - processes ALL clips in one pass."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    import warnings
    import logging
    import numpy as np

    # Suppress warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    _patch_torchvision_compat()

    if str(REPO_PATH) not in sys.path:
        sys.path.insert(0, str(REPO_PATH))

    try:
        from languagebind import LanguageBind
    except ImportError as e:
        print(f"Error: languagebind import failed: {e}")
        return

    print(f"Loading LanguageBind model on {device}...")
    clip_type = {'video': 'LanguageBind/LanguageBind_Video_FT'}
    model = LanguageBind(clip_type=clip_type, cache_dir='./cache')

    # Enable SDPA
    for key in model.modality_encoder:
        if hasattr(model.modality_encoder[key], 'config'):
            model.modality_encoder[key].config._attn_implementation = "sdpa"

    model = model.to(device)
    model.eval()

    # NO torch.compile - autotune overhead not worth it
    print("Model loaded (no torch.compile to avoid autotune overhead)")

    from languagebind.video.processing_video import get_video_transform

    video_config = model.modality_config['video']
    video_config.vision_config.video_decode_backend = 'decord'
    video_transform = get_video_transform(video_config)
    num_frames = video_config.vision_config.num_frames

    os.makedirs(embeddings_dir, exist_ok=True)

    # Build global clip list across ALL videos
    all_clips = []  # [(clip_path, video_name, clip_idx)]
    video_clip_counts = {}  # video_name -> num_clips

    for video_path in videos:
        video_name = Path(video_path).stem
        embedding_path = os.path.join(embeddings_dir, f"{video_name}.pkl")
        if os.path.exists(embedding_path):
            continue
        video_clips_dir = os.path.join(clips_dir, video_name)
        if not os.path.exists(video_clips_dir):
            continue
        clip_files = sorted([f for f in os.listdir(video_clips_dir) if f.endswith('.mp4')])
        if clip_files:
            video_clip_counts[video_name] = len(clip_files)
            for idx, f in enumerate(clip_files):
                all_clips.append((os.path.join(video_clips_dir, f), video_name, idx))

    if not all_clips:
        print("All embeddings already computed!")
        return

    print(f"Processing {len(all_clips)} clips from {len(video_clip_counts)} videos...")

    # Fast pyav loader
    def load_video_pyav(path, transform, num_frames):
        import av
        container = av.open(path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        total_frames = stream.frames or 100
        frame_indices = set(np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int))

        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in frame_indices:
                frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
            if len(frames) >= num_frames:
                break
        container.close()

        while len(frames) < num_frames:
            frames.append(frames[-1].clone() if frames else torch.zeros(3, 224, 224))

        return transform(torch.stack(frames[:num_frames], dim=1))

    class GlobalClipDataset(Dataset):
        def __init__(self, clips, transform, num_frames):
            self.clips = clips
            self.transform = transform
            self.num_frames = num_frames

        def __len__(self):
            return len(self.clips)

        def __getitem__(self, idx):
            clip_path, video_name, clip_idx = self.clips[idx]
            try:
                tensor = load_video_pyav(clip_path, self.transform, self.num_frames)
                return tensor, video_name, clip_idx, True
            except Exception:
                return torch.zeros(3, self.num_frames, 224, 224), video_name, clip_idx, False

    dataset = GlobalClipDataset(all_clips, video_transform, num_frames)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        shuffle=False
    )

    # Collect embeddings per video
    video_embeddings = {name: [None] * count for name, count in video_clip_counts.items()}

    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_tensors, video_names, clip_indices, valid_masks in tqdm(loader, desc="Computing embeddings"):
            valid = valid_masks.bool()
            if not valid.any():
                continue

            valid_tensors = batch_tensors[valid].to(device, non_blocking=True)
            inputs = {'video': {'pixel_values': valid_tensors}}
            embs = model(inputs)['video'].cpu()

            # Scatter results back
            emb_idx = 0
            for i, (vname, cidx, is_valid) in enumerate(zip(video_names, clip_indices, valid_masks)):
                if is_valid:
                    video_embeddings[vname][cidx.item()] = embs[emb_idx]
                    emb_idx += 1

    # Save all embeddings + matching clip-path list (retriever_languagebind.py
    # requires both {name}.pkl and {name}_clip_paths.pkl to short-circuit the
    # pre_calculate fallthrough).
    print("Saving embeddings...")
    for video_name, emb_list in video_embeddings.items():
        valid_pairs = [(i, e) for i, e in enumerate(emb_list) if e is not None]
        if not valid_pairs:
            continue
        video_clips_dir = os.path.join(clips_dir, video_name)
        clip_files = sorted([f for f in os.listdir(video_clips_dir) if f.endswith('.mp4')])
        valid_embs = [e for _, e in valid_pairs]
        valid_paths = [os.path.join(video_clips_dir, clip_files[i]) for i, _ in valid_pairs]

        embedding_path = os.path.join(embeddings_dir, f"{video_name}.pkl")
        clip_paths_path = os.path.join(embeddings_dir, f"{video_name}_clip_paths.pkl")
        final = torch.stack(valid_embs, dim=0)
        with open(embedding_path, 'wb') as f:
            pickle.dump(final, f)
        with open(clip_paths_path, 'wb') as f:
            pickle.dump(valid_paths, f)

    print(f"Saved {len(video_embeddings)} video embeddings + clip path lists")


def main():
    parser = argparse.ArgumentParser(description='Preprocess videos for VideoExplorer')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for clips and embeddings')
    parser.add_argument('--clip_duration', type=int, default=10,
                        help='Duration of each clip in seconds (default: 10)')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Number of parallel workers for clip cutting')
    parser.add_argument('--skip_embeddings', action='store_true',
                        help='Skip embedding computation (only cut clips)')
    parser.add_argument('--skip_clips', action='store_true',
                        help='Skip clip cutting (only compute embeddings)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device for embedding computation')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for embedding computation')
    args = parser.parse_args()

    # Setup paths
    clips_dir = os.path.join(args.output_dir, 'clips', str(args.clip_duration))
    embeddings_dir = os.path.join(args.output_dir, 'embeddings', str(args.clip_duration), 'languagebind')
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    # Find videos
    videos = get_video_files(args.video_dir)
    print(f"Found {len(videos)} videos in {args.video_dir}")

    if not videos:
        print("No videos found!")
        return

    # Step 1: Cut all videos into clips (parallel)
    if not args.skip_clips:
        print(f"\n{'='*50}")
        print(f"Step 1: Cutting videos into {args.clip_duration}s clips")
        print(f"Workers: {args.num_workers}")
        print(f"{'='*50}")

        cut_args = [(v, clips_dir, args.clip_duration) for v in videos]

        results = []
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_single_video, arg): arg for arg in cut_args}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Cutting clips"):
                results.append(future.result())

        successful = sum(1 for r in results if r['success'])
        total_clips = sum(r['num_clips'] for r in results)
        print(f"\nCreated {total_clips} clips from {successful}/{len(videos)} videos")

    # Step 2: Compute embeddings (GPU, sequential batched)
    if not args.skip_embeddings:
        print(f"\n{'='*50}")
        print(f"Step 2: Computing LanguageBind embeddings")
        print(f"GPU: {args.gpu}, Batch size: {args.batch_size}")
        print(f"{'='*50}")

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        compute_embeddings_batch(
            videos, clips_dir, embeddings_dir,
            args.clip_duration, args.batch_size, device,
            num_workers=8
        )

    print(f"\n{'='*50}")
    print("Preprocessing complete!")
    print(f"Clips: {clips_dir}")
    print(f"Embeddings: {embeddings_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
