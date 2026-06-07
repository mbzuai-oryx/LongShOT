"""
Video processing module for frame extraction using FFmpeg pipe streaming.
Frames are extracted as raw numpy arrays (512x512 RGB uint8) — no PIL.

Supports parallel extraction: splits video into N time segments and runs
N FFmpeg processes concurrently, streaming chunks as they become available.
"""

import logging
from typing import List, Optional
import numpy as np
import subprocess
import json
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

logger = logging.getLogger(__name__)

FRAME_W = 512
FRAME_H = 512
FRAME_BYTES = FRAME_W * FRAME_H * 3  # RGB


@dataclass
class VideoFrame:
    """Represents a video frame with timing information."""
    image: np.ndarray   # (H, W, 3) uint8 RGB, or None after embedding
    timestamp: float
    frame_number: int


class VideoProcessor:
    """
    Extracts frames at a target FPS using parallel FFmpeg pipes.
    Streams chunks to the consumer as they are extracted — no buffering
    of all frames before yielding.
    """

    def __init__(self, target_fps: float = 5.0, enable_gpu_acceleration: bool = True,
                 num_extract_workers: int = 4):
        self.target_fps = target_fps
        self.num_extract_workers = num_extract_workers
        self.gpu_decoder = self._detect_gpu_acceleration() if enable_gpu_acceleration else None
        self.ffmpeg_threads = max(2, (os.cpu_count() or 8) // max(num_extract_workers, 1))

    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata using ffprobe."""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe = json.loads(result.stdout)

        video_stream = next(
            (s for s in probe['streams'] if s['codec_type'] == 'video'), None
        )
        if video_stream is None:
            raise ValueError("No video stream found in file")

        fps_str = video_stream['r_frame_rate']
        fps = float(fps_str.split('/')[0]) / float(fps_str.split('/')[1]) if '/' in fps_str else float(fps_str)

        return {
            'duration': float(probe['format']['duration']),
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'fps': fps,
            'codec': video_stream['codec_name']
        }

    def _detect_gpu_acceleration(self) -> Optional[str]:
        """Detect available GPU acceleration for FFmpeg decoding."""
        for decoder in ["h264_nvdec", "hevc_nvdec"]:
            try:
                test_cmd = ['ffmpeg', '-hide_banner', '-f', 'lavfi', '-i',
                           'testsrc=duration=0.1:size=64x64',
                           '-c:v', decoder, '-f', 'null', '-']
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logger.info(f"GPU decode available: {decoder}")
                    return decoder
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                continue
        return None

    def _build_segment_cmd(self, video_path: str, start_time: float, duration: float) -> list:
        """Build FFmpeg command for extracting a time segment."""
        cmd = ['ffmpeg', '-nostdin']

        if self.gpu_decoder and 'nvdec' in self.gpu_decoder:
            cmd.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])

        cmd.extend(['-ss', f'{start_time:.4f}', '-t', f'{duration:.4f}', '-i', str(video_path)])

        if self.gpu_decoder and 'nvdec' in self.gpu_decoder:
            vf = f'fps={self.target_fps},scale_cuda={FRAME_W}:{FRAME_H}:format=yuv420p,hwdownload,format=rgb24'
        else:
            vf = f'fps={self.target_fps},scale={FRAME_W}:{FRAME_H}:flags=fast_bilinear'

        cmd.extend([
            '-vf', vf,
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-threads', str(self.ffmpeg_threads),
            '-an',
            'pipe:1'
        ])
        return cmd

    def _stream_segment(self, video_path: str, start_time: float, duration: float,
                        start_frame_num: int, chunk_size: int, out_queue: Queue,
                        segment_id: int):
        """
        Extract frames from a segment and push chunks directly to the output queue.
        Each chunk is (segment_id, frame_number_of_first_frame, frames_list, pixels_array).
        """
        cmd = self._build_segment_cmd(video_path, start_time, duration)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   bufsize=FRAME_BYTES * chunk_size * 2)

        frame_num = start_frame_num
        try:
            while True:
                raw = process.stdout.read(FRAME_BYTES * chunk_size)
                if not raw:
                    break

                n_complete = len(raw) // FRAME_BYTES
                if n_complete == 0:
                    break

                pixels = np.frombuffer(raw[:n_complete * FRAME_BYTES], dtype=np.uint8).reshape(
                    n_complete, FRAME_H, FRAME_W, 3
                )
                frames = [
                    VideoFrame(image=pixels[i], timestamp=(frame_num + i) / self.target_fps,
                              frame_number=frame_num + i)
                    for i in range(n_complete)
                ]
                out_queue.put((segment_id, frame_num, frames, pixels))
                frame_num += n_complete
        finally:
            process.terminate()
            process.wait()

    def extract_frames_chunked(self, video_path: str, chunk_size: int = 64):
        """
        Generator that yields (frames, pixels) chunks.

        For long videos: launches N parallel FFmpeg workers, each streaming
        chunks into a shared queue. A reorder buffer ensures chunks are yielded
        in frame-number order for correct embedding storage.

        For short videos: single-process streaming.
        """
        info = self.get_video_info(video_path)
        duration = info['duration']
        expected = int(duration * self.target_fps)
        n_workers = min(self.num_extract_workers, max(1, expected // 200))

        if n_workers <= 1 or duration < 10:
            logger.info(f"Single-process extraction: ~{expected} frames, chunks of {chunk_size}")
            yield from self._extract_single_process(video_path, duration, chunk_size)
            return

        logger.info(f"Parallel streaming extraction: ~{expected} frames, {n_workers} workers, chunks of {chunk_size}")

        # Split video into time segments
        segment_duration = duration / n_workers
        out_queue = Queue(maxsize=n_workers * 2)

        # Launch worker threads
        workers = []
        for i in range(n_workers):
            seg_start = i * segment_duration
            seg_dur = segment_duration if i < n_workers - 1 else (duration - seg_start)
            seg_frame_start = int(seg_start * self.target_fps)
            t = Thread(target=self._stream_segment, daemon=True,
                       args=(video_path, seg_start, seg_dur, seg_frame_start, chunk_size, out_queue, i))
            t.start()
            workers.append(t)

        # Sentinel counter — when all workers finish, they stop pushing
        def _wait_and_sentinel():
            for t in workers:
                t.join()
            out_queue.put(None)

        sentinel_thread = Thread(target=_wait_and_sentinel, daemon=True)
        sentinel_thread.start()

        # Reorder buffer: segments may interleave, we yield in frame-number order
        buffer = {}  # frame_num -> (frames, pixels)
        next_expected = 0
        seen_frames = set()
        total = 0

        while True:
            item = out_queue.get()
            if item is None:
                break

            seg_id, first_frame, frames, pixels = item

            # Deduplicate at segment boundaries
            clean_frames = []
            clean_indices = []
            for i, f in enumerate(frames):
                if f.frame_number not in seen_frames:
                    seen_frames.add(f.frame_number)
                    clean_frames.append(f)
                    clean_indices.append(i)

            if not clean_frames:
                continue

            clean_pixels = pixels[clean_indices] if len(clean_indices) < len(frames) else pixels

            # Buffer for reordering
            buffer[clean_frames[0].frame_number] = (clean_frames, clean_pixels)

            # Yield all consecutive chunks starting from next_expected
            while next_expected in buffer:
                out_frames, out_pixels = buffer.pop(next_expected)
                total += len(out_frames)
                next_expected = out_frames[-1].frame_number + 1
                yield out_frames, out_pixels

        # Flush remaining buffer in order
        for key in sorted(buffer.keys()):
            out_frames, out_pixels = buffer[key]
            total += len(out_frames)
            yield out_frames, out_pixels

        sentinel_thread.join(timeout=5)
        logger.info(f"Parallel extraction complete: {total} frames from {n_workers} workers")

    def _extract_single_process(self, video_path: str, duration: float, chunk_size: int):
        """Single-process streaming extraction for short videos."""
        cmd = self._build_segment_cmd(video_path, 0.0, duration)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   bufsize=FRAME_BYTES * chunk_size * 2)

        frame_num = 0
        total = 0
        try:
            while True:
                raw = process.stdout.read(FRAME_BYTES * chunk_size)
                if not raw:
                    break

                n_complete = len(raw) // FRAME_BYTES
                if n_complete == 0:
                    break

                pixels = np.frombuffer(raw[:n_complete * FRAME_BYTES], dtype=np.uint8).reshape(
                    n_complete, FRAME_H, FRAME_W, 3
                )
                frames = [
                    VideoFrame(image=pixels[i], timestamp=(frame_num + i) / self.target_fps,
                              frame_number=frame_num + i)
                    for i in range(n_complete)
                ]
                frame_num += n_complete
                total += n_complete
                yield frames, pixels
        finally:
            process.terminate()
            process.wait()

        logger.info(f"Single-process extraction complete: {total} frames")

    def extract_frames_list(self, video_path: str) -> List[VideoFrame]:
        """Extract all frames into a list."""
        info = self.get_video_info(video_path)
        logger.info(f"Extracting ~{int(info['duration'] * self.target_fps)} frames")

        frames = []
        for chunk_frames, _ in self.extract_frames_chunked(video_path, chunk_size=256):
            frames.extend(chunk_frames)

        logger.info(f"Extracted {len(frames)} frames")
        return frames
