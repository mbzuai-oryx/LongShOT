#!/usr/bin/env python3
"""
Main entry point for the video preprocessing pipeline.

This script processes videos to extract audio transcriptions and visual embeddings
using efficient 5 FPS sampling for optimal speed and quality balance.
All frames are directly sampled at 5 FPS during extraction for maximum efficiency.
"""

import logging
import os
import sys
import click
from pathlib import Path
from typing import Optional, Dict, Any
import concurrent.futures
import time

from preprocessing import (
    AudioProcessor,
    VideoProcessor,
    ImageEmbedder,
    VectorStore,
    CacheManager,
)

from agent import VideoAgent
from config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("video_preprocessing.log"),
    ],
)

logger = logging.getLogger(__name__)


class VideoPreprocessingPipeline:
    """
    Main pipeline for processing videos and storing embeddings.

    Uses efficient 5 FPS direct sampling to extract frames optimized for
    SigLIP model processing. No filtering needed - frames are sampled
    during extraction for maximum speed.
    """

    def __init__(
        self,
        whisper_model: str = "small",
        siglip_model: str = "ViT-B-16-SigLIP-512",
        cache_dir: str = "./cache",
        db_path: str = "./chroma_db",
        vector_store: Optional[VectorStore] = None,
        device: str = "auto",
        batch_size: int = 32,
        audio_batch_size: int = 16,
        batch_mode: bool = False,
        text_model_instances: int = 4,
        image_model_instances: int = 2,
        max_workers: int = 16,
        frame_extraction_workers: int = 2,
        audio_processing_workers: int = 2,
        database_storage_workers: int = 2,
        chunk_size_multiplier: float = 1.0,
        enable_memory_optimization: bool = True,
        clear_cache_between_videos: bool = False,
        immediate_database_sync: bool = True,
        cpu_audio_batch_size: int = 8,
        cpu_audio_processing_workers: int = 4,
    ):
        """
        Initialize the processing pipeline.

        Args:
            whisper_model: Whisper model size for audio transcription
            siglip_model: SigLIP model for visual embeddings
            cache_dir: Directory for caching processed data
            db_path: Path to ChromaDB database
            vector_store: Optional shared VectorStore instance
            device: Device to use for processing ("cpu", "cuda", "auto")
            batch_size: Batch size for visual embedding processing
            audio_batch_size: Batch size for BatchedInferencePipeline processing
            batch_mode: If True, prevents redundant model loading for batch processing
            text_model_instances: Number of text embedding model instances on CUDA:0
            image_model_instances: Number of image embedding model instances on CUDA:1
            max_workers: Maximum number of worker threads for parallel operations
            frame_extraction_workers: Number of workers for frame extraction
            audio_processing_workers: Number of workers for audio processing
            database_storage_workers: Number of workers for database storage
            chunk_size_multiplier: Multiplier for chunk sizes (affects memory usage)
            enable_memory_optimization: Enable memory optimization techniques
            clear_cache_between_videos: Clear GPU cache between videos in batch processing
            immediate_database_sync: Immediately sync each video to database after processing
            cpu_audio_batch_size: Batch size for CPU audio processing (faster-whisper)
            cpu_audio_processing_workers: Number of OMP threads for CPU audio processing
        """
        self.config = {
            "whisper_model": whisper_model,
            "siglip_model": siglip_model,
            "device": device,
            "batch_size": batch_size,
            "audio_batch_size": audio_batch_size,
            "batch_mode": batch_mode,
            "text_model_instances": text_model_instances,
            "image_model_instances": image_model_instances,
            "max_workers": max_workers,
            "frame_extraction_workers": frame_extraction_workers,
            "audio_processing_workers": audio_processing_workers,
            "database_storage_workers": database_storage_workers,
            "chunk_size_multiplier": chunk_size_multiplier,
            "enable_memory_optimization": enable_memory_optimization,
            "clear_cache_between_videos": clear_cache_between_videos,
            "immediate_database_sync": immediate_database_sync,
            "cpu_audio_batch_size": cpu_audio_batch_size,
            "cpu_audio_processing_workers": cpu_audio_processing_workers,
            "metadata_version": "v6_5fps_direct_sampling",  # Updated for 5 FPS direct sampling
        }

        self.audio_processor = AudioProcessor(
            model_size=whisper_model,
            device=device,
            cpu_threads=cpu_audio_processing_workers,
        )

        self.video_processor = VideoProcessor(
            num_extract_workers=self.config.get("frame_extraction_workers", 4)
        )
        self.image_embedder = ImageEmbedder(num_instances=image_model_instances)

        self.vector_store = vector_store or VectorStore(db_path=db_path)
        self.cache_manager = CacheManager(cache_dir=cache_dir)
        self.quiet = False

        # Preload Whisper + SigLIP + text models in parallel at t=0
        self._preload_models()

    def _preload_models(self):
        """Preload all GPU/CPU models in parallel so first video starts immediately."""

        def _load_whisper():
            self.audio_processor._get_batched_model()
            logger.info("Whisper model preloaded")

        def _load_siglip():
            self.image_embedder._load_models()
            logger.info("SigLIP model preloaded")

        def _load_text():
            self._initialize_text_embedders()
            logger.info("Text embedding model preloaded")

        logger.info("Preloading models in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(_load_whisper),
                pool.submit(_load_siglip),
                pool.submit(_load_text),
            ]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()  # Raise any exceptions
        logger.info("All models preloaded")

    def process_video(
        self,
        video_path: str,
        video_id: Optional[str] = None,
        force_reprocess: bool = False,
        language: Optional[str] = None,
        log_fn=None,
    ) -> Dict[str, Any]:
        """
        Process a video file through the complete pipeline.

        Args:
            video_path: Path to the video file
            video_id: Unique identifier for the video (auto-generated if None)
            force_reprocess: Whether to reprocess even if cached
            language: Language code for transcription (auto-detect if None)
            log_fn: Callable for progress messages (default: logger.info).
                     Batch mode passes tqdm.write for clean console output.

        Returns:
            Dictionary containing processing results and statistics
        """
        if log_fn is None:
            log_fn = logger.info
        video_path = str(Path(video_path).resolve())

        if video_id is None:
            video_id = Path(video_path).stem

        logger.info(f"Processing video: {video_path} (ID: {video_id})")

        # Check cache first
        if not force_reprocess and self.cache_manager.is_cached(
            video_path, self.config
        ):
            logger.info("Loading from cache...")
            cached_data = self.cache_manager.load_from_cache(video_path)

            if cached_data:
                # Add to vector store if not already there
                self._add_cached_data_to_vector_store(video_id, cached_data)

                return {
                    "video_id": video_id,
                    "video_path": video_path,
                    "from_cache": True,
                    "audio_segments": len(cached_data["audio_segments"]),
                    "visual_frames": len(cached_data["visual_frames"]),
                    "processing_time": 0,
                }

        start_time = time.time()
        stage_times = {}

        try:
            batch_size = self.config["batch_size"]

            tag = f"[{video_id}]"

            # ---- Audio pipeline (runs in its own thread) ----
            def _audio_pipeline():
                log_fn(f"  {tag} audio: extracting + transcribing...")
                t0 = time.time()
                results = self.audio_processor.process_video_audio_batch(
                    [video_path],
                    language=language,
                    batch_size=self.config["audio_batch_size"],
                    cpu_batch_size=self.config["cpu_audio_batch_size"],
                )
                segments, info = results[0]
                stage_times["audio"] = time.time() - t0
                log_fn(
                    f"  {tag} audio: {len(segments)} segments, {stage_times['audio']:.1f}s"
                )

                t1 = time.time()
                embeddings = self._generate_text_embeddings([s.text for s in segments])
                stage_times["text_embed"] = time.time() - t1
                log_fn(
                    f"  {tag} text embed: {len(embeddings)} embeddings, {stage_times['text_embed']:.1f}s"
                )
                return segments, info, embeddings

            # ---- Visual pipeline: producer-consumer frame→embed overlap ----
            def _visual_pipeline():
                from queue import Queue
                from threading import Thread
                from preprocessing.image_embedder import ImageEmbedding
                from preprocessing.video_processor import VideoFrame

                # Queue carries (frames_list, pixels_batch) tuples from producer
                frame_queue = Queue(maxsize=2)
                all_frames = []
                all_embeddings = []
                extract_error = [None]

                def _producer():
                    try:
                        for (
                            frames,
                            pixels,
                        ) in self.video_processor.extract_frames_chunked(
                            video_path, chunk_size=batch_size
                        ):
                            frame_queue.put((frames, pixels))
                    except Exception as e:
                        extract_error[0] = e
                    finally:
                        frame_queue.put(None)

                producer = Thread(target=_producer, daemon=True)
                producer.start()

                log_fn(f"  {tag} visual: extracting + embedding (pipelined)...")
                t0 = time.time()
                while True:
                    item = frame_queue.get()
                    if item is None:
                        break
                    if extract_error[0]:
                        raise extract_error[0]

                    frames, pixels = item

                    # Fast path: numpy batch → tensor → SigLIP (no PIL)
                    emb_array = self.image_embedder.encode_numpy_batch(pixels)
                    for frame, emb in zip(frames, emb_array):
                        all_frames.append(
                            VideoFrame(
                                image=None,
                                timestamp=frame.timestamp,
                                frame_number=frame.frame_number,
                            )
                        )
                        all_embeddings.append(
                            ImageEmbedding(
                                embedding=emb,
                                timestamp=frame.timestamp,
                                frame_number=frame.frame_number,
                                image_size=(512, 512),
                            )
                        )
                    del frames, pixels  # Free immediately

                producer.join()
                if extract_error[0]:
                    raise extract_error[0]

                stage_times["visual"] = time.time() - t0
                log_fn(
                    f"  {tag} visual: {len(all_frames)} frames, {stage_times['visual']:.1f}s"
                )
                return all_frames, all_embeddings

            # ---- Run audio + visual in parallel ----
            # Whichever finishes first triggers its storage immediately
            # while the other continues computing.
            audio_result = [None]
            visual_result = [None]
            audio_store_future = None

            def _audio_pipeline_and_store():
                """Audio pipeline + immediate DB storage when done."""
                segments, info, embeddings = _audio_pipeline()
                audio_result[0] = (segments, info, embeddings)
                # Store audio to DB immediately — don't wait for visual
                self.vector_store.add_video_metadata(
                    video_id,
                    os.path.abspath(video_path),
                    {
                        "audio_duration": info.get("duration", 0),
                        "language": info.get("language", "unknown"),
                        "sampling_fps": 5.0,
                    },
                )
                self.vector_store.add_audio_embeddings(video_id, segments, embeddings)
                log_fn(f"  {tag} audio stored (early)")

            def _visual_pipeline_wrapper():
                visual_result[0] = _visual_pipeline()

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                af = pool.submit(_audio_pipeline_and_store)
                vf = pool.submit(_visual_pipeline_wrapper)
                af.result()
                vf.result()

            audio_segments, audio_info, audio_embeddings = audio_result[0]
            visual_frames, visual_embeddings = visual_result[0]

            compute_time = time.time() - start_time
            stage_times["compute"] = compute_time
            log_fn(
                f"  {tag} compute done: {len(audio_segments)} audio + {len(visual_frames)} visual in {compute_time:.1f}s"
            )

            result = {
                "video_id": video_id,
                "video_path": video_path,
                "from_cache": False,
                "audio_segments": len(audio_segments),
                "visual_frames": len(visual_frames),
                "processing_time": compute_time,
                "audio_duration": audio_info.get("duration", 0),
                "language": audio_info.get("language", "unknown"),
                "stage_times": stage_times,
                "_audio_segments_data": audio_segments,
                "_audio_info": audio_info,
                "_audio_embeddings": audio_embeddings,
                "_audio_stored_early": True,  # Flag: audio already in DB
                "_visual_frames_data": visual_frames,
                "_visual_embeddings": visual_embeddings,
            }
            return result

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            raise

    def store_results(self, result: Dict[str, Any], log_fn=None):
        """
        Store computed results to vector DB + cache. Can run in background
        while next video's compute starts.

        If audio was already stored early (during visual compute), skips
        audio DB write and only stores visual + cache.
        """
        if log_fn is None:
            log_fn = logger.info

        if result.get("from_cache"):
            return

        video_id = result["video_id"]
        video_path = result["video_path"]
        tag = f"[{video_id}]"
        audio_stored_early = result.pop("_audio_stored_early", False)

        audio_segments = result.pop("_audio_segments_data")
        audio_info = result.pop("_audio_info")
        audio_embeddings = result.pop("_audio_embeddings")
        visual_frames = result.pop("_visual_frames_data")
        visual_embeddings = result.pop("_visual_embeddings")
        visual_embedding_arrays = [emb.embedding for emb in visual_embeddings]

        log_fn(f"  {tag} storing visual + cache...")
        store_start = time.time()

        # Update metadata with total_frames (may have been set without it during early store)
        self.vector_store.add_video_metadata(
            video_id,
            os.path.abspath(video_path),
            {
                "audio_duration": audio_info.get("duration", 0),
                "language": audio_info.get("language", "unknown"),
                "total_frames": len(visual_frames),
                "sampling_fps": 5.0,
            },
        )

        storage_tasks = [
            lambda: self.vector_store.add_visual_embeddings(
                video_id, visual_embeddings
            ),
            lambda: self.cache_manager.save_to_cache(
                video_path,
                audio_segments,
                audio_info,
                visual_frames,
                audio_embeddings,
                visual_embedding_arrays,
                self.config,
            ),
        ]
        if not audio_stored_early:
            storage_tasks.append(
                lambda: self.vector_store.add_audio_embeddings(
                    video_id, audio_segments, audio_embeddings
                )
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(storage_tasks)
        ) as pool:
            futures = [pool.submit(task) for task in storage_tasks]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

        store_time = time.time() - store_start
        log_fn(f"  {tag} stored in {store_time:.1f}s")

    def _initialize_text_embedders(self):
        """Initialize text embedding model instances."""
        from sentence_transformers import SentenceTransformer

        if hasattr(self, "_text_models"):
            return

        text_instances = self.config.get("text_model_instances", 1)
        self._text_models = []
        text_device = self._determine_text_embedding_device()

        for i in range(text_instances):
            try:
                model = SentenceTransformer("all-MiniLM-L6-v2", device=text_device)
                self._text_models.append(model)
            except Exception as e:
                if "cuda" in text_device:
                    logger.warning(
                        f"CUDA failed for text models ({e}), falling back to CPU"
                    )
                    text_device = "cpu"
                    model = SentenceTransformer("all-MiniLM-L6-v2", device=text_device)
                    self._text_models.append(model)
                else:
                    raise

        logger.info(f"Loaded {len(self._text_models)} text model(s) on {text_device}")

    def _determine_text_embedding_device(self) -> str:
        """Determine device for text embeddings. Respects CUDA_VISIBLE_DEVICES."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    def _generate_text_embeddings(self, texts: list) -> list:
        """Generate text embeddings. Uses parallel instances if available."""
        self._initialize_text_embedders()

        if not texts:
            return []

        text_instances = len(self._text_models)

        if text_instances == 1:
            # Single instance — encode directly, no threading overhead
            return list(
                self._text_models[0].encode(
                    texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False
                )
            )

        # Multiple instances — split and parallelize
        chunk_size = len(texts) // text_instances
        remainder = len(texts) % text_instances
        chunks = []
        idx = 0
        for i in range(text_instances):
            end = idx + chunk_size + (1 if i < remainder else 0)
            chunks.append(texts[idx:end])
            idx = end

        all_embeddings = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=text_instances
        ) as executor:
            futures = [
                executor.submit(
                    self._text_models[i].encode,
                    chunk,
                    batch_size=32,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                for i, chunk in enumerate(chunks)
                if chunk
            ]
            for future in futures:
                all_embeddings.extend(future.result())

        return all_embeddings

    def _add_cached_data_to_vector_store(
        self, video_id: str, cached_data: dict
    ) -> None:
        """Add cached data to vector store if not already present."""
        # Check if data already exists in vector store
        existing_metadata = self.vector_store.get_video_metadata(video_id)

        if existing_metadata["total_embeddings"] == 0:
            # Add to vector store
            self.vector_store.add_audio_embeddings(
                video_id, cached_data["audio_segments"], cached_data["audio_embeddings"]
            )

            # Convert cached visual embeddings back to ImageEmbedding objects
            visual_embeddings = []
            for frame_meta, emb_array in zip(
                cached_data["visual_frames"], cached_data["visual_embeddings"]
            ):
                from preprocessing.image_embedder import ImageEmbedding

                visual_embeddings.append(
                    ImageEmbedding(
                        embedding=emb_array,
                        timestamp=frame_meta["timestamp"],
                        frame_number=frame_meta["frame_number"],
                        image_size=frame_meta["image_size"],
                    )
                )

            self.vector_store.add_visual_embeddings(video_id, visual_embeddings)
            logger.info(f"Added cached data to vector store for video {video_id}")


# CLI Interface
@click.group()
def cli():
    """
    Video Agent Pipeline - AI-powered video understanding system.

    A comprehensive pipeline for processing and querying video content using:
    - Intelligent frame selection to reduce redundancy
    - Audio transcription with Whisper
    - Visual embeddings with SigLIP
    - Vector database storage with ChromaDB
    - LLM-powered interactive querying

    All settings are configured via config.yaml. Use --config to specify
    a different configuration file.

    Examples:
        # Process a video
        python main.py process video.mp4

        # Chat about a processed video
        python main.py chat video_id

        # View system statistics
        python main.py stats

        # Use custom config
        python main.py --config my_config.yaml process video.mp4
    """
    pass


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--video-id", help="Custom video identifier")
@click.option(
    "--config", "-c", default="config.yaml", help="Path to configuration file"
)
@click.option("--language", help="Language code for transcription")
@click.option(
    "--force-reprocess", is_flag=True, help="Force reprocessing even if cached"
)
def process(video_path, video_id, config, language, force_reprocess):
    """
    Process a video file through the preprocessing pipeline.

    Extracts frames at native FPS, applies intelligent frame selection to reduce
    redundancy, then generates audio and visual embeddings for vector storage.

    All processing settings are configured via the config file.
    """

    try:
        app_config = load_config(config_path=config)
        click.echo(f"📋 Loaded configuration from: {config}")
    except Exception as e:
        click.echo(f"⚠️  Error loading config: {e}, using defaults")
        app_config = load_config()

    pipeline = VideoPreprocessingPipeline(
        whisper_model=app_config.preprocessing.whisper_model,
        siglip_model=app_config.preprocessing.siglip_model,
        cache_dir=app_config.preprocessing.cache_dir,
        db_path=app_config.preprocessing.db_path,
        device=app_config.preprocessing.device,
        batch_size=getattr(app_config.preprocessing, "batch_size", 256),
        audio_batch_size=getattr(app_config.preprocessing, "audio_batch_size", 32),
        text_model_instances=getattr(
            app_config.preprocessing, "text_model_instances", 4
        ),
        image_model_instances=getattr(
            app_config.preprocessing, "image_model_instances", 3
        ),
        max_workers=getattr(app_config.preprocessing, "max_workers", 16),
        frame_extraction_workers=getattr(
            app_config.preprocessing, "frame_extraction_workers", 2
        ),
        audio_processing_workers=getattr(
            app_config.preprocessing, "audio_processing_workers", 2
        ),
        database_storage_workers=getattr(
            app_config.preprocessing, "database_storage_workers", 2
        ),
        chunk_size_multiplier=getattr(
            app_config.preprocessing, "chunk_size_multiplier", 1.0
        ),
        enable_memory_optimization=getattr(
            app_config.preprocessing, "enable_memory_optimization", True
        ),
        clear_cache_between_videos=getattr(
            app_config.preprocessing, "clear_cache_between_videos", False
        ),
        immediate_database_sync=getattr(
            app_config.preprocessing, "immediate_database_sync", True
        ),
        cpu_audio_batch_size=getattr(
            app_config.preprocessing, "cpu_audio_batch_size", 8
        ),
        cpu_audio_processing_workers=getattr(
            app_config.preprocessing, "cpu_audio_processing_workers", 4
        ),
    )

    # Models will load in background during video processing

    results = pipeline.process_video(
        video_path=video_path,
        video_id=video_id,
        force_reprocess=force_reprocess,
        language=language,
    )

    click.echo("\n" + "=" * 50)
    click.echo("PROCESSING RESULTS")
    click.echo("=" * 50)
    click.echo(f"Video ID: {results['video_id']}")
    click.echo(f"Video Path: {results['video_path']}")
    click.echo(f"From Cache: {results['from_cache']}")
    click.echo(f"Audio Segments: {results['audio_segments']}")
    click.echo(f"Visual Frames: {results['visual_frames']}")
    click.echo(f"Processing Time: {results['processing_time']:.2f}s")

    if "audio_duration" in results:
        click.echo(f"Audio Duration: {results['audio_duration']:.2f}s")
        click.echo(f"Language: {results['language']}")

    click.echo("=" * 50)


@cli.command()
@click.option(
    "--config", "-c", default="config.yaml", help="Path to configuration file"
)
def stats(config):
    """Show statistics about cached data and vector database."""

    try:
        app_config = load_config(config_path=config)
    except Exception as e:
        click.echo(f"⚠️  Error loading config: {e}, using defaults")
        app_config = load_config()

    cache_manager = CacheManager(app_config.preprocessing.cache_dir)
    vector_store = VectorStore(app_config.preprocessing.db_path)

    cache_stats = cache_manager.get_cache_stats()
    db_stats = vector_store.get_collection_stats()

    click.echo("\n" + "=" * 50)
    click.echo("SYSTEM STATISTICS")
    click.echo("=" * 50)
    click.echo("Cache Statistics:")
    click.echo(f"  Total Videos Cached: {cache_stats['total_videos_cached']}")
    click.echo(f"  Total Cache Size: {cache_stats['total_cache_size_mb']:.2f} MB")
    click.echo(f"  Cache Directory: {cache_stats['cache_directory']}")

    click.echo("\nVector Database Statistics:")
    click.echo(f"  Audio Embeddings: {db_stats['audio_embeddings']}")
    click.echo(f"  Visual Embeddings: {db_stats['visual_embeddings']}")
    click.echo(f"  Total Embeddings: {db_stats['total_embeddings']}")
    click.echo(f"  Unique Videos: {db_stats['unique_videos']}")
    click.echo(f"  Database Path: {db_stats['database_path']}")
    click.echo("=" * 50)


@cli.command()
@click.option(
    "--config", "-c", default="config.yaml", help="Path to configuration file"
)
def list_videos(config):
    """List all cached videos."""

    try:
        app_config = load_config(config_path=config)
    except Exception as e:
        click.echo(f"⚠️  Error loading config: {e}, using defaults")
        app_config = load_config()

    cache_manager = CacheManager(app_config.preprocessing.cache_dir)
    videos = cache_manager.list_cached_videos()

    if not videos:
        click.echo("No videos found in cache.")
        return

    click.echo("\n" + "=" * 80)
    click.echo("CACHED VIDEOS")
    click.echo("=" * 80)
    click.echo(
        f"{'Video Path':<40} {'Segments':<8} {'Frames':<8} {'Duration':<10} {'Size (MB)':<10}"
    )
    click.echo("-" * 80)

    for video in videos:
        duration = f"{video['duration']:.1f}s" if video["duration"] else "N/A"
        click.echo(
            f"{video['video_path'][-38:]:<40} "
            f"{video['audio_segments']:<8} "
            f"{video['visual_frames']:<8} "
            f"{duration:<10} "
            f"{video['file_size_mb']:.1f}"
        )

    click.echo("=" * 80)


@cli.command()
@click.argument("video_path", required=False)
@click.option(
    "--config", "-c", default="config.yaml", help="Path to configuration file"
)
@click.option("--all", is_flag=True, help="Clear all cached data")
def clear_cache(video_path, config, all):
    """Clear cache for a specific video or all videos."""

    try:
        app_config = load_config(config_path=config)
    except Exception as e:
        click.echo(f"⚠️  Error loading config: {e}, using defaults")
        app_config = load_config()

    cache_manager = CacheManager(app_config.preprocessing.cache_dir)

    if all:
        if click.confirm("Are you sure you want to clear all cached data?"):
            cache_manager.clear_cache()
            click.echo("All cached data cleared.")
    elif video_path:
        cache_manager.clear_cache(video_path)
        click.echo(f"Cache cleared for {video_path}")
    else:
        click.echo("Please specify a video path or use --all flag.")


@cli.command()
@click.argument("video_id")
@click.option(
    "--config", "-c", default="config.yaml", help="Path to configuration file"
)
def chat(video_id, config):
    """
    Start an interactive chat session about a specific video.

    VIDEO_ID is the unique identifier of the video you want to ask questions about.
    Make sure the video has been processed first using the 'process' command.

    All agent settings are configured via the config file.
    """
    click.echo(f"\n🎬 Starting chat session for video: {video_id}")
    click.echo(
        "Type your questions about the video. Type 'quit' or 'exit' to end the session.\n"
    )

    try:
        app_config = load_config(config_path=config)
        click.echo(f"📋 Loaded configuration from: {config}")

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, app_config.logging.level.upper()),
            format=app_config.logging.format,
            handlers=[
                logging.StreamHandler(sys.stdout)
                if app_config.logging.console_output
                else None,
                logging.FileHandler(app_config.logging.file_path)
                if app_config.logging.file_path
                else None,
            ],
        )

    except Exception as e:
        click.echo(f"⚠️  Error loading config: {e}, using defaults")
        app_config = load_config()

    # Initialize the agent
    try:
        agent = VideoAgent(
            vllm_base_url=app_config.agent.vllm_base_url,
            model_name=app_config.agent.model_name,
            db_path=app_config.agent.db_path,
            vlm_base_url=app_config.agent.vlm_base_url,
            vlm_model_name=app_config.agent.vlm_model_name,
            alm_base_url=getattr(
                app_config.agent, "alm_base_url", "http://localhost:8013/v1"
            ),
            alm_model_name=getattr(
                app_config.agent, "alm_model_name", "nvidia/audio-flamingo-3-hf"
            ),
            text_embedding_url=getattr(
                app_config.agent, "text_embedding_url", "http://localhost:8014/v1"
            ),
            visual_embedding_url=getattr(
                app_config.agent, "visual_embedding_url", "http://localhost:8018/v1"
            ),
            videos_dir=app_config.agent.videos_dir,
            video_search_paths=getattr(
                app_config.agent, "video_search_paths", [app_config.agent.videos_dir]
            ),
        )
        click.echo("✅ Connected to LLM server and loaded video database.")

        # Preload models for faster responses
        click.echo("🔄 Preloading models...")
        agent._get_search_executor()  # Load embedding models
        agent._get_video_refiner()  # Initialize video refiner
        click.echo("✅ Models loaded and ready.\n")

    except Exception as e:
        click.echo(f"❌ Error initializing agent: {e}")
        return

    # Interactive chat loop
    while True:
        try:
            # Get user input
            user_input = click.prompt("You", type=str, prompt_suffix=" 🎯 ").strip()

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "q"]:
                click.echo("\n👋 Goodbye! Thanks for using the video agent.")
                break

            # Check for clear history command
            if user_input.lower() in ["clear", "reset", "clear history"]:
                agent.clear_conversation_history(video_id)
                click.echo("\n🔄 Conversation history cleared.\n")
                continue

            # Skip empty input
            if not user_input:
                continue

            # Process the question
            click.echo("\n🤖 Assistant: ", nl=False)
            response = agent.chat_with_video(video_id, user_input)
            click.echo(response)
            click.echo()

        except (EOFError, KeyboardInterrupt):
            click.echo("\n\n👋 Goodbye! Thanks for using the video agent.")
            break
        except Exception as e:
            click.echo(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    cli()
