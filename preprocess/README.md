# Video Preprocessing Pipeline

This directory contains the preprocessing pipeline for the LongShOT dataset. The pipeline downloads videos from YouTube and extracts comprehensive multimodal information including video descriptions, audio descriptions, speech transcriptions, and metadata.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Pipeline Stages](#pipeline-stages)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Stage-by-Stage Execution](#stage-by-stage-execution)
  - [Individual Component Execution](#individual-component-execution)
- [Output Structure](#output-structure)

## Overview

The preprocessing pipeline consists of two main processing phases:

1. **VLM Stages**: Video download, audio extraction, caption generation, video descriptions, and audio descriptions
2. **LLM Stages**: Temporal alignment, multimodal understanding, key events extraction, metadata generation, and final consolidation

The pipeline is designed to process long-form videos (1-100+ minutes) and extract rich multimodal information for downstream tasks.

## Installation

1. **Create a conda environment:**

```bash
conda create -n longshot python=3.11 -y
conda activate longshot
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> **Note:** If you encounter CUDNN issues:
> ```bash
> conda install -c conda-forge cudnn=8
> ```

## Configuration

### Main Configuration File (`config.py`)

The `config.py` file contains all pipeline settings:

```python
# Dataset directories
DATASET_DIR = "./dataset"
VIDEO_DIR = f"{DATASET_DIR}/videos"
AUDIO_DIR = f"{DATASET_DIR}/audio"
CAPTIONS_DIR = f"{DATASET_DIR}/captions"
VIDEO_DESCRIPTIONS_DIR = f"{DATASET_DIR}/video_descriptions"
AUDIO_DESCRIPTIONS_DIR = f"{DATASET_DIR}/audio_descriptions"
MULTIMODAL_UNDERSTANDING_DIR = f"{DATASET_DIR}/multimodal_understanding"
KEY_EVENTS_DIR = f"{DATASET_DIR}/key_events"
METADATA_DIR = f"{DATASET_DIR}/metadata"
FINAL_DIR = f"{DATASET_DIR}/final"

# Model configurations
VIDEO_DESCRIPTION_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
LLM_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
ASR_MODEL = "openai/whisper-large-v3"
AUDIO_FLAMINGO_MODEL_PATH = "nvidia/audio-flamingo-3"

# Server endpoints
VLLM_SERVER_URL = "http://localhost:8000/v1"  # VLM server
LLM_SERVER_URL = "http://localhost:8001/v1"   # LLM server

# Video constraints
MIN_VIDEO_DURATION = 60      # seconds
MAX_VIDEO_DURATION = 60000   # seconds (100 minutes)
```

### Video Selection

Edit `video_ids.txt` to specify which YouTube videos to process:

```
VIDEO_ID_1    Optional description
VIDEO_ID_2    Optional description
# Comments are allowed
```

Each line should contain an 11-character YouTube video ID.

### Download Audio Flamingo Model

Download the Audio Flamingo model required for audio description generation:

```bash
hf download nvidia/audio-flamingo-3 --local-dir ./MODELS/audio-flamingo-3/
```

### YouTube Authentication

To download videos, you need to provide YouTube cookies:

1. Install the [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) Chrome extension
2. Navigate to YouTube in your browser
3. Click the extension icon and export cookies
4. Save the cookies to `preprocess/cookies.txt`

## Pipeline Stages

### VLM Stages (Vision-Language Model)

These stages use the vision-language model for visual understanding:

1. **Video Download**: Downloads videos from YouTube based on `video_ids.txt`
2. **Audio Extraction**: Extracts audio tracks from videos in WAV format
3. **Caption Generation**: Generates timestamped speech transcriptions using Whisper Large V3
4. **Video Descriptions**: Creates detailed visual descriptions for 5-second video segments
5. **Audio Descriptions**: Generates descriptions of non-speech audio (music, sound effects, ambient sounds)

### LLM Stages (Language Model)

These stages use the language model for consolidation and reasoning:

6. **Video Description Alignment**: Ensures temporal continuity across video description segments
7. **Audio Description Alignment**: Aligns audio descriptions with video context
8. **Multimodal Understanding**: Combines captions, video descriptions, and audio descriptions into comprehensive segment understanding
9. **Multimodal Understanding Alignment**: Refines multimodal understanding for temporal coherence
10. **Key Events Generation**: Extracts key events and timestamps from aligned descriptions
11. **Metadata Generation**: Creates video-level summaries, categories, and metadata
12. **Final Consolidation**: Combines all outputs into training-ready JSONL format

## Usage

### Quick Start

The pipeline is designed to run in two sequential phases:

#### Phase 1: VLM Stages

Start the VLM server in a separate terminal:

```bash
./vllm_start.sh vlm
```

This starts the Qwen2.5-VL-32B model on port 8000 with:
- Tensor parallel size: 4 (using 4 GPUs)
- GPU memory utilization: 0.75
- Max model length: 65536 tokens
- Max sequences: 32

Then run the VLM processing stages:

```bash
python run.py vlm-stages
```

This executes:
1. Video download from YouTube
2. Audio extraction and preprocessing
3. Speech transcription with Whisper
4. Video description generation (visual analysis)
5. Audio description generation (audio content analysis)

#### Phase 2: LLM Stages

After VLM stages complete, start the LLM server in a separate terminal:

```bash
./vllm_start.sh llm
```

This starts the Qwen3-30B model on port 8001 with:
- Tensor parallel size: 4 (using 4 GPUs)
- GPU memory utilization: 0.95
- Max model length: 65536 tokens
- Max sequences: 64

Then run the LLM processing stages:

```bash
python run.py llm-stages
```

This executes:
1. Video description alignment (temporal refinement)
2. Audio description alignment (contextual refinement)
3. Multimodal understanding generation (fusion)
4. Multimodal understanding alignment (coherence)
5. Key events extraction
6. Metadata generation
7. Final JSONL consolidation

### Stage-by-Stage Execution

For more control, run individual stages:

#### 1. Download Videos

```bash
python download_youtube_videos.py
```

This script:
- Reads video IDs from `video_ids.txt`
- Checks for already downloaded videos
- Downloads missing videos up to 720p quality
- Embeds English subtitles if available
- Saves to `dataset/videos/{VIDEO_ID}.mp4`

**Options:**
- Videos are automatically skipped if already present
- Failed downloads are logged with error messages

#### 2. Generate Video Descriptions

Requires: VLM server running on port 8000

```bash
python run.py video-descriptions [OPTIONS]
```

**Options:**
- `--video-ids VIDEO_ID1 VIDEO_ID2`: Process specific videos
- `--max-videos N`: Limit processing to N videos
- `--max-workers N`: Number of concurrent workers (default: 8)

**Example:**
```bash
# Process all videos with 8 workers
python run.py video-descriptions

# Process specific videos
python run.py video-descriptions --video-ids abc123xyz def456uvw

# Process up to 10 videos with 4 workers
python run.py video-descriptions --max-videos 10 --max-workers 4
```

**Output:** `dataset/video_descriptions/{VIDEO_ID}_descriptions.json`

Each segment contains:
- `segment_id`: Unique identifier
- `start_time`: Start timestamp in seconds
- `end_time`: End timestamp in seconds
- `visual_description`: Detailed visual analysis
- `frames`: Frame indices used for analysis

#### 3. Generate Audio Descriptions

Requires: Videos with completed video descriptions

```bash
python run.py audio-descriptions [OPTIONS]
```

**Options:**
- `--video-ids VIDEO_ID1 VIDEO_ID2`: Process specific videos
- `--max-videos N`: Limit processing to N videos
- `--audio-model-path PATH`: Custom Audio Flamingo model path
- `--audio-batch-size N`: Batch size (default: 8)
- `--audio-gpus N`: Number of GPUs to use

**Example:**
```bash
# Process all videos needing audio descriptions
python run.py audio-descriptions

# Process with custom batch size
python run.py audio-descriptions --audio-batch-size 4
```

**Output:** `dataset/audio_descriptions/{VIDEO_ID}_audio_descriptions.json`

Each segment contains:
- `segment_id`: Unique identifier
- `start_time`: Start timestamp
- `end_time`: End timestamp
- `audio_description`: Description of non-speech audio
- `audio_type`: Classification (music, effects, ambient, etc.)

#### 4. Align Video Descriptions

Requires: LLM server running on port 8001

```bash
python run.py video-description-alignment [OPTIONS]
```

**Options:**
- `--video-ids VIDEO_ID1 VIDEO_ID2`: Process specific videos
- `--max-videos N`: Limit processing to N videos
- `--max-workers N`: Number of concurrent workers (default: 8)
- `--model MODEL_NAME`: Override model from config

**Example:**
```bash
# Align all video descriptions
python run.py video-description-alignment

# Align specific videos with 4 workers
python run.py video-description-alignment --video-ids abc123 --max-workers 4
```

**Output:** `dataset/video_descriptions/{VIDEO_ID}_descriptions_aligned.json`

The alignment process:
- Analyzes context from neighboring segments
- Ensures smooth transitions between segments
- Resolves ambiguities using temporal context
- Maintains visual continuity

#### 5. Align Audio Descriptions

Requires: LLM server running on port 8001

```bash
python run.py audio-description-alignment [OPTIONS]
```

**Options:**
- `--video-ids VIDEO_ID1 VIDEO_ID2`: Process specific videos
- `--max-videos N`: Limit processing to N videos
- `--max-workers N`: Concurrent workers (default: 8)
- `--context-segments N`: Context window size (default: 3)

**Example:**
```bash
# Align all audio descriptions
python run.py audio-description-alignment

# Align with larger context window
python run.py audio-description-alignment --context-segments 5
```

**Output:** `dataset/audio_descriptions/{VIDEO_ID}_audio_descriptions_aligned.json`

#### 6. Generate Multimodal Understanding

Requires: LLM server running on port 8001

```bash
python run.py multimodal-understanding [OPTIONS]
```

**Options:**
- `--video-ids VIDEO_ID1 VIDEO_ID2`: Process specific videos
- `--max-videos N`: Limit processing to N videos
- `--max-workers N`: Concurrent workers (default: 8)
- `--sequential`: Disable concurrent processing

**Example:**
```bash
# Generate for all videos
python run.py multimodal-understanding

# Process sequentially (for debugging)
python run.py multimodal-understanding --sequential
```

**Output:** `dataset/multimodal_understanding/{VIDEO_ID}_understanding.json`

Each segment contains:
- `segment_id`: Unique identifier
- `start_time` / `end_time`: Temporal bounds
- `visual_description`: Aligned visual information
- `audio_description`: Aligned audio information
- `speech_transcript`: Synchronized captions
- `comprehensive_understanding`: Unified multimodal description

#### 7. Align Multimodal Understanding

Requires: LLM server running on port 8001

```bash
python run.py multimodal-understanding-alignment [OPTIONS]
```

**Options:**
- `--video-ids VIDEO_ID1 VIDEO_ID2`: Process specific videos
- `--max-videos N`: Limit processing to N videos
- `--max-workers N`: Concurrent workers (default: 8)

**Output:** `dataset/multimodal_understanding/{VIDEO_ID}_understanding_aligned.json`

#### 8. Generate Metadata

Requires: LLM server running on port 8001

```bash
python run.py metadata [OPTIONS]
```

**Options:**
- `--video-ids VIDEO_ID1 VIDEO_ID2`: Process specific videos
- `--max-videos N`: Limit processing to N videos
- `--max-workers N`: Concurrent workers (default: 8)

**Example:**
```bash
# Generate metadata for all videos
python run.py metadata

# Generate for specific videos
python run.py metadata --video-ids abc123 def456
```

**Output:** `dataset/metadata/{VIDEO_ID}_metadata.json`

Metadata includes:
- `summary`: Comprehensive video summary
- `categories`: Topical categorization
- `key_moments`: Important timestamps
- `content_type`: Video classification
- `themes`: Identified themes and topics
- `duration`: Video length in seconds

#### 9. Consolidate Final Dataset

```bash
python run.py consolidate [OPTIONS]
```

**Options:**
- `--video-ids VIDEO_ID1 VIDEO_ID2`: Consolidate specific videos
- `--max-videos N`: Limit to N videos

**Example:**
```bash
# Consolidate all processed videos
python run.py consolidate

# Consolidate specific videos
python run.py consolidate --video-ids abc123 def456
```

**Output:** `dataset/final/{VIDEO_ID}.jsonl`

Each line in the JSONL file contains:
```json
{
  "video_id": "abc123xyz",
  "segment_id": "abc123xyz_seg_0001",
  "start_time": 0.0,
  "end_time": 5.0,
  "visual_description": "...",
  "audio_description": "...",
  "speech_transcript": "...",
  "comprehensive_understanding": "...",
  "metadata": {...}
}
```

#### 10. Generate Dataset Summary

```bash
python run.py summary
```

**Output:** `dataset/dataset_summary.json`

Summary statistics include:
- Total number of videos processed
- Total duration of all videos
- Average video length
- Number of segments generated
- Processing completion rates per stage
- File size statistics

### Individual Component Execution

For development or debugging, run standalone scripts:

```bash
# Video descriptions only
python run_video_descriptions.py --max-videos 5

# Audio descriptions only
python run_audio_descriptions.py --max-videos 5

# Video description alignment only
python run_audio_description_alignment.py --video-ids abc123

# Multimodal understanding only
python run_multimodal_understanding.py --max-workers 4

# Multimodal alignment only
python run_multimodal_understanding_alignment.py

# Key events generation only
python run_key_events_generation.py

# Metadata generation only
python run_metadata_generation.py --max-videos 10
```

## Output Structure

The pipeline creates the following directory structure:

```
dataset/
├── videos/                          # Downloaded video files
│   └── {VIDEO_ID}.mp4
├── audio/                          # Extracted audio files
│   └── {VIDEO_ID}.wav
├── captions/                       # Speech transcriptions
│   └── {VIDEO_ID}_captions.json
├── video_descriptions/             # Visual descriptions
│   ├── {VIDEO_ID}_descriptions.json
│   └── {VIDEO_ID}_descriptions_aligned.json
├── audio_descriptions/             # Audio descriptions
│   ├── {VIDEO_ID}_audio_descriptions.json
│   └── {VIDEO_ID}_audio_descriptions_aligned.json
├── multimodal_understanding/       # Fused understanding
│   ├── {VIDEO_ID}_understanding.json
│   └── {VIDEO_ID}_understanding_aligned.json
├── key_events/                     # Extracted key events
│   └── {VIDEO_ID}_key_events.json
├── metadata/                       # Video-level metadata
│   └── {VIDEO_ID}_metadata.json
├── final/                          # Training-ready JSONL
│   └── {VIDEO_ID}.jsonl
└── dataset_summary.json            # Overall statistics
```

### File Format Details

#### Captions Format (`captions/{VIDEO_ID}_captions.json`)

```json
{
  "video_id": "abc123xyz",
  "duration": 120.5,
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "Welcome to this video tutorial.",
      "words": [
        {"word": "Welcome", "start": 0.0, "end": 0.5},
        {"word": "to", "start": 0.5, "end": 0.7}
      ]
    }
  ]
}
```

#### Video Descriptions Format

```json
{
  "video_id": "abc123xyz",
  "segments": [
    {
      "segment_id": "abc123xyz_seg_0001",
      "start_time": 0.0,
      "end_time": 5.0,
      "visual_description": "A person sitting at a desk with a laptop...",
      "frames": [0, 30, 60, 90, 120],
      "scene_type": "indoor",
      "visual_elements": ["person", "laptop", "desk"]
    }
  ]
}
```

#### Final JSONL Format (`final/{VIDEO_ID}.jsonl`)

Each line is a complete segment:

```json
{
  "video_id": "abc123xyz",
  "segment_id": "abc123xyz_seg_0001",
  "start_time": 0.0,
  "end_time": 5.0,
  "duration": 5.0,
  "visual_description": "A person sitting at a desk...",
  "audio_description": "Soft background music playing...",
  "speech_transcript": "Welcome to this tutorial.",
  "comprehensive_understanding": "The video begins with a person at a desk...",
  "key_events": ["Video introduction", "Tutorial begins"],
  "metadata": {
    "scene_type": "indoor",
    "content_type": "tutorial",
    "visual_elements": ["person", "laptop"],
    "audio_elements": ["speech", "music"]
  }
}
```
