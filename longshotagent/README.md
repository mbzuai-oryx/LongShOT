# Video Agent: Multimodal Video Understanding System

A multimodal RAG system for video understanding that combines audio transcription, visual embeddings, and LLM-powered conversational querying. Process videos into searchable vector embeddings, then ask natural language questions about their content.

## Architecture

```
Video Input
    ├── Audio Track ─> faster-whisper ─> Text Embeddings (all-MiniLM-L6-v2) ─> Qdrant
    └── Video Track ─> FFmpeg (5 FPS, 512x512) ─> SigLIP Embeddings ─> Qdrant
                                                                            │
User Query ─> VideoAgent (LLM + tool calling) ─────────────────────────────┘
                  ├── search_video: semantic search across embeddings
                  ├── refine_video: segment extraction + Whisper/VLM analysis
                  └── verify_claim: fact-check claims against video content
```

### Models

| Component | Model | Purpose |
|-----------|-------|---------|
| LLM | Configurable (default: Gemma-4-31B) | Reasoning, tool calling (vLLM) |
| VLM | Configurable (default: Gemma-4-31B) | Visual segment analysis (vLLM) |
| ALM | nvidia/audio-flamingo-3-hf | Audio language understanding |
| Audio | faster-whisper (large-v3) | Speech transcription |
| Visual | SigLIP ViT-B-16-SigLIP-512 | Image embeddings |
| Text | all-MiniLM-L6-v2 | Text embeddings for audio search |

## Installation

```bash
pip install -r requirements.txt
```

FFmpeg is required:
```bash
sudo apt install ffmpeg
```

## Quick Start

### 1. Start the model servers

```bash
bash llm.sh       # LLM server (port 8010)
bash vlm.sh       # VLM server (port 8011)
bash alm.sh       # ALM server (port 8013)
bash text_embed.sh # Text embedding server (port 8014)
bash visual_embed.sh # Visual embedding server (port 8018)
```

### 2. Process a video

```bash
python main.py process /path/to/video.mp4
python main.py process /path/to/video.mp4 --video-id my_video --force-reprocess
```

### 3. Chat about it

```bash
python main.py chat my_video
```

### 4. Or use the API server

```bash
python server.py --port 8012
```

## Configuration

All settings are managed via `config.yaml`. See the file for full documentation.

## How It Works

### Preprocessing Pipeline

1. **Audio extraction** -- FFmpeg extracts audio, faster-whisper transcribes into timestamped segments
2. **Frame extraction** -- FFmpeg samples frames at target FPS, scaled to 512x512
3. **Embedding generation** -- Text segments embedded with all-MiniLM-L6-v2, frames with SigLIP
4. **Storage** -- Embeddings stored in Qdrant vector database and cached locally

### Query Flow

1. User asks a question via CLI or API
2. LLM decides which tool to call (search_video, refine_video, verify_claim)
3. **search_video**: encodes query, runs vector similarity search across audio/visual embeddings
4. **refine_video**: extracts a specific video segment, analyzes with Whisper (audio) or VLM (visual)
5. **verify_claim**: fact-checks specific claims against video evidence
6. LLM synthesizes tool results into a response
