# Configuration settings for the caption pipeline

# Dataset storage configuration
DATASET_DIR = "./dataset"
VIDEO_DIR = f"{DATASET_DIR}/videos"
AUDIO_DIR = f"{DATASET_DIR}/audio"
CAPTIONS_DIR = f"{DATASET_DIR}/captions"
VIDEO_DESCRIPTIONS_DIR = f"{DATASET_DIR}/video_descriptions"
METADATA_DIR = f"{DATASET_DIR}/metadata"
MULTIMODAL_UNDERSTANDING_DIR = f"{DATASET_DIR}/multimodal_understanding"
KEY_EVENTS_DIR = f"{DATASET_DIR}/key_events"
FINAL_DIR = f"{DATASET_DIR}/final"

# Make VIDEO_DESCRIPTIONS_DIR available directly in config
# This is for backward compatibility with existing code
if 'VIDEO_DESCRIPTIONS_DIR' not in locals():
    VIDEO_DESCRIPTIONS_DIR = f"{DATASET_DIR}/video_descriptions"

# Database configuration
DATABASE_URI = "sqlite:///caption_pipeline.db"  # Use relative path, Flask will handle instance folder

# Video download configuration
VIDEO_IDS_FILE = "video_ids.txt"  # File containing YouTube video IDs
MIN_VIDEO_DURATION = 60  # In seconds
MAX_VIDEO_DURATION = 60000  # In seconds (100 minutes)

# Caption generation configuration
ASR_MODEL = "openai/whisper-large-v3"  # Multilingual ASR model with timestamp support
BATCH_SIZE = 16
GPU_DEVICE = 0  # Set to -1 for CPU only

# Video description configuration
ENABLE_VIDEO_DESCRIPTIONS = True  # Enable/disable video description generation
VIDEO_DESCRIPTION_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"  # Vision-language model for descriptions
# VIDEO_DESCRIPTION_MODEL = "OpenGVLab/InternVL3_5-14B-Instruct"  # Vision-language model for descriptions
VLLM_SERVER_URL = "http://localhost:8000/v1"  # vLLM server endpoint (VLM model)
LLM_SERVER_URL = "http://localhost:8001/v1"  # vLLM server endpoint (LLM model)
LLM_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # LLM model for descriptions

# Audio description configuration
ENABLE_AUDIO_DESCRIPTIONS = False  # Enable/disable audio description generation using Audio Flamingo 3
AUDIO_FLAMINGO_MODEL_PATH = "nvidia/audio-flamingo-3"  # Path to Audio Flamingo 3 model
QWEN_AUDIO_MODEL = "Qwen/Qwen2-Audio-7B"   # Audio model for descriptions
AUDIO_DESCRIPTION_BATCH_SIZE = 8  # Batch size for audio description processing
AUDIO_DESCRIPTION_TEXT_PROMPT = "Please describe the audio in detail, focusing on music, sounds, ambience, and any non-speech audio content."  # Text prompt for audio description
AUDIO_DESCRIPTIONS_DIR = f"{DATASET_DIR}/audio_descriptions"  # Directory for audio description outputs

# Web interface configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
SECRET_KEY = "change-this-to-a-secure-random-key"
SESSION_TYPE = "filesystem"

# User configuration
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "change-this-password"
ALLOW_REGISTRATION = True  # Set to False to disable new user registration
