#!/usr/bin/env python3
"""
Utility functions for the simplified video analysis pipeline.
"""

import json
import logging
import fcntl
from pathlib import Path
from typing import Dict, Any, List


def load_schema(schema_filename: str) -> Dict[str, Any]:
    """Load JSON schema from schemas directory."""
    schema_path = Path(__file__).parent / "schemas" / schema_filename
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompt(prompt_filename: str) -> str:
    """Load prompt template from prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / prompt_filename
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def save_jsonl(data: List[Dict[str, Any]], file_path: Path) -> None:
    """Save data to JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def append_jsonl(item: Dict[str, Any], file_path: Path) -> None:
    """Append a single item to JSONL file with file locking for thread safety."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp file for atomic write
    temp_file = file_path.with_suffix('.tmp')
    
    with open(temp_file, 'a', encoding='utf-8') as f:
        # Use file locking to prevent race conditions
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    # Atomically move temp file to final location
    temp_file.replace(file_path)


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    if not file_path.exists():
        return []
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging based on configuration."""
    log_config = config.get('system', {})
    log_level = config.get('processing', {}).get('log_level', 'INFO')
    log_format = log_config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('log_file', './logs/video_analysis.log')
    
    # Create log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
