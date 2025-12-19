#!/usr/bin/env python3
"""
Script to parse Video-MME parquet file and extract long videos.
Creates a JSON file with long video data and analysis, and a TXT file with video IDs only.
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

def parse_parquet_and_extract_long_videos():
    """Parse parquet file and extract long duration videos with analysis."""
    
    # File paths
    parquet_file = Path("/data1/jaseel/caption_pipeline/video-mme/test-00000-of-00001.parquet")
    output_dir = Path("/data1/jaseel/caption_pipeline/video-mme")
    json_output = output_dir / "long_videos_data.json"
    txt_output = output_dir / "long_video_ids.txt"
    
    print(f"Loading parquet file: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    
    # Filter long duration videos
    long_videos_df = df[df['duration'] == 'long'].copy()
    
    print(f"Total videos: {len(df)}")
    print(f"Long duration videos: {len(long_videos_df)}")
    
    # Convert to list of dictionaries for JSON serialization
    # Handle numpy data types for JSON serialization
    long_videos_data = json.loads(long_videos_df.to_json(orient='records'))
    
    # Generate analysis
    analysis = {
        "total_videos_in_dataset": int(len(df)),
        "long_videos_count": int(len(long_videos_df)),
        "unique_long_video_ids": int(len(long_videos_df['video_id'].unique())),
        "unique_youtube_ids": int(len(long_videos_df['videoID'].unique())),
        "domains": {k: int(v) for k, v in long_videos_df['domain'].value_counts().items()},
        "sub_categories": {k: int(v) for k, v in long_videos_df['sub_category'].value_counts().items()},
        "task_types": {k: int(v) for k, v in long_videos_df['task_type'].value_counts().items()},
        "questions_per_video": {str(k): int(v) for k, v in long_videos_df['video_id'].value_counts().items()},
        "duration_distribution": {k: int(v) for k, v in df['duration'].value_counts().items()}
    }
    
    # Create final JSON structure
    output_data = {
        "metadata": {
            "description": "Long duration videos extracted from Video-MME dataset",
            "source_file": str(parquet_file),
            "extraction_date": pd.Timestamp.now().isoformat(),
            "columns": list(long_videos_df.columns)
        },
        "analysis": analysis,
        "long_videos": long_videos_data
    }
    
    # Save JSON file
    print(f"Saving JSON data to: {json_output}")
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Extract unique YouTube video IDs and save to TXT file
    unique_youtube_ids = sorted(long_videos_df['videoID'].unique())
    
    print(f"Saving YouTube video IDs to: {txt_output}")
    with open(txt_output, 'w', encoding='utf-8') as f:
        for video_id in unique_youtube_ids:
            f.write(f"{video_id}\n")
    
    # Print summary
    print("\n=== EXTRACTION SUMMARY ===")
    print(f"Total long videos extracted: {len(long_videos_df)}")
    print(f"Unique YouTube video IDs: {len(unique_youtube_ids)}")
    print(f"Domain distribution: {dict(long_videos_df['domain'].value_counts())}")
    print(f"Files created:")
    print(f"  - JSON with data and analysis: {json_output}")
    print(f"  - TXT with YouTube video IDs: {txt_output}")
    
    return output_data

if __name__ == "__main__":
    result = parse_parquet_and_extract_long_videos()