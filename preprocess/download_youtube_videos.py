#!/usr/bin/env python3
"""
YouTube video download script using yt-dlp
Downloads videos from video_ids.txt that are not already in dataset_our_v2/videos/
"""

import os
import subprocess
import sys
from pathlib import Path
import re

def get_video_ids_from_file(file_path):
    """Extract video IDs from the video_ids.txt file"""
    video_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract just the video ID (11 characters for YouTube)
                video_id = line.split()[0]
                if len(video_id) == 11:
                    video_ids.append(video_id)
    return video_ids

def get_existing_videos(videos_dir):
    """Get list of already downloaded video IDs"""
    existing_ids = []
    videos_path = Path(videos_dir)
    if videos_path.exists():
        for video_file in videos_path.glob('*.mp4'):
            video_id = video_file.stem
            existing_ids.append(video_id)
    return existing_ids

def download_video(video_id, output_dir):
    """Download a single YouTube video using yt-dlp"""
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")
    
    cmd = [
        'yt-dlp',
        '--format', 'best[height<=720]',  # Download best quality up to 720p
        '--output', output_template,
        '--no-playlist',
        '--embed-subs',
        '--write-auto-sub',
        '--sub-lang', 'en',
        '--convert-subs', 'srt',
        '--cookies' , 'cookies.txt',
        '--progress',
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ Successfully downloaded: {video_id}")
            return True
        else:
            print(f"❌ Failed to download {video_id}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout downloading {video_id}")
        return False
    except Exception as e:
        print(f"❌ Error downloading {video_id}: {e}")
        return False

def main():
    # Paths
    video_ids_file = "video_ids.txt"
    videos_output_dir = "dataset/videos"
    
    # Create output directory if it doesn't exist
    os.makedirs(videos_output_dir, exist_ok=True)
    
    # Get video IDs to download
    print("📋 Reading video IDs from file...")
    all_video_ids = get_video_ids_from_file(video_ids_file)
    print(f"Found {len(all_video_ids)} video IDs in {video_ids_file}")
    
    # Get already downloaded videos
    print("📁 Checking existing downloads...")
    existing_ids = get_existing_videos(videos_output_dir)
    print(f"Found {len(existing_ids)} already downloaded videos")
    
    # Find missing videos
    missing_ids = [vid_id for vid_id in all_video_ids if vid_id not in existing_ids]
    print(f"🔍 Need to download {len(missing_ids)} missing videos")
    
    if not missing_ids:
        print("✅ All videos already downloaded!")
        return
    
    # Download missing videos
    successful = 0
    failed = 0
    
    print(f"\n📥 Starting downloads...")
    for i, video_id in enumerate(missing_ids, 1):
        print(f"\n[{i}/{len(missing_ids)}] Downloading {video_id}...")
        if download_video(video_id, videos_output_dir):
            successful += 1
        else:
            failed += 1
    
    print(f"\n📊 Download Summary:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Total videos in directory: {len(existing_ids) + successful}")

if __name__ == "__main__":
    main()