#!/usr/bin/env python3
"""
Simplified video analysis pipeline with high-performance processing using direct HTTP requests.
"""

import argparse
import asyncio
import json
import logging
import time
import yaml
from pathlib import Path
from typing import Dict, Any, List

import aiohttp
from tqdm import tqdm

from pipeline import VideoPipeline
from utils import load_schema, load_prompt, setup_logging


async def main_async(config_path: str, sample_size: int = None, stage: str = "all", debug_samples: int = None) -> None:
    """Main async processing function."""
    start_time = time.time()
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Add debug_samples to config for stages to access
    if debug_samples:
        config['debug_samples'] = debug_samples
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Extract configuration
    vllm_config = config['vllm']
    processing_config = config['processing']
    
    # Initialize HTTP session and connection timeout
    timeout = aiohttp.ClientTimeout(total=vllm_config.get('timeout_seconds', 120))
    connector = aiohttp.TCPConnector(limit=processing_config.get('max_concurrent_requests', 100))
    session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    
    # Setup pipeline
    pipeline = VideoPipeline(
        session=session,
        base_url=vllm_config['base_url'],
        model_name=vllm_config['model'],
        max_concurrent=processing_config.get('max_concurrent_requests', 100),
        config=config
    )
    
    # Find video files
    input_path = Path(processing_config['input_directory'])
    video_files = list(input_path.glob(processing_config['file_pattern']))
    
    if not video_files:
        logger.error(f"No files found matching {processing_config['file_pattern']} in {input_path}")
        return
    
    # Apply sample limit if specified
    if sample_size:
        video_files = video_files[:sample_size]
    
    print(f"\n🚀 DATA GENERATION PIPELINE")
    print(f"{'='*60}")
    print(f"📊 Processing stage: {stage.upper()}")
    print(f"📹 Processing {len(video_files)} videos")
    if debug_samples:
        print(f"🐛 Debug mode: limiting to {debug_samples} samples per stage")
    print(f"⚡ Max concurrent requests: {processing_config.get('max_concurrent_requests', 100)}")
    print(f"🔗 vLLM Server: {vllm_config['base_url']} (chat/completions)")
    print(f"{'='*60}")
    
    try:
        # Run pipeline based on stage
        if stage in ["analysis", "all"]:
            print(f"\n🔍 STAGE 1: SCENARIO ANALYSIS")
            await pipeline.process_analysis(video_files)
            if stage == "analysis":
                return
        
        if stage in ["question_types", "all"]:
            print(f"\n🎯 STAGE 2: QUESTION TYPES GENERATION")
            await pipeline.process_question_types()
            if stage == "question_types":
                return
        
        if stage in ["questions", "all"]:
            print(f"\n❓ STAGE 3: QUESTION GENERATION")
            await pipeline.process_questions()
            if stage == "questions":
                return
        
        if stage in ["answers", "all"]:
            print(f"\n💬 STAGE 4: ANSWER GENERATION")
            await pipeline.process_answers()
            if stage == "answers":
                return
        
        if stage in ["criteria", "all"]:
            print(f"\n📋 STAGE 5: CRITERIA GENERATION")
            await pipeline.process_criteria()
        
        if stage in ["finalize", "all"]:
            print(f"\n🎯 STAGE 6: DATASET FINALIZATION")
            await pipeline.finalize_dataset()
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"🎉 PIPELINE EXECUTION SUMMARY")
        print(f"{'='*70}")
        print(f"⏱️  Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"✅ Pipeline execution completed successfully!")
        print(f"{'='*70}")
    
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        logger.exception("Pipeline error")
    finally:
        await session.close()
    
    logger.info("Processing completed successfully")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Video Analysis System")
    parser.add_argument("--config", "-c", default="config.yaml", 
                       help="Configuration file path (default: config.yaml)")
    parser.add_argument("--sample", "-s", type=int, 
                       help="Process only N samples for testing")
    parser.add_argument("--debug-samples", "-d", type=int, 
                       help="Limit to N samples per stage for debugging")
    parser.add_argument("--stage", choices=["analysis", "question_types", "questions", "answers", "criteria", "finalize", "all"], 
                       default="all",
                       help="Processing stage to run (default: all)")
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file {args.config} not found")
        return 1
    
    try:
        asyncio.run(main_async(args.config, args.sample, args.stage, args.debug_samples))
        return 0
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
