"""
Video metadata generator module for enhancing video metadata.
This module processes video descriptions to extract metadata like summaries and content categories.
"""

import os
import json
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import time
from datetime import datetime
import cv2

# Import project configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (LLM_SERVER_URL, METADATA_DIR, LLM_MODEL, VIDEO_DIR, VIDEO_DESCRIPTIONS_DIR,
                   MULTIMODAL_UNDERSTANDING_DIR)

# Import rich console utilities
from caption_pipeline.utils.rich_console import get_console

# Try to import vLLM components
try:
    from openai import OpenAI
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    OpenAI = None

# Set up logging
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'metadata_generator.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
rich_console = get_console()

# Constants for category classification
VIDEO_CATEGORIES = [
    "Documentary", "Sports", "Lifestyle", "Interview", "Food/Cooking", 
    "Travel", "Adventure", "Reality Show", "Talent Show", "Comedy", 
    "Educational", "Podcasts", "Series/Movies", "Tutorials", 
    "Surveillance", "Kids video", "Movie Trailers", "Makeup tutorials", 
    "News/Weather", "Music Videos", "Product Reviews/Unboxing", 
    "Live Streams", "Meetings/Conferences", "Security Footage", 
    "Vlogs", "Fitness/Workout", "Customer Support Calls", 
    "Dashcam Videos", "Emergency/Incident Footage"
]

# Prompts
SUMMARY_PROMPT = """
You are an expert content analyst specializing in precise, objective video summaries for benchmarking multimodal understanding across diverse video domains (e.g., sports, news, animation, documentaries). Your task is to create a concise, factual summary that captures the video’s core content, structure, and purpose without embellishment or speculation.

Video context:
{video_context}

ANALYSIS FRAMEWORK:
Content Overview:

Identify the video’s primary purpose, genre, and intended audience.
Classify the narrative structure (e.g., linear, episodic, instructional, narrative).
Highlight main topics, objectives, or messages based solely on provided data.
Note production quality (e.g., clarity, editing) relevant to the content.

Subjects and Roles:

Describe key subjects (e.g., people, characters, objects) and their roles.
Outline observable interactions or relationships without inferring motivations.
Specify any expertise or distinct traits explicitly shown.

Narrative Flow:

Summarize the chronological sequence of events or information.
Highlight key transitions, shifts, or focal points in the content.
Identify recurring elements (e.g., themes, visuals, or actions) grounded in the data.
Maintain logical progression without adding external context.

Context and Setting:

Describe the setting (e.g., location, environment) and its relevance to the content.
Note observable contextual elements (e.g., time, cultural markers) if present.
Mention production elements (e.g., overlays, audio) that support the narrative.

SYNTHESIS REQUIREMENTS:
Create a concise, structured summary (150–250 words) in 2–3 paragraphs that:

Opens with a clear statement of the video’s purpose, genre, and focus.
Details key events, subjects, and settings in chronological order, emphasizing transitions.
Concludes with the video’s primary takeaway or objective, grounded in the content.
Uses neutral, precise language, avoiding subjective terms (e.g., no “compelling,” “sophisticated”).
Eliminates repetitive or vague phrasing for clarity and reproducibility.
Ensures applicability to any video domain, avoiding genre-specific bias.
Integrates all elements (subjects, narrative, setting) into a cohesive, factual account.

Source Material:
{descriptions}
COMPREHENSIVE VIDEO SUMMARY:
"""

CATEGORY_PROMPT = """
You are an expert content taxonomist and media analyst with deep understanding of video genres, audience preferences, and content classification systems. Your task is to precisely categorize this video using sophisticated content analysis.

**AVAILABLE CATEGORIES:**
{categories}

{video_context}

**CLASSIFICATION METHODOLOGY:**

**Content Analysis:**
• Examine the video's primary purpose, format, and production style
• Identify target audience demographics and engagement patterns  
• Analyze narrative structure, presentation style, and content depth
• Assess educational vs. entertainment value and content sophistication

**Genre Recognition:**
• Determine primary and secondary genre characteristics
• Identify subgenres and niche category elements
• Recognize hybrid formats that blend multiple category types
• Consider cultural context and platform-specific conventions

**Audience Matching:**
• Analyze content appropriateness for different age groups
• Identify specific interest communities and fan bases
• Consider viewing context (education, entertainment, information, etc.)
• Assess content complexity and engagement level requirements

**Quality & Context Assessment:**
• Evaluate production values and professional quality markers
• Identify format-specific characteristics (tutorial steps, narrative arcs, etc.)
• Analyze cultural or temporal context that influences categorization
• Consider algorithmic and search optimization factors

**CLASSIFICATION REQUIREMENTS:**
Provide 1-3 most accurate categories as a comma-separated list. Consider:
- Primary category should represent the main content type and purpose
- Secondary categories should capture important subcategories or cross-genre elements
- Avoid over-categorization - select only clearly applicable categories
- Prioritize categories that best serve content discovery and audience matching
- Consider both explicit content features and implicit audience targeting

**Source Analysis:**
{summary}

**PRECISE CATEGORY CLASSIFICATION:**
"""


class VideoMetadataGenerator:
    """Class to generate enhanced metadata from video descriptions."""
    
    def __init__(self, api_base="http://localhost:8000/v1", 
                 model_name=LLM_MODEL,
                 max_workers=4):
        """Initialize VideoMetadataGenerator."""
        self.api_base = api_base
        self.model_name = model_name
        self.max_workers = max_workers
        
        # Initialize paths from config
        self.metadata_dir = METADATA_DIR
        self.video_dir = VIDEO_DIR
        self.video_descriptions_dir = VIDEO_DESCRIPTIONS_DIR
        self.multimodal_understanding_dir = MULTIMODAL_UNDERSTANDING_DIR
        
        # Load metadata
        self.metadata_file = os.path.join(self.metadata_dir, 'video_metadata.csv')
        self._load_metadata()
        
        if not VLLM_AVAILABLE:
            rich_console.print_warning("vLLM not available. Metadata generation may be limited.")
        
        # Test server connection
        try:
            client = self._create_client()
            if client:
                rich_console.print_success(f"Successfully connected to vLLM server at {self.api_base}")
            else:
                rich_console.print_warning("Failed to create vLLM client. Text generation features will be disabled.")
        except Exception as e:
            rich_console.print_error(f"Error connecting to vLLM server: {e}")
    
    def _create_client(self):
        """Create a new OpenAI client instance for this thread."""
        try:
            client = OpenAI(api_key="EMPTY", base_url=self.api_base)
            return client
        except Exception as e:
            rich_console.print_error(f"Failed to create OpenAI client: {e}")
            return None
    
    def _load_metadata(self):
        """Load the existing metadata CSV file."""
        if os.path.exists(self.metadata_file):
            self.metadata_df = pd.read_csv(self.metadata_file)
            rich_console.print_info(f"Loaded metadata with {len(self.metadata_df)} videos")
        else:
            rich_console.print_warning(f"Metadata file {self.metadata_file} not found. Creating empty DataFrame.")
            self.metadata_df = pd.DataFrame(columns=[
                'video_id', 'title', 'channel', 'duration', 'view_count', 
                'publish_date', 'description', 'tags', 'download_date',
                'file_path', 'status', 'caption_path'
            ])
    
    def _save_metadata(self):
        """Save the updated metadata back to the CSV file."""
        self.metadata_df.to_csv(self.metadata_file, index=False)
        rich_console.print_info(f"Saved updated metadata with {len(self.metadata_df)} videos")
    
    def _load_multimodal_understanding(self, video_id):
        """Load multimodal understanding data for a video."""
        # Try aligned version first, fall back to non-aligned
        aligned_file = os.path.join(self.multimodal_understanding_dir, f"{video_id}_multimodal_understanding_aligned.json")
        non_aligned_file = os.path.join(self.multimodal_understanding_dir, f"{video_id}_multimodal_understanding.json")
        
        try:
            if os.path.exists(aligned_file):
                with open(aligned_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif os.path.exists(non_aligned_file):
                logger.debug(f"Using non-aligned multimodal understanding for {video_id}")
                with open(non_aligned_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.debug(f"No multimodal understanding data found for video {video_id}")
                return None
        except Exception as e:
            logger.debug(f"Error loading multimodal understanding data for video {video_id}: {e}")
            return None

    def _load_video_descriptions(self, video_id):
        """Load video descriptions from the JSON file."""
        descriptions_file = os.path.join(self.video_descriptions_dir, f"{video_id}_descriptions_aligned.json")
        if not os.path.exists(descriptions_file):
            rich_console.print_error(f"Descriptions file not found for video {video_id}")
            return None
        
        try:
            with open(descriptions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            rich_console.print_error(f"Error loading descriptions for video {video_id}: {e}")
            return None
    
    def _get_video_context(self, video_id):
        """Get video context information (title and channel) from metadata."""
        if not video_id:
            return ""
        
        try:
            video_data = self.metadata_df[self.metadata_df['video_id'] == video_id]
            if len(video_data) == 0:
                return ""
            
            title = None
            channel = None
            
            if 'title' in video_data.columns:
                title_value = video_data['title'].values[0]
                if pd.notna(title_value) and str(title_value).strip():
                    title = str(title_value).strip()
            
            if 'channel' in video_data.columns:
                channel_value = video_data['channel'].values[0]
                if pd.notna(channel_value) and str(channel_value).strip():
                    channel = str(channel_value).strip()
            
            if title or channel:
                video_context = "Video Information:\n"
                if title:
                    video_context += f"- Title: {title}\n"
                if channel:
                    video_context += f"- Channel: {channel}\n"
                return video_context
            
            return ""
        except Exception as e:
            logger.debug(f"Error getting video context for {video_id}: {e}")
            return ""
    
    def _generate_summary(self, descriptions, multimodal_understanding=None, client=None, video_id=None):
        """Generate a summary from video descriptions using vLLM."""
        if not client:
            client = self._create_client()
            if not client:
                return "Summary generation failed due to server connection issues."
        
        # Use comprehensive multimodal understanding if available (preferred)
        if multimodal_understanding and multimodal_understanding.get('segments'):
            multimodal_descriptions = []
            for segment in multimodal_understanding.get('segments', []):
                if 'multimodal_understanding' in segment and segment['multimodal_understanding']:
                    multimodal_descriptions.append(segment['multimodal_understanding'])
            
            if multimodal_descriptions:
                # Use the multimodal understanding
                combined_descriptions = "\n\n".join(multimodal_descriptions)
                description_source = "multimodal understanding"
            else:
                # Fallback to individual modalities
                combined_descriptions = self._fallback_to_individual_modalities(descriptions)
                description_source = "individual modalities (fallback)"
        else:
            # Fallback to individual modalities
            combined_descriptions = self._fallback_to_individual_modalities(descriptions)
            description_source = "individual modalities"
        
        if not combined_descriptions or combined_descriptions.strip() == "":
            return "No descriptions available to generate summary."
        
        # Limit length to avoid token limitations
        if len(combined_descriptions) > 15000:  # Arbitrary limit to avoid token issues
            combined_descriptions = combined_descriptions[:15000] + "...[truncated for length]"
        
        # Get video context from metadata
        video_context = self._get_video_context(video_id)
        
        prompt = SUMMARY_PROMPT.format(
            video_context=video_context,
            descriptions=combined_descriptions
        )
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant that creates concise video summaries based on {description_source}."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            rich_console.print_error(f"Error generating summary: {e}")
            return f"Summary generation failed: {str(e)}"
    
    def _fallback_to_individual_modalities(self, descriptions):
        """Fallback method to combine individual modality descriptions."""
        visual_descriptions = []
        for segment in descriptions.get('segments', []):
            if 'visual_description' in segment and segment['visual_description']:
                # Include audio text if available for better context
                if 'audio_text' in segment and segment['audio_text'] and not segment.get('is_silent', False):
                    visual_descriptions.append(f"Visual: {segment['visual_description']}\nAudio: {segment['audio_text']}")
                else:
                    visual_descriptions.append(segment['visual_description'])
        
        return "\n\n".join(visual_descriptions)
    
    def _classify_categories(self, summary, client=None, video_id=None):
        """Classify the video into predefined categories based on the summary."""
        if not client:
            client = self._create_client()
            if not client:
                return "Uncategorized"
        
        # Get video context from metadata
        video_context = self._get_video_context(video_id)
        
        prompt = CATEGORY_PROMPT.format(
            categories="\n".join([f"- {cat}" for cat in VIDEO_CATEGORIES]),
            video_context=video_context,
            summary=summary
        )
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies videos into appropriate categories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=100,
            )
            
            categories = response.choices[0].message.content.strip()
            # Clean up and validate categories
            category_list = [cat.strip() for cat in categories.split(',')]
            valid_categories = [cat for cat in category_list if cat in VIDEO_CATEGORIES]
            
            if not valid_categories:
                return "Uncategorized"
            return ", ".join(valid_categories)
        except Exception as e:
            rich_console.print_error(f"Error classifying categories: {e}")
            return "Uncategorized"
    
    def _get_video_duration(self, video_id):
        """Get the duration of a video in seconds."""
        # First check if duration is already in metadata
        video_data = self.metadata_df[self.metadata_df['video_id'] == video_id]
        if len(video_data) > 0 and 'duration' in video_data.columns:
            duration = video_data['duration'].values[0]
            if pd.notna(duration) and duration > 0:
                return duration
        
        # Try to get duration from video file
        video_path = None
        if len(video_data) > 0 and 'file_path' in video_data.columns:
            potential_path = video_data['file_path'].values[0]
            if os.path.exists(potential_path):
                video_path = potential_path
        
        # Check common video file patterns if path not found
        if not video_path:
            for ext in ['.mp4', '.avi', '.mkv', '.webm']:
                potential_path = os.path.join(self.video_dir, f"{video_id}{ext}")
                if os.path.exists(potential_path):
                    video_path = potential_path
                    break
        
        if not video_path or not os.path.exists(video_path):
            rich_console.print_warning(f"Video file not found for {video_id}")
            return None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                rich_console.print_error(f"Could not open video {video_path}")
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if fps > 0 and total_frames > 0:
                duration = total_frames / fps
                return duration
            return None
        except Exception as e:
            rich_console.print_error(f"Error getting video duration: {e}")
            return None
    
    def process_video(self, video_id):
        """Process metadata for a single video."""
        rich_console.print_info(f"Generating metadata for video {video_id}")
        
        # Try to load multimodal understanding first (preferred)
        multimodal_understanding = self._load_multimodal_understanding(video_id)
        
        # Load video descriptions as fallback
        descriptions = self._load_video_descriptions(video_id)
        if not descriptions and not multimodal_understanding:
            rich_console.print_warning(f"No descriptions or multimodal understanding found for video {video_id}")
            return False
        
        client = self._create_client()
        if not client:
            rich_console.print_error("Failed to create vLLM client for metadata generation")
            return False
        
        try:
            # Generate summary using multimodal understanding if available, otherwise fallback to descriptions
            if multimodal_understanding:
                summary = self._generate_summary(descriptions, multimodal_understanding, client, video_id)
                rich_console.print_success(f"Generated summary for video {video_id} using comprehensive multimodal understanding")
            else:
                summary = self._generate_summary(descriptions, None, client, video_id)
                rich_console.print_success(f"Generated summary for video {video_id} using individual modalities")
            
            # Classify categories
            categories = self._classify_categories(summary, client, video_id)
            rich_console.print_success(f"Classified video {video_id} into categories: {categories}")
            
            # Get video duration
            duration = self._get_video_duration(video_id)
            if duration:
                rich_console.print_info(f"Video {video_id} duration: {duration:.2f} seconds")
            
            # Update metadata
            self._update_video_metadata(video_id, {
                'content_categories': categories,
                'duration': duration if duration else None,
                # Replace the original description with the AI-generated summary
                'original_description': self.metadata_df.loc[self.metadata_df['video_id'] == video_id, 'description'].values[0] if video_id in self.metadata_df['video_id'].values else None,
                'description': summary,
                'data_source': 'multimodal_understanding' if multimodal_understanding else 'individual_modalities'
            })
            
            return True
        except Exception as e:
            rich_console.print_error(f"Error processing metadata for video {video_id}: {e}")
            return False
    
    def _update_video_metadata(self, video_id, metadata_updates):
        """Update the metadata for a specific video."""
        # Check if video exists in metadata
        if video_id not in self.metadata_df['video_id'].values:
            rich_console.print_warning(f"Video {video_id} not found in metadata")
            return False
        
        # Update the metadata
        for key, value in metadata_updates.items():
            if key in self.metadata_df.columns:
                self.metadata_df.loc[self.metadata_df['video_id'] == video_id, key] = value
            else:
                # Add new column if it doesn't exist
                self.metadata_df[key] = None
                self.metadata_df.loc[self.metadata_df['video_id'] == video_id, key] = value
        
        # Save the updated metadata
        self._save_metadata()
        return True
    
    def process_videos(self, video_ids=None, use_concurrent=True):
        """Process metadata for multiple videos."""
        if not video_ids:
            # Get all videos that have descriptions but need metadata
            descriptions_files = [f[:-26] for f in os.listdir(self.video_descriptions_dir) 
                                if f.endswith('_descriptions_aligned.json')]
            video_ids = descriptions_files
        
        total_videos = len(video_ids)
        
        rich_console.print_component_header("Metadata Generation", f"Processing {total_videos} videos")
        
        # Create rich progress bar
        progress, task_id = rich_console.create_metadata_progress(total_videos)
        
        with progress:
            results = {}
            
            if use_concurrent and total_videos > 1 and self.max_workers > 1:
                # Process videos concurrently
                with ThreadPoolExecutor(max_workers=min(self.max_workers, total_videos)) as executor:
                    # Submit all tasks
                    futures = {executor.submit(self.process_video, vid): vid for vid in video_ids}
                    
                    # Process results as they complete
                    for future in as_completed(futures):
                        video_id = futures[future]
                        try:
                            result = future.result()
                            results[video_id] = result
                            if result:
                                rich_console.print_success(f"✓ Completed metadata for {video_id}")
                            else:
                                rich_console.print_error(f"✗ Failed metadata for {video_id}")
                            progress.update(task_id, advance=1)
                        except Exception as e:
                            rich_console.print_error(f"✗ Error processing video {video_id}: {e}")
                            results[video_id] = False
                            progress.update(task_id, advance=1)
            else:
                # Process videos sequentially
                for video_id in video_ids:
                    result = self.process_video(video_id)
                    results[video_id] = result
                    if result:
                        rich_console.print_success(f"✓ Completed metadata for {video_id}")
                    else:
                        rich_console.print_error(f"✗ Failed metadata for {video_id}")
                    progress.update(task_id, advance=1)
        
        # Summarize results
        successful = sum(1 for result in results.values() if result)
        
        rich_console.print_completion_message("Metadata Generation", {
            'total': total_videos,
            'successful': successful,
            'duration': 0  # Duration calculated externally
        })
        
        return results


def main():
    """Main function for testing the metadata generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate enhanced metadata from video descriptions')
    parser.add_argument('--video-id', type=str, help='Specific video ID to process')
    parser.add_argument('--api-base', type=str, default=LLM_SERVER_URL, help='vLLM server API base URL')
    parser.add_argument('--model', type=str, default=LLM_MODEL, help='Language model to use')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of concurrent workers')
    parser.add_argument('--no-concurrent', action='store_true', help='Disable concurrent processing')
    
    args = parser.parse_args()
    
    # Initialize metadata generator
    generator = VideoMetadataGenerator(
        api_base=args.api_base,
        model_name=args.model,
        max_workers=args.max_workers
    )
    
    use_concurrent = not args.no_concurrent
    
    if args.video_id:
        # Process a single video
        success = generator.process_video(args.video_id)
        if success:
            print(f"Successfully generated metadata for video {args.video_id}")
        else:
            print(f"Failed to generate metadata for video {args.video_id}")
    else:
        # Process all videos
        generator.process_videos(use_concurrent=use_concurrent)


if __name__ == "__main__":
    main()
