#!/usr/bin/env python
"""
Generate comprehensive dataset summary and statistics for the video benchmark dataset.
This script analyzes all processed videos and generates a detailed overview of the dataset.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DATASET_DIR
from caption_pipeline.utils.dataset_analyzer import DatasetAnalyzer
from caption_pipeline.utils.rich_console import get_console

rich_console = get_console()


def main():
    """Main function to generate and save dataset summary."""
    rich_console.print_header("Dataset Summary Generator", 
                            "Generating comprehensive dataset statistics and overview")
    
    analyzer = DatasetAnalyzer(DATASET_DIR)
    
    # Generate and save summary
    summary_file = analyzer.generate_and_save_summary()
    rich_console.print_success(f"Detailed summary available in: {summary_file}")


if __name__ == "__main__":
    main()