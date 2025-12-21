"""
Utility functions for text processing.
"""

import re
import unicodedata
import string
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Standard punctuation marks
ALL_PUNCTUATIONS = string.punctuation


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from text.

    Args:
        text: Input text

    Returns:
        Text without punctuation
    """
    if not text:
        return ""

    # Create translation table
    translator = str.maketrans('', '', ALL_PUNCTUATIONS)

    # Apply translation
    return text.translate(translator)


def split_text_into_sentences(text: str, max_length: int = 100) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Input text to split
        max_length: Maximum character length per sentence

    Returns:
        List of sentence strings
    """
    if not text or len(text.strip()) == 0:
        return []

    text = text.strip()

    # Sentence ending patterns
    sentence_endings = [
        '.',   # Period
        '!',   # Exclamation mark
        '?',   # Question mark
    ]

    # Create pattern for sentence splitting
    # Split on sentence endings followed by whitespace or end of string
    pattern = '([' + ''.join(re.escape(ending) for ending in sentence_endings) + '])'

    # Split while keeping the delimiters
    parts = re.split(pattern, text)

    sentences = []
    current_sentence = ""

    i = 0
    while i < len(parts):
        part = parts[i].strip()

        if not part:
            i += 1
            continue

        # If this is a sentence ending punctuation
        if part in sentence_endings:
            current_sentence += part
            # Look ahead to see if there's more content
            if i + 1 < len(parts) and parts[i + 1].strip():
                # End current sentence and start new one
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                # This is the end of the text
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
        else:
            # Regular text part
            current_sentence += part

        i += 1

    # Add any remaining text
    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    # Post-process: split overly long sentences
    final_sentences = []
    for sentence in sentences:
        if len(sentence) <= max_length:
            final_sentences.append(sentence)
        else:
            # Split long sentence at natural break points
            split_sentences = split_long_sentence(sentence, max_length)
            final_sentences.extend(split_sentences)

    # Filter out very short sentences (likely artifacts)
    final_sentences = [s for s in final_sentences if len(s.strip()) > 3]

    return final_sentences


def split_long_sentence(sentence: str, max_length: int) -> List[str]:
    """
    Split a long sentence into smaller parts at natural break points.

    Args:
        sentence: Long sentence to split
        max_length: Maximum length per part

    Returns:
        List of sentence parts
    """
    if len(sentence) <= max_length:
        return [sentence]

    # Natural break points in order of preference (English conjunctions and prepositions)
    break_patterns = [
        ' and ',
        ' or ',
        ' but ',
        ' if ',
        ' when ',
        ' where ',
        ' which ',
        ' that ',
        ', ',      # Comma with space
        ' ',       # Space as last resort
    ]

    parts = []
    remaining = sentence

    while len(remaining) > max_length:
        best_split = -1
        best_pattern = None

        # Find the best split point within the max_length
        for pattern in break_patterns:
            # Look for pattern within max_length from the start
            search_text = remaining[:max_length]
            last_occurrence = search_text.rfind(pattern)

            if last_occurrence > len(remaining) * 0.3:  # Don't split too early
                best_split = last_occurrence + len(pattern)
                best_pattern = pattern
                break

        if best_split > 0:
            # Split at the best point found
            part = remaining[:best_split].strip()
            if part:
                parts.append(part)
            remaining = remaining[best_split:].strip()
        else:
            # No good split point found, force split at max_length
            part = remaining[:max_length].strip()
            if part:
                parts.append(part)
            remaining = remaining[max_length:].strip()

    # Add the remaining part
    if remaining.strip():
        parts.append(remaining.strip())

    return parts


def split_caption_segments(segments: List[Dict], max_segment_length: int = 100) -> List[Dict]:
    """
    Split long caption segments into smaller segments with proper timestamp interpolation.

    Args:
        segments: List of caption segments with 'start', 'end', 'text' keys
        max_segment_length: Maximum character length per segment

    Returns:
        List of split segments with interpolated timestamps
    """
    if not segments:
        return []

    new_segments = []

    for i, segment in enumerate(segments):
        text = segment.get('text', '').strip()
        start_time = segment.get('start', 0.0)
        end_time = segment.get('end', start_time + 5.0)

        if not text:
            continue

        # Split the text into sentences
        sentences = split_text_into_sentences(text, max_segment_length)

        if len(sentences) <= 1:
            # No splitting needed
            new_segments.append(segment.copy())
        else:
            # Split into multiple segments with interpolated timestamps
            segment_duration = end_time - start_time

            # Calculate time per character (with minimum duration per segment)
            total_chars = sum(len(s) for s in sentences)
            min_duration_per_segment = 1.0  # Minimum 1 second per segment

            # Distribute time based on character count, but ensure minimum duration
            time_per_char = max(segment_duration / total_chars,
                              min_duration_per_segment / max(len(sentences), 1))

            current_time = start_time

            for j, sentence in enumerate(sentences):
                # Calculate duration for this sentence
                sentence_duration = max(len(sentence) * time_per_char, min_duration_per_segment)

                # Ensure we don't exceed the original end time
                sentence_end = min(current_time + sentence_duration, end_time)

                # If this is the last sentence, make sure it ends at the original end time
                if j == len(sentences) - 1:
                    sentence_end = end_time

                new_segment = {
                    'id': len(new_segments),
                    'start': current_time,
                    'end': sentence_end,
                    'text': sentence.strip(),
                    'words': []  # Word-level timestamps would need more complex interpolation
                }

                # Copy other fields from original segment
                for key, value in segment.items():
                    if key not in ['id', 'start', 'end', 'text', 'words']:
                        new_segment[key] = value

                new_segments.append(new_segment)
                current_time = sentence_end

    return new_segments


def merge_short_segments(segments: List[Dict], min_duration: float = 1.0, min_length: int = 20) -> List[Dict]:
    """
    Merge very short segments with adjacent segments to avoid overly fragmented captions.

    Args:
        segments: List of caption segments
        min_duration: Minimum duration in seconds
        min_length: Minimum text length in characters

    Returns:
        List of merged segments
    """
    if not segments:
        return []

    # Sort segments by start time to ensure proper order
    segments = sorted(segments, key=lambda x: x.get('start', 0))

    merged_segments = []
    i = 0

    while i < len(segments):
        current_segment = segments[i].copy()
        current_duration = current_segment.get('end', 0) - current_segment.get('start', 0)
        current_text = current_segment.get('text', '').strip()

        # Check if current segment is too short
        if (current_duration < min_duration or len(current_text) < min_length) and i + 1 < len(segments):
            # Try to merge with next segment
            next_segment = segments[i + 1]
            next_text = next_segment.get('text', '').strip()

            # Merge texts
            merged_text = f"{current_text} {next_text}".strip()

            # Update segment
            current_segment['text'] = merged_text
            current_segment['end'] = next_segment.get('end', current_segment['end'])
            current_segment['id'] = len(merged_segments)

            # Skip the next segment since we merged it
            i += 2
        else:
            # Keep current segment as is
            current_segment['id'] = len(merged_segments)
            i += 1

        merged_segments.append(current_segment)

    return merged_segments
