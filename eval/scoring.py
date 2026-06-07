import os
import json
import time
from datetime import datetime
from utils import get_judge_artifact_paths, load_jsonl, save_timing_data

# Task categories in display order
TASK_CATEGORIES = {
    "Core Perception Tasks": [
        "entity_recognition",
        "event_understanding",
        "temporal_reasoning",
        "audio_understanding"
    ],
    "Reasoning Tasks": [
        "causal_reasoning",
        "quantitative_reasoning",
        "compositional_reasoning",
        "comparative_analysis"
    ],
    "Information Tasks": [
        "information_retrieval",
        "summarization",
        "instruction_extraction",
        "sentiment_analysis"
    ],
    "Multimodal Tasks": [
        "multimodal_synthesis",
        "cross_modal_verification",
        "audio_visual_alignment",
        "motion_analysis"
    ]
}

# Task remapping: merge source tasks into canonical task names
TASK_REMAP = {
    "ocr_text_extraction": "entity_recognition",
    "visual_scene_understanding": "event_understanding",
    "spatial_reasoning": "temporal_reasoning",
    "speech_transcription": "audio_understanding",
    "speaker_identification": "audio_understanding",
    "commonsense_reasoning": "causal_reasoning",
    "mathematical_reasoning": "quantitative_reasoning",
    "pattern_recognition": "quantitative_reasoning",
    "reasoning_tasks": "quantitative_reasoning",
    "counterfactual_reasoning": "compositional_reasoning",
    "question_answering": "summarization",
    "ethical_reasoning": "sentiment_analysis",
    "multimodal_translation": "motion_analysis",
}

# Modality grouping: map raw modality tags to display groups
MODALITY_GROUPS = {
    "visual": "visual",
    "audio_environment": "audio",
    "speech": "speech",
    "text": "visual",
    "text_extraction": "visual",
    "text_overlay": "visual",
    "cross_modal_information": "visual",
    "timestamp_identification": "visual",
    "implicit_information": "visual",
    "contextual_interpretation": "visual",
}

MODALITY_DISPLAY_NAMES = {
    "visual": "Visual",
    "audio": "Audio",
    "speech": "Speech",
}

# Duration buckets (label, min_seconds, max_seconds)
DURATION_BUCKETS = [
    ("short", 0, 1800),       # <30 min
    ("medium", 1800, 2700),   # 30-45 min
    ("long", 2700, float('inf')),  # >45 min
]

DURATION_DISPLAY_NAMES = {
    "short": "Short (<30m)",
    "medium": "Medium (30-45m)",
    "long": "Long (>45m)",
}

# Human-readable task names
TASK_DISPLAY_NAMES = {
    "entity_recognition": "Entity Recognition",
    "event_understanding": "Event Understanding",
    "temporal_reasoning": "Temporal Reasoning",
    "audio_understanding": "Audio Understanding",
    "causal_reasoning": "Causal Reasoning",
    "quantitative_reasoning": "Quantitative Reasoning",
    "compositional_reasoning": "Compositional Reasoning",
    "comparative_analysis": "Comparative Analysis",
    "information_retrieval": "Information Retrieval",
    "summarization": "Summarization",
    "instruction_extraction": "Instruction Extraction",
    "sentiment_analysis": "Sentiment Analysis",
    "multimodal_synthesis": "Multimodal Synthesis",
    "cross_modal_verification": "Cross Modal Verification",
    "audio_visual_alignment": "Audio Visual Alignment",
    "motion_analysis": "Motion Analysis",
}


def _load_stage_runs(timing_file, stage):
    """Load run timestamps for a single stage from a timing file."""
    timestamps = []
    if os.path.exists(timing_file):
        try:
            with open(timing_file, 'r') as f:
                timing = json.load(f)
            stage_data = timing.get(stage, {})
            sample_field = 'samples_processed' if stage == 'generation' else 'samples_evaluated'
            for run in stage_data.get('runs', []):
                ts = run.get('run_timestamp', '')
                samples = run.get(sample_field, 0)
                interrupted = run.get('interrupted', False)
                status = "interrupted" if interrupted else "completed"
                timestamps.append(f"  {stage:<12} {ts}  ({samples} samples, {status})")
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return timestamps


def _load_run_timestamps(generation_timing_file, judge_timing_file=None):
    """Load generation runs from the model root and evaluation runs from the judge subdir."""
    timestamps = []
    timestamps.extend(_load_stage_runs(generation_timing_file, 'generation'))
    eval_timing_file = judge_timing_file if judge_timing_file and os.path.exists(judge_timing_file) else generation_timing_file
    timestamps.extend(_load_stage_runs(eval_timing_file, 'evaluation'))
    return timestamps


def _compute_scores(eval_results):
    """Compute per-task accuracy from evaluation results."""
    task_performance = {}
    for result in eval_results:
        task_type = TASK_REMAP.get(result.get('task', ''), result.get('task', 'unknown'))
        if task_type not in task_performance:
            task_performance[task_type] = {"obtained": 0, "total": 0, "count": 0}

        obtained = 0
        max_score = 0
        for turn in result['conversations']:
            if turn['role'] == 'assistant':
                for criteria in turn['criteria']:
                    if criteria['criteria_met'] and not criteria['is_penalty']:
                        obtained += criteria['weight']
                    if criteria['weight'] > 0:
                        max_score += criteria['weight']

        task_performance[task_type]['obtained'] += obtained
        task_performance[task_type]['total'] += max_score
        task_performance[task_type]['count'] += 1

    accuracies = {}
    counts = {}
    for task, perf in task_performance.items():
        counts[task] = perf['count']
        accuracies[task] = perf['obtained'] / perf['total'] if perf['total'] > 0 else 0.0

    return accuracies, counts


def _compute_breakdown_scores(eval_results):
    """Compute accuracy breakdowns by modality and video duration."""

    def _sample_score(result):
        """Return (obtained, max_score) for a single sample."""
        obtained = 0
        max_score = 0
        for turn in result['conversations']:
            if turn['role'] == 'assistant':
                for criteria in turn['criteria']:
                    if criteria['criteria_met'] and not criteria['is_penalty']:
                        obtained += criteria['weight']
                    if criteria['weight'] > 0:
                        max_score += criteria['weight']
        return obtained, max_score

    def _sample_modalities(result):
        """Return the set of grouped modality keys for a sample."""
        raw_mods = set()
        for turn in result['conversations']:
            if turn['role'] == 'user':
                for m in turn.get('modalities', []):
                    raw_mods.add(m)
        grouped = set()
        for m in raw_mods:
            g = MODALITY_GROUPS.get(m, "visual")
            grouped.add(g)
        return grouped

    def _duration_bucket(duration):
        """Return the duration bucket label for a given duration in seconds."""
        for label, lo, hi in DURATION_BUCKETS:
            if lo <= duration < hi:
                return label
        return DURATION_BUCKETS[-1][0]

    # Accumulators: {key: {"obtained": int, "total": int, "count": int}}
    modality_perf = {}
    duration_perf = {}

    for result in eval_results:
        obtained, max_score = _sample_score(result)

        # Modality breakdown
        for mod in _sample_modalities(result):
            if mod not in modality_perf:
                modality_perf[mod] = {"obtained": 0, "total": 0, "count": 0}
            modality_perf[mod]["obtained"] += obtained
            modality_perf[mod]["total"] += max_score
            modality_perf[mod]["count"] += 1

        # Duration breakdown
        bucket = _duration_bucket(result.get('duration', 0))
        if bucket not in duration_perf:
            duration_perf[bucket] = {"obtained": 0, "total": 0, "count": 0}
        duration_perf[bucket]["obtained"] += obtained
        duration_perf[bucket]["total"] += max_score
        duration_perf[bucket]["count"] += 1

    # Convert to accuracies
    modality_scores = {}
    for mod, perf in modality_perf.items():
        modality_scores[mod] = {
            "accuracy": perf["obtained"] / perf["total"] if perf["total"] > 0 else 0.0,
            "samples": perf["count"],
        }

    duration_scores = {}
    for bucket, perf in duration_perf.items():
        duration_scores[bucket] = {
            "accuracy": perf["obtained"] / perf["total"] if perf["total"] > 0 else 0.0,
            "samples": perf["count"],
        }

    return modality_scores, duration_scores


def _format_report(model_name, accuracies, counts, run_timestamps,
                   modality_scores=None, duration_scores=None,
                   judge_model=None, judge_tag=None):
    """Build the full score report string (written to file and printed)."""
    W = 64
    lines = []

    # Header
    lines.append("=" * W)
    lines.append("  LongShOT Bench Evaluation Results")
    lines.append("=" * W)
    lines.append("")
    lines.append(f"  Model:  {model_name}")
    if judge_model:
        lines.append(f"  Judge:  {judge_model}")
    if judge_tag:
        lines.append(f"  Tag:    {judge_tag}")
    lines.append(f"  Scored: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if run_timestamps:
        lines.append("")
        lines.append("  Run History:")
        lines.extend(run_timestamps)
    lines.append("")

    if not accuracies:
        lines.append("  No evaluation results found.")
        return "\n".join(lines)

    # --- Table 1: Detailed subtask results ---
    lines.append("=" * W)
    lines.append("  TABLE 1: Subtask Results")
    lines.append("=" * W)
    lines.append(f"  {'Task':<32} {'N':>5}  {'Accuracy':>8}")
    lines.append("-" * W)

    category_avgs = {}
    total_samples = 0

    for category, tasks in TASK_CATEGORIES.items():
        lines.append(f"  {category}")
        lines.append("-" * W)

        cat_accs = []
        cat_count = 0
        for task in tasks:
            if task in accuracies:
                name = TASK_DISPLAY_NAMES.get(task, task)
                count = counts.get(task, 0)
                acc = accuracies[task]
                lines.append(f"    {name:<30} {count:>5}  {acc*100:7.2f}%")
                cat_accs.append(acc)
                cat_count += count

        if cat_accs:
            cat_avg = sum(cat_accs) / len(cat_accs)
            category_avgs[category] = cat_avg
            total_samples += cat_count
            lines.append(f"    {'Avg.':<30} {cat_count:>5}  {cat_avg*100:7.2f}%")
        lines.append("")

    if category_avgs:
        overall = sum(category_avgs.values()) / len(category_avgs)
        lines.append("=" * W)
        lines.append(f"  {'OVERALL':<34} {total_samples:>5}  {overall*100:7.2f}%")
        lines.append("=" * W)

    # --- Table 2: Category summary ---
    lines.append("")
    lines.append("=" * W)
    lines.append("  TABLE 2: Category Summary")
    lines.append("=" * W)
    lines.append(f"  {'Category':<34} {'N':>5}  {'Accuracy':>8}")
    lines.append("-" * W)

    for category, tasks in TASK_CATEGORIES.items():
        if category in category_avgs:
            cat_count = sum(counts.get(t, 0) for t in tasks)
            lines.append(f"  {category:<34} {cat_count:>5}  {category_avgs[category]*100:7.2f}%")

    if category_avgs:
        overall = sum(category_avgs.values()) / len(category_avgs)
        lines.append("-" * W)
        lines.append(f"  {'OVERALL':<34} {total_samples:>5}  {overall*100:7.2f}%")
        lines.append("=" * W)

    # --- Table 3: Modality breakdown ---
    if modality_scores:
        lines.append("")
        lines.append("=" * W)
        lines.append("  TABLE 3: Modality Breakdown")
        lines.append("=" * W)
        lines.append(f"  {'Modality':<34} {'N':>5}  {'Accuracy':>8}")
        lines.append("-" * W)

        display_order = ["visual", "audio", "speech"]
        for mod in display_order:
            if mod in modality_scores:
                name = MODALITY_DISPLAY_NAMES.get(mod, mod)
                s = modality_scores[mod]
                lines.append(f"  {name:<34} {s['samples']:>5}  {s['accuracy']*100:7.2f}%")
        lines.append("=" * W)
        lines.append("  Note: Samples may appear in multiple modalities.")

    # --- Table 4: Duration breakdown ---
    if duration_scores:
        lines.append("")
        lines.append("=" * W)
        lines.append("  TABLE 4: Video Duration Breakdown")
        lines.append("=" * W)
        lines.append(f"  {'Duration':<34} {'N':>5}  {'Accuracy':>8}")
        lines.append("-" * W)

        for label, _, _ in DURATION_BUCKETS:
            if label in duration_scores:
                name = DURATION_DISPLAY_NAMES.get(label, label)
                s = duration_scores[label]
                lines.append(f"  {name:<34} {s['samples']:>5}  {s['accuracy']*100:7.2f}%")
        lines.append("=" * W)

    return "\n".join(lines)


def _build_score_json(model_name, accuracies, counts,
                      modality_scores=None, duration_scores=None,
                      judge_model=None, judge_tag=None):
    """Build structured JSON score data."""
    category_avgs = {}
    total_samples = 0
    subtasks = {}

    for category, tasks in TASK_CATEGORIES.items():
        cat_accs = []
        cat_count = 0
        for task in tasks:
            if task in accuracies:
                count = counts.get(task, 0)
                acc = round(accuracies[task] * 100, 2)
                subtasks[task] = {"accuracy": acc, "samples": count, "category": category}
                cat_accs.append(accuracies[task])
                cat_count += count

        if cat_accs:
            cat_avg = sum(cat_accs) / len(cat_accs)
            category_avgs[category] = {"accuracy": round(cat_avg * 100, 2), "samples": cat_count}
            total_samples += cat_count

    overall = round(sum(c["accuracy"] for c in category_avgs.values()) / len(category_avgs), 2) if category_avgs else 0

    result = {
        "model": model_name,
        "judge_model": judge_model,
        "judge_tag": judge_tag,
        "scored_at": datetime.now().isoformat(),
        "overall": {"accuracy": overall, "samples": total_samples},
        "categories": category_avgs,
        "subtasks": subtasks,
    }

    if modality_scores:
        result["modality_breakdown"] = {
            mod: {"accuracy": round(s["accuracy"] * 100, 2), "samples": s["samples"]}
            for mod, s in modality_scores.items()
        }

    if duration_scores:
        result["duration_breakdown"] = {
            label: {
                "accuracy": round(s["accuracy"] * 100, 2),
                "samples": s["samples"],
                "range": DURATION_DISPLAY_NAMES.get(label, label),
            }
            for label, s in duration_scores.items()
        }

    return result


def calculate_and_save_scores(args, model_name, model_name_underscored, tasks_to_load,
                            overall_start_time, eval_file, generation_timing_file,
                            judge_timing_file, eval_model, eval_tag):
    """Calculate scores for evaluation results and save them to file."""
    scoring_start_time = time.time()

    judge_paths = get_judge_artifact_paths(
        args.output_dir, model_name_underscored, eval_model, eval_tag
    )
    os.makedirs(judge_paths["judge_dir"], exist_ok=True)
    score_file = judge_paths["score_file"]
    score_json_file = judge_paths["score_json_file"]

    # Load and score
    accuracies, counts = {}, {}
    modality_scores, duration_scores = None, None
    if os.path.exists(eval_file):
        eval_results = load_jsonl(eval_file)
        if eval_results:
            accuracies, counts = _compute_scores(eval_results)
            modality_scores, duration_scores = _compute_breakdown_scores(eval_results)
    else:
        print(f"Warning: Evaluation file not found: {eval_file}")

    # Load run history from timing file
    run_timestamps = _load_run_timestamps(generation_timing_file, judge_timing_file)

    # Write text report
    report = _format_report(model_name, accuracies, counts, run_timestamps,
                            modality_scores, duration_scores,
                            judge_model=eval_model, judge_tag=eval_tag)
    with open(score_file, 'w') as f:
        f.write(report + "\n")

    # Write JSON report
    score_data = _build_score_json(model_name, accuracies, counts,
                                   modality_scores, duration_scores,
                                   judge_model=eval_model, judge_tag=eval_tag)
    with open(score_json_file, 'w') as f:
        json.dump(score_data, f, indent=2)

    # Print to console
    print("\n" + report)

    # Save timing
    scoring_duration = time.time() - scoring_start_time
    total_duration = time.time() - overall_start_time
    save_timing_data(judge_timing_file, 'scoring', scoring_duration)
    save_timing_data(judge_timing_file, 'total', total_duration)

    print(f"\nResults saved to \033[1m{score_file}\033[0m")
    print(f"Timing data saved to \033[1m{judge_timing_file}\033[0m\n")
