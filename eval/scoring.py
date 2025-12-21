import os
import time
from datetime import datetime
from utils import load_jsonl, save_timing_data

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

# Flat ordered list of all tasks
TASK_ORDER = [task for tasks in TASK_CATEGORIES.values() for task in tasks]


def calculate_and_save_scores(args, model_name, model_name_underscored, tasks_to_load,
                            timing_results, overall_start_time, eval_file):
    """
    Calculate scores for evaluation results and save them to file.

    Args:
        args: Command line arguments object
        model_name: Name of the model being evaluated
        model_name_underscored: Underscored version of model name for file paths
        tasks_to_load: List of tasks that were evaluated
        timing_results: Dictionary to store timing information
        overall_start_time: Start time of the overall evaluation
        eval_file: Path to the evaluation results file
    """
    # Scoring phase timing
    scoring_start_time = time.time()

    score_file = os.path.join(args.output_dir, model_name_underscored, f"{model_name_underscored}_score.txt")
    timing_file = os.path.join(args.output_dir, model_name_underscored, f"{model_name_underscored}_timing.json")

    all_task_accuracies = {}

    if not os.path.exists(eval_file):
        print(f"Warning: Evaluation file not found: {eval_file}")
    else:
        eval_results = load_jsonl(eval_file)
        if eval_results:
            # Calculate scores
            task_performance = {}
            for result in eval_results:
                task_type = result.get('task', 'unknown_task')
                if task_type not in task_performance:
                    task_performance[task_type] = {"score_obtained": 0, "score_total": 0}

                obtained_score = 0
                max_score = 0

                for turn in result['conversations']:
                    if turn['role'] == 'assistant':
                        for criteria in turn['criteria']:
                            if criteria['criteria_met'] and not criteria['is_penalty']:
                                obtained_score += criteria['weight']
                            max_score += criteria['weight'] if criteria['weight'] > 0 else 0

                task_performance[task_type]['score_obtained'] += obtained_score
                task_performance[task_type]['score_total'] += max_score

            # Calculate accuracies
            for task_type in task_performance:
                if task_performance[task_type]['score_total'] > 0:
                    accuracy = task_performance[task_type]['score_obtained'] / task_performance[task_type]['score_total']
                    all_task_accuracies[task_type] = accuracy
                else:
                    all_task_accuracies[task_type] = 0.0

    # Append consolidated results
    with open(score_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"  LongShOT Bench Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Model:  {model_name}\n")
        f.write(f"  Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if all_task_accuracies:
            all_accuracies = []
            category_accuracies = {}

            for category, tasks in TASK_CATEGORIES.items():
                f.write("-" * 60 + "\n")
                f.write(f"  {category}\n")
                f.write("-" * 60 + "\n")

                cat_accuracies = []
                for task in tasks:
                    if task in all_task_accuracies:
                        accuracy = all_task_accuracies[task]
                        f.write(f"    {task:<30} {accuracy*100:6.2f}%\n")
                        all_accuracies.append(accuracy)
                        cat_accuracies.append(accuracy)

                # Category average
                if cat_accuracies:
                    cat_avg = sum(cat_accuracies) / len(cat_accuracies)
                    category_accuracies[category] = cat_avg
                    f.write(f"    {'Category Average':<30} {cat_avg*100:6.2f}%\n")
                f.write("\n")

            # Overall summary (average of category averages)
            if category_accuracies:
                overall_accuracy = sum(category_accuracies.values()) / len(category_accuracies)
                f.write("=" * 60 + "\n")
                f.write(f"  OVERALL ACCURACY: {overall_accuracy*100:6.2f}%\n")
                f.write("=" * 60 + "\n")
    
    # Display results
    print("\n" + "=" * 60)
    print("  LongShOT Bench Evaluation Results")
    print("=" * 60 + "\n")

    if all_task_accuracies:
        category_accuracies_display = {}

        for category, tasks in TASK_CATEGORIES.items():
            print("-" * 60)
            print(f"  {category}")
            print("-" * 60)

            cat_accuracies = []
            for task in tasks:
                if task in all_task_accuracies:
                    accuracy = all_task_accuracies[task]
                    print(f"    {task:<30} {accuracy*100:6.2f}%")
                    cat_accuracies.append(accuracy)

            if cat_accuracies:
                cat_avg = sum(cat_accuracies) / len(cat_accuracies)
                category_accuracies_display[category] = cat_avg
                print(f"    {'Category Average':<30} {cat_avg*100:6.2f}%")
            print()

        # Overall (average of category averages)
        if category_accuracies_display:
            overall_accuracy = sum(category_accuracies_display.values()) / len(category_accuracies_display)
            print("=" * 60)
            print(f"  OVERALL ACCURACY: {overall_accuracy*100:6.2f}%")
            print("=" * 60)

    scoring_end_time = time.time()
    scoring_duration = scoring_end_time - scoring_start_time
    timing_results['scoring'] = scoring_duration

    # Calculate overall timing
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    timing_results['total'] = total_duration
    
    # Save scoring and total timing
    save_timing_data(timing_file, 'scoring', scoring_duration)
    save_timing_data(timing_file, 'total', total_duration)

    print(f"\nResults saved to \033[1m{score_file}\033[0m")
    print(f"Timing data saved to \033[1m{timing_file}\033[0m\n")
    
    return timing_results