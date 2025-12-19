import os
import time
from datetime import datetime
from utils import load_jsonl, save_timing_data


def calculate_and_save_scores(args, model_name, model_name_underscored, test_types, tasks_to_load, 
                            timing_results, overall_start_time, get_file_paths):
    """
    Calculate scores for evaluation results and save them to file.
    
    Args:
        args: Command line arguments object
        model_name: Name of the model being evaluated
        model_name_underscored: Underscored version of model name for file paths
        test_types: List of test types to process
        tasks_to_load: List of tasks that were evaluated
        timing_results: Dictionary to store timing information
        overall_start_time: Start time of the overall evaluation
        get_file_paths: Function to get output and eval file paths for a test type
    """
    # Scoring phase timing
    scoring_start_time = time.time()
    
    print("\nAggregating video benchmark results...")
    print(tasks_to_load)
    
    score_file = os.path.join(args.output_dir, model_name_underscored, f"{model_name_underscored}_score.txt")
    timing_file = os.path.join(args.output_dir, model_name_underscored, f"{model_name_underscored}_timing.json")
    normal_results = []
    hallucination_results = []
    
    for test_type in test_types:
        _, eval_file = get_file_paths(test_type)
        
        if not os.path.exists(eval_file):
            print(f"Warning: Evaluation file not found: {eval_file}")
            continue
        
        eval_results = load_jsonl(eval_file)
        if not eval_results:
            continue
        
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
        task_accuracies = {}
        for task_type in task_performance:
            if task_performance[task_type]['score_total'] > 0:
                accuracy = task_performance[task_type]['score_obtained'] / task_performance[task_type]['score_total']
                task_accuracies[task_type] = accuracy
            else:
                task_accuracies[task_type] = 0.0
        
        # Store results
        if is_hallucination_test(eval_file):
            hallucination_results.append({
                'test_type': test_type,
                'task_accuracies': task_accuracies
            })
        else:
            normal_results.append({
                'test_type': test_type, 
                'task_accuracies': task_accuracies
            })
    
    # Append consolidated results
    with open(score_file, 'a') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Normal results section
        if normal_results:
            f.write("Video Benchmark Task Results:\n")
            f.write("                    Task  Accuracy Accuracy (%)\n")
            
            # Collect all task accuracies and calculate overall
            all_accuracies = []
            task_index = 0
            for result in normal_results:
                for task, accuracy in result['task_accuracies'].items():
                    f.write(f"{task_index:2d}  {task:>20}  {accuracy:.6f}       {accuracy*100:5.2f}%\n")
                    all_accuracies.append(accuracy)
                    task_index += 1
            
            # Calculate and write overall accuracy
            if all_accuracies:
                overall_accuracy = sum(all_accuracies) / len(all_accuracies)
                f.write(f"{task_index:2d}  {'Overall':>20}  {overall_accuracy:.6f}       {overall_accuracy*100:5.2f}%\n\n")
        
        # Hallucination results section with columns for each test type
        if hallucination_results:
            f.write("Hallucination Test Results:\n")
            f.write("(Higher score = More Hallucination)\n")
            f.write("                    Task  No Video (%)  With Notice (%)\n")
            
            # Organize hallucination data by task
            hal_data = {}
            for result in hallucination_results:
                test_type = result['test_type']
                for task, accuracy in result['task_accuracies'].items():
                    if task not in hal_data:
                        hal_data[task] = {}
                    hal_data[task][test_type] = accuracy
            
            hal_task_index = 0
            for task, scores in hal_data.items():
                no_video_score = scores.get('no_video', 0) * 100
                with_notice_score = scores.get('with_notice', 0) * 100
                
                f.write(f"{hal_task_index:2d}  {task:>20}  {no_video_score:8.2f}%    {with_notice_score:10.2f}%\n")
                hal_task_index += 1
            
            f.write("\n")
    
    # Display results  
    print("\n" + "="*50)
    if normal_results:
        print("Video Benchmark Task Results:")
        print("                    Task  Accuracy Accuracy (%)")
        
        # Collect all task accuracies and calculate overall
        all_accuracies = []
        task_index = 0
        for result in normal_results:
            for task, accuracy in result['task_accuracies'].items():
                print(f"{task_index:2d}  {task:>20}  {accuracy:.6f}       {accuracy*100:5.2f}%")
                all_accuracies.append(accuracy)
                task_index += 1
        
        # Calculate and display overall accuracy
        if all_accuracies:
            overall_accuracy = sum(all_accuracies) / len(all_accuracies)
            print(f"{task_index:2d}  {'Overall':>20}  {overall_accuracy:.6f}       {overall_accuracy*100:5.2f}%")
    
    if hallucination_results:
        print("\nHALLUCINATION TEST RESULTS:")
        print("(Higher score = More Hallucination)")
        print("                    Task  No Video (%)  With Notice (%)")
        
        # Organize hallucination data by task
        hal_data = {}
        for result in hallucination_results:
            test_type = result['test_type']
            for task, accuracy in result['task_accuracies'].items():
                if task not in hal_data:
                    hal_data[task] = {}
                hal_data[task][test_type] = accuracy
        
        hal_task_index = 0
        for task, scores in hal_data.items():
            no_video_score = scores.get('no_video', 0) * 100
            with_notice_score = scores.get('with_notice', 0) * 100
            
            print(f"{hal_task_index:2d}  {task:>20}  {no_video_score:8.2f}%    {with_notice_score:10.2f}%")
            hal_task_index += 1
    print("="*50)

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