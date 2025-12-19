import argparse
from tqdm import tqdm
import yaml
import os
import json
import concurrent.futures
import time
from utils import *
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from scoring import calculate_and_save_scores
load_dotenv()
print("Loaded environment variables from .env file")

def str2bool(v):
    """Convert string to boolean value."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    # Start overall timing
    overall_start_time = time.time()
    timing_results = {}

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate models on video benchmark tasks')
    parser.add_argument('--tasks', '-t', nargs='+', required=True, help='List of video benchmark tasks to evaluate on')
    parser.add_argument('--config', '-c', type=str, default='tasks.yaml', help='Path to the video benchmark configuration YAML file')
    parser.add_argument('--config-file', type=str, default='config.yaml', help='Path to the system configuration YAML file')
    parser.add_argument('--output_dir', '-o', type=str, default='results', help='Directory to save evaluation results')
    parser.add_argument('--num_workers', '-n', type=int, default=1, help='Number of workers for parallel processing')
    parser.add_argument('--generate', '-g', type=str2bool, default=False, help='Generate responses for the given tasks')
    parser.add_argument('--evaluate', type=str2bool, default=False, help='Evaluate the generated responses')
    parser.add_argument('--score', type=str2bool, default=False, help='Calculate scores for the evaluation results')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-11B-Vision-Instruct', help='Model name for generation')
    parser.add_argument('--eval_model', type=str, default='Qwen/Qwen3-14B', help='Model name for evaluation')
    parser.add_argument('--tensor_parallel_size', type=str, default=1, help='Tensor parallel size for vllm server')
    parser.add_argument('--external-server', action='store_true', help='Use external vLLM server (do not start/stop server automatically)')
    parser.add_argument('--hallucination-test', type=str, default='none', choices=['none', 'no_video', 'with_notice', 'both'], help='Enable hallucination testing: none (default), no_video (question only), with_notice (question + notice about missing video), both (run both tests)')
    parser.add_argument('--tool', action='store_true', help='Enable tool calling mode for agentic tasks')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_file)

    # Setup phase timing
    setup_start_time = time.time()

    # Load dataset configurations from YAML file
    if os.path.exists(args.config):
        with open(args.config, 'r') as file:
            dataset_configs = yaml.safe_load(file)
            print(f"Loaded dataset configurations from {args.config}")
    else:
        print(f"Configuration file {args.config} not found.")
        return

    # Identify which video benchmark tasks need to be loaded
    all_tasks = []
    task_configs = {}
    for cat_tasks in dataset_configs.values():
        all_tasks.extend(cat_tasks.keys())
        task_configs.update(cat_tasks)

    tasks_to_load = []
    for task in args.tasks:
        if task == "all":
            tasks_to_load = all_tasks
            break
        elif task in all_tasks:
            tasks_to_load.append(task)
        else:
            print(f"Warning: Unknown task '{task}'")

    if not tasks_to_load:
        print("No valid tasks selected. Available options:")
        print("- Individual tasks:", ", ".join(all_tasks))
        print("- Groups: all, vlm")
        return
   
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get model name and create output file path
    model_name = args.model   
    model_name_underscored = convert_to_underscored(model_name)
    print(f"Generating responses using model: {model_name}")

    os.makedirs(os.path.join(args.output_dir, model_name_underscored), exist_ok=True)
    
    def get_file_paths(test_type=""):
        """Get output and eval file paths for a given test type"""
        suffix = f"_hal_{test_type}" if test_type else ""
        output = os.path.join(args.output_dir, model_name_underscored, f"{model_name_underscored}{suffix}.jsonl")
        eval_path = os.path.join(args.output_dir, model_name_underscored, f"{model_name_underscored}{suffix}_eval.jsonl")
        return output, eval_path
    
    # Determine which tests to run - separate normal and hallucination tests
    normal_tests = []
    hallucination_tests = []
    
    # Always include normal tests when generate is enabled or when evaluating/scoring existing results
    if args.generate or args.evaluate or args.score:
        normal_tests = [""]
    
    # Add hallucination tests based on the argument
    if args.hallucination_test == 'no_video':
        hallucination_tests = ["no_video"]
    elif args.hallucination_test == 'with_notice':
        hallucination_tests = ["with_notice"]
    elif args.hallucination_test == 'both':
        hallucination_tests = ["no_video", "with_notice"]
    
    # Combine for compatibility with existing code
    test_types = normal_tests + hallucination_tests
    
    timing_file = os.path.join(args.output_dir, model_name_underscored, f"{model_name_underscored}_timing.json")
    
    print(f"Running {len(test_types)} test(s): {test_types}")
    for test_type in test_types:
        output_file, eval_file = get_file_paths(test_type)
        print(f"Results for {test_type or 'normal'} will be saved to: {output_file}")

    # Print selected tasks
    print(f"Selected tasks: {tasks_to_load}")

    candidate_server = False
    eval_server = False

    setup_end_time = time.time()
    setup_duration = setup_end_time - setup_start_time
    timing_results['setup'] = setup_duration
    
    # Save setup timing
    save_timing_data(timing_file, 'setup', setup_duration)

    if args.generate:
        generation_start_time = time.time()
        
        # Load test samples from all tasks
        test_samples = []
        for task_name in tasks_to_load:
            print(f"Loading task: {task_name}")
            test_samples += load_dataset_with_params(task_configs[task_name], task_name, config)

        test_samples = sorted(test_samples, key=lambda x: x["video_id"])
        print(f"\nModel: {model_name}\nGenerating responses for {len(test_samples)} samples using {args.num_workers} workers...")

        # Start server if we have samples to process
        server_start_time = time.time()
        candidate_server = None
        candidate_log_file = None
        # Skip vLLM server for video-agent models
        if test_samples and not args.external_server and not is_video_agent_model(model_name):
            candidate_server, candidate_log_file = start_vllm(model_name, args.tensor_parallel_size, "candidate", config)
        server_setup_time = time.time() - server_start_time
        
        # Process normal tests first, then hallucination tests
        inference_start_time = time.time()
        
        # First process normal tests
        for test_type in normal_tests:
            output_file, _ = get_file_paths(test_type)
            
            # Load already processed samples for this test type
            completed_ids = set()
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        completed_ids.add(data.get("sample_id"))
            
            # Filter samples for this test type
            current_samples = [s for s in test_samples if s.get("sample_id") not in completed_ids]
            if not current_samples:
                continue
                
            print(f"Processing {len(current_samples)} samples for {test_type or 'normal'} test...")
            
            if args.num_workers <= 1:
                for sample in tqdm(current_samples, desc=f"{test_type or 'normal'} test"):
                    process_sample(sample, output_file, model_name, config, tool_mode=args.tool)
            else:
                from functools import partial
                worker_func = partial(process_sample, output_file=output_file, model_name=model_name, config=config, tool_mode=args.tool)
                with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                    futures = [executor.submit(worker_func, sample) for sample in current_samples]
                    for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{test_type or 'normal'} test"):
                        pass
        
        # Then process hallucination tests after normal tests are complete
        for test_type in hallucination_tests:
            output_file, _ = get_file_paths(test_type)
            
            # Load already processed samples for this test type
            completed_ids = set()
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        completed_ids.add(data.get("sample_id"))
            
            # Filter samples for this test type
            current_samples = [s for s in test_samples if s.get("sample_id") not in completed_ids]
            if not current_samples:
                continue
                
            print(f"Processing {len(current_samples)} samples for {test_type} hallucination test...")
            
            if args.num_workers <= 1:
                for sample in tqdm(current_samples, desc=f"{test_type} hallucination test"):
                    process_sample(sample, output_file, model_name, config, tool_mode=args.tool)
            else:
                from functools import partial
                worker_func = partial(process_sample, output_file=output_file, model_name=model_name, config=config, tool_mode=args.tool)
                with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                    futures = [executor.submit(worker_func, sample) for sample in current_samples]
                    for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{test_type} hallucination test"):
                        pass
                        
        inference_end_time = time.time()

        # Clean up server
        cleanup_start_time = time.time()
        if candidate_server and not args.external_server and not is_video_agent_model(model_name):
            kill_candidate_server(candidate_server, candidate_log_file)
        cleanup_end_time = time.time()

        generation_end_time = time.time()
        generation_duration = generation_end_time - generation_start_time
        timing_results['generation'] = {
            'total': generation_duration,
            'server_setup': server_setup_time,
            'inference': inference_end_time - inference_start_time,
            'cleanup': cleanup_end_time - cleanup_start_time,
            'samples_processed': len(test_samples)
        }
        
        save_timing_data(timing_file, 'generation', timing_results['generation'])

    if args.evaluate:
        evaluation_start_time = time.time()
        eval_model = args.eval_model
        print("Evaluating results...")

        # Start server
        eval_server_start_time = time.time()
        eval_server = None
        eval_log_file = None
        
        # Check if there are any samples to evaluate before starting server
        has_samples_to_evaluate = False
        for test_type in normal_tests + hallucination_tests:
            output_file, eval_file = get_file_paths(test_type)
            if os.path.exists(output_file):
                responses = load_jsonl(output_file)
                evaluated_ids = set()
                if os.path.exists(eval_file):
                    with open(eval_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                evaluated_ids.add(data.get("sample_id"))
                remaining = [sample for sample in responses if sample.get("sample_id") not in evaluated_ids]
                if remaining:
                    has_samples_to_evaluate = True
                    break
        
        if not has_samples_to_evaluate:
            print("All samples already evaluated, skipping evaluation phase.")
            eval_server_setup_time = 0
        elif not args.external_server:
            eval_server, eval_log_file = start_vllm(eval_model, args.tensor_parallel_size, "evaluation", config)
            eval_server_setup_time = time.time() - eval_server_start_time
        else:
            eval_server_setup_time = time.time() - eval_server_start_time

        # Process normal tests first, then hallucination tests
        eval_inference_start_time = time.time()
        total_evaluated = 0
        
        # First process normal test evaluations
        for test_type in normal_tests:
            output_file, eval_file = get_file_paths(test_type)
            
            if not os.path.exists(output_file):
                print(f"Warning: Output file {output_file} not found, skipping evaluation for {test_type or 'normal'}")
                continue
                
            responses = load_jsonl(output_file)
            print(f"Evaluating {test_type or 'normal'} using model: {eval_model}")
            
            # Load already evaluated samples
            evaluated_ids = set()
            if os.path.exists(eval_file):
                with open(eval_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            evaluated_ids.add(data.get("sample_id"))
            
            # Filter samples
            responses = [sample for sample in responses if sample.get("sample_id") not in evaluated_ids]
            if not responses:
                continue
                
            print(f"Evaluating {len(responses)} {test_type or 'normal'} responses...")
            total_evaluated += len(responses)
            
            if args.num_workers <= 1:
                for sample in tqdm(responses, desc=f"Evaluating {test_type or 'normal'}"):
                    process_video_evaluation(sample, eval_file, eval_model, config)
            else:
                from functools import partial
                eval_worker_func = partial(process_video_evaluation, eval_file=eval_file, eval_model=eval_model, config=config)
                with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                    futures = [executor.submit(eval_worker_func, sample) for sample in responses]
                    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Evaluating {test_type or 'normal'}"):
                        f.result()
        
        # Then process hallucination test evaluations after normal tests are complete
        for test_type in hallucination_tests:
            output_file, eval_file = get_file_paths(test_type)
            
            if not os.path.exists(output_file):
                print(f"Warning: Output file {output_file} not found, skipping evaluation for {test_type} hallucination test")
                continue
                
            responses = load_jsonl(output_file)
            print(f"Evaluating {test_type} hallucination test using model: {eval_model}")
            
            # Load already evaluated samples
            evaluated_ids = set()
            if os.path.exists(eval_file):
                with open(eval_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            evaluated_ids.add(data.get("sample_id"))
            
            # Filter samples
            responses = [sample for sample in responses if sample.get("sample_id") not in evaluated_ids]
            if not responses:
                continue
                
            print(f"Evaluating {len(responses)} {test_type} hallucination responses...")
            total_evaluated += len(responses)
            
            if args.num_workers <= 1:
                for sample in tqdm(responses, desc=f"Evaluating {test_type} hallucination"):
                    process_video_evaluation(sample, eval_file, eval_model, config)
            else:
                from functools import partial
                eval_worker_func = partial(process_video_evaluation, eval_file=eval_file, eval_model=eval_model, config=config)
                with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                    futures = [executor.submit(eval_worker_func, sample) for sample in responses]
                    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Evaluating {test_type} hallucination"):
                        f.result()
                        
        eval_inference_end_time = time.time()
        print("\nEvaluation completed...")

        # Clean up server
        eval_cleanup_start_time = time.time()
        if eval_server and not args.external_server:
            kill_candidate_server(eval_server, eval_log_file)
        eval_cleanup_end_time = time.time()

        evaluation_end_time = time.time()
        evaluation_duration = evaluation_end_time - evaluation_start_time
        timing_results['evaluation'] = {
            'total': evaluation_duration,
            'server_setup': eval_server_setup_time,
            'inference': eval_inference_end_time - eval_inference_start_time,
            'cleanup': eval_cleanup_end_time - eval_cleanup_start_time,
            'samples_evaluated': total_evaluated
        }
        
        save_timing_data(timing_file, 'evaluation', timing_results['evaluation'])

    if args.score:
        timing_results = calculate_and_save_scores(
            args, model_name, model_name_underscored, test_types, tasks_to_load, 
            timing_results, overall_start_time, get_file_paths
        )
    
    # Final timing save for runs that don't include scoring
    if 'total' not in timing_results:
        final_end_time = time.time()
        total_duration = final_end_time - overall_start_time
        timing_results['total'] = total_duration
        save_timing_data(timing_file, 'total', total_duration)


if __name__ == "__main__":
    main()