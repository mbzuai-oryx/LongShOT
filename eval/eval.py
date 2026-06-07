import argparse
from tqdm import tqdm
import yaml
import os
import json
import re
import concurrent.futures
import signal
import time
from utils import *
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dotenv import load_dotenv
from scoring import calculate_and_save_scores
from agents import get_agent, list_agents, load_agent_config, run_agent_inference
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
    overall_start_time = time.time()

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
    parser.add_argument('--eval_tag', type=str, default=None, help='Filesystem-safe tag for judge-specific eval outputs')
    parser.add_argument('--tensor_parallel_size', type=str, default=1, help='Tensor parallel size for vllm server')
    parser.add_argument('--external-server', action='store_true', help='Use external vLLM server (do not start/stop server automatically)')
    parser.add_argument('--port', type=str, default=None, help='Override vLLM server port (for parallel runs)')
    parser.add_argument('--omni', action='store_true', help='Enable audio extraction from video (Qwen Omni / MiniCPM-o)')
    parser.add_argument('--backend', default='vllm', choices=['vllm', 'hf', 'api'], help='Serving backend (vllm, hf, or api for OpenRouter/cloud APIs)')
    parser.add_argument('--max-frames', type=int, default=0, help='Use pre-extracted frames instead of video (0=use video)')
    parser.add_argument('--audio-only', action='store_true', help='Audio-only mode: send only audio (no video) to the model')
    parser.add_argument('--alias', type=str, default=None, help='Override output directory name (for A/B comparisons with same model)')
    parser.add_argument('--model-path', type=str, default=None, help='Local path to model weights (e.g. hf-mount path). Model name is still used for output naming.')
    args, unknown = parser.parse_known_args()

    # Collect arbitrary --agent-* flags into args.agent_meta dict
    # e.g. --agent-llm Foo --agent-vlm Bar --agent-omni Baz
    #   => {"llm": "Foo", "vlm": "Bar", "omni": "Baz"}
    args.agent_meta = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--agent-'):
            key = unknown[i][len('--agent-'):]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                args.agent_meta[key] = unknown[i + 1]
                i += 2
            else:
                args.agent_meta[key] = True
                i += 1
        else:
            print(f"Warning: unrecognized argument: {unknown[i]}")
            i += 1

    # Load configuration
    config = load_config(args.config_file)

    # Override port if specified (for parallel execution)
    if args.port:
        config['server']['port'] = args.port
        config['server']['base_url'] = f"http://localhost:{args.port}/v1/"

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
    model_name_underscored = args.alias if args.alias else convert_to_underscored(model_name)
    print(f"Generating responses using model: {model_name}"
          + (f" (alias: {args.alias})" if args.alias else ""))

    generation_paths = get_generation_artifact_paths(args.output_dir, model_name_underscored)
    judge_paths = get_judge_artifact_paths(
        args.output_dir, model_name_underscored, args.eval_model, args.eval_tag
    )
    eval_tag = judge_paths["eval_tag"]

    os.makedirs(generation_paths["model_dir"], exist_ok=True)
    if args.evaluate or args.score:
        os.makedirs(judge_paths["judge_dir"], exist_ok=True)

    output_file = generation_paths["output_file"]
    generation_timing_file = generation_paths["timing_file"]
    eval_file = judge_paths["eval_file"]
    judge_timing_file = judge_paths["timing_file"]

    if args.generate:
        generation_start_time = time.time()
        server_setup_time = 0
        inference_start_time = None
        completed_ids = set()
        current_samples = []
        test_samples = []
        candidate_server = None
        candidate_log_file = None
        interrupted = False

        try:
            # Load test samples from all tasks
            for task_name in tasks_to_load:
                print(f"Loading task: {task_name}")
                test_samples += load_dataset_with_params(task_configs[task_name], task_name, config)

            test_samples = sorted(test_samples, key=lambda x: x["video_id"])
            print(f"\nModel: {model_name}\nGenerating responses for {len(test_samples)} samples using {args.num_workers} workers...")

            # Start server if we have samples to process
            server_start_time = time.time()
            # Skip server for video-agent models and API backends (OpenRouter etc.)
            if test_samples and not args.external_server and not is_video_agent_model(model_name) and args.backend != "api":
                if args.backend == "hf":
                    candidate_server, candidate_log_file = start_transformers_server(
                        model_name, args.tensor_parallel_size, config, omni=args.omni,
                        max_frames=args.max_frames)
                else:
                    serve_name = args.model_path if args.model_path else model_name
                    candidate_server, candidate_log_file = start_vllm(
                        serve_name, args.tensor_parallel_size, "candidate", config, omni=args.omni,
                        max_frames=args.max_frames, audio_only=args.audio_only,
                        served_model_name=model_name if args.model_path else None)
            server_setup_time = time.time() - server_start_time

            # Load already processed sample IDs (extract just sample_id without full JSON parse)
            if os.path.exists(output_file):
                sid_pattern = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')
                with open(output_file, 'rb') as f:
                    for line in f:
                        m = sid_pattern.search(line.decode('utf-8', errors='ignore'))
                        if m:
                            completed_ids.add(m.group(1))

            # Filter samples
            current_samples = [s for s in test_samples if s.get("sample_id") not in completed_ids]

            # Process samples
            inference_start_time = time.time()

            if current_samples:
                # Build video path map for fast lookup (avoid os.walk per sample)
                video_base_path = config['paths']['video_path']
                if not os.path.isabs(video_base_path):
                    video_base_path = os.path.abspath(video_base_path)
                video_path_map = build_video_path_map(video_base_path)
                print(f"Built video path map: {len(video_path_map)} videos indexed")

                # Branch: video-agent models use agents framework
                # Only route to agents framework when the extracted name is a
                # registered agent (e.g. video_explorer, videomind).  Custom
                # versioned agents use an external server
                # and go through the standard VLM path instead.
                agent_name = get_agent_name(model_name) if is_video_agent_model(model_name) else None
                use_agent_framework = agent_name is not None and agent_name in list_agents()
                if use_agent_framework:
                    print(f"Using agent framework: {agent_name}")
                    agent_config = load_agent_config(agent_name)
                    agent = get_agent(agent_name, agent_config)

                    try:
                        agent.start_servers()
                        print(f"Agent servers started: {list(agent.get_server_endpoints().keys())}")

                        run_agent_inference(
                            agent=agent,
                            samples=current_samples,
                            video_path_map=video_path_map,
                            output_file=output_file,
                            extract_question=extract_question_for_agent,
                            inject_response=inject_agent_response,
                            concurrency_override=args.num_workers,
                            agent_config=agent_config,
                        )
                    finally:
                        agent.stop_servers()
                        print("Agent servers stopped")

                # Standard VLM path
                else:
                    # Create shared client: OpenRouter for API backend, local OpenAI for vllm/hf
                    if args.backend == "api":
                        from openrouter_client import create_openrouter_client
                        client = create_openrouter_client(config)
                    else:
                        client = create_openai_client(config)

                    if args.num_workers <= 1:
                        for sample in tqdm(current_samples, desc="Generating"):
                            process_sample(sample, output_file, model_name, config,
                                           client=client, video_path_map=video_path_map, omni=args.omni,
                                           max_frames=args.max_frames, audio_only=args.audio_only,
                                           backend=args.backend)
                    else:
                        # Flatten all samples into individual turn-level tasks so
                        # exactly num_workers API calls are in flight at all times,
                        # keeping vLLM continuously saturated.
                        import threading

                        all_turn_tasks = []   # (sample_idx, conv_idx, messages, extra_body)
                        no_turn_count = 0

                        for si, sample in enumerate(current_samples):
                            try:
                                tasks = build_turn_tasks(sample, model_name, config, video_path_map, omni=args.omni, max_frames=args.max_frames, audio_only=args.audio_only)
                            except Exception as e:
                                print(f"\n[WARNING] Build failed for {sample.get('sample_id')}: {e}")
                                continue
                            if not tasks:
                                # No assistant turns — write as-is
                                with FileLock(f"{output_file}.lock"):
                                    with open(output_file, 'a') as f:
                                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                                no_turn_count += 1
                                continue
                            for conv_idx, msgs, extra_body in tasks:
                                all_turn_tasks.append((si, conv_idx, msgs, extra_body))

                        print(f"Dispatching {len(all_turn_tasks)} turn-level tasks "
                              f"across {args.num_workers} workers")

                        # Per-sample turn completion tracking
                        turns_remaining = {}
                        for si, _, _, _ in all_turn_tasks:
                            turns_remaining[si] = turns_remaining.get(si, 0) + 1

                        completed_lock = threading.Lock()
                        failed_turns = []
                        pbar = tqdm(total=len(current_samples), initial=no_turn_count,
                                    desc="Generating")

                        def _has_empty_responses(sample):
                            for t in sample.get('conversations', []):
                                if t.get('role') == 'assistant' and not t.get('candidate_response'):
                                    return True
                            return False

                        def _on_turn_done(si):
                            """Write sample to disk once all its turns complete."""
                            do_write = False
                            with completed_lock:
                                turns_remaining[si] -= 1
                                if turns_remaining[si] == 0:
                                    do_write = True
                            if do_write:
                                if _has_empty_responses(current_samples[si]):
                                    print(f"\n[SKIP] {current_samples[si].get('sample_id')}: empty response, will retry next run")
                                else:
                                    with FileLock(f"{output_file}.lock"):
                                        with open(output_file, 'a') as f:
                                            f.write(json.dumps(current_samples[si], ensure_ascii=False) + '\n')
                                pbar.update(1)

                        # Build video prefetch pipeline: for each sample, identify
                        # the next 1-2 distinct videos that will be needed.
                        # Preloading happens via the server's /v1/preload endpoint.
                        _video_boundaries = {}  # si -> list of next video paths to prefetch
                        if args.backend == "hf" and video_path_map:
                            seen_vids = []
                            for s in current_samples:
                                vid = s.get("video_id", "")
                                if not seen_vids or seen_vids[-1] != vid:
                                    seen_vids.append(vid)
                            vid_to_next = {}
                            for i, vid in enumerate(seen_vids):
                                upcoming = []
                                for j in range(i + 1, min(i + 3, len(seen_vids))):
                                    p = video_path_map.get(seen_vids[j])
                                    if p:
                                        upcoming.append(p)
                                vid_to_next[vid] = upcoming
                            # Map sample index to prefetch paths (only at video boundaries)
                            prev_vid = None
                            for si, sample in enumerate(current_samples):
                                vid = sample.get("video_id", "")
                                if vid != prev_vid:
                                    paths = vid_to_next.get(vid, [])
                                    if paths:
                                        _video_boundaries[si] = paths
                                    prev_vid = vid
                            if _video_boundaries:
                                print(f"Video prefetch: {len(_video_boundaries)} boundary points, "
                                      f"prefetching n+1/n+2 videos ahead")

                        _prefetch_submitted = set()

                        def _maybe_prefetch(si):
                            """Submit /v1/preload for upcoming videos at video boundaries."""
                            paths = _video_boundaries.get(si)
                            if not paths:
                                return
                            # Only prefetch each path once
                            to_preload = [p for p in paths if p not in _prefetch_submitted]
                            if not to_preload:
                                return
                            for p in to_preload:
                                _prefetch_submitted.add(p)
                            try:
                                import requests as _req
                                base_url = config['server']['base_url'].rstrip('/')
                                _req.post(f"{base_url}/preload",
                                          json={"paths": to_preload, "media_type": "video"},
                                          timeout=2)
                            except Exception:
                                pass  # best-effort

                        if args.backend == "api":
                            from openrouter_client import execute_turn_openrouter as _execute_turn_fn
                        else:
                            _execute_turn_fn = execute_turn

                        def _process_turn(task):
                            si, conv_idx, messages, extra_body = task
                            _maybe_prefetch(si)
                            resp = _execute_turn_fn(client, model_name, messages, extra_body, config)
                            current_samples[si]['conversations'][conv_idx]['candidate_response'] = resp
                            _on_turn_done(si)

                        # Worker-based dispatch: submit all turn tasks at once and
                        # let the thread pool drain them in arrival order.
                        executor = ThreadPoolExecutor(max_workers=args.num_workers)

                        turn_futures = {executor.submit(_process_turn, t): t
                                        for t in all_turn_tasks}

                        try:
                            for future in concurrent.futures.as_completed(turn_futures):
                                try:
                                    future.result(timeout=300)
                                except Exception as e:
                                    task = turn_futures[future]
                                    failed_turns.append(task)
                                    print(f"\n[WARNING] Turn failed for "
                                          f"{current_samples[task[0]].get('sample_id')}: {e}")
                        except KeyboardInterrupt:
                            print("\n[INTERRUPTED] Cancelling pending generation tasks...")
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise
                        finally:
                            executor.shutdown(wait=False)

                        # Retry failed turns in parallel
                        if failed_turns:
                            print(f"\n[WARNING] {len(failed_turns)} turns failed, retrying...")
                            retry_executor = ThreadPoolExecutor(max_workers=args.num_workers)
                            retry_futures = {retry_executor.submit(_process_turn, t): t
                                             for t in failed_turns}
                            try:
                                for future in tqdm(concurrent.futures.as_completed(retry_futures),
                                                   total=len(retry_futures), desc="Retrying"):
                                    try:
                                        future.result(timeout=300)
                                    except Exception as e:
                                        task = retry_futures[future]
                                        print(f"[ERROR] Retry failed for {current_samples[task[0]].get('sample_id')}: {e}")
                            except KeyboardInterrupt:
                                for f in retry_futures:
                                    f.cancel()
                                retry_executor.shutdown(wait=False, cancel_futures=True)
                                raise
                            finally:
                                retry_executor.shutdown(wait=False)

                        pbar.close()

        except KeyboardInterrupt:
            # Block further SIGINTs so the finally block runs to completion
            # (the scheduler may send a redundant SIGINT during its cleanup)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            print("\n[INTERRUPTED] Saving partial timing before exit...")
            interrupted = True
        finally:
            # Clean up server
            if candidate_server and not args.external_server and not is_video_agent_model(model_name):
                kill_candidate_server(candidate_server, candidate_log_file)

            # Count actual samples written to disk (not just what we attempted)
            actual_completed = 0
            if os.path.exists(output_file):
                with open(output_file, 'rb') as f:
                    actual_completed = sum(1 for _ in f)
            samples_this_run = actual_completed - len(completed_ids)

            # Save timing — works for both normal completion and interrupts
            timing_data = {
                'total': time.time() - generation_start_time,
                'server_setup': server_setup_time,
                'inference': (time.time() - inference_start_time) if inference_start_time else 0,
                'samples_processed': samples_this_run,
                'samples_skipped': len(completed_ids),
                'samples_total': len(test_samples),
                'interrupted': interrupted,
            }
            if args.agent_meta:
                timing_data['agent_config'] = args.agent_meta
            save_timing_data(generation_timing_file, 'generation', timing_data)

            if interrupted:
                print(f"[INTERRUPTED] Timing saved. {samples_this_run} samples completed this run.")
                return

    # Restore default SIGINT handler (generation may have set SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    if args.evaluate:
        evaluation_start_time = time.time()
        eval_model = args.eval_model
        eval_server_setup_time = 0
        eval_inference_start_time = None
        eval_server = None
        eval_log_file = None
        total_evaluated = 0
        interrupted = False
        print(f"Evaluating results with judge: {eval_model} (tag: {eval_tag})")

        try:
            # Load responses and filter already-evaluated samples (single load)
            responses = []
            if os.path.exists(output_file):
                responses = load_jsonl(output_file)
                evaluated_ids = set()
                if os.path.exists(eval_file):
                    sid_pattern = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')
                    with open(eval_file, 'rb') as f:
                        for line in f:
                            m = sid_pattern.search(line.decode('utf-8', errors='ignore'))
                            if m:
                                evaluated_ids.add(m.group(1))
                responses = [sample for sample in responses if sample.get("sample_id") not in evaluated_ids]

            if not responses:
                print("All samples already evaluated, skipping evaluation phase.")
            elif not args.external_server:
                eval_server_start_time = time.time()
                eval_server, eval_log_file = start_vllm(eval_model, args.tensor_parallel_size, "evaluation", config)
                eval_server_setup_time = time.time() - eval_server_start_time

            total_evaluated = len(responses)
            eval_inference_start_time = time.time()

            if responses:
                # Create shared OpenAI client for connection reuse across eval workers
                eval_client = create_openai_client(config)

                if args.num_workers <= 1:
                    for sample in tqdm(responses, desc="Evaluating"):
                        process_video_evaluation(sample, eval_file, eval_model, config, client=eval_client)
                else:
                    # Flatten all criterion-level tasks across samples into one
                    # thread pool — avoids nested ThreadPoolExecutors (previously
                    # 256 outer * 8 inner = 2048 threads contending on vLLM).
                    import threading
                    from utils import _evaluate_single_criterion

                    all_crit_tasks = []   # (sample_idx, criterion, ground_truth, model_response)
                    no_crit_count = 0

                    for si, sample in enumerate(responses):
                        if 'conversations' not in sample:
                            with FileLock(f"{eval_file}.lock"):
                                with open(eval_file, 'a') as f:
                                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                            no_crit_count += 1
                            continue
                        has_criteria = False
                        for turn in sample['conversations']:
                            if turn['role'] != 'assistant':
                                continue
                            gt = turn['content']
                            mr = turn.get('candidate_response', '')
                            for criterion in turn.get('criteria', []):
                                all_crit_tasks.append((si, criterion, gt, mr))
                                has_criteria = True
                        if not has_criteria:
                            with FileLock(f"{eval_file}.lock"):
                                with open(eval_file, 'a') as f:
                                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                            no_crit_count += 1

                    print(f"Dispatching {len(all_crit_tasks)} criterion-level tasks "
                          f"across {args.num_workers} workers")

                    # Per-sample criterion completion tracking
                    crits_remaining = {}
                    for si, _, _, _ in all_crit_tasks:
                        crits_remaining[si] = crits_remaining.get(si, 0) + 1

                    completed_lock = threading.Lock()
                    failed_crits = []
                    pbar = tqdm(total=len(responses), initial=no_crit_count,
                                desc="Evaluating")

                    def _on_crit_done(si):
                        """Write sample to disk once all its criteria are evaluated."""
                        do_write = False
                        with completed_lock:
                            crits_remaining[si] -= 1
                            if crits_remaining[si] == 0:
                                do_write = True
                        if do_write:
                            with FileLock(f"{eval_file}.lock"):
                                with open(eval_file, 'a') as f:
                                    f.write(json.dumps(responses[si], ensure_ascii=False) + '\n')
                            pbar.update(1)

                    def _process_criterion(task):
                        si, criterion, gt, mr = task
                        _evaluate_single_criterion(
                            criterion, gt, mr, eval_model, config,
                            responses[si].get('sample_id'), eval_client)
                        _on_crit_done(si)

                    executor = ThreadPoolExecutor(max_workers=args.num_workers)
                    crit_futures = {executor.submit(_process_criterion, t): t
                                    for t in all_crit_tasks}

                    try:
                        for future in concurrent.futures.as_completed(crit_futures):
                            try:
                                future.result(timeout=300)
                            except Exception as e:
                                task = crit_futures[future]
                                print(f"\n[WARNING] Criterion failed for {responses[task[0]].get('sample_id')}: {e}")
                                failed_crits.append(task)
                    except KeyboardInterrupt:
                        print("\n[INTERRUPTED] Cancelling pending eval tasks...")
                        for f in crit_futures:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    finally:
                        executor.shutdown(wait=False)

                    # Retry failed criteria
                    if failed_crits:
                        print(f"\n[WARNING] {len(failed_crits)} criterion evals failed, retrying...")
                        retry_executor = ThreadPoolExecutor(max_workers=args.num_workers)
                        retry_futures = {retry_executor.submit(_process_criterion, t): t
                                         for t in failed_crits}
                        try:
                            for future in tqdm(concurrent.futures.as_completed(retry_futures),
                                               total=len(retry_futures), desc="Retrying eval"):
                                try:
                                    future.result(timeout=300)
                                except Exception as e:
                                    task = retry_futures[future]
                                    print(f"[ERROR] Retry failed for {responses[task[0]].get('sample_id')}: {e}")
                        except KeyboardInterrupt:
                            for f in retry_futures:
                                f.cancel()
                            retry_executor.shutdown(wait=False, cancel_futures=True)
                            raise
                        finally:
                            retry_executor.shutdown(wait=False)

                    pbar.close()

        except KeyboardInterrupt:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            print("\n[INTERRUPTED] Saving partial evaluation timing before exit...")
            interrupted = True
        finally:
            # Clean up server
            if eval_server and not args.external_server:
                kill_candidate_server(eval_server, eval_log_file)

            save_timing_data(judge_timing_file, 'evaluation', {
                'total': time.time() - evaluation_start_time,
                'server_setup': eval_server_setup_time,
                'inference': (time.time() - eval_inference_start_time) if eval_inference_start_time else 0,
                'samples_evaluated': total_evaluated,
                'interrupted': interrupted,
                'judge_model': eval_model,
                'judge_tag': eval_tag,
            })

            if interrupted:
                return

    if args.score:
        calculate_and_save_scores(
            args, model_name, model_name_underscored, tasks_to_load,
            overall_start_time,
            eval_file=eval_file,
            generation_timing_file=generation_timing_file,
            judge_timing_file=judge_timing_file,
            eval_model=args.eval_model,
            eval_tag=eval_tag,
        )


if __name__ == "__main__":
    main()
