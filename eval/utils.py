import json
from datasets import load_dataset
from tqdm import tqdm
import requests
import re
from openai import OpenAI
import base64
import io
from filelock import FileLock
import subprocess
import time
import psutil
import os
from PIL import Image
from datetime import datetime
import yaml

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def create_logs_directory(config=None):
    """Create logs directory if it doesn't exist"""
    if config is None:
        config = load_config()
    logs_dir = config['paths']['logs_dir']
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    return logs_dir

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def load_dataset_with_params(params, task_name, config=None):
    """Load a video benchmark task dataset using stored parameters"""
    if config is None:
        config = load_config()
    
    path = params["path"]
    name = params.get("name", "default")
    split = params.get("split", "test")
    
    # Load dataset with or without name parameter
    if name == "default":
        ds = load_dataset(path)[split]
    else:
        ds = load_dataset(path, name=name)[split]
    
    # Convert to list with progress bar
    dataset_list = list(tqdm(ds))
    
    # Apply dataset limit from config (0 or None means no limit)
    dataset_limit = config['system'].get('dataset_limit', 0)
    if dataset_limit and dataset_limit > 0:
        return dataset_list[:dataset_limit]
    
    # Load only samples with video_id: eno3UMEMQJI and b0HfmY64eSE for testing
    # test_video_ids = {"eno3UMEMQJI", "b0HfmY64eSE"}
    # dataset_list = [sample for sample in dataset_list if sample.get("video_id") in test_video_ids]
    
    return dataset_list

def get_model_name(config=None):
    """Get the model name from the API"""
    if config is None:
        config = load_config()
    
    url = f"{config['server']['base_url']}models"
    response = requests.get(url)
    response.raise_for_status()
    models_data = response.json()
    
    # Return first model ID if available, otherwise default
    if models_data.get("data"):
        return models_data["data"][0]["id"]
    return "unknown_model"

def convert_to_underscored(model_name):
    """Convert model name to underscored format by replacing non-alphanumeric chars with underscore"""
    return re.sub(r'[^a-zA-Z0-9]', '_', (lambda split: split[-2] if split[-1].strip() == "" else split[-1])(model_name.split('/')))

def is_video_agent_model(model_name):
    """Check if the model is a video-agent model"""
    return "video-agent" in model_name.lower()

def generate_candidate_response(sample, model_name, config=None):
    """Generate model response for a video benchmark sample"""
    if config is None:
        config = load_config()

    # Get video identifier
    video_id = sample.get("video_id")

    # Check if this is a video-agent model
    is_video_agent = is_video_agent_model(model_name)

    if not is_video_agent:
        # For non-video-agent models, get video file path
        video_base_path = config['paths']['video_path']

        # Convert to absolute path if relative
        if not os.path.isabs(video_base_path):
            video_base_path = os.path.abspath(video_base_path)

        # Search for video file in nested subdirectories
        video_path = None
        if video_id:
            # print(video_id)
            for root, _, files in os.walk(video_base_path):
                if f"{video_id}.mp4" in files:
                    video_path = os.path.join(root, f"{video_id}.mp4")
                    break
    else:
        # For video-agent models, validate video_id is present
        if not video_id:
            raise ValueError("Video ID is required for video-agent models")
    # Extract conversation turns from the video benchmark sample
    conversations = sample.get('conversations', sample.get('question', []))
    base_url = config['server']['base_url']
    api_key = config['server']['api_key']

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Set up system message based on model type
    if is_video_agent:
        messages = []
        if config['system']['system_prompt']:
            messages.append({"role": "system", "content": config['system']['system_prompt']})
    else:
        messages = [{"role": "system", "content": [{"type": "text", "text": config['system']['system_prompt']}]}] if config['system']['system_prompt'] else []

    if isinstance(conversations, list):
        for i, turn in enumerate(conversations):
            if turn['role'] == 'user':
                if is_video_agent:
                    # For video-agent, add user message as simple text
                    messages.append({"role": "user", "content": turn['content']})
                else:
                    # Handle user turns for non-video-agent models
                    if i == 0:
                        # First user turn: validate video and include video reference
                        print(video_path)
                        if not video_path or not os.path.exists(video_path):
                            raise FileNotFoundError(f"Video file not found: {video_path}")

                        content = [
                            {"type": "video_url", "video_url": {"url": f"file://{video_path}"}},
                            {"type": "text", "text": turn['content']}
                        ]
                    else:
                        # Subsequent user turns: add only question
                        content = [{"type": "text", "text": turn['content']}]

                    messages.append({"role": "user", "content": content})

            elif turn['role'] == 'assistant':
                # Generate model response for assistant turns
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": config['generation']['max_tokens'],
                    "temperature": config['generation']['temperature'],
                    "top_p": config['generation']['top_p'],
                }

                # Add video-agent specific parameters
                if is_video_agent:
                    payload["video_id"] = video_id
                    payload["stream"] = False

                response = requests.post(
                    f"{base_url}chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=config['generation']['timeout']
                )
                response.raise_for_status()
                completion = response.json()

                turn['candidate_response'] = completion["choices"][0]["message"]["content"]

                # Add assistant response to message history
                if is_video_agent:
                    messages.append({
                        "role": "assistant",
                        "content": turn['content']
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": turn['content']}]
                    })

    return conversations


def evaluate_candidate_response(sample, eval_model="Qwen/Qwen3-14B", config=None, eval_file_path=""):
    """Evaluate a video benchmark response against ground truth criteria"""
    if config is None:
        config = load_config()

    if 'conversations' not in sample:
        return sample

    # Check if this is a tool calling task
    if is_tool_calling_task(sample):
        from tool_handler import evaluate_tool_calling_response
        return evaluate_tool_calling_response(sample, eval_model, config, eval_file_path)

    # Check if this is a hallucination test
    from hal_test import is_hallucination_test, evaluate_hallucination_response
    if is_hallucination_test(eval_file_path):
        return evaluate_hallucination_response(sample, eval_model, config, eval_file_path)

    # Use original ground truth evaluation for all evaluation models
    json_schema = {
        "type": "object",
        "properties": {
            "criteria_met": {"type": "boolean"}
        },
        "required": ["criteria_met"]
    }

    for turn in sample['conversations']:
        if turn['role'] != 'assistant':
            continue

        ground_truth_response = turn['content']
        model_response = turn['candidate_response']

        for criterion in turn['criteria']:
            prompt = f"""You are an expert evaluator specializing in video content analysis and multimodal understanding.

Your task is to evaluate the Model Response against the **single evaluation criterion** provided, using the Ground Truth Response as a reference.

Ground Truth Response:
{ground_truth_response}

Model Response:
{model_response}

Evaluation Criterion:
{criterion['description']}

Instructions:
- Evaluate ONLY the provided criterion in this assessment.
- Compare the Model Response to the Ground Truth Response and determine if the criterion is satisfied.
- If the Model Response satisfies the criterion, set "criteria_met" to true; otherwise, set it to false.
- Focus on video content understanding, temporal relationships, and multimodal analysis. /no_think"""

            payload = {
                "model": eval_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config['evaluation']['temperature'],
                "max_tokens": config['evaluation']['max_tokens'],
                "top_p": config['evaluation']['top_p'],
                "guided_json": json_schema
            }

            # Retry logic for API calls
            max_retries = 3
            retry_delay = 0.5

            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{config['server']['base_url']}chat/completions",
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=config['evaluation']['timeout']
                    )
                    response.raise_for_status()

                    result = response.json()

                    # Validate API response structure
                    if "choices" not in result or not result["choices"]:
                        raise ValueError("API response missing or empty 'choices' field")

                    message = result["choices"][0].get("message")
                    if message is None:
                        raise ValueError("API response missing 'message' field")

                    eval_content = message.get("content")
                    if eval_content is None:
                        finish_reason = result["choices"][0].get("finish_reason")
                        raise ValueError(f"API returned None content. Finish reason: {finish_reason}")

                    # Try to parse the JSON response
                    parsed_result = json.loads(eval_content)

                    # Successfully parsed - set the result and break
                    criterion['criteria_met'] = parsed_result['criteria_met']
                    break

                except (requests.exceptions.RequestException, ValueError, json.JSONDecodeError) as e:
                    if attempt < max_retries - 1:
                        print(f"[WARNING] Retry {attempt + 1}/{max_retries - 1} for sample {sample.get('sample_id')}: {str(e)[:100]}")
                        time.sleep(retry_delay)
                    else:
                        print(f"[ERROR] Failed after {max_retries} attempts for sample {sample.get('sample_id')}, criterion '{criterion.get('name')}': {e}")
                        # Set criteria_met to None to indicate evaluation failure
                        criterion['criteria_met'] = None
                        criterion['evaluation_error'] = str(e)

    return sample


def process_sample(sample, output_file, model_name, config=None, tool_mode=False):
    """Process a single sample - handles normal, hallucination, and tool calling tests"""
    # Check if this is a tool calling task
    if tool_mode or is_tool_calling_task(sample):
        response = generate_tool_calling_response(sample, model_name, config)
    else:
        response = generate_candidate_response(sample, model_name, config)

    sample['conversations'] = response

    lock_file = f"{output_file}.lock"
    with FileLock(lock_file):
        with open(output_file, 'a') as f:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    return sample

def process_video_sample(sample, output_file, model_name, config=None):
    """Legacy function - calls unified process_sample"""
    return process_sample(sample, output_file, model_name, config)

def process_video_evaluation(sample, eval_file, eval_model, config=None):
    """Process a single video benchmark evaluation and save result to the evaluation file"""
    evaluated_sample = evaluate_candidate_response(sample, eval_model, config, eval_file)

    lock_file = f"{eval_file}.lock"
    with FileLock(lock_file):
        with open(eval_file, 'a') as f:
            f.write(json.dumps(evaluated_sample, ensure_ascii=False) + '\n')
    
    return evaluated_sample

def start_vllm(model_name, tensor_parallel_size, config_type="candidate", config=None):
    """Start the VLLM server for video benchmark evaluation with the specified model"""
    if config is None:
        config = load_config()
    
    # Get vLLM configuration based on type (candidate or evaluation)
    vllm_config_key = f"vllm_{config_type}"
    if vllm_config_key not in config:
        vllm_config_key = "vllm_candidate"  # fallback to candidate config
    
    vllm_config = config[vllm_config_key]
    
    # Setup logging
    logs_dir = create_logs_directory(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = convert_to_underscored(model_name)
    log_file_path = os.path.join(logs_dir, f"vllm_{config_type}_{model_name_safe}_{timestamp}.log")

    # Build command
    cmd = [
        "vllm", "serve", model_name,
        "--tensor-parallel-size", str(tensor_parallel_size),
    ]
    
    # Add max-model-len only if specified
    if vllm_config['max_model_len']:
        cmd.extend(["--max-model-len", str(vllm_config['max_model_len'])])
    
    cmd.extend([
        "--max-num-seqs", str(vllm_config['max_num_seqs']), 
        "--port", config['server']['port'],
    ])
    
    # Add optional flags
    if vllm_config.get('trust_remote_code', True):
        cmd.append("--trust-remote-code")
    
    if 'allowed_local_media_path' in vllm_config:
        cmd.extend(["--allowed-local-media-path", vllm_config['allowed_local_media_path']])
    
    if vllm_config.get('enable_prefix_caching', True):
        cmd.append("--enable-prefix-caching")
    
    # Add any additional arguments
    if 'additional_args' in vllm_config and vllm_config['additional_args']:
        cmd.extend(vllm_config['additional_args'])
    
    print(f"Starting VLLM server ({config_type}) for video benchmark with model {model_name}...")
    print(f"Logs will be written to: {log_file_path}")

    # Start server process with logging
    log_file = open(log_file_path, 'w')
    server_process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    # Wait for server to be ready
    print(f"Waiting for {model_name} server to become available...")
    server_url = config['server']['base_url']
    url = f"{server_url}models"
    
    while True:
        try:
            requests.get(url, timeout=config['system']['server_check_timeout']).raise_for_status()
            print(f"\n✅ {model_name} server is up and ready for evaluation.")
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print(".", end="", flush=True)
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"\nUnexpected error: {e}")
            break
    
    return server_process, log_file

def kill_candidate_server(process, log_file=None):
    """Kill the VLLM video model server process and its children"""
    if not process:
        return
    
    print("Shutting down video model server...")
    try:
        parent = psutil.Process(process.pid)
        
        # Terminate all children first, then parent
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
        
        # Force kill any processes that didn't terminate gracefully
        _, still_alive = psutil.wait_procs([parent], timeout=10)
        for p in still_alive:
            p.kill()
        
        print("Video model server process terminated successfully")
    except Exception as e:
        print(f"Error killing video model server: {e}")
        process.kill()
    finally:
        # Close log file if it exists and is open
        if log_file and not log_file.closed:
            log_file.close()
            print("Log file closed")

def save_timing_data(timing_file, stage_name, time_data):
    """Save timing data for a specific stage to JSON file"""
    # Load existing timing data if file exists
    timing_data = {}
    if os.path.exists(timing_file):
        try:
            with open(timing_file, 'r') as f:
                timing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            timing_data = {}

    # Update with new stage timing
    timing_data[stage_name] = time_data

    # Save updated data
    with open(timing_file, 'w') as f:
        json.dump(timing_data, f, indent=2)

def is_tool_calling_task(sample):
    """Check if a sample requires tool calling"""
    # Check if sample has tool_calls in conversations
    if 'conversations' in sample:
        for turn in sample['conversations']:
            if turn.get('role') == 'assistant' and 'tool_calls' in turn:
                return True

    # Check if sample has required_tools field
    if 'required_tools' in sample:
        return True

    # Check if task is agentic_task
    if sample.get('task') == 'agentic_task':
        return True

    return False

def generate_tool_calling_response(sample, model_name, config=None):
    """Generate model response for a tool calling sample"""
    from tool_handler import process_tool_calling_sample
    return process_tool_calling_sample(sample, model_name, config)