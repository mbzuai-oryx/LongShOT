import json
import re
import os
import requests
from utils import load_config, is_video_agent_model

def load_tools_definitions(tools_file="tools.json"):
    """Load tool definitions from JSON file"""
    if not os.path.exists(tools_file):
        raise FileNotFoundError(f"Tools file not found: {tools_file}")

    with open(tools_file, 'r') as f:
        tools = json.load(f)

    return tools

def create_tools_prompt(tools):
    """Create a prompt section describing available tools"""
    tools_desc = "Available tools:\n\n"

    for tool in tools:
        name = tool['name']
        inputs = tool['inputs']
        outputs = tool['outputs']
        description = tool['description']

        # Format inputs with optional indicators
        formatted_inputs = []
        for inp in inputs:
            if inp.endswith('?'):
                formatted_inputs.append(f"{inp[:-1]} (optional)")
            elif inp.endswith('[]'):
                formatted_inputs.append(f"{inp[:-2]} (array, optional if no ? marker)")
            else:
                formatted_inputs.append(f"{inp} (required)")

        inputs_str = ', '.join(formatted_inputs)

        tools_desc += f"- **{name}**: {description}\n"
        tools_desc += f"  Parameters: {inputs_str}\n"
        tools_desc += f"  Returns: {outputs}\n\n"

    return tools_desc

def extract_json_from_response(response_text):
    """Extract JSON from model response, handling various formats"""
    if not response_text:
        return None

    # Try direct JSON parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from code blocks
    json_pattern = r'```(?:json)?\s*(\{[^`]*\})\s*```'
    matches = re.findall(json_pattern, response_text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Try finding JSON-like structures in the text
    json_pattern = r'\{[^{}]*"criteria_met"[^{}]*\}'
    matches = re.findall(json_pattern, response_text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Try extracting just the boolean value
    bool_pattern = r'"criteria_met":\s*(true|false)'
    match = re.search(bool_pattern, response_text, re.IGNORECASE)
    if match:
        return {"criteria_met": match.group(1).lower() == "true"}

    # Last resort - look for yes/no patterns
    if re.search(r'\b(yes|true|correct|met)\b', response_text, re.IGNORECASE):
        return {"criteria_met": True}
    elif re.search(r'\b(no|false|incorrect|not met|failed)\b', response_text, re.IGNORECASE):
        return {"criteria_met": False}

    # Default fallback
    print(f"Warning: Could not parse JSON from response: {response_text[:200]}...")
    return {"criteria_met": False}

def parse_tool_calls_from_response(response_text):
    """Parse tool calls from model response - supports both JSON and text formats"""
    tool_calls = []

    # First try to parse JSON format (preferred)
    try:
        # Look for JSON blocks containing tool_calls
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)

        for json_match in json_matches:
            try:
                parsed_json = json.loads(json_match)
                if 'tool_calls' in parsed_json:
                    for tool_call in parsed_json['tool_calls']:
                        if 'tool' in tool_call:
                            tool_calls.append({
                                'tool': tool_call['tool'],
                                'parameters': tool_call.get('parameters', {})
                            })
                    break  # Found valid JSON, stop looking
            except json.JSONDecodeError:
                continue

        # If no JSON blocks found, try to find inline JSON
        if not tool_calls:
            # Look for {"tool_calls": [...]} pattern
            inline_pattern = r'\{[^{}]*"tool_calls"[^{}]*\[[^\]]*\][^{}]*\}'
            inline_matches = re.findall(inline_pattern, response_text, re.DOTALL)

            for inline_match in inline_matches:
                try:
                    parsed_json = json.loads(inline_match)
                    if 'tool_calls' in parsed_json:
                        for tool_call in parsed_json['tool_calls']:
                            if 'tool' in tool_call:
                                tool_calls.append({
                                    'tool': tool_call['tool'],
                                    'parameters': tool_call.get('parameters', {})
                                })
                        break
                except json.JSONDecodeError:
                    continue

    except Exception:
        pass

    # Fallback to text format parsing if JSON parsing failed
    if not tool_calls:
        # Look for TOOL_CALL: pattern
        pattern = r'TOOL_CALL:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)'
        matches = re.findall(pattern, response_text, re.DOTALL)

        for match in matches:
            tool_name = match[0]
            params_str = match[1].strip()

            # Parse parameters
            parameters = {}
            if params_str:
                # Simple parameter parsing - handles key="value" format
                param_pattern = r'(\w+)\s*=\s*["\']([^"\']*)["\']'
                param_matches = re.findall(param_pattern, params_str)
                for param_match in param_matches:
                    key, value = param_match
                    # Try to convert to appropriate type
                    if value.startswith('[') and value.endswith(']'):
                        # Handle array format like ["val1", "val2"]
                        try:
                            parameters[key] = json.loads(value)
                        except:
                            parameters[key] = value
                    elif value.replace('.', '').isdigit():
                        # Handle numeric values
                        parameters[key] = float(value) if '.' in value else int(value)
                    else:
                        parameters[key] = value

            tool_calls.append({
                'tool': tool_name,
                'parameters': parameters
            })

    return tool_calls

def simulate_tool_execution(tool_call, expected_tool_calls):
    """Simulate tool execution by finding matching expected response"""
    tool_name = tool_call['tool']

    # Find matching expected tool call
    for expected_call in expected_tool_calls:
        if expected_call['tool'] == tool_name:
            # Return the expected response
            return expected_call['expected_response']

    # If no match found, return a generic response
    return f'{{"status": "tool_not_found", "message": "Tool {tool_name} not found in expected calls"}}'

def get_tool_calls_from_turn(turn):
    """Extract tool calls from turn, handling both JSON string list and object list formats"""
    if 'tool_calls' in turn and turn['tool_calls']:
        tool_calls = turn['tool_calls']

        if isinstance(tool_calls, list):
            parsed_tool_calls = []
            for tool_call in tool_calls:
                if isinstance(tool_call, str):
                    # New format: tool_call is a JSON string
                    try:
                        parsed_tool_calls.append(json.loads(tool_call))
                    except (json.JSONDecodeError, TypeError):
                        continue
                elif isinstance(tool_call, dict):
                    # Original format: tool_call is already an object
                    parsed_tool_calls.append(tool_call)
            return parsed_tool_calls
        else:
            return []

    # Handle legacy JSON string format (if it exists)
    if 'tool_calls_json' in turn:
        try:
            return json.loads(turn['tool_calls_json'])
        except (json.JSONDecodeError, TypeError):
            return []

    return []

def get_valid_tool_parameters(tools_definitions):
    """Extract all valid parameters from tool definitions"""
    valid_params = set()
    for tool in tools_definitions:
        for param in tool['inputs']:
            # Clean parameter name (remove ?, [], etc.)
            clean_param = param.rstrip('?[]')
            valid_params.add(clean_param)
    return valid_params

def count_tool_parameters(tool_name, tools_definitions):
    """Count required and optional parameters for a tool"""
    for tool in tools_definitions:
        if tool['name'] == tool_name:
            required_params = 0
            optional_params = 0
            for param in tool['inputs']:
                if param.endswith('?'):
                    optional_params += 1
                elif param.endswith('[]'):
                    # Arrays can be required or optional depending on context
                    # For now, treat as optional since models might not always provide
                    optional_params += 1
                elif param.startswith('timestamp'):
                    # Timestamps are optional as per system prompt instructions
                    optional_params += 1
                elif param.startswith('language'):
                    # Language is often optional
                    optional_params += 1
                else:
                    required_params += 1
            return required_params, optional_params
    return 0, 0

def generate_tool_criteria(expected_tool_calls, tools_definitions):
    """Auto-generate evaluation criteria based on expected tool calls and 6_tools.txt rules"""
    criteria = []

    if not expected_tool_calls:
        return criteria

    # Get valid tool parameters from definitions to filter out invalid ones
    valid_params = get_valid_tool_parameters(tools_definitions)

    # Known API-level parameters that should be filtered out (not tool parameters)
    invalid_api_params = {'video_id', 'stream', 'api_key', 'model', 'temperature', 'max_tokens'}

    # Filter expected tool calls to only include valid tool parameters
    filtered_tool_calls = []
    for call in expected_tool_calls:
        filtered_call = call.copy()
        if 'parameters' in call and call['parameters']:
            # Filter out invalid parameters
            original_params = call['parameters']
            filtered_params = {k: v for k, v in original_params.items()
                             if k in valid_params and k not in invalid_api_params}

            # Log filtered out parameters for debugging
            removed_params = set(original_params.keys()) - set(filtered_params.keys())
            if removed_params:
                print(f"Filtered out invalid parameters for tool '{call['tool']}': {removed_params}")

            filtered_call['parameters'] = filtered_params
        filtered_tool_calls.append(filtered_call)

    # Extract expected tools
    expected_tools = [call['tool'] for call in filtered_tool_calls]
    unique_tools = list(set(expected_tools))

    # 1. Tool Selection Accuracy (+1 each correct tool)
    criteria.append({
        "name": "tool_selection_accuracy",
        "description": f"Must use exactly these tools: {unique_tools}. No other tools allowed. +1 point for each correct tool used.",
        "weight": len(unique_tools),
        "is_penalty": False
    })

    # 2. Required Parameter Accuracy (+1 each required param)
    total_required_params = 0
    total_optional_params = 0

    for call in filtered_tool_calls:
        tool_name = call['tool']
        req_params, opt_params = count_tool_parameters(tool_name, tools_definitions)
        total_required_params += req_params
        total_optional_params += opt_params

    if total_required_params > 0:
        criteria.append({
            "name": "required_parameter_accuracy",
            "description": f"Must provide {total_required_params} required parameters correctly across all tool calls.",
            "weight": total_required_params,
            "is_penalty": False
        })

    # 3. Optional Parameter Precision (+0.25 bonus each)
    if total_optional_params > 0:
        criteria.append({
            "name": "optional_parameter_precision",
            "description": f"Award +0.25 bonus for each of {total_optional_params} optional parameters provided correctly (e.g., timestamp, language).",
            "weight": total_optional_params * 0.25,
            "is_penalty": False
        })

    # 4. Sequence Logic (+1 for correct sequence)
    criteria.append({
        "name": "sequence_logic",
        "description": f"Must call {len(filtered_tool_calls)} tool(s) in logical sequence. No tools should be called before identifying the appropriate context.",
        "weight": 1,
        "is_penalty": False
    })

    # 5. Efficiency Penalty (-1 per extra tool call, cap -5)
    criteria.append({
        "name": "efficiency_penalty",
        "description": f"Penalty for extra tool calls beyond the {len(filtered_tool_calls)} needed. -1 per extra call, maximum penalty capped at -5.",
        "weight": 1,
        "is_penalty": True
    })

    # 6. Wrong Tool Call Penalty (-2 each incorrect tool)
    criteria.append({
        "name": "wrong_tool_call",
        "description": "Penalty for using incorrect tools not in the expected set. -2 for each wrong tool used.",
        "weight": 2,
        "is_penalty": True
    })

    # 7. Final Answer Correctness (+3 for complete accuracy)
    criteria.append({
        "name": "final_answer_correctness",
        "description": "Must provide correct final answer that integrates tool results appropriately and addresses the original question completely.",
        "weight": 3,
        "is_penalty": False
    })

    return criteria

def process_tool_calling_sample(sample, model_name, config=None):
    """Generate model response for a tool calling sample using 2-step process"""
    if config is None:
        config = load_config()

    # Load tools definitions
    tools = load_tools_definitions()

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
            for root, _, files in os.walk(video_base_path):
                if f"{video_id}.mp4" in files:
                    video_path = os.path.join(root, f"{video_id}.mp4")
                    break
    else:
        # For video-agent models, validate video_id is present
        if not video_id:
            raise ValueError("Video ID is required for video-agent models")

    # Extract conversation turns from the sample
    conversations = sample.get('conversations', [])

    base_url = config['server']['base_url']
    api_key = config['server']['api_key']

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Create tools prompt for step 1 (tool selection)
    tools_prompt = create_tools_prompt(tools)

    # Step 1 system prompt - ask model to select tools
    step1_system_prompt = f"""You are a helpful assistant that can analyze videos and use tools to answer questions.

{tools_prompt}

Your task is to analyze the user's question about the video and determine which tools you need to use to provide a complete answer.

Respond with JSON in this format:

```json
{{
  "tool_calls": [
    {{
      "tool": "tool_name",
      "parameters": {{
        "required_param": "value",
        "optional_param": "value"
      }}
    }}
  ],
  "reasoning": "Why you selected these tools and how they will help answer the question"
}}
```

Important notes:
- Only select the tools you actually need to answer the question
- **Timestamps are optional** - only include if you're confident about the timing
- You can call multiple tools by adding more objects to the tool_calls array
- Provide clear reasoning for your tool selection
- Do NOT provide a final answer yet - just select the appropriate tools

Fallback: If JSON doesn't work, use: TOOL_CALL: tool_name(param="value")"""

    # Step 2 system prompt - provide final answer with tool results
    step2_system_prompt = """You are a helpful assistant analyzing videos. You previously selected tools to help answer a question.

The tools have now been executed and their results are provided below. Use these tool results to provide a complete and accurate final answer to the user's original question.

Focus on:
- Integrating the tool results appropriately
- Providing a clear, comprehensive answer
- Addressing all aspects of the original question
- Being accurate based on the tool outputs provided"""

    # Set up initial messages based on model type
    if is_video_agent:
        messages = []
        if step1_system_prompt:
            messages.append({"role": "system", "content": step1_system_prompt})
    else:
        messages = [{"role": "system", "content": [{"type": "text", "text": step1_system_prompt}]}] if step1_system_prompt else []

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
                # === STEP 1: Tool Selection ===
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

                step1_response = requests.post(
                    f"{base_url}chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=config['generation']['timeout']
                )
                step1_response.raise_for_status()
                step1_completion = step1_response.json()

                step1_model_response = step1_completion["choices"][0]["message"]["content"]

                # Parse tool calls from step 1 response
                parsed_tool_calls = parse_tool_calls_from_response(step1_model_response)

                # Get expected tool calls from sample (handle both formats)
                expected_tool_calls = get_tool_calls_from_turn(turn)

                # Simulate tool execution to get results for step 2
                simulated_results = []
                for tool_call in expected_tool_calls:  # Use expected tool calls for simulation
                    tool_result = simulate_tool_execution(tool_call, expected_tool_calls)
                    simulated_results.append({
                        'tool_call': tool_call,
                        'result': tool_result
                    })

                # === STEP 2: Final Answer with Tool Results ===
                # Create step 2 messages with tool results
                step2_messages = []
                if is_video_agent:
                    step2_messages.append({"role": "system", "content": step2_system_prompt})
                    step2_messages.append({"role": "user", "content": turn['content']})
                else:
                    step2_messages.append({"role": "system", "content": [{"type": "text", "text": step2_system_prompt}]})
                    # Include video for first user turn in step 2
                    if i == 0:
                        content = [
                            {"type": "video_url", "video_url": {"url": f"file://{video_path}"}},
                            {"type": "text", "text": turn['content']}
                        ]
                    else:
                        content = [{"type": "text", "text": turn['content']}]
                    step2_messages.append({"role": "user", "content": content})

                # Add tool results to step 2 conversation
                tool_results_text = "\n\nTool Results:\n"
                for result in simulated_results:
                    tool_name = result['tool_call']['tool']
                    tool_params = result['tool_call'].get('parameters', {})
                    tool_output = result['result']
                    tool_results_text += f"Tool: {tool_name}\nParameters: {json.dumps(tool_params, indent=2)}\nResult: {tool_output}\n\n"

                tool_results_text += "Please provide your final answer based on these tool results:"

                if is_video_agent:
                    step2_messages.append({"role": "user", "content": tool_results_text})
                else:
                    step2_messages.append({"role": "user", "content": [{"type": "text", "text": tool_results_text}]})

                # Make step 2 API call
                step2_payload = {
                    "model": model_name,
                    "messages": step2_messages,
                    "max_tokens": config['generation']['max_tokens'],
                    "temperature": config['generation']['temperature'],
                    "top_p": config['generation']['top_p'],
                }

                if is_video_agent:
                    step2_payload["video_id"] = video_id
                    step2_payload["stream"] = False

                step2_response = requests.post(
                    f"{base_url}chat/completions",
                    headers=headers,
                    json=step2_payload,
                    timeout=config['generation']['timeout']
                )
                step2_response.raise_for_status()
                step2_completion = step2_response.json()

                step2_model_response = step2_completion["choices"][0]["message"]["content"]

                # Store all responses and metadata
                turn['tool_selection_response'] = step1_model_response  # Step 1: Tool selection
                turn['final_response'] = step2_model_response  # Step 2: Final answer
                turn['candidate_response'] = step2_model_response  # For backward compatibility
                turn['parsed_tool_calls'] = parsed_tool_calls  # From step 1
                turn['simulated_results'] = simulated_results
                turn['raw_model_response'] = step1_model_response  # Keep step 1 as raw response

                # Add assistant response to message history (use ground truth for controlled evaluation)
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

def evaluate_tool_calling_response(sample, eval_model="Qwen/Qwen3-14B", config=None, eval_file_path=""):
    """Evaluate tool calling response using auto-generated criteria and LLM-based evaluation"""
    if config is None:
        config = load_config()

    if 'conversations' not in sample:
        return sample

    # Load tools definitions for criteria generation
    tools_definitions = load_tools_definitions()

    # Use auto-generated evaluation criteria for tool calling
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

        # Get responses from 2-step process
        ground_truth_response = turn['content']
        tool_selection_response = turn.get('tool_selection_response', '')
        final_answer_response = turn.get('final_response', '')
        model_response = turn.get('candidate_response', final_answer_response)  # Backward compatibility

        # Get expected tool calls and auto-generate criteria (handle both formats)
        expected_tool_calls = get_tool_calls_from_turn(turn)
        parsed_tool_calls = turn.get('parsed_tool_calls', [])

        # Auto-generate criteria based on expected tool calls and 6_tools.txt rules
        auto_criteria = generate_tool_criteria(expected_tool_calls, tools_definitions)

        # Replace manual criteria with auto-generated ones
        turn['criteria'] = auto_criteria

        for criterion in auto_criteria:
            # Create evaluation prompt based on criterion type
            if 'tool_selection_accuracy' in criterion['name']:
                # Use tool selection response for tool-related criteria
                prompt = create_tool_selection_prompt(criterion, expected_tool_calls, parsed_tool_calls, tool_selection_response)
            elif 'parameter_accuracy' in criterion['name']:
                # Use tool selection response for parameter accuracy
                prompt = create_parameter_accuracy_prompt(criterion, expected_tool_calls, parsed_tool_calls, tool_selection_response)
            elif 'sequence_logic' in criterion['name']:
                # Use tool selection response for sequence logic
                prompt = create_sequence_logic_prompt(criterion, expected_tool_calls, parsed_tool_calls, tool_selection_response)
            elif 'efficiency_penalty' in criterion['name']:
                # Use tool selection response for efficiency
                prompt = create_efficiency_prompt(criterion, expected_tool_calls, parsed_tool_calls, tool_selection_response)
            elif 'wrong_tool_call' in criterion['name']:
                # Use tool selection response for wrong tool evaluation
                prompt = create_wrong_tool_prompt(criterion, expected_tool_calls, parsed_tool_calls, tool_selection_response)
            elif 'final_answer_correctness' in criterion['name']:
                # Use final answer response for answer correctness
                prompt = create_final_answer_prompt(criterion, ground_truth_response, final_answer_response)
            else:
                # Generic evaluation prompt - use tool selection for tool-related, final answer for content
                if 'tool' in criterion['name'].lower():
                    evaluation_response = tool_selection_response
                else:
                    evaluation_response = final_answer_response

                prompt = f"""You are an expert evaluator for tool calling systems.

Evaluation Criterion:
{criterion['description']}

Expected Tool Calls:
{json.dumps(expected_tool_calls, indent=2)}

Model's Tool Calls:
{json.dumps(parsed_tool_calls, indent=2)}

Model Response:
{evaluation_response}

Ground Truth Response:
{ground_truth_response}

Instructions:
- Evaluate ONLY the provided criterion.
- Set "criteria_met" to true if the criterion is satisfied, false otherwise.
- Consider the tool calling context and requirements."""

            payload = {
                "model": eval_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config['evaluation']['temperature'],
                "max_tokens": config['evaluation']['max_tokens'],
                "top_p": config['evaluation']['top_p'],
                "guided_json": json_schema
            }

            response = requests.post(
                f"{config['server']['base_url']}chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=config['evaluation']['timeout']
            )
            response.raise_for_status()

            result = response.json()
            eval_content = result["choices"][0]["message"]["content"]
            parsed_result = extract_json_from_response(eval_content)
            criterion['criteria_met'] = parsed_result['criteria_met'] if parsed_result else False

    return sample

def create_tool_selection_prompt(criterion, expected_tools, parsed_tools, model_response):
    """Create evaluation prompt for tool selection accuracy"""
    expected_tool_names = [tool['tool'] for tool in expected_tools]
    parsed_tool_names = [tool['tool'] for tool in parsed_tools]

    return f"""Evaluate tool selection accuracy.

Criterion: {criterion['description']}

Expected Tools: {expected_tool_names}
Model's Tools: {parsed_tool_names}

Model Response:
{model_response}

Did the model use exactly the correct tools as specified? Set "criteria_met" to true if yes, false if wrong tools were used."""

def create_parameter_accuracy_prompt(criterion, expected_tools, parsed_tools, model_response):
    """Create evaluation prompt for parameter accuracy"""
    return f"""Evaluate parameter accuracy for tool calls.

Criterion: {criterion['description']}

Expected Tool Calls:
{json.dumps(expected_tools, indent=2)}

Model's Tool Calls:
{json.dumps(parsed_tools, indent=2)}

Did the model provide the required parameters correctly? Consider both required and optional parameters as specified in the criterion."""

def create_sequence_logic_prompt(criterion, expected_tools, parsed_tools, model_response):
    """Create evaluation prompt for sequence logic"""
    return f"""Evaluate the logical sequence of tool calls.

Criterion: {criterion['description']}

Expected sequence (based on expected tools): {len(expected_tools)} tool calls
Model's sequence: {len(parsed_tools)} tool calls

Tool sequence: {[tool['tool'] for tool in parsed_tools]}

Did the model follow the correct sequence logic for tool calling?"""

def create_efficiency_prompt(criterion, expected_tools, parsed_tools, model_response):
    """Create evaluation prompt for efficiency (penalty for extra tools)"""
    expected_count = len(expected_tools)
    actual_count = len(parsed_tools)
    extra_calls = max(0, actual_count - expected_count)

    return f"""Evaluate efficiency - penalize extra tool calls.

Criterion: {criterion['description']}

Expected number of tool calls: {expected_count}
Actual number of tool calls: {actual_count}
Extra calls: {extra_calls}

If there are extra tool calls beyond what's needed, this criterion should NOT be met (criteria_met = false)."""

def create_wrong_tool_prompt(criterion, expected_tools, parsed_tools, model_response):
    """Create evaluation prompt for wrong tool call penalty"""
    expected_tool_names = set(tool['tool'] for tool in expected_tools)
    parsed_tool_names = [tool['tool'] for tool in parsed_tools]
    wrong_tools = [tool for tool in parsed_tool_names if tool not in expected_tool_names]

    return f"""Evaluate wrong tool call penalty.

Criterion: {criterion['description']}

Expected Tools: {list(expected_tool_names)}
Model's Tools: {parsed_tool_names}
Wrong Tools Used: {wrong_tools}

Model Response:
{model_response}

Did the model use any incorrect tools not in the expected set? Set "criteria_met" to false if wrong tools were used, true if only correct tools were used."""

def create_final_answer_prompt(criterion, ground_truth, model_response):
    """Create evaluation prompt for final answer correctness"""
    return f"""Evaluate final answer correctness.

Criterion: {criterion['description']}

Ground Truth Response:
{ground_truth}

Model Response:
{model_response}

Does the model's final answer correctly address the question based on the tool results? The answer should integrate information from tool outputs appropriately."""