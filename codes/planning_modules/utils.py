# Just copying utils.py logic here as it belongs to the package context, 
# or we can import from parent if we structure packages correctly. 
# For simplicity, I'll assume we can use relative import if running as module, 
# or we copy the content. Given the constraints, let's copy the minimal needed functions 
# or import from the original utils.py location if we are running the main script from `codes`.

import json
import os

# --- Copy of relevant utils functions ---

def cal_cost(response_json, model_name):
    model_cost = {
        # gpt-4.1
        "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
        "gpt-4.1-2025-04-14": {"input": 2.00, "cached_input": 0.50, "output": 8.00},

        # gpt-4.1-mini
        "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
        "gpt-4.1-mini-2025-04-14": {"input": 0.40, "cached_input": 0.10, "output": 1.60},

        # gpt-4.1-nano
        "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
        "gpt-4.1-nano-2025-04-14": {"input": 0.10, "cached_input": 0.025, "output": 0.40},

        # gpt-5-mini
        "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},

        # gpt-4.5-preview
        "gpt-4.5-preview": {"input": 75.00, "cached_input": 37.50, "output": 150.00},
        "gpt-4.5-preview-2025-02-27": {"input": 75.00, "cached_input": 37.50, "output": 150.00},

        # gpt-4o
        "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
        "gpt-4o-2024-08-06": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
        "gpt-4o-2024-11-20": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
        "gpt-4o-2024-05-13": {"input": 5.00, "cached_input": None, "output": 15.00},

        # gpt-4o-audio-preview
        "gpt-4o-audio-preview": {"input": 2.50, "cached_input": None, "output": 10.00},
        "gpt-4o-audio-preview-2024-12-17": {"input": 2.50, "cached_input": None, "output": 10.00},
        "gpt-4o-audio-preview-2024-10-01": {"input": 2.50, "cached_input": None, "output": 10.00},

        # gpt-4o-realtime-preview
        "gpt-4o-realtime-preview": {"input": 5.00, "cached_input": 2.50, "output": 20.00},
        "gpt-4o-realtime-preview-2024-12-17": {"input": 5.00, "cached_input": 2.50, "output": 20.00},
        "gpt-4o-realtime-preview-2024-10-01": {"input": 5.00, "cached_input": 2.50, "output": 20.00},

        # gpt-4o-mini
        "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
        "gpt-4o-mini-2024-07-18": {"input": 0.15, "cached_input": 0.075, "output": 0.60},

        # gpt-4o-mini-audio-preview
        "gpt-4o-mini-audio-preview": {"input": 0.15, "cached_input": None, "output": 0.60},
        "gpt-4o-mini-audio-preview-2024-12-17": {"input": 0.15, "cached_input": None, "output": 0.60},

        # gpt-4o-mini-realtime-preview
        "gpt-4o-mini-realtime-preview": {"input": 0.60, "cached_input": 0.30, "output": 2.40},
        "gpt-4o-mini-realtime-preview-2024-12-17": {"input": 0.60, "cached_input": 0.30, "output": 2.40},

        # o1
        "o1": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
        "o1-2024-12-17": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
        "o1-preview-2024-09-12": {"input": 15.00, "cached_input": 7.50, "output": 60.00},

        # o1-pro
        "o1-pro": {"input": 150.00, "cached_input": None, "output": 600.00},
        "o1-pro-2025-03-19": {"input": 150.00, "cached_input": None, "output": 600.00},

        # o3
        "o3": {"input": 10.00, "cached_input": 2.50, "output": 40.00},
        "o3-2025-04-16": {"input": 10.00, "cached_input": 2.50, "output": 40.00},

        # o4-mini
        "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
        "o4-mini-2025-04-16": {"input": 1.10, "cached_input": 0.275, "output": 4.40},

        # o3-mini
        "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
        "o3-mini-2025-01-31": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
    }

    
    prompt_tokens = response_json["usage"]["prompt_tokens"]
    completion_tokens = response_json["usage"]["completion_tokens"]
    cached_tokens = response_json["usage"]["prompt_tokens_details"].get("cached_tokens", 0)

    # input token = (prompt_tokens - cached_tokens)
    actual_input_tokens = prompt_tokens - cached_tokens
    output_tokens = completion_tokens

    cost_info = model_cost.get(model_name, {"input": 0, "cached_input": 0, "output": 0})
    if model_name not in model_cost:
        # Check prefix match for future proofing or fallback
        pass

    input_cost = (actual_input_tokens / 1_000_000) * cost_info['input']
    
    # Handle case where cached_input might be None in dict
    cached_unit_cost = cost_info['cached_input'] if cost_info['cached_input'] is not None else cost_info['input']
    cached_input_cost = (cached_tokens / 1_000_000) * cached_unit_cost
    
    output_cost = (output_tokens / 1_000_000) * cost_info['output']

    total_cost = input_cost + cached_input_cost + output_cost

    return {
        'model_name': model_name,
        'actual_input_tokens': actual_input_tokens,
        'input_cost': input_cost,
        'cached_tokens': cached_tokens,
        'cached_input_cost': cached_input_cost,
        'output_tokens': output_tokens,
        'output_cost': output_cost,
        'total_cost': total_cost,
    }

def print_response(completion_json, is_llm=False):
    print("============================================")
    if is_llm:
        print(completion_json['text'])
    else:
        print(completion_json['choices'][0]['message']['content'])
    print("============================================\n")

def print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost):
    usage_info = cal_cost(completion_json, gpt_version)

    current_cost = usage_info['total_cost']
    total_accumulated_cost += current_cost

    output_lines = []
    output_lines.append("🌟 Usage Summary 🌟")
    output_lines.append(f"{current_stage}")
    output_lines.append(f"🛠️ Model: {usage_info['model_name']}")
    output_lines.append(f"📥 Input tokens: {usage_info['actual_input_tokens']} (Cost: ${usage_info['input_cost']:.8f})")
    output_lines.append(f"📦 Cached input tokens: {usage_info['cached_tokens']} (Cost: ${usage_info['cached_input_cost']:.8f})")
    output_lines.append(f"📤 Output tokens: {usage_info['output_tokens']} (Cost: ${usage_info['output_cost']:.8f})")
    output_lines.append(f"💵 Current total cost: ${current_cost:.8f}")
    output_lines.append(f"🪙 Accumulated total cost so far: ${total_accumulated_cost:.8f}")
    output_lines.append("============================================\n")

    output_text = "\n".join(output_lines)
    
    print(output_text)

    # Make sure output_dir exists before writing
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/cost_info.log", "a", encoding="utf-8") as f:
            f.write(output_text + "\n")
    
    return total_accumulated_cost

def load_accumulated_cost(accumulated_cost_file):
    if os.path.exists(accumulated_cost_file):
        with open(accumulated_cost_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("total_cost", 0.0)
    else:
        return 0.0

def save_accumulated_cost(accumulated_cost_file, cost):
    os.makedirs(os.path.dirname(accumulated_cost_file), exist_ok=True)
    with open(accumulated_cost_file, "w", encoding="utf-8") as f:
        json.dump({"total_cost": cost}, f)
