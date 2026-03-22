from openai import OpenAI
import json
import os
from .utils import print_response, print_log_cost

def execute_logic_stage(client, trajectories, gpt_version, output_dir, total_accumulated_cost, core_map_path=None, ref_snippets_path=None):
    current_stage = "[Planning] Logic design"
    print(current_stage)

    # 1. 读取上下文（与 step2 保持一致）
    extra_context = ""
    if core_map_path and os.path.exists(core_map_path):
        with open(core_map_path, 'r', encoding='utf-8') as f:
            extra_context += f"\n[BEGIN CORE MAP]\n{f.read()}\n[END CORE MAP]\n"
    
    if ref_snippets_path and os.path.exists(ref_snippets_path):
        with open(ref_snippets_path, 'r', encoding='utf-8') as f:
            extra_context += f"\n[BEGIN REFERENCE SNIPPETS]\n{f.read()}\n[END REFERENCE SNIPPETS]\n"

    # 2. 动态生成 Prompt
    if extra_context:
        prompt_content = f"""You are a senior AI engineer. We are reproducing a paper by modifying an existing codebase.
Here is the codebase skeleton and reference snippets: 
{extra_context}

CRITICAL INSTRUCTION: Your task is to break down the implementation logic specifically for the files identified as [EDIT], [CREATE] and [REUSE] in the previous Architecture stage.
- For [EDIT] files: You MUST explicitly state which existing functions or classes (referencing the CORE MAP) need to be changed, and describe the exact mathematical or algorithmic patches required.
- For [CREATE] files: Describe the required classes, functions, and their internal logic from scratch.
- For [REUSE] files: You MUST include them in the logic analysis and explicitly state '[REUSE] Keep exactly as original'. Do not write detailed logic for them.
"""
    else:
        prompt_content = """Your goal is to break down tasks according to the PRD/technical design, generate a task list, and analyze task dependencies from scratch.
The Logic Analysis should not only consider the dependencies between files but also provide detailed descriptions to assist in writing the new code needed to reproduce the paper.
"""

    prompt_content += """
-----

## Format Example
[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0"  
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "models/resnet.py",
            "[EDIT] In the `ResNet` class, modify the `forward` function. Remove the existing Global Average Pooling (GAP) layer. Instead, import and instantiate the new `MILPooling` layer. Pass the feature map to this new pooling layer before the final linear classifier."
        ],
        [
            "models/mil_pooling.py",
            "[CREATE] Create a new file. Implement a `MILPooling` class inheriting from `nn.Module`. It should contain an attention network (a two-layer MLP with a hidden dimension of 8 and a Sigmoid activation) to compute instance weights. Multiply the features by these weights and sum them up."
        ],
        [
            "train.py",
            "[EDIT] In the `train_epoch` function, update the criterion to include the new regularizer mentioned in Equation 5 of the paper."
        ]
    ],
    "Shared Knowledge": "The `d_model` dimension is strictly set to 128 across all backbone outputs before entering the MIL pooling head.",
    "Anything UNCLEAR": "Are there specific initialization strategies required for the attention weights?"
}
[/CONTENT]

## Nodes: "<node>: <type>  # <instruction>"
- Required packages: typing.List[str]  # Required python packages for reproducing the paper.
- Required Other language third-party packages: typing.List[str]  # E.g., specific system dependencies or C++ extensions.
- Logic Analysis: typing.List[typing.List[str]]  # A list of [file_path, logic_description]. You MUST include ALL files mentioned in the architecture, including the [REUSE] ones. For [REUSE] files, just set the description to '[REUSE] Keep exactly as original'. For [EDIT] and [CREATE] files, give EXTREMELY SPECIFIC instructions (e.g., function names, variable changes, math formulas) so a coder can directly write/edit the code.
- Shared Knowledge: <class 'str'>  # Detail any shared knowledge, like common utility functions or configuration variables.
- Anything UNCLEAR: <class 'str'>  # Mention any unresolved questions.

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT] like the format example, nothing else.

## Action
Follow the node instructions above, generate your output accordingly, and ensure it follows the given format example.
"""

    task_list_msg = [{'role': 'user', 'content': prompt_content}]
    messages_to_send = list(trajectories) + task_list_msg
    
    if "o3-mini" in gpt_version:
        completion = client.chat.completions.create(
            model=gpt_version, 
            reasoning_effort="high",
            messages=messages_to_send
        )
    else:
        completion = client.chat.completions.create(
            model=gpt_version, 
            messages=messages_to_send
        )

    completion_json = json.loads(completion.model_dump_json())
    
    print_response(completion_json)
    new_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
    
    updated_trajectories = list(trajectories) + task_list_msg
    response_message = completion.choices[0].message
    updated_trajectories.append({'role': response_message.role, 'content': response_message.content})
    
    return completion_json, updated_trajectories, new_cost