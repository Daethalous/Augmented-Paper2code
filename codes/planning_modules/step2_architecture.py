from openai import OpenAI
import json
import os
from .utils import print_response, print_log_cost

def execute_architecture_stage(client, trajectories, gpt_version, output_dir, total_accumulated_cost, core_map_path=None, ref_snippets_path=None):
    current_stage = "[Planning] Architecture design"
    print(current_stage)

    extra_context = ""
    if core_map_path and os.path.exists(core_map_path):
        with open(core_map_path, 'r', encoding='utf-8') as f:
            extra_context += f"\n[BEGIN CORE MAP]\n{f.read()}\n[END CORE MAP]\n"
    
    if ref_snippets_path and os.path.exists(ref_snippets_path):
        with open(ref_snippets_path, 'r', encoding='utf-8') as f:
            extra_context += f"\n[BEGIN REFERENCE SNIPPETS]\n{f.read()}\n[END REFERENCE SNIPPETS]\n"

    if extra_context:
        prompt_content = f"""You are a senior system architect and an aggressive Code Pruner.
We have determined the [Must-Have Logic Components] necessary to reproduce this paper (provided in the previous response).
We also have the AST overview and reference snippets of the existing codebase skeleton:
{extra_context}

CRITICAL INSTRUCTION: Your goal is to map the [Must-Have Logic Components] to the existing codebase AST using a strict "Opt-in" strategy.

[Forced Mapping Rules]:
1. Default-DROP Principle: ANY file in the AST that does NOT directly serve one of the [Must-Have Logic Components] MUST be explicitly marked as [DROP]. Absolutely DO NOT retain unrelated datasets, legacy experiment scripts, or obsolete visualizers.
2. Inheritance and Modification: Traverse the [Must-Have Logic Components] and find the best matching file in the AST:
   - If a file matches the component's structure perfectly without changes, mark it as [REUSE].
   - If a file is structurally similar but requires code modifications based on the paper, mark it as [EDIT], and provide a brief summary of the required function/class edits.
   - If the AST completely lacks a file that can fulfill the specific logic, ONLY THEN are you allowed to mark it as [CREATE].
3. DRY Principle (Don't Repeat Yourself): If multiple baselines or logical components share a common function (e.g., identical loss or shared attention module), you MUST map that logic to a unified common file (e.g., utils.py) marked as [CREATE] or [EDIT]. DO NOT [CREATE] duplicate logic across different scripts.

Please analyze the provided AST and output a [Mutation List] clearly specifying all [REUSE], [EDIT], [CREATE], and [DROP] files with their justification.
For the "File list", ONLY include the relative paths of the files that survived (i.e. REUSE, EDIT, and CREATE). Do NOT include DROP files in the final File list.
"""
    else:
        prompt_content = """Your goal is to create a concise, usable, and complete software software architecture design for reproducing the paper's method from scratch.
Use appropriate open-source libraries and keep the overall architecture simple.
Based on the abstract plan for reproducing the paper's main method, please design a concise, usable, and complete software system file tree.
"""

    # 统一的格式要求，动态调整了示例内容以适应 Mutation 逻辑
    prompt_content += """
-----

## Format Example
[CONTENT]
{
    "Implementation approach": "We will map the paper's required Logic Components to the baseline repository. We observe the core attention mechanism is shared and must be newly created in `models/attention.py`, while we reuse the data loaders and discard old irrelevant experiments.",
    "Mutation List": "1. [REUSE] core/utils.py (Perfectly matches our data scaling component).\n2. [EDIT] models/resnet.py (Requires a new forward method injection of Spatial Attention as per the paper).\n3. [CREATE] models/attention.py (Newly identified mechanism missing from the AST).\n4. [DROP] visualization_old.py (Doesn't serve any new required logic).\n5. [DROP] data/mnist_loader.py (Paper only tests CIFAR/ImageNet, not MNIST).",
    "File list": [
        "train.py",  
        "dataset_loader.py", 
        "models/resnet.py",  
        "models/attention.py",
        "core/utils.py" 
    ],
    "Data structures and interfaces": "\nclassDiagram\n    class Main {\n        +__init__()\n        +run_experiment()\n    }\n    class ResNet {\n        +__init__(params: dict)\n        +forward(x: Tensor) -> Tensor\n    }\n    class Attention {\n        +__init__(dim: int)\n        +forward(x: Tensor) -> Tensor\n    }\n    Main --> ResNet\n    ResNet --> Attention\n",
    "Program call flow": "\nsequenceDiagram\n    participant M as Main\n    participant RN as ResNet\n    participant Att as Attention\n    M->>RN: forward(x)\n    RN->>Att: apply(features)\n    Att-->>RN: enhanced_features\n    RN-->>M: predictions\n",
    "Anything UNCLEAR": "Need clarification on the exact attention hidden dimension used."
}
[/CONTENT]

## Nodes: "<node>: <type>  # <instruction>"
- Implementation approach: <class 'str'>  # Summarize the mapping outcome and explain how it solves the Logic Component requirements.
- Mutation List: typing.Optional[str]  # MANDATORY if baseline map is provided. Detail the exact outcomes of [REUSE], [EDIT], [CREATE], and [DROP] files with rigorous justifications matching the Drop Principle.
- File list: typing.List[str]  # List all relative paths involved in this reproduction. NO [DROP] files.
- Data structures and interfaces: typing.Optional[str]  # Use mermaid classDiagram code syntax.
- Program call flow: typing.Optional[str] # Use sequenceDiagram code syntax.
- Anything UNCLEAR: <class 'str'>  # Mention ambiguities.

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT] like the format example, nothing else.

## Action
Follow the instructions for the nodes, generate the output, and ensure it follows the format example.
"""

    file_list_msg = [{"role": "user", "content": prompt_content}]

    messages_to_send = list(trajectories) + file_list_msg
    
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
    
    updated_trajectories = list(trajectories) + file_list_msg
    response_message = completion.choices[0].message
    updated_trajectories.append({'role': response_message.role, 'content': response_message.content})
    
    return completion_json, updated_trajectories, new_cost