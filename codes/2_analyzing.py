from openai import OpenAI
import json
import os
from tqdm import tqdm
import sys
from utils import extract_planning, content_to_json, print_response, print_log_cost, load_accumulated_cost, save_accumulated_cost
import copy

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name',type=str)
parser.add_argument('--gpt_version',type=str, default="o3-mini")
parser.add_argument('--paper_format',type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str) # json format
parser.add_argument('--pdf_latex_path', type=str) # latex format
parser.add_argument('--output_dir',type=str, default="")

args    = parser.parse_args()

client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

paper_name = args.paper_name
gpt_version = args.gpt_version
paper_format = args.paper_format
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path
output_dir = args.output_dir
    
if paper_format == "JSON":
    with open(f'{pdf_json_path}') as f:
        paper_content = json.load(f)
elif paper_format == "LaTeX":
    with open(f'{pdf_latex_path}') as f:
        paper_content = f.read()
else:
    print(f"[ERROR] Invalid paper format. Please select either 'JSON' or 'LaTeX.")
    sys.exit(0)


with open(f'{output_dir}/planning_config.yaml') as f: 
    config_yaml = f.read()

context_lst = extract_planning(f'{output_dir}/planning_trajectories.json')

# 0: overview, 1: detailed, 2: PRD
if os.path.exists(f'{output_dir}/task_list.json'):
    with open(f'{output_dir}/task_list.json') as f:
        task_list = json.load(f)
else:
    task_list = content_to_json(context_lst[2])

if 'Task list' in task_list:
    todo_file_lst = task_list['Task list']
elif 'task_list' in task_list:
    todo_file_lst = task_list['task_list']
elif 'task list' in task_list:
    todo_file_lst = task_list['task list']
elif 'Logic Analysis' in task_list:
    # Extract file list directly from Logic Analysis if Task list is missing
    todo_file_lst = [item[0] for item in task_list['Logic Analysis'] if isinstance(item, list) and len(item) > 0]
else:
    print(f"[ERROR] 'Task list' does not exist. Please re-generate the planning.")
    sys.exit(0)

if 'Logic Analysis' in task_list:
    logic_analysis = task_list['Logic Analysis']
elif 'logic_analysis' in task_list:
    logic_analysis = task_list['logic_analysis']
elif 'logic analysis' in task_list:
    logic_analysis = task_list['logic analysis']
else:
    print(f"[ERROR] 'Logic Analysis' does not exist. Please re-generate the planning.")
    sys.exit(0)
    
done_file_lst = ['config.yaml']
logic_analysis_dict = {}
for desc in task_list['Logic Analysis']:
    logic_analysis_dict[desc[0]] = desc[1]

def read_original_source_code(file_path, base_output_dir):
    cloned_repos_dir = os.path.join(base_output_dir, "cloned_repos")
    if not os.path.exists(cloned_repos_dir):
        return None
    for d in os.listdir(cloned_repos_dir):
        if d.startswith("core_"):
            # Try to match the exact file_path inside the cloned repo
            target_path = os.path.join(cloned_repos_dir, d, file_path)
            if os.path.exists(target_path):
                with open(target_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
    return None

def get_dynamic_prompt(todo_file_name, todo_file_desc, action_tag, original_code, core_map_content, paper_content, context_lst, config_yaml):
    if action_tag == "EDIT" and original_code:
        instruction = f"""INSTRUCTION:
You are modifying an existing file as part of replicating a target research paper. 

Below is the GLOBAL REPOSITORY MAP defining the skeleton of the baseline framework:
[GLOBAL REPO MAP START]
{core_map_content}
[GLOBAL REPO MAP END]

Below is the ORIGINAL SOURCE CODE of the file you need to edit:
[ORIGINAL SOURCE CODE START]
{original_code}
[ORIGINAL SOURCE CODE END]

Do NOT write code from scratch. Instead, write a highly precise "Modification Guide".
Combine the paper's specific methodology with the provided source code and explicitly state:
1. Which specific classes or functions must be retained.
2. Which specific classes or functions need to be modified.
3. Detailed logical differences (e.g., "In the `forward` function, replace operation A with operation B").
Be analytical, actionable, and exact. Write the logic analysis for '{todo_file_name}'."""
    else:
        instruction = f"""INSTRUCTION:
You are creating a newly authored file within an existing system architecture to replicate a target research paper.

Below is the GLOBAL REPOSITORY MAP defining the skeleton of the underlying framework:
[GLOBAL REPO MAP START]
{core_map_content}
[GLOBAL REPO MAP END]

Please refer to the global repository map, configuration, and architecture plan to thoroughly design the logic for this new file '{todo_file_name}'.
Write out the detailed module design, variable states, core function signatures, and explicit frontend/backend dependencies or inputs/outputs within the existing architecture. 
Focus strictly on a complete analytical blueprint rather than direct line-by-line coding.
"""

    write_msg=[{'role': 'system', "content": """You are an expert researcher, strategic analyzer and software engineer with a deep understanding of experimental design and reproducibility.
Your task is to conduct a comprehensive logic analysis to accurately reproduce the experiments described in the research paper. 
1. Align with the Paper: Strictly follow the methods, methodologies, and setups.
2. Be Clear and Structured: Present analysis in a logical, well-organized format.
3. Follow design: Follow "Data structures and interfaces" strictly.
"""},
    {'role': 'user', "content": f"""## Target File Description
{todo_file_desc}

-----
## Paper
{paper_content}

-----
## Overview of the plan
{context_lst[0]}

-----
## Design
{context_lst[1]}

-----
## Configuration file
```yaml
{config_yaml}
```
-----
{instruction}

-----
## Logic Analysis: {todo_file_name}"""}]
    
    return write_msg


def api_call(msg):
    if "o3-mini" in gpt_version:
        completion = client.chat.completions.create(
            model=gpt_version, 
            reasoning_effort="high",
            messages=msg
        )
    else:
        completion = client.chat.completions.create(
            model=gpt_version, 
            messages=msg
        )
    return completion


artifact_output_dir=f'{output_dir}/analyzing_artifacts'
os.makedirs(artifact_output_dir, exist_ok=True)

core_map_content = ""
core_map_path = os.path.join(output_dir, "core_repo_map.txt")
if os.path.exists(core_map_path):
    with open(core_map_path, "r", encoding="utf-8") as f:
        core_map_content = f.read()

total_accumulated_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")
for todo_file_name in tqdm(todo_file_lst):
    responses = []
    trajectories = []

    current_stage=f"[ANALYSIS] {todo_file_name}"
    print(f"\n{current_stage}")
    if todo_file_name == "config.yaml":
        continue
    
    if todo_file_name not in logic_analysis_dict:
        logic_analysis_dict[todo_file_name] = ""
        
    desc = logic_analysis_dict[todo_file_name]
    
    # Parse Action Tags
    arch_design_context = context_lst[1] if len(context_lst) > 1 else ""
    
    if "[REUSE]" in desc or f"[REUSE] {todo_file_name}" in arch_design_context:
        print(f"> Skipping {todo_file_name} (Tagged as REUSE)")
        done_file_lst.append(todo_file_name)
        continue
        
    action_tag = "CREATE"
    if "[EDIT]" in desc or f"[EDIT] {todo_file_name}" in arch_design_context:
        action_tag = "EDIT"
        
    original_code = None
    if action_tag == "EDIT":
        original_code = read_original_source_code(todo_file_name, output_dir)
        if not original_code:
            print(f"> [WARNING] Could not find local source for {todo_file_name}. Falling back to CREATE.")
            action_tag = "CREATE"
        else:
            print(f"> Action: EDIT (Local source found)")
    else:
        print(f"> Action: CREATE")

    instruction_msg = get_dynamic_prompt(
        todo_file_name, desc, action_tag, original_code, 
        core_map_content, paper_content, context_lst, config_yaml
    )
    trajectories.extend(instruction_msg)
        
    completion = api_call(trajectories)
    
    # response
    completion_json = json.loads(completion.model_dump_json())
    responses.append(completion_json)
    
    # trajectories
    message = completion.choices[0].message
    trajectories.append({'role': message.role, 'content': message.content})

    # print and logging
    print_response(completion_json)
    temp_total_accumulated_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
    total_accumulated_cost = temp_total_accumulated_cost

    # save
    analysis_file_path = f'{artifact_output_dir}/{todo_file_name}_simple_analysis.txt'
    os.makedirs(os.path.dirname(analysis_file_path), exist_ok=True)
    with open(analysis_file_path, 'w') as f:
        f.write(completion_json['choices'][0]['message']['content'])


    done_file_lst.append(todo_file_name)

    # save for next stage(coding)
    todo_file_name = todo_file_name.replace("/", "_") 
    with open(f'{output_dir}/{todo_file_name}_simple_analysis_response.json', 'w') as f:
        json.dump(responses, f)

    with open(f'{output_dir}/{todo_file_name}_simple_analysis_trajectories.json', 'w') as f:
        json.dump(trajectories, f)

save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)
