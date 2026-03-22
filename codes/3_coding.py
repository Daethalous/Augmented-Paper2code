from openai import OpenAI
import json
import os
from tqdm import tqdm
import re
import sys
import copy
from utils import extract_planning, content_to_json, extract_code_from_content, print_response, print_log_cost, load_accumulated_cost, save_accumulated_cost
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name',type=str)
parser.add_argument('--gpt_version',type=str, default="o3-mini")
parser.add_argument('--paper_format',type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str) # json format
parser.add_argument('--pdf_latex_path', type=str) # latex format
parser.add_argument('--output_dir',type=str, default="")
parser.add_argument('--output_repo_dir',type=str, default="")

args    = parser.parse_args()
client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

paper_name = args.paper_name
gpt_version = args.gpt_version
paper_format = args.paper_format
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path
output_dir = args.output_dir
output_repo_dir = args.output_repo_dir

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
# file_list = content_to_json(context_lst[1])
task_list = content_to_json(context_lst[2])

if 'Task list' in task_list:
    todo_file_lst = task_list['Task list']
elif 'task_list' in task_list:
    todo_file_lst = task_list['task_list']
elif 'task list' in task_list:
    todo_file_lst = task_list['task list']
elif 'Logic Analysis' in task_list:
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
    
logic_analysis_dict = {}
for desc in task_list['Logic Analysis']:
    logic_analysis_dict[desc[0]] = desc[1]

done_file_lst = ['config.yaml']
done_file_dict = {}

def read_original_source_code(file_path, base_output_dir):
    cloned_repos_dir = os.path.join(base_output_dir, "cloned_repos")
    if not os.path.exists(cloned_repos_dir):
        return None
    for d in os.listdir(cloned_repos_dir):
        if d.startswith("core_"):
            target_path = os.path.join(cloned_repos_dir, d, file_path)
            if os.path.exists(target_path):
                with open(target_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            # If not found directly, try stripping leading modifiers or find recursively
            basename = os.path.basename(file_path)
            for root, dirs, files in os.walk(os.path.join(cloned_repos_dir, d)):
                if basename in files:
                    with open(os.path.join(root, basename), 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()
    return None

code_msg = [
    {"role": "system", "content": f"""You are an expert researcher and software engineer with a deep understanding of experimental design and reproducibility in scientific research.
You will receive a research paper in {paper_format} format, an overview of the plan, a Design in JSON format consisting of "Implementation approach", "File list", "Data structures and interfaces", and "Program call flow", followed by a Task in JSON format that includes "Required packages", "Required other language third-party packages", "Logic Analysis", and "Task list", along with a configuration file named "config.yaml". 
Your task is to write code to reproduce the experiments and methodologies described in the paper. 

The code you write must be elegant, modular, and maintainable, adhering to Google-style guidelines. 
The code must strictly align with the paper's methodology, experimental setup, and evaluation metrics. 
Write code with triple quoto."""}]

def get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst, action_tag="CREATE", original_code=None): 
    code_files = ""
    for done_file in done_file_lst:
        if done_file.endswith(".yaml"): continue
        
        file_content = done_file_dict[done_file]
        # Option B: Read-Only Fence instead of skeleton. We supply the full context 
        # so LLM sees exact tensor shapes and attributes, but we strictly forbid modification.
        if action_tags_dict.get(done_file, "") == "REUSE":
            code_files += f"""
```python
# [FILE: {done_file}] (REUSED MODULE - READ ONLY)
# Context only: Understand API APIs, data structures, and tensor shapes.
{file_content}
```

"""
        else:
            code_files += f"""
```python
# [FILE: {done_file}]
{file_content}
```

"""
    
    reference_code_prompt = ""
    if action_tag == "EDIT" and original_code:
        reference_code_prompt = f"""
## Original Source Code for Reference
You are tasked with EDITING this file. Please refer to this original code carefully to ensure compatibility and correctness in your rewritten version.
```python
{original_code}
```
-----
"""

    write_msg=[
{'role': 'user', "content": f"""# Context
## Paper
{paper_content}

-----

## Overview of the plan
{context_lst[0]}

-----

## Design
{context_lst[1]}

-----

## Task
{context_lst[2]}

-----

## Configuration file
```yaml
{config_yaml}
```
-----

## Code Files
{code_files}

-----
{reference_code_prompt}

# Format example
## Code: {todo_file_name}
```python
## {todo_file_name}
...
```

-----

# Instruction
Based on the paper, plan, design, task and configuration file(config.yaml) specified previously, follow "Format example", write the code. 

We have {done_file_lst}.
Next, you must write only the "{todo_file_name}".
1. Only One file: do your best to implement THIS ONLY ONE FILE.
2. COMPLETE CODE: Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.
3. Set default value: If there is any setting, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE. AVOID circular import.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.
5. CAREFULLY CHECK THAT YOU DONT MISS ANY NECESSARY CLASS/FUNCTION IN THIS FILE.
6. Before using a external variable/module, make sure you import it first.
7. Write out EVERY CODE DETAIL, DON'T LEAVE TODO. IF YOU ARE EDITING A FILE, YOU MUST OUTPUT THE COMPLETE FILE! DO NOT LEAVE SNIPPETS LIKE '# ...existing code...', WRITE EVERY SINGLE CHARACTER SO THE FILE CAN COMPILE.
8. REFER TO CONFIGURATION: you must use configuration from "config.yaml". DO NOT FABRICATE any configuration values.

{detailed_logic_analysis}

## Code: {todo_file_name}"""}]
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
    

# testing for checking
detailed_logic_analysis_dict = {}
retrieved_section_dict = {}

# Keep track of action tags like analyzing
action_tags_dict = {}

arch_design_context = context_lst[1] if len(context_lst) > 1 else ""

# Regex extract [REUSE] file paths from arch design to double guarantee they are in todo_file_lst
reuse_pattern = re.compile(r'\[REUSE\]\s+([\w\./-]+|\*\.\w+)')
found_reuse_files = reuse_pattern.findall(arch_design_context)
for rf in found_reuse_files:
    if rf not in todo_file_lst:
        todo_file_lst.append(rf)

for todo_file_name in todo_file_lst:
    save_todo_file_name = todo_file_name.replace("/", "_")

    if todo_file_name == "config.yaml":
        continue

    desc = logic_analysis_dict.get(todo_file_name, "")
    
    # Parse Action Tags
    if "[REUSE]" in desc or f"[REUSE] {todo_file_name}" in arch_design_context:
        action_tags_dict[todo_file_name] = "REUSE"
    elif "[EDIT]" in desc or f"[EDIT] {todo_file_name}" in arch_design_context:
        action_tags_dict[todo_file_name] = "EDIT"
    else:
        action_tags_dict[todo_file_name] = "CREATE"
    
    if action_tags_dict[todo_file_name] != "REUSE":
        try:
            with open(f"{output_dir}/{save_todo_file_name}_simple_analysis_response.json") as f:
                detailed_logic_analysis_response = json.load(f)
            detailed_logic_analysis_dict[todo_file_name] = detailed_logic_analysis_response[0]['choices'][0]['message']['content']
        except FileNotFoundError:
            # Fallback to the physical txt artifact if JSON is missing
            txt_path = f"{output_dir}/analyzing_artifacts/{save_todo_file_name}_simple_analysis.txt"
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    detailed_logic_analysis_dict[todo_file_name] = f.read()
            else:
                detailed_logic_analysis_dict[todo_file_name] = ""


artifact_output_dir=f'{output_dir}/coding_artifacts'
os.makedirs(artifact_output_dir, exist_ok=True)

total_accumulated_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")
for todo_idx, todo_file_name in enumerate(tqdm(todo_file_lst)):
    responses = []
    trajectories = copy.deepcopy(code_msg)

    current_stage = f"[CODING] {todo_file_name}"
    print(f"\n{current_stage}")

    if todo_file_name == "config.yaml":
        continue
        
    action_tag = action_tags_dict.get(todo_file_name, "CREATE")

    if action_tag == "REUSE":
        print(f"> Action: REUSE (Copying directly from clone)")
        original_code = read_original_source_code(todo_file_name, output_dir)
        if original_code is None:
            print(f"> [WARNING] Could not find local source for REUSE: {todo_file_name}. Ignoring.")
            continue
            
        code = original_code
        done_file_lst.append(todo_file_name)
        done_file_dict[todo_file_name] = code
        todo_file_dir = '/'.join(todo_file_name.split("/")[:-1])
        if todo_file_dir:
            os.makedirs(f"{output_repo_dir}/{todo_file_dir}", exist_ok=True)
        with open(f"{output_repo_dir}/{todo_file_name}", 'w', encoding='utf-8') as f:
            f.write(code)
        continue
        
    original_code = None
    if action_tag == "EDIT":
        original_code = read_original_source_code(todo_file_name, output_dir)
        if not original_code:
            print(f"> [WARNING] Could not find local source for {todo_file_name}. Falling back to CREATE.")
            action_tag = "CREATE"
        else:
            print(f"> Action: EDIT (Local source found, inject info in Prompts)")
    else:
        print(f"> Action: CREATE")

    instruction_msg = get_write_msg(todo_file_name, detailed_logic_analysis_dict[todo_file_name], done_file_lst, action_tag, original_code)
    trajectories.extend(instruction_msg)

    completion = api_call(trajectories)
    # print(completion.choices[0].message)
    
    # response
    completion_json = json.loads(completion.model_dump_json())
    responses.append(completion_json)

    # trajectories
    message = completion.choices[0].message
    trajectories.append({'role': message.role, 'content': message.content})

    done_file_lst.append(todo_file_name)

    # save
    # save_dir_name = f"{paper_name}_repo"
    os.makedirs(f'{output_repo_dir}', exist_ok=True)
    save_todo_file_name = todo_file_name.replace("/", "_")


    # print and logging
    print_response(completion_json)
    temp_total_accumulated_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
    total_accumulated_cost = temp_total_accumulated_cost

    # save artifacts
    with open(f'{artifact_output_dir}/{save_todo_file_name}_coding.txt', 'w', encoding='utf-8') as f:
        f.write(completion_json['choices'][0]['message']['content'])


    # extract code save 
    code = extract_code_from_content(message.content)
    if len(code) == 0:
        code = message.content 

    done_file_dict[todo_file_name] = code
    if save_todo_file_name != todo_file_name:
        todo_file_dir = '/'.join(todo_file_name.split("/")[:-1])
        os.makedirs(f"{output_repo_dir}/{todo_file_dir}", exist_ok=True)

    with open(f"{output_repo_dir}/{todo_file_name}", 'w', encoding='utf-8') as f:
        f.write(code)

save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)
