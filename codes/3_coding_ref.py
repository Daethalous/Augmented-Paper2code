from openai import OpenAI
import json
import os
from tqdm import tqdm
import re
import sys
import copy
from utils import extract_planning, content_to_json, extract_code_from_content, print_response, print_log_cost, load_accumulated_cost, save_accumulated_cost, read_python_files, format_json_data
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

todo_file_lst = task_list['Task list']
done_file_lst = ['config.yaml']
done_file_dict = {}

code_msg = [
    {"role": "system", "content": f"""You are an expert researcher and software engineer with a deep understanding of experimental design and reproducibility in scientific research.
You will receive a research paper in {paper_format} format, an overview of the plan, a Design in JSON format consisting of "Implementation approach", "File list", "Data structures and interfaces", and "Program call flow", followed by a Task in JSON format that includes "Required packages", "Required other language third-party packages", "Logic Analysis", and "Task list", along with a configuration file named "config.yaml". 
Your task is to write code to reproduce the experiments and methodologies described in the paper. 

The code you write must be elegant, modular, and maintainable, adhering to Google-style guidelines. 
The code must strictly align with the paper's methodology, experimental setup, and evaluation metrics. 
Write code with triple quoto."""}]

baseline_content_dict = {}
if os.path.exists(f'{output_repo_dir}'):
    baseline_content_dict = read_python_files(output_repo_dir)

def get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst, new_definitions=[]): 
    code_files = ""
    for done_file in done_file_lst:
        if done_file.endswith(".yaml"): continue
        code_files += f"""
```python
{done_file_dict[done_file]}
```

"""

    baseline_msg = ""
    if todo_file_name in baseline_content_dict:
        baseline_msg = f"""## Baseline Code ({todo_file_name})
The file '{todo_file_name}' already exists in the repository. You must modify/refine it based on the logic analysis.

```python
{baseline_content_dict[todo_file_name]}
```
        """
        instruction_desc = f"""We are refining an existing file '{todo_file_name}'.
1. Read the provided "Baseline Code" carefully.
2. Based on the logic analysis, REFINE the code to meet the requirements.
3. OUTPUT THE COMPLETE FILE CONTENT (including unmodified parts). Do not return diffs or partial updates."""
    
    else:
        existing_files_summary = ""
        # If the file doesn't exist, we might still want to show what files exist in the baseline
        # to help with imports if they are not in todo list
        if len(baseline_content_dict) > 0:
             existing_files = list(baseline_content_dict.keys())
             # Filter out files that are in todo list (they will be overwritten/created anyway)
             existing_files = [f for f in existing_files if f not in todo_file_lst]
             existing_files_summary = f"Existing files in baseline (not modified in this plan): {existing_files}"

        instruction_desc = f"""We are creating a new file '{todo_file_name}'.
1. Follow the logic analysis to implement this file.
2. COMPLETE CODE: Implement complete, reliable, reusable code snippets.
{existing_files_summary}"""

    # --- Integration Check Logic ---
    is_entry_point = any(name in todo_file_name.lower() for name in ["train", "main", "run", "eval", "test", "__init__", "factory", "builder", "config", "registry", "loss", "model"])
    if is_entry_point and new_definitions:
        instruction_desc += f"""
\n\n[CRITICAL INTEGRATION STEP]
You have implemented the following NEW components in this session: {new_definitions}.
The baseline code typically DOES NOT use them (it uses default/old implementations).
You MUST Modify the code logic (e.g. argument parsing, model instantiation, loss function selection, factory registration) to INTEGRATE these new components.
Do NOT output the gathered baseline code blindly. 
Example: 
- If this is a Factory/__init__: Add the new class to the export/builder map.
- If this is a Config parser: Ensure arguments allow selecting the new method.
- If this is Main/Train: Replace the hardcoded class instantiation with dynamic selection based on config.
"""
    # -------------------------------

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

## Codes of dependencies (Newly implemented/modified in this session)
{code_files}

{baseline_msg}

-----

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
{instruction_desc}
3. Set default value: If there is any setting, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE. AVOID circular import.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.
5. CAREFULLY CHECK THAT YOU DONT MISS ANY NECESSARY CLASS/FUNCTION IN THIS FILE.
6. Before using a external variable/module, make sure you import it first.
7. Write out EVERY CODE DETAIL, DON'T LEAVE TODO.
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
for todo_file_name in todo_file_lst:
    # simple analysis
    
    # Cleaning the filename if it contains descriptions (e.g., "1. utils/foo.py -- description")
    # Same logic as in 2_analyzing_ref.py
    clean_todo_file_name = todo_file_name
    if " " in todo_file_name:
        parts = todo_file_name.split(" ")
        for part in parts:
            if ("/" in part or part.endswith(".py") or part.endswith(".txt") or part.endswith(".sh")) and not part.endswith("."):
                clean_todo_file_name = part
                break
    
    # Handle legacy saving format (flat directory structure with _ replacement)
    save_todo_file_name_flat = clean_todo_file_name.replace("/", "_")
    
    # Try different paths
    possible_paths = [
        f"{output_dir}/{save_todo_file_name_flat}_simple_analysis_response.json", # Legacy flat
        f"{output_dir}/analyzing_artifacts/{clean_todo_file_name}_simple_analysis_response.json" # Nested (future proof)
    ]

    if todo_file_name == "config.yaml":
        continue
    
    json_path = f"{output_dir}/{save_todo_file_name_flat}_simple_analysis_response.json"
    
    if os.path.exists(json_path):
        with open(json_path) as f:
            detailed_logic_analysis_response = json.load(f)
        detailed_logic_analysis_dict[todo_file_name] = detailed_logic_analysis_response[0]['choices'][0]['message']['content']
    else:
        print(f"[WARNING] Analysis file not found: {json_path}")
        detailed_logic_analysis_dict[todo_file_name] = ""


artifact_output_dir=f'{output_dir}/coding_artifacts'
os.makedirs(artifact_output_dir, exist_ok=True)

total_accumulated_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")
newly_defined_symbols = []
for todo_idx, todo_file_name in enumerate(tqdm(todo_file_lst)):
    # Clean file name for prompt usage (same cleaning logic)
    clean_todo_file_name = todo_file_name
    if " " in todo_file_name:
        parts = todo_file_name.split(" ")
        for part in parts:
            if ("/" in part or part.endswith(".py") or part.endswith(".txt") or part.endswith(".sh")) and not part.endswith("."):
                clean_todo_file_name = part
                break
                
    responses = []
    trajectories = copy.deepcopy(code_msg)

    current_stage = f"[CODING] {clean_todo_file_name}"
    print(current_stage)

    if clean_todo_file_name == "config.yaml":
        continue

    instruction_msg = get_write_msg(clean_todo_file_name, detailed_logic_analysis_dict[todo_file_name], done_file_lst, newly_defined_symbols)
    trajectories.extend(instruction_msg)

    completion = api_call(trajectories)
    # print(completion.choices[0].message)
    
    # response
    completion_json = json.loads(completion.model_dump_json())
    responses.append(completion_json)

    # trajectories
    message = completion.choices[0].message
    trajectories.append({'role': message.role, 'content': message.content})

    done_file_lst.append(clean_todo_file_name)

    # save
    # save_dir_name = f"{paper_name}_repo"
    os.makedirs(f'{output_repo_dir}', exist_ok=True)
    save_todo_file_name = clean_todo_file_name.replace("/", "_")


    # print and logging
    print_response(completion_json)
    temp_total_accumulated_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
    total_accumulated_cost = temp_total_accumulated_cost

    # save artifacts
    with open(f'{artifact_output_dir}/{save_todo_file_name}_coding.txt', 'w') as f:
        f.write(completion_json['choices'][0]['message']['content'])


    # extract code save 
    code = extract_code_from_content(message.content)
    if len(code) == 0:
        code = message.content 

    try:
        # Simple extraction of new definitions for context
        defined_classes = re.findall(r'^class\s+(\w+)', code, re.MULTILINE)
        defined_funcs = re.findall(r'^def\s+(\w+)', code, re.MULTILINE)
        defined_funcs = [f for f in defined_funcs if not f.startswith("_")]
        if defined_classes or defined_funcs:
            newly_defined_symbols.append(f"File: {clean_todo_file_name} (Classes: {defined_classes}, Functions: {defined_funcs})")
    except:
        pass
    
    if len(clean_todo_file_name.strip()) == 0:
        print(f"[WARNING] Filename is empty. Skipping save.")
        continue

    done_file_dict[clean_todo_file_name] = code
    if save_todo_file_name != clean_todo_file_name:
        todo_file_dir = '/'.join(clean_todo_file_name.split("/")[:-1])
        if todo_file_dir: # avoid empty string if file is in root
            os.makedirs(f"{output_repo_dir}/{todo_file_dir}", exist_ok=True)

    file_path = f"{output_repo_dir}/{clean_todo_file_name}"
    if os.path.isdir(file_path):
         print(f"[ERROR] Target path is a directory: {file_path}. Skipping.")
    else: 
        with open(file_path, 'w') as f:
            f.write(code)

save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)
