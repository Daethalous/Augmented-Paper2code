from openai import OpenAI
import json
import argparse
import os
import sys

# Import functions from the newly created modules
# Ensure the planning_modules package is accessible. 
# Since this script is in `codes/`, and `planning_modules` is in `codes/`, 
# we can import directly.
from planning_modules import (
    execute_plan_stage,
    execute_repo_triage_stage,
    execute_architecture_stage,
    execute_logic_stage,
    execute_config_stage
)
from planning_modules.utils import load_accumulated_cost, save_accumulated_cost

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--paper_name', type=str)
parser.add_argument('--gpt_version', type=str)
parser.add_argument('--paper_format', type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str) # json format
parser.add_argument('--pdf_latex_path', type=str) # latex format
parser.add_argument('--repo_json_path', type=str, default="") # new: github repos info
parser.add_argument('--output_dir', type=str, default="")

args = parser.parse_args()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

paper_name = args.paper_name
gpt_version = args.gpt_version
paper_format = args.paper_format
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path
repo_json_path = args.repo_json_path
output_dir = args.output_dir

# Load paper content
if paper_format == "JSON":
    with open(f'{pdf_json_path}') as f:
        paper_content = json.load(f)
elif paper_format == "LaTeX":
    try:
        with open(f'{pdf_latex_path}', 'r', encoding='utf-8') as f:
            paper_content = f.read()
    except UnicodeDecodeError:
        with open(f'{pdf_latex_path}', 'r', encoding='latin1') as f: # Fallback encoding if utf-8 fails
            paper_content = f.read()

else:
    print(f"[ERROR] Invalid paper format. Please select either 'JSON' or 'LaTeX.")
    sys.exit(0)

responses = []
total_accumulated_cost = 0

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------------
# Step 0.5: Repo Triage (Optional)
# ------------------------------------------------------------------
if repo_json_path:
    # 独立调用，不包含全局trajectories
    _, total_accumulated_cost, core_map_path, ref_snippets_path = execute_repo_triage_stage(
        client,
        [],
        gpt_version,
        output_dir,
        total_accumulated_cost,
        repo_json_path,
        paper_content
    )
else:
    core_map_path, ref_snippets_path = None, None

# ------------------------------------------------------------------
# Step 1: Overall Plan (Paper Only - Isolation)
# ------------------------------------------------------------------
completion_json, _, total_accumulated_cost = execute_plan_stage(
    client,
    [],
    gpt_version,
    paper_content,
    output_dir,
    total_accumulated_cost,
    None, # Do not pass AST map here
    None  # Do not pass ref snippets here
)
responses.append(completion_json)
plan_text = completion_json['choices'][0]['message']['content']

# ------------------------------------------------------------------
# Step 2: Architecture Design
# ------------------------------------------------------------------
# Construct minimal context for Step 2
minimal_traj_for_step2 = [
    {"role": "user", "content": f"## Paper\n{paper_content}"},
    {"role": "assistant", "content": plan_text}
]
completion_json, _, total_accumulated_cost = execute_architecture_stage(
    client,
    minimal_traj_for_step2,
    gpt_version,
    output_dir,
    total_accumulated_cost,
    core_map_path,
    ref_snippets_path
)
responses.append(completion_json)
arch_text = completion_json['choices'][0]['message']['content']

# ------------------------------------------------------------------
# Step 3: Logic Design
# ------------------------------------------------------------------
# Construct minimal context for Step 3
minimal_traj_for_step3 = [
    {"role": "user", "content": f"## Paper\n{paper_content}"},
    {"role": "assistant", "content": plan_text},
    {"role": "user", "content": "Please output the architecture design based on the above."},
    {"role": "assistant", "content": arch_text}
]
completion_json, _, total_accumulated_cost = execute_logic_stage(
    client,
    minimal_traj_for_step3,
    gpt_version,
    output_dir,
    total_accumulated_cost,
    core_map_path,
    ref_snippets_path
)
responses.append(completion_json)
logic_text = completion_json['choices'][0]['message']['content']

# ------------------------------------------------------------------
# Step 4: Configuration File Generation
# ------------------------------------------------------------------
# Construct minimal context for Step 4
minimal_traj_for_step4 = [
    {"role": "user", "content": f"## Paper\n{paper_content}"},
    {"role": "assistant", "content": arch_text}
]
completion_json, _, total_accumulated_cost = execute_config_stage(
    client,
    minimal_traj_for_step4,
    gpt_version,
    output_dir,
    total_accumulated_cost
)
responses.append(completion_json)
config_text = completion_json['choices'][0]['message']['content']

# ------------------------------------------------------------------
# Final Save
# ------------------------------------------------------------------
save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)

with open(f'{output_dir}/planning_response.json', 'w') as f:
    json.dump(responses, f)

# We reconstruct the dummy trajectories array exactly as legacy 2_analyzing and 3_coding expects:
# A list of objects where role="assistant" gives us the 4 texts in index 0,1,2,3
legacy_trajectories = [
    {"role": "assistant", "content": plan_text},
    {"role": "assistant", "content": arch_text},
    {"role": "assistant", "content": logic_text},
    {"role": "assistant", "content": config_text}
]

with open(f'{output_dir}/planning_trajectories.json', 'w') as f:
    json.dump(legacy_trajectories, f)
print(f"Planning completed. Results saved to {output_dir}")
