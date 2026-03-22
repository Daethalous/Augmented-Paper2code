from openai import OpenAI
import json
from tqdm import tqdm
import argparse
import os
import sys
from utils import print_response, print_log_cost, load_accumulated_cost, save_accumulated_cost

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name',type=str)
parser.add_argument('--gpt_version',type=str)
parser.add_argument('--paper_format',type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str) # json format
parser.add_argument('--pdf_latex_path', type=str) # latex format
parser.add_argument('--output_dir',type=str, default="")
parser.add_argument("--output_repo_dir", type=str, default="./cloned_repo")

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

def get_repo_content(repo_dir):
    repo_content = ""
    for root, dirs, files in os.walk(repo_dir):
        for file in files:
            if file.endswith((".py", ".yaml", ".yml", ".json")) and not file.startswith(("test", "setup")):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_dir)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        repo_content += f"\n\n## File: {rel_path}\n```python\n{content}\n```"
                except Exception as e:
                    print(f"Skipping file {file_path}: {e}")
    return repo_content

repo_content = get_repo_content(args.output_repo_dir) if os.path.exists(args.output_repo_dir) else ""

plan_msg = [
        {'role': "system", "content": f"""You are an expert researcher and strategic planner with a deep understanding of experimental design and reproducibility in scientific research. 
You will receive a research paper in {paper_format} format. 
Your task is to create a detailed and efficient plan to reproduce the experiments and methodologies described in the paper.
This plan should align precisely with the paper's methodology, experimental setup, and evaluation metrics. 

Instructions:

1. Align with the Paper: Your plan must strictly follow the methods, datasets, model configurations, hyperparameters, and experimental setups described in the paper.
2. Be Clear and Structured: Present the plan in a well-organized and easy-to-follow format, breaking it down into actionable steps.
3. Prioritize Efficiency: Optimize the plan for clarity and practical implementation while ensuring fidelity to the original experiments."""},
        {"role": "user",
         "content" : f"""## Paper
{paper_content}

## Task
1. We want to reproduce the method described in the attached paper. 
2. The authors did not release any official code, so we have to plan our own implementation.
3. Before writing any Python code, please outline a comprehensive plan that covers:
   - Key details from the paper's **Methodology**.
   - Important aspects of **Experiments**, including dataset requirements, experimental settings, hyperparameters, or evaluation metrics.
4. The plan should be as **detailed and informative** as possible to help us write the final code later.

## Requirements
- You don't need to provide the actual code yet; focus on a **thorough, clear strategy**.
- If something is unclear from the paper, mention it explicitly.

## Instruction
The response should give us a strong roadmap, making it easier to write the code later."""}]

file_list_msg = [
        {"role": "user", "content": f"""Your goal is to create a concise, usable, and complete software system design for reproducing the paper's method based on an existing reference implementation.
             
We have a reference implementation in `{args.output_repo_dir}`.
The following is the content of the reference repository:
{repo_content}

Based on the plan for reproducing the paper’s main method and the provided reference code, please design modifications to the software system. 
- If a file exists in the reference repo and needs modification to align with the paper, list it in the "File list".
- If a file is new, list it in the "File list".
- If a file exists and does NOT need modification, list it in the "File list" AND ADD AN "<unchanged>" tag to it, also include it in the "Data structures and interfaces" and "Program call flow" diagrams if relevant.

-----

## Format Example
[CONTENT]
{{
    "Implementation approach": "We will base our implementation on the reference code. We will modify `model.py` to include the new attention mechanism proposed in the paper and update `trainer.py` to support the new loss function. `dataset_loader.py` will be reused as is.",
    "File list": [
        "model.py",  
        "trainer.py",
        "new_module.py"
    ],
    "Data structures and interfaces": "\\nclassDiagram\\n    class Main {{\\n        +__init__()\\n        +run_experiment()\\n    }}\\n    class DatasetLoader {{\\n        +__init__(config: dict)\\n        +load_data() -> Any\\n    }}\\n    class Model {{\\n        +__init__(params: dict)\\n        +forward(x: Tensor) -> Tensor\\n        +new_method()  # New method added\\n    }}\\n    class Trainer {{\\n        +__init__(model: Model, data: Any)\\n        +train() -> None\\n    }}\\n    Main --> DatasetLoader\\n    Main --> Trainer\\n    Trainer --> Model\\n",
    "Program call flow": "\\nsequenceDiagram\\n    participant M as Main\\n    participant DL as DatasetLoader\\n    participant MD as Model\\n    participant TR as Trainer\\n    M->>DL: load_data()\\n    DL-->>M: return dataset\\n    M->>MD: initialize model()\\n    M->>TR: train(model, dataset)\\n    TR->>MD: forward(x)\\n    MD-->>TR: predictions\\n    TR-->>M: training complete\\n",
    "Anything UNCLEAR": "Need clarification on..."
}}
[/CONTENT]

## Nodes: "<node>: <type>  # <instruction>"
- Implementation approach: <class 'str'>  # Summarize the chosen solution strategy, explicitly mentioning which files from the reference repo will be reused, modified, or created.
- File list: typing.List[str]  # Only need relative paths. List ONLY files that need to be created or modified.
- Data structures and interfaces: typing.Optional[str]  # Use mermaid classDiagram code syntax. Include BOTH modified/new classes AND critical existing classes. Mark new/modified methods or classes clearly.
- Program call flow: typing.Optional[str] # Use sequenceDiagram code syntax. Show the flow including interactions with reuse existing components.
- Anything UNCLEAR: <class 'str'>  # Mention ambiguities.

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT] like the format example, nothing else.

## Action
Follow the instructions for the nodes, generate the output, and ensure it follows the format example."""}
    ]

task_list_msg = [
        {'role': 'user', 'content': """Your goal is break down tasks according to PRD/technical design, generate a task list, and analyze task dependencies. 
You will break down tasks, analyze dependencies.
             
You outline a clear PRD/technical design for reproducing the paper’s method and experiments based on the reference code. 

Now, let's break down tasks according to PRD/technical design, generate a task list, and analyze task dependencies.
The Logic Analysis should not only consider the dependencies between files but also provide detailed descriptions to assist in writing the code needed to reproduce the paper.

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
            "data_preprocessing.py",
            "DataPreprocessing class ........"
        ],
        [
            "trainer.py",
            "Trainer ....... "
        ],
        [
            "existing_file.py",
            "No modification needed."
        ]
    ],
    "Task list": [
        "dataset_loader.py", 
        "model.py",  
        "trainer.py", 
        "evaluation.py",
        "main.py"  
    ],
    "Full API spec": "openapi: 3.0.0 ...",
    "Shared Knowledge": "Both data_preprocessing.py and trainer.py share ........",
    "Anything UNCLEAR": "Clarification needed on recommended hardware configuration for large-scale experiments."
}

[/CONTENT]

## Nodes: "<node>: <type>  # <instruction>"
- Required packages: typing.Optional[typing.List[str]]  # Provide required third-party packages in requirements.txt format.(e.g., 'numpy==1.21.0').
- Required Other language third-party packages: typing.List[str]  # List down packages required for non-Python languages. If none, specify "No third-party dependencies required".
- Logic Analysis: typing.List[typing.List[str]]  # Provide a list of files with the classes/methods/functions to be implemented, including dependency analysis and imports. Include as much detailed description as possible. IF A FILE DOES NOT NEED MODIFICATION, mark it as "No modification needed" or "Omitted".
- Task list: typing.List[str]  # Break down the tasks into a list of filenames, prioritized based on dependency order. The task list must include the previously generated file list (files that need modification/creation).
- Full API spec: <class 'str'>  # Describe all APIs using OpenAPI 3.0 spec that may be used by both frontend and backend. If front-end and back-end communication is not required, leave it blank.
- Shared Knowledge: <class 'str'>  # Detail any shared knowledge, like common utility functions or configuration variables.
- Anything UNCLEAR: <class 'str'>  # Mention any unresolved questions or clarifications needed from the paper or project scope.

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT] like the format example, nothing else.

## Action
Follow the node instructions above, generate your output accordingly, and ensure it follows the given format example."""}]

# config
config_msg = [
        {'role': 'user', 'content': """You write elegant, modular, and maintainable code. Adhere to Google-style guidelines.

Based on the paper, plan, design specified previously, follow the "Format Example" and generate the code. 
Extract the training details from the above paper (e.g., learning rate, batch size, epochs, etc.), follow the "Format example" and generate/modify the configuration file.
DO NOT FABRICATE DETAILS — only use what the paper provides.

You must write `config.yaml`.
If `config.yaml` already exists in the reference code, modify it to align with the paper's experiments. If not, create a new one.

ATTENTION: Use '##' to SPLIT SECTIONS, not '#'. Your output format must follow the example below exactly.

-----

# Format Example
## Code: config.yaml
```yaml
## config.yaml
training:
  learning_rate: ...
  batch_size: ...
  epochs: ...
...
```

-----

## Code: config.yaml
"""
    }]

def api_call(msg, gpt_version):
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

responses = []
trajectories = []
total_accumulated_cost = 0

for idx, instruction_msg in enumerate([plan_msg, file_list_msg, task_list_msg, config_msg]):
    current_stage = ""
    if idx == 0 :
        current_stage = f"[Planning] Overall plan"
    elif idx == 1:
        current_stage = f"[Planning] Architecture design"
    elif idx == 2:
        current_stage = f"[Planning] Logic design"
    elif idx == 3:
        current_stage = f"[Planning] Configuration file generation"
    print(current_stage)

    trajectories.extend(instruction_msg)

    completion = api_call(trajectories, gpt_version)
    
    # response
    completion_json = json.loads(completion.model_dump_json())

    # print and logging
    print_response(completion_json)
    temp_total_accumulated_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
    total_accumulated_cost = temp_total_accumulated_cost

    responses.append(completion_json)

    # trajectories
    message = completion.choices[0].message
    trajectories.append({'role': message.role, 'content': message.content})


# save
save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)

os.makedirs(output_dir, exist_ok=True)

with open(f'{output_dir}/planning_response.json', 'w') as f:
    json.dump(responses, f)

with open(f'{output_dir}/planning_trajectories.json', 'w') as f:
    json.dump(trajectories, f)
