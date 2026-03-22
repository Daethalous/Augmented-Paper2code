from openai import OpenAI
import json
import os
from .utils import print_response, print_log_cost

def execute_plan_stage(client, trajectories, gpt_version, paper_content, output_dir, total_accumulated_cost, core_map_path=None, ref_snippets_path=None):
    current_stage = "[Planning] Overall plan"
    print(current_stage)

    plan_msg = [
        {'role': "system", "content": f"""You are an expert researcher and strategic planner with a deep understanding of experimental design and reproducibility in scientific research.
Your task is to create a detailed and highly abstract plan to reproduce the experiments and methodologies described in the paper, WITHOUT considering any specific existing codebase.
This plan should align precisely with the paper's methodology, experimental setup, and evaluation metrics.

Instructions:
1. Align with the Paper: Your plan must strictly follow the methods, datasets, model configurations, hyperparameters, and experimental setups described in the paper.
2. Abstract Component Extraction: Determine the "Must-Have Logic Components" (e.g., Core Data Pipeline, Specific Loss Function, Neural Network Sub-modules) necessary to build the system from scratch.
3. Identify Shared/Common Modules: If the paper describes multiple baselines or experiments that share components, explicitly group them into "Shared_Components" or "Common Modules". Do not attach them to specific hypothetical files yet.
4. Focus on Logic and Data Flow: Define the boundary of each component's inputs and outputs."""},
        {"role": "user",
         "content" : f"""## Paper
{paper_content}

## Task
1. We want to reproduce the method described in the attached paper.
2. Based entirely on the theoretical paper content, outline a comprehensive plan covering:
   - Key details from the paper's **Methodology** and **Experiments**.
   - A list of **Must-Have Logic Components** (e.g., specific dataset handlers, novel neural network blocks, training loops, evaluation metrics).
3. Do NOT generate a specific file tree or file paths. Instead, output the required abstract software requirements and dataflow.
4. If multiple experimental setups share the same logical block, clearly identify it as a "Shared Component" to avoid redundant implementations.
"""}]

    # Update trajectories with new user/system messages
    # Note: trajectories should only contain what happened before. 
    # But here we are appending the *new* turn. 
    # To keep context, we usually append to existing trajectories.
    # However, for the first step, trajectories might be empty.
    
    # In original script, `trajectories` accumulates everything.
    # We will append the current message to trajectories to form the input prompt.
    
    # Create a local copy to send to API
    messages_to_send = list(trajectories) + plan_msg
    
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
    
    # Update trajectories with the message sequence used + response
    # The original script extends trajectories with the instruction_msg (plan_msg here)
    # AND the response.
    
    updated_trajectories = list(trajectories) + plan_msg
    response_message = completion.choices[0].message
    updated_trajectories.append({'role': response_message.role, 'content': response_message.content})
    
    return completion_json, updated_trajectories, new_cost
