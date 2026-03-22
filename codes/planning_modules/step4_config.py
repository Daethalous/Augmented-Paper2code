from openai import OpenAI
import json
import os
from .utils import print_response, print_log_cost

def execute_config_stage(client, trajectories, gpt_version, output_dir, total_accumulated_cost):
    current_stage = "[Planning] Configuration file generation"
    print(current_stage)

    config_msg = [
        {'role': 'user', 'content': """You write elegant, modular, and maintainable code. Adhere to Google-style guidelines.

Based on the paper, plan, design specified previously, follow the "Format Example" and generate the code. 
Extract the training details from the above paper (e.g., learning rate, batch size, epochs, etc.), follow the "Format example" and generate the code. 
DO NOT FABRICATE DETAILS — only use what the paper provides.

You must write `config.yaml`.

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

    messages_to_send = list(trajectories) + config_msg
    
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
    
    updated_trajectories = list(trajectories) + config_msg
    response_message = completion.choices[0].message
    updated_trajectories.append({'role': response_message.role, 'content': response_message.content})
    
    return completion_json, updated_trajectories, new_cost
