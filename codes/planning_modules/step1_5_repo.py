import json
import os
import ast
import re
import subprocess
import shutil
from openai import OpenAI
from .utils import print_response, print_log_cost

def clone_repo(url, dest):
    print(f"Cloning {url} to {dest}...")
    try:
        # Use GIT_TERMINAL_PROMPT=0 to prevent hanging on password prompts if the repo doesn't exist (returns 404)
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        
        subprocess.run(
            ["git", "clone", "--depth", "1", url, dest], 
            stdout=subprocess.DEVNULL, 
            env=env,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to clone repository (it might be private or not exist).")
    except Exception as e:
        print(f"[ERROR] Exception during clone: {str(e)}")

def clean_up(dest):
    if os.path.exists(dest):
        # Handle read-only files on Windows for shutil.rmtree
        def onerror(func, path, exc_info):
            import stat
            if not os.access(path, os.W_OK):
                os.chmod(path, stat.S_IWUSR)
                func(path)
            else:
                pass
        shutil.rmtree(dest, onerror=onerror)

def parse_ast_for_file(filepath, base_path):
    rel_path = os.path.relpath(filepath, base_path)
    output = f"📂 {rel_path}\n"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
    except Exception as e:
        return output + f"# Could not read file: {str(e)}\n\n"
        
    try:
        tree = ast.parse(source)
    except Exception as e:
        return output + f"# Syntax error or parsing issue: {str(e)}\n\n"

    def format_args(args_node):
        args_list = []
        for arg in args_node.args:
            arg_str = arg.arg
            if hasattr(arg, 'annotation') and arg.annotation is not None:
                try:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                except:
                    pass
            args_list.append(arg_str)
        if args_node.vararg:
            args_list.append(f"*{args_node.vararg.arg}")
        if args_node.kwarg:
            args_list.append(f"**{args_node.kwarg.arg}")
        return ", ".join(args_list)

    def extract_func_details(func_node):
        details = []
        is_init = func_node.name == '__init__'
        
        # We relax the condition here: if it's 'forward', 'build', or has 'model', 'net' in name, we treat it as core logic
        is_forward_or_build = func_node.name == 'forward' or 'build' in func_node.name.lower() or 'model' in func_node.name.lower()

        def visit(node):
            if isinstance(node, ast.Assign):
                if is_init:
                    for target in node.targets:
                        if isinstance(target, ast.Attribute) and getattr(target.value, 'id', None) == 'self':
                            details.append(f"self.{target.attr} = ...")
                if is_forward_or_build:
                    if isinstance(node.value, ast.Call):
                        try:
                            tgt = ast.unparse(node.targets[0]).replace('\n', ' ')
                            func_str = ast.unparse(node.value.func).replace('\n', ' ')
                            details.append(f"{tgt} = {func_str}(...)")
                        except:
                            pass
            elif isinstance(node, ast.Return):
                try:
                    if node.value:
                        ret_val = ast.unparse(node.value).replace('\n', ' ')
                        if len(ret_val) > 60: ret_val = ret_val[:57] + "..."
                        details.append(f"return {ret_val}")
                    else:
                        details.append("return")
                except:
                    pass

            for child in ast.iter_child_nodes(node):
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    visit(child)
        
        for stmt in func_node.body:
            visit(stmt)
            
        seen = set()
        res = []
        for d in details:
            if d not in seen:
                seen.add(d)
                res.append(d)
        if not res:
            res.append("pass")
        return res

    has_content = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            has_content = True
            bases = [ast.unparse(b) for b in node.bases]
            bases_str = f"({', '.join(bases)})" if bases else ""
            output += f"class {node.name}{bases_str}:\n"
            doc = ast.get_docstring(node)
            if doc:
                output += f'    """{doc.strip().split(chr(10))[0]}..."""\n'
            
            methods_found = False
            for body_node in node.body:
                if isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods_found = True
                    args_str = format_args(body_node.args)
                    returns_str = f" -> {ast.unparse(body_node.returns)}" if hasattr(body_node, 'returns') and body_node.returns else ""
                    is_async = "async " if isinstance(body_node, ast.AsyncFunctionDef) else ""
                    output += f"    {is_async}def {body_node.name}({args_str}){returns_str}: \n"
                    method_doc = ast.get_docstring(body_node)
                    if method_doc:
                        output += f'        """{method_doc.strip().split(chr(10))[0]}..."""\n'
                        
                    func_details = extract_func_details(body_node)
                    for detail in func_details:
                        output += f"        {detail}\n"
            if not methods_found:
                output += "    pass\n"
            output += "\n"
            
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            has_content = True
            args_str = format_args(node.args)
            returns_str = f" -> {ast.unparse(node.returns)}" if hasattr(node, 'returns') and node.returns else ""
            is_async = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            output += f"{is_async}def {node.name}({args_str}){returns_str}:\n"
            doc = ast.get_docstring(node)
            if doc:
                output += f'    """{doc.strip().split(chr(10))[0]}..."""\n'
                
            func_details = extract_func_details(node)
            for detail in func_details:
                output += f"    {detail}\n"
            output += "\n"

    if not has_content:
        output += "# No classes or functions found.\n\n"
        
    return output

def extract_core_baseline(clone_dir):
    extracted_text = ""
    if not clone_dir or not os.path.exists(clone_dir):
        return extracted_text

    for root, dirs, files in os.walk(clone_dir):
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', 'tests']]
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                extracted_text += parse_ast_for_file(filepath, clone_dir)       

    return extracted_text

def extract_reference_tool(clone_dir, url):
    extracted_text = ""
    if not clone_dir or not os.path.exists(clone_dir):
        return extracted_text

    readme_path = None
    for file in os.listdir(clone_dir):
        if file.lower().startswith("readme"):
            readme_path = os.path.join(clone_dir, file)
            break

    if readme_path:
        with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:    
            content = f.read()

        pattern = re.compile(r'(?i)^(#{1,3}\s*(installation|usage|quickstart|example|getting started)[\s\S]*?)(?=^#{1,3}\s|\Z)', re.MULTILINE)
        matches = pattern.findall(content)
        for match in matches:
            extracted_text += match[0].strip() + "\n\n"

    return extracted_text

def extract_repo_preview(url, dest_dir):
    repo_name = url.rstrip('/').split('/')[-1]
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]
    
    clone_dir = os.path.join(dest_dir, "cloned_repos", f"temp_{repo_name}")
    clean_up(clone_dir)
    os.makedirs(os.path.dirname(clone_dir), exist_ok=True)
    clone_repo(url, clone_dir)
    
    if not os.path.exists(clone_dir):
        return f"# Failed to clone {url}\n\n", None
        
    readme_content = ""
    for file in os.listdir(clone_dir):
        if file.lower().startswith("readme"):
            readme_path = os.path.join(clone_dir, file)
            with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                content_lines = f.readlines()
                readme_content = "".join(content_lines[:100])
            break
            
    top_level_items = []
    for item in os.listdir(clone_dir):
        if item not in ['.git', '__pycache__']:
            top_level_items.append(item)
            
    preview = f"[Repository: {url}]\n"
    preview += f"Top-level structure: {', '.join(top_level_items)}\n"
    preview += f"README snippet:\n{readme_content}\n"
    preview += "="*40 + "\n"
    
    return preview, clone_dir

def execute_repo_triage_stage(client, trajectories, gpt_version, output_dir, total_accumulated_cost, repo_json_path, paper_content):
    if not repo_json_path or not os.path.exists(repo_json_path):
        return trajectories, total_accumulated_cost, None, None

    current_stage = "[Planning] Repo Triage & Pruning"
    print(current_stage)

    with open(repo_json_path, 'r', encoding='utf-8') as f:
        repo_data = json.load(f)

    # 1. Clone & Generate Previews
    repo_previews = ""
    temp_dirs = {}
    for repo_item in repo_data:
        url = repo_item.get("target_url")
        if not url: continue
        preview, clone_dir = extract_repo_preview(url, output_dir)
        repo_previews += preview
        if clone_dir:
            temp_dirs[url] = clone_dir

    # 2. Prepare Paper Context
    paper_intro = ""
    if isinstance(paper_content, dict):
        paper_intro += "Title: " + str(paper_content.get('title', '')) + "\n"
        paper_intro += "Abstract: " + str(paper_content.get('abstract', '')) + "\n"
        if 'pdf_parse' in paper_content:
            intro_text = ""
            for section in paper_content['pdf_parse'].get('body_text', []):
                if 'intro' in str(section.get('section', '')).lower() or 'method' in str(section.get('section', '')).lower():
                    intro_text += section.get('text', '') + "\n"
            paper_intro += "Introduction: " + intro_text[:2000]
    else:
        paper_intro = str(paper_content)[:3000]

    prompt_system = """You are a senior AI architect. Your task is to evaluate a provided set of GitHub repositories and determine their classification in the "reproduce target paper" task.

[Classification Criteria]
1. `INFRA_LIB` (Infrastructure Library):
   - Description: Extremely large generic underlying frameworks (e.g., PyTorch, TensorFlow, HuggingFace transformers, mmdetection).
   - Action: The system should ignore them as large models already have built-in knowledge.
2. `CORE_BASELINE` (Core Baseline Library):
   - Description: The target paper explicitly builds upon this codebase for secondary development (e.g., modifying Loss, adding new modules), or it is the official code of a heavily relied-upon preceding work.
   - Action: The system must shallow clone it locally and extract the code directory tree and function skeletons.
3. `REFERENCE_TOOL` (Reference Tool Library):
   - Description: Used only for specific small functions (e.g., specific evaluation scripts, dataset processing tools, specific weight compression algorithms). No need to modify its source code; we just need to know how to install and call it.
   - Action: The system only needs to extract the Usage/Installation sections from its README.

[Output Requirements]
Please output a strict JSON object containing a "repos" key with an array of repository classifications.
Format requirement:
{
  "repos": [
    {
      "target_url": "...",
      "category": "INFRA_LIB or CORE_BASELINE or REFERENCE_TOOL",
      "reason": "..."
    }
  ]
}
"""

    prompt_user = f"[Paper Introduction/Abstract]\n{paper_intro}\n\n[Repository Previews]\n{repo_previews}"

    triage_msg = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]

    if "o3-mini" in gpt_version:
        completion = client.chat.completions.create(
            model=gpt_version,
            reasoning_effort="high",
            messages=triage_msg,
            response_format={"type": "json_object"}
        )
    else:
        completion = client.chat.completions.create(
            model=gpt_version,
            messages=triage_msg,
            response_format={"type": "json_object"} if "gpt" in gpt_version else None
        )

    completion_json = json.loads(completion.model_dump_json())
    print_response(completion_json)
    new_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)

    content_str = completion.choices[0].message.content

    json_match = re.search(r'```json\n(.*?)```', content_str, re.DOTALL)      
    if json_match:
        content_str = json_match.group(1).strip()

    try:
        triage_results = json.loads(content_str)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse triage results as JSON. \n{content_str}")
        triage_results = {"repos": []}
        
    # === NEW: OUTPUT REPO ANALYSIS ===
    artifacts_dir = os.path.join(output_dir, "planning_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    analysis_log_path = os.path.join(artifacts_dir, "1.0_repo_analysis.txt")
    with open(analysis_log_path, 'w', encoding='utf-8') as f:
        f.write("=== System Prompt ===\n")
        f.write(prompt_system + "\n\n")
        f.write("=== User Prompt (Paper + Repos) ===\n")
        f.write(prompt_user + "\n\n")
        f.write("=== LLM Response ===\n")
        f.write(content_str + "\n")

    core_map_output = ""
    ref_snippets_output = ""

    for repo_info in triage_results.get("repos", []):
        url = repo_info.get("target_url")
        category = repo_info.get("category")
        
        orig_clone_dir = temp_dirs.get(url)
        if not orig_clone_dir or not os.path.exists(orig_clone_dir):
            continue

        repo_name = url.rstrip('/').split('/')[-1]
        parent_dir = os.path.dirname(orig_clone_dir)

        if category == "CORE_BASELINE":
            new_clone_dir = os.path.join(parent_dir, f"core_{repo_name}")
            clean_up(new_clone_dir)
            os.rename(orig_clone_dir, new_clone_dir)
            
            core_map_output += f"[Repository Map: {url}]\n"
            core_map_output += extract_core_baseline(new_clone_dir)
            core_map_output += "\n" + "="*40 + "\n\n"
            
        elif category == "REFERENCE_TOOL":
            new_clone_dir = os.path.join(parent_dir, f"ref_{repo_name}")
            clean_up(new_clone_dir)
            os.rename(orig_clone_dir, new_clone_dir)
            
            ref_snippets_output += f"[Reference Snippets from: {url}]\n"        
            ref_snippets_output += extract_reference_tool(new_clone_dir, url)      
            ref_snippets_output += "\n" + "="*40 + "\n\n"
            
        else:
            clean_up(orig_clone_dir)

    core_map_path = os.path.join(artifacts_dir, "1.0_core_repo_map.txt")
    ref_snippets_path = os.path.join(artifacts_dir, "1.0_reference_snippets.txt")      
    triage_result_path = os.path.join(artifacts_dir, "repo_triage_result.json")    

    if core_map_output:
        with open(core_map_path, "w", encoding="utf-8") as f:
            f.write(core_map_output)
    if ref_snippets_output:
        with open(ref_snippets_path, "w", encoding="utf-8") as f:
            f.write(ref_snippets_output)

    with open(triage_result_path, "w", encoding="utf-8") as f:
        json.dump(triage_results, f, indent=2)

    return trajectories, new_cost, core_map_path if core_map_output else None, ref_snippets_path if ref_snippets_output else None
