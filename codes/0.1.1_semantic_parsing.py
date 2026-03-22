import json
import argparse
import os
import sys
from typing import Dict, Any, List, Optional
from openai import OpenAI

# Add current directory to path so we can import utils if running from elsewhere
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils import load_json_file, save_json_file, resolve_s2orc_citations
except ImportError:
    # If running from root without codes in path
    try:
        from codes.utils import load_json_file, save_json_file, resolve_s2orc_citations
    except ImportError:
        print("[Error] Could not import utils. Please ensure you are running from the project root or codes directory.")
        sys.exit(1)

# --- Constants & Prompts ---

# 把纯指令部分剥离出来作为 System Prompt
SYSTEM_PROMPT = """
You are an Expert AI Systems Architect. Your task is to analyze a highly condensed research paper text and extract its technical blueprint to help a downstream engineering system prepare a reproducible code baseline.

Your objective is NOT to understand every mathematical detail, but to identify the macro-level engineering requirements: What domain is this? What backbone models are used? Does the author explicitly mention building upon a specific open-source library or a prior paper's official codebase?

Analyze the text and output a strictly valid JSON object adhering to the following rules:

1. `task_domain`: Choose exactly one from [CV, NLP, Audio, TimeSeries, Graph, ReinforcementLearning, Other].
2. `core_task`: A concise 1-3 word description of the task (e.g., "Knowledge Distillation", "Image Classification").
3. `base_architecture`:
   - `has_clear_backbone`: true if a specific model family (e.g., ResNet, LLaMA, Transformer) is used as the foundation.
   - `backbone_names`: List the specific models mentioned. Leave empty if none.
4. `implementation_dependencies`: Look VERY carefully at the Experiments and Implementation sections.
   - `relies_on_existing_codebase`: true ONLY if they mention building the training pipeline on general libraries like 'transformers', 'torchdistill', 'DeepSpeed', etc.
   - `explicit_codebase_names`: Extract those framework/library names. 
   - `based_on_prior_paper_code`: true if they explicitly state they used a prior paper's codebase, OR if they mention "adopting the same training strategy/pipeline", "following the standard settings of", or "for fair comparison, we use" regarding a specific prior paper (e.g., CRD, ITRD).
   - `prior_paper_titles`: The exact titles of those prior works. STRICTLY EXCLUDE: datasets, pure mathematical theories, and evaluation metrics.
5. `algorithmic_nature`: Choose the best fit from:
   - "New_Architecture" (Proposes a completely new network topology from scratch)
   - "Loss_Modification" (Keeps standard networks but changes the loss/objective function)
   - "Plugin_Module" (Proposes a small module like a new Attention mechanism to plug into existing networks)
   - "Evaluation_Framework" (Not a standard training task, but an evaluation/agentic pipeline)
   - "Others" (Doesn't fit above categories, but still has clear engineering implications. Describe in `core_task`.)

Output ONLY the raw JSON object. Do not include markdown formatting like ```json.
"""

# --- Input Slicer ---

def extract_relevant_sections(paper_json: Dict, bib_data: Dict) -> str:
    """
    Extracts relevant sections adapting to S2ORC JSON format.
    Focuses on Abstract, Intro (last parts for contribution), Method, Experiments.
    Skips Related Work.
    Resolves citations using cite_spans.
    """
    relevant_text = []
    
    # 1. Abstract
    # S2ORC v1 might have abstract string or abstract list in pdf_parse
    if "abstract" in paper_json and isinstance(paper_json["abstract"], str):
         relevant_text.append(f"Abstract: {paper_json['abstract']}")
    elif "pdf_parse" in paper_json and "abstract" in paper_json["pdf_parse"]:
         # S2ORC abstract is often a list of dicts
         abstract_objs = paper_json["pdf_parse"]["abstract"]
         if isinstance(abstract_objs, list):
             abstract_texts = [p.get("text", "") for p in abstract_objs]
             relevant_text.append(f"Abstract: {' '.join(abstract_texts)}")
        
    # 2. Body Text (Handling S2ORC structure)
    if "pdf_parse" in paper_json and "body_text" in paper_json["pdf_parse"]:
        intro_paragraphs = []
        
        for paragraph in paper_json["pdf_parse"]["body_text"]:
            sec_title = paragraph.get("section", "")
            if not sec_title: 
                # Sometimes null section, treat as content if we are in main flow? 
                # Or skip. Usually skip if unknown context.
                pass
                
            sec_title_lower = str(sec_title).lower() if sec_title else ""
            
            # Resolve citations FIRST
            # This mutates the text for our purpose (but doesn't change original JSON unless we reassign, 
            # here we just get the string)
            resolved_para_text = resolve_s2orc_citations(paragraph.get("text", ""), paragraph.get("cite_spans", []), bib_data)
            full_para = f"Section [{sec_title}]: {resolved_para_text}"

            # Heuristic routing
            if "introduction" in sec_title_lower:
                intro_paragraphs.append(resolved_para_text)
            elif "related work" in sec_title_lower or "background" in sec_title_lower:
                continue # Explicitly skip
            elif any(k in sec_title_lower for k in ["method", "proposed", "approach", "architecture", "model"]):
                relevant_text.append(full_para)
            elif any(k in sec_title_lower for k in ["experiment", "evaluation", "implementation", "training", "setup", "result"]):
                relevant_text.append(full_para)
                
        # Only take the last 2 paragraphs of the introduction (usually contains the contributions)
        if intro_paragraphs:
            num_intro_to_keep = min(2, len(intro_paragraphs))
            # Join with newline?
            intro_segment = "\n".join(intro_paragraphs[-num_intro_to_keep:])
            relevant_text.append(f"Section [Introduction Contributions]: {intro_segment}")

    return "\n\n".join(relevant_text)

def prepare_llm_input(paper_json: Dict, bib_data: Dict) -> str:
    """
    Orchestrates the input preparation: slicing -> citation resolution.
    Note: Citation resolution happens INSIDE slicing now to use cite_spans.
    """
    return extract_relevant_sections(paper_json, bib_data)

# --- Semantic Parsing Layer ---

class SemanticParser:
    def __init__(self, client: OpenAI, model_version: str = "gpt-5-mini"):
        self.client = client
        self.model = model_version

    def parse(self, resolved_text: str) -> Dict:
        """
        Sends the resolved text to LLM and returns the parsed JSON structure.
        """
        user_content = f"Please analyze the following paper text:\n\n[PAPER TEXT START]\n{resolved_text}\n[PAPER TEXT END]"
        
        try:
            # Handle temperature constraints for reasoning models (o1, o3, gpt-5-mini, etc.)
            temperature = 0.1
            if self.model.startswith("o") or "gpt-5" in self.model:
                temperature = 1

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=temperature
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            print(f"[SemanticParser] Error: {e}")
            return {}

# --- Main Pipeline ---

def main():
    parser = argparse.ArgumentParser(description="0.1.1 Semantic Parsing Layer")
    parser.add_argument("--input_json_path", type=str, required=True, help="Path to paper JSON")
    parser.add_argument("--input_bib_path", type=str, help="Path to bibliography JSON (optional if embedded)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSON")
    parser.add_argument("--api_key", type=str, help="OpenAI API Key")
    parser.add_argument("--gpt_version", type=str, default="gpt-5-mini")
    
    args = parser.parse_args()
    
    # 1. Setup
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[Error] OpenAI API Key is required.")
        # return # Allow running for testing extraction if skipped? No, parser needs key.
        # But for testing slicing logic we might want to skip.
        # Let's enforce it for now unless mocked.
    
    client = OpenAI(api_key=api_key) if api_key else None
    
    # 2. Load Data
    print(f"[Info] Loading data from {args.input_json_path}...")
    paper_data = load_json_file(args.input_json_path)
    
    bib_data = None
    if args.input_bib_path:
        bib_data = load_json_file(args.input_bib_path)
    
    # Fallback: check internal bib
    if not bib_data:
        # Check standard S2ORC keys
        if "bib_entries" in paper_data:
            print("[Info] Using embedded bibliography (bib_entries).")
            bib_data = paper_data["bib_entries"]
        elif "pdf_parse" in paper_data and "bib_entries" in paper_data["pdf_parse"]:
            print("[Info] Using embedded bibliography (pdf_parse.bib_entries).")
            bib_data = paper_data["pdf_parse"]["bib_entries"]
        elif "ref_entries" in paper_data:
            # ref_entries usually contains FIGREF, TABREF, BIBREF. Filter for BIBREF?
            # Or just pass it all, utils.py handles key lookup.
             print("[Info] Using embedded bibliography (ref_entries).")
             bib_data = paper_data["ref_entries"]
    
    if not paper_data:
        print("[Error] Failed to load input paper.")
        return

    # 3. Prepare Input (Slicing + Resolution)
    print("[Info] Preparing LLM input (Slicing & Citation Resolution)...")
    resolved_text = prepare_llm_input(paper_data, bib_data)
    
    # Debug: Save intermediate resolved text? (Optional)
    # save_json_file({"text": resolved_text}, args.output_path + ".debug.json")

    # 4. Semantic Parsing
    if client:
        print(f"[Info] Running Semantic Analysis with {args.gpt_version}...")
        parser = SemanticParser(client, args.gpt_version)
        result = parser.parse(resolved_text)
        
        # 5. Save Output
        if result:
            print(f"[Info] Semantic analysis successful. Saving to {args.output_path}")
            save_json_file(result, args.output_path)
        else:
            print("[Error] Semantic analysis failed.")
    else:
        print("[Info] No API Key provided. Outputting prepared text for debug.")
        save_json_file({"extracted_text": resolved_text}, args.output_path + ".debug.json")

if __name__ == "__main__":
    main()
