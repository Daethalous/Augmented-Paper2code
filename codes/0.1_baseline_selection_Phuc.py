import json
import argparse
import os
import re
import sys
import time
import glob
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime

from openai import OpenAI
import arxiv

# --- PROMPTS ---

CLASSIFICATION_SYSTEM_PROMPT = """
# Role
You are a research paper analyst for the LIRA (Lineage Informed Reproduce Agent) framework.
Your goal is to classify the algorithmic nature of the target paper based on its content.

# Classification Tags
Choose exactly one of the following tags:
- "SOTA_Inherited": The paper directly builds upon one prior work's codebase as its starting point.
- "Plugin_Module": The paper introduces a new component that is plugged into multiple existing baselines.
- "Loss_Modification": The paper only modifies the training objective; the backbone architecture is unchanged.
- "Evaluation_Framework": The paper proposes a new benchmark or evaluation protocol rather than a new model.
- "New_Architecture": The paper proposes an entirely new model design with no single clear prior codebase.

# Strict Rules
1. Base your classification only on the paper content provided.
2. Your reasoning must reference specific phrases or sentences from the paper.
3. signals must be exact phrases copied from the paper text.

# Base Architecture Detection
Identify whether the paper builds upon or repurposes a known backbone architecture.
- Set "has_clear_backbone" to true if the paper explicitly uses or extends a known architecture.
- Set "backbone_names" to the list of known architectures used. Common examples include:
  Transformer, Informer, Reformer, Autoformer, Flowformer, Flashformer, TimesNet,
  ResNet, ViT, BERT, GPT, LLaMA, U-Net, GAN, Diffusion Model, LSTM, TCN.
- If no clear backbone is identified, set "has_clear_backbone" to false and "backbone_names" to [].

# Output Format (JSON)
{
  "algorithmic_nature": "<one of the 5 tags>",
  "reasoning": "<explanation referencing specific paper text>",
  "signals": ["<exact phrase from paper>", ...],
  "base_architecture": {
    "has_clear_backbone": <true or false>,
    "backbone_names": ["<architecture name>", ...]
  }
}
"""

CLASSIFICATION_USER_PROMPT_TEMPLATE = """
# Target Paper
Title: {target_paper_title}

# Paper Content
{target_paper_context}

# Reference List
{reference_list}

# Task
Classify the algorithmic nature of this paper using the tags defined in the system prompt.
"""

EXTRACTION_SYSTEM_PROMPT = """
# Role
You are the **Reproduction Dependency Analyst** for the LIRA (Lineage Informed Reproduce Agent) framework.
Your goal is to identify **all existing works strictly needed to reproduce the target paper's implementation**
by reasoning through each candidate work step by step before making a verdict.

# Input Data
- **Target Paper Content**: Full paper text including methodology, implementation details, and experiments.
- **Reference List**: The complete bibliography of the paper.

# Domain Tags
Assign exactly one domain tag to each INCLUDED work:
- **"algorithm_implementation"**: The core method or architecture directly relies on this work.
- **"data_pipeline"**: This work provides dataset loading, preprocessing, or augmentation logic.
- **"evaluation"**: This work defines metrics, benchmarking setup, or evaluation protocol.
- **"training_infrastructure"**: This work provides optimizer, scheduler, loss function, or training loop.

# General Work Classification
For each INCLUDED work, determine if it is a general/foundational work:
- Set "is_general" to true if the work is a widely known foundational method whose implementation
  is common knowledge and does not require cloning a specific repository. Examples include:
  Transformer, BERT, ResNet, Adam optimizer, Layer Normalization, Dropout, BatchNorm,
  attention mechanisms, positional encodings, standard loss functions.
- Set "is_general" to false if the work is a specific codebase, benchmark repository, or
  domain-specific implementation that must be obtained from a particular source to reproduce
  the paper's experiments.

# STRICT COMPLIANCE RULES (VIOLATIONS WILL CAUSE SYSTEM FAILURE)
1. **NO EXTERNAL KNOWLEDGE**: You are a string extractor. Copy titles EXACTLY as they appear in the Reference List. Do NOT add nicknames or abbreviations (e.g. if the reference says "Attention is all you need", do NOT write "Transformer (Attention is all you need)").
2. **NO HALLUCINATION**: You can ONLY select works physically present in the Reference List. If a work is not in the Reference List, do NOT include it even if the paper discusses it.
3. **EXACT STRING COPY**: Your output title must match the Reference List characters exactly, including punctuation and capitalization.

# CoT Reasoning Per Entry
For each candidate work you discover, reason through the following steps IN ORDER:

**Step 1 — What is this work?**
Briefly describe what this work is and what it contributes to the field.

**Step 2 — Where exactly is it mentioned in the paper?**
Copy the exact sentence(s) from the paper that mention this work. Do NOT paraphrase.

**Step 3 — Should this be discarded? (Early Exit)**
DISCARD immediately if ANY of the following are true:
- The work appears only in a comparison table or results section ("compared to X", "X achieves Y", "outperforms X")
- The work is a general-purpose library (e.g. PyTorch, NumPy, scikit-learn, pandas)
- The work is cited only to motivate the problem or provide background, with no implementation reuse
- The paper does not reuse any code, data, protocol, or component from this work
→ If ANY condition is true: DISCARD. Do NOT proceed to Steps 4 and 5. Do NOT emit this work.

**Step 4 — What domain does it serve?**
Based on how this work is used in the paper, classify it into exactly one of the following domains:
- **"algorithm_implementation"**: The paper reuses or builds upon this work's model architecture, modules, or core algorithmic components.
- **"data_pipeline"**: The paper reuses this work's datasets, data splits, or preprocessing protocols.
- **"evaluation"**: The paper reuses this work's evaluation metrics, benchmarks, or analysis tools.
- **"training_infrastructure"**: The paper reuses this work's training setup, optimizer, loss function, or learning rate schedule.

**Step 5 — Final Check Before Verdict**
Can you point to a specific sentence in the paper proving this work directly contributes
code, data, or protocol to the reproduction?
- If YES → **INCLUDE**
- If NO → **DISCARD**

# Output Format (JSON)
Only emit works with final verdict INCLUDE.
{
  "relevant_works": [
    {
      "title": "EXACT_COPY_OF_TITLE_FROM_REFERENCE_LIST",
      "domain_tag": "<one of the 4 domain tags>",
      "is_general": <true or false>,
      "context_snippet": "<exact sentence(s) from paper, no paraphrasing>",
      "reasoning": "<Step 1 → Step 2 → Step 3 (not discarded because...) → Step 4 → Step 5 verdict>"
    }
  ]
}
"""

EXTRACTION_USER_PROMPT_TEMPLATE = """
# Target Paper
Title: {target_paper_title}

# Paper Content
{target_paper_context}

# Reference List
{reference_list}

# Task
Scan the full paper content. For each work you identify as potentially relevant for reproduction,
reason through it step by step before deciding to include or discard it.
Only include works with verdict INCLUDE in your output.
"""

# --- HELPER CLASSES (Inline implementation of LIRA/src/tools) ---

class ArxivSearch:
    """Replication of LIRA's ArxivSearch"""
    def __init__(self):
        if arxiv:
            self.sch_engine = arxiv.Client()
        else:
            self.sch_engine = None
        
    def get_arxiv_id_by_title(self, paper_title):
        if not self.sch_engine: return None
        search = arxiv.Search(
            query=f'ti:"{paper_title}"',
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance
        )
        try:
            # Check results
            results = list(self.sch_engine.results(search))
            if results:
                result = results[0]
                return result.entry_id.split('/')[-1]
            return None
        except Exception as e:
            print(f"[ArxivSearch] Search failed for '{paper_title}': {e}")
            return None

class LineageManager:
    """Replication of LIRA's LineageManager"""
    def __init__(self, openai_client=None, gpt_version="gpt-5-mini"):
        self.arxiv = ArxivSearch()
        self.client = openai_client
        self.gpt_version = gpt_version
        # We skip logging to file in this script for simplicity, or print to stdout

    def _compress_bib(self, bib_json: Dict) -> str:
        """Compress bibliography to a simplified string for prompt injection."""
        simple_bib = {}
        if isinstance(bib_json, dict):
            for k, v in bib_json.items():
                if isinstance(v, dict):
                    title = v.get('title', 'Unknown')
                    simple_bib[k] = f"\"title\": \"{title}\""
                else:
                    simple_bib[k] = str(v)[:200]
        return json.dumps(simple_bib, indent=2)[:25000]

    def classify_paper(self, target_paper_json: Dict, target_paper_title: str, bib_json: Dict, log_file: str = None) -> Dict:
        """Stage 1: Classify the algorithmic nature of the paper."""
        if not self.client:
            raise ValueError("OpenAI client required.")

        bib_str = self._compress_bib(bib_json)
        system_prompt = CLASSIFICATION_SYSTEM_PROMPT
        user_prompt = CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
            target_paper_title=target_paper_title,
            target_paper_context=target_paper_json,
            reference_list=bib_str
        )

        try:
            response = self.client.chat.completions.create(
                model=self.gpt_version,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            print(f"[Classification] LLM Response: {content[:100]}...")

            if log_file:
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "model": self.gpt_version,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "response": content
                    }, f, indent=2, ensure_ascii=False)
                print(f"[Classification] Log saved to {log_file}")

            return json.loads(content)

        except Exception as e:
            print(f"[Classification] Error: {e}")
            if log_file:
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"ERROR: {str(e)}\n")
            return {
                "algorithmic_nature": "Unknown",
                "reasoning": f"Error: {e}",
                "signals": []
            }

    def extract_relevant_works(self, target_paper_json: Dict, target_paper_title: str, bib_json: Dict, log_file: str = None) -> List:
        """Stage 2: Extract relevant works via inline CoT reasoning."""
        if not self.client:
            raise ValueError("OpenAI client required.")

        bib_str = self._compress_bib(bib_json)
        system_prompt = EXTRACTION_SYSTEM_PROMPT
        user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
            target_paper_title=target_paper_title,
            target_paper_context=target_paper_json,
            reference_list=bib_str
        )

        try:
            response = self.client.chat.completions.create(
                model=self.gpt_version,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            print(f"[Extraction] LLM Response: {content[:100]}...")

            if log_file:
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "model": self.gpt_version,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "response": content
                    }, f, indent=2, ensure_ascii=False)
                print(f"[Extraction] Log saved to {log_file}")

            result = json.loads(content)
            return result.get("relevant_works", [])

        except Exception as e:
            print(f"[Extraction] Error: {e}")
            if log_file:
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"ERROR: {str(e)}\n")
            return []

# --- MAIN LOGIC (Replicating PaperPdfParser.parse but with P2C inputs) ---

def main(args):
    # Setup
    api_key = os.environ.get("OPENAI_API_KEY") or args.api_key
    model_id = args.gpt_version
    
    if not api_key:
        print("[ERROR] OpenAI API Key not found.")
        return

    client = OpenAI(api_key=api_key)
    
    # Init LIRA tools
    lm = LineageManager(openai_client=client, gpt_version=model_id)
    arxiv_searcher = ArxivSearch()

    # Load Inputs
    print(f"[Parser] Loading inputs from {args.input_json_path}")
    with open(args.input_json_path, 'r', encoding='utf-8') as f:
        paper_data = json.load(f)
    print(f"[Parser] Loading bib from {args.input_bib_path}")
    with open(args.input_bib_path, 'r', encoding='utf-8') as f:
        bib_data = json.load(f)

    # Convert to MD (The "LIRA-Input" Adaptation)
    print("[Parser] Reconstructing Markdown content...")
    paper_title = paper_data.get("title", "Unknown Paper")
    print(f"[Parser] Title extracted: {paper_title}")

    # --- Step 2: Classify Paper ---
    print("[Parser] Classifying paper...")
    preprocess_dir = os.path.join(args.output_dir, "preprocess_artifacts")
    os.makedirs(preprocess_dir, exist_ok=True)

    classification = lm.classify_paper(
        target_paper_json=paper_data,
        target_paper_title=paper_title,
        bib_json=bib_data,
        log_file=os.path.join(preprocess_dir, "llm_classification_log.json")
    )
    print(f"[Parser] Classification: {classification.get('algorithmic_nature')}")

    # --- Step 3: Extract Relevant Works ---
    print("[Parser] Extracting relevant works...")
    relevant_works = lm.extract_relevant_works(
        target_paper_json=paper_data,
        target_paper_title=paper_title,
        bib_json=bib_data,
        log_file=os.path.join(preprocess_dir, "llm_extraction_log.json")
    )
    print(f"[Parser] Found {len(relevant_works)} relevant works")

    # --- Step 4: Assemble Enriched Semantic JSON ---
    base_arch = classification.pop("base_architecture", {"has_clear_backbone": None, "backbone_names": []})
    output = {
        "paper_info": {
            "title": paper_title,
            "processed_time": datetime.now().isoformat()
        },
        "task_domain": None,
        "core_task": None,
        "base_architecture": base_arch,
        "algorithmic_nature": classification,
        "implementation_dependencies": {
            "relies_on_existing_codebase": None,
            "explicit_codebase_names": [],
            "based_on_prior_paper_code": None,
            "prior_paper_titles": []
        },
        "relevant_works": relevant_works
    }

    output_path = os.path.join(preprocess_dir, "enriched_semantic.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[Parser] Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # P2C Inputs
    parser.add_argument('--paper_name',type=str)
    parser.add_argument('--gpt_version',type=str, default="gpt-5-mini")
    parser.add_argument('--paper_format',type=str, default="JSON", choices=["JSON", "LaTeX"])
    parser.add_argument("--input_json_path", type=str, required=True) # P2C specific
    parser.add_argument("--input_bib_path", type=str, required=True) # P2C specific
    parser.add_argument('--output_dir',type=str, default="outputs")
    parser.add_argument("--api_key", type=str, default=None)

    # Ignored args to maintain compatibility if called by existing scripts
    parser.add_argument('--pdf_json_path', type=str) 
    parser.add_argument('--pdf_latex_path', type=str)

    args = parser.parse_args()
    main(args)