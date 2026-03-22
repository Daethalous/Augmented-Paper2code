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
import fitz

# --- PROMPTS (From LIRA/src/prompts/lineage_prompts.py) ---
BASELINE_SELECTION_SYSTEM_PROMPT = """
# Role
You are the **Code Lineage Detective** for the LIRA(Lineage Informed Reproduce Agent) framework.
Your goal is to identify the **single best existing codebase** from the Reference List that serves as the starting point for reproducing the target paper, and **most relevant related works** to refine the code further.

# Input Data
- **Target Paper Context**: Excerpts from the paper (focus on Implementation Details), with reference citations concluded.

# STRICT COMPLIANCE RULES (VIOLATIONS WILL CAUSE SYSTEM FAILURE)
1. **NO EXTERNAL KNOWLEDGE**: You are a specific string extractor. If a paper is famously known as "DeiT" but the reference list only says "Training data-efficient image transformers...", you MUST return "Training data-efficient image transformers...". Do NOT add "DeiT:".
2. **NO HALLUCINATION**: You can ONLY select papers that are physically present in the "Reference List" text below. If "Faster R-CNN" is not there, do NOT return it, even if the paper talks about it.
3. **EXACT STRING COPY**: Your output title must match the reference list characters exactly (including punctuation).

# Task
1. **Match Citations**: Find citation numbers in Context (e.g. [1]) and locate them in Reference List.
2. **Select Baseline**:
   - Limit choice to papers explicitly cited in the "Reference List".
   - If the authors indicated that their works was build upon earlier works, e.g. "Our work bases on [11]","We follow [63]", go to the reference and copy the title *exactly*.

# Output Format (JSON)
{
  "baseline_candidate": {
    "title": "EXACT_COPY_OF_TITLE_FROM_LIST",
    "is_explicitly_mentioned_for_code": true,
    "reasoning": "Reason for choosing the baseline like \\"Context says 'We follow [...]', and Ref [...] is '...'\\""
  },
  "related_work_candidates": [
    "Exact Title 1",
    "Exact Title 2",
    ... up to 5 titles
  ]
}
"""

BASELINE_SELECTION_USER_PROMPT_TEMPLATE = """
# Target Paper
Title: {target_paper_title}

# Full Paper Context
{target_paper_context}

# Reference List
{reference_list}

# Task
Identify the Baseline and Top 5 Related Papers from the Reference List above according to the Engineering Ancestor criteria.
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

    def retrieve_full_paper_text(self, query):
        if not self.sch_engine: return "ARXIV_MODULE_MISSING"
        pdf_text = str()
        filename = f"temp_{query}.pdf"
        try:
            search = arxiv.Search(id_list=[query])
            results = list(self.sch_engine.results(search))
            if not results:
                 return "DOWNLOAD FAILED: Paper not found"
            paper = results[0]
            # Download the PDF to a temp location
            paper.download_pdf(filename=filename)
        except Exception as e:
             return f"DOWNLOAD FAILED: {str(e)}"
        
        if not os.path.exists(filename):
            return "DOWNLOAD FAILED: File not found" 
            
        try:
            if not fitz:
                return "PYMUPDF_MISSING"
            # creating a pdf reader object
            doc = fitz.open(filename)
            # Iterate over all the pages
            for page_number in range(len(doc)):
                # Extract text from the page
                page = doc.load_page(page_number)
                text = page.get_text("text")
                
                # Do something with the text (e.g., print it)
                pdf_text += f"--- Page {page_number} ---"
                pdf_text += text
                pdf_text += "\n"
            doc.close()
        except Exception as e:
            return f"TEXT EXTRACTION FAILED: {str(e)}"
        finally:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except:
                    pass
        return pdf_text

def get_repo_details(repo_path: str) -> Optional[Dict]:
    """Replication of LIRA's github_ops.get_repo_details"""
    # Simple GitHub API check without auth (rate limited but functional for basic checks)
    # LIRA imports this, we implement a basic version
    if repo_path.startswith("http"):
        # Extract owner/repo
        match = re.search(r'github\.com/([^/]+/[^/]+)', repo_path)
        if match:
            repo_path = match.group(1)
        else:
            return None
            
    api_url = f"https://api.github.com/repos/{repo_path}"
    try:
        # Check if repo exists
        # NOTE: Ideally use Authorization header if token is available
        cmds = ["curl", "-s", api_url]
        ret = subprocess.run(cmds, capture_output=True, text=True)
        if ret.returncode == 0:
            data = json.loads(ret.stdout)
            if "name" in data and "html_url" in data:
                 return {
                     "name": data.get("full_name"),
                     "url": data.get("html_url"),
                     "stars": data.get("stargazers_count", 0),
                     "description": data.get("description")
                 }
    except Exception as e:
        print(f"GitHub check failed: {e}")
    return None

class LineageManager:
    """Replication of LIRA's LineageManager"""
    def __init__(self, openai_client=None, gpt_version="gpt-5-mini"):
        self.arxiv = ArxivSearch()
        self.client = openai_client
        self.gpt_version = gpt_version
        # We skip logging to file in this script for simplicity, or print to stdout

    def select_baseline_and_related(self, target_paper_json: Dict, target_paper_title: str, bib_json: Dict, log_file: str = None) -> Dict:
        if not self.client:
            raise ValueError("OpenAI client required.")
            
        # 0. Helper function (moved inside or reference from self)     
        
        # Compress Bibliography: simplified view
        simple_bib = {}
        if isinstance(bib_json, dict):
            for k, v in bib_json.items():
                # Keep ID, Title, and Raw Text (truncated)
                # Some bib entries might be lists or strings if malformed, handle safely
                if isinstance(v, dict):
                    title = v.get('title', 'Unknown')
                    simple_bib[k] = f"\"title\": \"{title}\""
                else:
                    simple_bib[k] = str(v)[:200]
        
        # Limit bib string length to avoid context overflow (approx 5-8k tokens)
        bib_str = json.dumps(simple_bib, indent=2)[:25000] 

        # 1. Prepare Content directly from JSON
        # Concatenate JSON content into a structured prompt string
        system_prompt = BASELINE_SELECTION_SYSTEM_PROMPT
        user_prompt = BASELINE_SELECTION_USER_PROMPT_TEMPLATE.format(
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
            print(f"[Lineage] LLM Response: {content[:100]}...")

            if log_file:
                with open(log_file, "w", encoding="utf-8") as f:
                    log_data = {
                        "model": self.gpt_version,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "response": content
                    }
                    json.dump(log_data, f, indent=2, ensure_ascii=False)
                print(f"[Lineage] LLM interaction log saved to {log_file}")
            
            result = json.loads(content)
            baseline_info = result.get("baseline_candidate", {})
            
            return {
                "baseline_query": baseline_info.get("title"),
                "baseline_reasoning": baseline_info.get("reasoning", ""),
                "related_queries": result.get("related_work_candidates", []),
                "related_reasoning": "" 
            }

        except Exception as e:
            print(f"[LineageManager] Error: {e}")
            if log_file:
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"ERROR: {str(e)}\n")
            return {
                "baseline_query": None,
                "baseline_reasoning": f"Error: {e}",
                "related_queries": [],
                "related_reasoning": ""
            }

    def find_paper_implementation(self, paper_info: Dict) -> Dict:
        title = paper_info.get('title', '')
        arxiv_id = paper_info.get('arxiv_id', paper_info.get('doi', ''))
        
        repo_data = None
        used_query = ""
        strategy = ""

        # --- Strategy: Full Text Extraction ---
        if arxiv_id:
            print(f"[Lineage] Strategy: Downloading/Parsing Full Text for {arxiv_id}...")
            paper_text = self.arxiv.retrieve_full_paper_text(arxiv_id)
            
            if paper_text and "DOWNLOAD FAILED" not in paper_text:
                # Regex to find github.com/user/repo
                github_pattern = r'(?:https?://)?github\.com/([a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+)'
                found_repos = re.findall(github_pattern, paper_text)
                
                valid_repos = []
                for r_path in found_repos:
                    r_path = r_path.rstrip(".,)]}'\"")
                    if r_path.lower() in ['pytorch/pytorch', 'tensorflow/tensorflow', 'huggingface/transformers']:
                         continue
                    valid_repos.append(r_path)
                
                if valid_repos:
                    unique_candidates = list(set(valid_repos))
                    print(f"[Lineage] Found candidates in text: {unique_candidates}")
                    
                    for candidate in unique_candidates[:3]: # check top 3
                        details = get_repo_details(candidate)
                        if details:
                            repo_data = details
                            strategy = "full_text_link_extraction"
                            used_query = f"extract:{candidate}"
                            print(f"[Lineage] Success: Verified {candidate}")
                            break
            else:
                 print("[Lineage] Failed to retrieve full text for extraction.")

        if not repo_data:
             print("[Lineage] No official repository link found in paper text.")

        return {
            "paper_title": title,
            "arxiv_id": arxiv_id,
            "search_query": used_query,
            "search_strategy": strategy,
            "official_repo": repo_data['url'] if repo_data else None,
            "repo_stars": repo_data['stars'] if repo_data else 0,
            "found": repo_data is not None
        }

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

    # --- Step 2: Identify Credits (LIRA logic) ---
    print("[Parser] Identifying Baseline & Related from References...")
    
    # Prepare logging path
    preprocess_dir = os.path.join(args.output_dir, "preprocess_artifacts")
    os.makedirs(preprocess_dir, exist_ok=True)
    log_file_path = os.path.join(preprocess_dir, "llm_selection_log.json")
    
    credits_result = lm.select_baseline_and_related(paper_data, paper_title, bib_data, log_file=log_file_path)
    
    # --- Step 3: Resolve ArXiv Metadata (LIRA logic) ---
    print("[Parser] Resolving ArXiv Metadata...")
    resolved_baseline = None
    resolved_related = []
    
    # 3.1 Resolve Baseline
    b_query = credits_result.get('baseline_query')
    if b_query:
        print(f"[Parser] Resolving Baseline: '{b_query}'")
        if arxiv_searcher.sch_engine:
            resolved_baseline = None
            arxiv_id = arxiv_searcher.get_arxiv_id_by_title(b_query)
            if arxiv_id:
                try:
                    search = arxiv_searcher.sch_engine.results(arxiv.Search(id_list=[arxiv_id]))
                    # Handle generator
                    results = list(search)
                    if results:
                        r = results[0]
                        resolved_baseline = {
                            "source": "arxiv",
                            "title": r.title,
                            "summary": r.summary,
                            "arxiv_id": r.entry_id.split('/')[-1],
                            "date": str(r.published).split(" ")[0]
                        }
                except Exception as e:
                     print(f"[Parser] Metadata fetch failed for {arxiv_id}: {e}")
            if not resolved_baseline:
                print(f"[Parser] Baseline not found on ArXiv: {b_query}")
        else:
            print("[Parser] Arxiv module missing, cannot resolve baseline.")

    
    # 3.2 Resolve Related
    r_queries = credits_result.get('related_queries', [])
    for q in r_queries:
        if not q: continue
        time.sleep(1.0) # Rate limit
        if arxiv_searcher.sch_engine:
             arxiv_id = arxiv_searcher.get_arxiv_id_by_title(q)
             if arxiv_id:
                try:
                    search = arxiv_searcher.sch_engine.results(arxiv.Search(id_list=[arxiv_id]))
                    results = list(search)
                    if results:
                        r = results[0]
                        resolved_related.append({
                            "source": "arxiv",
                            "title": r.title,
                            "summary": r.summary,
                            "arxiv_id": r.entry_id.split('/')[-1],
                            "date": str(r.published).split(" ")[0]
                        })
                except:
                    pass
    
    print(f"[Parser] Resolved {len(resolved_related)}/{len(r_queries)} related papers")

    # --- Step 4: Find Implementations (LIRA logic) ---
    print("[Parser] Finding GitHub Implementations...")
    impl_results = {"baseline": None, "related": []}
    
    if resolved_baseline:
        print(f"[Parser] Searching for Baseline Code: {resolved_baseline['title']}...")
        impl = lm.find_paper_implementation(resolved_baseline)
        impl_results["baseline"] = impl
    
    for rp in resolved_related:
        print(f"[Parser] Searching Related: {rp['title'][:30]}...")
        impl = lm.find_paper_implementation(rp)
        impl_results["related"].append(impl)

    # --- Step 5: Assemble Report ---
    final_report = {
        "paper_info": {
            "title": paper_title,
            "processed_time": datetime.now().isoformat()
        },
        "baseline_selection": {
            "query_from_llm": credits_result.get('baseline_query'),
            "reasoning": credits_result.get('baseline_reasoning'),
            "resolved_metadata": resolved_baseline,
            "implementation": impl_results['baseline']
        },
        "related_selection": {
            "queries_from_llm": credits_result.get('related_queries'),
            "resolved_metadata": resolved_related,
            "implementations": impl_results['related']
        }
    }
    
    # Save Report
    # preprocess_dir is already created
    report_path = os.path.join(preprocess_dir, "baseline_analysis.json")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2)
    print(f"[Parser] Report saved: {report_path}")

    # Step 6: Clone (Adaptation: LIRA Agent does this in baseline.py, so we include it here as requested)
    target_repo = None
    if impl_results["baseline"] and impl_results["baseline"].get("official_repo"):
        target_repo = impl_results["baseline"]["official_repo"]
    
    if target_repo:
        print(f"[INFO] Cloning identified repo: {target_repo}")
        run_git_clone(target_repo, args.output_repo_dir)
    else:
        print("[INFO] No official baseline repository found via direct lineage search.")


def run_git_clone(repo_url: str, target_dir: str):
    try:
        subprocess.run(["git", "clone", repo_url, target_dir], check=True)
        print(f"[SUCCESS] Cloned {repo_url}")
    except Exception as e:
        print(f"[ERROR] Clone failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # P2C Inputs
    parser.add_argument('--paper_name',type=str)
    parser.add_argument('--gpt_version',type=str, default="gpt-5-mini")
    parser.add_argument('--paper_format',type=str, default="JSON", choices=["JSON", "LaTeX"])
    parser.add_argument("--input_json_path", type=str, required=True) # P2C specific
    parser.add_argument("--input_bib_path", type=str, required=True) # P2C specific
    parser.add_argument('--output_dir',type=str, default="outputs")
    parser.add_argument("--output_repo_dir", type=str, default="./cloned_repo")
    parser.add_argument("--api_key", type=str, default=None)

    # Ignored args to maintain compatibility if called by existing scripts
    parser.add_argument('--pdf_json_path', type=str) 
    parser.add_argument('--pdf_latex_path', type=str)

    args = parser.parse_args()
    main(args)