from openai import OpenAI
import requests
import json
import os
import time
import re

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") 


def search_github_url(query_term: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant that finds official GitHub repositories "
                        "for academic papers or software tools. "
                        "Return only the URL of the most relevant official repository. "
                        "If you are not confident, return 'NOT_FOUND' instead of guessing."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Find the official GitHub repository for: '{query_term}'. "
                        "Return only the raw GitHub URL, no explanation, no markdown, nothing else."
                    )
                }
            ]
        )

        result = response.choices[0].message.content.strip()
        if result == "NOT_FOUND":
            return None
        for word in result.split():
            word = word.strip(".,\n")
            if "github.com" in word:
                return word
        return None

    except Exception as e:
        print(f"[Search] Error for '{query_term}': {e}")
        return None


def extract_repo_path(url: str) -> str:
    """Extract owner/repo from GitHub URL."""
    match = re.search(r'github\.com/([^/]+/[^/]+)', url)
    if match:
        return match.group(1).rstrip("/")
    return None


def get_repo_metadata(github_url: str) -> dict:
    """Fetch stars, description, and readme snippet from GitHub API."""
    repo_path = extract_repo_path(github_url)
    if not repo_path:
        return {"stars": 0, "description": None, "readme_snippet": None}

    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    meta = {"stars": 0, "description": None, "readme_snippet": None}

    # Fetch repo info
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{repo_path}",
            headers=headers,
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            meta["stars"] = data.get("stargazers_count", 0)
            meta["description"] = data.get("description")
    except Exception as e:
        print(f"[GitHub] Repo fetch failed for {repo_path}: {e}")

    time.sleep(0.5)

    # Fetch README
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{repo_path}/readme",
            headers={**headers, "Accept": "application/vnd.github.raw"},
            timeout=10
        )
        if resp.status_code == 200:
            readme = resp.text
            # Take first 500 chars as snippet
            meta["readme_snippet"] = readme[:500].strip()
    except Exception as e:
        print(f"[GitHub] README fetch failed for {repo_path}: {e}")

    return meta


def process_semantic_json(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        data = json.load(f)

    relevant_works = data.get("relevant_works", [])
    results = []

    for work in relevant_works:
        query_term = work.get("title", "")
        query_source = work.get("domain_tag", "unknown")
        is_general = work.get("is_general", False)

        if not query_term:
            continue

        if is_general:
            print(f"[Skipping] General work, no repo needed: '{query_term}'")
            continue

        print(f"\n[Processing] source={query_source} | query='{query_term}'")

        github_url = search_github_url(query_term)
        print(f"[URL] {github_url}")

        if not github_url:
            print("[Info] No URL found")
            repo_meta = None
        else:
            time.sleep(1.5)
            repo_meta = get_repo_metadata(github_url)

        result = {
            "query_source": query_source,
            "query_term": query_term,
            "target_url": github_url,
            "repo_meta": repo_meta
        }

        results.append(result)
        print(f"[Done] stars={repo_meta['stars'] if repo_meta else 'N/A'}")

        time.sleep(2.0)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {len(results)} results → {output_path}")


if __name__ == "__main__":
    process_semantic_json(
        input_path="codes/baseline_agent/outputs/iTransformer/preprocess_artifacts/enriched_semantic.json",
        output_path="codes/baseline_agent/outputs/iTransformer/output_repo_urls.json"
    )
