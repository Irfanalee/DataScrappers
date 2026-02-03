"""
GitHub PR Review Comment Scraper
Collects code review comments from high-quality repos for fine-tuning.
"""

import requests
import json
import time
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Configuration
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]  # Raises KeyError if not set

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

BASE_URL = "https://api.github.com"

# Target repos - high quality Python projects with good review culture
TARGET_REPOS = [
    "fastapi/fastapi",
    "pydantic/pydantic",
    "psf/requests",
    "encode/httpx",
    "astral-sh/ruff",
    "tiangolo/sqlmodel",
    "pallets/flask",
    "django/django",
    "pytorch/pytorch",
    "huggingface/transformers",
]

# Filtering thresholds
MIN_COMMENT_LENGTH = 50  # Skip short comments like "LGTM", "nit"
MAX_COMMENT_LENGTH = 2000  # Skip overly long comments (usually not focused)
MIN_CODE_CONTEXT_LINES = 3


def rate_limit_check(response):
    """Check rate limit and sleep if needed."""
    remaining = int(response.headers.get("X-RateLimit-Remaining", 1))
    if remaining < 10:
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        sleep_time = max(reset_time - time.time(), 0) + 5
        print(f"Rate limit low ({remaining}). Sleeping {sleep_time:.0f}s...")
        time.sleep(sleep_time)


def get_merged_prs(repo, per_page=100, max_prs=500):
    """Fetch merged PRs from a repo."""
    prs = []
    page = 1
    
    while len(prs) < max_prs:
        url = f"{BASE_URL}/repos/{repo}/pulls"
        params = {
            "state": "closed",
            "sort": "updated",
            "direction": "desc",
            "per_page": per_page,
            "page": page
        }
        
        response = requests.get(url, headers=HEADERS, params=params)
        rate_limit_check(response)
        
        if response.status_code != 200:
            print(f"Error fetching PRs from {repo}: {response.status_code}")
            break
        
        data = response.json()
        if not data:
            break
        
        # Filter for merged PRs only
        merged = [pr for pr in data if pr.get("merged_at")]
        prs.extend(merged)
        
        print(f"  Fetched page {page}: {len(merged)} merged PRs (total: {len(prs)})")
        page += 1
        
        if len(data) < per_page:
            break
        
        time.sleep(0.5)  # Be nice to GitHub
    
    return prs[:max_prs]


def get_pr_review_comments(repo, pr_number):
    """Fetch review comments for a specific PR."""
    url = f"{BASE_URL}/repos/{repo}/pulls/{pr_number}/comments"
    
    all_comments = []
    page = 1
    
    while True:
        params = {"per_page": 100, "page": page}
        response = requests.get(url, headers=HEADERS, params=params)
        rate_limit_check(response)
        
        if response.status_code != 200:
            break
        
        data = response.json()
        if not data:
            break
        
        all_comments.extend(data)
        page += 1
        
        if len(data) < 100:
            break
        
        time.sleep(0.3)
    
    return all_comments


def filter_comment(comment):
    """Filter out low-quality comments."""
    body = comment.get("body", "")
    
    # Length checks
    if len(body) < MIN_COMMENT_LENGTH:
        return False, "too_short"
    if len(body) > MAX_COMMENT_LENGTH:
        return False, "too_long"
    
    # Skip common low-value patterns
    lower_body = body.lower().strip()
    skip_patterns = [
        "lgtm",
        "looks good",
        "nit:",
        "nit ",
        "+1",
        "thanks!",
        "thank you",
        "nice!",
        "great!",
        "awesome",
        "ship it",
        "approved",
    ]
    
    for pattern in skip_patterns:
        if lower_body.startswith(pattern) and len(body) < 100:
            return False, f"skip_pattern:{pattern}"
    
    # Must have code context
    if not comment.get("diff_hunk"):
        return False, "no_diff_hunk"
    
    # Only Python files
    path = comment.get("path", "")
    if not path.endswith(".py"):
        return False, "not_python"
    
    return True, "ok"


def extract_training_example(comment, repo, pr_number):
    """Convert a review comment into a training example."""
    return {
        "repo": repo,
        "pr_number": pr_number,
        "file_path": comment.get("path"),
        "line": comment.get("original_line") or comment.get("line"),
        "side": comment.get("side"),  # LEFT or RIGHT
        "diff_hunk": comment.get("diff_hunk"),
        "comment": comment.get("body"),
        "comment_id": comment.get("id"),
        "user": comment.get("user", {}).get("login"),
        "created_at": comment.get("created_at"),
        "url": comment.get("html_url"),
    }


def scrape_repo(repo, max_prs=500, output_dir="data"):
    """Scrape review comments from a single repo."""
    print(f"\n{'='*60}")
    print(f"Scraping: {repo}")
    print(f"{'='*60}")
    
    # Get merged PRs
    print(f"Fetching merged PRs...")
    prs = get_merged_prs(repo, max_prs=max_prs)
    print(f"Found {len(prs)} merged PRs")
    
    examples = []
    stats = {"total_comments": 0, "filtered": {}, "kept": 0}
    
    for i, pr in enumerate(prs):
        pr_number = pr["number"]
        
        # Get review comments
        comments = get_pr_review_comments(repo, pr_number)
        stats["total_comments"] += len(comments)
        
        for comment in comments:
            passed, reason = filter_comment(comment)
            
            if passed:
                example = extract_training_example(comment, repo, pr_number)
                examples.append(example)
                stats["kept"] += 1
            else:
                stats["filtered"][reason] = stats["filtered"].get(reason, 0) + 1
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(prs)} PRs, {stats['kept']} examples collected")
        
        time.sleep(0.2)
    
    # Save to file
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    repo_name = repo.replace("/", "_")
    output_file = f"{output_dir}/{repo_name}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "repo": repo,
            "scraped_at": datetime.now().isoformat(),
            "stats": stats,
            "examples": examples
        }, f, indent=2)
    
    print(f"\nRepo complete: {repo}")
    print(f"  Total comments scanned: {stats['total_comments']}")
    print(f"  Examples kept: {stats['kept']}")
    print(f"  Filtered: {stats['filtered']}")
    print(f"  Saved to: {output_file}")
    
    return examples, stats


def main():
    """Main scraping loop."""
    print("GitHub PR Review Scraper")
    print(f"Target repos: {len(TARGET_REPOS)}")
    print(f"Token: {'*' * 10}{GITHUB_TOKEN[-4:]}")
    
    all_examples = []
    all_stats = {"total_comments": 0, "kept": 0, "by_repo": {}}
    
    for repo in TARGET_REPOS:
        try:
            examples, stats = scrape_repo(repo, max_prs=300)
            all_examples.extend(examples)
            all_stats["total_comments"] += stats["total_comments"]
            all_stats["kept"] += stats["kept"]
            all_stats["by_repo"][repo] = stats["kept"]
        except Exception as e:
            print(f"Error scraping {repo}: {e}")
            continue
        
        # Save combined file periodically
        with open("data/all_examples.json", "w") as f:
            json.dump({
                "scraped_at": datetime.now().isoformat(),
                "total_examples": len(all_examples),
                "stats": all_stats,
                "examples": all_examples
            }, f, indent=2)
        
        print(f"\n>>> Running total: {len(all_examples)} examples")
        time.sleep(1)
    
    print("\n" + "="*60)
    print("SCRAPING COMPLETE")
    print("="*60)
    print(f"Total examples collected: {len(all_examples)}")
    print(f"By repo: {json.dumps(all_stats['by_repo'], indent=2)}")
    print(f"Saved to: data/all_examples.json")


if __name__ == "__main__":
    main()
