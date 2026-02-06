"""
GitHub Discussions Scraper for DevOps Incident Responder
Scrapes Q&A discussions with accepted answers from 2021+
Uses GraphQL API for discussions
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

# =============================================================================
# CONFIGURATION
# =============================================================================

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Set GITHUB_TOKEN environment variable")

HEADERS = {
    "Authorization": f"bearer {GITHUB_TOKEN}",
    "Content-Type": "application/json",
}

GRAPHQL_URL = "https://api.github.com/graphql"
OUTPUT_DIR = "data/github_discussions"

# Date filter
MIN_DATE = "2021-01-01"

# Repos with active discussions
TARGET_REPOS = {
    "kubernetes": [
        ("kubernetes", "kubernetes"),
        ("kubernetes", "minikube"),
    ],
    "docker": [
        ("docker", "compose"),
    ],
    "terraform": [
        ("hashicorp", "terraform"),
    ],
    "nodejs": [
        ("vercel", "next.js"),
        ("nodejs", "node"),
    ],
    "azure": [
        ("Azure", "azure-cli"),
    ],
    "nginx": [
        ("nginxinc", "kubernetes-ingress"),
    ],
}


def run_graphql_query(query: str, variables: dict = None) -> dict:
    """Execute a GraphQL query."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    response = requests.post(GRAPHQL_URL, headers=HEADERS, json=payload)
    
    if response.status_code != 200:
        print(f"GraphQL error: {response.status_code}")
        print(response.text)
        return None
    
    data = response.json()
    
    if "errors" in data:
        print(f"GraphQL errors: {data['errors']}")
        return None
    
    return data


def get_discussions(owner: str, repo: str, max_discussions: int = 200) -> list:
    """Fetch discussions from a repo using GraphQL."""
    
    query = """
    query($owner: String!, $repo: String!, $cursor: String) {
      repository(owner: $owner, name: $repo) {
        discussions(first: 50, after: $cursor, orderBy: {field: UPDATED_AT, direction: DESC}) {
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            number
            title
            body
            createdAt
            updatedAt
            url
            answer {
              body
              createdAt
              author {
                login
              }
            }
            author {
              login
            }
            category {
              name
            }
            comments(first: 10) {
              nodes {
                body
                createdAt
                author {
                  login
                }
                isAnswer
              }
            }
          }
        }
      }
    }
    """
    
    discussions = []
    cursor = None
    
    while len(discussions) < max_discussions:
        variables = {
            "owner": owner,
            "repo": repo,
            "cursor": cursor
        }
        
        result = run_graphql_query(query, variables)
        
        if not result:
            break
        
        repo_data = result.get("data", {}).get("repository")
        if not repo_data:
            print(f"  No discussions found for {owner}/{repo}")
            break
        
        disc_data = repo_data.get("discussions", {})
        nodes = disc_data.get("nodes", [])
        
        if not nodes:
            break
        
        # Filter by date
        for node in nodes:
            created = node.get("createdAt", "")[:10]
            if created >= MIN_DATE:
                discussions.append(node)
        
        print(f"    Fetched {len(nodes)} discussions (total: {len(discussions)})")
        
        page_info = disc_data.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            break
        
        cursor = page_info.get("endCursor")
        time.sleep(0.5)
    
    return discussions[:max_discussions]


def extract_discussion_data(discussion: dict, repo: str, tech: str) -> dict:
    """Extract Q&A data from a discussion."""
    
    # Get the question
    question = discussion.get("body") or ""
    title = discussion.get("title") or ""
    
    # Get the answer (prioritize marked answer)
    answer = ""
    answer_obj = discussion.get("answer")
    
    if answer_obj:
        answer = answer_obj.get("body") or ""
    else:
        # Look for answer in comments
        comments = discussion.get("comments", {}).get("nodes", [])
        for comment in comments:
            if comment.get("isAnswer"):
                answer = comment.get("body") or ""
                break
        
        # If no marked answer, take longest helpful comment
        if not answer:
            helpful_comments = [
                c.get("body", "") for c in comments 
                if len(c.get("body", "")) > 100
            ]
            if helpful_comments:
                answer = max(helpful_comments, key=len)
    
    return {
        "tech": tech,
        "repo": repo,
        "discussion_number": discussion.get("number"),
        "title": title,
        "question": question,
        "answer": answer,
        "category": discussion.get("category", {}).get("name"),
        "created_at": discussion.get("createdAt"),
        "url": discussion.get("url"),
        "has_official_answer": discussion.get("answer") is not None,
    }


def is_quality_discussion(discussion: dict) -> bool:
    """Filter for quality Q&A discussions."""
    
    body = discussion.get("body") or ""
    title = discussion.get("title") or ""
    category = (discussion.get("category") or {}).get("name", "").lower()
    
    # Must have substantial question
    if len(body) < 50:
        return False
    
    # Prefer Q&A, Help, Troubleshooting categories
    good_categories = ["q&a", "help", "troubleshooting", "support", "question"]
    if category and not any(cat in category for cat in good_categories):
        # Still allow if it looks like an error/issue
        pass
    
    # Skip announcements, show & tell
    skip_categories = ["announcement", "show", "ideas", "rfc"]
    if any(skip in category for skip in skip_categories):
        return False
    
    # Look for error/issue indicators
    combined = (title + " " + body).lower()
    error_indicators = [
        "error", "fail", "crash", "exception", "timeout",
        "not working", "broken", "issue", "problem", "help",
        "how to fix", "how do i", "why does", "doesn't work",
    ]
    
    has_error = any(indicator in combined for indicator in error_indicators)
    
    return has_error


def scrape_repo_discussions(owner: str, repo: str, tech: str) -> list:
    """Scrape discussions from a single repo."""
    
    print(f"\n  Scraping: {owner}/{repo}")
    
    # Get discussions
    discussions = get_discussions(owner, repo)
    print(f"  Found {len(discussions)} discussions (2021+)")
    
    # Filter for quality
    quality = [d for d in discussions if is_quality_discussion(d)]
    print(f"  Quality Q&A: {len(quality)}")
    
    examples = []
    
    for discussion in quality:
        data = extract_discussion_data(discussion, f"{owner}/{repo}", tech)
        
        # Skip if no answer
        if not data["answer"] or len(data["answer"]) < 50:
            continue
        
        examples.append(data)
    
    print(f"  With answers: {len(examples)}")
    return examples


def scrape_all_discussions():
    """Scrape all target repos."""
    
    print("=" * 60)
    print("GITHUB DISCUSSIONS SCRAPER - DevOps Q&A Data")
    print("=" * 60)
    print(f"Date filter: >= {MIN_DATE}")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    all_examples = []
    stats = {"by_tech": {}, "by_repo": {}}
    
    for tech, repos in TARGET_REPOS.items():
        print(f"\n{'='*60}")
        print(f"Technology: {tech.upper()}")
        print(f"{'='*60}")
        
        tech_examples = []
        
        for owner, repo in repos:
            try:
                examples = scrape_repo_discussions(owner, repo, tech)
                tech_examples.extend(examples)
                stats["by_repo"][f"{owner}/{repo}"] = len(examples)
                print(f"  ✓ {owner}/{repo}: {len(examples)} examples")
            except Exception as e:
                print(f"  ✗ {owner}/{repo}: Error - {e}")
                continue
            
            time.sleep(1)
        
        all_examples.extend(tech_examples)
        stats["by_tech"][tech] = len(tech_examples)
    
    # Save combined file
    combined_file = f"{OUTPUT_DIR}/all_discussions.json"
    with open(combined_file, "w") as f:
        json.dump({
            "scraped_at": datetime.now().isoformat(),
            "min_date": MIN_DATE,
            "total": len(all_examples),
            "stats": stats,
            "examples": all_examples
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"Total examples: {len(all_examples)}")
    print(f"\nBy technology:")
    for tech, count in stats["by_tech"].items():
        print(f"  {tech}: {count}")
    print(f"\nSaved to: {combined_file}")
    
    return all_examples, stats


if __name__ == "__main__":
    scrape_all_discussions()
