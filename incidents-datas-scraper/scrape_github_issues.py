"""
GitHub Issues Scraper for DevOps Incident Responder
Scrapes closed issues with solutions from major DevOps repos (2021+)
"""

import requests
import json
import time
import os
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Set GITHUB_TOKEN environment variable")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

BASE_URL = "https://api.github.com"
OUTPUT_DIR = "data/github_issues"

# Date filter - only issues from 2021 onwards
MIN_DATE = "2021-01-01"

# Target repos organized by technology
TARGET_REPOS = {
    "kubernetes": [
        "kubernetes/kubernetes",
        "kubernetes/minikube",
        "kubernetes/kubectl",
    ],
    "docker": [
        "docker/compose",
        "docker/cli",
        "moby/moby",
    ],
    "terraform": [
        "hashicorp/terraform",
        "hashicorp/terraform-provider-aws",
        "hashicorp/terraform-provider-azurerm",
        "hashicorp/terraform-provider-google",
    ],
    "azure": [
        "Azure/azure-cli",
        "Azure/azure-sdk-for-python",
        "Azure/azure-functions-python-worker",
    ],
    "gcp": [
        "googleapis/google-cloud-python",
        "GoogleCloudPlatform/python-docs-samples",
    ],
    "nodejs": [
        "nodejs/node",
        "vercel/next.js",
        "expressjs/express",
    ],
    "redis": [
        "redis/redis",
        "redis/redis-py",
    ],
    "mongodb": [
        "mongodb/mongo",
        "mongodb/mongo-python-driver",
    ],
    "nginx": [
        "nginx/nginx",
        "nginxinc/kubernetes-ingress",
    ],
    "postgresql": [
        "postgres/postgres",
        "psycopg/psycopg2",
    ],
    "influxdb": [
        "influxdata/influxdb",
        "influxdata/telegraf",
    ],
}

# Labels that indicate bugs/issues with fixes
BUG_LABELS = [
    "bug",
    "fix",
    "fixed",
    "resolved",
    "type/bug",
    "kind/bug",
    "type:bug",
    "kind:bug",
]


def rate_limit_check(response):
    """Check rate limit and sleep if needed."""
    remaining = int(response.headers.get("X-RateLimit-Remaining", 1))
    if remaining < 10:
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        sleep_time = max(reset_time - time.time(), 0) + 5
        print(f"  Rate limit low ({remaining}). Sleeping {sleep_time:.0f}s...")
        time.sleep(sleep_time)


def get_closed_issues(repo: str, max_issues: int = 500) -> list:
    """Fetch closed issues from a repo created after MIN_DATE."""
    issues = []
    page = 1
    per_page = 100
    
    while len(issues) < max_issues:
        url = f"{BASE_URL}/repos/{repo}/issues"
        params = {
            "state": "closed",
            "sort": "updated",
            "direction": "desc",
            "per_page": per_page,
            "page": page,
            "since": f"{MIN_DATE}T00:00:00Z",
        }
        
        response = requests.get(url, headers=HEADERS, params=params)
        rate_limit_check(response)
        
        if response.status_code != 200:
            print(f"  Error fetching issues: {response.status_code}")
            break
        
        data = response.json()
        if not data:
            break
        
        # Filter out pull requests (they come through issues API too)
        real_issues = [i for i in data if "pull_request" not in i]
        
        # Filter for issues created after MIN_DATE
        for issue in real_issues:
            created = issue.get("created_at", "")[:10]
            if created >= MIN_DATE:
                issues.append(issue)
        
        print(f"    Page {page}: {len(real_issues)} issues (total: {len(issues)})")
        
        if len(data) < per_page:
            break
        
        page += 1
        time.sleep(0.3)
    
    return issues[:max_issues]


def get_issue_comments(repo: str, issue_number: int) -> list:
    """Fetch comments for a specific issue."""
    url = f"{BASE_URL}/repos/{repo}/issues/{issue_number}/comments"
    
    response = requests.get(url, headers=HEADERS)
    rate_limit_check(response)
    
    if response.status_code != 200:
        return []
    
    return response.json()


def has_solution_indicators(issue: dict, comments: list) -> bool:
    """Check if issue + comments indicate a solution was found."""
    # Check issue body for solution indicators
    body = (issue.get("body") or "").lower()
    
    # Check comments
    all_text = body + " ".join([(c.get("body") or "").lower() for c in comments])
    
    solution_phrases = [
        "fixed by",
        "solved by",
        "the fix is",
        "the solution is",
        "this was fixed",
        "this was resolved",
        "workaround:",
        "workaround is",
        "i fixed it by",
        "i solved it by",
        "the issue was",
        "root cause",
        "the problem was",
        "this happens because",
        "you need to",
        "try this:",
        "the answer is",
        "solution:",
    ]
    
    for phrase in solution_phrases:
        if phrase in all_text:
            return True
    
    # Check if closed by a commit/PR (indicates fix)
    if issue.get("closed_at") and len(comments) > 0:
        return True
    
    return False


def extract_incident_data(issue: dict, comments: list, repo: str, tech: str) -> dict:
    """Extract incident data from issue + comments."""
    
    # Get the problem description (issue body)
    problem = issue.get("body") or ""
    
    # Get solution from comments (prioritize maintainer/author comments)
    solution_comments = []
    issue_author = issue.get("user", {}).get("login", "")
    
    for comment in comments:
        commenter = comment.get("user", {}).get("login", "")
        body = comment.get("body") or ""
        
        # Skip very short comments
        if len(body) < 50:
            continue
        
        # Prioritize comments that look like solutions
        body_lower = body.lower()
        is_solution = any(phrase in body_lower for phrase in [
            "fixed", "solved", "solution", "workaround", "try this",
            "you need to", "the issue", "the problem", "root cause"
        ])
        
        if is_solution:
            solution_comments.append({
                "body": body,
                "is_author": commenter == issue_author,
                "commenter": commenter,
            })
    
    # Combine best solution comments
    solution = "\n\n---\n\n".join([c["body"] for c in solution_comments[:3]])
    
    return {
        "tech": tech,
        "repo": repo,
        "issue_number": issue.get("number"),
        "title": issue.get("title"),
        "problem": problem,
        "solution": solution,
        "labels": [l.get("name") for l in issue.get("labels", [])],
        "created_at": issue.get("created_at"),
        "closed_at": issue.get("closed_at"),
        "url": issue.get("html_url"),
        "comments_count": len(comments),
    }


def is_quality_issue(issue: dict) -> bool:
    """Filter for quality issues worth scraping."""
    body = issue.get("body") or ""
    title = issue.get("title") or ""
    
    # Must have some description
    if len(body) < 100:
        return False
    
    # Skip feature requests
    title_lower = title.lower()
    if any(skip in title_lower for skip in ["feature request", "enhancement", "proposal", "[rfc]"]):
        return False
    
    # Look for error indicators
    error_indicators = [
        "error", "fail", "crash", "exception", "timeout",
        "refused", "denied", "not working", "broken",
        "issue", "problem", "bug", "stack trace",
    ]
    
    combined = (title + " " + body).lower()
    has_error = any(indicator in combined for indicator in error_indicators)
    
    return has_error


def scrape_repo(repo: str, tech: str, max_issues: int = 300) -> list:
    """Scrape issues from a single repo."""
    print(f"\n  Scraping: {repo}")
    
    # Get closed issues
    issues = get_closed_issues(repo, max_issues)
    print(f"  Found {len(issues)} closed issues (2021+)")
    
    # Filter for quality
    quality_issues = [i for i in issues if is_quality_issue(i)]
    print(f"  Quality issues: {len(quality_issues)}")
    
    examples = []
    
    for i, issue in enumerate(quality_issues[:100]):  # Limit per repo
        # Get comments
        comments = get_issue_comments(repo, issue["number"])
        
        # Check for solution
        if not has_solution_indicators(issue, comments):
            continue
        
        # Extract data
        data = extract_incident_data(issue, comments, repo, tech)
        
        # Skip if no solution found
        if not data["solution"] or len(data["solution"]) < 50:
            continue
        
        examples.append(data)
        
        if (i + 1) % 20 == 0:
            print(f"    Processed {i+1} issues, {len(examples)} with solutions")
        
        time.sleep(0.2)
    
    return examples


def scrape_all_repos():
    """Scrape all target repos."""
    print("=" * 60)
    print("GITHUB ISSUES SCRAPER - DevOps Incident Data")
    print("=" * 60)
    print(f"Date filter: >= {MIN_DATE}")
    print(f"Technologies: {list(TARGET_REPOS.keys())}")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    all_examples = []
    stats = {"by_tech": {}, "by_repo": {}}
    
    for tech, repos in TARGET_REPOS.items():
        print(f"\n{'='*60}")
        print(f"Technology: {tech.upper()}")
        print(f"{'='*60}")
        
        tech_examples = []
        
        for repo in repos:
            try:
                examples = scrape_repo(repo, tech)
                tech_examples.extend(examples)
                stats["by_repo"][repo] = len(examples)
                print(f"  ✓ {repo}: {len(examples)} examples")
            except Exception as e:
                print(f"  ✗ {repo}: Error - {e}")
                continue
            
            time.sleep(1)
        
        all_examples.extend(tech_examples)
        stats["by_tech"][tech] = len(tech_examples)
        
        # Save per-tech file
        tech_file = f"{OUTPUT_DIR}/{tech}_issues.json"
        with open(tech_file, "w") as f:
            json.dump({
                "tech": tech,
                "count": len(tech_examples),
                "examples": tech_examples
            }, f, indent=2)
        print(f"  Saved: {tech_file}")
    
    # Save combined file
    combined_file = f"{OUTPUT_DIR}/all_issues.json"
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
    scrape_all_repos()
