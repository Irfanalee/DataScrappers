"""
Stack Overflow Scraper for DevOps Incident Responder
Scrapes questions with accepted answers from 2021+ using Stack Exchange API
"""

import requests
import json
import time
import html
import re
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = "data/stackoverflow"

# Stack Exchange API (no auth needed, but rate limited)
# With API key: 10,000 requests/day
# Without: 300 requests/day
API_KEY = os.environ.get("STACKEXCHANGE_API_KEY", "")

BASE_URL = "https://api.stackexchange.com/2.3"

# Date filter - Unix timestamp for 2021-01-01
MIN_DATE_UNIX = 1609459200  # 2021-01-01 00:00:00 UTC

# Tags to scrape - your tech stack
TECH_TAGS = {
    "kubernetes": ["kubernetes", "k8s", "kubectl", "minikube", "helm"],
    "docker": ["docker", "docker-compose", "dockerfile", "docker-swarm"],
    "terraform": ["terraform", "terraform-provider-aws", "terraform-provider-azure", "hcl"],
    "azure": ["azure", "azure-devops", "azure-functions", "azure-cli"],
    "gcp": ["google-cloud-platform", "gcloud", "google-cloud-functions", "bigquery"],
    "nodejs": ["node.js", "express", "npm", "nestjs"],
    "redis": ["redis", "redis-cluster", "redis-sentinel"],
    "mongodb": ["mongodb", "mongoose", "mongodb-query", "pymongo"],
    "nginx": ["nginx", "nginx-config", "nginx-reverse-proxy", "nginx-ingress"],
    "postgresql": ["postgresql", "psycopg2", "postgres", "plpgsql"],
    "influxdb": ["influxdb", "influxdb-2", "telegraf", "flux"],
}

# Minimum score for quality filtering
MIN_SCORE = 1
MIN_ANSWER_SCORE = 1


def make_api_request(endpoint: str, params: dict) -> dict:
    """Make a request to Stack Exchange API."""
    params["site"] = "stackoverflow"
    params["filter"] = "withbody"  # Include body content
    
    if API_KEY:
        params["key"] = API_KEY
    
    url = f"{BASE_URL}/{endpoint}"
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"  API error: {response.status_code}")
        return None
    
    data = response.json()
    
    # Check quota
    quota_remaining = data.get("quota_remaining", 0)
    if quota_remaining < 10:
        print(f"  Warning: API quota low ({quota_remaining} remaining)")
    
    # Handle backoff
    if "backoff" in data:
        backoff = data["backoff"]
        print(f"  Backoff requested: {backoff}s")
        time.sleep(backoff)
    
    return data


def clean_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Keep code blocks but mark them
    text = re.sub(r'<code>(.*?)</code>', r'`\1`', text, flags=re.DOTALL)
    text = re.sub(r'<pre>(.*?)</pre>', r'```\n\1\n```', text, flags=re.DOTALL)
    
    # Remove other HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return text


def get_questions_by_tag(tag: str, max_questions: int = 200) -> list:
    """Fetch questions with accepted answers for a tag."""
    
    questions = []
    page = 1
    
    while len(questions) < max_questions:
        params = {
            "tagged": tag,
            "sort": "votes",
            "order": "desc",
            "pagesize": 100,
            "page": page,
            "fromdate": MIN_DATE_UNIX,
            "filter": "!nNPvSNdWme",  # Include body, answers
        }
        
        data = make_api_request("questions", params)
        
        if not data or "items" not in data:
            break
        
        items = data["items"]
        if not items:
            break
        
        # Filter for questions with accepted answers
        for q in items:
            if q.get("accepted_answer_id") and q.get("score", 0) >= MIN_SCORE:
                questions.append(q)
        
        print(f"    Page {page}: {len(items)} questions, {len(questions)} with accepted answers")
        
        if not data.get("has_more", False):
            break
        
        page += 1
        time.sleep(0.5)  # Rate limiting
        
        if page > 10:  # Safety limit
            break
    
    return questions[:max_questions]


def get_answer(answer_id: int) -> dict:
    """Fetch a specific answer by ID."""
    
    params = {
        "filter": "!nNPvSNdWme",  # Include body
    }
    
    data = make_api_request(f"answers/{answer_id}", params)
    
    if data and "items" in data and data["items"]:
        return data["items"][0]
    
    return None


def extract_incident_data(question: dict, answer: dict, tech: str, tag: str) -> dict:
    """Extract incident data from question + answer."""
    
    problem = clean_html(question.get("body", ""))
    solution = clean_html(answer.get("body", ""))
    
    return {
        "tech": tech,
        "source": "stackoverflow",
        "tag": tag,
        "question_id": question.get("question_id"),
        "title": question.get("title", ""),
        "problem": problem,
        "solution": solution,
        "question_score": question.get("score", 0),
        "answer_score": answer.get("score", 0),
        "created_at": datetime.fromtimestamp(question.get("creation_date", 0)).isoformat(),
        "url": question.get("link", ""),
    }


def is_quality_qa(question: dict, answer: dict) -> bool:
    """Filter for quality Q&A pairs."""
    
    # Score filters
    if question.get("score", 0) < MIN_SCORE:
        return False
    if answer.get("score", 0) < MIN_ANSWER_SCORE:
        return False
    
    # Length filters
    q_body = question.get("body", "")
    a_body = answer.get("body", "")
    
    if len(q_body) < 100:
        return False
    if len(a_body) < 100:
        return False
    
    # Error indicators in question
    title = question.get("title", "").lower()
    body_lower = q_body.lower()
    
    error_indicators = [
        "error", "fail", "exception", "crash", "timeout",
        "not working", "broken", "issue", "problem",
        "unable to", "cannot", "can't", "doesn't work",
        "refused", "denied", "rejected", "invalid",
    ]
    
    combined = title + " " + body_lower
    has_error = any(indicator in combined for indicator in error_indicators)
    
    return has_error


def scrape_tag(tag: str, tech: str, max_per_tag: int = 150) -> list:
    """Scrape questions for a single tag."""
    
    print(f"  Tag: {tag}")
    
    # Get questions
    questions = get_questions_by_tag(tag, max_per_tag)
    print(f"    Found {len(questions)} questions with accepted answers")
    
    examples = []
    
    for q in questions:
        # Get the accepted answer
        answer_id = q.get("accepted_answer_id")
        if not answer_id:
            continue
        
        # Check if answer is in the question data
        answer = None
        for a in q.get("answers", []):
            if a.get("answer_id") == answer_id:
                answer = a
                break
        
        # If not, fetch it separately
        if not answer:
            answer = get_answer(answer_id)
            time.sleep(0.3)
        
        if not answer:
            continue
        
        # Quality filter
        if not is_quality_qa(q, answer):
            continue
        
        # Extract data
        data = extract_incident_data(q, answer, tech, tag)
        examples.append(data)
    
    print(f"    Quality examples: {len(examples)}")
    return examples


def scrape_all_tags():
    """Scrape all tags for all technologies."""
    
    print("=" * 60)
    print("STACK OVERFLOW SCRAPER - DevOps Incident Data")
    print("=" * 60)
    print(f"Date filter: >= 2021-01-01")
    print(f"Technologies: {list(TECH_TAGS.keys())}")
    if API_KEY:
        print("Using API key (10k requests/day)")
    else:
        print("No API key (300 requests/day - consider adding one)")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    all_examples = []
    stats = {"by_tech": {}, "by_tag": {}}
    
    for tech, tags in TECH_TAGS.items():
        print(f"\n{'='*60}")
        print(f"Technology: {tech.upper()}")
        print(f"{'='*60}")
        
        tech_examples = []
        
        for tag in tags:
            try:
                examples = scrape_tag(tag, tech)
                tech_examples.extend(examples)
                stats["by_tag"][tag] = len(examples)
            except Exception as e:
                print(f"    Error scraping {tag}: {e}")
                continue
            
            time.sleep(1)
        
        # Deduplicate by question_id within tech
        seen_ids = set()
        unique_examples = []
        for ex in tech_examples:
            qid = ex.get("question_id")
            if qid not in seen_ids:
                seen_ids.add(qid)
                unique_examples.append(ex)
        
        all_examples.extend(unique_examples)
        stats["by_tech"][tech] = len(unique_examples)
        
        print(f"\n  {tech} total (deduplicated): {len(unique_examples)}")
        
        # Save per-tech file
        tech_file = f"{OUTPUT_DIR}/{tech}_stackoverflow.json"
        with open(tech_file, "w") as f:
            json.dump({
                "tech": tech,
                "count": len(unique_examples),
                "examples": unique_examples
            }, f, indent=2)
    
    # Save combined file
    combined_file = f"{OUTPUT_DIR}/all_stackoverflow.json"
    with open(combined_file, "w") as f:
        json.dump({
            "scraped_at": datetime.now().isoformat(),
            "source": "stackoverflow",
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
    scrape_all_tags()
