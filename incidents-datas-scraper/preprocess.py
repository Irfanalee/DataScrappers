"""
Preprocess incident data from all sources into training format.
Combines GitHub Issues, GitHub Discussions, and Stack Overflow.
"""

import json
import random
import re
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILES = {
    "github_issues": "data/github_issues/all_issues.json",
    "github_discussions": "data/github_discussions/all_discussions.json",
    "stackoverflow": "data/stackoverflow/all_stackoverflow.json",
}

OUTPUT_DIR = "data/processed"
TRAIN_FILE = f"{OUTPUT_DIR}/train.jsonl"
EVAL_FILE = f"{OUTPUT_DIR}/eval.jsonl"

TRAIN_RATIO = 0.9

# System prompt for the model
SYSTEM_PROMPT = """You are an expert DevOps engineer and SRE. Analyze the provided error logs, stack traces, or incident descriptions. 

Your response should include:
1. **Root Cause**: What is causing this issue
2. **Severity**: Low / Medium / High / Critical
3. **Fix**: Step-by-step solution to resolve the issue
4. **Prevention**: How to prevent this in the future (optional)

Be direct, specific, and actionable. Reference exact commands, config changes, or code fixes when applicable."""


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove URLs (optional - they often break)
    # text = re.sub(r'https?://\S+', '[URL]', text)
    
    return text.strip()


def extract_error_snippet(text: str, max_length: int = 1500) -> str:
    """Extract the most relevant error portion from text."""
    if len(text) <= max_length:
        return text
    
    # Look for error indicators
    error_patterns = [
        r'error[:\s].*',
        r'exception[:\s].*',
        r'traceback.*',
        r'failed[:\s].*',
        r'fatal[:\s].*',
    ]
    
    lines = text.split('\n')
    relevant_lines = []
    in_error_block = False
    
    for line in lines:
        line_lower = line.lower()
        
        # Check if this line starts an error block
        if any(re.search(p, line_lower) for p in error_patterns):
            in_error_block = True
        
        if in_error_block or any(kw in line_lower for kw in ['error', 'exception', 'fail', 'denied', 'refused']):
            relevant_lines.append(line)
        
        if len('\n'.join(relevant_lines)) > max_length:
            break
    
    if relevant_lines:
        return '\n'.join(relevant_lines)[:max_length]
    
    # Fallback: just truncate
    return text[:max_length] + "\n... (truncated)"


def format_solution(solution: str, tech: str) -> str:
    """Format solution into structured response."""
    solution = clean_text(solution)
    
    # If solution is already well-structured, return as-is
    if any(marker in solution.lower() for marker in ['root cause', 'fix:', 'solution:', 'the issue']):
        return solution
    
    # Otherwise, return cleaned solution
    return solution


def process_github_issue(item: dict) -> dict:
    """Process a GitHub issue into training format."""
    problem = clean_text(item.get("problem", ""))
    solution = clean_text(item.get("solution", ""))
    tech = item.get("tech", "unknown")
    title = item.get("title", "")
    
    # Combine title + problem
    if title and title.lower() not in problem.lower():
        problem = f"{title}\n\n{problem}"
    
    problem = extract_error_snippet(problem)
    
    return {
        "tech": tech,
        "source": "github_issues",
        "problem": problem,
        "solution": solution,
        "url": item.get("url", ""),
    }


def process_github_discussion(item: dict) -> dict:
    """Process a GitHub discussion into training format."""
    problem = clean_text(item.get("question", ""))
    solution = clean_text(item.get("answer", ""))
    tech = item.get("tech", "unknown")
    title = item.get("title", "")
    
    if title and title.lower() not in problem.lower():
        problem = f"{title}\n\n{problem}"
    
    problem = extract_error_snippet(problem)
    
    return {
        "tech": tech,
        "source": "github_discussions",
        "problem": problem,
        "solution": solution,
        "url": item.get("url", ""),
    }


def process_stackoverflow(item: dict) -> dict:
    """Process a Stack Overflow Q&A into training format."""
    problem = clean_text(item.get("problem", ""))
    solution = clean_text(item.get("solution", ""))
    tech = item.get("tech", "unknown")
    title = item.get("title", "")
    
    if title and title.lower() not in problem.lower():
        problem = f"{title}\n\n{problem}"
    
    problem = extract_error_snippet(problem)
    
    return {
        "tech": tech,
        "source": "stackoverflow",
        "problem": problem,
        "solution": solution,
        "url": item.get("url", ""),
    }


def is_quality_example(item: dict) -> tuple[bool, str]:
    """Check if example meets quality standards."""
    problem = item.get("problem", "")
    solution = item.get("solution", "")
    
    # Length checks
    if len(problem) < 50:
        return False, "problem_too_short"
    if len(problem) > 5000:
        return False, "problem_too_long"
    if len(solution) < 50:
        return False, "solution_too_short"
    if len(solution) > 3000:
        return False, "solution_too_long"
    
    # Must have error-like content in problem
    problem_lower = problem.lower()
    error_indicators = [
        "error", "fail", "exception", "crash", "timeout",
        "not working", "broken", "issue", "problem",
        "unable", "cannot", "can't", "doesn't", "refused",
        "denied", "rejected", "invalid", "missing",
    ]
    
    has_error = any(ind in problem_lower for ind in error_indicators)
    if not has_error:
        return False, "no_error_indicator"
    
    # Solution should have actionable content
    solution_lower = solution.lower()
    action_indicators = [
        "try", "use", "change", "set", "add", "remove",
        "install", "update", "run", "execute", "configure",
        "the issue", "the problem", "because", "caused by",
        "solution", "fix", "resolve", "workaround",
    ]
    
    has_action = any(ind in solution_lower for ind in action_indicators)
    if not has_action:
        return False, "no_actionable_solution"
    
    return True, "ok"


def format_training_example(item: dict) -> dict:
    """Format into ChatML training structure."""
    tech = item.get("tech", "unknown")
    problem = item["problem"]
    solution = item["solution"]
    
    user_content = f"Analyze this {tech} incident and provide diagnosis and fix:\n\n```\n{problem}\n```"
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": solution}
        ],
        "_meta": {
            "tech": tech,
            "source": item.get("source", "unknown"),
            "url": item.get("url", ""),
        }
    }


def load_and_process_source(filepath: str, processor_func) -> list:
    """Load a source file and process its examples."""
    print(f"  Loading {filepath}...")
    
    try:
        with open(filepath) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"    File not found, skipping")
        return []
    
    examples = data.get("examples", [])
    print(f"    Raw examples: {len(examples)}")
    
    processed = []
    for item in examples:
        processed.append(processor_func(item))
    
    return processed


def main():
    print("=" * 60)
    print("INCIDENT RESPONDER - DATA PREPROCESSING")
    print("=" * 60)
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load and process all sources
    all_examples = []
    source_counts = {}
    
    # GitHub Issues
    print("\n[1/3] GitHub Issues")
    items = load_and_process_source(INPUT_FILES["github_issues"], process_github_issue)
    all_examples.extend(items)
    source_counts["github_issues"] = len(items)
    
    # GitHub Discussions
    print("\n[2/3] GitHub Discussions")
    items = load_and_process_source(INPUT_FILES["github_discussions"], process_github_discussion)
    all_examples.extend(items)
    source_counts["github_discussions"] = len(items)
    
    # Stack Overflow
    print("\n[3/3] Stack Overflow")
    items = load_and_process_source(INPUT_FILES["stackoverflow"], process_stackoverflow)
    all_examples.extend(items)
    source_counts["stackoverflow"] = len(items)
    
    print(f"\nTotal raw examples: {len(all_examples)}")
    
    # Quality filtering
    print("\nFiltering for quality...")
    filtered = []
    filter_stats = {}
    
    for item in all_examples:
        passed, reason = is_quality_example(item)
        if passed:
            filtered.append(item)
        else:
            filter_stats[reason] = filter_stats.get(reason, 0) + 1
    
    print(f"After filtering: {len(filtered)}")
    print(f"Filtered out: {len(all_examples) - len(filtered)}")
    for reason, count in sorted(filter_stats.items(), key=lambda x: -x[1]):
        print(f"  - {reason}: {count}")
    
    # Format for training
    print("\nFormatting for training...")
    formatted = [format_training_example(item) for item in filtered]
    
    # Shuffle
    random.seed(42)
    random.shuffle(formatted)
    
    # Split train/eval
    split_idx = int(len(formatted) * TRAIN_RATIO)
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_data)}")
    print(f"  Eval: {len(eval_data)}")
    
    # Tech distribution in training data
    tech_counts = {}
    for item in train_data:
        tech = item["_meta"]["tech"]
        tech_counts[tech] = tech_counts.get(tech, 0) + 1
    
    print(f"\nTraining data by tech:")
    for tech, count in sorted(tech_counts.items(), key=lambda x: -x[1]):
        print(f"  {tech}: {count}")
    
    # Save
    print(f"\nSaving...")
    
    with open(TRAIN_FILE, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    print(f"  {TRAIN_FILE}")
    
    with open(EVAL_FILE, "w") as f:
        for item in eval_data:
            f.write(json.dumps(item) + "\n")
    print(f"  {EVAL_FILE}")
    
    # Save stats
    stats_file = f"{OUTPUT_DIR}/preprocessing_stats.json"
    with open(stats_file, "w") as f:
        json.dump({
            "processed_at": datetime.now().isoformat(),
            "source_counts": source_counts,
            "total_raw": len(all_examples),
            "total_filtered": len(filtered),
            "filter_stats": filter_stats,
            "train_count": len(train_data),
            "eval_count": len(eval_data),
            "tech_distribution": tech_counts,
        }, f, indent=2)
    print(f"  {stats_file}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Training examples: {len(train_data)}")
    print(f"Eval examples: {len(eval_data)}")
    
    # Preview
    print("\n--- Sample Training Example ---")
    sample = train_data[0]
    print(f"Tech: {sample['_meta']['tech']}")
    print(f"Source: {sample['_meta']['source']}")
    print(f"\n[USER]:\n{sample['messages'][1]['content'][:500]}...")
    print(f"\n[ASSISTANT]:\n{sample['messages'][2]['content'][:500]}...")


if __name__ == "__main__":
    main()
