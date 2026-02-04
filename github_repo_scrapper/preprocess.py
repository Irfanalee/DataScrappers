"""
Preprocess scraped GitHub PR comments into training format.
Converts raw data into prompt-completion pairs for fine-tuning.
"""

import json
import random
import re
from pathlib import Path


def clean_diff_hunk(diff_hunk: str) -> str:
    """
    Extract clean code from diff hunk.
    Removes diff markers (+/-) and keeps relevant code context.
    """
    if not diff_hunk:
        return ""
    
    lines = diff_hunk.split("\n")
    cleaned_lines = []
    
    for line in lines:
        # Skip diff header lines
        if line.startswith("@@"):
            continue
        # Remove the +/- prefix but keep the line
        if line.startswith("+") or line.startswith("-"):
            cleaned_lines.append(line[1:])
        elif line.startswith(" "):
            cleaned_lines.append(line[1:])
        else:
            cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines).strip()


def clean_comment(comment: str) -> str:
    """
    Clean up the review comment.
    Removes excessive whitespace, GitHub mentions, etc.
    """
    if not comment:
        return ""
    
    # Remove GitHub @mentions
    comment = re.sub(r"@[\w-]+", "", comment)
    
    # Remove image links
    comment = re.sub(r"!\[.*?\]\(.*?\)", "", comment)
    
    # Remove regular links but keep text
    comment = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", comment)
    
    # Clean up excessive whitespace
    comment = re.sub(r"\n{3,}", "\n\n", comment)
    comment = comment.strip()
    
    return comment


def is_quality_example(code: str, comment: str) -> tuple[bool, str]:
    """
    Additional quality filters beyond the scraper.
    Returns (passed, reason).
    """
    # Code checks
    if len(code) < 20:
        return False, "code_too_short"
    if len(code) > 3000:
        return False, "code_too_long"
    
    # Comment checks
    if len(comment) < 30:
        return False, "comment_too_short"
    if len(comment) > 1500:
        return False, "comment_too_long"
    
    # Skip if comment is mostly code (likely a suggestion, not explanation)
    code_block_ratio = comment.count("```") / max(len(comment), 1)
    if code_block_ratio > 0.01 and comment.count("```") >= 4:
        return False, "mostly_code_blocks"
    
    # Skip non-English (rough heuristic)
    ascii_ratio = sum(1 for c in comment if ord(c) < 128) / max(len(comment), 1)
    if ascii_ratio < 0.8:
        return False, "likely_non_english"
    
    # Skip if just asking a question (not giving feedback)
    lower_comment = comment.lower().strip()
    if lower_comment.endswith("?") and comment.count("?") > comment.count("."):
        return False, "mostly_questions"
    
    return True, "ok"


def format_training_example(code: str, comment: str, file_path: str) -> dict:
    """
    Format into the training structure.
    Using ChatML-style format compatible with most fine-tuning frameworks.
    """
    
    # Extract filename for context
    filename = file_path.split("/")[-1] if file_path else "code.py"
    
    system_prompt = "You are an expert code reviewer. Analyze the provided Python code and give constructive, specific feedback. Focus on bugs, potential issues, code quality, and improvements. Be direct and actionable."
    
    user_prompt = f"""Review this Python code from `{filename}`:

```python
{code}
```"""

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": comment}
        ]
    }


def format_alpaca_style(code: str, comment: str, file_path: str) -> dict:
    """
    Alternative format: Alpaca-style instruction/input/output.
    Some training frameworks prefer this.
    """
    filename = file_path.split("/")[-1] if file_path else "code.py"
    
    return {
        "instruction": "Review the following Python code and provide constructive, specific feedback on potential bugs, issues, and improvements.",
        "input": f"File: {filename}\n\n```python\n{code}\n```",
        "output": comment
    }


def process_dataset(
    input_file: str = "data/all_examples.json",
    output_dir: str = "data/processed",
    train_ratio: float = 0.9,
    format_type: str = "chatml"  # "chatml" or "alpaca"
):
    """
    Process the raw dataset into training format.
    Splits into train/eval sets.
    """
    print(f"Loading data from {input_file}...")
    with open(input_file) as f:
        data = json.load(f)
    
    examples = data["examples"]
    print(f"Loaded {len(examples)} raw examples")
    
    # Process and filter
    processed = []
    stats = {"total": len(examples), "kept": 0, "filtered": {}}
    
    for ex in examples:
        code = clean_diff_hunk(ex.get("diff_hunk", ""))
        comment = clean_comment(ex.get("comment", ""))
        file_path = ex.get("file_path", "")
        
        # Quality check
        passed, reason = is_quality_example(code, comment)
        
        if not passed:
            stats["filtered"][reason] = stats["filtered"].get(reason, 0) + 1
            continue
        
        # Format based on chosen style
        if format_type == "chatml":
            formatted = format_training_example(code, comment, file_path)
        else:
            formatted = format_alpaca_style(code, comment, file_path)
        
        # Add metadata for debugging (won't be used in training)
        formatted["_meta"] = {
            "repo": ex.get("repo"),
            "pr_number": ex.get("pr_number"),
            "url": ex.get("url")
        }
        
        processed.append(formatted)
        stats["kept"] += 1
    
    print(f"\nProcessing complete:")
    print(f"  Kept: {stats['kept']}")
    print(f"  Filtered: {sum(stats['filtered'].values())}")
    print(f"  Filter breakdown: {json.dumps(stats['filtered'], indent=4)}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(processed)
    
    split_idx = int(len(processed) * train_ratio)
    train_data = processed[:split_idx]
    eval_data = processed[split_idx:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_data)}")
    print(f"  Eval: {len(eval_data)}")
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    train_file = f"{output_dir}/train.jsonl"
    eval_file = f"{output_dir}/eval.jsonl"
    
    with open(train_file, "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")
    
    with open(eval_file, "w") as f:
        for ex in eval_data:
            f.write(json.dumps(ex) + "\n")
    
    # Also save a combined JSON for inspection
    with open(f"{output_dir}/all_processed.json", "w") as f:
        json.dump({
            "format": format_type,
            "stats": stats,
            "train_count": len(train_data),
            "eval_count": len(eval_data),
            "examples": processed[:10]  # First 10 for inspection
        }, f, indent=2)
    
    print(f"\nSaved:")
    print(f"  {train_file} ({len(train_data)} examples)")
    print(f"  {eval_file} ({len(eval_data)} examples)")
    print(f"  {output_dir}/all_processed.json (stats + 10 samples)")
    
    return train_data, eval_data, stats


def preview_examples(input_file: str = "data/processed/train.jsonl", n: int = 3):
    """Preview a few processed examples."""
    print(f"\n{'='*60}")
    print(f"PREVIEWING {n} EXAMPLES")
    print(f"{'='*60}\n")
    
    with open(input_file) as f:
        lines = f.readlines()
    
    samples = random.sample(lines, min(n, len(lines)))
    
    for i, line in enumerate(samples, 1):
        ex = json.loads(line)
        print(f"--- Example {i} ---")
        
        if "messages" in ex:  # ChatML format
            for msg in ex["messages"]:
                role = msg["role"].upper()
                content = msg["content"]
                if len(content) > 300:
                    content = content[:300] + "..."
                print(f"[{role}]: {content}\n")
        else:  # Alpaca format
            print(f"[INSTRUCTION]: {ex['instruction']}")
            print(f"[INPUT]: {ex['input'][:300]}...")
            print(f"[OUTPUT]: {ex['output'][:300]}...")
        
        print(f"\nSource: {ex.get('_meta', {}).get('repo', 'unknown')}")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess code review data")
    parser.add_argument("--input", default="data/all_examples.json", help="Input JSON file")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--format", choices=["chatml", "alpaca"], default="chatml", help="Output format")
    parser.add_argument("--preview", action="store_true", help="Preview examples after processing")
    
    args = parser.parse_args()
    
    train_data, eval_data, stats = process_dataset(
        input_file=args.input,
        output_dir=args.output,
        format_type=args.format
    )
    
    if args.preview:
        preview_examples(f"{args.output}/train.jsonl")