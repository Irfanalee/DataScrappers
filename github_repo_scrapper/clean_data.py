"""
Clean training data by removing author responses and low-quality examples.
"""

import json
import re
from pathlib import Path

INPUT_FILE = "data/processed/train_with_synthetic.jsonl"
OUTPUT_FILE = "data/processed/train_cleaned.jsonl"

# Patterns that indicate author responses (not reviewer feedback)
AUTHOR_RESPONSE_PATTERNS = [
    r"^I've fixed",
    r"^I'll fix",
    r"^I've updated",
    r"^I'll update",
    r"^I've changed",
    r"^I'll change",
    r"^I've removed",
    r"^I'll remove",
    r"^I've added",
    r"^I'll add",
    r"^Fixed it",
    r"^Done!",
    r"^Done\.",
    r"^Good catch!",
    r"^Thanks for",
    r"^Thank you for",
    r"^Addressed",
    r"^Updated",
    r"^Changed as suggested",
    r"^Applied",
    r"^Resolved",
    r"^Good point!",
    r"^Nice catch",
    r"^Ah yes",
    r"^Ah,",
    r"^Oops",
    r"^My bad",
    r"^You're right",
    r"^Makes sense",
]

# Compile patterns for efficiency
AUTHOR_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in AUTHOR_RESPONSE_PATTERNS]

# Additional low-quality patterns
LOW_QUALITY_PATTERNS = [
    r"^LGTM",
    r"^\+1",
    r"^Looks good",
    r"^Ship it",
    r"^Approved",
    r"^It's a draft",
    r"^draft version",
    r"^WIP",
    r"^TODO",
]

LOW_QUALITY_COMPILED = [re.compile(p, re.IGNORECASE) for p in LOW_QUALITY_PATTERNS]


def is_author_response(comment: str) -> bool:
    """Check if the comment is an author response rather than a review."""
    for pattern in AUTHOR_PATTERNS_COMPILED:
        if pattern.search(comment):
            return True
    return False


def is_low_quality(comment: str) -> bool:
    """Check if the comment is low quality."""
    for pattern in LOW_QUALITY_COMPILED:
        if pattern.search(comment):
            return True
    return False


def clean_dataset():
    """Clean the training dataset."""
    
    print(f"Loading {INPUT_FILE}...")
    
    examples = []
    with open(INPUT_FILE) as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples")
    
    # Filter
    cleaned = []
    removed = {"author_response": 0, "low_quality": 0, "too_short": 0}
    
    for ex in examples:
        # Get the assistant's response
        messages = ex.get("messages", [])
        assistant_msg = None
        for msg in messages:
            if msg["role"] == "assistant":
                assistant_msg = msg["content"]
                break
        
        if not assistant_msg:
            removed["too_short"] += 1
            continue
        
        # Check for author responses
        if is_author_response(assistant_msg):
            removed["author_response"] += 1
            continue
        
        # Check for low quality
        if is_low_quality(assistant_msg):
            removed["low_quality"] += 1
            continue
        
        # Check minimum length
        if len(assistant_msg) < 50:
            removed["too_short"] += 1
            continue
        
        cleaned.append(ex)
    
    print(f"\nCleaning complete:")
    print(f"  Original: {len(examples)}")
    print(f"  Cleaned: {len(cleaned)}")
    print(f"  Removed: {sum(removed.values())}")
    print(f"    - Author responses: {removed['author_response']}")
    print(f"    - Low quality: {removed['low_quality']}")
    print(f"    - Too short: {removed['too_short']}")
    
    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for ex in cleaned:
            f.write(json.dumps(ex) + "\n")
    
    print("Done!")
    
    # Show a few examples of what was removed
    print("\n--- Examples of removed author responses ---")
    count = 0
    with open(INPUT_FILE) as f:
        for line in f:
            ex = json.loads(line)
            for msg in ex.get("messages", []):
                if msg["role"] == "assistant":
                    if is_author_response(msg["content"]):
                        print(f"  • {msg['content'][:100]}...")
                        count += 1
                        if count >= 5:
                            break
            if count >= 5:
                break
    
    return cleaned


if __name__ == "__main__":
    clean_dataset()
