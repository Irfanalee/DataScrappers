"""
Synthetic Data Generator for Code Review Critic
Generates buggy code + expert reviews using Claude Haiku 3.5

Estimated cost: ~$1.50 for 1,500 examples
"""

import anthropic
import json
import random
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = "data/synthetic"
OUTPUT_FILE = f"{OUTPUT_DIR}/synthetic_examples.json"
TARGET_EXAMPLES = 1500
MODEL = "claude-3-5-haiku-latest"

# Rate limiting
REQUESTS_PER_MINUTE = 50
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE  # 1.2 seconds

# =============================================================================
# BUG TEMPLATES
# =============================================================================

BUG_TEMPLATES = [
    # 1. Null/None access
    {
        "category": "null_access",
        "description": "Accessing attribute on potentially None value",
        "templates": [
            {
                "code": '''def get_user_email(user_id):
    user = db.query(User).filter_by(id=user_id).first()
    return user.email''',
                "bug": "No null check before accessing .email",
                "vars": ["user", "email", "User", "user_id"]
            },
            {
                "code": '''def get_order_total(order_id):
    order = Order.objects.filter(id=order_id).first()
    return order.total_amount''',
                "bug": "No null check before accessing .total_amount",
                "vars": ["order", "total_amount", "Order", "order_id"]
            },
            {
                "code": '''def get_config_value(key):
    config = load_config().get(key)
    return config.value''',
                "bug": "No null check - .get() can return None",
                "vars": ["config", "value", "key"]
            },
        ]
    },
    # 2. Missing error handling
    {
        "category": "missing_error_handling",
        "description": "No try/except around operations that can fail",
        "templates": [
            {
                "code": '''def read_json_file(filepath):
    with open(filepath) as f:
        return json.load(f)''',
                "bug": "No handling for FileNotFoundError or JSONDecodeError",
                "vars": ["filepath", "json"]
            },
            {
                "code": '''def fetch_api_data(url):
    response = requests.get(url)
    return response.json()''',
                "bug": "No handling for network errors or invalid JSON",
                "vars": ["url", "requests", "response"]
            },
            {
                "code": '''def parse_int(value):
    return int(value)''',
                "bug": "No handling for ValueError if value isn't numeric",
                "vars": ["value", "int"]
            },
        ]
    },
    # 3. Resource leaks
    {
        "category": "resource_leak",
        "description": "Resource not properly closed",
        "templates": [
            {
                "code": '''def read_file(path):
    f = open(path, 'r')
    content = f.read()
    return content''',
                "bug": "File handle never closed - should use 'with' statement",
                "vars": ["path", "f", "content"]
            },
            {
                "code": '''def query_database(query):
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()''',
                "bug": "Database connection never closed",
                "vars": ["query", "conn", "cursor"]
            },
            {
                "code": '''def download_file(url, dest):
    response = urllib.request.urlopen(url)
    data = response.read()
    with open(dest, 'wb') as f:
        f.write(data)''',
                "bug": "URL connection not closed if error occurs during write",
                "vars": ["url", "dest", "response"]
            },
        ]
    },
    # 4. Index/Key errors
    {
        "category": "index_error",
        "description": "Accessing index/key that may not exist",
        "templates": [
            {
                "code": '''def get_first_item(items):
    return items[0]''',
                "bug": "No check if list is empty - IndexError risk",
                "vars": ["items"]
            },
            {
                "code": '''def get_nested_value(data):
    return data["config"]["database"]["host"]''',
                "bug": "No check if nested keys exist - KeyError risk",
                "vars": ["data", "config", "database", "host"]
            },
            {
                "code": '''def get_last_element(results):
    return results[-1]''',
                "bug": "No check if list is empty before accessing last element",
                "vars": ["results"]
            },
        ]
    },
    # 5. Type errors
    {
        "category": "type_error",
        "description": "Type mismatch or wrong type handling",
        "templates": [
            {
                "code": '''def calculate_average(numbers):
    return sum(numbers) / len(numbers)''',
                "bug": "ZeroDivisionError if numbers is empty",
                "vars": ["numbers"]
            },
            {
                "code": '''def concat_strings(a, b):
    return a + b''',
                "bug": "No type validation - will fail if non-strings passed",
                "vars": ["a", "b"]
            },
            {
                "code": '''def format_price(price):
    return f"${price:.2f}"''',
                "bug": "Will fail if price is None or non-numeric",
                "vars": ["price"]
            },
        ]
    },
    # 6. Logic errors
    {
        "category": "logic_error",
        "description": "Off-by-one or incorrect logic",
        "templates": [
            {
                "code": '''def is_valid_age(age):
    return age > 0 and age < 120''',
                "bug": "Should be >= and <= for boundary inclusivity, also no type check",
                "vars": ["age"]
            },
            {
                "code": '''def get_items_in_range(items, start, end):
    return items[start:end-1]''',
                "bug": "Off-by-one error - end-1 excludes the intended last item",
                "vars": ["items", "start", "end"]
            },
            {
                "code": '''def is_leap_year(year):
    return year % 4 == 0''',
                "bug": "Incomplete leap year logic - missing century exception",
                "vars": ["year"]
            },
        ]
    },
    # 7. Security issues
    {
        "category": "security",
        "description": "Security vulnerabilities",
        "templates": [
            {
                "code": '''def get_user_by_name(name):
    query = f"SELECT * FROM users WHERE name = '{name}'"
    return db.execute(query)''',
                "bug": "SQL injection vulnerability - use parameterized queries",
                "vars": ["name", "query", "users"]
            },
            {
                "code": '''def run_command(user_input):
    import os
    os.system(f"echo {user_input}")''',
                "bug": "Command injection vulnerability - user input in shell command",
                "vars": ["user_input", "os"]
            },
            {
                "code": '''API_KEY = "sk-1234567890abcdef"

def call_api():
    return requests.get(URL, headers={"Authorization": API_KEY})''',
                "bug": "Hardcoded API key - should use environment variables",
                "vars": ["API_KEY", "URL"]
            },
        ]
    },
    # 8. Concurrency issues
    {
        "category": "concurrency",
        "description": "Race conditions and thread safety",
        "templates": [
            {
                "code": '''counter = 0

def increment():
    global counter
    counter += 1''',
                "bug": "Race condition - counter increment is not atomic",
                "vars": ["counter"]
            },
            {
                "code": '''cache = {}

def get_or_compute(key, func):
    if key not in cache:
        cache[key] = func()
    return cache[key]''',
                "bug": "Race condition - check-then-act pattern is not thread-safe",
                "vars": ["cache", "key", "func"]
            },
        ]
    },
    # 9. Memory/Performance issues
    {
        "category": "performance",
        "description": "Memory leaks or inefficient code",
        "templates": [
            {
                "code": '''def read_large_file(path):
    with open(path) as f:
        return f.read()''',
                "bug": "Reads entire file into memory - use iteration for large files",
                "vars": ["path", "f"]
            },
            {
                "code": '''def find_duplicates(items):
    duplicates = []
    for i, item in enumerate(items):
        if item in items[i+1:]:
            duplicates.append(item)
    return duplicates''',
                "bug": "O(n²) complexity - use a set for O(n) duplicate detection",
                "vars": ["items", "duplicates"]
            },
            {
                "code": '''def build_string(items):
    result = ""
    for item in items:
        result += str(item) + ","
    return result''',
                "bug": "String concatenation in loop is O(n²) - use join()",
                "vars": ["items", "result"]
            },
        ]
    },
    # 10. API misuse
    {
        "category": "api_misuse",
        "description": "Incorrect use of standard library or frameworks",
        "templates": [
            {
                "code": '''def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")''',
                "bug": "No error handling for invalid date format",
                "vars": ["date_str", "datetime"]
            },
            {
                "code": '''async def fetch_all(urls):
    results = []
    for url in urls:
        results.append(await fetch(url))
    return results''',
                "bug": "Sequential awaits - use asyncio.gather() for parallel execution",
                "vars": ["urls", "results", "fetch"]
            },
            {
                "code": '''def update_dict(d, key, value):
    d[key] = value
    return d''',
                "bug": "Mutates input dict and returns it - confusing API, pick one pattern",
                "vars": ["d", "key", "value"]
            },
        ]
    },
    # 11. Boolean/Comparison errors  
    {
        "category": "comparison_error",
        "description": "Incorrect boolean logic or comparisons",
        "templates": [
            {
                "code": '''def is_empty(value):
    if value == None or value == "":
        return True
    return False''',
                "bug": "Use 'is None' not '== None', also consider using 'not value'",
                "vars": ["value"]
            },
            {
                "code": '''def check_range(x, low, high):
    return low < x < high''',
                "bug": "May want inclusive bounds (<=) depending on use case, unclear API",
                "vars": ["x", "low", "high"]
            },
            {
                "code": '''def is_valid(items):
    return items != None and len(items) > 0''',
                "bug": "Use 'is not None', could simplify to 'return bool(items)'",
                "vars": ["items"]
            },
        ]
    },
    # 12. Mutable default arguments
    {
        "category": "mutable_default",
        "description": "Mutable default argument pitfall",
        "templates": [
            {
                "code": '''def add_item(item, items=[]):
    items.append(item)
    return items''',
                "bug": "Mutable default argument - list persists between calls",
                "vars": ["item", "items"]
            },
            {
                "code": '''def create_user(name, roles=[], metadata={}):
    return {"name": name, "roles": roles, "metadata": metadata}''',
                "bug": "Mutable default arguments - use None and create inside function",
                "vars": ["name", "roles", "metadata"]
            },
        ]
    },
    # 13. Import/Dependency issues
    {
        "category": "import_issues",
        "description": "Import problems or circular dependencies",
        "templates": [
            {
                "code": '''from module import *

def process():
    return helper_function()''',
                "bug": "Wildcard import - unclear which names are imported, pollution risk",
                "vars": ["module", "helper_function"]
            },
            {
                "code": '''import pandas as pd
import numpy as np
import requests
import json
import os

def get_value():
    return json.loads("{}")''',
                "bug": "Unused imports (pandas, numpy, requests, os) - remove them",
                "vars": ["pandas", "numpy", "requests", "os"]
            },
        ]
    },
    # 14. Exception handling anti-patterns
    {
        "category": "exception_antipattern",
        "description": "Poor exception handling practices",
        "templates": [
            {
                "code": '''def safe_divide(a, b):
    try:
        return a / b
    except:
        return 0''',
                "bug": "Bare except catches everything including KeyboardInterrupt - too broad",
                "vars": ["a", "b"]
            },
            {
                "code": '''def process_data(data):
    try:
        result = transform(data)
        save(result)
        notify()
    except Exception as e:
        pass''',
                "bug": "Silently swallowing exceptions - at minimum log the error",
                "vars": ["data", "result"]
            },
            {
                "code": '''def load_config():
    try:
        return json.load(open("config.json"))
    except Exception as e:
        raise Exception("Config error")''',
                "bug": "Losing original exception context - use 'raise ... from e'",
                "vars": ["json", "config"]
            },
        ]
    },
    # 15. Return value issues
    {
        "category": "return_issues",
        "description": "Inconsistent or missing return values",
        "templates": [
            {
                "code": '''def find_item(items, target):
    for item in items:
        if item == target:
            return item''',
                "bug": "Implicit None return if not found - make it explicit",
                "vars": ["items", "target", "item"]
            },
            {
                "code": '''def validate(data):
    if not data:
        return False
    if "id" not in data:
        return
    return True''',
                "bug": "Inconsistent returns - None vs False vs True",
                "vars": ["data"]
            },
            {
                "code": '''def get_status(code):
    if code == 200:
        return "OK"
    elif code == 404:
        return "Not Found"
    elif code == 500:
        return "Error"''',
                "bug": "No default return for unknown codes - returns None implicitly",
                "vars": ["code"]
            },
        ]
    },
]

# =============================================================================
# VARIATION GENERATORS
# =============================================================================

FUNCTION_NAMES = [
    "process", "handle", "get", "fetch", "load", "parse", "validate",
    "check", "compute", "calculate", "transform", "convert", "extract",
    "find", "search", "filter", "update", "save", "create", "delete",
    "init", "setup", "run", "execute", "build", "generate", "format"
]

VARIABLE_NAMES = [
    "data", "result", "value", "item", "record", "entry", "obj",
    "response", "request", "payload", "content", "body", "params",
    "config", "settings", "options", "args", "kwargs", "context"
]

ENTITY_NAMES = [
    "User", "Order", "Product", "Customer", "Account", "Transaction",
    "Session", "Request", "Event", "Message", "Task", "Job", "Item"
]


def generate_variation(template: dict) -> dict:
    """Generate a variation of a code template."""
    code = template["code"]
    
    # Random function name substitution
    if "def " in code:
        for func in FUNCTION_NAMES[:5]:  # Check common function starts
            if f"def {func}" in code or f"def get_" in code:
                new_func = random.choice(FUNCTION_NAMES)
                # Simple substitution - keeps it readable
                break
    
    return {
        "code": code,
        "bug": template["bug"],
    }


# =============================================================================
# CLAUDE API
# =============================================================================

def generate_review(client: anthropic.Anthropic, code: str, bug_hint: str, category: str) -> str:
    """Generate a code review using Claude Haiku."""
    
    prompt = f"""You are an expert Python code reviewer. Review this code and provide constructive feedback.

The code has an issue related to: {category}
Hint: {bug_hint}

Code:
```python
{code}
```

Write a concise, actionable code review comment (2-4 sentences). Be specific about:
1. What the problem is
2. What could go wrong
3. How to fix it

Do NOT use phrases like "Great code!" or "Nice work!". Be direct and technical.
Do NOT include code blocks in your response - just explain in prose.
Do NOT start with "The code" or "This code" - vary your opening."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text.strip()


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate_synthetic_dataset(target_count: int = TARGET_EXAMPLES):
    """Generate synthetic training examples."""
    
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    examples = []
    total_templates = sum(len(cat["templates"]) for cat in BUG_TEMPLATES)
    examples_per_template = target_count // total_templates + 1
    
    print("=" * 60)
    print("SYNTHETIC DATA GENERATOR")
    print("=" * 60)
    print(f"Target examples: {target_count}")
    print(f"Bug categories: {len(BUG_TEMPLATES)}")
    print(f"Total templates: {total_templates}")
    print(f"Examples per template: ~{examples_per_template}")
    print(f"Model: {MODEL}")
    print()
    
    stats = {"total": 0, "by_category": {}}
    
    for category_data in BUG_TEMPLATES:
        category = category_data["category"]
        description = category_data["description"]
        templates = category_data["templates"]
        
        print(f"\n[{category}] {description}")
        stats["by_category"][category] = 0
        
        for template in templates:
            # Generate multiple variations per template
            for i in range(examples_per_template):
                if stats["total"] >= target_count:
                    break
                
                try:
                    variation = generate_variation(template)
                    
                    review = generate_review(
                        client,
                        variation["code"],
                        variation["bug"],
                        category
                    )
                    
                    # Format as training example
                    example = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert code reviewer. Analyze the provided Python code and give constructive, specific feedback. Focus on bugs, potential issues, code quality, and improvements. Be direct and actionable."
                            },
                            {
                                "role": "user", 
                                "content": f"Review this Python code:\n\n```python\n{variation['code']}\n```"
                            },
                            {
                                "role": "assistant",
                                "content": review
                            }
                        ],
                        "_meta": {
                            "source": "synthetic",
                            "category": category,
                            "bug_type": variation["bug"]
                        }
                    }
                    
                    examples.append(example)
                    stats["total"] += 1
                    stats["by_category"][category] += 1
                    
                    if stats["total"] % 50 == 0:
                        print(f"  Generated {stats['total']}/{target_count} examples...")
                        # Save checkpoint
                        save_examples(examples, stats)
                    
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                    
                except Exception as e:
                    print(f"  Error generating example: {e}")
                    time.sleep(2)  # Back off on error
                    continue
            
            if stats["total"] >= target_count:
                break
        
        if stats["total"] >= target_count:
            break
    
    # Final save
    save_examples(examples, stats)
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total examples: {stats['total']}")
    print(f"By category: {json.dumps(stats['by_category'], indent=2)}")
    print(f"Saved to: {OUTPUT_FILE}")
    
    return examples, stats


def save_examples(examples: list, stats: dict):
    """Save examples to file."""
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "model": MODEL,
            "stats": stats,
            "examples": examples
        }, f, indent=2)


def convert_to_training_format(input_file: str = OUTPUT_FILE, output_dir: str = "data/processed"):
    """Convert synthetic examples to JSONL and merge with existing data."""
    
    print(f"\nConverting {input_file} to training format...")
    
    with open(input_file) as f:
        data = json.load(f)
    
    synthetic_examples = data["examples"]
    print(f"Loaded {len(synthetic_examples)} synthetic examples")
    
    # Save as separate JSONL
    synthetic_jsonl = f"{output_dir}/synthetic.jsonl"
    with open(synthetic_jsonl, "w") as f:
        for ex in synthetic_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved to {synthetic_jsonl}")
    
    # Merge with existing training data
    existing_train = f"{output_dir}/train.jsonl"
    merged_train = f"{output_dir}/train_with_synthetic.jsonl"
    
    existing_examples = []
    with open(existing_train) as f:
        for line in f:
            existing_examples.append(json.loads(line))
    
    print(f"Existing training examples: {len(existing_examples)}")
    
    # Combine and shuffle
    combined = existing_examples + synthetic_examples
    random.seed(42)
    random.shuffle(combined)
    
    with open(merged_train, "w") as f:
        for ex in combined:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Merged dataset: {len(combined)} examples")
    print(f"Saved to {merged_train}")
    
    return merged_train


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic code review data")
    parser.add_argument("--count", type=int, default=TARGET_EXAMPLES, help="Number of examples to generate")
    parser.add_argument("--convert-only", action="store_true", help="Only convert existing synthetic data")
    
    args = parser.parse_args()
    
    if args.convert_only:
        convert_to_training_format()
    else:
        generate_synthetic_dataset(args.count)
        print("\nTo merge with training data, run:")
        print("  python generate_synthetic.py --convert-only")
