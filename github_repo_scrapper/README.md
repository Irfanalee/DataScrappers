# Code Review Scraper

Collects high-quality code review comments from GitHub PRs for fine-tuning an LLM.

## Setup

```bash
pip install -r requirements.txt
export GITHUB_TOKEN="your_github_token_here"
```

## Run

```bash
python scraper.py
```

## Output

- `data/<repo_name>.json` — Per-repo data
- `data/all_examples.json` — Combined dataset

## What It Does

1. Fetches merged PRs from high-quality Python repos
2. Extracts review comments with code context (diff hunks)
3. Filters out low-quality comments (LGTM, too short, non-Python)
4. Saves in JSON format ready for training

## Target Repos

- fastapi/fastapi
- pydantic/pydantic
- psf/requests
- encode/httpx
- astral-sh/ruff
- tiangolo/sqlmodel
- pallets/flask
- django/django
- pytorch/pytorch
- huggingface/transformers

## Estimated Output

~5-10k examples, 20-50MB total
