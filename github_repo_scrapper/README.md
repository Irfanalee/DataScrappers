# Code Review Scraper

Collects high-quality code review comments from GitHub PRs for fine-tuning an LLM.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root directory:
```bash
cp .env.example .env
```

Or manually create `.env` with your GitHub token:
```
GITHUB_TOKEN=your_github_token_here
```

> **Note:** Get your GitHub token from [GitHub Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens). You need `repo` scope to access private PR data.

## Run

```bash
python scraper.py
```

## Output

- `data/<repo_name>.json` — Per-repo data
- `data/all_examples.json` — Combined dataset

## Configuration

The scraper uses a `.env` file to load sensitive data like your GitHub token. The `.env` file is automatically ignored by Git (see `.gitignore`), so your token stays secure.

**Environment Variables:**
- `GITHUB_TOKEN` (required) — Personal access token for GitHub API authentication

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
