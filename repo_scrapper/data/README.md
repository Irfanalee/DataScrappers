---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - code-review
  - python
  - github
  - llm-training
size_categories:
  - 1K<n<10K
---

# GitHub Code Review Comments Dataset

A dataset of real code review comments scraped from popular Python GitHub repositories.

## Dataset Description

This dataset contains code review comments from pull requests across major Python open-source projects. Each example includes:
- The code being reviewed
- The reviewer's feedback
- Metadata (repository, PR number, URL)

## Dataset Structure

### Files
- `train_cleaned.jsonl` - Training set (~8,275 examples)
- `eval.jsonl` - Evaluation set (~780 examples)

## Data Format

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert code reviewer..."},
    {"role": "user", "content": "Review this Python code:\n\n```python\n...```"},
    {"role": "assistant", "content": "Feedback..."}
  ],
  "_meta": {
    "repo": "pydantic/pydantic",
    "pr_number": 10597,
    "url": "https://github.com/..."
  }
}
```

## Source Repositories

- huggingface/transformers
- apache/airflow
- pydantic/pydantic
- django/django
- scrapy/scrapy
- celery/celery
- psf/black
- fastapi/fastapi
- encode/httpx
- psf/requests
- tiangolo/sqlmodel
- astral-sh/ruff
- pallets/flask
- ansible/ansible
- python/cpython
- pytorch/pytorch

## Intended Use

- Fine-tuning LLMs for code review tasks
- Training code quality assistants
- Research on code review patterns

## Limitations

- Python-focused (limited other languages)
- Reflects patterns and biases of source repositories
- Some comments may be conversational rather than actionable

## License

Apache 2.0
