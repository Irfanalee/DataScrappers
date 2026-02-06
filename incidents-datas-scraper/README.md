# DevOps Incident Responder - Data Collection

## Overview

Fine-tuned LLM that analyzes error logs and incidents, provides root cause analysis, and suggests fixes.

## Tech Stack Coverage

- Kubernetes
- Docker  
- Azure
- GCP
- Node.js
- Redis
- MongoDB
- Nginx
- Terraform
- PostgreSQL
- InfluxDB

## Data Sources

### 1. GitHub Issues (Primary)
Closed issues with solutions from major DevOps repos.

```bash
python scrape_github_issues.py
```

### 2. GitHub Discussions
Q&A discussions with accepted answers.

```bash
python scrape_github_discussions.py
```

## Setup

```bash
# Create directory
mkdir -p incident-responder
cd incident-responder

# Set GitHub token
export GITHUB_TOKEN="your_token_here"

# Install dependencies
pip install requests

# Run scrapers
python scrape_github_issues.py
python scrape_github_discussions.py
```

## Output Structure

```
data/
├── github_issues/
│   ├── kubernetes_issues.json
│   ├── docker_issues.json
│   ├── terraform_issues.json
│   └── all_issues.json
├── github_discussions/
│   └── all_discussions.json
└── processed/
    ├── train.jsonl
    └── eval.jsonl
```

## Data Format

Each example contains:
- `tech`: Technology (kubernetes, docker, etc.)
- `repo`: Source repository
- `problem`: Error description / issue body
- `solution`: Fix / resolution from comments
- `url`: Link to original issue

## Filtering

- Date: 2021-01-01 onwards (recent data only)
- Quality: Must have error indicators + solution
- Length: Problem > 100 chars, Solution > 50 chars

## Expected Yield

| Source | Est. Examples |
|--------|---------------|
| GitHub Issues | 2,000 - 4,000 |
| GitHub Discussions | 500 - 1,500 |
| **Total** | **2,500 - 5,500** |

## Next Steps

1. Run scrapers
2. Preprocess into training format
3. Add synthetic data if needed
4. Fine-tune model
