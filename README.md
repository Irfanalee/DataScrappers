# Data Scrapers

A collection of specialized data scrapers for gathering training datasets from various sources.

## Overview

This project contains tools to efficiently scrape and process data from popular platforms and repositories. Each scraper is designed to extract high-quality, structured data suitable for fine-tuning machine learning models.

## Folders

### `github_repo_scrapper/`
Scrapes code review comments from high-quality GitHub repositories. Collects review feedback and code context from merged pull requests to create a dataset for fine-tuning LLMs on code review practices.

**Key Features:**
- Fetches merged PRs from curated Python repositories
- Extracts inline review comments with code diff context
- Filters low-quality comments (e.g., "LGTM", too short)
- Outputs structured JSON data ready for model training
- Rate-limited API calls to respect GitHub limits

**Getting Started:**
```bash
cd github_repo_scrapper
pip install -r requirements.txt
cp .env.example .env
# Add your GitHub token to .env
python scraper.py
```

**Output:** `data/all_examples.json` containing ~5-10k code review examples

## Requirements

- Python 3.8+
- Individual scraper requirements listed in respective folders

## Project Structure

```
DataScrappers/
├── README.md (this file)
├── github_repo_scrapper/
│   ├── README.md
│   ├── scraper.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── .gitignore
│   └── data/
```

## Future Scrapers

- Documentation scrapers
- Blog/article extractors
- Dataset collectors from public APIs

## License

See individual scraper directories for license information.
