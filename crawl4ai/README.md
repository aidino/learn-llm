## Installation & Setup (2023 Edition)

### 1. Basic Installation

```bash
pip install crawl4ai
```

### 2. Run the Setup Command

Installs or updates required Playwright browsers (Chromium, Firefox, etc.) - Performs OS-level checks (e.g., missing libs on Linux) - Confirms your environment is ready to crawl

```bash
crawl4ai-setup

```

### 3. Diagnostics

```bash
crawl4ai-doctor

```

### 4. Advanced Installation (Optional)

- Text Clustering (Torch)

Installs PyTorch-based features (e.g., cosine similarity or advanced semantic chunking).


```bash
pip install crawl4ai[torch]
crawl4ai-setup

```

- Transformers

Adds Hugging Face-based summarization or generation strategies.

```bash
pip install crawl4ai[transformer]
crawl4ai-setup

```

- All Features

```bash
pip install crawl4ai[all]
crawl4ai-setup
```

- (Optional) Pre-Fetching Models

```bash
crawl4ai-download-models

```

## Crawl4AI CLI Guide

### Basic Usage

```bash
# Basic crawling
crwl https://example.com

# Get markdown output
crwl https://example.com -o markdown -O output/example.com.md

# Verbose JSON output with cache bypass
crwl https://example.com -o json -v --bypass-cache

# See usage examples
crwl --example
```