# Clipboard Analysis Search Engine



# Clipboard Analysis Tool

This tool analyzes clipboard content using various search and analysis methods, including semantic search and hybrid search capabilities.

## Prerequisites

- Python 3.8+
- Docker
- OpenAI API key

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Start Elasticsearch:
```bash
docker run -d -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "xpack.security.http.ssl.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.12.1
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

4. Run the application:
```bash
streamlit run clipboard_analysis/alfred_clipboard_exploration.py
```

## Features

- Semantic search using embeddings
- Hybrid search combining semantic and keyword search
- Clipboard content analysis
- Filtering and visualization of clipboard data

## Architecture

The tool uses:
- Streamlit for the web interface
- Elasticsearch for hybrid search capabilities
- OpenAI embeddings for semantic search
- SQLite for clipboard data storage
