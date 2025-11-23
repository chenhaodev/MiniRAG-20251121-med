# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniRAG is a lightweight Retrieval-Augmented Generation (RAG) framework designed for small language models (SLMs). It uses heterogeneous graph indexing and topology-enhanced retrieval to achieve good RAG performance with minimal resources.

**Paper**: [arXiv:2501.06713](https://arxiv.org/abs/2501.06713)

## Common Commands

### Installation

```bash
# Using uv (recommended)
uv sync                # Install dependencies
uv sync --extra api    # With API server support

# Using pip
pip install -e .
pip install -e ".[api]"  # With API server support

# From PyPI
pip install minirag-hku
```

### Running

```bash
# Index a dataset
python ./reproduce/Step_0_index.py

# Run Q&A evaluation
python ./reproduce/Step_1_QA.py

# Start API server (requires [api] extras)
minirag-server

# API server with custom config
minirag-server --llm-binding ollama --llm-model mistral-nemo:latest --port 9721
```

### Code Quality

```bash
# Run pre-commit hooks
pre-commit run --all-files

# Format code (ruff)
ruff format .

# Lint with auto-fix (ignores E402 for import order)
ruff check --fix --ignore=E402 .
```

## Architecture

### Core Components

- **`minirag/minirag.py`** - Main `MiniRAG` class with `insert()`, `query()`, `delete_by_entity()` methods (async versions prefixed with `a`)
- **`minirag/operate.py`** - Text chunking, entity/relationship extraction, and query processing logic
- **`minirag/base.py`** - Abstract base classes: `BaseVectorStorage`, `BaseKVStorage`, `BaseGraphStorage`

### Storage Layer (`minirag/kg/`)

18+ storage backend implementations:
- **Graph**: Neo4j, NetworkX, MongoDB, PostgreSQL, Gremlin, AGE, Oracle, Weaviate
- **Vector**: NanoVectorDB, Milvus, ChromaDB, TiDB, PostgreSQL, Weaviate
- **Key-Value**: JsonKVStorage, MongoDB, PostgreSQL, Redis, Oracle, Weaviate

### LLM Integration (`minirag/llm/`)

12+ provider integrations: OpenAI, Azure OpenAI, Bedrock, Ollama, LoLLMs, LMDeploy, HuggingFace, Zhipu (GLM), SiliconCloud, Jina, Nvidia

### API Server (`minirag/api/`)

FastAPI-based server with Ollama-compatible endpoints. Swagger docs at `/docs` when running.

### Data Flow

```
Documents → Chunking → Entity/Relationship Extraction → Graph Indexing
Query → Entity/Chunk Retrieval → Context Aggregation → LLM Response
```

### Query Modes

- **`"mini"`** (default) - Lightweight mode optimized for SLMs
- **`"naive"`** - Simple chunk retrieval
- **`"light"`** - Lightweight topology-enhanced retrieval

## Key Patterns

- **Dataclass configuration**: `MiniRAG` and `QueryParam` use Python dataclasses
- **Async-first**: Core methods have async variants (`insert()` / `ainsert()`)
- **Lazy imports**: Optional dependencies loaded on demand in storage implementations
- **Logger**: `logging.getLogger("minirag")`

## Basic Usage Example

```python
from minirag import MiniRAG, QueryParam

rag = MiniRAG(working_dir, llm_model_func, embedding_func)
rag.insert(text)
result = rag.query("question", param=QueryParam(mode="mini"))
```
