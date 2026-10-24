# Enterprise RAG Platform

A production-oriented Retrieval-Augmented Generation (RAG) platform built with **FastAPI, PostgreSQL, pgvector, and local LLMs**. The project explores reliable document ingestion, semantic retrieval, reranking, grounded answer generation, and evaluation.

> **Project status:** Under active development. See the roadmap below for implementation progress.

## Overview

Large language models can generate incorrect or unsupported answers when they lack access to relevant domain-specific information. Retrieval-Augmented Generation addresses this by retrieving relevant document passages and using them as context for answer generation.

This project implements a modular RAG backend designed around clear service boundaries, database persistence, configurable components, and automated testing.

## Key Features

- **REST API:** FastAPI endpoints for health checks, document management, search, and answer generation.
- **Document ingestion:** Store uploaded document content and track processing status.
- **Text chunking:** Split documents into smaller passages for retrieval.
- **Embeddings:** Convert text into vector representations.
- **Vector search:** Use PostgreSQL with pgvector for semantic similarity search.
- **Relevance filtering:** Apply configurable distance thresholds to retrieved chunks.
- **Reranking:** Reorder retrieval candidates using a reranking model.
- **LLM integration:** Support local model inference through Ollama.
- **Database migrations:** Manage schema changes with Alembic.
- **Observability:** Add structured logging and request correlation for troubleshooting.
- **Automated tests:** Test API behavior, persistence, retrieval, and related services.

Feature availability and configuration may vary as development continues.

## Technology Stack

| Component | Technology |
|---|---|
| API framework | FastAPI |
| Language | Python 3.12 |
| Database | PostgreSQL |
| Vector search | pgvector |
| ORM | SQLAlchemy |
| Database migrations | Alembic |
| LLM runtime | Ollama |
| Testing | pytest |
| Code quality | Ruff |
| Environment and dependencies | uv |
| Local infrastructure | Docker Compose |

## Architecture

```text
Client
  |
  v
FastAPI Application
  |
  +--> Health API
  +--> Document API
  +--> Search API
  +--> Answer API
          |
          v
     Application Services
          |
          +--> Document ingestion and chunking
          +--> Embedding generation
          +--> Vector retrieval
          +--> Relevance filtering
          +--> Reranking
          +--> Answer generation
          |
          +--> PostgreSQL + pgvector
          |
          +--> Ollama
```

### Document ingestion flow

1. Receive a document through the API.
2. Extract or accept its text.
3. Split the text into chunks.
4. Generate embeddings for the chunks.
5. Persist document and chunk data.
6. Update the processing status.

### Question-answering flow

1. Receive a user question.
2. Generate an embedding for the question.
3. Retrieve relevant document chunks.
4. Filter results by the configured relevance threshold.
5. Rerank candidates when enabled.
6. Use the retrieved context to generate an answer.

## Repository Structure

Enterprise_RAG_System/
├── migrations/
│   └── versions/
├── src/
│   └── enterprise_rag_platform/
│       ├── api/
│       │   ├── routes/
│       │   │   ├── answers.py
│       │   │   ├── documents.py
│       │   │   ├── health.py
│       │   │   └── search.py
│       │   ├── exception_handlers.py
│       │   └── middleware.py
│       ├── core/
│       │   └── logging.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── performance.py
│       │   ├── performance_runner.py
│       │   ├── quality_gate.py
│       │   └── runner.py
│       ├── models/
│       ├── schemas/
│       ├── services/
│       │   ├── chunking.py
│       │   ├── chunk_persistence.py
│       │   ├── embedding.py
│       │   ├── ingestion.py
│       │   ├── llm.py
│       │   ├── rag.py
│       │   ├── reranking.py
│       │   └── retrieval.py
│       ├── config.py
│       ├── database.py
│       └── main.py
├── tests/
│   ├── integration/
│   └── unit/
├── data/                 # Local data; ignored by Git
├── reports/              # Generated reports; ignored by Git
├── .env.example
├── .gitignore
├── alembic.ini
├── compose.yaml
├── pyproject.toml
├── uv.lock
└── README.md
``` id="z6qvkd"


The tree illustrates the intended high-level organization; update it to match the actual repository files.

## Prerequisites

Install the following before running the application:

- Python 3.12
- uv
- Docker Desktop with Docker Compose
- Git
- Ollama, if using local LLM inference

## Local Setup

### 1. Clone the repository

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Enterprise_RAG_System
```

Replace the placeholder with your repository URL.

### 2. Install dependencies

```powershell
uv sync
```

### 3. Configure environment variables

Create your local `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Review the values in `.env` and adjust them for your local environment. Never commit credentials or private configuration.

### 4. Start PostgreSQL and pgvector

```powershell
docker compose up -d
```

Ensure the database port and credentials in your configuration match the Compose setup.

### 5. Apply database migrations

```powershell
uv run alembic upgrade head
```

### 6. Start the API

Use the application's configured development entry point. For example, if `enterprise_rag_platform.main:app` is the import path in your project:

```powershell
uv run uvicorn enterprise_rag_platform.main:app --reload --app-dir src
```

Check the actual health endpoint and API prefix in your configuration before making requests.

## Testing

Run an individual test during development:

```powershell
uv run pytest tests/integration/test_retrieval.py::test_search_chunks_ranks_by_cosine_distance -v
```

Run a specific test file:

```powershell
uv run pytest tests/integration/test_retrieval.py -v
```

Run the full test suite before submitting a change:

```powershell
uv run pytest
```

Check code quality:

```powershell
uv run ruff check .
```

## API Exploration

Once the API is running, open FastAPI's interactive documentation at the configured Swagger UI route, typically:

```text
/docs
```

Use the documented endpoints to explore health checks, document ingestion, semantic search, and answer generation.

## Configuration and Security

- Keep secrets in environment variables or a local `.env` file.
- Commit `.env.example`, not `.env`.
- Do not expose internal documents or database credentials in logs.
- Validate uploaded content and enforce configured upload-size limits.
- Apply authentication and authorization before exposing the service to untrusted users.
- Restrict retrieved context to the documents the requesting user is authorized to access.
- Treat retrieved document content as untrusted input when generating model responses.

This is a development project; production deployment requires appropriate authentication, authorization, monitoring, resource limits, and security review.

## Development Roadmap

- [ ] FastAPI application foundation
- [ ] PostgreSQL and pgvector integration
- [ ] SQLAlchemy models and Alembic migrations
- [ ] Document and chunk persistence
- [ ] Embedding-based semantic retrieval
- [ ] Retrieval and reranking service integration
- [ ] API routes for core workflows

Update these checkboxes as each capability is verified.

## Design Goals

- Modular, maintainable service boundaries
- Configuration-driven behavior
- Testable components and dependency injection
- Reliable persistence and document processing
- Relevant, context-grounded answers
- Observable retrieval and generation workflows
- Incremental evolution toward production readiness

## Contributing

1. Create a feature branch.
2. Make a focused change.
3. Run the relevant tests.
4. Run Ruff.
5. Document configuration or API changes.
6. Submit a pull request with a concise description.

## License

No license has been specified yet. Add a `LICENSE` file before publishing the project if you want to define how others may use, modify, and distribute it.
