# Agentic RAG: Meditations by Marcus Aurelius

This is a portfolio project showcasing an **Agentic RAG (Retrieval-Augmented Generation)** application.
The system allows users to query *Meditations* by Marcus Aurelius, providing accurate answers, summaries, and philosophical insights grounded in the text.

Unlike standard RAG, this project employs an **Agentic workflow** using **LangGraph**, enabling the system to reason about queries, perform multi-step retrieval, self-correct, and evaluate its own answers.

## 🚀 Key Features

### 🤖 Agentic RAG Workflow (LangGraph)
The core of the application is a stateful agent that orchestrates the retrieval and generation process:
- **Controller Node**: Analyzes the user query and decides the next action (Search, Answer, Clarify, or Switch Model).
- **Retriever Node**: Executes search queries using the hybrid retrieval service.
- **Generator Node**: Synthesizes answers from retrieved context.
- **Evaluator Node**: Checks the generated answer for groundedness and relevance. If the answer is poor, it loops back to the controller with feedback.

### 🔍 Advanced Retrieval Engine
- **Hybrid Search**: Combines dense vector retrieval from document chunks and generated questions.
- **Score Fusion**: Weighted combination of chunk scores and question scores (`final_score = chunk_score + α * question_score`).
- **Metadata Filtering**: Leveraging extracted metadata (topics, chapters) for precise filtering.

### 🛡️ Production Engineering
- **Rate Limiting**: Implemented using `slowapi` with configurable strategies (Token Bucket) to ensure service stability.
- **Security**: API Key authentication middleware protecting sensitive endpoints.
- **Robust Error Handling**: Standardized exception handling and logging for easier debugging and reliability.
- **Observability**: Structured logging configuration for tracking request flows and system health.

### ⚡ High-Performance Ingestion
- **Semantic Chunking**: Intelligently splits text based on semantic similarity.
- **Metadata Extraction**: Uses LLMs to extract topics, keywords, and philosophical concepts for each chunk.
- **Synthetic Questions**: Generates potential user questions for each chunk to improve retrieval matching.

### 🛠️ Modern Tech Stack
- **Frameworks**: LangGraph, LlamaIndex, FastAPI
- **Vector DB**: Qdrant
- **LLMs**: OpenAI, OpenRouter, Gemini, Local (llama.cpp, VLLM)
- **Tooling**: `uv` for dependency management, Docker for deployment
- **Reliability**: SlowAPI (Rate Limiting), Pydantic (Validation)

### 🌐 Local LLM Support
The application supports running with local LLMs via OpenAI-compatible APIs:
- **llama.cpp**: For local model serving
- **VLLM**: For high-performance inference on Kubernetes/GKE
- Configure via environment variables to point to your local server

### ☁️ Cloud LLM Support
In addition to local LLMs, the application seamlessly integrates with cloud-based models for flexibility and scalability:
- **OpenAI**: GPT-4 and GPT-3.5-turbo for high-quality generation and reasoning.
- **Gemini**: Google's multimodal models for advanced text understanding and synthesis.
- **OpenRouter**: A unified API for accessing multiple providers (e.g., Anthropic Claude, Mistral) with load balancing and failover.

Configuration is handled via environment variables, allowing easy switching between providers without code changes. This hybrid approach ensures optimal performance, cost-efficiency, and access to the latest model capabilities.

## 🏗️ Architecture

### Ingestion Pipeline
1. **Load**: PDF loading with PyMuPDF.
2. **Chunk**: Semantic chunking.
3. **Enrich**: Extract metadata & generate synthetic questions.
4. **Embed**: Generate embeddings (OpenAI/Gemini/Local).
5. **Store**: Upsert to Qdrant.

### Agentic Pipeline
The agentic pipeline is built with LangGraph:
1. **Controller** receives query.
2. **Controller** decides to search.
3. **Retriever** fetches documents.
4. **Controller** reviews docs, decides to answer or search again.
5. **Generator** drafts answer.
6. **Evaluator** critiques answer.
7. **Final Answer** returned to user.

## 🛠️ Getting Started

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- [uv](https://github.com/astral-sh/uv) (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd meditations-rag
   ```

2. **Set up environment variables**
   Copy `.env.example` to `.env` and configure your API keys (OpenAI, Qdrant, etc.).
   ```bash
   cp .env.example .env
   ```

3. **Build the Docker image**
   ```bash
   docker-compose build
   ```

### Running the Application

1. **Start Infrastructure (Qdrant)**
   ```bash
   docker-compose up -d qdrant
   ```

2. **Run Ingestion (First time only)**
   Process the book and populate the vector database.
   ```bash
   docker-compose run --rm ingest
   ```

3. **Start the API Server**
   ```bash
   docker-compose up api
   ```
   The API will be available at `http://localhost:8000`.
   - Swagger UI: `http://localhost:8000/docs`
   - Agentic Query: `POST /agentic/query`

## 📂 Project Structure

```
src/meditations_rag/
├── api/            # FastAPI application, routers & security
├── core/           # Core logic (LLM, Embedding, Rate Limiting)
├── pipelines/      # RAG & Ingestion pipelines (LangGraph)
├── services/       # Business logic (Loader, Chunker, Retrieval)
└── main.py         # Ingestion entry point
examples/           # Standalone examples for each component
scripts/
└── ingest.py       # Standalone ingestion script
```

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:
- [Retrieval Service](docs/RETRIEVAL_SERVICE.md)
- [RAG Pipeline](docs/RAG_PIPELINE.md)
- [Metadata Extraction](docs/metadata_extraction.md)

Check the `examples/` directory for runnable scripts demonstrating individual components like rate limiting, error handling, and specific LLM integrations.
