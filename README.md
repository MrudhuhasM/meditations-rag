
This is a portfolio project to showcase my work to potential employers.

In this project i am building an Agentic RAG application 
My source for this project is Meditations by Marcus Aurelius, this is philosophical text that contains personal writings by Marcus Aurelius, Roman Emperor from 161 to 180 AD, in which he outlines his Stoic philosophy.
The application will be able to answer questions about the text, provide summaries, and generate insights based on the content of the book.

I will use the following technologies in this project:

LLM: OpenAI, OpenRouter, Local(Open AI), Gemini
Vector DB: Qdrant
Embedding: OpenAI, Local(Open AI), Gemini
Agent Framework: Langgraph

## LLM Providers

This project supports multiple LLM providers:

- **OpenAI**: Industry-standard models (GPT-4, GPT-4o, GPT-4o-mini)
- **OpenRouter**: Unified access to 100+ models from multiple providers (OpenAI, Anthropic, Google, Meta, etc.)
- **Gemini**: Google's latest models with competitive pricing
- **Local LLM**: Privacy-focused local models (Ollama, etc.)

All providers include:
- ✅ Automatic rate limiting
- ✅ Retry logic with exponential backoff
- ✅ Structured output support (Pydantic models)
- ✅ Comprehensive error handling
- ✅ Easy configuration via environment variables

## Ingestion Pipeline

The ingestion pipeline processes documents and stores them in a vector database for retrieval:

**Pipeline Stages:**
1. **Document Loading**: Load documents using PyMuPDF (PDF support)
2. **Semantic Chunking**: Intelligently split documents using semantic similarity
3. **Metadata Extraction**: Extract structured metadata (questions, keywords, topics, entities, philosophical concepts)
4. **Embedding Generation**: Generate vector embeddings for chunks and questions
5. **Vector Store**: Upsert chunks and questions to Qdrant vector database

**Features:**
- ✅ Batch processing with configurable batch sizes
- ✅ Async/await for efficient I/O operations
- ✅ Optional metadata extraction via LLM
- ✅ Dual collection strategy (chunks + questions for hybrid retrieval)
- ✅ Comprehensive logging and error handling

## Retrieval Service

The retrieval service implements a **hybrid retrieval strategy** combining dense retrieval from both chunks and questions:

**Retrieval Strategy:**
1. **Dense Retrieval from Chunks**: Semantic search over chunk embeddings (direct content matching)
2. **Dense Retrieval from Questions**: Semantic search over question embeddings (conceptual matching)
3. **Score Fusion**: Linear combination using configurable alpha parameter

**Score Fusion Formula:**
```
final_score = chunk_score + α * question_score
```

Where `α` (alpha) controls the weight of question-based retrieval (0.0 to 1.0).

**Features:**
- ✅ Three retrieval modes: Hybrid (default), Chunk-only, Question-only
- ✅ Configurable score fusion with alpha parameter
- ✅ Parallel retrieval for performance
- ✅ LLM-ready context formatting
- ✅ Comprehensive metadata in results
- ✅ Flexible top-k configuration
- ✅ Production-grade error handling

**Quick Example:**
```python
from meditations_rag.services import RetrievalService

# Initialize
retrieval_service = RetrievalService(
    vector_store=vector_store,
    embedding_service=embedding_service,
    alpha=0.3,  # Score fusion weight
    top_k=5     # Number of results
)

# Retrieve
results = await retrieval_service.retrieve(
    query="How should I prepare for difficult people?"
)

# Use results
for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"Text: {result.text[:200]}...")
```

**Documentation:**
- See [docs/RETRIEVAL_SERVICE.md](docs/RETRIEVAL_SERVICE.md) for comprehensive documentation
- See [docs/RETRIEVAL_QUICK_REF.md](docs/RETRIEVAL_QUICK_REF.md) for quick reference
- See [examples/retrieval_example.py](examples/retrieval_example.py) for working examples
