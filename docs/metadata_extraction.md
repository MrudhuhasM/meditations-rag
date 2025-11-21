# Metadata Extraction Service

## Overview

The Metadata Extraction Service generates structured metadata from document chunks to significantly enhance RAG retrieval quality. By extracting semantic information beyond simple embeddings, we enable:

- **Question-based retrieval**: Match user queries to questions that chunks can answer
- **Keyword filtering**: Lexical search augmentation for semantic search
- **Topical filtering**: Category-based retrieval refinement
- **Entity recognition**: Fact-based query routing
- **Concept mapping**: Philosophical theme navigation

## Architecture

### Design Patterns

1. **Strategy Pattern**: `MetadataExtractionStrategy` allows different extraction approaches
2. **Factory Pattern**: `create_llm()` abstracts LLM provider instantiation
3. **Dependency Inversion**: Service depends on `LLMBase` abstraction, not concrete implementations
4. **Single Responsibility**: Service solely handles metadata extraction, not storage or retrieval

### Core Components

```
MetadataExtractorService
├── LLMBase (via Factory)
│   ├── OpenAILLM
│   ├── GeminiLLM
│   └── LocalLLM
├── MetadataExtractionStrategy
└── ChunkMetadata (Pydantic Model)
```

## Metadata Schema

### ChunkMetadata

```python
class ChunkMetadata(BaseModel):
    questions: List[str]              # 5 questions chunk can answer
    keywords: List[str]                # 8-12 distinctive keywords
    topic: MeditationsTopic            # Single primary topic (enum)
    entities: List[str]                # Named entities (people, places, etc.)
    philosophical_concepts: List[str]  # Stoic/philosophical concepts
    stoic_practices: List[str]         # Specific Stoic exercises mentioned
```

### Predefined Topics (MeditationsTopic Enum)

Based on core themes in Marcus Aurelius' Meditations:

1. **Virtue and Character** - Moral excellence, integrity, honor
2. **Death and Mortality** - Impermanence, acceptance of death
3. **Reason and Rationality** - Logic, clear thinking, judgment
4. **Duty and Service** - Obligation, public service, responsibility
5. **Control and Acceptance** - Dichotomy of control, acceptance
6. **Nature and Universe** - Cosmic order, natural law
7. **Justice and Community** - Fairness, social responsibility
8. **Self-Discipline** - Restraint, moderation, temperance
9. **Pleasure and Pain** - Hedonism, suffering, indifference
10. **Time and Impermanence** - Present moment, change
11. **Anger and Emotions** - Passion management, emotional control
12. **Wisdom and Learning** - Knowledge, philosophy, education
13. **Simplicity and Humility** - Modesty, plain living
14. **Gratitude and Appreciation** - Thankfulness, recognition
15. **Leadership and Governance** - Rulership, authority, power

## Usage

### Basic Usage

```python
import asyncio
from meditations_rag.core.llm import create_llm
from meditations_rag.services.metadata import MetadataExtractorService

async def extract_metadata():
    # Create LLM (uses settings.rag.llm_provider)
    llm = create_llm()

    # Initialize extractor
    extractor = MetadataExtractorService(
        llm=llm,
        batch_size=10,
        max_concurrent=5
    )

    # Extract from single chunk
    chunk_text = "Your document chunk text here..."
    metadata = await extractor.extract_metadata(chunk_text)

    print(f"Topic: {metadata.topic}")
    print(f"Questions: {metadata.questions}")
    print(f"Keywords: {metadata.keywords}")
```

### Batch Processing

```python
async def extract_batch():
    llm = create_llm()
    extractor = MetadataExtractorService(llm=llm)

    chunks = ["chunk1 text", "chunk2 text", "chunk3 text"]

    # Returns List[ChunkMetadata | None]
    # None for failed extractions
    metadata_list = await extractor.extract_batch_metadata(chunks)

    for chunk, meta in zip(chunks, metadata_list):
        if meta:
            print(f"✓ Extracted: {meta.topic}")
        else:
            print("✗ Extraction failed")
```

### Integration with Ingestion Pipeline

```python
from meditations_rag.services.loader import DocumentLoaderService
from meditations_rag.services.chunker import ChunkerService
from meditations_rag.services.metadata import MetadataExtractorService
from meditations_rag.pipelines.ingest import IngestPipeline
from meditations_rag.core.llm import create_llm

# Setup services
loader = DocumentLoaderService()
chunker = ChunkerService(embed_model=...)
llm = create_llm()
metadata_extractor = MetadataExtractorService(llm=llm)

# Create pipeline
pipeline = IngestPipeline(
    loader=loader,
    chunk_service=chunker,
    metadata_extractor=metadata_extractor
)

# Ingest and extract
chunks, metadata_list = await pipeline.ingest("path/to/document.pdf")
```

### Custom Extraction Strategy

```python
from meditations_rag.services.metadata import MetadataExtractionStrategy

class CustomStrategy(MetadataExtractionStrategy):
    def build_prompt(self, chunk_text: str, available_topics: List[str]) -> str:
        # Custom prompt engineering
        return f"""
        Custom extraction instructions for: {chunk_text}
        Topics: {available_topics}
        """

# Use custom strategy
extractor = MetadataExtractorService(
    llm=llm,
    strategy=CustomStrategy()
)
```

## Configuration

### Environment Variables

```bash
# Enable/disable metadata extraction
RAG_METADATA_EXTRACTION_ENABLED=true

# Batch processing
RAG_METADATA_BATCH_SIZE=10
RAG_METADATA_MAX_CONCURRENT=5

# LLM provider for extraction
RAG_LLM_PROVIDER=openai  # or 'gemini', 'local'

# Failure tolerance
RAG_FAILURE_THRESHOLD=0.2  # 20% max failures
```

### Settings Object

```python
from meditations_rag.config import settings

# Access metadata settings
enabled = settings.rag.metadata_extraction_enabled
batch_size = settings.rag.metadata_batch_size
max_concurrent = settings.rag.metadata_max_concurrent
```

## Error Handling

The service implements robust error handling:

1. **Individual Failures**: Failed chunks return `None` without blocking batch
2. **Failure Threshold**: If failures exceed `settings.rag.failure_threshold`, raises `ValueError`
3. **Retry Logic**: LLM calls use tenacity retry with exponential backoff
4. **Structured Validation**: Pydantic enforces schema compliance

```python
try:
    metadata_list = await extractor.extract_batch_metadata(chunks)
except ValueError as e:
    # Too many failures
    logger.error(f"Extraction failed: {e}")
```

## Performance Considerations

### Concurrency Control

- Uses `asyncio.Semaphore` to limit concurrent LLM requests
- Default: `max_concurrent=5` (configurable)
- Prevents rate limit issues and resource exhaustion

### Batching Strategy

- Processes chunks in batches for monitoring
- Default batch size: 10 chunks
- Provides granular progress logging

### Cost Optimization

**OpenAI API Cost Estimate** (gpt-4o-mini):
- Average tokens per extraction: ~1500 (prompt) + ~300 (response)
- Cost per 1K chunks: ~$2.70 (at $0.15/1M input, $0.60/1M output)

**Recommendations**:
- Use smaller/cheaper models for metadata extraction
- Cache metadata to avoid re-extraction
- Consider local models for high-volume scenarios

## Retrieval Enhancement

### Question-Based Retrieval

```python
# User query: "How does Marcus view death?"
# Match against metadata.questions:
# - "Why does Marcus consider death natural?"
# - "How should one prepare for death according to Stoicism?"
```

### Hybrid Search

```python
# Combine:
# 1. Semantic embedding search (top-K)
# 2. Keyword filtering (metadata.keywords)
# 3. Topic filtering (metadata.topic == "Death and Mortality")
# 4. Entity filtering (metadata.entities contains "Marcus Aurelius")
```

### Multi-Vector Indexing

Consider indexing:
- Chunk embeddings (semantic search)
- Question embeddings (question similarity)
- Keyword vectors (BM25/lexical)
- Topic clusters (categorical filtering)

## Best Practices

1. **Pre-extraction**: Run metadata extraction during ingestion, not query time
2. **Validation**: Always check for `None` in batch results
3. **Monitoring**: Track success rates and extraction latency
4. **Versioning**: Include metadata schema version for future migrations
5. **Testing**: Validate extraction quality on representative samples

## Troubleshooting

### Low Success Rate

```python
# Increase retries
settings.openai.max_retries = 5

# Reduce concurrency (avoid rate limits)
extractor = MetadataExtractorService(llm=llm, max_concurrent=3)

# Check LLM provider status
```

### Inconsistent Topics

```python
# Topics are constrained by MeditationsTopic enum
# If classification is poor, consider:
# 1. Improving chunk quality (better boundaries)
# 2. Adding examples to extraction prompt
# 3. Using a stronger LLM model
```

### High Latency

```python
# Optimize concurrency
max_concurrent = 10  # if rate limits allow

# Use faster LLM model
RAG_LLM_PROVIDER=gemini  # Gemini Flash is faster

# Parallelize batches (advanced)
```

## Example Output

```json
{
  "questions": [
    "How does Marcus define what is in our control versus not in our control?",
    "Why should one not fear external events according to Stoicism?",
    "What is the Stoic view on virtue versus external goods?",
    "How can one maintain tranquility amid difficulties?",
    "What role does reason play in managing emotions?"
  ],
  "keywords": [
    "control",
    "dichotomy",
    "virtue",
    "external goods",
    "reason",
    "tranquility",
    "Stoicism",
    "emotions",
    "judgment",
    "acceptance"
  ],
  "topic": "Control and Acceptance",
  "entities": [
    "Marcus Aurelius",
    "Stoics",
    "Epictetus"
  ],
  "philosophical_concepts": [
    "prohairesis",
    "apatheia",
    "dichotomy of control",
    "virtue ethics"
  ],
  "stoic_practices": [
    "premeditatio malorum",
    "examination of impressions"
  ]
}
```

## Testing

Run the example script:

```bash
uv run python examples/metadata_extraction_example.py
```

Expected output: Structured metadata for 3 sample chunks from Meditations.

## References

- [Pydantic Structured Outputs](https://docs.pydantic.dev/latest/concepts/json_schema/)
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [Gemini JSON Mode](https://ai.google.dev/gemini-api/docs/json-mode)
- [Marcus Aurelius - Meditations](https://en.wikipedia.org/wiki/Meditations)
