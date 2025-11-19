# Retrieval Service Documentation

## Overview

The `RetrievalService` implements a **hybrid retrieval strategy** for the RAG pipeline, combining:

1. **Dense retrieval from chunks** - Semantic similarity search over chunk embeddings
2. **Dense retrieval from questions** - Semantic similarity search over question embeddings  
3. **Score fusion** - Linear combination of chunk and question scores

This dual-path approach improves retrieval quality by matching both:
- **Direct content similarity** (chunk embeddings)
- **Conceptual similarity** (question embeddings)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Retrieval Service                         │
│                                                               │
│  User Query                                                   │
│      │                                                        │
│      ├─────────────────┬─────────────────┐                  │
│      ▼                 ▼                 ▼                   │
│  Embed Query     Embed Query       Embed Query              │
│      │                 │                 │                   │
│      ▼                 ▼                 ▼                   │
│  Search            Search            Search                  │
│  Chunks         Questions         (Optional)                 │
│  Collection     Collection                                   │
│      │                 │                                     │
│      └────────┬────────┘                                     │
│               ▼                                               │
│         Score Fusion                                          │
│    final = chunk_score + α * question_score                  │
│               │                                               │
│               ▼                                               │
│      Ranked Results                                           │
└───────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### 1. Dual-Collection Strategy

**Chunks Collection:**
- Contains full chunk text embeddings
- Stores complete chunk content and metadata
- Best for: Direct content matching, specific passages

**Questions Collection:**
- Contains embeddings of generated questions from chunks
- Links questions back to parent chunks via `chunk_id`
- Best for: Conceptual matching, query reformulation, understanding user intent

### 2. Score Fusion Formula

```
final_score = chunk_score + α * question_score
```

Where:
- `chunk_score`: Cosine similarity from chunks collection (0.0 to 1.0)
- `question_score`: Cosine similarity from questions collection (0.0 to 1.0)
- `α` (alpha): Weight parameter controlling question influence (0.0 to 1.0)

**Alpha Parameter Guidelines:**

| Alpha | Strategy | Use Case |
|-------|----------|----------|
| 0.0 | Chunk-only | When you want exact passage matching |
| 0.3 | Chunk-focused | Default - slight question boost |
| 0.5 | Balanced | Equal weight to both approaches |
| 0.7 | Question-focused | When user intent matters more |
| 1.0 | Question-only | Pure conceptual matching |

### 3. Retrieval Modes

The service supports multiple retrieval modes:

1. **Hybrid (Default)** - Combines both chunk and question retrieval
2. **Chunk-Only** - Only searches chunk embeddings
3. **Question-Only** - Only searches question embeddings

---

## API Reference

### RetrievalService

```python
from meditations_rag.services import RetrievalService

retrieval_service = RetrievalService(
    vector_store=vector_store,
    embedding_service=embedding_service,
    alpha=0.3,              # Score fusion weight
    top_k=5,                # Final number of results
    question_top_k=10       # Results from question collection
)
```

**Constructor Parameters:**

- `vector_store` (QdrantVectorStore): Vector database client
- `embedding_service` (VectorEmbeddingService): Embedding generation service
- `alpha` (float): Score fusion weight (0.0-1.0), default from config
- `top_k` (int): Number of final results to return
- `question_top_k` (int, optional): Results from question collection (defaults to top_k)

### Core Methods

#### retrieve()

Main retrieval method with flexible options.

```python
results = await retrieval_service.retrieve(
    query="How should I deal with difficult people?",
    top_k=5,              # Override default
    alpha=0.3,            # Override default
    chunk_only=False,     # Only search chunks
    question_only=False   # Only search questions
)
```

**Parameters:**
- `query` (str): User query string
- `top_k` (int, optional): Override default top_k
- `alpha` (float, optional): Override default alpha
- `chunk_only` (bool): Disable question retrieval
- `question_only` (bool): Disable chunk retrieval

**Returns:** `List[RetrievalResult]`

**RetrievalResult Schema:**
```python
{
    "chunk_id": str,           # Unique chunk identifier
    "score": float,            # Final fused score
    "chunk_score": float,      # Score from chunks (if available)
    "question_score": float,   # Score from questions (if available)
    "text": str,               # Retrieved chunk text
    "metadata": dict,          # Chunk metadata
    "matched_question": str    # Matched question (if from question retrieval)
}
```

#### retrieve_with_context()

Retrieve and format results as LLM-ready context.

```python
context_data = await retrieval_service.retrieve_with_context(
    query="What does Marcus say about death?",
    top_k=3,
    alpha=0.5,
    include_metadata=True
)
```

**Returns:**
```python
{
    "query": str,
    "context": str,              # Formatted context string
    "num_results": int,
    "results": List[dict],       # Full result objects
    "retrieval_metadata": {
        "top_k": int,
        "alpha": float,
        "avg_score": float,
        "max_score": float,
        "min_score": float
    }
}
```

---

## Configuration

Set retrieval parameters in `.env`:

```bash
# Retrieval Settings
RAG_RETRIEVAL_TOP_K=5              # Number of final results
RAG_RETRIEVAL_QUESTION_TOP_K=10    # Results from question collection
RAG_RETRIEVAL_ALPHA=0.3            # Score fusion weight (0.0-1.0)
```

Access in code:
```python
from meditations_rag.config import settings

top_k = settings.rag.retrieval_top_k
alpha = settings.rag.retrieval_alpha
```

---

## Usage Examples

### Example 1: Basic Hybrid Retrieval

```python
import asyncio
from meditations_rag.services import RetrievalService
from meditations_rag.services.vector_store import QdrantVectorStore, VectorEmbeddingService
from meditations_rag.core.embedding import EmbeddingFactory

async def basic_retrieval():
    # Setup
    embedding_provider = EmbeddingFactory().create_embedding_provider()
    embedding_service = VectorEmbeddingService(embedding_provider)
    vector_store = QdrantVectorStore()
    
    retrieval_service = RetrievalService(
        vector_store=vector_store,
        embedding_service=embedding_service,
        alpha=0.3,
        top_k=5
    )
    
    # Retrieve
    results = await retrieval_service.retrieve(
        query="How should I prepare for challenges each morning?"
    )
    
    # Display
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.score:.4f}")
        print(f"Chunk: {result.chunk_id}")
        print(f"Text: {result.text[:200]}...")
        if result.matched_question:
            print(f"Matched Q: {result.matched_question}")

asyncio.run(basic_retrieval())
```

### Example 2: Chunk-Only vs Question-Only

```python
async def compare_modes():
    # ... setup code ...
    
    query = "What did Marcus learn from his teachers?"
    
    # Chunk-only retrieval
    chunk_results = await retrieval_service.retrieve(
        query=query,
        chunk_only=True,
        top_k=3
    )
    
    # Question-only retrieval
    question_results = await retrieval_service.retrieve(
        query=query,
        question_only=True,
        top_k=3
    )
    
    print("Chunk-only top result:", chunk_results[0].chunk_id)
    print("Question-only top result:", question_results[0].chunk_id)
    print("Matched question:", question_results[0].matched_question)
```

### Example 3: Alpha Parameter Tuning

```python
async def tune_alpha():
    query = "What does Marcus say about death?"
    
    # Test different alpha values
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        results = await retrieval_service.retrieve(
            query=query,
            alpha=alpha,
            top_k=3
        )
        
        print(f"\nAlpha = {alpha}")
        print(f"Top chunk: {results[0].chunk_id}")
        print(f"Score: {results[0].score:.4f}")
        print(f"  Chunk: {results[0].chunk_score:.4f}")
        print(f"  Question: {results[0].question_score:.4f}")
```

### Example 4: LLM-Ready Context

```python
async def get_llm_context():
    context_data = await retrieval_service.retrieve_with_context(
        query="How can philosophy help in daily life?",
        top_k=3,
        include_metadata=True
    )
    
    # Use in LLM prompt
    prompt = f"""
Context from Marcus Aurelius's Meditations:
{context_data['context']}

User Question: {context_data['query']}

Please answer based on the provided context.
"""
    
    print(prompt)
    print(f"\nRetrieval Stats:")
    print(f"  Avg Score: {context_data['retrieval_metadata']['avg_score']:.4f}")
```

---

## Best Practices

### 1. Choosing Alpha

**Start with default (0.3):**
- Good balance for most queries
- Chunk content dominates, questions provide boost

**Increase alpha (0.5-0.7) when:**
- User queries are conceptual or abstract
- Questions are high-quality and diverse
- Query intent matters more than exact wording

**Decrease alpha (0.1-0.2) when:**
- Precision is critical
- Looking for specific passages or quotes
- Questions are noisy or low-quality

**Dynamic alpha:**
```python
# Adjust based on query type
def get_alpha(query: str) -> float:
    if any(word in query.lower() for word in ["how", "why", "explain"]):
        return 0.5  # Conceptual - higher alpha
    elif any(word in query.lower() for word in ["who", "when", "where"]):
        return 0.2  # Factual - lower alpha
    else:
        return 0.3  # Default
```

### 2. Setting top_k Values

**Final top_k (results returned):**
- Start with 3-5 for LLM context
- Increase to 10-20 for reranking pipelines
- Keep low to avoid context overflow

**question_top_k (candidates from questions):**
- Set 2-3x higher than final top_k
- Ensures good question coverage
- Default: 10 when final top_k = 5

### 3. Performance Optimization

**Parallel retrieval:**
```python
# Built-in - both collections searched in parallel
results = await retrieval_service.retrieve(query)
```

**Batch queries:**
```python
async def batch_retrieve(queries: List[str]):
    tasks = [retrieval_service.retrieve(q) for q in queries]
    return await asyncio.gather(*tasks)
```

**Caching embeddings:**
```python
# Cache query embeddings for repeated queries
from functools import lru_cache

@lru_cache(maxsize=100)
def cache_query_embedding(query: str):
    # Implement caching logic
    pass
```

### 4. Error Handling

```python
from meditations_rag.core.exceptions import VectorStoreQueryError

try:
    results = await retrieval_service.retrieve(query)
except VectorStoreQueryError as e:
    logger.error(f"Retrieval failed: {e}")
    # Fallback logic
    results = []
except ValueError as e:
    logger.error(f"Invalid parameters: {e}")
    # Handle bad input
```

---

## Evaluation & Metrics

### Retrieval Quality Metrics

**1. Score Distribution:**
```python
scores = [r.score for r in results]
avg_score = sum(scores) / len(scores)
score_variance = statistics.variance(scores)

print(f"Average score: {avg_score:.4f}")
print(f"Variance: {score_variance:.4f}")
```

**2. Source Analysis:**
```python
chunk_sources = sum(1 for r in results if r.chunk_score is not None)
question_sources = sum(1 for r in results if r.question_score is not None)
hybrid_sources = sum(1 for r in results if r.chunk_score and r.question_score)

print(f"From chunks only: {chunk_sources}")
print(f"From questions only: {question_sources}")
print(f"From both: {hybrid_sources}")
```

**3. Comparing Strategies:**
```python
async def evaluate_strategies(query: str, ground_truth_chunk_id: str):
    strategies = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    for alpha in strategies:
        results = await retrieval_service.retrieve(query, alpha=alpha, top_k=10)
        
        # Find rank of ground truth
        rank = next(
            (i for i, r in enumerate(results, 1) if r.chunk_id == ground_truth_chunk_id),
            None
        )
        
        print(f"Alpha {alpha}: Ground truth at rank {rank}")
```

---

## Troubleshooting

### Issue: Low scores across all results

**Symptoms:** All scores < 0.3

**Causes & Solutions:**
1. **Poor query embedding** - Check embedding model quality
2. **Mismatched domains** - Ensure embeddings from same model
3. **Empty collections** - Verify data was ingested correctly

```python
# Check collection status
info = vector_store.client.get_collection(vector_store.main_collection)
print(f"Chunks: {info.points_count}")

info = vector_store.client.get_collection(vector_store.question_collection)
print(f"Questions: {info.points_count}")
```

### Issue: Question scores always 0

**Symptoms:** `question_score` is None or 0 for all results

**Causes:**
1. Questions collection is empty
2. No question embeddings were generated during ingestion
3. Questions don't match query semantically

**Solution:**
```python
# Verify questions were ingested
results = vector_store.client.scroll(
    collection_name=vector_store.question_collection,
    limit=10
)
print(f"Found {len(results[0])} questions")
for point in results[0][:3]:
    print(f"  Q: {point.payload.get('question', 'N/A')}")
```

### Issue: Identical results regardless of alpha

**Symptoms:** Same chunks returned for alpha=0.0 and alpha=1.0

**Causes:**
1. Only one collection has data
2. Question collection not properly linked to chunks

**Debug:**
```python
# Test each mode separately
chunk_only = await retrieval_service.retrieve(query, chunk_only=True)
question_only = await retrieval_service.retrieve(query, question_only=True)

print(f"Chunk-only: {[r.chunk_id for r in chunk_only]}")
print(f"Question-only: {[r.chunk_id for r in question_only]}")
```

---

## Advanced Topics

### Custom Score Fusion

Extend `RetrievalService` for custom fusion logic:

```python
from meditations_rag.services.retrieval import RetrievalService

class CustomRetrievalService(RetrievalService):
    def _fuse_scores(self, chunk_results, question_results):
        # Custom fusion: Reciprocal Rank Fusion (RRF)
        k = 60
        fused_scores = {}
        
        for rank, point in enumerate(chunk_results, 1):
            chunk_id = point.id
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (k + rank)
        
        for rank, point in enumerate(question_results, 1):
            chunk_id = point.payload.get("chunk_id")
            if chunk_id:
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (k + rank)
        
        # Convert to RetrievalResult objects
        # ... implementation ...
```

### Metadata Filtering

Add filters to retrieval:

```python
# Modify _retrieve_from_chunks to include filters
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = self.vector_store.client.search(
    collection_name=self.vector_store.main_collection,
    query_vector=query_vector,
    limit=top_k,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="topic",
                match=MatchValue(value="death")
            )
        ]
    )
)
```

---

## Related Documentation

- [Vector Store Implementation](VECTOR_STORE_IMPLEMENTATION.md)
- [Embedding Service](embedding_factory.md)
- [Metadata Extraction](metadata_extraction.md)
- [Qdrant Documentation](https://qdrant.tech/documentation/)

---

## References

### Academic Papers

1. **Hybrid Search:**
   - Luan et al. (2021) - "Sparse, Dense, and Attentional Representations for Text Retrieval"
   - Ma et al. (2021) - "A Replica Study on Hybrid Retrieval"

2. **Score Fusion:**
   - Cormack et al. (2009) - "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
   - Shaw & Fox (1994) - "Combination of Multiple Searches"

3. **RAG Systems:**
   - Lewis et al. (2020) - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
   - Gao et al. (2023) - "Retrieval-Augmented Generation for Large Language Models: A Survey"

### Online Resources

- [Pinecone: Hybrid Search Intro](https://www.pinecone.io/learn/hybrid-search-intro/)
- [Qdrant: Search Concepts](https://qdrant.tech/documentation/concepts/search/)
- [Building RAG Systems](https://www.anthropic.com/index/retrieval-augmented-generation)

---

## Changelog

**v0.1.0** (2025-11-19)
- Initial implementation
- Dense retrieval from chunks and questions
- Linear score fusion with alpha parameter
- Support for chunk-only, question-only, and hybrid modes
- LLM context formatting
- Comprehensive documentation and examples
