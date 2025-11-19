# RAG Pipeline Documentation

## Overview

The RAG (Retrieval-Augmented Generation) pipeline orchestrates the complete question-answering workflow.

## Basic Usage

```python
from meditations_rag.pipelines.rag import create_rag_pipeline
from meditations_rag.core.llm.openai import OpenAILLM
from meditations_rag.core.embedding.openai import OpenAIEmbedding

# Initialize
llm = OpenAILLM()
embedder = OpenAIEmbedding()
pipeline = create_rag_pipeline(llm, embedder)

# Query
response = await pipeline.query("What does Marcus say about virtue?")
print(response.answer)
```

## Features

- Query rewriting and metadata extraction
- Hybrid retrieval (chunks + questions + metadata)
- Configurable retrieval parameters
- Batch query processing
- Comprehensive error handling

## See examples/rag_pipeline_example.py for detailed usage examples.
