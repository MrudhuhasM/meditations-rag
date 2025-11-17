"""
RAG (Retrieval-Augmented Generation) configuration settings.

This module contains settings specific to RAG pipeline operations
including document processing, retrieval, and generation.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from meditations_rag.config.base import BaseAppSettings


class DocumentProcessingSettings(BaseAppSettings):
    """Document processing and chunking settings."""

    # Document Loading
    documents_path: Path = Field(
        default=Path("data/documents"),
        description="Path to source documents",
    )
    supported_formats: list[str] = Field(
        default=["txt", "pdf", "md", "html", "docx"],
        description="Supported document formats",
    )

    # Text Chunking
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Target size for text chunks (in tokens/characters)",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between consecutive chunks",
    )
    chunk_strategy: Literal["fixed", "semantic", "recursive"] = Field(
        default="recursive",
        description="Chunking strategy to use",
    )

    # Text Cleaning
    remove_extra_whitespace: bool = Field(
        default=True,
        description="Remove extra whitespace from text",
    )
    normalize_unicode: bool = Field(
        default=True,
        description="Normalize unicode characters",
    )
    remove_urls: bool = Field(
        default=False,
        description="Remove URLs from text",
    )
    lowercase: bool = Field(
        default=False,
        description="Convert text to lowercase",
    )

    # Metadata Extraction
    extract_metadata: bool = Field(
        default=True,
        description="Extract metadata from documents",
    )
    metadata_fields: list[str] = Field(
        default=["title", "author", "date", "source"],
        description="Metadata fields to extract",
    )

    @field_validator("documents_path")
    @classmethod
    def create_documents_directory(cls, v: Path) -> Path:
        """Ensure documents directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class RetrievalSettings(BaseAppSettings):
    """Document retrieval and search settings."""

    # Retrieval Strategy
    retrieval_strategy: Literal["similarity", "mmr", "hybrid"] = Field(
        default="similarity",
        description="Document retrieval strategy",
    )

    # Search Parameters
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of documents to retrieve",
    )
    score_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for retrieval",
    )

    # MMR (Maximal Marginal Relevance) Settings
    mmr_diversity_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Diversity score for MMR (0=similarity only, 1=diversity only)",
    )
    mmr_fetch_k: int = Field(
        default=20,
        ge=1,
        description="Number of documents to fetch before MMR reranking",
    )

    # Hybrid Search Settings
    hybrid_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Balance between vector and keyword search (0=keyword, 1=vector)",
    )

    # Reranking
    rerank_enabled: bool = Field(
        default=False,
        description="Enable reranking of retrieved documents",
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        description="Model to use for reranking",
    )
    rerank_top_k: int = Field(
        default=3,
        ge=1,
        description="Number of documents after reranking",
    )

    # Filtering
    filter_enabled: bool = Field(
        default=True,
        description="Enable metadata filtering",
    )
    filter_conditions: dict[str, str] = Field(
        default_factory=dict,
        description="Default filter conditions",
    )


class GenerationSettings(BaseAppSettings):
    """Response generation settings."""

    # Prompt Engineering
    system_prompt: str = Field(
        default="You are a helpful assistant that answers questions based on the provided context from Marcus Aurelius's Meditations.",
        description="System prompt for the LLM",
    )
    prompt_template: str = Field(
        default="""Context:
{context}

Question: {question}

Answer based on the context above. If the answer cannot be found in the context, say so.""",
        description="Template for constructing prompts",
    )

    # Generation Parameters
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation",
    )
    max_tokens: int = Field(
        default=500,
        ge=1,
        description="Maximum tokens in generated response",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling parameter",
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty (-2.0 to 2.0)",
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty (-2.0 to 2.0)",
    )

    # Response Control
    include_sources: bool = Field(
        default=True,
        description="Include source documents in response",
    )
    include_confidence: bool = Field(
        default=True,
        description="Include confidence scores in response",
    )
    streaming: bool = Field(
        default=False,
        description="Stream responses token by token",
    )

    # Safety and Quality
    max_context_length: int = Field(
        default=4000,
        ge=100,
        description="Maximum context length in tokens",
    )
    truncate_context: bool = Field(
        default=True,
        description="Truncate context if it exceeds max length",
    )
    fallback_response: str = Field(
        default="I couldn't find relevant information to answer your question.",
        description="Fallback response when no good answer is found",
    )


class RAGSettings(BaseAppSettings):
    """Unified RAG pipeline configuration."""

    # Pipeline Control
    rag_enabled: bool = Field(
        default=True,
        description="Enable RAG pipeline",
    )
    rag_mode: Literal["simple", "advanced", "agentic"] = Field(
        default="simple",
        description="RAG pipeline mode",
    )

    # Caching
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching of queries and responses",
    )
    cache_ttl: int = Field(
        default=3600,
        ge=0,
        description="Cache time-to-live in seconds",
    )
    cache_max_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of cached items",
    )

    # Performance
    async_enabled: bool = Field(
        default=True,
        description="Enable async processing",
    )
    batch_processing: bool = Field(
        default=False,
        description="Enable batch processing of queries",
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Batch size for processing",
    )

    # Monitoring
    track_metrics: bool = Field(
        default=True,
        description="Track RAG pipeline metrics",
    )
    log_queries: bool = Field(
        default=True,
        description="Log user queries (disable in production if PII concerns)",
    )
    log_responses: bool = Field(
        default=False,
        description="Log generated responses",
    )

    # Component Settings
    document_processing: DocumentProcessingSettings = Field(
        default_factory=DocumentProcessingSettings
    )
    retrieval: RetrievalSettings = Field(
        default_factory=RetrievalSettings
    )
    generation: GenerationSettings = Field(
        default_factory=GenerationSettings
    )

    @field_validator("rag_mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Normalize RAG mode."""
        return v.lower()

    def model_post_init(self, __context) -> None:
        """Post-initialization validation."""
        # Ensure batch size makes sense
        if self.batch_processing and self.batch_size < 1:
            raise ValueError("batch_size must be >= 1 when batch_processing is enabled")
