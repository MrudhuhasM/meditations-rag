"""
Metadata Extraction Service for RAG chunks.

This module provides structured metadata extraction from document chunks
to enhance retrieval quality. Uses LLM-based structured output generation.
"""

import asyncio
from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator

from meditations_rag.config import get_logger, settings
from meditations_rag.core.llm.base import LLMBase

logger = get_logger(__name__)


class MeditationsTopic(str, Enum):
    """
    Predefined topics from Marcus Aurelius' Meditations.
    Based on core Stoic philosophy themes.
    """

    VIRTUE_AND_CHARACTER = "Virtue and Character"
    DEATH_AND_MORTALITY = "Death and Mortality"
    REASON_AND_RATIONALITY = "Reason and Rationality"
    DUTY_AND_SERVICE = "Duty and Service"
    CONTROL_AND_ACCEPTANCE = "Control and Acceptance"
    NATURE_AND_UNIVERSE = "Nature and Universe"
    JUSTICE_AND_COMMUNITY = "Justice and Community"
    SELF_DISCIPLINE = "Self-Discipline"
    PLEASURE_AND_PAIN = "Pleasure and Pain"
    TIME_AND_IMPERMANENCE = "Time and Impermanence"
    ANGER_AND_EMOTIONS = "Anger and Emotions"
    WISDOM_AND_LEARNING = "Wisdom and Learning"
    SIMPLICITY_AND_HUMILITY = "Simplicity and Humility"
    GRATITUDE_AND_APPRECIATION = "Gratitude and Appreciation"
    LEADERSHIP_AND_GOVERNANCE = "Leadership and Governance"


class ChunkMetadata(BaseModel):
    """
    Structured metadata extracted from a document chunk.

    This metadata enhances retrieval by providing:
    - Semantic questions the chunk can answer
    - Key terms for lexical matching
    - Topical classification for filtering
    - Named entities for fact-based queries
    - Philosophical concepts for thematic search
    """

    questions: List[str] = Field(
        description="Five specific questions that this chunk can answer comprehensively",
        min_length=5,
        max_length=5,
    )

    keywords: List[str] = Field(
        description="8-12 distinctive keywords or phrases representing core concepts in the chunk",
        min_length=8,
    )

    @field_validator("keywords", mode="after")
    @classmethod
    def validate_keywords_length(cls, v: List[str]) -> List[str]:
        """
        Validate and truncate keywords list if LLM returns more than expected.

        This provides resilience when the LLM doesn't follow instructions perfectly.
        We keep the first 12 keywords as they're typically the most important.
        """
        if len(v) > 12:
            logger.warning(f"LLM returned {len(v)} keywords, truncating to 12")
            return v[:12]
        return v

    topic: MeditationsTopic = Field(
        description="Single primary topic from predefined Stoic philosophy themes"
    )

    entities: List[str] = Field(
        description="Named entities: people, places, philosophical schools, historical events, etc.",
        default_factory=list,
    )

    philosophical_concepts: List[str] = Field(
        description="Abstract Stoic or philosophical concepts discussed (e.g., 'logos', 'apatheia', 'prohairesis')",
        default_factory=list,
    )

    stoic_practices: List[str] = Field(
        description="Specific Stoic exercises or practices mentioned (e.g., 'negative visualization', 'view from above')",
        default_factory=list,
    )


class MetadataExtractionStrategy(BaseModel):
    """Strategy pattern for different extraction approaches."""

    def build_prompt(self, chunk_text: str, available_topics: List[str]) -> str:
        """Build extraction prompt. Override for custom strategies."""
        topics_str = "\n".join([f"- {topic}" for topic in available_topics])

        return f"""You are analyzing a passage from Marcus Aurelius' "Meditations", a foundational Stoic philosophy text.

Extract comprehensive metadata to enable precise retrieval and semantic search.

**Passage:**
{chunk_text}

**Instructions:**

1. **Questions (exactly 5)**: Generate 5 specific questions this passage can definitively answer. Make them diverse:
   - 1-2 factual questions (who, what, when)
   - 2-3 conceptual questions (why, how, what does X mean)
   - 1 application question (how should one...)

2. **Keywords (8-12)**: Extract distinctive terms, including:
   - Key Stoic terminology
   - Important verbs and concepts
   - Unique phrases from the text

3. **Topic**: Select the ONE most appropriate topic from this list:
{topics_str}

4. **Entities**: Extract named entities such as:
   - People (Marcus Aurelius, Socrates, etc.)
   - Places (Rome, Athens, etc.)
   - Philosophical schools (Stoicism, Epicureanism, etc.)
   - Historical events or periods

5. **Philosophical Concepts**: Identify abstract Stoic/philosophical concepts:
   - Greek terms (logos, pneuma, hegemonikon, etc.)
   - Core ideas (virtue, wisdom, temperance, etc.)

6. **Stoic Practices**: Identify any mentioned Stoic exercises:
   - Meditative practices
   - Mental techniques
   - Practical disciplines

Provide precise, high-quality metadata that maximizes retrieval accuracy."""


class MetadataExtractorService:
    """
    Service for extracting structured metadata from document chunks.

    Uses Strategy pattern for flexible extraction approaches and
    Factory pattern for LLM provider abstraction.

    Design Principles:
    - Single Responsibility: Only handles metadata extraction
    - Open/Closed: Extensible via strategy pattern
    - Dependency Inversion: Depends on LLMBase abstraction
    """

    def __init__(
        self,
        llm: LLMBase,
        strategy: MetadataExtractionStrategy | None = None,
        batch_size: int = 10,
        max_concurrent: int = 5,
    ):
        """
        Initialize metadata extractor.

        Args:
            llm: LLM implementation for structured generation
            strategy: Extraction strategy (uses default if None)
            batch_size: Number of chunks to process in parallel batches
            max_concurrent: Max concurrent LLM requests
        """
        self.llm = llm
        self.strategy = strategy or MetadataExtractionStrategy()
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.available_topics = [topic.value for topic in MeditationsTopic]

    async def extract_metadata(self, chunk_text: str) -> ChunkMetadata:
        """
        Extract metadata from a single chunk.

        Args:
            chunk_text: Text content of the chunk

        Returns:
            Structured metadata

        Raises:
            ValueError: If LLM fails to generate valid metadata
        """
        try:
            prompt = self.strategy.build_prompt(chunk_text, self.available_topics)

            result = await self.llm.generate_structured(
                prompt=prompt, response_model=ChunkMetadata
            )

            # Type narrowing - we know generate_structured returns ChunkMetadata
            if not isinstance(result, ChunkMetadata):
                raise ValueError("Invalid metadata type returned")

            logger.debug(f"Extracted metadata with topic: {result.topic}")
            return result

        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            raise ValueError(f"Metadata extraction failed: {e}") from e

    async def _extract_with_semaphore(
        self, semaphore: asyncio.Semaphore, chunk_text: str, chunk_index: int
    ) -> tuple[int, ChunkMetadata | None]:
        """
        Extract metadata with concurrency control.

        Args:
            semaphore: Asyncio semaphore for rate limiting
            chunk_text: Chunk text content
            chunk_index: Index for tracking

        Returns:
            Tuple of (index, metadata or None if failed)
        """
        async with semaphore:
            try:
                metadata = await self.extract_metadata(chunk_text)
                return (chunk_index, metadata)
            except Exception as e:
                logger.warning(f"Chunk {chunk_index} metadata extraction failed: {e}")
                return (chunk_index, None)

    async def extract_batch_metadata(
        self, chunks: List[str]
    ) -> List[ChunkMetadata | None]:
        """
        Extract metadata from multiple chunks with batching and concurrency control.

        Args:
            chunks: List of chunk text contents

        Returns:
            List of metadata objects (None for failed extractions)
        """
        logger.info(f"Starting metadata extraction for {len(chunks)} chunks")

        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: List[ChunkMetadata | None] = [None] * len(chunks)

        # Process in batches for better monitoring
        for batch_start in range(0, len(chunks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start // self.batch_size + 1}: chunks {batch_start}-{batch_end - 1}"
            )

            # Create tasks for this batch
            tasks = [
                self._extract_with_semaphore(semaphore, chunk, batch_start + i)
                for i, chunk in enumerate(batch_chunks)
            ]

            # Wait for batch completion
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Store results
            success_count = 0
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
                    continue

                # Type narrowing - result is tuple[int, ChunkMetadata | None]
                if not isinstance(result, tuple):
                    logger.error(f"Unexpected result type: {type(result)}")
                    continue

                idx, metadata = result
                results[idx] = metadata
                if metadata is not None:
                    success_count += 1

            logger.info(
                f"Batch completed: {success_count}/{len(batch_chunks)} successful extractions"
            )

        # Calculate overall stats
        total_success = sum(1 for r in results if r is not None)
        success_rate = (total_success / len(chunks)) * 100 if chunks else 0

        logger.info(
            f"Metadata extraction completed: {total_success}/{len(chunks)} ({success_rate:.1f}%) successful"
        )

        # Check failure threshold
        failure_rate = 1 - (total_success / len(chunks)) if chunks else 0
        if failure_rate > settings.rag.failure_threshold:
            logger.error(
                f"Metadata extraction failure rate ({failure_rate:.1%}) exceeds threshold "
                f"({settings.rag.failure_threshold:.1%})"
            )
            raise ValueError("Metadata extraction failed: too many failures")

        return results
