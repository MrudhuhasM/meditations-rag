"""
RAG Pipeline for Meditations Question Answering.

Complete end-to-end pipeline that:
1. Takes a user query
2. Retrieves relevant context from vector store
3. Generates an answer using LLM
4. Returns structured response with sources

Design Principles:
- Single Responsibility: Orchestrates retrieval + generation
- Dependency Inversion: Uses abstractions for all services
- Production-Ready: Comprehensive error handling and logging
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from meditations_rag.services.retrieval import RetrievalService, QueryRewriter, RetrievalResult
from meditations_rag.services.vector_store import QdrantVectorStore, VectorEmbeddingService
from meditations_rag.core.llm.base import LLMBase
from meditations_rag.core.embedding.base import EmbeddingBase
from meditations_rag.config import get_logger, settings
from meditations_rag.core.exceptions import (
    MeditationsRAGException,
    LLMException,
    VectorStoreException
)
import time


logger = get_logger(__name__)


class RAGResponse(BaseModel):
    """
    Structured response from RAG pipeline.
    
    Contains the generated answer along with metadata about
    the retrieval and generation process.
    """
    
    query: str = Field(description="Original user query")
    answer: str = Field(description="Generated answer from LLM")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved sources used to generate the answer"
    )
    retrieval_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the retrieval process"
    )
    generation_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the generation process"
    )
    total_time_seconds: float = Field(
        default=0.0,
        description="Total pipeline execution time"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How should I prepare myself each morning?",
                "answer": "According to Marcus Aurelius in Meditations...",
                "sources": [
                    {
                        "chunk_id": "chunk_123",
                        "text": "When you wake at dawn...",
                        "score": 0.89,
                        "topic": "daily_practice"
                    }
                ],
                "retrieval_metadata": {
                    "num_results": 5,
                    "avg_score": 0.85,
                    "retrieval_time_seconds": 0.3
                },
                "generation_metadata": {
                    "model": "gpt-4",
                    "generation_time_seconds": 1.2
                },
                "total_time_seconds": 1.5
            }
        }


class RAGPipeline:
    """
    Complete RAG pipeline for Meditations question answering.
    
    Orchestrates:
    1. Query processing and metadata extraction
    2. Hybrid retrieval from vector store
    3. Context formatting
    4. Answer generation with LLM
    5. Response structuring with sources
    
    Usage:
        pipeline = RAGPipeline(llm, embedding_provider)
        response = await pipeline.query("What does Marcus say about virtue?")
        print(response.answer)
    """
    
    def __init__(
        self,
        llm: LLMBase,
        embedding_provider: EmbeddingBase,
        vector_store: Optional[QdrantVectorStore] = None,
        enable_query_rewriting: bool = True,
        retrieval_top_k: Optional[int] = None,
        retrieval_alpha: Optional[float] = None,
        enable_metadata_retrieval: bool = True,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            llm: LLM provider for answer generation
            embedding_provider: Embedding provider for query encoding
            vector_store: Qdrant vector store (creates default if None)
            enable_query_rewriting: Enable query rewriting + metadata extraction
            retrieval_top_k: Number of results to retrieve (defaults to config)
            retrieval_alpha: Weight for question scores (defaults to config)
            enable_metadata_retrieval: Enable metadata-aware retrieval
            system_prompt: Custom system prompt for answer generation
        """
        self.llm = llm
        self.embedding_provider = embedding_provider
        
        # Initialize vector store
        self.vector_store = vector_store or QdrantVectorStore()
        
        # Initialize embedding service
        self.embedding_service = VectorEmbeddingService(
            embedding_provider=self.embedding_provider
        )
        
        # Initialize query rewriter if enabled
        self.query_rewriter = None
        if enable_query_rewriting:
            self.query_rewriter = QueryRewriter(llm=self.llm)
        
        # Initialize retrieval service
        self.retrieval_service = RetrievalService(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            query_rewriter=self.query_rewriter,
            alpha=retrieval_alpha or settings.rag.retrieval_alpha,
            top_k=retrieval_top_k or settings.rag.retrieval_top_k,
            enable_metadata_retrieval=enable_metadata_retrieval
        )
        
        # System prompt for answer generation
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        logger.info(
            f"Initialized RAG Pipeline: "
            f"query_rewriting={enable_query_rewriting}, "
            f"top_k={self.retrieval_service.top_k}, "
            f"alpha={self.retrieval_service.alpha}, "
            f"metadata_retrieval={enable_metadata_retrieval}"
        )
    
    def _default_system_prompt(self) -> str:
        """
        Generate default system prompt for answer generation.
        
        Returns:
            System prompt string
        """
        return """You are a knowledgeable assistant specializing in Marcus Aurelius' "Meditations" and Stoic philosophy.

Your task is to answer questions about Meditations based SOLELY on the provided context passages. Follow these guidelines:

1. **Accuracy First**: Only provide information directly supported by the context. If the context doesn't contain enough information to answer the question fully, acknowledge this limitation.

2. **Quote Directly**: When possible, include relevant quotes from the text to support your answer.

3. **Attribution**: Reference the specific passages or books when citing ideas.

4. **Stoic Context**: Explain philosophical concepts clearly, making them accessible to modern readers while maintaining their original meaning.

5. **Honest Limitations**: If the provided context doesn't address the question, say so explicitly. Don't hallucinate or invent information.

6. **Practical Wisdom**: When appropriate, connect Marcus' teachings to practical application, but always ground it in the text.

7. **Concise but Complete**: Provide thorough answers without unnecessary verbosity.

Remember: Your authority comes from the text itself. Stay faithful to what Marcus actually wrote."""
    
    def _format_context(
        self,
        results: List[RetrievalResult],
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieval results as context for LLM.
        
        Args:
            results: List of retrieval results
            include_metadata: Include metadata in context
            
        Returns:
            Formatted context string
        """
        if not results:
            return "[No relevant context found]"
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            context_block = f"[Passage {i}]\n{result.text}"
            
            if include_metadata:
                metadata_lines = []
                
                # Add topic if available
                if result.metadata.get("topic"):
                    metadata_lines.append(f"Topic: {result.metadata['topic']}")
                
                # Add book/chapter if available
                if result.metadata.get("book"):
                    metadata_lines.append(f"Book: {result.metadata['book']}")
                if result.metadata.get("chapter"):
                    metadata_lines.append(f"Chapter: {result.metadata['chapter']}")
                
                # Add matched question if available
                if result.matched_question:
                    metadata_lines.append(f"Related Question: {result.matched_question}")
                
                if metadata_lines:
                    context_block += "\n" + "\n".join(metadata_lines)
            
            context_parts.append(context_block)
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Build complete prompt for LLM answer generation.
        
        Args:
            query: User query
            context: Formatted context from retrieval
            
        Returns:
            Complete prompt string
        """
        prompt = f"""Based on the following passages from Marcus Aurelius' "Meditations", please answer the user's question.

**Context from Meditations:**

{context}

**User Question:**
{query}

**Your Answer:**"""
        
        return prompt
    
    async def _generate_answer(
        self,
        query: str,
        context: str
    ) -> tuple[str, Dict[str, Any]]:
        """
        Generate answer using LLM.
        
        Args:
            query: User query
            context: Formatted context
            
        Returns:
            Tuple of (answer string, generation metadata)
        """
        start_time = time.time()
        
        try:
            # Build complete prompt
            prompt = self._build_prompt(query, context)
            
            logger.debug(f"Generating answer for query: '{query[:100]}...'")
            
            # Generate answer
            # Note: We use the system prompt via a messages format if the LLM supports it
            # For now, we'll include it in the prompt
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            
            answer = await self.llm.generate(full_prompt)
            
            generation_time = time.time() - start_time
            
            logger.info(f"Answer generated in {generation_time:.2f}s")
            
            # Build generation metadata
            # Try to get model name from LLM settings if available
            model_name = "unknown"
            if hasattr(self.llm, "settings"):
                settings_obj = getattr(self.llm, "settings")
                model_name = getattr(settings_obj, "llm_model_name", "unknown")
            
            metadata = {
                "generation_time_seconds": generation_time,
                "model": model_name
            }
            
            return answer, metadata
            
        except LLMException as e:
            logger.error(f"LLM generation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during answer generation: {e}", exc_info=True)
            raise MeditationsRAGException(
                f"Failed to generate answer: {e}",
                details={"query": query, "error": str(e)}
            ) from e
    
    async def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_sources: bool = True,
        include_metadata: bool = True
    ) -> RAGResponse:
        """
        Execute complete RAG pipeline for a query.
        
        This is the main entry point for the RAG pipeline.
        
        Args:
            query: User question about Meditations
            top_k: Override default number of retrieval results
            include_sources: Include source passages in response
            include_metadata: Include metadata in context and response
            
        Returns:
            RAGResponse with answer and sources
            
        Example:
            >>> pipeline = RAGPipeline(llm, embedding_provider)
            >>> response = await pipeline.query("What does Marcus say about virtue?")
            >>> print(response.answer)
            >>> for source in response.sources:
            ...     print(f"  - {source['text'][:100]}...")
        """
        pipeline_start_time = time.time()
        
        logger.info(f"Starting RAG pipeline for query: '{query[:100]}...'")
        
        try:
            # Step 1: Retrieve relevant context
            logger.debug("Step 1: Retrieving relevant context")
            retrieval_start = time.time()
            
            retrieval_results = await self.retrieval_service.retrieve(
                query=query,
                top_k=top_k
            )
            
            retrieval_time = time.time() - retrieval_start
            
            logger.info(
                f"Retrieved {len(retrieval_results)} results in {retrieval_time:.2f}s"
            )
            
            if not retrieval_results:
                logger.warning("No results retrieved - returning empty answer")
                return RAGResponse(
                    query=query,
                    answer="I couldn't find any relevant passages in Meditations to answer your question. Could you rephrase or ask something else?",
                    sources=[],
                    retrieval_metadata={
                        "num_results": 0,
                        "retrieval_time_seconds": retrieval_time
                    },
                    generation_metadata={},
                    total_time_seconds=time.time() - pipeline_start_time
                )
            
            # Step 2: Format context
            logger.debug("Step 2: Formatting context for LLM")
            context = self._format_context(retrieval_results, include_metadata)
            
            # Step 3: Generate answer
            logger.debug("Step 3: Generating answer with LLM")
            answer, generation_metadata = await self._generate_answer(query, context)
            
            # Step 4: Build response
            logger.debug("Step 4: Building structured response")
            
            # Prepare sources if requested
            sources = []
            if include_sources:
                for result in retrieval_results:
                    source = {
                        "chunk_id": result.chunk_id,
                        "text": result.text,
                        "score": round(result.score, 4),
                    }
                    
                    # Add optional fields
                    if result.chunk_score is not None:
                        source["chunk_score"] = round(result.chunk_score, 4)
                    if result.question_score is not None:
                        source["question_score"] = round(result.question_score, 4)
                    if result.matched_question:
                        source["matched_question"] = result.matched_question
                    
                    # Add metadata if requested
                    if include_metadata and result.metadata:
                        # Include only relevant metadata fields
                        for key in ["topic", "book", "chapter", "keywords"]:
                            if key in result.metadata:
                                source[key] = result.metadata[key]
                    
                    sources.append(source)
            
            # Build retrieval metadata
            retrieval_metadata = {
                "num_results": len(retrieval_results),
                "retrieval_time_seconds": retrieval_time,
                "top_k": top_k or self.retrieval_service.top_k,
                "alpha": self.retrieval_service.alpha,
                "avg_score": sum(r.score for r in retrieval_results) / len(retrieval_results),
                "max_score": max(r.score for r in retrieval_results),
                "min_score": min(r.score for r in retrieval_results)
            }
            
            # Calculate total time
            total_time = time.time() - pipeline_start_time
            
            logger.info(f"RAG pipeline completed in {total_time:.2f}s")
            
            return RAGResponse(
                query=query,
                answer=answer,
                sources=sources,
                retrieval_metadata=retrieval_metadata,
                generation_metadata=generation_metadata,
                total_time_seconds=total_time
            )
            
        except VectorStoreException as e:
            logger.error(f"Vector store error in RAG pipeline: {e}")
            raise
        except LLMException as e:
            logger.error(f"LLM error in RAG pipeline: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in RAG pipeline: {e}", exc_info=True)
            raise MeditationsRAGException(
                f"RAG pipeline failed: {e}",
                details={"query": query, "error": str(e)}
            ) from e
    
    async def batch_query(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        include_sources: bool = True,
        include_metadata: bool = True,
        max_concurrent: int = 3
    ) -> List[RAGResponse]:
        """
        Execute RAG pipeline for multiple queries.
        
        Processes queries with controlled concurrency to avoid
        overwhelming API rate limits.
        
        Args:
            queries: List of user questions
            top_k: Override default number of retrieval results
            include_sources: Include source passages in responses
            include_metadata: Include metadata in context and responses
            max_concurrent: Maximum concurrent pipeline executions
            
        Returns:
            List of RAGResponse objects (same order as input queries)
        """
        import asyncio
        from asyncio import Semaphore
        
        logger.info(f"Starting batch RAG pipeline for {len(queries)} queries")
        
        semaphore = Semaphore(max_concurrent)
        
        async def process_with_semaphore(query: str) -> RAGResponse:
            async with semaphore:
                return await self.query(
                    query=query,
                    top_k=top_k,
                    include_sources=include_sources,
                    include_metadata=include_metadata
                )
        
        start_time = time.time()
        
        # Process all queries
        responses = await asyncio.gather(
            *[process_with_semaphore(q) for q in queries],
            return_exceptions=True
        )
        
        total_time = time.time() - start_time
        
        # Handle exceptions
        successful = sum(1 for r in responses if not isinstance(r, Exception))
        failed = len(responses) - successful
        
        logger.info(
            f"Batch RAG pipeline completed in {total_time:.2f}s: "
            f"{successful}/{len(queries)} successful, {failed} failed"
        )
        
        # Convert exceptions to error responses
        final_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Query {i+1} failed: {response}")
                final_responses.append(
                    RAGResponse(
                        query=queries[i],
                        answer=f"Error: {str(response)}",
                        sources=[],
                        retrieval_metadata={"error": str(response)},
                        generation_metadata={},
                        total_time_seconds=0.0
                    )
                )
            else:
                final_responses.append(response)
        
        return final_responses
    
    async def stream_answer(
        self,
        query: str,
        top_k: Optional[int] = None
    ):
        """
        Stream answer generation (if LLM supports streaming).
        
        Note: This is a placeholder for future streaming implementation.
        Current implementation returns the full answer at once.
        
        Args:
            query: User question
            top_k: Override default number of retrieval results
            
        Yields:
            Answer chunks as they're generated
        """
        logger.warning("Streaming not yet implemented - falling back to standard query")
        response = await self.query(query, top_k)
        yield response.answer


def create_rag_pipeline(
    llm: LLMBase,
    embedding_provider: EmbeddingBase,
    **kwargs
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline with standard configuration.
    
    Args:
        llm: LLM provider instance
        embedding_provider: Embedding provider instance
        **kwargs: Additional arguments to pass to RAGPipeline
        
    Returns:
        Configured RAGPipeline instance
        
    Example:
        >>> from meditations_rag.core.llm.openai import OpenAILLM
        >>> from meditations_rag.core.embedding.openai import OpenAIEmbedding
        >>> 
        >>> llm = OpenAILLM()
        >>> embedder = OpenAIEmbedding()
        >>> pipeline = create_rag_pipeline(llm, embedder)
        >>> 
        >>> response = await pipeline.query("What does Marcus say about virtue?")
    """
    logger.info("Creating RAG pipeline with factory function")
    return RAGPipeline(llm, embedding_provider, **kwargs)
