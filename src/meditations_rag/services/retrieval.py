"""
Retrieval Service for RAG Pipeline.

Implements hybrid retrieval strategy with metadata-aware enhancement:
1. Dense retrieval from chunks (semantic similarity on chunk text)
2. Dense retrieval from questions (semantic similarity on generated questions)
3. Metadata-filtered retrieval (keywords, topics, concepts, practices)
4. Score fusion combining all retrieval methods with metadata-based boosting

Design Principles:
- Single Responsibility: Only handles retrieval and ranking
- Dependency Inversion: Uses abstractions for vector store and embeddings
- Configurable: Alpha parameter controls chunk vs question weighting
- Extensible: Supports metadata-based filtering and boosting
"""

from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from meditations_rag.services.vector_store import QdrantVectorStore, VectorEmbeddingService
from meditations_rag.services.metadata import MeditationsTopic
from meditations_rag.config import get_logger, settings
from meditations_rag.core.exceptions import VectorStoreQueryError
from meditations_rag.core.llm.base import LLMBase
from qdrant_client.models import ScoredPoint, Filter, FieldCondition, MatchValue, MatchAny
import asyncio
from enum import Enum


logger = get_logger(__name__)


class QueryMetadata(BaseModel):
    """
    Metadata extracted from user query for enhanced retrieval.
    
    Used for:
    - Metadata-filtered retrieval
    - Score boosting based on metadata matches
    - Multi-faceted retrieval strategy
    """
    
    rewritten_query: str = Field(description="Rewritten/clarified version of the original query")
    keywords: List[str] = Field(default_factory=list, description="Key terms extracted from query")
    topic: Optional[MeditationsTopic] = Field(default=None, description="Primary topic if identifiable")
    entities: List[str] = Field(default_factory=list, description="Named entities mentioned")
    philosophical_concepts: List[str] = Field(default_factory=list, description="Stoic/philosophical concepts")
    stoic_practices: List[str] = Field(default_factory=list, description="Stoic practices mentioned")


class QueryRewriter:
    """
    Service for query rewriting and metadata extraction.
    
    Takes raw user queries and:
    1. Rewrites them for better semantic matching
    2. Extracts metadata for filtered retrieval
    3. Identifies key concepts for score boosting
    """
    
    def __init__(self, llm: LLMBase):
        """
        Initialize query rewriter.
        
        Args:
            llm: LLM implementation for structured generation
        """
        self.llm = llm
        self.available_topics = [topic.value for topic in MeditationsTopic]
        logger.info("Initialized QueryRewriter")
    
    async def rewrite_and_extract(self, query: str) -> QueryMetadata:
        """
        Rewrite query and extract metadata for enhanced retrieval.
        
        Args:
            query: Raw user query
            
        Returns:
            QueryMetadata with rewritten query and extracted metadata
        """
        topics_str = "\n".join([f"- {topic}" for topic in self.available_topics])
        
        prompt = f"""You are analyzing a user question about Marcus Aurelius' "Meditations" to optimize retrieval.

**User Question:**
{query}

**Task:**
1. **Rewrite the query** to be clearer and more specific for semantic search. Make it a complete, well-formed question or statement.

2. **Extract keywords** (3-8 terms): Key concepts, important verbs, distinctive terms that would appear in relevant passages.

3. **Identify topic** (if applicable): Select ONE topic from this list if the query clearly relates to it, otherwise leave null:
{topics_str}

4. **Extract entities**: Any people, places, philosophical schools, or specific references mentioned.

5. **Identify philosophical concepts**: Abstract Stoic or philosophical concepts the query is about (e.g., 'virtue', 'logos', 'apatheia').

6. **Identify stoic practices**: Any Stoic exercises or practical techniques the query asks about (e.g., 'negative visualization', 'view from above').

**Guidelines:**
- Be precise and specific
- Only extract what's actually present or strongly implied
- Rewritten query should preserve the user's intent while being more retrievable
- If uncertain about topic, leave it null rather than guessing

Provide high-quality metadata to maximize retrieval precision."""

        try:
            result = await self.llm.generate_structured(
                prompt=prompt,
                response_model=QueryMetadata
            )
            
            if not isinstance(result, QueryMetadata):
                raise ValueError("Invalid query metadata type returned")
            
            logger.info(
                f"Query rewritten: '{query[:50]}...' -> '{result.rewritten_query[:50]}...'"
            )
            logger.debug(
                f"Extracted metadata: keywords={len(result.keywords)}, "
                f"topic={result.topic}, concepts={len(result.philosophical_concepts)}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            # Fallback: return query as-is with minimal metadata
            logger.warning("Falling back to original query without metadata")
            return QueryMetadata(
                rewritten_query=query,
                keywords=[],
                topic=None,
                entities=[],
                philosophical_concepts=[],
                stoic_practices=[]
            )


class RetrievalResult(BaseModel):
    """Single retrieval result with metadata."""
    
    chunk_id: str = Field(description="ID of the retrieved chunk")
    score: float = Field(description="Final fused score")
    chunk_score: Optional[float] = Field(default=None, description="Score from chunk collection")
    question_score: Optional[float] = Field(default=None, description="Score from question collection")
    text: str = Field(description="Retrieved chunk text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    matched_question: Optional[str] = Field(default=None, description="Question that matched (if from question retrieval)")
    
    class Config:
        arbitrary_types_allowed = True


class RetrievalService:
    """
    Hybrid retrieval service with metadata-aware enhancement.
    
    Implements multi-faceted retrieval:
    1. Semantic search over chunk embeddings (direct content matching)
    2. Semantic search over question embeddings (conceptual matching)
    3. Metadata-filtered retrieval (keywords, topics, concepts, practices)
    4. Score fusion with metadata-based boosting
    
    Score Fusion Formula:
        final_score = chunk_score + λ * question_score + boost_score
        
    Where:
    - λ (lambda/alpha) controls the weight of question-based retrieval
    - boost_score is added when metadata matches (configurable per field)
    """
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_service: VectorEmbeddingService,
        query_rewriter: Optional[QueryRewriter] = None,
        alpha: float = 0.3,
        top_k: int = 6,
        question_top_k: Optional[int] = None,
        metadata_top_k: int = 5,
        enable_metadata_retrieval: bool = True,
        metadata_boost_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize retrieval service.
        
        Args:
            vector_store: Qdrant vector store instance
            embedding_service: Embedding service for query encoding
            query_rewriter: Optional query rewriting service for metadata extraction
            alpha: Weight for question scores in fusion (0.0 to 1.0)
            top_k: Number of final results to return (default 6 for metadata-aware)
            question_top_k: Number of results to fetch from question collection
            metadata_top_k: Number of results per metadata filter
            enable_metadata_retrieval: Enable metadata-filtered retrieval
            metadata_boost_weights: Score boost per metadata field match
                                   (e.g., {'keyword': 0.1, 'topic': 0.2})
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.query_rewriter = query_rewriter
        self.alpha = alpha
        self.top_k = top_k
        self.question_top_k = question_top_k or top_k
        self.metadata_top_k = metadata_top_k
        self.enable_metadata_retrieval = enable_metadata_retrieval
        
        # Default metadata boost weights
        self.metadata_boost_weights = metadata_boost_weights or {
            'keyword': 0.05,  # Small boost per keyword match
            'topic': 0.15,    # Moderate boost for topic match
            'concept': 0.10,  # Moderate boost for concept match
            'practice': 0.10, # Moderate boost for practice match
        }
        
        # Validate alpha parameter
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}")
        
        logger.info(
            f"Initialized RetrievalService: alpha={alpha}, top_k={top_k}, "
            f"question_top_k={self.question_top_k}, metadata_retrieval={enable_metadata_retrieval}"
        )
    
    async def _retrieve_from_chunks(
        self,
        query: str,
        top_k: int
    ) -> List[ScoredPoint]:
        """
        Retrieve results from the main chunks collection.
        
        Args:
            query: User query string
            top_k: Number of results to retrieve
            
        Returns:
            List of ScoredPoint objects from Qdrant
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_texts([query])
            query_vector = query_embedding[0]
            
            # Query chunks collection
            logger.debug(f"Querying chunks collection with top_k={top_k}")
            results = self.vector_store.client.query_points(
                collection_name=self.vector_store.main_collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            ).points
            
            logger.info(f"Retrieved {len(results)} results from chunks collection")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve from chunks collection: {e}")
            raise VectorStoreQueryError(f"Chunk retrieval failed: {e}") from e
    
    async def _retrieve_from_questions(
        self,
        query: str,
        top_k: int
    ) -> List[ScoredPoint]:
        """
        Retrieve results from the questions collection.
        
        Args:
            query: User query string
            top_k: Number of results to retrieve
            
        Returns:
            List of ScoredPoint objects from Qdrant
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_texts([query])
            query_vector = query_embedding[0]
            
            # Query questions collection
            logger.debug(f"Querying questions collection with top_k={top_k}")
            results = self.vector_store.client.query_points(
                collection_name=self.vector_store.question_collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            ).points
            
            logger.info(f"Retrieved {len(results)} results from questions collection")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve from questions collection: {e}")
            raise VectorStoreQueryError(f"Question retrieval failed: {e}") from e
    
    async def _retrieve_with_metadata_filter(
        self,
        query_vector: List[float],
        filter_field: str,
        filter_values: List[str],
        top_k: int
    ) -> List[ScoredPoint]:
        """
        Retrieve results with metadata filtering.
        
        Args:
            query_vector: Query embedding vector
            filter_field: Metadata field to filter on (e.g., 'keywords', 'topic')
            filter_values: Values to match (OR condition if multiple)
            top_k: Number of results to retrieve
            
        Returns:
            List of ScoredPoint objects matching the filter
        """
        try:
            # Build Qdrant filter
            # For array fields (keywords, concepts, practices), use MatchAny
            # For single fields (topic), use MatchValue
            if filter_field in ['keywords', 'philosophical_concepts', 'stoic_practices', 'entities']:
                # Array field - match any of the values
                filter_condition = Filter(
                    must=[
                        FieldCondition(
                            key=filter_field,
                            match=MatchAny(any=filter_values)
                        )
                    ]
                )
            else:
                # Single value field (topic)
                if len(filter_values) == 1:
                    filter_condition = Filter(
                        must=[
                            FieldCondition(
                                key=filter_field,
                                match=MatchValue(value=filter_values[0])
                            )
                        ]
                    )
                else:
                    # Multiple topic values (shouldn't happen, but handle it)
                    filter_condition = Filter(
                        must=[
                            FieldCondition(
                                key=filter_field,
                                match=MatchAny(any=filter_values)
                            )
                        ]
                    )
            
            logger.debug(
                f"Querying with filter: {filter_field} in {filter_values[:3]}... "
                f"(top_k={top_k})"
            )
            
            results = self.vector_store.client.query_points(
                collection_name=self.vector_store.main_collection,
                query=query_vector,
                query_filter=filter_condition,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            ).points
            
            logger.debug(
                f"Retrieved {len(results)} results with {filter_field} filter"
            )
            return results
            
        except Exception as e:
            logger.warning(
                f"Metadata-filtered retrieval failed for {filter_field}: {e}"
            )
            # Return empty list on filter failure (graceful degradation)
            return []
    
    async def _retrieve_metadata_aware(
        self,
        query_vector: List[float],
        query_metadata: QueryMetadata,
        top_k_per_filter: int
    ) -> Dict[str, List[ScoredPoint]]:
        """
        Retrieve results using metadata-based filtering.
        
        Performs parallel retrieval across different metadata facets:
        - Keywords: Top-K for each keyword
        - Topic: Top-K for the identified topic
        - Concepts: Top-K for each philosophical concept
        - Practices: Top-K for each stoic practice
        
        Args:
            query_vector: Query embedding vector
            query_metadata: Extracted query metadata
            top_k_per_filter: Number of results per metadata filter
            
        Returns:
            Dictionary mapping metadata type to retrieved results
        """
        retrieval_tasks = []
        metadata_types = []
        
        # Keyword-based retrieval (top-N keywords)
        if query_metadata.keywords:
            # Limit to top 3-5 keywords to avoid too many requests
            top_keywords = query_metadata.keywords[:5]
            if top_keywords:
                retrieval_tasks.append(
                    self._retrieve_with_metadata_filter(
                        query_vector, 'keywords', top_keywords, top_k_per_filter
                    )
                )
                metadata_types.append('keywords')
        
        # Topic-based retrieval
        if query_metadata.topic:
            retrieval_tasks.append(
                self._retrieve_with_metadata_filter(
                    query_vector, 'topic', [query_metadata.topic.value], top_k_per_filter
                )
            )
            metadata_types.append('topic')
        
        # Concept-based retrieval
        if query_metadata.philosophical_concepts:
            retrieval_tasks.append(
                self._retrieve_with_metadata_filter(
                    query_vector, 'philosophical_concepts', 
                    query_metadata.philosophical_concepts, top_k_per_filter
                )
            )
            metadata_types.append('concepts')
        
        # Practice-based retrieval
        if query_metadata.stoic_practices:
            retrieval_tasks.append(
                self._retrieve_with_metadata_filter(
                    query_vector, 'stoic_practices',
                    query_metadata.stoic_practices, top_k_per_filter
                )
            )
            metadata_types.append('practices')
        
        if not retrieval_tasks:
            logger.info("No metadata filters applicable, skipping metadata retrieval")
            return {}
        
        logger.info(
            f"Running {len(retrieval_tasks)} metadata-filtered retrievals: "
            f"{', '.join(metadata_types)}"
        )
        
        # Execute all metadata retrievals in parallel
        results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        
        # Build results dictionary
        metadata_results = {}
        for metadata_type, result in zip(metadata_types, results):
            if isinstance(result, Exception):
                logger.warning(f"Metadata retrieval failed for {metadata_type}: {result}")
                metadata_results[metadata_type] = []
            elif isinstance(result, list):
                metadata_results[metadata_type] = result
                logger.debug(
                    f"Retrieved {len(result)} results for {metadata_type}"
                )
            else:
                logger.warning(f"Unexpected result type for {metadata_type}: {type(result)}")
                metadata_results[metadata_type] = []
        
        return metadata_results
    
    def _fuse_scores_with_metadata(
        self,
        chunk_results: List[ScoredPoint],
        question_results: List[ScoredPoint],
        metadata_results: Dict[str, List[ScoredPoint]],
        query_metadata: Optional[QueryMetadata] = None
    ) -> List[RetrievalResult]:
        """
        Fuse scores from all retrieval sources with metadata-based boosting.
        
        Formula: final_score = chunk_score + α * question_score + boost_score
        
        Where boost_score = Σ(weight_i * match_count_i) for each metadata field
        
        Args:
            chunk_results: Results from chunks collection
            question_results: Results from questions collection
            metadata_results: Results from metadata-filtered retrievals
            query_metadata: Extracted query metadata for boost calculation
            
        Returns:
            List of RetrievalResult objects sorted by fused score
        """
        logger.debug(
            f"Fusing scores: {len(chunk_results)} chunk results, "
            f"{len(question_results)} question results, "
            f"{sum(len(v) for v in metadata_results.values())} metadata results, "
            f"alpha={self.alpha}"
        )
        
        # Build score maps
        chunk_scores: Dict[str, tuple] = {}  # chunk_id -> (score, point)
        for point in chunk_results:
            chunk_id = str(point.id)
            chunk_scores[chunk_id] = (point.score, point)
        
        question_scores: Dict[str, tuple] = {}  # chunk_id -> (score, question, point)
        for point in question_results:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            if chunk_id:
                chunk_id = str(chunk_id)
                if chunk_id not in question_scores or point.score > question_scores[chunk_id][0]:
                    matched_question = payload.get("question", "")
                    question_scores[chunk_id] = (point.score, matched_question, point)
        
        # Track metadata matches for boosting
        metadata_matches: Dict[str, Set[str]] = {}  # chunk_id -> set of metadata_types matched
        
        for metadata_type, points in metadata_results.items():
            for point in points:
                chunk_id = str(point.id)
                if chunk_id not in metadata_matches:
                    metadata_matches[chunk_id] = set()
                metadata_matches[chunk_id].add(metadata_type)
        
        # Collect all unique chunk IDs
        all_chunk_ids = (
            set(chunk_scores.keys()) | 
            set(question_scores.keys()) | 
            set(metadata_matches.keys())
        )
        
        logger.debug(
            f"Found {len(all_chunk_ids)} unique chunks across all retrievals "
            f"(chunks: {len(chunk_scores)}, questions: {len(question_scores)}, "
            f"metadata_matches: {len(metadata_matches)})"
        )
        
        # Fuse scores for each chunk
        fused_results = []
        
        for chunk_id in all_chunk_ids:
            chunk_score_val = chunk_scores.get(chunk_id, (0.0, None))[0]
            question_info = question_scores.get(chunk_id, (0.0, None, None))
            question_score_val = question_info[0]
            matched_question = question_info[1] if len(question_info) > 1 else None
            
            # Calculate metadata boost
            boost_score = 0.0
            matched_metadata_types = metadata_matches.get(chunk_id, set())
            
            for metadata_type in matched_metadata_types:
                # Map metadata type to boost weight
                if metadata_type == 'keywords':
                    boost_score += self.metadata_boost_weights.get('keyword', 0.05)
                elif metadata_type == 'topic':
                    boost_score += self.metadata_boost_weights.get('topic', 0.15)
                elif metadata_type == 'concepts':
                    boost_score += self.metadata_boost_weights.get('concept', 0.10)
                elif metadata_type == 'practices':
                    boost_score += self.metadata_boost_weights.get('practice', 0.10)
            
            # Apply fusion formula
            final_score = chunk_score_val + self.alpha * question_score_val + boost_score
            
            # Get chunk data (prefer from chunk_results, fallback to others)
            chunk_point = None
            if chunk_id in chunk_scores:
                chunk_point = chunk_scores[chunk_id][1]
            elif chunk_id in question_scores:
                chunk_point = question_scores[chunk_id][2]
            else:
                # Get from any metadata result
                for points in metadata_results.values():
                    for point in points:
                        if str(point.id) == chunk_id:
                            chunk_point = point
                            break
                    if chunk_point:
                        break
            
            if chunk_point is None:
                logger.warning(f"No point data found for chunk_id {chunk_id}, skipping")
                continue
            
            # Extract text and metadata
            payload = chunk_point.payload or {}
            text = payload.get("text") or payload.get("chunk_text", "")
            
            # Build metadata (exclude internal fields)
            metadata = {
                k: v for k, v in payload.items()
                if k not in ["text", "chunk_text", "question", "chunk_id"]
            }
            
            result = RetrievalResult(
                chunk_id=str(chunk_id),
                score=final_score,
                chunk_score=chunk_score_val if chunk_score_val > 0 else None,
                question_score=question_score_val if question_score_val > 0 else None,
                text=text,
                metadata=metadata,
                matched_question=matched_question
            )
            
            fused_results.append(result)
        
        # Sort by fused score (descending) and limit to top_k
        fused_results.sort(key=lambda x: x.score, reverse=True)
        fused_results = fused_results[:self.top_k]
        
        logger.info(
            f"Score fusion complete: {len(fused_results)} results. "
            f"Top score: {fused_results[0].score:.4f}" if fused_results else "No results"
        )
        
        return fused_results
    
    def _fuse_scores(
        self,
        chunk_results: List[ScoredPoint],
        question_results: List[ScoredPoint]
    ) -> List[RetrievalResult]:
        """
        Fuse scores from chunk and question retrievals (no metadata).
        
        This is a simplified version without metadata boosting.
        Use _fuse_scores_with_metadata for full metadata-aware retrieval.
        
        Formula: final_score = chunk_score + α * question_score
        
        Args:
            chunk_results: Results from chunks collection
            question_results: Results from questions collection
            
        Returns:
            List of RetrievalResult objects sorted by fused score
        """
        return self._fuse_scores_with_metadata(
            chunk_results, question_results, {}, None
        )
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        chunk_only: bool = False,
        question_only: bool = False,
        use_metadata: Optional[bool] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve and rank chunks using metadata-aware hybrid retrieval.
        
        Retrieval Strategy:
        1. Query rewriting + metadata extraction (if query_rewriter available)
        2. Dense retrieval from chunks collection (rewritten query embedding)
        3. Dense retrieval from questions collection (rewritten query embedding)
        4. Metadata-filtered retrieval (keywords, topic, concepts, practices)
        5. Score fusion with metadata-based boosting
        
        Args:
            query: User query string
            top_k: Override default top_k for this query
            alpha: Override default alpha for this query
            chunk_only: Only retrieve from chunks (disable question/metadata retrieval)
            question_only: Only retrieve from questions (disable chunk/metadata retrieval)
            use_metadata: Override enable_metadata_retrieval for this query
            
        Returns:
            List of RetrievalResult objects sorted by score
        """
        if chunk_only and question_only:
            raise ValueError("Cannot specify both chunk_only and question_only")
        
        # Use instance defaults or overrides
        k = top_k or self.top_k
        current_alpha = alpha if alpha is not None else self.alpha
        metadata_enabled = (
            use_metadata if use_metadata is not None 
            else self.enable_metadata_retrieval
        )
        
        # Step 1: Query rewriting + metadata extraction
        query_metadata: Optional[QueryMetadata] = None
        effective_query = query
        
        if metadata_enabled and self.query_rewriter and not (chunk_only or question_only):
            logger.info(f"Rewriting query and extracting metadata: '{query[:100]}...'")
            try:
                query_metadata = await self.query_rewriter.rewrite_and_extract(query)
                effective_query = query_metadata.rewritten_query
            except Exception as e:
                logger.warning(f"Query rewriting failed, using original query: {e}")
                effective_query = query
        
        logger.info(
            f"Retrieving for query: '{effective_query[:100]}...' "
            f"(top_k={k}, alpha={current_alpha}, chunk_only={chunk_only}, "
            f"question_only={question_only}, metadata_enabled={metadata_enabled})"
        )
        
        # Step 2-4: Parallel retrieval from all sources
        if question_only:
            # Only question retrieval
            question_results = await self._retrieve_from_questions(effective_query, k)
            chunk_results = []
            metadata_results = {}
        elif chunk_only:
            # Only chunk retrieval
            chunk_results = await self._retrieve_from_chunks(effective_query, k)
            question_results = []
            metadata_results = {}
        else:
            # Generate query embedding once for all retrievals
            query_embedding = await self.embedding_service.embed_texts([effective_query])
            query_vector = query_embedding[0]
            
            # Build parallel retrieval tasks as async functions
            async def retrieve_chunks():
                return self.vector_store.client.query_points(
                    collection_name=self.vector_store.main_collection,
                    query=query_vector,
                    limit=k,
                    with_payload=True,
                    with_vectors=False
                )
            
            async def retrieve_questions():
                return self.vector_store.client.query_points(
                    collection_name=self.vector_store.question_collection,
                    query=query_vector,
                    limit=self.question_top_k,
                    with_payload=True,
                    with_vectors=False
                )
            
            retrieval_tasks = []
            task_names = []
            
            # Add chunk retrieval task
            retrieval_tasks.append(retrieve_chunks())
            task_names.append('chunks')
            
            # Add question retrieval task
            retrieval_tasks.append(retrieve_questions())
            task_names.append('questions')
            
            # Metadata-filtered retrieval (if enabled and metadata available)
            if metadata_enabled and query_metadata:
                retrieval_tasks.append(
                    self._retrieve_metadata_aware(
                        query_vector, query_metadata, self.metadata_top_k
                    )
                )
                task_names.append('metadata')
            
            logger.info(f"Running {len(retrieval_tasks)} parallel retrievals: {', '.join(task_names)}")
            
            # Execute all retrievals in parallel
            results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
            
            # Unpack results
            chunk_response = results[0]
            question_response = results[1]
            metadata_results = results[2] if len(results) > 2 else {}
            
            # Handle exceptions and extract points
            if isinstance(chunk_response, Exception):
                logger.error(f"Chunk retrieval failed: {chunk_response}")
                chunk_results = []
            else:
                # chunk_response is QueryResponse
                chunk_results = getattr(chunk_response, 'points', [])
                logger.info(f"Retrieved {len(chunk_results)} chunks")
            
            if isinstance(question_response, Exception):
                logger.error(f"Question retrieval failed: {question_response}")
                question_results = []
            else:
                # question_response is QueryResponse
                question_results = getattr(question_response, 'points', [])
                logger.info(f"Retrieved {len(question_results)} questions")
            
            if isinstance(metadata_results, Exception):
                logger.warning(f"Metadata retrieval failed: {metadata_results}")
                metadata_results = {}
            elif isinstance(metadata_results, dict):
                total_metadata = sum(len(v) for v in metadata_results.values())
                logger.info(f"Retrieved {total_metadata} metadata-filtered chunks")
            else:
                logger.warning(f"Unexpected metadata result type: {type(metadata_results)}")
                metadata_results = {}
        
        # Step 5: Fuse scores
        if chunk_only or question_only:
            # Single-source retrieval - no fusion needed
            results_list = []
            source_results = chunk_results if chunk_only else question_results
            
            for point in source_results[:k]:
                payload = point.payload or {}
                text = payload.get("text") or payload.get("chunk_text", "")
                metadata = {
                    k: v for k, v in payload.items()
                    if k not in ["text", "chunk_text", "question", "chunk_id"]
                }
                
                # For question-only, extract chunk_id from payload
                chunk_id = str(point.id) if chunk_only else payload.get("chunk_id", str(point.id))
                matched_question = payload.get("question") if question_only else None
                
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    score=point.score,
                    chunk_score=point.score if chunk_only else None,
                    question_score=point.score if question_only else None,
                    text=text,
                    metadata=metadata,
                    matched_question=matched_question
                )
                results_list.append(result)
            
            logger.info(f"Single-source retrieval complete: {len(results_list)} results")
            return results_list
        else:
            # Hybrid retrieval with metadata-aware fusion
            if metadata_results:
                return self._fuse_scores_with_metadata(
                    chunk_results, question_results, metadata_results, query_metadata
                )
            else:
                # No metadata results, use standard fusion
                return self._fuse_scores(chunk_results, question_results)
    
    async def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve chunks and format as context for LLM prompting.
        
        Args:
            query: User query string
            top_k: Number of results to retrieve
            alpha: Score fusion weight
            include_metadata: Include metadata in context
            
        Returns:
            Dictionary with formatted context and retrieval metadata
        """
        results = await self.retrieve(query, top_k=top_k, alpha=alpha)
        
        # Format context for LLM
        context_parts = []
        
        for i, result in enumerate(results, 1):
            context_block = f"[Context {i}]\n{result.text}\n"
            
            if include_metadata and result.metadata:
                # Add relevant metadata
                if "topic" in result.metadata:
                    context_block += f"Topic: {result.metadata['topic']}\n"
                if "keywords" in result.metadata:
                    keywords = ", ".join(result.metadata.get("keywords", []))
                    context_block += f"Keywords: {keywords}\n"
                if result.matched_question:
                    context_block += f"Related Question: {result.matched_question}\n"
            
            context_parts.append(context_block)
        
        formatted_context = "\n".join(context_parts)
        
        return {
            "query": query,
            "context": formatted_context,
            "num_results": len(results),
            "results": [r.dict() for r in results],
            "retrieval_metadata": {
                "top_k": top_k or self.top_k,
                "alpha": alpha or self.alpha,
                "avg_score": sum(r.score for r in results) / len(results) if results else 0.0,
                "max_score": max(r.score for r in results) if results else 0.0,
                "min_score": min(r.score for r in results) if results else 0.0
            }
        }
