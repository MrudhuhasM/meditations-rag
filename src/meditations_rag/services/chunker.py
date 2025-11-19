"""
Document Chunking Service.

This module provides semantic chunking capabilities using LlamaIndex's
SemanticSplitterNodeParser. It processes documents in batches to manage
memory and handle errors gracefully.
"""

from typing import List
from llama_index.core.schema import Document, BaseNode
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.embeddings import BaseEmbedding
from meditations_rag.config import settings, get_logger

logger = get_logger(__name__)


class ChunkerService:
    """
    Service for chunking documents into semantic nodes.
    
    Uses semantic splitting to keep related concepts together based on
    embedding similarity.
    """
    
    def __init__(self, embed_model: BaseEmbedding):
        """
        Initialize chunker service.
        
        Args:
            embed_model: LlamaIndex embedding model for semantic splitting
        """
        self.node_parser = SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=settings.rag.buffer_size,
            breakpoint_percentile_threshold=settings.rag.break_point_threshold,
        )
        logger.info(
            f"Initialized ChunkerService with buffer_size={settings.rag.buffer_size}, "
            f"threshold={settings.rag.break_point_threshold}"
        )

    def _chunk_batch(self, documents: List[Document]) -> List[BaseNode]:
        """
        Process a single batch of documents.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked nodes (empty list on failure)
        """
        try:
            chunked_docs = self.node_parser.get_nodes_from_documents(documents)
            return chunked_docs
        except Exception as e:
            logger.error(f"Error during chunking batch: {e}")
            return []

    def chunk_documents(self, documents: List[Document]) -> List[BaseNode]:
        """
        Chunk a list of documents into semantic nodes.
        
        Processes documents in batches and monitors failure rates.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of all generated chunks
            
        Raises:
            Exception: If failure rate exceeds threshold
        """
        if not documents:
            logger.warning("No documents provided for chunking")
            return []
            
        logger.info(f"Starting document chunking for {len(documents)} documents...")
        
        batch_size = settings.rag.batch_size
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

        logger.info(f"Processing {len(documents)} documents in {len(batches)} batches")

        failure_count = 0
        chunks = []
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} documents")
            chunked_docs = self._chunk_batch(batch)
            
            if not chunked_docs and batch: # Only count as failure if batch was not empty but result is empty
                # Note: get_nodes_from_documents might return empty if docs are empty or very short
                # But here we assume it's a failure if we expected chunks
                failure_count += 1
                logger.warning(f"Batch {i+1} failed to chunk or produced no chunks. Total failures: {failure_count}")
            
            logger.info(f"Completed batch {i+1}/{len(batches)}: Generated {len(chunked_docs)} chunks")
            chunks.extend(chunked_docs)

        # Calculate failure rate
        failure_rate = failure_count / len(batches) if batches else 0

        if failure_rate > settings.rag.failure_threshold:
            logger.error(
                f"Chunking failed: {failure_count}/{len(batches)} batches failed "
                f"({failure_rate:.1%}), exceeding threshold ({settings.rag.failure_threshold:.1%})"
            )
            raise Exception("Chunking process failed due to high failure rate.")
        
        logger.info(f"Chunking completed: {len(chunks)} chunks created with {failure_count} failures.")
        return chunks


