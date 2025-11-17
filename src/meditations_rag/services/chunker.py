import asyncio
from llama_index.core.node_parser import SemanticSplitterNodeParser
from meditations_rag.config import settings, get_logger
from meditations_rag.core.chunk_embedding import get_chunk_embedding_model

logger = get_logger(__name__)

# Preload NLTK data to avoid lazy loading issues with multiprocessing/threading
# This ensures NLTK resources are loaded in the main thread before any concurrent processing
try:
    import nltk
    # Force load the punkt tokenizer and stopwords to avoid lazy loading
    nltk.data.find('tokenizers/punkt')
except (LookupError, ImportError):
    # If NLTK data is not found, it will be downloaded on first use
    # The lazy loader will handle it, but may cause issues with threading
    pass

class ChunkerService:
    """
    Service for chunking documents using semantic splitting.
    
    This service uses SemanticSplitterNodeParser with configurable
    embedding models to intelligently split documents at semantic boundaries.
    """
    
    def __init__(self, embed_model=None):
        """
        Initialize chunker service.
        
        Args:
            embed_model: Optional embedding model. If None, uses default from settings.
        """
        if embed_model is None:
            embed_model = get_chunk_embedding_model()
            
        self.node_parser = SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=settings.rag.buffer_size,
            breakpoint_percentile_threshold=settings.rag.break_point_threshold,
        )

    def _process_batch_sync(self, batch):
        """
        Synchronously process a single batch.
        
        This method is designed to be called from within a thread/process pool.
        It must be a regular function (not async) to work with executors.
        
        Args:
            batch: List of documents to process
            
        Returns:
            List of nodes
        """
        return self.node_parser.get_nodes_from_documents(batch)

    async def _process_batch_with_semaphore(self, batch, batch_idx, total_batches, semaphore):
        """
        Process a single batch with semaphore-controlled concurrency.
        
        Args:
            batch: List of documents to process
            batch_idx: Index of this batch (0-based)
            total_batches: Total number of batches
            semaphore: Asyncio semaphore for concurrency control
            
        Returns:
            Tuple of (success: bool, nodes: List or None, error: Exception or None)
        """
        async with semaphore:
            try:
                # Process batch synchronously within the semaphore context
                # This avoids threading/pickling issues with NLTK
                nodes = self._process_batch_sync(batch)
                logger.debug(f"Batch {batch_idx + 1}/{total_batches} processed successfully, added {len(nodes)} nodes")
                return (True, nodes, None)
            except Exception as batch_error:
                logger.warning(f"Chunking batch {batch_idx + 1}/{total_batches} failed with error: {str(batch_error)}")
                return (False, None, batch_error)

    async def chunk_documents(self, documents):
        """
        Chunk documents into semantic nodes concurrently with failure tolerance.
        
        Processes documents in batches with semaphore-controlled concurrency.
        Allows silent failures up to the configured failure threshold before aborting.
        
        Implementation Note:
        - Uses asyncio.Semaphore to limit concurrent batch processing
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of Node objects representing semantic chunks
            
        Raises:
            Exception: If failure rate exceeds the configured threshold
        """
        try:
            if not documents:
                logger.info("No documents provided for chunking")
                return []
            
            batches = [
                documents[i:i + settings.rag.batch_size]
                for i in range(0, len(documents), settings.rag.batch_size)
            ]
            
            logger.info(f"Starting to chunk {len(documents)} documents in {len(batches)} batches with max {settings.rag.max_concurrent_requests} concurrent requests")
            
            # Create semaphore to limit concurrent batch processing
            semaphore = asyncio.Semaphore(settings.rag.max_concurrent_requests)
            
            # Create tasks for all batches
            tasks = [
                self._process_batch_with_semaphore(batch, i, len(batches), semaphore)
                for i, batch in enumerate(batches)
            ]
            
            # Execute all tasks concurrently (controlled by semaphore)
            results = await asyncio.gather(*tasks)
            
            # Collect successful results and count failures
            all_nodes = []
            failures = 0
            
            for success, nodes, error in results:
                if success and nodes is not None:
                    all_nodes.extend(nodes)
                else:
                    failures += 1
            
            total_batches = len(batches)
            failure_rate = failures / total_batches if total_batches > 0 else 0
            
            logger.info(f"Chunking completed: {len(all_nodes)} total nodes from {total_batches - failures}/{total_batches} successful batches")
            
            if failure_rate > settings.rag.failure_threshold:
                error_msg = f"Chunking failure rate {failure_rate:.2%} exceeds threshold {settings.rag.failure_threshold:.2%} ({failures}/{total_batches} batches failed)"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            return all_nodes
            
        except Exception as e:
            logger.error(f"Unexpected error during document chunking: {str(e)}")
            raise