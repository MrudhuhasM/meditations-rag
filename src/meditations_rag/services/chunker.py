from llama_index.core.node_parser import SemanticSplitterNodeParser
from meditations_rag.config import settings, get_logger

logger = get_logger(__name__)


class ChunkerService:
    
    def __init__(self, embed_model):
            
        self.node_parser = SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=settings.rag.buffer_size,
            breakpoint_percentile_threshold=settings.rag.break_point_threshold,
        )

    def _chunk_batch(self, documents):
        try:
            chunked_docs = self.node_parser.get_nodes_from_documents(documents)
            return chunked_docs
        except Exception as e:
            logger.error(f"Error during chunking batch: {e}")
            return []

    def chunk_documents(self, documents):
        logger.info("Starting document chunking...")
        
        batches = [documents[i:i + settings.rag.batch_size] for i in range(0, len(documents), settings.rag.batch_size)]

        logger.info(f"processing {len(documents)} in {len(batches)} batches")

        failure_count = 0
        chunks = []
        
        for i,batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} documents")
            chunked_docs = self._chunk_batch(batch)
            if not chunked_docs:
                failure_count += 1
                logger.warning(f"Batch {i+1} failed to chunk. Total failures: {failure_count}")
            logger.info(f"Completed batch {i+1}/{len(batches)}: Generated {len(chunked_docs)} chunks")
            chunks.extend(chunked_docs)

        failure_threshold = failure_count / len(batches)

        if failure_threshold > settings.rag.failure_threshold:
            logger.error(f"Chunking failed: {failure_count} out of {len(batches)} batches failed, exceeding threshold.")
            raise Exception("Chunking process failed due to high failure rate.")
        
        logger.info(f"Chunking completed: {len(chunks)} chunks created with {failure_count} failures.")
        return chunks
            


