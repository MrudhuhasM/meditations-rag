from abc import ABC, abstractmethod


class EmbeddingBase(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embeddings for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Integer dimension of embedding vectors
        """
        pass
