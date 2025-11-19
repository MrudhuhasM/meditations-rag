"""
Document Loading Service.

This module handles loading documents from file paths using LlamaIndex readers.
Currently supports PDF loading via PyMuPDFReader.
"""

import asyncio
import os
from typing import List
from llama_index.core.schema import Document
from llama_index.readers.file import PyMuPDFReader
from meditations_rag.config import get_logger

logger = get_logger(__name__)


class DocumentLoaderService:
    """
    Service for loading documents from files.
    
    Wraps LlamaIndex readers to provide a consistent async interface
    for document loading.
    """
    
    def __init__(self):
        """Initialize document loader."""
        self.reader = PyMuPDFReader()
        logger.info("Initialized DocumentLoaderService with PyMuPDFReader")

    async def load_documents(self, file_path: str) -> List[Document]:
        """
        Load documents from a file path asynchronously.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            List of loaded documents
            
        Raises:
            FileNotFoundError: If file does not exist
            Exception: If loading fails
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.info(f"Loading documents from: {file_path}")
        
        try:
            # Run blocking IO in a separate thread
            documents = await asyncio.to_thread(self.reader.load_data, file_path)
            logger.info(f"Successfully loaded {len(documents)} documents/pages")
            return documents
        except Exception as e:
            logger.error(f"Failed to load documents from {file_path}: {e}")
            raise