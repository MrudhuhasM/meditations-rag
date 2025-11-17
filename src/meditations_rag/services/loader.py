import asyncio
from llama_index.readers.file import PyMuPDFReader


class DocumentLoaderService:
    def __init__(self):
        self.reader = PyMuPDFReader()

    async def load_documents(self, file_path: str):
        return await asyncio.to_thread(self.reader.load_data, file_path)