import asyncio


class DocumentLoaderService:
    def __init__(self, reader):
        self.reader = reader

    def load_documents(self, file_path: str):
        return asyncio.to_thread(self.reader.load, file_path)