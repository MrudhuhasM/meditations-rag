import asyncio

from meditations_rag.config import get_logger
from meditations_rag.services.vector_store import QdrantVectorStore

logger = get_logger(__name__)


async def create_indexes():
    """
    Script to manually create payload indexes on Qdrant.
    Useful when migrating to cloud or fixing missing index errors.
    """
    print("Initializing Qdrant connection...")
    try:
        store = QdrantVectorStore()
        print(f"Connected to Qdrant at {store.url or store.host}")

        print("Creating payload indexes...")
        store.create_payload_indexes()

        print("\nIndex creation process finished.")
        print("You can now run the application.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(create_indexes())
