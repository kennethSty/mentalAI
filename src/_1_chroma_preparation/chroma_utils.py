from langchain_chroma import Chroma
import chromadb

from src._1_chroma_preparation.embed_utils import EmbedModelWrapper, EmbeddingFunction

class ChromaCollectionManager:
    """Manages ChromaDB initialization and operations"""
    def __init__(self, persist_dir: str):
        """
        Initializes ChromaDB client.
        :param persist_dir: Directory to persist ChromaDB storage.
        """
        self.client = chromadb.PersistentClient(path=persist_dir)

    def create_empty_collection(self, collection_name: str):
        try:
            self.client.delete_collection(name=collection_name)
        except ValueError:
            pass

        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

        return collection

    def get_chroma_collection(self, collection_name: str, embed_fn: EmbeddingFunction) -> Chroma:
        """
        Returns the ChromaDB collection identified with collection_name
        :return: Chroma collection object.
        """
        return Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=embed_fn, #embed_fn setup: EmbeddingFunction(EmbedModelWrapper)
            collection_metadata={"hnsw:space": "cosine"},
        )
