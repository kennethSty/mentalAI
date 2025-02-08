import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores.chroma import Chroma
import numpy as np


class PubMedBert:
    def __init__(self, device):
        print("Initializing Embedder")
        self.device = device
        self.model = SentenceTransformer(
            "pritamdeka/S-PubMedBert-MS-MARCO", device=self.device
        )
        self.model.max_seq_length = 512
        print("Embedder initialized")

    def encode(self, doc_batch):
        batch_size = len(doc_batch)
        embeddings = self.model.encode(
            doc_batch, device=self.device, batch_size=batch_size
        )
        return np.stack(embeddings, axis=0).tolist()


class PubMedEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        return self.model.encode(input)

    def embed_query(self, query):
        return self.model.encode(query)

class ChromaDBManager:
    """Manages ChromaDB initialization and operations"""
    def __init__(self, device: str, persist_dir: str = "../../data/chroma_store"):
        """
        Initializes ChromaDB with PubMedBert embeddings.
        :param device: The device for PubMedBert (e.g., 'cpu' or 'cuda').
        :param persist_dir: Directory to persist ChromaDB storage.
        """
        self.device = device
        self.model = PubMedBert(device=self.device)
        self.embed_fn = PubMedEmbeddingFunction(self.model)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = "pubmed_embeddings"

    def get_chroma_collection(self) -> Chroma:
        """
        Returns the ChromaDB collection for managing PubMed embeddings.
        :return: Chroma collection object.
        """
        return Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embed_fn,
            collection_metadata={"hnsw:space": "cosine"},
        )
