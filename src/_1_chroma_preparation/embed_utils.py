import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores.chroma import Chroma
import numpy as np

class EmbedModelWrapper:
    """ Super Class Wrapping around a sentence transformer embedding model """
    def __init__(self, device, model, max_seq_length):
        print("Initializing Embedder")
        self.device = device
        self.model = model
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        print("Embedder initialized")

    def encode(self, doc_batch):
        """ Embeds a batch of documents """
        batch_size = len(doc_batch)
        embeddings = self.model.encode(
            doc_batch, device=self.device, batch_size=batch_size
        )
        return np.stack(embeddings, axis=0).tolist()

class PubMedBert(EmbedModelWrapper):
    """ PubMedBert Embedding Model """
    def __init__(self, device):
        super().__init__(
            self,
            device = device,
            model = SentenceTransformer(
            "pritamdeka/S-PubMedBert-MS-MARCO", device=self.device
            ),
            max_seq_length = 512
        )

class MiniLML6(EmbedModelWrapper):
    """ MiniLML6 Embedding Model """
    def __init__(self, device):
        super().__init__(
            self,
            device = device,
            model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=self.device
            ),
            max_seq_length = None
        )

class EmbeddingFunction(chromadb.EmbeddingFunction):
    """ Wraps an EmbedModelWrapper into a chromadb.EmbeddingFunction """
    def __init__(self, model: EmbedModelWrapper):
        self.model = model

    def __call__(self, input):
        return self.model.encode(input)

    def embed_query(self, query):
        return self.model.encode(query)

class ChromaDBManager:
    """Manages ChromaDB initialization and operations"""
    def __init__(self, model: EmbedModelWrapper, device: str, persist_dir: str, collection_name: str):
        """
        Initializes ChromaDB with embeddings.
        :param device: The device for PubMedBert (e.g., 'cpu' or 'cuda').
        :param persist_dir: Directory to persist ChromaDB storage.
        """
        self.device = device
        self.model = model
        self.embed_fn = EmbeddingFunction(self.model)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name

    def get_chroma_collection(self) -> Chroma:
        """
        Returns the ChromaDB collection for managing embeddings.
        :return: Chroma collection object.
        """
        return Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embed_fn,
            collection_metadata={"hnsw:space": "cosine"},
        )
