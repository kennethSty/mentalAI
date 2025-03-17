import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores.chroma import Chroma
import numpy as np

class EmbedModelWrapper:
    """ Super Class Wrapping around a sentence transformer embedding model """
    def __init__(self, device, model, max_seq_length, verbose=False):
        if verbose:
            print("Initializing Embedder")
        self.device = device
        self.model = model
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        if verbose:
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
            device = device,
            model = SentenceTransformer(
            "pritamdeka/S-PubMedBert-MS-MARCO", device=device
            ),
            max_seq_length = 512
        )

class MiniLML6(EmbedModelWrapper):
    """ MiniLML6 Embedding Model """
    def __init__(self, device):
        super().__init__(
            device=device,
            model=SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
            ),
            max_seq_length=None
        )

class EmbeddingFunction(chromadb.EmbeddingFunction):
    """ Wraps an EmbedModelWrapper into a chromadb.EmbeddingFunction """
    def __init__(self, model: EmbedModelWrapper):
        self.model = model

    def __call__(self, input):
        return self.model.encode(input)

    def embed_query(self, query):
        return self.model.encode(query)
