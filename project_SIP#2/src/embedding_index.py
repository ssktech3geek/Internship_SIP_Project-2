"""
embedding_index.py

Embeds document chunks with FinBERT and builds a FAISS index for
semantic retrieval.

WHY FinBERT over a generic sentence encoder: FinBERT is pretrained on
financial-domain text (earnings calls, filings, financial news), so its
embedding space places financially-related concepts closer together than
a generic encoder trained mostly on general web/Wikipedia text would.
This directly improves retrieval precision for financial queries.

WHY a flat FAISS index at this scale: IndexFlatL2 performs exact nearest-
neighbor search — no approximation error, simplest to reason about and
debug. This is the right choice at prototype/small-corpus scale (hundreds
to low thousands of chunks). At production scale (100k+ chunks), this
would be swapped for an IVF or HNSW index to keep search sub-linear —
noted here explicitly as a scaling decision, not implemented prematurely.
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch


@dataclass
class IndexedChunk:
    chunk_id: str
    text: str
    source_doc: str
    section: str


class FinBertEmbedder:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Mean-pools token embeddings from FinBERT's last hidden layer to
        produce one fixed-length vector per chunk. Mean pooling (rather
        than just the [CLS] token) is used because it tends to produce
        more robust sentence-level representations for similarity search,
        a well-established finding in sentence-embedding literature.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = summed / counts
            all_embeddings.append(mean_pooled.numpy())
        return np.vstack(all_embeddings).astype("float32")


class RetrievalIndex:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks: List[IndexedChunk] = []

    def add(self, embeddings: np.ndarray, chunks: List[IndexedChunk]):
        assert embeddings.shape[0] == len(chunks), "Embedding count must match chunk count"
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[IndexedChunk, float]]:
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(dist)))
        return results
