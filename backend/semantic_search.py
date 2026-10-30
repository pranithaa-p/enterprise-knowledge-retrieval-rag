
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class SemanticSearch:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.embeddings = None

    def build_index(self, chunks: list[dict]):
        if not chunks:
            self.index = None
            self.chunks = []
            self.embeddings = None
            return

        self.chunks = chunks

        texts = [chunk["text"] for chunk in chunks]

        self.embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index is None or not self.chunks:
            return []

        query_vector = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        k = min(top_k, len(self.chunks))

        scores, indices = self.index.search(query_vector, k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            result = self.chunks[int(index)].copy()
            result["score"] = round(float(score), 4)
            results.append(result)

        return results
