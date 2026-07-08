"""Semantic cache — restricted to non-personal answers only.

Embeds the incoming query and returns a cached answer if a previous query is semantically
near-identical (cosine similarity >= threshold). Reuses the SAME embedding model as
retrieval (bge-small), so vectors are normalized and cosine == dot product.

CRITICAL RESTRICTION for this project: NEVER cache or key on the user's personal financial
context. Two users asking "can I afford this phone" with different incomes must not collide.
So the cache is only consulted/populated for general-principle answers where NO personal
context was used and NO calculator tool ran. Enforced by the caller passing cacheable=False
whenever context was provided or a calculator was called.
"""
import numpy as np

SIMILARITY_THRESHOLD = 0.95


class SemanticCache:
    def __init__(self, embedding_model, threshold: float = SIMILARITY_THRESHOLD):
        self.model = embedding_model
        self.threshold = threshold
        self._vectors: list[np.ndarray] = []
        self._entries: list[dict] = []  # {"answer", "sources"}

    def _embed(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True).astype("float32")[0]

    def get(self, query: str) -> dict | None:
        if not self._vectors:
            return None
        q = self._embed(query)
        sims = np.array([float(np.dot(q, v)) for v in self._vectors])  # cosine (normalized)
        best = int(sims.argmax())
        if sims[best] >= self.threshold:
            return self._entries[best]
        return None

    def put(self, query: str, answer: str, sources: list[dict]) -> None:
        self._vectors.append(self._embed(query))
        self._entries.append({"answer": answer, "sources": sources})

    def __len__(self) -> int:
        return len(self._entries)
