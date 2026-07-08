from pathlib import Path
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "embeddings"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_DATASET = os.environ.get("INDEX_DATASET", "prashantgautam8077/domaingpt-index")


def _ensure_index_files(embeddings_dir: Path) -> None:
    """If the index isn't on disk (e.g. a fresh HF Space), download it from the
    private HF Dataset. Locally, files already exist and this is a no-op."""
    if (embeddings_dir / "corpus.index").exists() and (embeddings_dir / "chunks.json").exists():
        return
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["corpus.index", "chunks.json"]:
        downloaded = hf_hub_download(
            repo_id=INDEX_DATASET, filename=fname, repo_type="dataset",
            token=token, local_dir=str(embeddings_dir),
        )
        print(f"Downloaded {fname} from {INDEX_DATASET}")


def load_index(embeddings_dir: Path = EMBEDDINGS_DIR) -> tuple[faiss.Index, list[dict]]:
    _ensure_index_files(embeddings_dir)
    index = faiss.read_index(str(embeddings_dir / "corpus.index"))
    chunks = json.loads((embeddings_dir / "chunks.json").read_text())
    return index, chunks


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    return model.encode([query], normalize_embeddings=True).astype("float32")


def retrieve_top_k(query: str, index: faiss.Index, chunks: list[dict], model: SentenceTransformer, k: int = 5) -> list[dict]:
    query_vector = embed_query(query, model)
    distances, indices = index.search(query_vector, k)
    results = []
    for rank, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        results.append({**chunk, "distance": float(distances[0][rank])})
    return results
