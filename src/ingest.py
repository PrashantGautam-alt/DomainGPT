import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "embeddings"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Articles under this length are embedded whole. Above it, split by heading
# (falls back to fixed-size for headingless text, e.g. SEBI's long article).
SINGLE_CHUNK_THRESHOLD = 1800
FIXED_CHUNK_SIZE = 1200
FIXED_CHUNK_OVERLAP = 150
BOILERPLATE_HEADINGS = {"see also", "references", "further reading", "external links", "notes", "bibliography"}


def load_documents(data_dir: Path) -> list[dict]:
    ncfe = json.loads((data_dir / "ncfe_faqs.json").read_text())
    sebi = json.loads((data_dir / "sebi_articles.json").read_text())
    wiki = json.loads((data_dir / "wikipedia_articles.json").read_text())

    documents = []
    for d in ncfe:
        documents.append({
            "text": f"Q: {d['question']}\nA: {d['answer']}",
            "title": d["question"],
            "source_url": d["source_url"],
            "source_type": "ncfe",
            "category": d["category"],
        })
    for d in sebi:
        documents.append({
            "text": d["text"],
            "title": d["title"],
            "source_url": d["source_url"],
            "source_type": "sebi",
            "category": None,
        })
    for d in wiki:
        documents.append({
            "text": d["text"],
            "title": d["title"],
            "source_url": d["source_url"],
            "source_type": "wikipedia",
            "category": None,
        })
    return documents


def fixed_size_split(text: str, chunk_size: int = FIXED_CHUNK_SIZE, overlap: int = FIXED_CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    pieces = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return pieces


def split_by_heading(text: str) -> list[tuple[str, str]]:
    # MediaWiki API text uses "== Heading ==" for top-level sections. The
    # leading/trailing "\n" in the pattern is what keeps this matching only
    # level-2 markers, not "===" subheadings (those never have a bare "\n== "
    # or " ==\n" boundary around them).
    parts = re.split(r"\n== (.+?) ==\n", text)
    if len(parts) == 1:
        return [("", text)]
    sections = [("", parts[0])] if parts[0].strip() else []
    for i in range(1, len(parts), 2):
        heading = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((heading, content))
    return sections


def chunk_document(doc: dict) -> list[dict]:
    text = doc["text"]
    if doc["source_type"] == "ncfe" or len(text) <= SINGLE_CHUNK_THRESHOLD:
        return [{**doc, "chunk_text": text, "section": None}]

    chunks = []
    for heading, content in split_by_heading(text):
        if heading.strip().lower() in BOILERPLATE_HEADINGS:
            continue
        content = content.strip()
        if not content:
            continue
        section_text = f"{doc['title']} — {heading}\n{content}" if heading else content
        for piece in fixed_size_split(section_text):
            chunks.append({**doc, "chunk_text": piece, "section": heading or None})
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    chunks = []
    for doc in documents:
        chunks.extend(chunk_document(doc))
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = i
        del chunk["text"]
    return chunks


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    texts = [c["chunk_text"] for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(vectors, dtype="float32")


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index: faiss.Index, chunks: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "corpus.index"))
    (out_dir / "chunks.json").write_text(json.dumps(chunks, indent=2))


def main():
    documents = load_documents(DATA_DIR)
    chunks = chunk_documents(documents)
    print(f"{len(documents)} documents -> {len(chunks)} chunks")

    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = embed_chunks(chunks, model)
    index = build_faiss_index(vectors)
    save_index(index, chunks, OUT_DIR)
    print(f"Saved FAISS index ({index.ntotal} vectors, dim={vectors.shape[1]}) to {OUT_DIR}")


if __name__ == "__main__":
    main()
