"""Upload the built FAISS index + chunk texts to a PRIVATE HuggingFace Dataset.

Why a private Dataset instead of committing the index into the public GitHub repo
or the Space repo: the chunks contain scraped third-party text (NCFE/SEBI/Wikipedia)
verbatim, which we don't redistribute in a browsable public repo (see .gitignore).
A private Dataset keeps that text non-public while the deployed Space can still pull
it at runtime with HF_TOKEN. This is the standard "artifact registry" pattern.

Run this AFTER src/ingest.py has produced embeddings/corpus.index + chunks.json.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, whoami

load_dotenv()

EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "embeddings"
DATASET_NAME = "domaingpt-index"


def main():
    token = os.environ["HF_TOKEN"]
    username = whoami(token=token)["name"]
    repo_id = f"{username}/{DATASET_NAME}"

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    print(f"Dataset repo ready (private): {repo_id}")

    for fname in ["corpus.index", "chunks.json"]:
        local_path = EMBEDDINGS_DIR / fname
        if not local_path.exists():
            raise FileNotFoundError(f"{local_path} missing — run src/ingest.py first")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded {fname}")

    print(f"\nDone. Deployed app will download from: {repo_id}")


if __name__ == "__main__":
    main()
