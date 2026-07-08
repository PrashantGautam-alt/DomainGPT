from fastapi import FastAPI
from pydantic import BaseModel

from retrieval import load_index, load_embedding_model, retrieve_top_k
from generate import build_prompt, generate_response

app = FastAPI(title="DomainGPT")

_index, _chunks = load_index()
_embedding_model = load_embedding_model()


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    results = retrieve_top_k(request.query, _index, _chunks, _embedding_model, k=5)
    messages, sources = build_prompt(request.query, results)
    answer = generate_response(messages)
    return AskResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok"}
