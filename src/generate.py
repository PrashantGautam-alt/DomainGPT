import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = """You are DomainGPT, a financial-literacy assistant for students and early-career \
professionals in India.

Answer using ONLY the provided context below. If the context doesn't contain enough information to \
answer, say so honestly instead of guessing.

Rules:
- Explain tradeoffs and general principles. Ground factual claims in the provided context.
- You may be directive about arithmetic (e.g. how much to save, EMI vs. cash affordability) once the \
relevant numbers are known.
- You must NOT recommend a specific stock, mutual fund, or investment product to buy. Redirect to \
general principles and note that specific product decisions should go through a registered investment \
advisor (SEBI RIA) — this is an educational-only boundary, not a formality.
- Keep answers concise and practical.
"""


def build_prompt(query: str, retrieved_chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    context_lines = []
    sources = []
    seen_urls = set()
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_lines.append(f"[{i}] {chunk['chunk_text']}")
        if chunk["source_url"] not in seen_urls:
            sources.append({"title": chunk["title"], "source_url": chunk["source_url"]})
            seen_urls.add(chunk["source_url"])

    context_block = "\n\n".join(context_lines)
    user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    return messages, sources


PROVIDER_CONFIG = {
    "openai": {"base_url": None, "api_key_env": "OPENAI_API_KEY"},
    # Groq serves open-weight models (including our own base model, Llama-3.1-8B-Instruct)
    # through an OpenAI-compatible endpoint, so the same client class works for both.
    "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
}


def get_client(provider: str) -> OpenAI:
    config = PROVIDER_CONFIG[provider]
    api_key = os.environ[config["api_key_env"]]
    return OpenAI(api_key=api_key, base_url=config["base_url"]) if config["base_url"] else OpenAI(api_key=api_key)


def generate_response(messages: list[dict], provider: str = "groq", model: str = "llama-3.1-8b-instant") -> str:
    client = get_client(provider)
    response = client.chat.completions.create(model=model, messages=messages, temperature=0.3)
    return response.choices[0].message.content
