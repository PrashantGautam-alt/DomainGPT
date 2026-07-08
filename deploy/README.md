---
title: DomainGPT
emoji: 💰
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
short_description: Financial-literacy assistant for students (educational only)
---

# DomainGPT

A financial-literacy assistant for students and early-career professionals in India.
Retrieval-augmented answers grounded in NCFE / SEBI / Wikipedia investor-education
sources, with citations. **Educational information only — not personalized investment advice.**

This Space runs the RAG-only version. The FAISS index is downloaded at startup from a
private HuggingFace Dataset; the generation model is served via Groq.

## Space secrets required
- `GROQ_API_KEY` — for answer generation (Llama-3.1-8B via Groq)
- `HF_TOKEN` — to download the private index Dataset at startup
