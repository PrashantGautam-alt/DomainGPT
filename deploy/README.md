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

An agentic financial-decision assistant for students and early-career professionals in India.
Combines tool-calling calculators (affordability, EMI-vs-cash, budgeting, job-quit runway),
context elicitation (asks for missing numbers instead of guessing), and RAG grounded in
NCFE / SEBI / Wikipedia investor-education sources with citations.
**Educational information only — not personalized investment advice.**

Runs on CPU: the LLM is a Groq API call, so no GPU is needed. The FAISS index is downloaded at
startup from a private HuggingFace Dataset. The QLoRA fine-tuned model + its eval numbers live
separately on the Hub (huggingface.co/prashantgautam8077/domaingpt-v1).

## Space secrets required
- `GROQ_API_KEY` — for answer generation (Llama-3.1-8B via Groq)
- `HF_TOKEN` — to download the private index Dataset at startup
