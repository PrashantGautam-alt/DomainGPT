# DomainGPT — Agentic Personal-Finance Assistant

A deployed **agentic** assistant that helps students and early-career professionals in India make
everyday money decisions — *"can I afford this phone, cash or EMI?"*, *"should I use a credit card?"*,
*"how much should I save?"*, *"should we start trading to earn money?"* — by combining four techniques,
each doing a job the others can't:

- **RAG** grounds general financial principles in cited investor-education sources (NCFE / SEBI / Wikipedia)
- **Tool-calling calculators** do the arithmetic **deterministically** (never the LLM)
- **Context elicitation** asks for missing personal numbers instead of guessing them
- A **QLoRA fine-tune** of Llama-3.1-8B teaches the behavioral policy (when to ask, which tool, tradeoff-first tone, and the educational-only advice boundary)

Shipped with a **production eval harness** and **MLOps** (privacy-aware logging, restricted semantic
caching, a model fallback chain, and an ops dashboard). Personal data is **ephemeral only** — never
written to disk or a database.

> 📄 **[DEEP_DIVE.md](../DEEP_DIVE.md)** explains every component and design decision in interview-defense depth.

## Live demo
🔗 **[domaingpt-hkf4czjwdsbyxxxsbniddi.streamlit.app](https://domaingpt-hkf4czjwdsbyxxxsbniddi.streamlit.app/)**
· Educational information only — not personalized investment advice.

The QLoRA fine-tuned model is on the Hub:
**[huggingface.co/prashantgautam8077/domaingpt-v1](https://huggingface.co/prashantgautam8077/domaingpt-v1)**.
(The live app serves generation via Groq — a CPU-friendly API call; the fine-tuned model + its
eval numbers are the separate evidence on the Hub.)

---

## Architecture

```
Streamlit UI ── FinancialContext in session_state (ephemeral, never persisted)
     │
     ▼
service.answer() ─ semantic cache (general-principle answers only) ─ privacy-aware logging
     │
     ▼
agent (native tool-calling loop, from scratch — no framework) + fallback chain
   ├─ extract_context() → fill personal fields from the message
   ├─ missing a required personal field? → ASK, don't guess
   ├─ search_financial_knowledge → FAISS retrieval over the corpus → cited chunks
   └─ calculator → deterministic Python (tools.py), personal args taken from context
     │
     ▼
FAISS IndexFlatL2 (271 chunks, bge-small-en-v1.5) built by ingest.py from NCFE/SEBI/Wikipedia
```

Generation is served by **Groq** (Llama-3.1-8B, OpenAI-compatible API); the fine-tuned model is
uploaded to the HuggingFace Hub.

---

## Results

**Retrieval** (shared by all models — FAISS over the corpus): precision@5 = **0.667**, MRR = **0.833**.

**Zero-shot behavior — base vs. QLoRA fine-tuned** (same prompt, same eval set, measured with
`src/eval_finetuned.py` on the server): the headline result is what the fine-tune taught.

| Metric | Base Llama-3.1-8B | **QLoRA fine-tuned** |
|---|---|---|
| Calculator tool-selection accuracy | 0.00 | **1.00** |
| Context-elicitation ask-rate | 0.80 | **1.00** |
| Context-elicitation proceed-rate | 0.00 | **1.00** |

Base Llama, given only the system prompt, never emits a structured tool call (it answers calc
questions in prose). After QLoRA fine-tuning it emits correct, well-formed tool calls
(`{"name": "affordability_calculator", "parameters": {…}}`) while *keeping* its educational tone
and advice-boundary compliance (refuses to recommend specific products).

> Two honest framings of tool-selection: **zero-shot** (prompt only — base 0.00 → fine-tuned 1.00,
> above) and **with the provider's tool-calling API** scaffolding the format (base already 0.895 via
> the Groq tools API — the agent uses this path in production). The fine-tune's contribution is doing
> zero-shot what previously needed the API. See DEEP_DIVE §9 for the full discussion.

Eval harness: `src/eval.py` (retrieval + agent tool-selection + elicitation via the tools API),
`src/eval_finetuned.py` (zero-shot base-vs-fine-tuned via transformers), `src/eval_part2.py`
(faithfulness + advice-boundary, LLM-judged). Eval set: `eval/eval_set.json` (24 labeled examples).

---

## Repository layout

| Path | What |
|---|---|
| `src/ingest.py` | Build the FAISS index — non-uniform chunking, bge embeddings |
| `src/retrieval.py` | Load index, embed query, top-k retrieval (+ runtime index download) |
| `src/generate.py` | Provider clients (Groq/OpenAI), compliance system prompt, rate-limit backoff |
| `src/tools.py` | The four deterministic calculators (+ `tests/test_tools.py`, 10/10) |
| `src/agent.py` | Native tool-calling loop, RAG-as-a-tool, fallback chain |
| `src/context.py` | `FinancialContext` + slot-filling extraction (ephemeral) |
| `src/cache.py` | Restricted semantic cache (general-principle answers only) |
| `src/monitor.py` | Privacy-aware request logging + cost/latency |
| `src/service.py` | Orchestration: cache → agent+fallback → cache-write → log |
| `src/eval.py`, `src/eval_part2.py` | Eval harness (Parts 1 & 2) |
| `src/prepare_training_data.py`, `src/train_qlora.py` | QLoRA data generation + training |
| `app/ui.py`, `app/dashboard.py` | Chat UI + ops dashboard |
| `deploy/` | HF Space files + `build_space.sh` |

---

## The compliance line

Arithmetic (affordability, EMI-vs-cash, budget splits, savings runway) can be **directive** once the
numbers are known. Recommending a **specific investment product** (a stock/fund to buy) stays
**educational only** — because SEBI RIA rules regulate personalized product advice, not general math.

## Out of scope (v2 vision, deliberately deferred)

User accounts, persistent budgeting/goal-tracking (a database of anyone's financial history), and
monetization. These change the risk profile (sensitive data at rest, legal overhead) enough that they
shouldn't blend into this internship-timeline MVP.

## Corpus & licensing

NCFE (primary, official financial-literacy body), SEBI investor-education (secondary), Wikipedia
(CC BY-SA, supplementary) — 82 documents → 271 chunks. **Zerodha Varsity** (explicit reproduction
prohibition) and **RBI** (blocks non-browser access) were checked and rejected.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill GROQ_API_KEY, OPENAI_API_KEY (optional), HF_TOKEN
python src/ingest.py             # build the FAISS index
streamlit run app/ui.py          # chat UI
python src/eval.py               # baseline metrics
```
