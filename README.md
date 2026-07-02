# DomainGPT — Personal Financial Decision Assistant (Agentic MVP)

A deployed, **agentic QLoRA-fine-tuned + RAG + tool-calling** assistant that helps students and
early-career professionals make everyday financial decisions — "can I afford this phone, cash or EMI?",
"should I use a credit card?", "how much should I save?", "can I quit my job?" — by reasoning over their
actual personal context (income, expenses, debts, goals) with deterministic calculator tools, and
explaining general principles with citations rather than recommending specific investment products.
Built with a **production eval harness** and **MLOps**. The applied-AI flagship in Prashant's 2026 intern portfolio.

## Domain
**Personal financial decision-making for students/early professionals** — not a finance-facts chatbot, an
**agent that takes in someone's actual situation and reasons over it**. Real questions that motivated this
(asked by Prashant's own friends, verbatim-ish): *"we want to learn trading to earn money, our parents
don't earn much, what should we do," "I want to buy a phone, which one is a good investment, and should I
buy it on cash or EMI," "should I use a credit card," "how much should I save."*

**Scope decision (locked 2026-07-03):** this pivoted from a pure Q&A/RAG assistant into an agentic
decision assistant, but scoped as an **MVP, not a production product**, for the internship-timeline build.
See "What's explicitly out of scope for this build" below — accounts, persistent storage, and
monetization are deferred to a v2 vision, not part of this sprint.

**Why this is a stronger direction than plain Q&A:**
- Questions like "can I afford this phone" require reasoning over real numbers (income, expenses, existing
  debt) — a genuine case for **tool calling** (deterministic calculators), not just retrieval.
- It's the authentic use case — these are literally the questions Prashant's friends ask him.
- Tool-calling/agentic design is exactly the in-demand 2026 skill signal per `CONTEXT.md`'s market data —
  a step up from a static RAG-over-docs bot.

**The compliance line, restated precisely (this got sharper with the pivot):** affordability math,
EMI-vs-cash comparisons, budget splits, and savings-runway calculations are **arithmetic + general
principles** — safe territory, and the assistant can be directive here ("yes, you can afford this in 3
months" / "EMI costs you ₹X more than cash"). The line is **recommending specific investment products**
("buy this mutual fund," "invest in this stock") — that stays SEBI-RIA-regulated territory and the
assistant must stay educational/tradeoff-first there, same as before.

**Data sensitivity (new with this pivot, must be designed for, not bolted on):** income, expenses, debts,
and goals are sensitive personal data. The MVP is **session-based/ephemeral only** — no accounts, no
database of anyone's financial details, nothing persisted to disk. Logging must never store raw sensitive
fields in plaintext (see MLOps section in `ROADMAP.md`).

**Corpus (licensing-checked 2026-07-03 — see `CHECKPOINT.md` for the full check):**
- [NCFE — National Centre for Financial Education](https://ncfe.org.in/) — official RBI/SEBI/IRDAI/PFRDA-backed financial literacy body. **Primary source**, used for grounding explanations.
- SEBI investor education (investor.sebi.gov.in) — secondary source
- Wikipedia (CC BY-SA) — supplementary source for standard term definitions (SIP, PPF, NPS, EMI)
- Real friend questions (see above) — the authentic eval-set seed

**Rejected sources:** Zerodha Varsity (explicit "reproduction... is not permitted" copyright notice) and
RBI's main site (blocks non-browser/bot access outright). Don't scrape either.

**Target users:** Prashant's close circle — 4–5 friends to start, the same people already asking him these
questions in real life.

## What it is (the components)
1. **Personal context elicitation** — the agent asks for (or extracts from natural conversation) income, expenses, debts, goals — held in ephemeral session state only, never persisted
2. **Tool-calling calculators** — deterministic Python functions (affordability, EMI-vs-cash, budget/savings-rate split, job-quit runway) the LLM calls instead of doing arithmetic itself
3. **RAG** over a **vector database** of investor-education content, for grounding general-principle explanations with citations
4. **QLoRA fine-tune** of Llama — teaches when to ask a clarifying question vs. answer directly, which tool to call, and a structured tradeoff-first tone on the advice-boundary questions
5. **Production eval harness** — faithfulness, hallucination rate, retrieval precision@k, **tool-selection accuracy**, **context-elicitation appropriateness**, LLM-as-judge vs GPT
6. **MLOps** — latency/cost monitoring, quality-drift tracking, caching, fallback strategy, **privacy-aware logging** (no raw sensitive fields persisted)
7. **Deployment** — public app + real users (friends) + a feedback loop

## What's explicitly out of scope for this build (v2 vision, not blocking)
- User accounts / login
- Persistent budgeting or goal-tracking over time (anything requiring a database of someone's financial history)
- Monetization
These are real ambitions worth revisiting after the MVP ships and after internship-application season —
they change the risk profile (real financial data at rest, business/legal overhead) enough that they
shouldn't gate or blend into this sprint.

## The goal / what we are targeting
- **Primary:** a portfolio piece that wins **applied ML / AI Engineer / GenAI / MLOps** internships in 2026.
- **Secondary:** broad-acceptance — also reads well to SWE and inference screens (systems-thinking).
- It must out-signal a high-CPI generalist competitor whose "GenAI chatbot" is just an API wrapper.

## Definition of Done
- [ ] Public deployed app (live link) + real users (N+, starting with 4-5 friends)
- [ ] QLoRA fine-tuned model (uploaded to HuggingFace) beating a base/GPT baseline on a domain eval
- [ ] Working RAG over a real vector DB with measured retrieval precision
- [ ] Tool-calling calculators wired in and measured (tool-selection accuracy)
- [ ] Eval harness producing a metrics report (faithfulness, hallucination rate, precision@k, tool-selection accuracy, context-elicitation appropriateness)
- [ ] MLOps: monitoring dashboard + caching + fallback + privacy-aware logging (no raw sensitive fields stored)
- [ ] Clean README with architecture diagram + all metrics + demo
- [ ] Prashant can explain every component without notes (interview-ready)

## Status
In progress — build #2 in the pipeline (after the LocateAnything agent). Domain locked: personal financial
decision assistant, agentic MVP (pivoted 2026-07-02 from Indian Parliament Q&A → student personal-finance
literacy assistant → this agentic scope, confirmed 2026-07-03, on the day Day 1 build work started).
