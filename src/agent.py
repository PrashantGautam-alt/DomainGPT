"""The agent: native tool-calling loop over the deterministic calculators.

Flow (built from scratch, no agent framework):
  1. Send the conversation + tool schemas to the model.
  2. If the model responds with tool_calls, execute the matching Python function
     from tools.py and append the result as a `tool` message.
  3. Call the model again so it can compose a natural-language answer from the result.
  4. Repeat until the model answers with plain text (no more tool calls).

The model decides WHICH tool and WHAT arguments; tools.py does the arithmetic.
Context elicitation (asking when a required field is missing) is layered on in Day 6.
"""
import json
from dataclasses import asdict

from openai import BadRequestError

from generate import get_client, chat_completion
from context import FinancialContext, extract_context, PERSONAL_NUMERIC_FIELDS
from tools import (
    affordability_calculator,
    emi_vs_cash_calculator,
    budget_split_calculator,
    job_quit_runway_calculator,
)

TOOL_FUNCTIONS = {
    "affordability_calculator": affordability_calculator,
    "emi_vs_cash_calculator": emi_vs_cash_calculator,
    "budget_split_calculator": budget_split_calculator,
    "job_quit_runway_calculator": job_quit_runway_calculator,
}

SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_financial_knowledge",
        "description": "Search the investor-education knowledge base (NCFE/SEBI/Wikipedia) for general financial principles, definitions, and tradeoffs. Use for explanation questions like 'should I use a credit card', 'what is an emergency fund', 'is trading a good idea' — anything not a personal-number calculation. Returns relevant sourced passages to ground your answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up in the knowledge base"},
            },
            "required": ["query"],
        },
    },
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "affordability_calculator",
            "description": "Determine whether the user can afford a purchase from one month's surplus, and if not, how many months of saving it would take. Use when the user asks 'can I afford X'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "income": {"type": "number", "description": "Monthly income in rupees"},
                    "monthly_expenses": {"type": "number", "description": "Monthly living expenses in rupees"},
                    "existing_debt_payment": {"type": "number", "description": "Existing monthly debt/EMI payments in rupees (0 if none)"},
                    "item_cost": {"type": "number", "description": "Cost of the item to buy in rupees"},
                },
                "required": ["income", "monthly_expenses", "existing_debt_payment", "item_cost"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emi_vs_cash_calculator",
            "description": "Compare buying an item on EMI (loan) vs. paying cash, accounting for EMI interest and the opportunity cost of the cash. Use for 'should I buy on EMI or cash'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_cost": {"type": "number", "description": "Item price in rupees"},
                    "tenure_months": {"type": "integer", "description": "EMI tenure in months"},
                    "interest_rate_annual": {"type": "number", "description": "Annual EMI interest rate in percent"},
                    "opportunity_cost_rate_annual": {"type": "number", "description": "Annual return the cash could earn if invested instead, in percent (use ~8 if unknown)"},
                },
                "required": ["item_cost", "tenure_months", "interest_rate_annual", "opportunity_cost_rate_annual"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "budget_split_calculator",
            "description": "Compute a 50/30/20 budget breakdown (needs/wants/savings) from income, and the user's current savings rate if expenses are given. Use for 'how much should I save' / 'how should I budget'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "income": {"type": "number", "description": "Monthly income in rupees"},
                    "monthly_expenses": {"type": "number", "description": "Monthly expenses in rupees (optional)"},
                },
                "required": ["income"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "job_quit_runway_calculator",
            "description": "Compute how many months of expenses the user's savings cover with no income, and whether that covers an expected income gap. Use for 'can I afford to quit my job' / 'how long will my savings last'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "savings": {"type": "number", "description": "Total liquid savings in rupees"},
                    "monthly_expenses": {"type": "number", "description": "Monthly expenses in rupees"},
                    "expected_income_gap_months": {"type": "number", "description": "Expected months until new income starts"},
                },
                "required": ["savings", "monthly_expenses", "expected_income_gap_months"],
            },
        },
    },
]

WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the live web for a real-world fact you don't have, such as a product's current price (e.g. a bike, phone, or laptop). Use this BEFORE asking the user for a fact you could look up yourself. Returns short text snippets from the top results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up on the web, e.g. 'Royal Enfield Bullet 350 on-road price India'"},
            },
            "required": ["query"],
        },
    },
}

AGENT_SYSTEM_PROMPT = """You are NomosAI, a goal-oriented financial-decision assistant for students \
and early-career professionals in India. Reason like a smart human expert, not a form-filling chatbot.

Follow this decision process on every turn:
1. Identify the user's GOAL from the conversation and keep it active. Never lose track of what they \
are trying to work out, and never restart the conversation.
2. Check what you ALREADY KNOW from the conversation so far (the running notes are given below). \
Never ask for something the user already told you.
3. If a fact is missing but you can look it up yourself (like a product's price), use the web_search \
tool instead of asking the user.
4. Only if information is still missing and you cannot look it up, ask the user for the SINGLE most \
important missing thing. One short question at a time, never a list of fields.
5. As soon as you have enough to give a useful answer, ANSWER. Do not keep asking questions once you \
can make meaningful progress. It is better to answer with a stated assumption than to over-ask.

Tools: use the calculators for arithmetic (affordability, EMI-vs-cash, budgeting, job-quit runway) \
rather than computing in your head; use search_financial_knowledge for general financial principles; \
use web_search for real-world facts like prices. Never invent a personal financial value (income, \
expenses, debt, savings); those come only from what the user tells you.

Style: explain tradeoffs, be concise and practical. Never recommend a specific stock or fund to buy \
that stays educational only. Do not use the long dash character in your replies; write with commas, \
periods, or short dashes instead.
"""


def _execute_tool_call(name: str, args: dict) -> str:
    func = TOOL_FUNCTIONS[name]
    result = func(**args)
    return json.dumps(asdict(result))


def _strip_em_dash(text: str | None) -> str | None:
    """Remove the long dash characters the user asked never to show in the UI."""
    if not text:
        return text
    return text.replace(" — ", ", ").replace("—", "-").replace(" – ", ", ").replace("–", "-")


def _normalize_history(conversation) -> list[dict]:
    """Accept either a single user string (eval path) or a full [{role, content}] history
    (product path) and return clean {role, content} turns for the model."""
    if isinstance(conversation, str):
        return [{"role": "user", "content": conversation}]
    return [{"role": m["role"], "content": m["content"]} for m in conversation
            if m.get("role") in ("user", "assistant") and m.get("content")]


def run_agent(
    conversation,
    context: FinancialContext | None = None,
    retriever=None,
    web_search=None,
    provider: str = "groq",
    model: str = "llama-3.1-8b-instant",
    max_iterations: int = 6,
    decide_only: bool = False,
) -> dict:
    """Goal-oriented tool-calling loop over the FULL conversation history.

    `conversation` is either the latest user string (eval path) or a list of
    {role, content} turns (product path) so the model keeps the goal + what's known in view
    and never re-asks. `retriever` adds the RAG knowledge tool; `web_search` adds live web
    lookup (for facts like prices). Returns {"answer", "tool_calls", "asked_for", "sources"}.

    Missing personal values are never guessed: they come from `context` (extraction), and if a
    calculator needs one we don't have, the model is told to ask the user for that single thing.
    decide_only=True stops after the first tool decision (eval tool-selection).
    """
    client = get_client(provider)
    context = context if context is not None else FinancialContext()
    history = _normalize_history(conversation)
    latest_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    extract_context(latest_user, context, client, model)

    tool_schemas = list(TOOL_SCHEMAS)
    if retriever is not None:
        tool_schemas.append(SEARCH_TOOL_SCHEMA)
    if web_search is not None:
        tool_schemas.append(WEB_SEARCH_TOOL_SCHEMA)

    system_prompt = AGENT_SYSTEM_PROMPT + "\n\nRunning notes on this user: " + context.summary()
    messages = [{"role": "system", "content": system_prompt}, *history]
    tools_used, sources, seen_urls, pending_missing = [], [], set(), []

    for _ in range(max_iterations):
        try:
            response = chat_completion(
                client, model=model, messages=messages, tools=tool_schemas,
                tool_choice="auto", temperature=0.3,
            )
        except BadRequestError as e:
            if "tool_use_failed" not in str(e):
                raise
            response = chat_completion(client, model=model, messages=messages, temperature=0.3)
            return {"answer": _strip_em_dash(response.choices[0].message.content),
                    "tool_calls": tools_used, "asked_for": pending_missing,
                    "sources": sources, "degraded": True}
        msg = response.choices[0].message

        if not msg.tool_calls:
            return {"answer": _strip_em_dash(msg.content), "tool_calls": tools_used,
                    "asked_for": pending_missing, "sources": sources}

        if decide_only:
            return {"answer": None, "tool_calls": [tc.function.name for tc in msg.tool_calls],
                    "asked_for": [], "sources": sources}

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            name = tc.function.name
            tools_used.append(name)
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}

            if name == "search_financial_knowledge" and retriever is not None:
                chunks = retriever(args.get("query", ""))
                passages = []
                for i, ch in enumerate(chunks, 1):
                    passages.append(f"[{i}] {ch['chunk_text'][:600]}")
                    if ch["source_url"] not in seen_urls:
                        sources.append({"title": ch["title"], "source_url": ch["source_url"]})
                        seen_urls.add(ch["source_url"])
                result_json = "\n\n".join(passages)
            elif name == "web_search" and web_search is not None:
                found = web_search(args.get("query", ""))
                result_json = found or "No results found. Ask the user for this fact instead."
            else:
                # Personal values come only from context (never the model's guess).
                for pf in PERSONAL_NUMERIC_FIELDS:
                    if pf in args and getattr(context, pf) is not None:
                        args[pf] = getattr(context, pf)
                missing = context.missing_personal_for(name)
                if missing:
                    # Don't run with missing personal info; have the model ask for ONE thing.
                    pending_missing = missing
                    result_json = json.dumps({
                        "error": f"missing required info: {missing[0]}",
                        "instruction": "Ask the user for just this one thing in a short question. Do not guess it.",
                    })
                else:
                    try:
                        result_json = _execute_tool_call(name, args)
                    except (TypeError, ValueError) as e:
                        result_json = json.dumps({"error": str(e)})

            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "name": name, "content": result_json})

    return {"answer": "I need a bit more detail to answer that well. Could you rephrase or add one specific number?",
            "tool_calls": tools_used, "asked_for": pending_missing, "sources": sources}


def make_retriever(k: int = 5):
    """Load the index + embedding model once and return a callable(query) -> chunks."""
    from retrieval import load_index, load_embedding_model, retrieve_top_k
    index, chunks = load_index()
    embedding_model = load_embedding_model()
    return lambda query: retrieve_top_k(query, index, chunks, embedding_model, k=k)


# Fallback chain: try each (provider, model) in order until one answers. Order = best/most
# specialised first, most robust last. The fine-tuned model (once served) goes first; base
# Llama-3.1-8B is the everyday model; the 70B is the sturdier last resort. Keeps the app
# alive if the primary is down or erroring.
DEFAULT_FALLBACK_CHAIN = [
    ("groq", "llama-3.1-8b-instant"),
    ("groq", "llama-3.3-70b-versatile"),
]


def run_agent_with_fallback(conversation, context=None, retriever=None, web_search=None,
                            chain=DEFAULT_FALLBACK_CHAIN, **kwargs) -> dict:
    """Run the agent across a fallback chain of (provider, model) until one succeeds.
    Returns the result dict with an added 'model' field naming which one answered."""
    last_error = None
    for provider, model in chain:
        try:
            out = run_agent(conversation, context=context, retriever=retriever,
                            web_search=web_search, provider=provider, model=model, **kwargs)
            out["model"] = model
            return out
        except Exception as e:  # provider down, rate-limited past retries, etc.
            last_error = e
            continue
    raise RuntimeError(f"All fallback models failed. Last error: {last_error}")


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else \
        "I earn 15000 a month, spend 10000, no debt. Can I afford a 20000 phone?"
    retriever = make_retriever()
    out = run_agent(q, retriever=retriever)
    print("Q:", q)
    print("Tools used:", out["tool_calls"])
    print("Asked for:", out["asked_for"])
    print("Sources:", [s["title"] for s in out["sources"]])
    print("A:", out["answer"])
