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

from generate import get_client
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

AGENT_SYSTEM_PROMPT = """You are DomainGPT, a financial-decision assistant for students and \
early-career professionals in India. You have calculator tools for affordability, EMI-vs-cash, \
budgeting, and job-quit runway. Call the right tool for arithmetic questions rather than computing \
numbers yourself. Never invent a personal financial value (income, expenses, debt, savings) — if \
one is needed and unknown, ask the user for it. Explain tradeoffs in your final answer. You never \
recommend a specific stock or fund to buy — that stays educational only.
"""


def _execute_tool_call(name: str, args: dict) -> str:
    func = TOOL_FUNCTIONS[name]
    result = func(**args)
    return json.dumps(asdict(result))


def _clarifying_question(missing_fields: list[str]) -> str:
    labels = {
        "income": "your monthly income",
        "monthly_expenses": "your monthly expenses",
        "existing_debt_payment": "your current monthly debt/EMI payments (₹0 if none)",
        "savings": "your total savings",
    }
    asks = [labels.get(f, f) for f in missing_fields]
    if len(asks) == 1:
        return f"To work that out, could you tell me {asks[0]}?"
    return "To work that out, could you tell me " + ", ".join(asks[:-1]) + f", and {asks[-1]}?"


def run_agent(
    user_message: str,
    context: FinancialContext | None = None,
    retriever=None,
    provider: str = "groq",
    model: str = "llama-3.1-8b-instant",
    max_iterations: int = 5,
) -> dict:
    """Run the tool-calling loop with context elicitation and (optional) knowledge search.

    `retriever`, if given, is a callable(query:str) -> list[chunk dict] that adds the
    search_financial_knowledge tool. Returns
    {"answer", "tool_calls", "asked_for": [...], "sources": [...]}.
    Personal values are taken from `context` (populated by extraction), never from the
    model's guessed arguments. If a required personal field is missing, the agent asks.
    """
    client = get_client(provider)
    context = context if context is not None else FinancialContext()
    extract_context(user_message, context, client, model)

    tool_schemas = list(TOOL_SCHEMAS)
    if retriever is not None:
        tool_schemas.append(SEARCH_TOOL_SCHEMA)

    system_prompt = AGENT_SYSTEM_PROMPT + "\n\n" + context.summary()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    tools_used = []
    sources = []
    seen_urls = set()

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tool_schemas,
            tool_choice="auto", temperature=0.3,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return {"answer": msg.content, "tool_calls": tools_used,
                    "asked_for": [], "sources": sources}

        # Gate: if any calculator call needs a personal field we don't have, ask instead of guessing.
        for tc in msg.tool_calls:
            missing = context.missing_personal_for(tc.function.name)
            if missing:
                return {"answer": _clarifying_question(missing), "tool_calls": tools_used,
                        "asked_for": missing, "sources": sources}

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
            args = json.loads(tc.function.arguments)

            if name == "search_financial_knowledge":
                chunks = retriever(args["query"])
                passages = []
                for i, ch in enumerate(chunks, 1):
                    passages.append(f"[{i}] {ch['chunk_text']}")
                    if ch["source_url"] not in seen_urls:
                        sources.append({"title": ch["title"], "source_url": ch["source_url"]})
                        seen_urls.add(ch["source_url"])
                result_json = "\n\n".join(passages)
            else:
                # Override personal fields with authoritative context values (never trust the
                # model's version of a personal number); question-specific args pass through.
                for pf in PERSONAL_NUMERIC_FIELDS:
                    if pf in args and getattr(context, pf) is not None:
                        args[pf] = getattr(context, pf)
                result_json = _execute_tool_call(name, args)

            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "name": name, "content": result_json})

    return {"answer": "(agent stopped: max tool iterations reached)",
            "tool_calls": tools_used, "asked_for": [], "sources": sources}


def make_retriever(k: int = 5):
    """Load the index + embedding model once and return a callable(query) -> chunks."""
    from retrieval import load_index, load_embedding_model, retrieve_top_k
    index, chunks = load_index()
    embedding_model = load_embedding_model()
    return lambda query: retrieve_top_k(query, index, chunks, embedding_model, k=k)


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
