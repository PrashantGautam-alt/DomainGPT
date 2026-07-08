"""Generate the QLoRA supervised fine-tuning (SFT) dataset.

The fine-tune teaches BEHAVIORAL POLICY, not facts (facts come from RAG + the tools):
  1. tool-selection      — a numeric question with full info -> emit the RIGHT tool call
  2. context-elicitation — a question missing a required personal value -> ASK, don't guess
  3. advice-boundary     — a "which stock/fund should I buy" question -> redirect to
                           principles/tradeoffs, never name a product to buy

Each example is a chat conversation (list of messages) using the same tool schemas the
live agent uses, so the fine-tune's format matches inference exactly. We use Groq's larger
Llama-3.3-70B as a generator to produce diverse variants of the real friend-questions, then
hand-verify a sample in notebooks/finetune_prep.ipynb. Output: data/train.jsonl.
"""
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

from generate import get_client

load_dotenv()

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "train.jsonl"
GENERATOR_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are DomainGPT, a financial-decision assistant for students and early-career "
    "professionals in India. You have calculator tools for affordability, EMI-vs-cash, "
    "budgeting, and job-quit runway. Call the right tool for arithmetic questions rather "
    "than computing numbers yourself. Never invent a personal financial value (income, "
    "expenses, debt, savings) — if one is needed and unknown, ask the user for it. Explain "
    "tradeoffs. You never recommend a specific stock or fund to buy — that stays educational only."
)

# Seed question stems grounded in the real friend-questions + corpus topics.
TOOL_SELECTION_SEEDS = [
    ("affordability", "buying a {item} costing ₹{cost}", ["phone", "laptop", "bike", "watch"]),
    ("emi_vs_cash", "buying a ₹{cost} {item} on EMI vs paying cash", ["phone", "laptop", "TV"]),
    ("budget", "how to split a ₹{income} monthly income / how much to save", [""]),
    ("runway", "quitting a job with ₹{savings} savings for {months} months", [""]),
]

ELICITATION_SEEDS = [
    "Can I afford a ₹{cost} {item}?",
    "Should I buy this ₹{cost} {item} on EMI or cash?",
    "How much should I save each month?",
    "Can I afford to quit my job for a few months?",
]

ADVICE_BOUNDARY_SEEDS = [
    "Which mutual fund should I buy right now?",
    "We want to learn trading to earn money fast — what should we do?",
    "Which stock will give me the best returns this year?",
    "Should I put my money in {scheme} — is it a good buy?",
]


def _gen(client, instruction: str) -> str:
    resp = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": instruction}],
        temperature=0.8,
    )
    return resp.choices[0].message.content.strip()


def build_tool_selection_examples(client, n: int) -> list[dict]:
    """user (with all numbers) -> assistant emits the correct tool call."""
    examples = []
    tool_map = {
        "affordability": "affordability_calculator",
        "emi_vs_cash": "emi_vs_cash_calculator",
        "budget": "budget_split_calculator",
        "runway": "job_quit_runway_calculator",
    }
    for _ in range(n):
        kind, _desc, items = random.choice(TOOL_SELECTION_SEEDS)
        item = random.choice(items)
        cost = random.choice([15000, 20000, 30000, 50000, 80000])
        income = random.choice([15000, 25000, 40000, 60000])
        expenses = random.choice([8000, 12000, 20000, 30000])
        savings = random.choice([50000, 120000, 200000])
        months = random.choice([3, 6, 9])

        if kind == "affordability":
            user = f"I earn ₹{income} a month, spend ₹{expenses}, no debt. Can I afford a {item} costing ₹{cost}?"
            args = {"income": income, "monthly_expenses": expenses, "existing_debt_payment": 0, "item_cost": cost}
        elif kind == "emi_vs_cash":
            user = f"Should I buy a ₹{cost} {item} on a 12-month EMI at 14% interest, or pay cash? I could invest at 8%."
            args = {"item_cost": cost, "tenure_months": 12, "interest_rate_annual": 14, "opportunity_cost_rate_annual": 8}
        elif kind == "budget":
            user = f"I earn ₹{income} a month and spend ₹{expenses}. How should I budget and how much should I save?"
            args = {"income": income, "monthly_expenses": expenses}
        else:  # runway
            user = f"I have ₹{savings} saved and spend ₹{expenses} a month. Can I afford to quit my job for {months} months?"
            args = {"savings": savings, "monthly_expenses": expenses, "expected_income_gap_months": months}

        # Target: the assistant emits a structured tool call as JSON text (Llama-3.1's
        # native {"name", "parameters"} shape). We train on text content rather than the
        # template's tool_calls slot, because the Llama-3.1 chat template does not render
        # assistant tool_calls — this format is robust and matches Llama-3.1 tool output.
        tool_call_json = json.dumps({"name": tool_map[kind], "parameters": args})
        examples.append({
            "category": "tool_selection",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": tool_call_json},
            ],
        })
    return examples


def build_elicitation_examples(client, n: int) -> list[dict]:
    """user (missing personal info) -> assistant asks a concise clarifying question."""
    examples = []
    for _ in range(n):
        template = random.choice(ELICITATION_SEEDS)
        item = random.choice(["phone", "laptop", "bike"])
        cost = random.choice([20000, 30000, 50000])
        user = template.format(cost=cost, item=item)
        instruction = (
            f"A user asks a financial assistant: \"{user}\". The assistant does NOT know the user's "
            f"income, expenses, savings, or debt yet. Write ONLY the assistant's reply, which must ask "
            f"one concise clarifying question for the specific missing numbers it needs — and must NOT "
            f"guess or assume any figures. 1-2 sentences."
        )
        reply = _gen(client, instruction)
        examples.append({
            "category": "context_elicitation",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": reply},
            ],
        })
    return examples


def build_advice_boundary_examples(client, n: int) -> list[dict]:
    """user asks for a specific product pick -> assistant redirects to principles, no product."""
    examples = []
    for _ in range(n):
        template = random.choice(ADVICE_BOUNDARY_SEEDS)
        scheme = random.choice(["a crypto coin", "an IPO", "a chit fund", "a stock tip from a friend"])
        user = template.format(scheme=scheme)
        instruction = (
            f"A user asks a financial assistant: \"{user}\". The assistant must NOT recommend any "
            f"specific stock, fund, or product to buy (educational-only boundary, SEBI RIA rules). "
            f"Write ONLY the assistant's reply: gently decline to name a product, explain the relevant "
            f"tradeoffs/risks and how to think about it, and suggest consulting a SEBI-registered "
            f"advisor for personalised product decisions. 3-5 sentences, practical tone."
        )
        reply = _gen(client, instruction)
        examples.append({
            "category": "advice_boundary",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": reply},
            ],
        })
    return examples


def main(n_per_category: int = 50):
    client = get_client("groq")
    random.seed(42)

    print("Generating tool-selection examples...")
    tool_ex = build_tool_selection_examples(client, n_per_category)
    print("Generating context-elicitation examples...")
    elicit_ex = build_elicitation_examples(client, n_per_category)
    print("Generating advice-boundary examples...")
    boundary_ex = build_advice_boundary_examples(client, n_per_category)

    all_examples = tool_ex + elicit_ex + boundary_ex
    random.shuffle(all_examples)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    counts = {}
    for ex in all_examples:
        counts[ex["category"]] = counts.get(ex["category"], 0) + 1
    print(f"\nWrote {len(all_examples)} examples to {OUT_PATH}")
    print("Category balance:", counts)


if __name__ == "__main__":
    main()
