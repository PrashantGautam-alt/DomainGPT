"""Ephemeral personal financial context + extraction.

FinancialContext holds the sensitive personal fields (income, expenses, debt, savings,
goal, risk tolerance). It lives ONLY in memory / Streamlit session_state for the duration
of a session — never written to disk or a database. This is a structural privacy decision
(see SPEC.md data-sensitivity constraint), not a "we'll add auth later" placeholder.

The model never supplies these personal values directly into a tool call. Instead we
extract them from the user's message into this object, and the agent fills tool arguments
from it. If a required personal field is still missing, the agent asks rather than guesses.
"""
import json
from dataclasses import dataclass, field, fields, asdict

# Persistent personal fields (worth eliciting once and reusing across the session).
# Question-specific values (item_cost, tenure_months, rates, gap_months) are NOT here —
# they come fresh from each question.
PERSONAL_NUMERIC_FIELDS = ["income", "monthly_expenses", "existing_debt_payment", "savings"]

# Which personal fields each tool needs before it can run.
TOOL_REQUIRED_PERSONAL_FIELDS = {
    "affordability_calculator": ["income", "monthly_expenses", "existing_debt_payment"],
    "emi_vs_cash_calculator": [],
    "budget_split_calculator": ["income"],
    "job_quit_runway_calculator": ["savings", "monthly_expenses"],
}


@dataclass
class FinancialContext:
    income: float | None = None
    monthly_expenses: float | None = None
    existing_debt_payment: float | None = None
    savings: float | None = None
    goal: str | None = None
    risk_tolerance: str | None = None

    def known_fields(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}

    def missing_personal_for(self, tool_name: str) -> list[str]:
        required = TOOL_REQUIRED_PERSONAL_FIELDS.get(tool_name, [])
        return [name for name in required if getattr(self, name) is None]

    def summary(self) -> str:
        known = self.known_fields()
        if not known:
            return "Nothing known yet about the user's finances."
        return "Known about the user: " + ", ".join(f"{k}={v}" for k, v in known.items())


EXTRACTION_PROMPT = """Extract personal financial values EXPLICITLY stated in the user's message. \
Return ONLY a JSON object. Include a key ONLY if the user gives an actual value for it:
- income (monthly income, number)
- monthly_expenses (number)
- existing_debt_payment (monthly debt/EMI; include as 0 ONLY if the user explicitly says no debt)
- savings (number)
- goal (short string)
- risk_tolerance (string)
Do NOT include a key the user did not mention. Do NOT infer, assume, or default any value to 0. \
If a field is not stated, leave it out entirely. If nothing is stated, return {}.

Message: """


def extract_context(message: str, context: FinancialContext, client, model: str) -> FinancialContext:
    """Update `context` in place with any financial values stated in `message`."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT + message}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    try:
        extracted = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return context

    valid_keys = {f.name for f in fields(context)}
    for key, value in extracted.items():
        if key not in valid_keys or value is None:
            continue
        # Drop empty strings (models emit "" for string fields they were told to omit).
        if isinstance(value, str) and not value.strip():
            continue
        # Only overwrite a known value with a newly stated one; don't let a spurious
        # extraction clobber an existing field (extraction is per-message, best-effort).
        if getattr(context, key) is None:
            setattr(context, key, value)
    return context
