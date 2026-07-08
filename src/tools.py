"""Deterministic financial calculators.

These are plain Python — no LLM involved. The agent (agent.py) decides WHICH one to
call and with what arguments; these functions do the actual arithmetic so the numbers
are always correct (LLMs are unreliable at multi-step math). Every function is pure and
unit-testable in isolation. All money is in the same currency unit (₹) and rates are
annual percentages unless noted.
"""
from dataclasses import dataclass


@dataclass
class AffordabilityResult:
    can_afford_now: bool
    monthly_surplus: float
    months_to_save: float | None  # None if affordable outright or surplus <= 0
    note: str


def affordability_calculator(
    income: float,
    monthly_expenses: float,
    existing_debt_payment: float,
    item_cost: float,
) -> AffordabilityResult:
    """Can the user buy `item_cost` outright from one month's surplus, and if not,
    how many months of saving the surplus would it take?

    monthly_surplus = income - expenses - existing debt payments.
    """
    monthly_surplus = income - monthly_expenses - existing_debt_payment
    can_afford_now = monthly_surplus >= item_cost

    if can_afford_now:
        return AffordabilityResult(True, monthly_surplus, None,
                                   "Affordable from this month's surplus.")
    if monthly_surplus <= 0:
        return AffordabilityResult(False, monthly_surplus, None,
                                   "No monthly surplus — expenses and debt consume all income.")
    months_to_save = item_cost / monthly_surplus
    return AffordabilityResult(False, monthly_surplus, months_to_save,
                               f"Not affordable now; ~{months_to_save:.1f} months of saving the surplus.")


@dataclass
class EmiVsCashResult:
    monthly_emi: float
    total_emi_paid: float
    emi_interest_cost: float           # extra paid vs. the sticker price under EMI
    opportunity_cost_of_cash: float    # what the lump sum could have earned if invested instead
    cheaper_option: str                # "emi" or "cash"
    note: str


def emi_vs_cash_calculator(
    item_cost: float,
    tenure_months: int,
    interest_rate_annual: float,
    opportunity_cost_rate_annual: float,
) -> EmiVsCashResult:
    """Compare paying cash now vs. taking an EMI loan.

    EMI formula (standard reducing-balance):  EMI = P*r*(1+r)^n / ((1+r)^n - 1)
    where P = principal (item_cost), r = monthly rate, n = tenure in months.

    EMI extra cost  = total EMI paid - item_cost (the interest you pay the lender).
    Cash opportunity cost = interest the item_cost would have earned if invested for the
    same tenure at opportunity_cost_rate_annual (simple approximation over the period).

    We pick the cheaper option by comparing EMI's interest cost against the cash path's
    opportunity cost (paying cash frees you from EMI interest but forgoes investment growth).
    """
    r = interest_rate_annual / 100 / 12
    n = tenure_months
    if r == 0:
        monthly_emi = item_cost / n
    else:
        monthly_emi = item_cost * r * (1 + r) ** n / ((1 + r) ** n - 1)
    total_emi_paid = monthly_emi * n
    emi_interest_cost = total_emi_paid - item_cost

    # Opportunity cost of spending the lump sum now instead of investing it for the tenure.
    opp_r = opportunity_cost_rate_annual / 100 / 12
    opportunity_cost_of_cash = item_cost * ((1 + opp_r) ** n - 1)

    cheaper_option = "emi" if emi_interest_cost < opportunity_cost_of_cash else "cash"
    note = (
        f"EMI costs ₹{emi_interest_cost:,.0f} in interest; paying cash forgoes "
        f"~₹{opportunity_cost_of_cash:,.0f} of investment growth over {n} months."
    )
    return EmiVsCashResult(monthly_emi, total_emi_paid, emi_interest_cost,
                           opportunity_cost_of_cash, cheaper_option, note)


@dataclass
class BudgetSplitResult:
    needs_budget: float
    wants_budget: float
    savings_budget: float
    current_savings_rate: float | None  # fraction of income, if expenses given
    note: str


def budget_split_calculator(
    income: float,
    monthly_expenses: float | None = None,
) -> BudgetSplitResult:
    """50/30/20 rule: 50% needs, 30% wants, 20% savings. If monthly_expenses is given,
    also report the user's current savings rate for comparison."""
    needs_budget = income * 0.50
    wants_budget = income * 0.30
    savings_budget = income * 0.20

    current_savings_rate = None
    if monthly_expenses is not None:
        current_savings_rate = max(0.0, (income - monthly_expenses) / income)
    note = "50/30/20 guideline: 50% needs, 30% wants, 20% savings."
    return BudgetSplitResult(needs_budget, wants_budget, savings_budget,
                             current_savings_rate, note)


@dataclass
class RunwayResult:
    runway_months: float
    covers_gap: bool
    shortfall_months: float
    note: str


def job_quit_runway_calculator(
    savings: float,
    monthly_expenses: float,
    expected_income_gap_months: float,
) -> RunwayResult:
    """How many months the savings cover expenses with no income, and whether that
    covers the expected gap until new income starts."""
    if monthly_expenses <= 0:
        raise ValueError("monthly_expenses must be positive")
    runway_months = savings / monthly_expenses
    covers_gap = runway_months >= expected_income_gap_months
    shortfall_months = max(0.0, expected_income_gap_months - runway_months)
    if covers_gap:
        note = f"Savings cover ~{runway_months:.1f} months — enough for the {expected_income_gap_months:.0f}-month gap."
    else:
        note = f"Savings cover ~{runway_months:.1f} months — short by ~{shortfall_months:.1f} months."
    return RunwayResult(runway_months, covers_gap, shortfall_months, note)
