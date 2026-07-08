"""Hand-computed test cases for the deterministic calculators.
Run: .venv/bin/python3 -m pytest tests/ -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from tools import (
    affordability_calculator,
    emi_vs_cash_calculator,
    budget_split_calculator,
    job_quit_runway_calculator,
)


def test_emi_standard_reducing_balance():
    # ₹1,00,000 loan, 12 months, 12% annual → EMI ≈ 8884.88 (hand-computed)
    r = emi_vs_cash_calculator(100000, 12, 12.0, 8.0)
    assert r.monthly_emi == pytest.approx(8884.88, abs=0.1)
    assert r.emi_interest_cost == pytest.approx(6618.55, abs=1.0)


def test_emi_zero_interest():
    r = emi_vs_cash_calculator(12000, 12, 0.0, 8.0)
    assert r.monthly_emi == pytest.approx(1000.0)
    assert r.emi_interest_cost == pytest.approx(0.0)


def test_affordability_needs_saving():
    a = affordability_calculator(15000, 10000, 0, 20000)
    assert a.can_afford_now is False
    assert a.monthly_surplus == 5000
    assert a.months_to_save == pytest.approx(4.0)


def test_affordability_now():
    a = affordability_calculator(50000, 10000, 0, 20000)
    assert a.can_afford_now is True
    assert a.months_to_save is None


def test_affordability_no_surplus():
    a = affordability_calculator(10000, 10000, 2000, 5000)
    assert a.monthly_surplus == -2000
    assert a.months_to_save is None


def test_budget_split_50_30_20():
    b = budget_split_calculator(30000, 22000)
    assert (b.needs_budget, b.wants_budget, b.savings_budget) == (15000, 9000, 6000)
    assert b.current_savings_rate == pytest.approx(8000 / 30000)


def test_budget_split_no_expenses():
    b = budget_split_calculator(30000)
    assert b.current_savings_rate is None


def test_runway_short():
    q = job_quit_runway_calculator(120000, 20000, 8)
    assert q.runway_months == pytest.approx(6.0)
    assert q.covers_gap is False
    assert q.shortfall_months == pytest.approx(2.0)


def test_runway_covers():
    q = job_quit_runway_calculator(200000, 20000, 8)
    assert q.covers_gap is True
    assert q.shortfall_months == 0.0


def test_runway_zero_expenses_raises():
    with pytest.raises(ValueError):
        job_quit_runway_calculator(100000, 0, 6)
