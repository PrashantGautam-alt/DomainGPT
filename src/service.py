"""Request orchestration: ties together cache, agent+fallback, and privacy-aware logging.

This is the single entry point the UI/API calls. Flow:
  1. If the query is cacheable (no personal context yet), check the semantic cache.
  2. Otherwise run the agent across the fallback chain.
  3. Cache the answer only if it used NO personal context and NO calculator (general
     principle answers only — never cache personalised/tool results).
  4. Log the request with privacy-aware fields only.
"""
from context import FinancialContext
from agent import run_agent_with_fallback
from cache import SemanticCache
from monitor import Timer, build_log, log_request

# Tools whose outputs must never be cached (they depend on personal numbers).
CALCULATOR_TOOLS = {
    "affordability_calculator", "emi_vs_cash_calculator",
    "budget_split_calculator", "job_quit_runway_calculator",
}


def answer(conversation, context: FinancialContext, retriever,
           cache: SemanticCache | None = None, web_search=None) -> dict:
    """`conversation` is the full [{role, content}] history (or a single user string).
    Runs the goal-oriented agent over it with cache + fallback + privacy-aware logging."""
    if isinstance(conversation, str):
        history = [{"role": "user", "content": conversation}]
    else:
        history = conversation
    latest_query = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
    user_turns = sum(1 for m in history if m.get("role") == "user")

    had_context_before = bool(context.known_fields())
    # Cache only a first-turn, no-context, general-principle question (multi-turn is personal).
    cacheable = cache is not None and not had_context_before and user_turns <= 1

    if cacheable:
        cached = cache.get(latest_query)
        if cached is not None:
            _log(latency_ms=0, tools=[], context_provided=False, asked=False,
                 degraded=False, cache_hit=True, model="cache")
            return {**cached, "cache_hit": True, "model": "cache"}

    with Timer() as t:
        out = run_agent_with_fallback(history, context=context, retriever=retriever,
                                      web_search=web_search)

    used_calculator = any(tc in CALCULATOR_TOOLS for tc in out["tool_calls"])
    context_used = bool(context.known_fields())
    asked = bool(out["asked_for"])

    # Only cache pure general-principle answers: no personal context, no calculator, actually answered.
    if cacheable and not used_calculator and not context_used and not asked and out.get("answer"):
        cache.put(latest_query, out["answer"], out["sources"])

    _log(latency_ms=t.elapsed_ms, tools=out["tool_calls"], context_provided=context_used,
         asked=asked, degraded=out.get("degraded", False), cache_hit=False, model=out.get("model", "?"))

    out["cache_hit"] = False
    return out


def _log(latency_ms, tools, context_provided, asked, degraded, cache_hit, model):
    log_request(build_log(
        latency_ms=latency_ms, tools_called=tools, context_provided=context_provided,
        asked_for_missing=asked, degraded=degraded, cache_hit=cache_hit, model=model,
    ))
