"""Free web search (DuckDuckGo, no API key) for real-world facts the agent needs but the
finance corpus doesn't have — e.g. a product's current price. This is the "use tools before
asking the user" capability: the agent looks a fact up instead of interrogating the user.

Best-effort: DDG can rate-limit or return nothing; callers handle an empty result by asking
the user for the fact as a fallback.
"""


def web_search(query: str, max_results: int = 4) -> str:
    """Return a short text digest of the top results, or "" if the search fails/empties."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results)
    except Exception:
        return ""
    if not results:
        return ""
    lines = []
    for r in results:
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()
        lines.append(f"- {title}: {body}")
    return "\n".join(lines)
