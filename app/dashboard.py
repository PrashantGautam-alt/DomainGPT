"""MLOps dashboard — operational metrics from the privacy-aware request log.

Shows only non-sensitive operational data (volume, latency, cost, tool mix, cache hits) —
by construction the log contains no personal financial values, so nothing sensitive can
surface here. Run: streamlit run app/dashboard.py
"""
import sys
from pathlib import Path
from statistics import mean

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monitor import read_logs

st.set_page_config(page_title="DomainGPT — Ops", page_icon="📊")
st.title("DomainGPT — Ops Dashboard")
st.caption("Operational metrics only. No personal financial values are ever logged or shown.")

logs = read_logs()
if not logs:
    st.info("No requests logged yet. Use the app, then refresh.")
    st.stop()


def percentile(values, p):
    if not values:
        return 0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return s[idx]


latencies = [r["latency_ms"] for r in logs]
costs = [r["estimated_cost_usd"] for r in logs]
cache_hits = sum(1 for r in logs if r.get("cache_hit"))
degraded = sum(1 for r in logs if r.get("degraded"))
asked = sum(1 for r in logs if r.get("asked_for_missing"))

c1, c2, c3 = st.columns(3)
c1.metric("Total requests", len(logs))
c2.metric("p50 latency", f"{percentile(latencies, 50)} ms")
c3.metric("p95 latency", f"{percentile(latencies, 95)} ms")

c4, c5, c6 = st.columns(3)
c4.metric("Avg cost / query", f"${mean(costs):.5f}" if costs else "$0")
c5.metric("Cache hit rate", f"{cache_hits / len(logs) * 100:.0f}%")
c6.metric("Degraded (fallback)", degraded)

st.subheader("Tool usage")
tool_counts = {}
for r in logs:
    for t in r.get("tools_called", []):
        tool_counts[t] = tool_counts.get(t, 0) + 1
if tool_counts:
    st.bar_chart(tool_counts)
else:
    st.caption("No tool calls logged yet.")

st.subheader("Behavioral signals")
st.write(f"- Clarifying question asked (missing context): **{asked}** / {len(logs)} requests")
st.write(f"- Context provided: **{sum(1 for r in logs if r.get('context_provided'))}** / {len(logs)} requests")

st.subheader("Latency over time")
st.line_chart([r["latency_ms"] for r in logs])
