"""NomosAI - goal-oriented financial-decision assistant (Streamlit chat UI).

Multi-turn chat. The full conversation is passed to the agent each turn so it keeps the
user's goal and known facts in view (no forgetting / no re-asking). The ephemeral
FinancialContext lives in st.session_state and is never persisted.
"""
import os
import sys
from pathlib import Path

import streamlit as st

# On Streamlit Community Cloud, API keys come via st.secrets; our modules read os.environ.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import make_retriever
from context import FinancialContext
from cache import SemanticCache
from retrieval import load_embedding_model
from service import answer
from websearch import web_search

st.set_page_config(page_title="NomosAI", page_icon="🪙", layout="centered")


@st.cache_resource
def get_resources():
    retriever = make_retriever()
    cache = SemanticCache(load_embedding_model())
    return retriever, cache


st.markdown(
    """
    <div style="text-align:center; padding: 0.5rem 0 0.25rem;">
      <div style="font-size:2.4rem; font-weight:800; letter-spacing:-0.5px;">🪙 NomosAI</div>
      <div style="font-size:1.02rem; opacity:0.75; margin-top:0.15rem;">
        Smart money decisions for students and early-career professionals
      </div>
      <div style="font-size:0.82rem; opacity:0.55; margin-top:0.35rem;">
        Educational information only, not personalized investment advice.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Ephemeral per-session state, never persisted.
if "context" not in st.session_state:
    st.session_state.context = FinancialContext()
if "messages" not in st.session_state:
    st.session_state.messages = []

EXAMPLES = [
    "Should I use a credit card?",
    "I earn 25000, spend 15000, can I afford a 40000 phone?",
    "How much money should I earn a month to buy a Royal Enfield Bullet?",
]

# Render history.
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🪙" if m["role"] == "assistant" else None):
        st.markdown(m["content"])
        for s in m.get("sources", []):
            st.markdown(f"- [{s['title']}]({s['source_url']})")

# Example chips (only before the conversation starts).
pending = None
if not st.session_state.messages:
    st.caption("Try one of these:")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            pending = ex

typed = st.chat_input("Ask a money question")
prompt = typed or pending

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🪙"):
        retriever, cache = get_resources()
        with st.spinner("Thinking..."):
            out = answer(st.session_state.messages, st.session_state.context,
                         retriever, cache=cache, web_search=web_search)
        st.markdown(out["answer"])
        if out["sources"]:
            st.markdown("**Sources:**")
            for s in out["sources"]:
                st.markdown(f"- [{s['title']}]({s['source_url']})")
    st.session_state.messages.append(
        {"role": "assistant", "content": out["answer"], "sources": out["sources"]}
    )
    if pending:  # a chip was clicked; rerun so the input box clears and chips hide
        st.rerun()

with st.sidebar:
    st.subheader("What I know about you")
    known = st.session_state.context.known_fields()
    if known:
        pretty = {
            "income": "Monthly income", "monthly_expenses": "Monthly expenses",
            "existing_debt_payment": "Debt/EMI", "savings": "Savings",
            "goal": "Goal", "risk_tolerance": "Risk tolerance",
        }
        for k, v in known.items():
            label = pretty.get(k, k)
            val = f"₹{v:,}" if isinstance(v, (int, float)) else v
            st.markdown(f"**{label}:** {val}")
    else:
        st.caption("Nothing yet. I'll ask only what I need.")
    st.divider()
    st.caption("Kept only for this session and never saved to disk.")
    if st.button("Start over"):
        st.session_state.context = FinancialContext()
        st.session_state.messages = []
        st.rerun()
