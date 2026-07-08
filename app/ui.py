"""Full agentic DomainGPT chat UI.

Multi-turn chat where the ephemeral FinancialContext lives in st.session_state — it
persists across turns within a browser session and disappears when the session ends.
Nothing is written to disk or a database (structural privacy decision, see SPEC.md).
The agent (agent.run_agent) decides between calculators, knowledge search, or asking a
clarifying question.
"""
import os
import sys
from pathlib import Path

import streamlit as st

# On Streamlit Community Cloud, API keys are provided via st.secrets. Our src modules read
# them from os.environ, so copy them across before anything uses them. (Locally, .env is
# used instead and st.secrets is empty — the try/except keeps that path working too.)
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

st.set_page_config(page_title="DomainGPT", page_icon="💰")


@st.cache_resource
def get_resources():
    retriever = make_retriever()
    cache = SemanticCache(load_embedding_model())
    return retriever, cache


st.title("DomainGPT")
st.caption(
    "A financial-decision assistant for students and early-career professionals. "
    "**Educational information only — not personalized investment advice.**"
)

# Ephemeral per-session state — never persisted.
if "context" not in st.session_state:
    st.session_state.context = FinancialContext()
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for s in m.get("sources", []):
            st.markdown(f"- [{s['title']}]({s['source_url']})")

if prompt := st.chat_input("Ask a money question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        retriever, cache = get_resources()
        with st.spinner("Thinking…"):
            out = answer(prompt, st.session_state.context, retriever, cache=cache)
        st.markdown(out["answer"])
        if out["sources"]:
            st.markdown("**Sources:**")
            for s in out["sources"]:
                st.markdown(f"- [{s['title']}]({s['source_url']})")
    st.session_state.messages.append(
        {"role": "assistant", "content": out["answer"], "sources": out["sources"]}
    )

with st.sidebar:
    st.subheader("What I know so far")
    known = st.session_state.context.known_fields()
    if known:
        for k, v in known.items():
            st.text(f"{k}: {v}")
    else:
        st.caption("Nothing yet — I'll ask as needed.")
    st.caption("This is kept only for this session and never saved.")
