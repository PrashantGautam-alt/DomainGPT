"""HuggingFace Space entry point — FULL agentic DomainGPT (Streamlit).

Runs on free CPU: the LLM is a Groq API call (no local model), FAISS + bge embeddings are
small and CPU-friendly, and the calculators are plain Python. In the Space this file sits
flat alongside copies of the src modules (assembled by deploy/build_space.sh). The FAISS
index is downloaded at startup from the private HF Dataset via the HF_TOKEN secret.

Secrets required in the Space (Settings → Secrets): GROQ_API_KEY, HF_TOKEN.
"""
import streamlit as st

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
    st.caption("Kept only for this session, never saved.")
