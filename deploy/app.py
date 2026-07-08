"""HuggingFace Space entry point (Streamlit).

In the Space, this file sits flat alongside copies of retrieval.py and generate.py
(assembled by deploy/build_space.sh). The FAISS index is NOT committed here — it's
downloaded at startup from the private HF Dataset by retrieval.load_index(), using
the HF_TOKEN Space secret. RAG-only version (tools + fine-tuned model come later).
"""
import streamlit as st

from retrieval import load_index, load_embedding_model, retrieve_top_k
from generate import build_prompt, generate_response

st.set_page_config(page_title="DomainGPT", page_icon="💰")


@st.cache_resource
def get_resources():
    index, chunks = load_index()
    embedding_model = load_embedding_model()
    return index, chunks, embedding_model


st.title("DomainGPT")
st.caption(
    "A financial-literacy assistant for students and early-career professionals. "
    "**Educational information only — not personalized investment advice.**"
)

query = st.text_input("Ask a money question", placeholder="Should I use a credit card?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and query.strip():
    index, chunks, embedding_model = get_resources()
    with st.spinner("Thinking..."):
        results = retrieve_top_k(query, index, chunks, embedding_model, k=5)
        messages, sources = build_prompt(query, results)
        answer = generate_response(messages)

    st.markdown(answer)
    if sources:
        st.markdown("**Sources:**")
        for s in sources:
            st.markdown(f"- [{s['title']}]({s['source_url']})")
