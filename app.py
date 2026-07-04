"""Streamlit chat UI for the Upanishad RAG bot.

Run:  streamlit run app.py
"""

import streamlit as st

from src.rag import ask

st.set_page_config(page_title="Upanishad RAG", page_icon="🕉️")

st.title("🕉️ Upanishad RAG")
st.caption(
    "Ask about life, death, the Self, Brahman — answered strictly from the "
    "108 Upanishads, with verse citations."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Passages to retrieve", 3, 12, 6)
    st.markdown(
        "Every answer is generated **only** from retrieved verses. "
        "If the texts don't address a question, the bot says so."
    )

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("passages"):
            with st.expander("📜 Source verses"):
                for p in turn["passages"]:
                    st.markdown(
                        f"**{p['upanishad']} {p['ref']}** · similarity {p['score']}\n\n"
                        f"> {p['text']}"
                    )

if question := st.chat_input("e.g. What happens to the self after death?"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the Upanishads..."):
            try:
                result = ask(question, k=top_k)
            except RuntimeError as e:  # missing API key
                st.error(str(e))
                st.stop()
        st.markdown(result["answer"])
        with st.expander("📜 Source verses"):
            for p in result["passages"]:
                st.markdown(
                    f"**{p['upanishad']} {p['ref']}** · similarity {p['score']}\n\n"
                    f"> {p['text']}"
                )

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "passages": result["passages"],
        }
    )
