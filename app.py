"""Streamlit chat UI for the Upanishads · Ashtavakra Gita · Bhagavad Gita RAG bot.

Run:  streamlit run app.py
"""

import os

import streamlit as st

# Streamlit Cloud provides secrets via st.secrets, not os.environ. Bridge them
# into the environment so src/generate.py (which reads os.environ["GROQ_API_KEY"])
# finds the key. Wrapped in try/except so local runs using a .env still work.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

from src.rag import ask

st.set_page_config(page_title="Vedanta RAG", page_icon="🕉️")

st.title("🕉️ Upanishads · 🪷 Ashtavakra · 🏹 Gita")
st.caption(
    "Ask about life, death, the Self, Brahman — answered strictly from the "
    "108 Upanishads, the Ashtavakra Gita and the Bhagavad Gita (Shankara "
    "Bhashya ed.), with verse citations. With several texts selected, each "
    "answers separately and disagreements between them are highlighted."
)

SOURCE_TITLES = {
    "upanishads": "🕉️ Upanishads",
    "ashtavakra": "🪷 Ashtavakra Gita",
    "gita": "🏹 Bhagavad Gita",
}

with st.sidebar:
    st.header("Settings")
    scope = st.multiselect(
        "Texts to ask",
        list(SOURCE_TITLES),
        default=list(SOURCE_TITLES),
        format_func=SOURCE_TITLES.get,
    )
    top_k = st.slider("Passages to retrieve", 3, 12, 6)
    if len(scope) > 1:
        st.caption("Passages are retrieved from each selected text separately.")
    st.markdown(
        "Every answer is generated **only** from retrieved verses. "
        "If the texts don't address a question, the bot says so."
    )


def show_passages(passages: list[dict]) -> None:
    with st.expander("📜 Source verses"):
        by_source: dict[str, list[dict]] = {}
        for p in passages:
            by_source.setdefault(p.get("source", "upanishads"), []).append(p)
        for source, group in by_source.items():
            if len(by_source) > 1:
                st.markdown(f"##### {SOURCE_TITLES.get(source, source)}")
            for p in group:
                st.markdown(
                    f"**{p['upanishad']} {p['ref']}** · score {p['score']}\n\n"
                    f"> {p['text']}"
                )


if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("passages"):
            show_passages(turn["passages"])

if question := st.chat_input("e.g. What happens to the self after death?"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not scope:
            st.warning("Select at least one text in the sidebar.")
            st.stop()
        with st.spinner("Searching the texts..."):
            try:
                result = ask(question, k=top_k, scope=scope)
            except RuntimeError as e:  # missing API key
                st.error(str(e))
                st.stop()
        st.markdown(result["answer"])
        show_passages(result["passages"])

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "passages": result["passages"],
        }
    )
