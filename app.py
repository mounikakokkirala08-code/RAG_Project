
import sys
from pathlib import Path

import streamlit as st

# Make the src folder available for Python imports
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from generation import generate_answer


st.set_page_config(
    page_title="SBC Benefits Assistant",
    page_icon="📄",
    layout="centered",
)

st.title("📄 SBC Benefits Assistant")
st.write(
    "Ask questions about your Summary of Benefits and Coverage (SBC) "
    "document. Answers are generated from retrieved document passages."
)

st.info(
    "This tool helps you explore the document. Always verify important "
    "coverage decisions against the full plan documents."
)

# Store chat messages during the current session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept a new question
question = st.chat_input("Ask a question about your health plan...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the SBC document and generating an answer..."):
            try:
                answer = generate_answer(question, k=4)
            except Exception as error:
                answer = (
                    "Sorry, I couldn't generate an answer. "
                    "Please check that the FAISS index and Ollama are available. "
                    f"\n\nError details: `{error}`"
                )

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
