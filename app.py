import streamlit as st
from qa_chain import get_qa_chain

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📘"
)

st.title("📘 mediBot")

# Load QA chain once
@st.cache_resource
def load_chain():
    return get_qa_chain()

qa_chain = load_chain()

# User input
query = st.text_input("Ask a question from your documents")

# Answer
if query:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke(query)
        st.markdown("### Answer")
        st.write(response)
