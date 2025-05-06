# tesla_ui.py

# import streamlit as st
# from main import get_qa_chain

# st.set_page_config(page_title="Tesla Manual Chatbot")

# st.title("  Tesla Chat Assistant")

# # Cache the chain so it's not rebuilt every time
# @st.cache_resource
# def load_chain():
#     return get_qa_chain()

# qa_chain = load_chain()

# question = st.text_input("Ask a question about the Tesla Owner's Manual:")

# if question:
#     with st.spinner("Searching..."):
#         response = qa_chain.invoke(question)
#         st.success(response['result'])
import streamlit as st
from main import get_qa_chain

st.set_page_config(
    page_title="Tesla Manual Chatbot",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("<h1 style='text-align: center;'>🚗 Tesla Owner's Manual Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>Ask any question about your Tesla's features, safety, or maintenance.</p>",
    unsafe_allow_html=True
)

@st.cache_resource
def load_chain():
    return get_qa_chain()

qa_chain = load_chain()

with st.form("question_form"):
    question = st.text_input("Ask your question below:", placeholder="e.g., How do I enable autopilot?")
    submit = st.form_submit_button("🔍 Ask")

if submit and question:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke(question)
        st.markdown("#### ✅ Answer:")
        st.success(response['result'])
