import uuid
import streamlit as st
from modules.nav import MenuButtons
from pages.account import get_roles
from streamlit import session_state as ss
from streamlit_rag_model import llama_model

streamlit_rag_model = llama_model()
in_memory_cache = {}

if 'authentication_status' not in ss:
    st.switch_page('./pages/account.py')

MenuButtons(get_roles())
st.header("Interaction with PDF's")


def initialize_session_state():
    if 'student_id' not in st.session_state:
        st.session_state.student_id = str(uuid.uuid4())
        st.session_state.chat_history = {}
        st.session_state.session_data = {}


initialize_session_state()


def handle_query():
    student_id = st.session_state.student_id
    user_query = st.session_state['user_query']

    if not user_query:
        st.error("Please enter a query.")
        return

    answer = streamlit_rag_model._get_answer_to_query(user_query)
    if student_id not in st.session_state.chat_history:
        st.session_state.chat_history[student_id] = []
    st.session_state.chat_history[student_id].append({'query': user_query, 'answer': answer['answer']})
    st.success(answer['message'])
    st.write('Answer:', answer['answer'])
    st.write('Chat History:')
    for chat in st.session_state.chat_history.get(student_id, []):
        st.write(f"Query: {chat['query']}")
        st.write(f"Answer: {chat['answer']}")
        st.write("---")


st.title("RAG Model")
with st.form(key='query_form_multi', clear_on_submit=True):
    st.text_input("Enter your query:", key='user_query')
    submit_button = st.form_submit_button("Submit")

    if submit_button:
        handle_query()
