from dependency import *
from modules.nav import MenuButtons
from pages.account import get_roles
from streamlit_rag_model import llama_model

streamlit_rag_model = llama_model()

if 'authentication_status' not in ss:
    st.switch_page('./pages/account.py')

MenuButtons(get_roles())
st.header("UPLOAD YOUR PDF'S")

uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=["pdf"])

if st.button("Upload"):
    if not uploaded_files:
        st.error("Please upload PDF files")

    for file in uploaded_files:
        pdf_details = streamlit_rag_model._pdf_file_save(file)
        message = rag_function._create_embedding_all(pdf_details['pdf_id'], pdf_details['pdf_path'])
    
    st.success(message['message'])