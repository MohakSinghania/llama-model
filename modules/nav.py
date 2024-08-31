from dependency import *


def HomeNav():
    st.sidebar.page_link("streamlit_app.py", label="Home", icon='🏠')

def LoginNav():
    st.sidebar.page_link("/home/ubuntu/llama-model/pages/account.py", label="Account", icon='🔐')

def UploadPDFNav():
    st.sidebar.page_link("/home/ubuntu/llama-model/pages/upload_pdf_page.py", label="Upload PDF's", icon='✈️')

def RAGModelNav():
    st.sidebar.page_link("/home/ubuntu/llama-model/pages/rag_model_stream.py", label="RAG Model", icon='✈️')

def UploadPDFClassNav():
    st.sidebar.page_link("/home/ubuntu/llama-model/pages/upload_pdf_page_class.py", label="Upload PDF's Based On Class", icon='✈️')

def RAGModelClassNav(class_name):
    ss.class_name = class_name
    st.sidebar.page_link("/home/ubuntu/llama-model/pages/rag_model_stream_class.py", label="RAG Model Based On Class", icon='📚')

def Page1():
    st.sidebar.page_link("/home/ubuntu/llama-model/pages/page1.py", label="Upload PDF's Based On Hierarchical Architecture", icon='✈️')

def MenuButtons(class_name, user_roles=None):
    if user_roles is None:
        user_roles = {}
    
    if 'authentication_status' not in ss:
        ss.authentication_status = False

    # Always show the home and login navigators.
    HomeNav()
    LoginNav()

    # Show the other page navigators depending on the users' role.
    if ss["authentication_status"]:
        # (1) Only the admin role can access page 1 and other pages.
        # In a user roles get all the usernames with admin role.
        admins = [k for k, v in user_roles.items() if v == 'admin']

        # Show page 1 if the username that logged in is an admin.
        if ss.username in admins:
            UploadPDFNav()
            UploadPDFClassNav()
            Page1()

        # (2) users with user and admin roles have access to page 2.
        RAGModelNav()
        RAGModelClassNav(class_name)
    
