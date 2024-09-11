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

def RAGModelClassNav(data):
    ss.class_name = data['class_name']
    st.sidebar.page_link("/home/ubuntu/llama-model/pages/rag_model_stream_class.py", label="RAG Model Based On Class", icon='📚')

def UploadPDFHierarchicalNav():
    st.sidebar.page_link("/home/ubuntu/llama-model/pages/upload_pdf_page_hierarchical.py", label="Upload PDF's Based On Hierarchical Architecture", icon='✈️')

def RAGModelHierarchicalNav(data):
    ss.school_college_ce = data['school_college_ce']
    ss.board = data['board']
    ss.state_board = data['state_board']
    ss.class_name = data['class_name']
    ss.college_name = data['college_name']
    ss.stream_name = data['stream_name']
    ss.subject = data['subject']
    st.sidebar.page_link("/home/ubuntu/llama-model/pages/rag_model_stream_hierarchical.py", label="RAG Model Based On Hierarchical Architecture", icon='📚')


def MenuButtons(data, user_roles=None):
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
            UploadPDFHierarchicalNav()