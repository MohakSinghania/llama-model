from dependency import *
from modules.nav import MenuButtons
from pages.account import get_roles
from streamlit_rag_model import llama_model

rag_func = llama_model()

if 'authentication_status' not in ss:
    st.switch_page('./pages/account.py')

MenuButtons(get_roles())
st.header("UPLOAD YOUR PDF'S")
s_c_ce_type = ["school" , "college" , "competitve_exam"]
selected_s_c_ce_type = st.selectbox('Select Type', s_c_ce_type)
if selected_s_c_ce_type == "school":
    board_type = ["CBSE" , "ICSE" , "state_board"]
    selected_board_type = st.selectbox('Select Board Type', board_type)
    class_options = ['class_01', 'class_02', 'class_03', 'class_04', 'class_05', 'class_06', 'class_07', 'class_08', 'class_09', 'class_10', 'class_11', 'class_12']
    selected_class = st.selectbox('Select Class', class_options)



uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=["pdf"])

if st.button("Upload"):
    if not uploaded_files:
        st.error("Please upload PDF files")

    for file in uploaded_files:
        rag_func._pdf_file_save_s_c_ce_class(file, selected_s_c_ce_type, selected_board_type, selected_class)

    message = rag_func._create_embedding_s_c_ce(selected_s_c_ce_type, selected_board_type, selected_class)
    st.success(message['message'])