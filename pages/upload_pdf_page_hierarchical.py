from dependency import *
from modules.nav import MenuButtons
from pages.account import get_roles
from streamlit_rag_model import llama_model

streamlit_rag_model = llama_model()

if 'authentication_status' not in ss:
    st.switch_page('./pages/account.py')

MenuButtons(get_roles())
st.header("UPLOAD YOUR PDF'S")
s_c_ce_type = ["school" , "college" , "competitve_exam"]
selected_s_c_ce_type = st.selectbox('Select Type', s_c_ce_type)
if selected_s_c_ce_type == "school":
    board_type = ["CBSE" , "ICSE" , "state_board"]
    selected_board_type = st.selectbox('Select Board Type', board_type)
    if selected_board_type == "state_board":
        state_board = [
                        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
                        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
                        "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
                        "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", 
                        "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
                        "West Bengal"
                    ]
        selected_state_board = st.selectbox('Select State Board Type', state_board)
    else:
        selected_state_board = None
    class_options = ['class_01', 'class_02', 'class_03', 'class_04', 'class_05', 'class_06', 'class_07', 'class_08', 'class_09', 'class_10', 'class_11', 'class_12']
    selected_class = st.selectbox('Select Class', class_options)
elif selected_s_c_ce_type == "college":
    college_name_type = ['Govt Holkar Science College', 'DAVV', 'RGPV']
    selected_college_name_type = st.selectbox('Select College/University Name', college_name_type)
    stream_type = ['B.Sc', 'M.Sc', 'B.E']
    selected_stream_type = st.selectbox('Select Stream', stream_type)
    subject_type = ['Computer Science', 'Mathematics', 'Statistics']
    selected_subject_type = st.selectbox('Select Subject', subject_type)
else:
    pass

uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=["pdf"])

if st.button("Upload"):
    if not uploaded_files:
        st.error("Please upload PDF files")
    if selected_s_c_ce_type == 'school':
        for file in uploaded_files:
            pdf_details = streamlit_rag_model._pdf_file_save_selection(file, selected_s_c_ce_type, board_type=selected_board_type, state_board=selected_state_board, class_name=selected_class)
            message = streamlit_rag_model._create_embedding_selection(pdf_details['pdf_id'], pdf_details['pdf_path'], selected_s_c_ce_type, board_type=selected_board_type, state_board=selected_state_board, class_name=selected_class)
    elif selected_s_c_ce_type == 'college':
        for file in uploaded_files:
            pdf_details = streamlit_rag_model._pdf_file_save_selection(file, selected_s_c_ce_type, college_name=selected_college_name_type, stream_name=selected_stream_type, subject_name=selected_subject_type)
            message = streamlit_rag_model._create_embedding_selection(pdf_details['pdf_id'], pdf_details['pdf_path'], selected_s_c_ce_type, college_name=selected_college_name_type, stream_name=selected_stream_type, subject_name=selected_subject_type)
    else:
        pass
    st.success(message['message'])