import yaml
import constants
import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from modules.nav_1 import MenuButtons
from streamlit import session_state as ss

# Load the configuration file
with open(constants.CONFIG_FILENAME) as file:
    config = yaml.load(file, Loader=SafeLoader)


def get_roles():
    """Gets user roles based on config file."""
    with open(constants.CONFIG_FILENAME) as file:
        config = yaml.load(file, Loader=SafeLoader)

    if config is not None:
        cred = config['credentials']
    else:
        cred = {}

    return {username: user_info['role'] for username, user_info in cred['usernames'].items() if 'role' in user_info}


def get_user_data(username):
    """Retrieve the class name associated with the logged-in user."""
    if username in config['credentials']['usernames']:
        data = {
            'school_college_ce': config['credentials']['usernames'][username].get('school_college_ce', None),
            'Board': config['credentials']['usernames'][username].get('Board', None),
            'State_Board': config['credentials']['usernames'][username].get('State_Board', None),
            'Class': config['credentials']['usernames'][username].get('Class', None),
            'Stream': config['credentials']['usernames'][username].get('Stream', None),
            'College_Name': config['credentials']['usernames'][username].get('College_Name', None),
            'Course_Name': config['credentials']['usernames'][username].get('Course_Name', None),
            'Subject': config['credentials']['usernames'][username].get('Subject', None)
        }
        return data
    return None


st.header('Account page')

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

login_tab, register_tab = st.tabs(['Login', 'Register'])

with login_tab:
    authenticator.login(location='main')

    if ss.get("authentication_status"):
        # Retrieve the class name for the logged-in user
        ss.data = get_user_data(ss["username"])
        authenticator.logout(location='main')
        st.write(f'Welcome *{ss["name"]}*')

    elif ss.get("authentication_status") is False:
        st.error('Username/password is incorrect')
    elif ss.get("authentication_status") is None:
        st.warning('Please enter your username and password')

with register_tab:
    if not ss.get("authentication_status"):
        try:
            role_options = ['admin', 'user']  # Define your role options here
            selected_role = st.selectbox('Select Role', role_options)
            if selected_role == 'user':
                s_c_ce_type = ["school", "college", "competitve_exam"]
                selected_s_c_ce_type = st.selectbox('Select Type', s_c_ce_type)
                if selected_s_c_ce_type == "school":
                    board_type = ["CBSE", "ICSE", "state_board"]
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
                    class_options = [
                                    'class_01', 'class_02', 'class_03', 'class_04', 'class_05', 'class_06', 'class_07', 'class_08', 'class_09',
                                    'class_10', 'class_11', 'class_12'
                                    ]
                    selected_class = st.selectbox('Select Class', class_options)
                    if selected_class == 'class_11' or selected_class == 'class_12':
                        stream_type = ['Commerce', 'Science', 'Arts', 'Medical']
                        selected_stream = st.selectbox('Select Stream Type', stream_type)
                    else:
                        selected_stream = None
                    selected_college_name_type = None
                    selected_course_type = None
                    selected_subject_type = None
                elif selected_s_c_ce_type == "college":
                    college_name_type = ['Govt Holkar Science College', 'DAVV', 'RGPV']
                    selected_college_name_type = st.selectbox('Select College/University Name', college_name_type)
                    stream_type = ['Commerce', 'Science', 'Arts', 'Medical']
                    selected_stream = st.selectbox('Select Stream Type', stream_type)
                    course_type = ['B.Sc', 'M.Sc', 'B.E']
                    selected_course_type = st.selectbox('Select Course', course_type)
                    subject_type = ['Computer Science', 'Mathematics', 'Statistics']
                    selected_subject_type = st.selectbox('Select Subject', subject_type)
                    selected_board_type = None
                    selected_state_board = None
                    selected_class = None
                else:
                    pass
            else:
                selected_s_c_ce_type = None
                selected_board_type = None
                selected_state_board = None
                selected_class = None
                selected_stream = None
                selected_course_name_type = None
                selected_stream_type = None
                selected_subject_type = None

            email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(pre_authorization=False)
            if email_of_registered_user:
                # Add role to config
                config['credentials']['usernames'][username_of_registered_user]['role'] = selected_role
                # Initialize class for new user (optional)
                config['credentials']['usernames'][username_of_registered_user]['school_college_ce'] = selected_s_c_ce_type
                if selected_s_c_ce_type == 'school':
                    config['credentials']['usernames'][username_of_registered_user]['Board'] = selected_board_type
                    if selected_board_type != 'state_board':
                        config['credentials']['usernames'][username_of_registered_user]['State_Board'] = selected_state_board
                    else:
                        config['credentials']['usernames'][username_of_registered_user]['State_Board'] = None
                    config['credentials']['usernames'][username_of_registered_user]['Class'] = selected_class
                    if selected_class == 'class_11' or selected_class == 'class_12':
                        config['credentials']['usernames'][username_of_registered_user]['Stream'] = selected_stream
                    else:
                        config['credentials']['usernames'][username_of_registered_user]['Stream'] = None
                    config['credentials']['usernames'][username_of_registered_user]['College_Name'] = selected_college_name_type
                    config['credentials']['usernames'][username_of_registered_user]['Course_Name'] = selected_course_type
                    config['credentials']['usernames'][username_of_registered_user]['Subject'] = selected_subject_type

                elif selected_s_c_ce_type == 'college':
                    config['credentials']['usernames'][username_of_registered_user]['Board'] = selected_board_type
                    config['credentials']['usernames'][username_of_registered_user]['State_Board'] = selected_state_board
                    config['credentials']['usernames'][username_of_registered_user]['Class'] = selected_class
                    config['credentials']['usernames'][username_of_registered_user]['College_Name'] = selected_college_name_type
                    config['credentials']['usernames'][username_of_registered_user]['Stream'] = selected_stream
                    config['credentials']['usernames'][username_of_registered_user]['Course_Name'] = selected_course_type
                    config['credentials']['usernames'][username_of_registered_user]['Subject'] = selected_subject_type

                else:
                    config['credentials']['usernames'][username_of_registered_user]['Board'] = selected_board_type
                    config['credentials']['usernames'][username_of_registered_user]['State_Board'] = selected_state_board
                    config['credentials']['usernames'][username_of_registered_user]['Class'] = selected_class
                    config['credentials']['usernames'][username_of_registered_user]['Stream'] = selected_stream
                    config['credentials']['usernames'][username_of_registered_user]['College_Name'] = selected_college_name_type
                    config['credentials']['usernames'][username_of_registered_user]['Course_Name'] = selected_course_type
                    config['credentials']['usernames'][username_of_registered_user]['Subject'] = selected_subject_type

                st.success('User registered successfully')
        except Exception as e:
            st.error(e)

# Save the updated configuration file
with open(constants.CONFIG_FILENAME, 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

MenuButtons(ss.data, get_roles())
