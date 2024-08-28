from dependency import *
import constants
from modules.nav import MenuButtons


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

def get_class_name(username):
    """Retrieve the class name associated with the logged-in user."""
    if username in config['credentials']['usernames']:
        return config['credentials']['usernames'][username].get('class', None)
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
        ss.class_name = get_class_name(ss["username"])
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
                class_options = ['class_01', 'class_02', 'class_03', 'class_04', 'class_05', 'class_06', 'class_07', 'class_08', 'class_09', 'class_10', 'class_11', 'class_12']
                selected_class = st.selectbox('Select Class', class_options)
            else:
                selected_class = None
            email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(pre_authorization=False)
            if email_of_registered_user:
                # Add role to config
                config['credentials']['usernames'][username_of_registered_user]['role'] = selected_role
                # Initialize class for new user (optional)
                config['credentials']['usernames'][username_of_registered_user]['class'] = selected_class  # Set a default class if needed
                st.success('User registered successfully')
        except Exception as e:
            st.error(e)

# Save the updated configuration file
with open(constants.CONFIG_FILENAME, 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

MenuButtons(ss.class_name, get_roles())