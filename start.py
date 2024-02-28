# flake8: noqa

import os
from time import sleep
from dotenv import load_dotenv
import streamlit as st
from streamlit import session_state
from streamlit_cookies_manager import EncryptedCookieManager
from src import ldap_connector
from src.language_utils import initialize_language

load_dotenv()

st.set_page_config(
    page_title="Kiwi 🥝",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_language()

# For session management
# This should be on top of your script
cookies = EncryptedCookieManager(
    prefix=os.getenv("COOKIES_PREFIX"),
    password=os.getenv("COOKIES_PASSWORD")
)

if not cookies.ready():
    # Wait for the component to load and send us current cookies.
    st.spinner()
    st.stop()

current_language = st.query_params['lang']

if current_language == 'en':
    welcome_message = os.getenv('WELCOME_MESSAGE_EN')
else:
    welcome_message = os.getenv('WELCOME_MESSAGE_DE')


st.write(f"## {welcome_message}")

if "password_correct" not in session_state:
    session_state["password_correct"] = False

with st.sidebar:
    # Display the logo on the sidebar
    # Create three columns
    col1, col2, col3 = st.columns([1, 2, 1])

    # Use the middle column to display the logo
    with col2:
        st.image("img/logo.svg", width=100)

    def credentials_entered():
        """Checks whether a password entered by the user is correct."""
        user_found = ldap_connector.check_auth(username=session_state.username,
                                               password=session_state.password)
        if user_found:
            session_state["password_correct"] = True
            del session_state["password"]  # Don't store the password.
        else:
            session_state["password_correct"] = False

        session_state['credentials_checked'] = True

    st.write(session_state['_']("Login with your university credentials."))

    with st.form("login-form"):
        # Show input for password.
        st.text_input(
            session_state['_']("User name"), key="username"
        )

        # Show input for password.
        st.text_input(
            session_state['_']("Password"), type="password", key="password"
        )

        st.form_submit_button(session_state['_']("Login"), on_click=credentials_entered)

        if 'credentials_checked' in session_state and not session_state['password_correct']:
            st.error(session_state['_']("😕 Password incorrect"))

# Prepare links on legal stuff depending on the language chosen (German sites as default)
if 'DATENSCHUTZ_DE' in os.environ and 'IMPRESSUM_DE' in os.environ:
    dantenschutz_link = os.environ['DATENSCHUTZ_DE']
    impressum_link = os.environ['IMPRESSUM_DE']

# Use sites in english if the language changes and the sites are available
if current_language == 'en':
    if 'DATENSCHUTZ_EN' in os.environ and 'IMPRESSUM_EN' in os.environ:
        dantenschutz_link = os.environ['DATENSCHUTZ_EN']
        impressum_link = os.environ['IMPRESSUM_EN']


def check_password():
    return st.session_state["password_correct"]


md_msg = session_state['_']("""

This portal is an open source app allowing users to chat with OpenAI's GPT models
 without submitting personal data to OpenAI during the login process. Users should
  be aware that all information they enter, is submitted to OpenAI.

**👈 Login on the sidebar** to enter the chat area!
### Learn more about your rights as an user of this app in the [Privacy Policy]({DATENSCHUTZ}) and [Legal Notice]({IMPRESSUM}).

"""
                            ).format(DATENSCHUTZ=dantenschutz_link, IMPRESSUM=impressum_link)

st.markdown(md_msg)

# First check if there's a session already started
if cookies.get("session") != 'in':

    # If no session, then check password
    if not check_password():
        st.stop()
    else:
        # When the password is correct create a persistent session
        # Save cookie for the session. Use username as value, maybe it's useful at some point
        cookies["session"] = 'in'
        cookies.save()

if cookies['session'] == 'in':
    st.sidebar.success(session_state['_']("Logged in!"))
    # Wait a bit before redirecting
    sleep(0.5)

    # Redirect to app
    st.switch_page("pages/chatbot_app.py")
