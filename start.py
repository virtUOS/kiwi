import os
from time import sleep

from dotenv import load_dotenv
import streamlit as st
from streamlit import session_state as ss
from streamlit_cookies_manager import EncryptedCookieManager

from src import ldap_connector

from src.language_utils import initialize_language

load_dotenv()

st.set_page_config(
    page_title="UOS Welcomes YOU",
    page_icon="👋",
    layout="wide"
)

initialize_language()

# For session management.
# TODO: log out. There's a problem deleting cookies: https://github.com/ktosiek/streamlit-cookies-manager/issues/1

# This should be on top of your script
cookies = EncryptedCookieManager(
    # This prefix will get added to all your cookie names.
    # This way you can run your app on Streamlit Cloud without cookie name clashes with other apps.
    prefix="virtuos/ai-portal",
    # You should really setup a long COOKIES_PASSWORD secret if you're running on Streamlit Cloud.
    password=os.environ.get("COOKIES_PASSWORD"),
)

if not cookies.ready():
    # Wait for the component to load and send us current cookies.
    st.spinner()
    st.stop()

st.write(ss['_']("# Welcome to the AI Portal of Osnabrück University! 👋"))

if "password_correct" not in ss:
    ss["password_correct"] = False

with st.sidebar:
    # Display the logo on the sidebar
    # Create three columns
    col1, col2, col3 = st.columns([1, 2, 1])

    # Use the middle column to display the logo
    with col2:
        st.image("img/logo.svg", width=100)


    def credentials_entered():
        """Checks whether a password entered by the user is correct."""
        user_found = ldap_connector.check_auth(username=ss.username,
                                               password=ss.password)
        if user_found:
            ss["password_correct"] = True
            del ss["password"]  # Don't store the password.
        else:
            ss["password_correct"] = False

        ss['credentials_checked'] = True


    st.write(ss['_']("Login with your university credentials."))

    with st.form("login-form"):
        # Show input for password.
        st.text_input(
            ss['_']("User"), key="username"
        )

        # Show input for password.
        st.text_input(
            ss['_']("Password"), type="password", key="password"
        )

        st.form_submit_button(ss['_']("Login"), on_click=credentials_entered)

        if 'credentials_checked' in ss and not ss['password_correct']:
            st.error("😕 Password incorrect")


def check_password():
    return st.session_state["password_correct"]


md_msg = ss['_']("""
    This portal is an open-source app to allow users to chat with several chatbot experts from OpenAI's ChatGPT.

    **👈 Login on the sidebar** to enter the chat area!
    ### Want to learn more about your rights as an user for this app?
    - Check out the [Datenschutz]({DATENSCHUTZ})
    - Check out the [Impressum]({IMPRESSUM})


"""
                 ).format(DATENSCHUTZ=os.environ['DATENSCHUTZ'], IMPRESSUM=os.environ['IMPRESSUM'])

st.markdown(md_msg)

# First check if there's a session already started
if not cookies.get("session"):

    # If no session, then check password
    if not check_password():
        st.stop()
    else:
        # When the password is correct create a persistent session
        # Save cookie for the session. Use username as value, maybe it's useful at some point
        cookies["session"] = 'in'
        cookies.save()


if cookies['session'] == 'in':
    st.sidebar.success(ss['_']("Logged in!"))
    # Wait a bit before redirecting
    sleep(0.5)

    # Redirect to app
    st.switch_page("pages/chatbot_app.py")
