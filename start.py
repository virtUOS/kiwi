import os
from time import sleep

from dotenv import load_dotenv
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

from src import ldap_connector

load_dotenv()

st.set_page_config(
    page_title="UOS Welcomes YOU",
    page_icon="👋",
    layout="wide"

)

# For session management.
# TODO: log out. There's a problem deleting cookies: https://github.com/ktosiek/streamlit-cookies-manager/issues/1

# This should be on top of your script
cookies = EncryptedCookieManager(
    # This prefix will get added to all your cookie names.
    # This way you can run your app on Streamlit Cloud without cookie name clashes with other apps.
    prefix="ktosiek/streamlit-cookies-manager/",
    # You should really setup a long COOKIES_PASSWORD secret if you're running on Streamlit Cloud.
    password=os.environ.get("COOKIES_PASSWORD"),
)

if not cookies.ready():
    # Wait for the component to load and send us current cookies.
    st.spinner()
    st.stop()


st.write("# Welcome to the AI Portal of Osnabrück University! 👋")

if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False

with st.sidebar:
    # Display the logo on the sidebar
    # Create three columns
    col1, col2, col3 = st.columns([1, 2, 1])

    # Use the middle column to display the logo
    with col2:
        st.image("img/logo.svg", width=100)

    def credentials_entered():
        """Checks whether a password entered by the user is correct."""
        user_found = ldap_connector.check_auth(username=st.session_state.username,
                                               password=st.session_state.password)
        if user_found:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

        if ("password_correct" in st.session_state and
                not st.session_state["password_correct"]):
            st.error("😕 Password incorrect")

    st.write("Login with your university credentials.")

    # Show input for password.
    st.text_input(
        "User", key="username"
    )

    # Show input for password.
    st.text_input(
        "Password", type="password", key="password"
    )

    st.button("Login", on_click=credentials_entered)


def check_password():
    return st.session_state["password_correct"]


st.markdown(
    f"""
    This portal is an open-source app to allow users to chat with several chatbot experts from OpenAI's ChatGPT.

    **👈 Login on the sidebar** to enter the chat area!
    ### Want to learn more about your rights as an user for this app?
    - Check out the [Datenschutz]({os.environ['DATENSCHUTZ']})
    - Check out the [Impressum]({os.environ['IMPRESSUM']})
"""
)

# First check if there's a session already started
if not cookies.get("session"):

    # If no session, then check password
    if not check_password():
        st.stop()

st.sidebar.success("Logged in!")

# Save cookie for the session. Use username as value, maybe it's useful at some point
cookies["session"] = st.session_state.username
cookies.save()

# Wait a bit before redirecting
sleep(0.5)

# Redirect to app
st.switch_page("pages/app.py")
