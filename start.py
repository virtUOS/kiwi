import os
from time import sleep
import gettext
from dotenv import load_dotenv
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
from src import ldap_connector
from utils.page_language import set_language, set_default_language, translate, languages

load_dotenv()

st.set_page_config(
    page_title="UOS Welcomes YOU",
    page_icon="👋",
    layout="wide"

)



set_default_language()


# with st.sidebar:
#     st.markdown(_("# Language"))
#     selected_language = st.selectbox(_('Select your language:'),
#                                      list(languages.values()),
#                                      )
#     st.session_state["selected_language"] = selected_language
#     set_language()


sel_lang = st.radio(
'Languages',
    options=languages.values(),
    label_visibility='hidden',
    horizontal=True,

)

st.session_state["selected_language"] = sel_lang
set_language()

if "selected_language" in st.session_state:
    if st.session_state["selected_language"] == '🇩🇪':
        _ = translate()
    else:
        _ = gettext.gettext

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

st.write(_("# Welcome to the AI Portal of Osnabrück University! 👋"))

# st.write("# Welcome to the AI Portal of Osnabrück University! 👋")

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

    st.write(_("Login with your university credentials."))

    with st.form("login-form"):
        # Show input for password.
        st.text_input(
            _("User"), key="username"
        )

        # Show input for password.
        st.text_input(
            _("Password"), type="password", key="password"
        )

        st.form_submit_button("Login", on_click=credentials_entered)


def check_password():
    return st.session_state["password_correct"]

md_msg = _("""
    This portal is an open-source app to allow users to chat with several chatbot experts from OpenAI's ChatGPT.

    **👈 Login on the sidebar** to enter the chat area!
    ### Want to learn more about your rights as an user for this app?
    - Check out the [Datenschutz]({DATENSCHUTZ})
    - Check out the [Impressum]({IMPRESSUM})
    

"""
).format(DATENSCHUTZ=os.environ['DATENSCHUTZ'], IMPRESSUM=os.environ['IMPRESSUM'])

st.markdown(md_msg)

# First check if there's a session already started
# if not cookies.get("session"):
#
#     # If no session, then check password
#     if not check_password():
#         st.stop()

    # If no session, then check password
    # if not check_password():
    #     st.stop()
    # else:
    #     # When the password is correct create a persistent session
    #     # Save cookie for the session. Use username as value, maybe it's useful at some point
    #     cookies["session"] = st.session_state.username
    #     cookies.save()

st.sidebar.success(_("Logged in!"))
# Wait a bit before redirecting
sleep(10)

# Redirect to app
st.switch_page("pages/app.py")
