# flake8: noqa

import os
from time import sleep
from dotenv import load_dotenv
import streamlit as st
from streamlit import session_state
from streamlit_cookies_manager import EncryptedCookieManager
from src import ldap_connector
from src.language_utils import initialize_language, language_controls

load_dotenv()

st.set_page_config(
    page_title="kiwi",
    page_icon="🥝",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

initialize_language()

current_language = st.query_params['lang']

if 'password_correct' not in session_state:
    session_state['password_correct'] = False

if 'credentials_checked' not in session_state:
    session_state['credentials_checked'] = False

with st.sidebar:
    # Display the logo on the sidebar
    # Create three columns
    col1, col2, col3 = st.columns([1, 2, 1])

    # Use the middle column to display the logo
    with col2:
        st.image("img/logo.svg", width=100)

    language_controls()

    def credentials_entered():
        """Checks whether a password entered by the user is correct."""
        #user_found = ldap_connector.check_auth(username=session_state['username_input'],
        #                                       password=session_state['password_input'])
        user_found = True
        if user_found:
            session_state['password_correct'] = True
            del session_state['password_input']  # Don't store the password.
        else:
            session_state['password_correct'] = False

        session_state['credentials_checked'] = True

    st.write(session_state['_']("Login with your university credentials."))

    hide_submit_text = """
    <style>
    div[data-testid="InputInstructions"] > span:nth-child(1) {
        visibility: hidden;
    }
    </style>
    """
    st.markdown(hide_submit_text, unsafe_allow_html=True)

    with st.form("login-form"):
        # Show input for password.
        st.text_input(
            session_state['_']("User name"), key='username_input'
        )

        # Show input for password.
        st.text_input(
            session_state['_']("Password"), type="password", key='password_input'
        )

        st.form_submit_button(session_state['_']("Login"), on_click=credentials_entered)

        if session_state['credentials_checked'] and not session_state['password_correct']:
            st.error(session_state['_']("Password incorrect"))


def check_password():
    return session_state["password_correct"]


def prepare_streamlit_links_to_legal_pages(link):
    # Extract page name without .py extension
    page = link.split('/')[1][0:-3]
    return page


# Use data in english if the language changes and the sites are available (German default)
if current_language == 'en':
    institution_name = os.getenv('INSTITUTION_EN')
    APP_NAME = os.getenv('APP_NAME_EN')
    if 'DATENSCHUTZ_EN' in os.environ and 'IMPRESSUM_EN' in os.environ:
        datenschutz_link = os.environ['DATENSCHUTZ_EN']
        impressum_link = os.environ['IMPRESSUM_EN']

    if datenschutz_link.endswith(".py") and impressum_link.endswith(".py"):
        app_path = os.getenv('APP_PATH')
        datenschutz_page = prepare_streamlit_links_to_legal_pages(datenschutz_link)
        impressum_page = prepare_streamlit_links_to_legal_pages(impressum_link)

        md_msg = f"""

                # Welcome to {APP_NAME}!
                ## Use. ChatGPT. Securely.

                ##### {APP_NAME} is an open source app of {institution_name}: It provides data protection-compliant access to OpenAI's GPT models. In concrete terms, this means that you do not need an account and all input that you transmit to the GPT models via kiwi may not be used by OpenAI to train the models. Please note, however, that the information will still be sent to OpenAI. Students are required to observe the guidelines of their subjects when using the models in the context of their studies.

                ##### General legal information can be found in the <a href="{datenschutz_page}" target="_self">Privacy Policy</a> and <a href="{impressum_page}" target="_self">Legal Notice</a>.

                ##### **Login on the sidebar** to enter the chat area.
                """
    else:

        md_msg = ("""
    
        # Welcome to {APP_NAME}!
        ## Use. ChatGPT. Securely.
    
        ##### {APP_NAME} is an open source app of {INSTITUTION}: It provides data protection-compliant access to OpenAI's GPT models. In concrete terms, this means that you do not need an account and all input that you transmit to the GPT models via kiwi may not be used by OpenAI to train the models. Please note, however, that the information will still be sent to OpenAI. Students are required to observe the guidelines of their subjects when using the models in the context of their studies.
    
        ##### General legal information can be found in the [Privacy Policy]({DATENSCHUTZ}) and [Legal Notice]({IMPRESSUM}).
    
        ##### **Login on the sidebar** to enter the chat area.
        """
                  ).format(APP_NAME=APP_NAME, DATENSCHUTZ=datenschutz_link, IMPRESSUM=impressum_link,
                           INSTITUTION=institution_name)
else:
    institution_name = os.getenv('INSTITUTION_DE')
    APP_NAME = os.getenv('APP_NAME_DE')
    if 'DATENSCHUTZ_DE' in os.environ and 'IMPRESSUM_DE' in os.environ:
        datenschutz_link = os.environ['DATENSCHUTZ_DE']
        impressum_link = os.environ['IMPRESSUM_DE']

    if datenschutz_link.endswith(".py") and impressum_link.endswith(".py"):
        app_path = os.getenv('APP_PATH')
        datenschutz_page = prepare_streamlit_links_to_legal_pages(datenschutz_link)
        impressum_page = prepare_streamlit_links_to_legal_pages(impressum_link)

        md_msg = f"""

                # Herzlich Willkommen bei {APP_NAME}!
                ## ChatGPT. Sicher. Nutzen.

                ##### {APP_NAME} ist eine Open Source-Anwendung der {institution_name}: Sie ermöglicht bietet einen datenschutzkonformen Zugang zu OpenAI‘s GPT-Modellen. Das bedeutete konkret: Sie brauchen keinen Account und sämtliche Eingaben, die Sie über kiwi an die GPT-Modelle übermitteln, dürfen von OpenAI nicht zum Training der Modelle genutzt werden. Bitte beachten Sie jedoch, dass die Informationen trotzdem an OpenAI gesendet werden. Studierende sind dazu angehalten, bei der Nutzung im Kontext des Studiums die Vorgaben ihrer Fächer zu beachten.

                ##### Mehr zu den rechtlichen Hintergründen erfahren Sie in den <a href="{datenschutz_page}" target="_self">Datenschutzhinweisen</a> und im <a href="{impressum_page}" target="_self">Impressum</a>.

                ##### Um den Chat-Bereich zu betreten, **melden Sie sich in der Seitenleiste an**.
                """
    else:
        md_msg = ("""
    
        # Herzlich Willkommen bei {APP_NAME}!
        ## ChatGPT. Sicher. Nutzen.
    
        ##### {APP_NAME} ist eine Open Source-Anwendung der {INSTITUTION}: Sie ermöglicht bietet einen datenschutzkonformen Zugang zu OpenAI‘s GPT-Modellen. Das bedeutete konkret: Sie brauchen keinen Account und sämtliche Eingaben, die Sie über kiwi an die GPT-Modelle übermitteln, dürfen von OpenAI nicht zum Training der Modelle genutzt werden. Bitte beachten Sie jedoch, dass die Informationen trotzdem an OpenAI gesendet werden. Studierende sind dazu angehalten, bei der Nutzung im Kontext des Studiums die Vorgaben ihrer Fächer zu beachten.
    
        ##### Mehr zu den rechtlichen Hintergründen erfahren Sie in den [Datenschutzhinweisen]({DATENSCHUTZ}) und im [Impressum]({IMPRESSUM}).
        
        ##### Um den Chat-Bereich zu betreten, **melden Sie sich in der Seitenleiste an**.
        """
                  ).format(APP_NAME=APP_NAME, DATENSCHUTZ=datenschutz_link, IMPRESSUM=impressum_link,
                           INSTITUTION=institution_name)

st.markdown(md_msg, unsafe_allow_html=True)

# First check if there's a session already started
if cookies.get('session') != 'in':

    # If no session, then check password
    if not check_password():
        st.stop()
    else:
        # When the password is correct create a persistent session
        # Save cookie for the session. Use username as value, maybe it's useful at some point
        cookies['session'] = 'in'
        cookies['username'] = session_state['username_input']  # Persist username
        cookies.save()

if cookies['session'] == 'in':
    st.sidebar.success(session_state['_']("Logged in!"))
    # Wait a bit before redirecting
    sleep(0.5)

    # Redirect to app
    st.switch_page("pages/chatbot_app.py")
