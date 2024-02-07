import streamlit as st
from dotenv import load_dotenv
import os
import nginx_ldap_connector as nglc

load_dotenv()

st.set_page_config(
    page_title="UOS Welcomes YOU",
    page_icon="👋",
    layout="wide"

)

st.write("# Welcome to the AI Portal of Universität Osnabrück! 👋")

if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False

with (((st.sidebar))):
    # Display the logo on the sidebar
    # Create three columns
    col1, col2, col3 = st.columns([1, 2, 1])

    # Use the middle column to display the logo
    with col2:
        st.image("img/logo.svg", width=100)

    def credentials_entered():
        """Checks whether a password entered by the user is correct."""
        if nglc.ldap_login(username=st.session_state.username,
                           password=st.session_state.password):
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
    - Check out [Datenschutz]({os.environ['DATENSCHUTZ']})
    - Check out the [Impressum]({os.environ['IMPRESSUM']})
"""
)

if not check_password():
    st.stop()

st.success("ENTER")
st.page_link()
