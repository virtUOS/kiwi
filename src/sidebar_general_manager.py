import streamlit as st
from streamlit import session_state
from streamlit_option_menu import option_menu

import src.streamlit_styling as st_styling
import src.utils as utils
from src.ai_client import AIClient


class SidebarManager:

    def __init__(self, session_manager):
        super().__init__()
        self.version = "beta-2.0.3"
        self.advanced_model = "gpt-4o"
        self.ssm = session_manager

    @staticmethod
    def display_logo():
        """
        Display the application logo in the sidebar.

        Utilizes Streamlit columns to center the logo and injects CSS to hide the fullscreen
        button that appears on hover over the logo.
        """
        with st.sidebar:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("img/logo.svg", width=100)
                # Gets rid of full screen option for logo image
                hide_img_fs = '''
                <style>
                button[title="View fullscreen"]{
                    visibility: hidden;}
                </style>
                '''
                st.markdown(hide_img_fs, unsafe_allow_html=True)

    @staticmethod
    def display_pages_menu(default_id):
        pages = [session_state['_']("Chat with Documents")]

        with st.sidebar:
            page = option_menu("", pages, icons=['chat-dots'] * len(pages),
                               menu_icon="cast",
                               default_index=default_id)

        session_state['chosen_page_index'] = pages.index(page)
        session_state['selected_chatbot_path'] = [page]

    def _display_model_information(self):
        """
        Display OpenAI model information in the sidebar if the OpenAI model is the current selection.

        In the sidebar, this static method shows the model name and version pulled from environment variables
        if 'OpenAI' is selected as the model in the session state. The model information helps users
        identify the active model configuration.
        """
        with st.sidebar:
            if session_state['model_selection'] == 'OpenAI':
                AIClient.set_accessible_models()

                model_label = session_state['_']("Model:")
                # Show dropdown when multiple models are available
                index = 0
                if self.advanced_model in session_state['accessible_models']:  # Use most advanced model as default
                    index = session_state['accessible_models'].index('gpt-4o')
                st.selectbox(model_label,
                             session_state['accessible_models'],
                             index=index,
                             key='selected_model')

                model_label_extra = session_state['_']("Model for Summaries:")
                index_extra = 0
                # Show dropdown when multiple models are available
                st.selectbox(model_label_extra,
                             session_state['accessible_models_extra'],
                             index=index_extra,
                             key='selected_model_extra')

    def display_general_sidebar_controls(self):
        """
        Display sidebar controls and settings for the page.

        This function performs the following steps:
        1. Checks for changes in the selected chatbot path to update the session state accordingly.
        2. Displays model information for the selected model, e.g., OpenAI.
        """
        utils.update_path_in_session_state()
        self._display_model_information()

    def _logout_option(self):
        """
        Display a logout button in the sidebar, allowing users to end their session.

        This method renders a logout button in the sidebar. If clicked, it calls the `logout_and_redirect` method
        to clear session variables and cookies, securely logging out the user and redirecting them to the start page.
        """
        with st.sidebar:
            if st.button(session_state['_']('Logout')):
                self.ssm.logout_and_redirect()
            st.write(f"Version: *{self.version}*")

    def display_general_sidebar_bottom_controls(self):
        """
        Display bottom sidebar controls and settings for page.

        This function performs the following steps:
        1. Adds custom CSS to stylize the sidebar.
        2. Offers a logout option.
        """
        st_styling.add_custom_css()
        self._logout_option()
