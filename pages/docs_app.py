import streamlit as st
from streamlit import session_state

import src.utils as utils
import src.docs_utils as doc_utils
from src.docs_manager import DocsManager
from src.docs_sidebar_manager import SidebarDocsManager
from src.sidebar_general_manager import SidebarManager
from src.language_manager import LanguageManager
from src.session_manager import SessionManager
from src.ai_client import AIClient

# Streamlit page config
st.set_page_config(page_title="Chat with your document", layout="wide", page_icon="🥝")


# Main Application Class
class DocsApplication:

    def __init__(self):
        self.session_manager = SessionManager()
        self.language_manager = LanguageManager()
        self.sidebar_general_manager = SidebarManager(self.session_manager)
        self.docs_manager = DocsManager()
        self.sidebar_doc_manager = SidebarDocsManager()

    def initialize_app(self):
        """Initializes the app configurations, verifies user session, and sets up the UI components."""

        # Set and manage sidebar interface controls
        self.language_manager.initialize_language()
        self.language_manager.language_controls()
        session_state['USER'] = session_state['_']("User")
        utils.initialize_session_variables()
        doc_utils.initialize_docs_session_variables(typ='docs', language=self.language_manager.get_language())
        self.session_manager.initialize_cookies()
        self.session_manager.verify_and_set_user_session()
        self.sidebar_general_manager.display_logo()
        self.sidebar_general_manager.display_pages_menu(default_id=1)
        self.sidebar_general_manager.display_general_sidebar_controls()

        # Initialize the client
        client = AIClient()

        self.sidebar_doc_manager.set_client(client)
        self.sidebar_doc_manager.display_docs_sidebar_controls()
        self.sidebar_general_manager.display_general_sidebar_bottom_controls()

        # Set and manage chat interface display
        self.docs_manager.set_window_dimensions()
        self.docs_manager.set_client(client)
        self.docs_manager.display_chat_interface()

    def run(self):
        """Runs the main application."""
        self.initialize_app()


# Main entry point
if __name__ == "__main__":
    app = DocsApplication()
    app.run()
