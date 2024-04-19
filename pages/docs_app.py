import streamlit as st
from streamlit import session_state

from dotenv import load_dotenv
import src.utils as utils
import src.docs_utils as doc_utils
from src.docs_utils import SidebarDocsManager, DocsManager
from src.utils import CookiesManager, SidebarManager, GeneralManager, AIClient, SummarizationManager
from src.language_utils import initialize_language

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Chat with your document", layout="wide", page_icon="📖")


# Main Application Class
class DocsApplication:

    def __init__(self):
        self.cookies_manager = CookiesManager()
        self.sidebar_general_manager = SidebarManager()
        self.general_manager = GeneralManager()
        self.summarization_manager = SummarizationManager()
        self.docs_manager = DocsManager(summarization_manager=self.summarization_manager)
        self.sidebar_doc_manager = SidebarDocsManager()

    def initialize_app(self):
        """Initializes the app configurations, verifies user session, and sets up the UI components."""

        # Set and manage sidebar interface controls
        utils.initialize_session_variables(user=session_state['_']("User"), typ='docs')
        doc_utils.initialize_docs_session_variables()
        initialize_language()
        self.cookies_manager.initialize_cookies()
        self.cookies_manager.verify_and_set_user_session()
        self.sidebar_general_manager.display_logo()
        self.sidebar_general_manager.display_pages_menu(default_id=1)
        self.sidebar_general_manager.display_general_sidebar_controls()
        self.sidebar_doc_manager.display_docs_sidebar_controls()
        self.sidebar_general_manager.display_general_sidebar_bottom_controls()

        # Initialize the client
        client = AIClient()

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
