import streamlit as st
from streamlit import session_state

from dotenv import load_dotenv
from src.docs_utils import SidebarDocsControls, DocsManager
from src.utils import SidebarManager, AIClient

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Chat with your document", layout="wide", page_icon="📖")


# Main Application Class
class DocsApplication:

    def __init__(self):
        self.sidebar_manager = SidebarManager()
        self.docs_manager = DocsManager(user=session_state['_']("User"))
        self.sidebar_doc_controls = SidebarDocsControls()

    def initialize_app(self):
        """Initializes the app configurations, verifies user session, and sets up the UI components."""

        # Set and manage sidebar interface controls
        self.sidebar_manager.verify_user_session()
        self.sidebar_manager.initialize_session_variables()
        self.sidebar_doc_controls.initialize_docs_session_variables()
        self.sidebar_manager.display_logo()
        self.sidebar_manager.display_general_sidebar_controls()
        self.sidebar_doc_controls.display_docs_sidebar_controls()
        self.sidebar_manager.display_general_sidebar_bottom_controls()

        # Initialize the client
        client = AIClient()

        # Set and manage chat interface display
        self.docs_manager.set_client(client)
        self.docs_manager.load_doc_to_display()
        self.docs_manager.display_chat_interface()

    def run(self):
        """Runs the main application."""
        self.initialize_app()


# Main entry point
if __name__ == "__main__":
    app = DocsApplication()
    app.run()