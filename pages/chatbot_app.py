import streamlit as st
from streamlit import session_state
from dotenv import load_dotenv

from src.chatbot_utils import SidebarChatControls, ChatManager
from src.utils import SidebarManager, PagesManager, AIClient

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="kiwi", layout="wide", page_icon="🥝")


# Main Application Class
class ChatbotApplication:

    def __init__(self):
        self.sidebar_manager = SidebarManager()
        self.sidebar_chat_manager = SidebarChatControls()
        self.pages_manager = PagesManager("1")
        self.chat_manager = ChatManager(user=session_state['_']("User"))

    def initialize_app(self):
        """Initializes the app configurations, verifies user session, and sets up the UI components."""

        # Set and manage sidebar interface controls
        self.sidebar_manager.verify_user_session()
        self.sidebar_manager.initialize_session_variables()
        self.sidebar_chat_manager.initialize_chat_session_variables()
        self.sidebar_manager.display_logo()
        self.sidebar_manager.display_general_sidebar_controls()
        self.sidebar_manager.display_general_sidebar_bottom_controls()

        # Initialize the client
        client = AIClient()

        # Set and manage chat interface display
        self.chat_manager.set_client(client)
        self.chat_manager.display_chat_interface()

    def run(self):
        """Runs the main application."""
        self.initialize_app()


# Main entry point
if __name__ == "__main__":
    app = ChatbotApplication()
    app.run()
