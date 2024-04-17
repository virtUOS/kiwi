import streamlit as st
from streamlit import session_state
from dotenv import load_dotenv

from src.chatbot_utils import SidebarChatManager, ChatManager
from src.utils import SidebarManager, GeneralManager, AIClient
from src.language_utils import initialize_language

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="kiwi", layout="wide", page_icon="🥝")


# Main Application Class
class ChatbotApplication:

    def __init__(self):
        self.sidebar_general_manager = SidebarManager()
        self.general_manager = GeneralManager(sidebar_general_manager=self.sidebar_general_manager)
        self.sidebar_chat_manager = SidebarChatManager(sidebar_general_manager=self.sidebar_general_manager,
                                                       general_manager=self.general_manager)
        self.chat_manager = ChatManager(general_manager=self.general_manager)

    def initialize_app(self):
        """Initializes the app configurations, verifies user session, and sets up the UI components."""

        # Set and manage sidebar interface controls
        self.sidebar_general_manager.initialize_session_variables()
        self.sidebar_chat_manager.initialize_chat_session_variables()
        self.general_manager.display_pages_tabs(default_id=0)
        initialize_language()
        self.sidebar_general_manager.set_user(user=session_state['_']("User"))
        self.sidebar_general_manager.initialize_cookies()
        self.sidebar_general_manager.verify_and_set_user_session()
        self.sidebar_general_manager.display_logo()
        self.sidebar_chat_manager.display_chat_top_sidebar_controls()
        self.sidebar_general_manager.display_general_sidebar_controls()
        self.sidebar_chat_manager.display_chat_middle_sidebar_controls()
        self.sidebar_general_manager.display_general_sidebar_bottom_controls()

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
