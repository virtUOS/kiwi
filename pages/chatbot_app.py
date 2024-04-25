import streamlit as st
from streamlit import session_state
from dotenv import load_dotenv

import src.utils as utils
import src.chatbot_utils as chatbot_utils
from src.chatbot_manager import ChatManager
from src.chatbot_sidebar_manager import SidebarChatManager
from src.language_manager import LanguageManager
from src.session_manager import SessionManager
from src.sidebar_general_manager import SidebarManager
from src.ai_client import AIClient

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="kiwi", layout="wide", page_icon="🥝")


# Main Application Class
class ChatbotApplication:

    def __init__(self):
        self.cookies_manager = SessionManager()
        self.language_manager = LanguageManager()
        self.sidebar_general_manager = SidebarManager()
        self.sidebar_chat_manager = SidebarChatManager()
        self.chat_manager = ChatManager()

    def initialize_app(self):
        """Initializes the app configurations, verifies user session, and sets up the UI components."""

        # Set and manage sidebar interface controls
        self.language_manager.initialize_language()
        self.language_manager.language_controls()
        utils.initialize_session_variables(user=session_state['_']("User"), typ='chat')
        chatbot_utils.initialize_chat_session_variables()
        self.cookies_manager.initialize_cookies()
        self.cookies_manager.verify_and_set_user_session()
        self.sidebar_general_manager.display_logo()
        self.sidebar_general_manager.display_pages_menu(default_id=0)
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
