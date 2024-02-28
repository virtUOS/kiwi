import streamlit as st
from dotenv import load_dotenv
from src.chatbot_utils import SidebarManager, ChatManager, AIClient

from src.language_utils import initialize_language


# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Kiwi", layout="wide", page_icon="🥝")


# Main Application Class
class ChatbotApplication:

    def __init__(self, USER):
        self.sidebar_manager = SidebarManager()
        self.chat_manager = ChatManager(user=USER)

    def initialize_app(self):
        """Initializes the app configurations, verifies user session, and sets up the UI components."""
        # Set the language
        initialize_language()

        # Set and manage sidebar interface controls
        self.sidebar_manager.verify_user_session()
        self.sidebar_manager.initialize_session_variables()
        self.sidebar_manager.display_logo()
        self.sidebar_manager.display_sidebar_controls()

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
    USER = "User"
    app = ChatbotApplication(USER)
    app.run()
