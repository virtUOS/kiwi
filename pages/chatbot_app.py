import streamlit as st
from streamlit import session_state
from dotenv import load_dotenv
from src.chatbot_utils import SidebarManager, ChatManager, AIClient

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="kiwi",
                   layout="wide",
                   page_icon="🥝",
                   initial_sidebar_state="auto")


# Main Application Class
class ChatbotApplication:

    def __init__(self):
        advanced_model = "gpt-4o"
        self.sidebar_manager = SidebarManager(advanced_model=advanced_model)
        self.chat_manager = ChatManager(user=session_state['_']("User"), advanced_model=advanced_model)

    def initialize_app(self):
        """Initializes the app configurations, verifies user session, and sets up the UI components."""

        # Set and manage sidebar interface controls
        self.sidebar_manager.initialize_session_variables()
        self.sidebar_manager.display_logo()
        self.sidebar_manager.display_sidebar_controls()

        # Initialize the client
        client = AIClient()

        # Set and manage chat interface display
        self.chat_manager.set_client(client)
        self.sidebar_manager.verify_user_session()
        self.chat_manager.display_chat_interface()

    def run(self):
        """Runs the main application."""
        self.initialize_app()


# Main entry point
if __name__ == "__main__":
    app = ChatbotApplication()
    app.run()
