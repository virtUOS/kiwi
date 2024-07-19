import os
import json

from streamlit import session_state
from streamlit_float import *
from dotenv import load_dotenv
from src.chat_manager import ChatManager
from src.ai_client import AIClient
from src.sidebar_manager import SidebarManager
import src.styling as styling
from src.language_utils import language_controls


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
        # Float feature initialization
        float_init()
        # Styling initialization
        styling.apply_styling()
        # App set up
        default_model = os.getenv("DEFAULT_MODEL")
        multi_models = json.loads(os.getenv("MULTI_MODELS"))
        multi_models = multi_models['models']
        self.sidebar_manager = SidebarManager(default_model=default_model, multi_models=multi_models)
        self.chat_manager = ChatManager(user=session_state['_']("User"),
                                        multi_models=multi_models,
                                        sidebar_manager=self.sidebar_manager)

    def initialize_app(self):
        """Initializes the app configurations, verifies user session, and sets up the UI components."""

        # Set and manage sidebar interface controls
        self.sidebar_manager.initialize_session_variables()
        self.sidebar_manager.display_logo()
        language_controls()
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
