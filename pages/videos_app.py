import streamlit as st
from streamlit import session_state

from dotenv import load_dotenv
from src.videos_utils import SidebarVideosControls, VideosManager
from src.utils import SidebarManager, GeneralManager, AIClient

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Chat with your videos", layout="wide", page_icon=":video_camera:")


# Main Application Class
class DocsApplication:

    def __init__(self):
        self.sidebar_manager = SidebarManager()
        self.general_manager = GeneralManager()
        self.videos_manager = VideosManager(user=session_state['_']("User"), general_manager=self.general_manager)
        self.sidebar_vid_controls = SidebarVideosControls(sidebar_general_manager=self.sidebar_manager,
                                                          general_manager=self.general_manager)

    def initialize_app(self):
        """Initializes the app configurations, verifies user session, and sets up the UI components."""

        # Set and manage sidebar interface controls
        self.sidebar_manager.verify_user_session()
        self.sidebar_manager.initialize_session_variables()
        self.sidebar_vid_controls.initialize_videos_session_variables()
        self.sidebar_manager.display_logo()
        self.sidebar_manager.display_general_sidebar_controls(display_chatbot_menu=False)

        # Initialize the client
        client = AIClient()

        # Set and manage videos interface display
        self.sidebar_vid_controls.set_client(client)
        self.sidebar_vid_controls.display_videos_sidebar_controls()
        self.general_manager.display_pages_tabs("3")
        self.sidebar_manager.display_general_sidebar_bottom_controls()
        self.videos_manager.set_client(client)
        self.videos_manager.display_video_interface()

    def run(self):
        """Runs the main application."""
        self.initialize_app()


# Main entry point
if __name__ == "__main__":
    app = DocsApplication()
    app.run()
