from streamlit import session_state
from dotenv import load_dotenv

import src.utils as utils

# Load environment variables
load_dotenv()


def initialize_chat_session_variables():
    """
    Initialize chat session variables with default values.

    Sets up the initial state for the chat application to function correctly from the start.
    """
    utils.load_prompts(session_state['typ'])
    required_keys = {
        'edited_prompts': {},
        'disable_custom': False,
    }

    for key, default_value in required_keys.items():
        if key not in session_state:
            session_state[key] = default_value
