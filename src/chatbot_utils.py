from streamlit import session_state
from dotenv import load_dotenv

import src.utils as utils

# Load environment variables
load_dotenv()


def initialize_chat_session_variables(typ, language):
    """
    Initialize chat session variables with default values.

    Sets up the initial state for the chat application to function correctly from the start.
    """
    session_state['typ'] = typ
    session_state[f"prompt_options_{session_state['typ']}"] = utils.load_prompts(session_state['typ'], language)
    required_keys = {
        'edited_prompts': {},
        'disable_custom': False,
    }

    for key, default_value in required_keys.items():
        if key not in session_state:
            session_state[key] = default_value
