import re
import streamlit as st
from streamlit import session_state

from src import menu_utils


def initialize_session_variables():
    """
    Initialize essential session variables with default values.

    Sets up the initial state for model selection, chatbot paths, conversation histories,
    prompt options, etc., essential for the application to function correctly from the start.
    """
    required_keys = {
        'model_selection': "OpenAI",
        'conversation_histories': {},
        'selected_chatbot_path': [],
        'selected_chatbot_path_serialized': "",
        'chosen_page_index': "1",
        'current_page_index': "1",
        'typ': "chat",
    }

    for key, default_value in required_keys.items():
        if key not in session_state:
            session_state[key] = default_value


@st.cache_data
def load_prompts(typ, language=None):
    """
    Load chat prompts based on the selected or default language.

    This method enables dynamic loading of chat prompts to support multilingual chatbot interactions.
    If no language is specified in the query parameters, German ('de') is used as the default language.
    """
    # Use German language as default
    if not language:
        language = "de"

    return menu_utils.load_prompts_from_yaml(typ=typ, language=language)


def update_path_in_session_state():
    """
    Check if the selected chatbot path in the session state has changed and update the serialized path accordingly.

    This static method compares the current selected chatbot path with the serialized chatbot path saved
     in the session.
    If there is a change, it updates the session state with the new serialized path, allowing for tracking
     the selection changes.
    """
    path_has_changed = True
    if 'selected_chatbot_path_serialized' in session_state:
        path_has_changed = menu_utils.path_changed(session_state['selected_chatbot_path'],
                                                   session_state['selected_chatbot_path_serialized'])
    if path_has_changed:
        serialized_path = '/'.join(session_state['selected_chatbot_path'])
        session_state['selected_chatbot_path_serialized'] = serialized_path


# GENERAL MANAGEMENT #

def update_edited_prompt(chatbot):
    """
    Updates the edited prompt in the session state to reflect changes made by the user.

    Parameters:
    - chatbot: The chatbot for which the prompt will be updated.
    """
    session_state['edited_prompts'][chatbot] = session_state['edited_prompt']


def restore_prompt(chatbot):
    """
    Restores the original chatbot prompt by removing any user-made edits from the session state.

    Parameters:
    - chatbot: The chatbot for which the prompt will be restored.
    """
    if chatbot in session_state['edited_prompts']:
        del session_state['edited_prompts'][chatbot]


@st.cache_data
def fetch_chatbot_prompt(chatbot):
    """
    Fetches the prompt for the current chatbot based on the selected path.

    Parameters:
    - chatbot: The chatbot for which the prompt will be fetched.

    Returns:
    - A string containing the prompt of the current chatbot.
    """
    return menu_utils.get_final_prompt(chatbot,
                                       session_state[f"prompt_options_{session_state['typ']}"])


def get_prompt_to_use(chatbot, description):
    """
    Retrieves and returns the current prompt description for the chatbot, considering any user edits.

    Parameters:
    - chatbot: The chatbot for which the prompt will be returned.
    - default_description: The default description provided for the chatbot's prompt.

    Returns:
    - The current (possibly edited) description to be used as the prompt.
    """
    return session_state['edited_prompts'].get(chatbot, description)


def update_conversation_history(chatbot, current_history):
    """
    Updates the session state with the current conversation history.

    This method ensures that the conversation history for the currently selected chatbot path in the session state
    is updated to reflect any new messages or responses that have been added during the interaction.

    Parameters:
    - chatbot: The chatbot for which the conversation will be updated.
    - current_history: The current conversation history, which is a list of tuples. Each tuple contains
      the speaker ('USER' or 'Assistant'), the message, and optionally, the system description or prompt
      accompanying user messages (for user messages only).
    """
    session_state['conversation_histories'][chatbot] = current_history


def add_conversation_entry(chatbot, speaker, message, prompt=""):
    session_state['conversation_histories'][chatbot].append((speaker, message, prompt))
    return session_state['conversation_histories'][chatbot]


def get_conversation_history(chatbot):
    return session_state['conversation_histories'].get(chatbot, [])


def has_conversation_history(chatbot):
    if chatbot == 'docs':
        if (chatbot in session_state['conversation_histories']
                and session_state['uploaded_pdf_files']):
            return True
    else:
        if (chatbot in session_state['conversation_histories']
                and session_state['conversation_histories'][chatbot]):
            return True
    return False


def is_new_message(current_history, user_message):
    """
    Checks if the last message in history differs from the newly submitted user message.

    Parameters:
    - current_history: The current conversation history.
    - user_message: The newly submitted user message.

    Returns:
    - A boolean indicating whether this is a new unique message submission.
    """
    if (not current_history or current_history[-1][0] != session_state['USER']
            or current_history[-1][1] != user_message):
        return True
    return False


def create_history(chatbot):
    # If no conversation in history for this chatbot, then create an empty one
    session_state['conversation_histories'][chatbot] = [
    ]


def generate_chat_history_tuples(conversation):
    # Generate tuples with the chat history
    # We iterate with a step of 2 and always take
    # the current and next item assuming USER follows by Assistant
    if conversation:
        chat_history_tuples = [
            (conversation[i][1], conversation[i + 1][1])
            # Extract just the messages, ignoring the labels
            for i in range(0, len(conversation) - 1, 2)
            # We use len(conversation)-1 to avoid out-of-range errors
        ]
        return chat_history_tuples
    return []


def process_ai_response_for_suggestion_queries(model_output):
    # Ensure model_output is a single string for processing
    model_output_string = ' '.join(model_output) if isinstance(model_output, list) else model_output

    # Remove timestamps
    clean_output = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3}', '', model_output_string)

    # Removing content inside square brackets including the brackets themselves
    clean_output = re.sub(r'\[.*?\]', '', clean_output)

    # Splitting based on '?' to capture questions, adding '?' back to each split segment
    queries = [query + '?' for query in re.split(r'\?+\s*', clean_output) if query.strip()]

    cleaned_queries = []
    for query in queries:
        # Removing anything that is not a letter at the start of each query
        query_clean = re.sub(r'^[^A-Za-z]+', '', query)  # Adjust regex as needed
        # Attempt to more robustly remove quotes around the question
        query_clean = re.sub(r'(^\s*"\s*)|(\s*"$)', '', query_clean)  # Improved to target quotes more effectively
        # Remove numerical patterns and excessive whitespace
        query_clean = re.sub(r'\s*\d+[.)>\-]+\s*$', '', query_clean).strip()
        # Replace multiple spaces with a single space
        query_clean = re.sub(r'\s+', ' ', query_clean)
        if query_clean:
            cleaned_queries.append(query_clean)

    # Sort queries by length (descending) to prefer longer queries and select the first three
    if len(cleaned_queries) > 3:
        cleaned_queries = sorted(cleaned_queries, key=len, reverse=True)[:3]
    elif len(cleaned_queries) < 3:
        cleaned_queries = []

    return cleaned_queries


def display_conversation(conversation_history, container=None):
    """
    Displays the conversation history between the user and the assistant within the given container or globally.

    Parameters:
    - conversation_history: A list containing tuples of (speaker, message, system prompt)
     representing the conversation. The system prompt is optional.
    - container: The Streamlit container (e.g., column, expander) where the messages should be displayed.
      If None, messages will be displayed in the main app area.
    """
    chat_message_container = container if container else st
    for entry in conversation_history:
        speaker, message, *__ = entry + (None,)  # Ensure at least 3 elements
        print(session_state['USER'])
        if speaker == session_state['USER']:
            chat_message_container.chat_message("user").write(message)
        elif speaker == "Assistant":
            chat_message_container.chat_message("assistant").write(message)


def change_chatbot_style():
    # Set style of chat input so that it shows up at the bottom of the column
    chat_input_style = f"""
    <style>
        .stChatInput {{
          position: fixed;
          bottom: 3rem;
        }}
    </style>
    """
    st.markdown(chat_input_style, unsafe_allow_html=True)
