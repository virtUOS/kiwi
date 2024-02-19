# Import libraries
import os
import pandas as pd
import streamlit as st
from streamlit import session_state as ss
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from streamlit_cookies_manager import EncryptedCookieManager
from src import menu_utils
from openai import OpenAI

# Constants
USER = 'User'

# Load environment variables
load_dotenv()

# Initialize OpenAI client and Streamlit page config
client = OpenAI()
st.set_page_config(page_title="My UOS Chatbot Community", layout="wide", page_icon="🥝")


# Session and cookies management
def initialize_cookies():
    """Initialize cookies for session management."""
    cookies = EncryptedCookieManager(
        prefix="ktosiek/streamlit-cookies-manager/",
        password=os.environ.get("COOKIES_PASSWORD"),
    )

    if not cookies.ready():
        st.spinner()
        st.stop()

    return cookies


def verify_session(cookies):
    """Verify if a user session is already started; if not, redirect to start page."""
    if not cookies.get("session"):
        st.switch_page("start.py")


def initialize_session_variables():
    """Initialize essential session variables."""
    required_keys = {
        'selected_path': [],
        'conversation_histories': {},
        'selected_path_serialized': "",
        'prompt_options': menu_utils.load_prompts_from_yaml(),
        'edited_prompts': {},
        'disable_custom': False,
        'open-ai-model': os.environ['OPENAI_MODEL'],
    }

    for key, default_value in required_keys.items():
        if key not in ss:
            ss[key] = default_value


def load_personal_prompts_file():
    if ss["prompts_file"]:
        # Disable and deselect custom prompts
        ss["disable_custom"] = True
        ss["use_custom_prompts"] = False
        # Reinitialize the edited prompts
        ss["edited_prompts"] = {}
        # Set the prompt options to the uploaded ones
        ss["prompt_options"] = menu_utils.load_dict_from_yaml(ss["prompts_file"])
    else:
        # Enable the custom prompts checkbox in case it was disabled
        ss["disable_custom"] = False


def load_prompts():
    """Load chat prompts based on user selection or file upload."""
    if "prompts_file" in ss and ss["prompts_file"]:
        pass
    # Only if the checkbox was already rendered we want to load them
    elif "use_custom_prompts" in ss:
        ss["prompt_options"] = menu_utils.load_prompts_from_yaml(
            ss["use_custom_prompts"])
    else:
        ss["prompt_options"] = menu_utils.load_prompts_from_yaml()


# Sidebar and Page Setup
def display_logo():
    """Display the logo on the sidebar."""
    with st.sidebar:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("img/logo.svg", width=100)


def display_sidebar_menu(options, path=[]):
    """Display sidebar menu for chatbot selection."""
    if isinstance(options, dict) and options:  # Verify options is a dictionary and not empty
        next_level = list(options.keys())

        with st.sidebar:
            choice = option_menu("Chat Menu", next_level,
                                 icons=['chat-dots'] * len(next_level),
                                 menu_icon="cast",
                                 default_index=0)

        if choice:
            new_path = path + [choice]
            ss['selected_path'] = new_path
            display_sidebar_menu(options.get(choice, {}), new_path)


def display_sidebar_controls():
    """Display controls and settings in the sidebar."""

    # Store the serialized path in session state if different from current path
    # Do this here to handle conversation controls showing up when changing
    # chatbot and the conversation exists
    if menu_utils.path_changed(ss['selected_path'], ss['selected_path_serialized']):
        ss['selected_path_serialized'] = '/'.join(
            ss['selected_path'])  # Update the serialized path in session state

    # Custom prompt selection
    with st.sidebar:
        st.write("Selected Chatbot: " + " > ".join(ss['selected_path']))
        st.checkbox('Use predefined chatbots',
                    value=False,
                    help="We predefined prompts for different "
                         "chatbots that you might find useful "
                         "or fun to engage with.",
                    on_change=load_prompts,
                    key="use_custom_prompts",
                    disabled=ss["disable_custom"])

        st.markdown("""---""")

        with st.expander("**Personalized Chatbots Controls**", expanded=False):
            # Here we prepare the prompts for download by merging predefined prompts with edited prompts
            prompts_for_download = ss["prompt_options"].copy()  # Copy the predefined prompts
            # Update the copied prompts with any changes made by the user
            for path, edited_prompt in ss['edited_prompts'].items():
                menu_utils.set_prompt_for_path(prompts_for_download, path.split('/'), edited_prompt)

            # Convert the merged prompts into a YAML string for download
            merged_prompts_yaml = menu_utils.dict_to_yaml(prompts_for_download)

            st.download_button("Download YAML file with custom prompts",
                               data=merged_prompts_yaml,
                               file_name="prompts.yaml",
                               mime="application/x-yaml")
            st.file_uploader("Upload YAML file with custom prompts",
                             type='yaml',
                             key="prompts_file",
                             on_change=load_personal_prompts_file)

        # Show the conversation controls only if there's a conversation
        if ss['selected_path_serialized'] in ss['conversation_histories'] and ss[
                'conversation_histories'][ss['selected_path_serialized']]:

            st.markdown("""---""")

            st.write("**Conversation Controls**")
            col1, col2 = st.columns([1, 5])
            if col1.button("🗑️", help="Delete the Current Conversation"):
                # Clears the current chatbot's conversation history
                ss['conversation_histories'][ss['selected_path_serialized']] = [
                ]
                st.rerun()

            conversation_to_download = ss['conversation_histories'][
                ss['selected_path_serialized']]

            conversation_df = pd.DataFrame(conversation_to_download, columns=['Speaker', 'Message'])
            conversation_csv = conversation_df.to_csv(index=False).encode('utf-8')
            col2.download_button("💽",
                                 data=conversation_csv,
                                 file_name="conversation.csv",
                                 mime="text/csv",
                                 help="Download the Current Conversation")


def get_openai_response(prompt_text, system_prompt):
    """
    Sends a prompt to the OpenAI API including the history of the conversation
    adjusted for the selected description and returns the API's response.

    The function uses the description from the dialog flow (as determined by the user's menu
    navigation and the final selected_path_description).

    :param prompt_text: The message text submitted by the user.
    :param system_prompt: The system message (description) for the chosen expertise area.
    :return: The OpenAI API's response.
    """
    # Building the messages with the "system" message based on expertise area
    messages = [{
        "role": "system",
        "content": system_prompt
    }]

    # Add the history of the conversation
    for speaker, message in ss['conversation_histories'][ss['selected_path_serialized']]:
        role = 'user' if speaker == USER else 'assistant'
        messages.append({"role": role, "content": message})

    # Add the current user message
    messages.append({"role": "user", "content": prompt_text})

    # Send the request to the OpenAI API
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=ss['open-ai-model'],
            messages=messages,
            stream=True,
        )
        response = st.write_stream(stream)

    return response


def manage_user_input(description_to_use):
    """Handles user input and gets response from OpenAI."""
    if user_message := st.chat_input(USER):
        # Ensure there's a key for this conversation in the history
        if ss['selected_path_serialized'] not in ss['conversation_histories']:
            ss['conversation_histories'][ss['selected_path_serialized']] = []

        # Get the current conversation history
        current_history = ss['conversation_histories'][ss['selected_path_serialized']]

        # Prevent re-running the query on refresh or navigating back by checking
        # if the last stored message is not from the User or the history is empty
        if not current_history or current_history[-1][0] != USER or current_history[-1][1] != user_message:
            # Add user message to the conversation
            current_history.append((USER, user_message))

            # Print user message immediately after getting entered because we're streaming the chatbot output
            with st.chat_message("user"):
                st.markdown(user_message)

            # Get response from OpenAI API
            ai_response = get_openai_response(user_message, description_to_use)

            # Add AI response to the conversation
            current_history.append(('Assistant', ai_response))

            # Update the conversation history in the session state
            ss['conversation_histories'][ss['selected_path_serialized']] = current_history

            st.rerun()


def display_chat_interface():
    """Displays the chat interface and handles conversation display."""
    selected_path = ss["selected_path"]
    if selected_path:

        # Checking if the last selected option corresponds to a chatbot conversation
        if isinstance(menu_utils.get_final_description(selected_path, ss["prompt_options"]), str):
            expertise_area = selected_path[-1]
            description = menu_utils.get_final_description(selected_path, ss["prompt_options"])

            # Interface to chat with selected expert
            st.title(f"Chat with {expertise_area} :robot_face:")

            # Allow the user to edit the default prompt for the chatbot
            with st.expander("Edit Bot Prompt", expanded=False):
                # Check if there's an edited prompt for the current path
                current_path = "/".join(ss['selected_path'])
                current_edited_prompt = ss['edited_prompts'].get(current_path, description)

                edited_prompt = st.text_area("System Prompt",
                                             value=current_edited_prompt,
                                             help="Edit the system prompt for more customized responses.")

                # Update the session state with the edited prompt
                ss['edited_prompts'][current_path] = edited_prompt

                # Button to restore the original prompt
                if st.button("🔄", help="Restore Original Prompt"):
                    if current_path in ss['edited_prompts']:
                        del ss['edited_prompts'][current_path]  # Remove the edited prompt
                        st.rerun()  # Rerun the script to reflect the changes

            if current_path in ss['edited_prompts']:
                description_to_use = ss['edited_prompts'][current_path]
            else:
                description_to_use = description

            # If there's already a conversation in the history for this chatbot, display it.
            if ss['selected_path_serialized'] in ss['conversation_histories']:
                # Display conversation
                for speaker, message in (
                        ss['conversation_histories'][ss['selected_path_serialized']]):
                    if speaker == USER:
                        st.chat_message("user").write(message)
                    else:
                        st.chat_message("assistant").write(message)

            manage_user_input(description_to_use)


# Main
def main():
    cookies = initialize_cookies()
    verify_session(cookies)
    initialize_session_variables()

    display_logo()
    load_prompts()
    display_sidebar_menu(ss["prompt_options"])
    display_sidebar_controls()

    display_chat_interface()


if __name__ == "__main__":
    main()
