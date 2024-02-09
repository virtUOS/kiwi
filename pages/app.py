import os

from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
from streamlit_option_menu import option_menu

from src import menu_options

USER = 'User'

load_dotenv()
client = OpenAI()
st.set_page_config(layout="wide")


def load_prompts_neutral():
    # Uncheck the other check first
    if st.session_state["use_neutral_prompts"]:
        st.session_state["use_witty_prompts"] = False
    load_prompts()


def load_prompts_witty():
    # Uncheck the other check first
    if st.session_state["use_witty_prompts"]:
        st.session_state["use_neutral_prompts"] = False
    load_prompts()


def load_prompts():
    # Only if the checkboxes were already rendered we want to send their values
    if "use_neutral_prompts" in st.session_state:
        st.session_state["prompt_options"] = menu_options.load_prompts_from_yaml(
            st.session_state["use_neutral_prompts"],
            st.session_state["use_witty_prompts"])
    else:
        st.session_state["prompt_options"] = menu_options.load_prompts_from_yaml()


def initialize_session_variables():
    # Initialize or update chatbot expertise and conversation in session state
    if 'selected_path' not in st.session_state:
        st.session_state['selected_path'] = []

    if 'conversation_histories' not in st.session_state:
        st.session_state['conversation_histories'] = {}

    if 'selected_path_serialized' not in st.session_state:
        st.session_state['selected_path_serialized'] = ""

    # The default are basic prompts
    if 'prompt_options' not in st.session_state:
        st.session_state['prompt_options'] = menu_options.load_prompts_from_yaml()


initialize_session_variables()

# Display the logo on the sidebar
with st.sidebar:
    # Create three columns
    col1, col2, col3 = st.columns([1, 2, 1])

    # Use the middle column to display the logo
    with col2:
        st.image("img/logo.svg", width=100)


def display_sidebar_menu(options, path=[]):
    if isinstance(options, dict) and options:  # Verify options is a dictionary and not empty
        next_level = list(options.keys())

        with st.sidebar:
            choice = option_menu("Chat Menu", next_level,
                                 icons=['chat-dots', 'chat-dots', 'chat-dots',
                                        'chat-dots', 'chat-dots', 'chat-dots'],
                                 menu_icon="cast", default_index=0)

        if choice:
            new_path = path + [choice]
            st.session_state.selected_path = new_path
            display_sidebar_menu(options.get(choice, {}), new_path)


# Load prompts
load_prompts()

display_sidebar_menu(st.session_state["prompt_options"])

selected_path = st.session_state.selected_path

# Add a toggle to select between neutral and witty prompts
with st.sidebar:
    st.write("Selected Chatbot: " + " > ".join(selected_path))
    col4, col5 = st.columns([1, 1])
    col4.checkbox('Use our predefined neutral chatbots',
                  value=False,
                  help="We predefined prompts for different "
                       "chatbots that you might find useful "
                       "or fun to engage with.",
                  on_change=load_prompts_neutral,
                  key="use_neutral_prompts")
    col5.checkbox('Use our predefined witty chatbots',
                  value=False,
                  help="These prompts give the bots some "
                       "distinct personalities compared "
                       "to the default ones which are "
                       "fairly neutral. Take it as a way "
                       "to experiment with different user "
                       "experiences.",
                  on_change=load_prompts_witty,
                  key="use_witty_prompts")


def get_openai_response(prompt_text, description_to_use):
    """
    Sends a prompt to the OpenAI API including the history of the conversation
    adjusted for the selected description and returns the API's response.

    The function uses the description from the dialog flow (as determined by the user's menu
    navigation and the final selected_path_description).

    :param prompt_text: The message text submitted by the user.
    :param description_to_use: The system message (description) for the chosen expertise area.
    :return: The OpenAI API's response.
    """
    # Building the messages with the "system" message based on expertise area
    messages = [{
        "role": "system",
        "content": description_to_use
    }]

    # Add the history of the conversation
    for speaker, message in st.session_state['conversation_histories'][st.session_state['selected_path_serialized']]:
        role = 'user' if speaker == USER else 'assistant'
        messages.append({"role": role, "content": message})

    # Add the current user message
    messages.append({"role": "user", "content": prompt_text})

    # Send the request to the OpenAI API
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=os.environ['OPENAI_MODEL'],
            messages=messages,
            stream=True,
        )
        response = st.write_stream(stream)

    return response


if selected_path:

    # Store the serialized path in session state if different from current path
    if menu_options.path_changed(selected_path, st.session_state['selected_path_serialized']):
        st.session_state['selected_path_serialized'] = '/'.join(
            selected_path)  # Update the serialized path in session state

    # Checking if the last selected option corresponds to a chatbot conversation
    if isinstance(menu_options.get_final_description(selected_path, st.session_state["prompt_options"]), str):
        expertise_area = selected_path[-1]
        description = menu_options.get_final_description(selected_path, st.session_state["prompt_options"])

        # Interface to chat with selected expert
        st.title(f"Chat with {expertise_area} :robot_face:")

        # Allow the user to edit the default prompt for the chatbot
        with st.expander("Edit Bot Prompt", expanded=False):
            st.session_state['edited_description'] = st.text_area("System Prompt", value=description,
                                                                  help="Edit the system prompt "
                                                                       "for more customized responses.")
        # Use the edited description if available, otherwise use the original one
        description_to_use = st.session_state.get('edited_description', description)

        with st.sidebar:
            if st.button("Clear Conversation"):
                # Clears the current chatbot's conversation history
                st.session_state['conversation_histories'][st.session_state['selected_path_serialized']] = [
                ]

        # If there's already a conversation in the history for this chatbot, display it.
        if st.session_state['selected_path_serialized'] in st.session_state['conversation_histories']:
            # Display conversation
            for speaker, message in (
                    st.session_state['conversation_histories'][st.session_state['selected_path_serialized']]):
                if speaker == USER:
                    st.chat_message("user").write(message)
                else:
                    st.chat_message("assistant").write(message)
        else:
            # If no conversation in history for this chatbot, then create an empty one
            st.session_state['conversation_histories'][st.session_state['selected_path_serialized']] = [
            ]

        # Accept user input
        if user_message := st.chat_input(USER):

            # Prevent re-running the query on refresh or navigating back to the page by checking
            # if the last stored message is not from the User
            if (not st.session_state['conversation_histories'][st.session_state['selected_path_serialized']] or
                    st.session_state['conversation_histories'][st.session_state['selected_path_serialized']][-1]
                    != (USER, user_message)):
                # Add user message to the conversation
                st.session_state['conversation_histories'][st.session_state
                ['selected_path_serialized']].append((USER, user_message))

                # Print user message immediately after getting entered because we're streaming the chatbot output
                with st.chat_message("user"):
                    st.markdown(user_message)

                # Get response from OpenAI API
                ai_response = get_openai_response(user_message, description_to_use)

                # Add AI response to the conversation
                st.session_state['conversation_histories'][st.session_state
                ['selected_path_serialized']].append(('Assistant', ai_response))
