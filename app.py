import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

import menu_options

load_dotenv()

client = OpenAI()

USER = 'Renato'

# Initialize or update chatbot expertise and conversation in session state
if 'selected_path' not in st.session_state:
    st.session_state.selected_path = []


# Display the logo on the sidebar
st.sidebar.image("img/logo.svg", width=100)  # Adjust width as desired, ensures logo is at the sidebar's top


def display_sidebar_menu(options, path=[]):
    if isinstance(options, dict) and options:  # Verify options is a dictionary and not empty
        next_level = list(options.keys())
        choice = st.sidebar.selectbox("Choose an option:", [""] + next_level, index=0,
                                      format_func=lambda x: x if x else "-")
        if choice:
            new_path = path + [choice]
            st.session_state.selected_path = new_path
            display_sidebar_menu(options.get(choice, {}), new_path)


display_sidebar_menu(menu_options.options)


def get_openai_response(prompt_text, selected_path_description):
    """
    Sends a prompt to the OpenAI API including the history of the conversation
    adjusted for the selected description and returns the API's response.

    The function uses the description from the dialog flow (as determined by the user's menu
    navigation and the final selected_path_description).

    :param prompt_text: The message text submitted by the user.
    :param selected_path_description: The system message (description) for the chosen expertise area.
    :return: The OpenAI API's response.
    """
    # Building the messages with the "system" message based on expertise area
    messages = [{
        "role": "system",
        "content": selected_path_description
    }]

    # Add the history of the conversation
    for speaker, message in st.session_state['conversation']:
        role = 'user' if speaker == USER else 'assistant'
        messages.append({"role": role, "content": message})

    # Add the current user message
    messages.append({"role": "user", "content": prompt_text})

    # Send the request to the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    ai_message = response.choices[0].message.content

    return ai_message


selected_path = st.session_state.selected_path
if selected_path:
    st.sidebar.write("Selected Path: " + " > ".join(selected_path))

# Checking if the last selected option corresponds to a chatbot conversation
if selected_path and isinstance(menu_options.get_final_description(selected_path), str):
    expertise_area = selected_path[-1]  # This will give "Programming" or "Finances"
    description = menu_options.get_final_description(selected_path)

    # Add the logo at the top of the app
    st.image("img/logo.svg", width=100)  # Adjust width as desired

    # Interface to chat with selected expert
    st.title(f"Chat with {expertise_area}")

    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []

    user_message = st.chat_input(USER)

    if user_message:
        # Prevent re-running the query on refresh or navigating back to the page
        if not st.session_state['conversation'] or st.session_state['conversation'][-1] != (USER, user_message):
            # Add user message to the conversation
            st.session_state['conversation'].append((USER, user_message))

            # Get response from OpenAI API
            ai_response = get_openai_response(user_message, description)

            # Add AI response to the conversation
            st.session_state['conversation'].append(('Assistant', ai_response))

    # Display conversation
    for speaker, message in st.session_state['conversation']:
        if speaker == USER:
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)
