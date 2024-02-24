import os
import pandas as pd
from openai import OpenAI
import streamlit as st
from streamlit import session_state as ss
from streamlit_option_menu import option_menu
from streamlit_cookies_manager import EncryptedCookieManager
from src import menu_utils
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SidebarManager:

    def __init__(self):
        self.cookies = self.initialize_cookies()

    @staticmethod
    def initialize_cookies():
        """Initialize cookies for session management."""
        cookies = EncryptedCookieManager(
            prefix=os.getenv("COOKIES_PREFIX"),
            password=os.getenv("COOKIES_PASSWORD")
        )

        if not cookies.ready():
            st.spinner()
            st.stop()

        return cookies

    def verify_user_session(self):
        """Verify if a user session is already started; if not, redirect to start page."""
        if self.cookies.get("session") != 'in':
            st.switch_page("start.py")

    @staticmethod
    def initialize_session_variables():
        """Initialize essential session variables."""
        required_keys = {
            'model_selection': "OpenAI",
            'selected_chatbot_path': [],
            'conversation_histories': {},
            'selected_chatbot_path_serialized': "",
            'prompt_options': menu_utils.load_prompts_from_yaml(),
            'edited_prompts': {},
            'disable_custom': False,
        }

        for key, default_value in required_keys.items():
            if key not in ss:
                ss[key] = default_value

    @staticmethod
    def display_logo():
        """Display the logo on the sidebar."""
        with st.sidebar:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("img/logo.svg", width=100)

                # Gets rid of full screen option for logo image
                hide_img_fs = '''
                <style>
                button[title="View fullscreen"]{
                    visibility: hidden;}
                </style>
                '''

                st.markdown(hide_img_fs, unsafe_allow_html=True)

    @staticmethod
    def load_prompts():
        language = st.query_params.get('lang', False)
        # Use German language as default
        if not language:
            language = "de"

        """Load chat prompts based on language."""
        ss["prompt_options"] = menu_utils.load_prompts_from_yaml(language=language)

    def _display_chatbots_menu(self, options, path=[]):
        """Display sidebar menu for chatbot selection."""
        if isinstance(options, dict) and options:  # Verify options is a dictionary and not empty
            next_level = list(options.keys())

            with st.sidebar:
                choice = option_menu(ss['_']("Chat Menu"), next_level,
                                     icons=['chat-dots'] * len(next_level),
                                     menu_icon="cast",
                                     default_index=0)

            if choice:
                new_path = path + [choice]
                ss['selected_chatbot_path'] = new_path
                self._display_chatbots_menu(options.get(choice, {}), new_path)

    def display_sidebar_controls(self):
        """Display controls and settings in the sidebar."""

        # First load prompts
        self.load_prompts()

        # Display the menu for chatbots
        self._display_chatbots_menu(ss['prompt_options'])

        # Store the serialized path in session state if different from current path
        # Do this here to handle conversation controls showing up when changing
        # chatbot and the conversation exists
        if menu_utils.path_changed(ss['selected_chatbot_path'], ss['selected_chatbot_path_serialized']):
            ss['selected_chatbot_path_serialized'] = '/'.join(
                ss['selected_chatbot_path'])  # Update the serialized path in session state

        # Custom prompt selection
        with st.sidebar:
            if ss['model_selection'] == 'OpenAI':
                model_text = ss['_']("Model:")
                st.write(f"{model_text} {os.getenv('OPENAI_MODEL')}")

            # Show the conversation controls only if there's a conversation
            if ss['selected_chatbot_path_serialized'] in ss['conversation_histories'] and ss[
                    'conversation_histories'][ss['selected_chatbot_path_serialized']]:

                st.markdown("""---""")

                st.write(ss['_']("**Conversation Controls**"))
                col1, col2 = st.columns([1, 5])
                if col1.button("🗑️", help=ss['_']("Delete the Current Conversation")):
                    # Clears the current chatbot's conversation history
                    ss['conversation_histories'][ss['selected_chatbot_path_serialized']] = [
                    ]
                    st.rerun()

                conversation_to_download = ss['conversation_histories'][
                    ss['selected_chatbot_path_serialized']]

                conversation_df = pd.DataFrame(conversation_to_download,
                                               columns=[ss['_']('Speaker'), ss['_']('Message')])
                conversation_csv = conversation_df.to_csv(index=False).encode('utf-8')
                col2.download_button("💽",
                                     data=conversation_csv,
                                     file_name="conversation.csv",
                                     mime="text/csv",
                                     help=ss['_']("Download the Current Conversation"))

            color = st.get_option('theme.secondaryBackgroundColor')

            css = f'''
            [data-testid="stSidebarNav"] {{
                position:absolute;
                bottom: 0;
                z-index: 1;
                background: {color};
            }}
            [data-testid="stSidebarNav"] > ul {{
                padding-top: 2rem;
            }}
            [data-testid="stSidebarNav"] > div {{
                position:absolute;
                top: 0;
            }}
            [data-testid="stSidebarNav"] > div > svg {{
                transform: rotate(180deg) !important;
            }}
            [data-testid="stSidebarNav"] + div {{
                overflow: scroll;
                max-height: 66vh;
            }}
            '''

            st.markdown("""---""")

            if st.button('Logout'):
                self.cookies["session"] = 'out'

                st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
                st.switch_page('start.py')


class ChatManager:

    def __init__(self, user):
        self.client = None
        ss['USER'] = user

    def set_client(self, client):
        self.client = client

    @staticmethod
    def update_edited_prompt():
        ss['edited_prompts'][ss['selected_chatbot_path_serialized']] = ss['edited_prompt']


    def _display_prompt_editor(self, description):
        """Allows editing of the chatbot prompt."""
        current_chatbot_path_serialized = ss['selected_chatbot_path_serialized']
        current_edited_prompt = ss['edited_prompts'].get(current_chatbot_path_serialized, description)

        st.text_area(ss['_']("Edit Prompt"),
                     value=current_edited_prompt,
                     help=ss['_']("Edit the system prompt for more customized responses."),
                     on_change=self.update_edited_prompt,
                     key='edited_prompt')

        if st.button("🔄", help=ss['_']("Restore Original Prompt")):
            if current_chatbot_path_serialized in ss['edited_prompts']:
                del ss['edited_prompts'][current_chatbot_path_serialized]
                st.rerun()

    @staticmethod
    def _get_description_to_use(default_description):
        """Retrieves the prompt description to use, whether edited or default."""
        return ss['edited_prompts'].get(ss['selected_chatbot_path_serialized'], default_description)

    @staticmethod
    def _display_conversation():
        """Displays the conversation history for the selected path."""
        conversation_history = ss['conversation_histories'].get(ss['selected_chatbot_path_serialized'], [])

        if conversation_history:
            for speaker, message in conversation_history:
                if speaker == ss['USER']:
                    st.chat_message("user").write(message)
                else:
                    st.chat_message("assistant").write(message)

    def _handle_user_input(self, description_to_use):
        """Handles user input, sending it to OpenAI and displaying the response."""
        if user_message := st.chat_input(ss['USER']):
            # Ensure there's a key for this conversation in the history
            if ss['selected_chatbot_path_serialized'] not in ss['conversation_histories']:
                ss['conversation_histories'][ss['selected_chatbot_path_serialized']] = []

            # Get the current conversation history
            current_history = ss['conversation_histories'][ss['selected_chatbot_path_serialized']]

            # Prevent re-running the query on refresh or navigating back by checking
            # if the last stored message is not from the User or the history is empty
            if not current_history or current_history[-1][0] != ss['USER'] or current_history[-1][1] != user_message:
                # Add user message to the conversation
                current_history.append((ss['USER'], user_message))

                # Print user message immediately after getting entered because we're streaming the chatbot output
                with st.chat_message("user"):
                    st.markdown(user_message)

                response = self.client.get_response(user_message, description_to_use)

                # Add AI response to the conversation
                current_history.append(('Assistant', response))

                # Update the conversation history in the session state
                ss['conversation_histories'][ss['selected_chatbot_path_serialized']] = current_history

                st.rerun()

    def display_chat_interface(self):
        """Displays the chat interface and manages conversation display and input."""
        if 'selected_chatbot_path' in ss and ss["selected_chatbot_path"]:
            selected_chatbot_path = ss["selected_chatbot_path"]

            if isinstance(menu_utils.get_final_description(selected_chatbot_path, ss["prompt_options"]), str):
                expertise_area = selected_chatbot_path[-1]
                description = menu_utils.get_final_description(selected_chatbot_path, ss["prompt_options"])

                # Display title and prompt editing interface
                chat_text_msg = ss['_']("Chat with")
                st.title(f"{chat_text_msg} {expertise_area} 🤖")

                with st.expander(ss['_']("Edit Bot Prompt"), expanded=False):
                    self._display_prompt_editor(description)

                description_to_use = self._get_description_to_use(description)

                self._display_conversation()
                self._handle_user_input(description_to_use)


class AIClient:

    def __init__(self):
        if ss['model_selection'] == 'OpenAI':

            @st.cache_resource
            def load_openai_data():
                return OpenAI(), os.getenv('OPENAI_MODEL')

            self.client, self.model = load_openai_data()
        else:
            pass
            # Add here client initialization for local model
            # For example using OpenLLM as inference server
            # ss["inference_server_url"] = "http://localhost:3000/v1"
            # self.client = OpenAI(base_url=ss['inference_server_url'])
            # models = ss["client"].models.list()
            # self.model = models.data[0].id

    def get_response(self, prompt, description_to_use):
        """
        Sends a prompt to the OpenAI API and returns the API's response.

        Parameters:
            prompt (str): The user's message or question.
            description_to_use (str): Additional context or instructions to provide to the model.

        Returns:
            str: The response from the chatbot.
        """
        try:
            # Prepare the full prompt and messages with context or instructions
            messages = self._prepare_full_prompt_and_messages(prompt, description_to_use)

            # Send the request to the OpenAI API
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                )
                response = st.write_stream(stream)

            return response

        except Exception as e:
            print(f"An error occurred while fetching the OpenAI response: {e}")
            # Optionally, return a default error message or handle the error appropriately.
            return "Sorry, I couldn't process that request."

    @staticmethod
    def _prepare_full_prompt_and_messages(user_prompt, description_to_use):
        """
        Prepares the full prompt and messages combining user input and additional descriptions.

        Parameters:
            user_prompt (str): The original prompt from the user.
            description_to_use (str): Additional context or instructions to provide to the model.

        Returns:
            list: List of dictionaries containing messages for the chat completion.
        """

        if ss["model_selection"] == 'OpenAI':
            messages = [{
                'role': "system",
                'content': description_to_use
            }]
        else:  # Add here conditions for different types of models
            messages = []

        # Add the history of the conversation
        for speaker, message in ss['conversation_histories'][ss['selected_chatbot_path_serialized']]:
            role = 'user' if speaker == ss['USER'] else 'assistant'
            messages.append({'role': role, 'content': message})

        # Building the messages with the "system" message based on expertise area
        if ss["model_selection"] == 'OpenAI':
            messages.append({"role": "user", "content": user_prompt})
        else:
            # Add the description to the current user message to simulate a system prompt.
            # This is for models that were trained with no system prompt: LLaMA/Mistral/Mixtral.
            # Source: https://github.com/huggingface/transformers/issues/28366
            # Maybe find a better solution or avoid any system prompt for these models
            messages.append({"role": "user", "content": description_to_use + "\n\n" + user_prompt})

        # This method can be expanded based on how you want to structure the prompt
        # For example, you might prepend the description_to_use before the user_prompt
        return messages
