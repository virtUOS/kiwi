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
            prefix="virtuos/ai-portal",
            password=os.getenv("COOKIES_PASSWORD")
        )

        if not cookies.ready():
            st.spinner()
            st.stop()

        return cookies

    def verify_user_session(self):
        """Verify if a user session is already started; if not, redirect to start page."""
        if not self.cookies.get("session"):
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

    @staticmethod
    def _load_personal_prompts_file():
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

    @staticmethod
    def _load_prompts():
        """Load chat prompts based on user selection or file upload."""
        if "prompts_file" in ss and ss["prompts_file"]:
            pass
        # Only if the checkbox was already rendered we want to load them
        elif "use_custom_prompts" in ss:
            ss["prompt_options"] = menu_utils.load_prompts_from_yaml(
                ss["use_custom_prompts"], language='en')
        else:
            ss["prompt_options"] = menu_utils.load_prompts_from_yaml()

    def _display_chatbots_menu(self, options, path=[]):
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
                ss['selected_chatbot_path'] = new_path
                self._display_chatbots_menu(options.get(choice, {}), new_path)

    def display_sidebar_controls(self):
        """Display controls and settings in the sidebar."""

        # First load prompts
        self._load_prompts()

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
            st.write("Selected Chatbot: " + " > ".join(ss['selected_chatbot_path']))
            st.checkbox('Use predefined chatbots',
                        value=False,
                        help="We predefined prompts for different "
                             "chatbots that you might find useful "
                             "or fun to engage with.",
                        on_change=self._load_prompts,
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
                                 on_change=self._load_personal_prompts_file)

            # Show the conversation controls only if there's a conversation
            if ss['selected_chatbot_path_serialized'] in ss['conversation_histories'] and ss[
                    'conversation_histories'][ss['selected_chatbot_path_serialized']]:

                st.markdown("""---""")

                st.write("**Conversation Controls**")
                col1, col2 = st.columns([1, 5])
                if col1.button("🗑️", help="Delete the Current Conversation"):
                    # Clears the current chatbot's conversation history
                    ss['conversation_histories'][ss['selected_chatbot_path_serialized']] = [
                    ]
                    st.rerun()

                conversation_to_download = ss['conversation_histories'][
                    ss['selected_chatbot_path_serialized']]

                conversation_df = pd.DataFrame(conversation_to_download, columns=['Speaker', 'Message'])
                conversation_csv = conversation_df.to_csv(index=False).encode('utf-8')
                col2.download_button("💽",
                                     data=conversation_csv,
                                     file_name="conversation.csv",
                                     mime="text/csv",
                                     help="Download the Current Conversation")


class ChatManager:

    def __init__(self, user):
        self.client = None
        ss['USER'] = user

    def set_client(self, client):
        self.client = client

    @staticmethod
    def _display_prompt_editor(description):
        """Allows editing of the chatbot prompt."""
        current_chatbot_path_serialized = ss['selected_chatbot_path_serialized']
        current_edited_prompt = ss['edited_prompts'].get(current_chatbot_path_serialized, description)

        edited_prompt = st.text_area("Edit Prompt",
                                     value=current_edited_prompt,
                                     help="Edit the system prompt for more customized responses.")
        ss['edited_prompts'][current_chatbot_path_serialized] = edited_prompt

        if st.button("🔄", help="Restore Original Prompt"):
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
                st.title(f"Chat with {expertise_area} Expert")

                with st.expander("Edit Bot Prompt", expanded=False):
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
