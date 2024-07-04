import os
import time
import pandas as pd

import streamlit as st
from streamlit import session_state
from streamlit_cookies_manager import EncryptedCookieManager

from src.language_utils import initialize_language
from src import menu_utils
from src.ai_client import AIClient


class SidebarManager:

    def __init__(self, advanced_model):
        """
        Initialize the SidebarManager instance by setting up session cookies and initializing the language.
        """
        self.cookies = self.initialize_cookies()
        initialize_language()
        st.info(session_state['_']("The outputs of the chat assistant may be erroneous - therefore, "
                                   "always check the answers for their accuracy. Remember not to enter "
                                   "any personal information and copyrighted materials."))
        self.advanced_model = advanced_model

    @staticmethod
    def initialize_cookies():
        """
        Initialize cookies for managing user sessions with enhanced security.

        Uses an encrypted cookie manager and detects the Safari browser to handle specific JavaScript-based delays.
        This ensures a smooth user experience across different browsers.
        """
        cookies = EncryptedCookieManager(
            prefix=os.getenv("COOKIES_PREFIX"),
            password=os.getenv("COOKIES_PASSWORD")
        )

        if not cookies.ready():
            st.spinner()
            st.stop()

        return cookies

    def verify_user_session(self):
        """
        Verify the user's session validity. If cookies indicate the session is not yet initiated,
        then redirect the user to the start page.

        This function ensures that unauthorized users are not able to access application sections
        that require a valid session.
        """
        if self.cookies.get("session") != 'in':
            st.switch_page("start.py")

    def initialize_session_variables(self):
        """
        Initialize essential session variables with default values.

        Sets up the initial state for username, model selection, chatbot paths, conversation histories,
        prompt options, etc., essential for the application to function correctly from the start.
        """
        required_keys = {
            'username': self.cookies.get('username'),
            'model_selection': "OpenAI",
            'selected_chatbot_path': [],
            'conversation_histories': {},
            'selected_chatbot_path_serialized': "",
            'prompt_options': menu_utils.load_prompts_from_yaml(),
            'edited_prompts': {},
            'images_key': 0,
            'urls_key': -1,
            'image_urls': [],
            'uploaded_images': [],
            'current_uploaded_images': [],
            'current_image_urls': [],
            'image_content': [],
            'photo_to_use': [],
            'activate_camera': False,
            'toggle_camera_label': "Activate camera",
            'images_state': -1,  # -1 is for no images yet. 0 images uploaded, 1 user interacted with images
        }

        for key, default_value in required_keys.items():
            if key not in session_state:
                session_state[key] = default_value

    @staticmethod
    def reset_images_widgets():
        session_state['images_key'] += 1
        session_state['urls_key'] -= 1
        session_state['image_urls'] = []
        session_state['photo_to_use'] = []

    def clear_images_callback(self):
        session_state['uploaded_images'] = []
        session_state['current_uploaded_images'] = []
        session_state['current_image_urls'] = []
        session_state['current_photo_to_use'] = []
        session_state['images_state'] = -1
        self.reset_images_widgets()

    def _delete_conversation_callback(self):
        """
        Callback function to delete the current conversation history.

        This method clears the conversation history for the selected chatbot path stored in the session state.
        It sets the conversation history to an empty list.
        """
        session_state['conversation_histories'][session_state['selected_chatbot_path_serialized']] = []
        self.clear_images_callback()

    @staticmethod
    def display_logo():
        """
        Display the application logo in the sidebar.

        Utilizes Streamlit columns to center the logo
        """
        with st.sidebar:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("img/logo.svg", width=100)

    @staticmethod
    def load_prompts():
        """
        Load chat prompts based on the selected or default language.

        This method enables dynamic loading of chat prompts to support multi-lingual chatbot interactions.
        If no language is specified in the query parameters, German ('de') is used as the default language.
        """
        language = st.query_params.get('lang', False)
        # Use German language as default
        if not language:
            language = "de"

        # Load chat prompts based on language.
        session_state["prompt_options"] = menu_utils.load_prompts_from_yaml(language=language)

    def logout_and_redirect(self):
        """
        Perform logout operations and redirect the user to the start page.

        This involves resetting cookies and session variables that signify the user's logged-in state,
        thereby securely logging out the user.
        """
        self.cookies['session'] = 'out'
        session_state['password_correct'] = False
        session_state['credentials_checked'] = False
        st.switch_page('start.py')

    def _display_chatbots_menu(self, options, path=[]):
        """
        Recursively display sidebar menu for chatbot selection based on a nested dictionary structure.

        Allows users to navigate and select chatbots in a hierarchical manner, updating the session state
        to reflect the current selection path.

        :param options: A dictionary containing chatbot options and potential sub-options.
        :param path: A list representing the current selection path as the user navigates the menu.
        """
        if isinstance(options, dict) and options:  # Verify options is a dictionary and not empty
            next_level = list(options.keys())

            #with st.sidebar:
                #choice = option_menu("", next_level,
                #                     # icons=['chat-dots'] * len(next_level),
                #                     # menu_icon="cast",
                #                     default_index=0)

            # For now keep the choice always on the first level since we don't have more chatbots or features
            choice = next_level[0]

            if choice:
                new_path = path + [choice]
                session_state['selected_chatbot_path'] = new_path
                self._display_chatbots_menu(options.get(choice, {}), new_path)

    def display_sidebar_controls(self):
        """
        Display sidebar controls and settings for chatbot conversations.

        This function performs the following steps to provide an interactive and informative sidebar for the chatbot
         application:

        1. **Load and Display Prompts**:
           - Calls `_load_and_display_prompts` to load prompt options and display them in the chatbot's menu.

        2. **Update Path in Session State**:
           - Calls `_update_path_in_session_state` to check for changes in the selected chatbot path and update the
            session state accordingly.

        3. **Display Model Information**:
           - Calls `_display_model_information` to display model information for the selected model (e.g., OpenAI).

        4. **Style Language Uploader**:
           - Calls `_style_language_uploader` to customize the appearance of the file uploader component based on the
            selected language.

        5. **Show Conversation Controls**:
           - Uses `_show_conversation_controls` to provide options for the user to interact with the conversation
            history, including deleting and downloading conversations.

        6. **Show Images Controls** (Conditional):
           - If the selected model is the advanced model, calls `_show_images_controls` to provide options for managing
            images (uploading images, entering image URLs).

        7. **Add Custom CSS**:
           - Applies custom CSS by calling `_add_custom_css` to stylize the sidebar for better user experience.

        8. **Logout Option**:
           - Finally, calls `_logout_option` to offer a logout option to the user.
        """
        self._load_and_display_prompts()

        self._update_path_in_session_state()

        self._display_model_information()

        with st.sidebar:
            st.markdown("---")
            st.write(session_state['_']("**Options**"))

        self._show_conversation_controls()

        with st.sidebar:
            st.markdown("---")

        self._logout_option()

    def _load_and_display_prompts(self):
        """
        Load chat prompts from the session state and display them in the chatbots menu in the sidebar.

        This method first loads the prompts by calling the `load_prompts` method and
        then displays the chatbots menu with the loaded prompts using the `_display_chatbots_menu` method.
        """
        self.load_prompts()
        self._display_chatbots_menu(session_state['prompt_options'])

    @staticmethod
    def _update_path_in_session_state():
        """
        Check if the selected chatbot path in the session state has changed and update the serialized path accordingly.

        This static method compares the current selected chatbot path with the serialized chatbot path saved
         in the session.
        If there is a change, it updates the session state with the new serialized path, allowing for tracking
         the selection changes.
        """
        path_has_changed = menu_utils.path_changed(session_state['selected_chatbot_path'],
                                                   session_state['selected_chatbot_path_serialized'])
        if path_has_changed:
            serialized_path = '/'.join(session_state['selected_chatbot_path'])
            session_state['selected_chatbot_path_serialized'] = serialized_path

    def _display_model_information(self):
        """
        Display OpenAI model information in the sidebar if the OpenAI model is the current selection.

        In the sidebar, this static method shows the model name and version pulled from environment variables
        if 'OpenAI' is selected as the model in the session state. The model information helps users
        identify the active model configuration.
        """
        with st.sidebar:
            accessible_models = AIClient.get_accessible_models()
            fastchat_models = AIClient.get_fastchat_models()

            all_accessible_models = accessible_models + fastchat_models

            # Show dropdown when multiple models are available
            if all_accessible_models:
                model_label = session_state['_']("Model:")
                index = 0
                if self.advanced_model in all_accessible_models:  # Use most advanced model as default
                    index = accessible_models.index(self.advanced_model)
                st.selectbox(model_label,
                             all_accessible_models,
                             index=index,
                             key='selected_model')

                if session_state['selected_model'] in fastchat_models:
                    session_state['model_selection'] = "OTHER"
                else:
                    session_state['model_selection'] = "OpenAI"

                # If the model is changed to one that doesn't support images
                # we have to clear the widgets from images. For the upload widget,
                # the way to do this is by generating a new widget with a new key
                # as described in:
                # https://discuss.streamlit.io/t/
                # are-there-any-ways-to-clear-file-uploader-values-without-using-streamlit-form/40903
                if session_state['selected_model'] != self.advanced_model and (
                        session_state['image_urls'] or
                        session_state['uploaded_images'] or
                        session_state['photo_to_use']):
                    session_state['images_key'] += 1
                    session_state['image_urls'] = []
                    session_state['uploaded_images'] = []
                    session_state['image_content'] = []
                    session_state['photo_to_use'] = []
                    session_state['activate_camera'] = False
                    st.rerun()

    def _display_delete_conversation_button(self, container):
        """
        Render a button in the specified column that allows the user to delete the active chatbot's conversation history.

        On button click, this method removes the conversation history from the session state for the current chatbot path
        and triggers a page rerun to refresh the state.
        """
        delete_label = session_state['_']("Delete Conversation")
        container.button("🗑️", on_click=self._delete_conversation_callback, help=delete_label)

    def _show_conversation_controls(self):
        """
        Display buttons for conversation management, including deleting and downloading conversation history,
         in the sidebar.

        This method checks if there is an existing conversation history for the currently selected chatbot path and,
         if so,
        displays options to either delete this history or download it as a CSV file. It leverages the
        `_delete_conversation_button` and `_download_conversation_button` methods to render these options.
        """
        conversation_key = session_state['selected_chatbot_path_serialized']
        with st.sidebar:
            col1, col2, col3 = st.columns([1, 1, 1])
        self._upload_conversation_button(col1, conversation_key)
        if conversation_key in session_state['conversation_histories'] and session_state[
                'conversation_histories'][conversation_key]:
            self._download_conversation_button(col2, conversation_key)
            self._display_delete_conversation_button(col3)

    def _upload_conversation_button(self, container, conversation_key):
        """
        Renders an upload button in the specified Streamlit container to allow users to upload the conversation
        history from a CSV file. This function invokes another method to process the uploaded file upon the
        button click.

        Parameters:
        - container (st.delta_generator.DeltaGenerator): The Streamlit container (e.g., a sidebar or a column)
          where the upload button will be displayed.
        - conversation_key (str): A unique string identifier for the conversation history to be upl
        """
        upload_label = session_state['_']("Upload Conversation")
        with container:
            with st.popover("⬆️", help=upload_label):
                st.file_uploader(session_state['_']("**Upload Conversation**"),
                                 type=['csv'],
                                 key='file_uploader_conversation',
                                 on_change=self._process_uploaded_conversation_file,
                                 args=(conversation_key,)
                                 )

    @staticmethod
    def _download_conversation_button(container, conversation_key):
        """
        Renders a download button within a specified Streamlit container, allowing users to download their
        conversation history as a CSV file. This method first structures the conversation history into a
        pandas DataFrame, then converts it to CSV format for download.

        Parameters:
        - container (st.delta_generator.DeltaGenerator): The Streamlit container (e.g., a sidebar or a column)
          where the download button will be displayed.
        - conversation_key (str): The key used to identify the conversation history within the session state,
          ensuring the correct conversation history is made downloadable.
        """
        download_label = session_state['_']("Download Conversation")
        conversation_to_download = session_state['conversation_histories'][conversation_key]
        # Preprocess the conversation to remove the 'Images' entries
        conversation_to_download = [(speaker, message, prompt) for speaker, message, prompt, _ in
                                    conversation_to_download]
        conversation_df = pd.DataFrame(conversation_to_download,
                                       columns=[session_state['_']('Speaker'),
                                                session_state['_']('Message'),
                                                session_state['_']('System prompt')])
        conversation_csv = conversation_df.to_csv(index=False).encode('utf-8')
        container.download_button("⬇️️", data=conversation_csv, file_name="conversation.csv",
                                  mime="text/csv",
                                  help=download_label)

    @staticmethod
    def _process_uploaded_conversation_file(conversation_key):
        """
        Handles the processing of a conversation history file uploaded by the user. Validates the uploaded CSV
        file for required columns, updates the session state with the uploaded conversation, and extracts
        the latest 'System prompt' based on user's message from the uploaded conversation for further processing.

        Parameters:
        - conversation_key (str): A unique identifier for the conversation history to correctly update the session
          state with the uploaded content.
        """
        if session_state['file_uploader_conversation']:
            try:
                session_state['work_around_for_broken_ui_conversation'] = True
                # Read the uploaded CSV file into a DataFrame
                conversation_df = pd.read_csv(session_state['file_uploader_conversation'])

                # Validate necessary columns exist
                required_columns = [session_state['_']('Speaker'),
                                    session_state['_']('Message'),
                                    session_state['_']('System prompt')]
                if not all(column in conversation_df.columns for column in required_columns):
                    st.error(
                        session_state['_']("The uploaded file is missing one or more required columns:"
                                           " 'Speaker', 'Message', 'System prompt'"))
                    return

                # Add a fourth value (None) to each tuple in the conversation list
                conversation_list = []
                for row in conversation_df[required_columns].itertuples(index=False):
                    conversation_list.append(tuple(list(row) + [None]))  # Add None as the fourth value

                # Update the session state with the uploaded conversation
                session_state['conversation_histories'][conversation_key] = conversation_list

                # Extract the last "System prompt" from a user's message
                # in the uploaded conversation for "edited_prompts"
                user_prompts = [tuple_conv for tuple_conv in conversation_list if tuple_conv[0] ==
                                session_state['USER']]
                if user_prompts:  # Make sure the list is not empty
                    last_user_prompt = user_prompts[-1][2]  # Safely access the last tuple
                    session_state['edited_prompts'][conversation_key] = last_user_prompt
                else:
                    return
                st.success(session_state['_']("Successfully uploaded the conversation."))
                time.sleep(2)
            except Exception as e:
                st.error(session_state['_']("Failed to process the uploaded file. Error: "))
                st.error(e)

    def _logout_option(self):
        """
        Display a logout button in the sidebar, allowing users to end their session.

        This method renders a logout button in the sidebar. If clicked, it calls the `logout_and_redirect` method
        to clear session variables and cookies, securely logging out the user and redirecting them to the start page.
        """
        with st.sidebar:
            if st.button(session_state['_']('Logout')):
                self.logout_and_redirect()
            st.write(f"Version: *v1.2.0*")
