import json
import os
import time
import base64
import re
from PIL import Image
from io import BytesIO

import pandas as pd
from openai import OpenAI
from streamlit import session_state
from streamlit_option_menu import option_menu
from streamlit_cookies_manager import EncryptedCookieManager
from streamlit_float import *
from src import menu_utils
from dotenv import load_dotenv

from src.language_utils import initialize_language, language_controls

# Load environment variables
load_dotenv()


class SidebarManager:

    def __init__(self):
        """
        Initialize the SidebarManager instance by setting up session cookies and initializing the language.
        """
        self.cookies = self.initialize_cookies()
        initialize_language()
        st.info(session_state['_']("The outputs of the chat assistant may be erroneous - therefore, "
                                   "always check the answers for their accuracy. Remember not to enter "
                                   "any personal information and copyrighted materials."))
        language_controls()
        self.advanced_model = "gpt-4o"
        # Float feature initialization
        float_init()

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
            'images_histories': {},
            'selected_chatbot_path_serialized': "",
            'prompt_options': menu_utils.load_prompts_from_yaml(),
            'edited_prompts': {},
            'disable_custom': False,
            'images_key': 0,
            'toggle_key': -1,
            'image_urls': [],
            'uploaded_images': [],
            'image_content': [],
            'photo_to_use': [],
            'activate_camera': False,
        }

        for key, default_value in required_keys.items():
            if key not in session_state:
                session_state[key] = default_value

    @staticmethod
    def display_logo():
        """
        Display the application logo in the sidebar.

        Utilizes Streamlit columns to center the logo and injects CSS to hide the fullscreen
        button that appears on hover over the logo.
        """
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
        """
        Load chat prompts based on the selected or default language.

        This method enables dynamic loading of chat prompts to support multi-lingual chatbot interactions.
        If no language is specified in the query parameters, German ('de') is used as the default language.
        """
        language = st.query_params.get('lang', False)
        # Use German language as default
        if not language:
            language = "de"

        """Load chat prompts based on language."""
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

            with st.sidebar:
                choice = option_menu("", next_level,
                                     # icons=['chat-dots'] * len(next_level),
                                     # menu_icon="cast",
                                     default_index=0)

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

        self._style_language_uploader()

        with st.sidebar:
            st.markdown("---")
            st.write(session_state['_']("**Options**"))

        self._show_conversation_controls()

        if 'selected_model' in session_state and session_state['selected_model'] == self.advanced_model:
            self._show_images_controls()

        self._add_custom_css()

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
            if session_state['model_selection'] == 'OpenAI':
                accessible_models = AIClient.get_accessible_models()

                # Show dropdown when multiple models are available
                if len(accessible_models) > 1:
                    model_label = session_state['_']("Model:")
                    index = 0
                    if self.advanced_model in accessible_models:  # Use most advanced model as default
                        index = accessible_models.index('gpt-4o')
                    st.selectbox(model_label,
                                 accessible_models,
                                 index=index,
                                 key='selected_model')

                    # If the model is changed to one that doesn't support images
                    # we have to clear the widgets from images. For the upload widget,
                    # the way to do this is by generating a new widget with a new key
                    # as described in:
                    # https://discuss.streamlit.io/t/
                    # are-there-any-ways-to-clear-file-uploader-values-without-using-streamlit-form/40903
                    if session_state['selected_model'] != self.advanced_model and (
                            session_state['image_urls'] or
                            session_state['uploaded_images'] or
                            session_state['camera_image_content']):
                        session_state['images_key'] += 1
                        session_state['toggle_key'] -= 1
                        session_state['image_urls'] = []
                        session_state['uploaded_images'] = []
                        session_state['image_content'] = []
                        session_state['photo_to_use'] = []
                        session_state['activate_camera'] = False
                        st.rerun()

                else:
                    model_text = session_state['_']("Model:")
                    st.write(f"{model_text} {AIClient.get_selected_model()}")

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
            # self._delete_conversation_button(col3)

    @staticmethod
    def _get_rid_of_submit_text():
        """
        Hides the default submit text in the Streamlit text area component.

        This method injects custom CSS into the Streamlit app to hide the default submit text
        that appears in the text area component.
        This enhances the user interface by removing unnecessary visual elements.
        """
        hide_submit_text = """
        <style>
        div[data-testid="InputInstructions"] > span:nth-child(1) {
            visibility: hidden;
        }
        </style>
        """
        st.markdown(hide_submit_text, unsafe_allow_html=True)

    @staticmethod
    def _close_sidepanel_callback():
        """
        Callback function to manage the state of the sidebar based on camera activation.

        This method sets the state of the sidebar (collapsed or expanded) based
        on whether the camera input is active or not.
        If the camera is activated, the sidebar is collapsed; otherwise, it is expanded.
        """
        if session_state['activate_camera']:
            session_state['sidebar_state'] = "collapsed"
        else:
            session_state['sidebar_state'] = "expanded"

    def _show_images_controls(self):
        """
        Display buttons for image management, including uploading images, in the sidebar.

        This method presents an expander with the label "Images Controls" in the sidebar, inside which there are various options
        to manage images. Users can activate camera input, upload multiple images of supported formats, and enter image URLs manually.

        Functionalities provided:
        - Toggle to activate camera input.
        - File uploader to upload images with supported formats.
        - Text area to manually enter image URLs, which are processed and stored in the session state on button click.

        The method also makes sure to remove any unnecessary submit text from the text area widget.
        """
        with st.sidebar:
            with st.expander(session_state['_']("Images")):
                # Checkbox to activate camera input
                # st.toggle(session_state['_']("Activate camera"),
                #          key='activate_camera',
                #          on_change=self._close_sidepanel_callback)

                # Widget to upload images
                session_state['uploaded_images'] = st.file_uploader(session_state['_']("Upload Images"),
                                                                    type=['png', 'jpeg', 'jpg', 'gif', 'webp'],
                                                                    accept_multiple_files=True,
                                                                    key=session_state['images_key'])

                # Text area for image URLs (Get rid of the submit text on the text area because it's useless here)
                self._get_rid_of_submit_text()
                urls = st.text_area(session_state['_']("Enter Image URLs (one per line)"))

                if st.button(session_state['_']("Add URLs")):
                    # Process the URLs
                    url_list = urls.strip().split('\n')
                    if 'image_urls' not in session_state:
                        session_state['image_urls'] = []
                    session_state['image_urls'].extend([url.strip() for url in url_list if url.strip()])

    def _upload_conversation_button(self, container, conversation_key):
        """
        Renders an upload button in the specified Streamlit container to allow users to upload the conversation
        history from a CSV file. This function invokes another method to process the uploaded file upon the
        button click.

        Parameters:
        - container (st.delta_generator.DeltaGenerator): The Streamlit container (e.g., a sidebar or a column)
          where the upload button will be displayed.
        - conversation_key (str): A unique string identifier for the conversation history to be uploaded. This
          is used to correctly associate the uploaded conversation with its relevant session state.
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

                # Convert the DataFrame to a list of tuples (converting each row to a tuple)
                conversation_list = conversation_df[required_columns].to_records(index=False).tolist()

                # Update the session state with the uploaded conversation
                session_state['conversation_histories'][conversation_key] = conversation_list
                # Initialize images_histories with lists of empty dictionaries
                session_state['images_histories'][conversation_key] = [{} for _ in range(len(conversation_list))]

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

    @staticmethod
    def _style_language_uploader():
        """
        Styles the file uploader component based on the selected language.

        This function customizes the instructions and limits text shown in the file
        uploader based on the user's language preference.
        It supports English and German language instructions.

        The style is applied using specific Streamlit DOM elements.

        It uses a simple data structure to switch between languages and generate the appropriate HTML
        and CSS modifications.

        Supported languages:
        - English ("en")
        - German ("de")

        Changes the visibility of default text and replaces it with translated instructions and file limits.
        """
        lang = 'de'
        if 'lang' in st.query_params:
            lang = st.query_params['lang']

        languages = {
            "en": {
                "instructions": "Drag and drop files here",
                "limits": "Limit 20MB per file",
            },
            "de": {
                "instructions": "Dateien hierher ziehen und ablegen",
                "limits": "Limit 20MB pro Datei",
            },
        }

        hide_label = (
            """
        <style>
            div[data-testid="stFileUploaderDropzoneInstructions"]>div>span {
               visibility:hidden;
            }
            div[data-testid="stFileUploaderDropzoneInstructions"]>div>span::after {
               content:"INSTRUCTIONS_TEXT";
               visibility:visible;
               display:block;
            }
             div[data-testid="stFileUploaderDropzoneInstructions"]>div>small {
               visibility:hidden;
            }
            div[data-testid="stFileUploaderDropzoneInstructions"]>div>small::before {
               content:"FILE_LIMITS";
               visibility:visible;
               display:block;
            }
        </style>
        """.replace("INSTRUCTIONS_TEXT", languages.get(lang).get("instructions"))
            .replace("FILE_LIMITS", languages.get(lang).get("limits"))
        )

        st.markdown(hide_label, unsafe_allow_html=True)

    @staticmethod
    def _add_custom_css():
        """
        Inject custom CSS into the sidebar to enhance its aesthetic according to the current theme.

        This static method retrieves the secondary background color from Streamlit's theme options
        and applies custom styles to various sidebar elements, enhancing the user interface.
        """
        color = st.get_option('theme.secondaryBackgroundColor')
        css = f"""
                [data-testid="stSidebarNav"] {{
                    position:absolute;
                    bottom: 0;
                    z-index: 1;
                    background: {color};
                }}
                """
        with st.sidebar:
            st.markdown("---")
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    def _logout_option(self):
        """
        Display a logout button in the sidebar, allowing users to end their session.

        This method renders a logout button in the sidebar. If clicked, it calls the `logout_and_redirect` method
        to clear session variables and cookies, securely logging out the user and redirecting them to the start page.
        """
        with st.sidebar:
            if st.button(session_state['_']('Logout')):
                self.logout_and_redirect()
            st.write(f"Version: *v1.1.5*")


class ChatManager:

    def __init__(self, user):
        """
        Initializes the ChatManager instance with the user's identifier.

        Parameters:
        - user: A string identifier for the user, used to differentiate messages in the conversation.
        """
        self.client = None
        session_state['USER'] = user

    def set_client(self, client):
        """
        Sets the client for API requests, typically used to interface with chat models.

        Parameters:
        - client: The client instance responsible for handling requests to chat models.
        """
        self.client = client

    @staticmethod
    def _update_edited_prompt():
        """
        Updates the edited prompt in the session state to reflect changes made by the user.
        """
        session_state['edited_prompts'][session_state[
            'selected_chatbot_path_serialized']] = session_state['edited_prompt']

    @staticmethod
    def _restore_prompt():
        """
        Restores the original chatbot prompt by removing any user-made edits from the session state.
        """
        if session_state['selected_chatbot_path_serialized'] in session_state['edited_prompts']:
            del session_state['edited_prompts'][session_state['selected_chatbot_path_serialized']]

    def _display_prompt_editor(self, description):
        """
        Displays an editable text area for the current chatbot prompt, allowing the user to make changes.

        Parameters:
        - description: The default chatbot prompt description which can be edited by the user.
        """
        current_chatbot_path_serialized = session_state['selected_chatbot_path_serialized']
        current_edited_prompt = session_state['edited_prompts'].get(current_chatbot_path_serialized, description)

        st.text_area(session_state['_']("System prompt"),
                     value=current_edited_prompt,
                     on_change=self._update_edited_prompt,
                     key='edited_prompt',
                     label_visibility='hidden')

        st.button("🔄", help=session_state['_']("Restore Original Prompt"), on_click=self._restore_prompt)

    @staticmethod
    def _get_description_to_use(default_description):
        """
        Retrieves and returns the current prompt description for the chatbot, considering any user edits.

        Parameters:
        - default_description: The default description provided for the chatbot's prompt.

        Returns:
        - The current (possibly edited) description to be used as the prompt.
        """
        return session_state['edited_prompts'].get(session_state[
                                                       'selected_chatbot_path_serialized'], default_description)

    @staticmethod
    def _display_images_inside_message(images, user_message_container=None):
        user_message_container = user_message_container if user_message_container else st
        # Check if there are images associated with this message
        if images:
            with user_message_container.expander("View Images"):
                for image_data in images:
                    for i, (name, image) in enumerate(image_data.items()):
                        st.write(f"{i} - {name}")
                        if isinstance(image, str):  # It's an image URL
                            st.write(image)
                        else:  # It's an image object
                            st.image(image)

    def _display_conversation(self, conversation_history, images_history, container=None):
        """Displays the conversation history between the user and the assistant within the given container or globally.

        Parameters:
        - conversation_history: A list containing tuples of (speaker, message) representing the conversation.
        - images_histories: A list containing dictionaries of images corresponding to each conversation message.
        - container: The Streamlit container (e.g., column, expander) where the messages should be displayed.
        If None, messages will be displayed in the main app area.
        """
        for (speaker, message, __), images in zip(conversation_history, images_history):
            chat_message_container = container if container else st
            if speaker == session_state['USER']:
                user_message_container = chat_message_container.chat_message("user")
                user_message_container.write(message)
                self._display_images_inside_message(images, user_message_container)

            elif speaker == "Assistant":
                chat_message_container.chat_message("assistant").write(message)

    @staticmethod
    def _get_current_conversation_history():
        """
        Retrieves the current conversation history for the selected chatbot path from the session state.

        Returns:
        - A list representing the conversation history.
        """
        serialized_path = session_state['selected_chatbot_path_serialized']
        if serialized_path not in session_state['conversation_histories']:
            session_state['conversation_histories'][serialized_path] = []
            session_state['images_histories'][serialized_path] = []
        return session_state['conversation_histories'][serialized_path]

    @staticmethod
    def _is_new_message(current_history, user_message):
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

    @staticmethod
    def _update_conversation_history(current_history):
        """
        Updates the session state with the current conversation history.

        This method ensures that the conversation history for the currently selected chatbot path in the session state
        is updated to reflect any new messages or responses that have been added during the interaction.

        Parameters:
        - current_history: The current conversation history, which is a list of tuples. Each tuple contains
          the speaker ('USER' or 'Assistant'), the message, and optionally, the system description or prompt
          accompanying user messages (for user messages only).
        """
        session_state['conversation_histories'][session_state['selected_chatbot_path_serialized']] = current_history

    @staticmethod
    def _append_user_message_to_history(current_history, user_message, description_to_use):
        """
        Appends the user's message to the conversation history in the session state.

        Parameters:
        - current_history: The current conversation history.
        - user_message: The message input by the user.
        - description_to_use: The system description or prompt that should accompany user messages.
        """
        # Adding user message and description to the current conversation
        current_history.append((session_state['USER'], user_message, description_to_use))

    def _process_response(self, current_history, user_message, description_to_use):
        """
        Submits the user message to OpenAI, retrieves the response, and updates the conversation history.

        Parameters:
        - current_history: The current conversation history.
        - user_message: The message input by the user.
        - description_to_use: The current system description or prompt associated with the user message.
        """
        response = self.client.get_response(user_message, description_to_use)
        # Add AI response to the history
        current_history.append(('Assistant', response, ""))

    def _handle_user_input(self, description_to_use, images, container_chat):
        """
        Handles the user input: sends the message to OpenAI, prints it, and updates the conversation history.

        This method is responsible for capturing the user input from the chat interface, sending it to the
        language model for a response, displaying the user message immediately, and updating the conversation
        history to include both the user message and the model's response. It ensures that the user input is
        processed only once per new message.

        Parameters:
        - description_to_use (str): The current system description or prompt used alongside user messages.
        - container_chat (streamlit.container): The Streamlit container where the chat messages are displayed.
        """
        selected_chatbot = session_state['selected_chatbot_path_serialized']
        if user_message := st.chat_input(session_state['USER']):
            current_history = self._get_current_conversation_history()

            with container_chat:
                # Ensures unique submission to prevent re-running on refresh
                if self._is_new_message(current_history, user_message):
                    self._append_user_message_to_history(current_history, user_message, description_to_use)

                    # Print user message immediately after getting entered because we're streaming the chatbot output
                    with st.chat_message("user"):
                        st.markdown(user_message)
                        self._display_images_inside_message(images)

                    # Process and display response
                    self._update_conversation_history(current_history)
                    self._process_response(current_history, user_message, description_to_use)
                    # Add extra empty one for the system message
                    session_state['images_histories'][selected_chatbot].append({})
                    st.rerun()

    @staticmethod
    def _fetch_chatbot_description():
        """
        Fetches the description for the current chatbot based on the selected path.

        Returns:
        - A string containing the description of the current chatbot.
        """
        return menu_utils.get_final_description(session_state["selected_chatbot_path"], session_state["prompt_options"])

    @staticmethod
    def _display_openai_model_info():
        """
        Displays information about the current OpenAI model in use.
        """
        using_text = session_state['_']("You're using the following OpenAI model:")
        model_info = f"{using_text} **{AIClient.get_selected_model()}**."
        st.write(model_info)
        st.write(session_state['_']("Each time you enter information,"
                                    " a system prompt is sent to the chat model by default."))

    def _display_chat_interface_header(self):
        """
        Displays headers and model information on the chat interface based on the user's selection.
        """
        st.markdown("""---""")  # Separator for layout
        if (session_state['selected_chatbot_path_serialized'] not in session_state['conversation_histories']
                or not session_state['conversation_histories'][session_state['selected_chatbot_path_serialized']]):
            st.header(session_state['_']("How can I help you?"))
            if session_state['model_selection'] == 'OpenAI':
                self._display_openai_model_info()

    @staticmethod
    def _clear_photo_callback():
        session_state['photo_to_use'] = []

    @staticmethod
    def _resize_image_and_get_base64(image, thumbnail_size):
        img = Image.open(image)
        resized_img = img.resize(thumbnail_size, Image.Resampling.BILINEAR)

        # Convert RGBA to RGB if necessary
        if resized_img.mode == 'RGBA':
            resized_img = resized_img.convert('RGB')

        buffer = BytesIO()
        resized_img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str, resized_img

    def _display_images_column(self, uploaded_images, image_urls, photo_to_use, selected_chatbot):
        """
        Displays uploaded images, image URLs, and user photo in a specified column.

        Parameters:
        - uploaded_images (list): List of uploaded images.
        - image_urls (list): List of image URLs.
        - photo_to_use (UploadedFile): Photo captured via camera input.
        """
        session_state['image_content'] = []
        thumbnail_dim = 100
        thumbnail_size = (thumbnail_dim, thumbnail_dim)

        # Create a new dictionary to hold all images for populating alongside the conversation history
        images_dict = {}

        if uploaded_images:
            st.markdown(session_state['_']("### Uploaded Images"))
            for image in uploaded_images:
                image64, resized_image = self._resize_image_and_get_base64(image, thumbnail_size)
                images_dict[image.name] = resized_image
                session_state['image_content'].append({
                    'type': "image_url",
                    'image_url': {"url": f"data:image/jpeg;base64,{image64}"}
                })
                st.image(image)

        if image_urls:
            st.markdown(session_state['_']("### Image URLs"))
            for i, url in enumerate(image_urls):
                images_dict[f"url-{i}"] = url
                session_state['image_content'].append({
                    'type': "image_url",
                    'image_url': {"url": url}
                })
                st.write(url)

        if photo_to_use:
            st.markdown(session_state['_']("### Photo"))
            photo64, resized_photo = self._resize_image_and_get_base64(photo_to_use, thumbnail_size)
            images_dict['photo'] = resized_photo
            session_state['image_content'].append({
                'type': "image_url",
                'image_url': {"url": f"data:image/jpeg;base64,{photo64}"}
            })
            st.image(photo_to_use)
            st.button("Clear photo 🧹", on_click=self._clear_photo_callback)

        # Append the dictionary of images to the chatbot's list of histories
        session_state['images_histories'][selected_chatbot].append(images_dict)

    @staticmethod
    def _display_camera():
        """
        Renders the camera input widget and displays the captured photo.
        """
        col3, col4, col5 = st.columns([1, 1, 1])
        with col4:
            st.camera_input(
                session_state['_']("Take a photo"),
                key='your_photo',
                label_visibility="hidden")
            if session_state['your_photo']:
                st.button("Use photo",
                          key='use_photo_button',
                          use_container_width=True)
            float_parent(f"bottom: 20rem; background-color: var(--default-backgroundColor); padding-top: 1rem;")
            if session_state['your_photo']:
                if session_state['use_photo_button']:
                    session_state['photo_to_use'] = session_state['your_photo']
                    session_state['toggle_key'] -= 1
                    session_state['activate_camera'] = False
                    st.rerun()

    @staticmethod
    def _toggle_camera_callback():
        if not session_state['activate_camera']:
            session_state['activate_camera'] = True
        else:
            session_state['activate_camera'] = False
            session_state['photo_to_use'] = []

    @staticmethod
    def _delete_conversation_callback():
        """
        Callback function to delete the current conversation history.

        This method clears the conversation history for the selected chatbot path stored in the session state.
        It sets the conversation history to an empty list.
        """
        session_state['conversation_histories'][session_state['selected_chatbot_path_serialized']] = []
        session_state['images_histories'][session_state['selected_chatbot_path_serialized']] = []

    def _delete_conversation_button(self, container):
        """
        Render a button in the specified column that allows the user
        to delete the active chatbot's conversation history.

        On button click, this static method removes the conversation history from the session state for the
        current chatbot path and triggers a page rerun to refresh the state.

        :param container: The container in Streamlit where the button should be placed.
        This should be a Streamlit column object.
        """
        delete_label = session_state['_']("Delete Conversation")
        container.button("🗑️", on_click=self._delete_conversation_callback, help=delete_label)

    def _display_chat_buttons(self):
        chat_buttons = st.container()
        chat_buttons.float(
            "bottom: 6.5rem;background-color: var(--default-backgroundColor); "
            "padding-top: 1rem; max-width: 80vw; flex-wrap: nowrap"
        )

        cols_dimensions = [99, 7, 6]

        conversation_key = session_state['selected_chatbot_path_serialized']

        with chat_buttons:
            col0, col1, col2 = st.columns(
                cols_dimensions
            )

            col0.toggle("📷",
                        key=session_state['toggle_key'],
                        value=False,
                        help=session_state['_']("Activate Camera"),
                        on_change=self._toggle_camera_callback)

            if conversation_key in session_state['conversation_histories'] and session_state[
                    'conversation_histories'][conversation_key]:
                self._delete_conversation_button(col2)

    def display_chat_interface(self):
        """
        Displays the chat interface, manages the display of conversation history, and handles user input.

        This method sets up the interface for the chat, including:
        - Fetching and displaying the system prompt.
        - Updating session states as necessary.
        - Displaying conversation history if any.
        - Handling user input.
        - Displaying uploaded images and URLs if provided.

        It also provides an option to view or edit the system prompt through an expander in the layout.
        """
        if 'selected_chatbot_path' in session_state and session_state["selected_chatbot_path"]:
            selected_chatbot = session_state['selected_chatbot_path_serialized']
            description = self._fetch_chatbot_description()

            if isinstance(description, str) and description.strip():
                if session_state.get('activate_camera', None):
                    self._display_camera()

                # Initialize variables for uploaded content
                uploaded_images = session_state.get('uploaded_images', [])
                image_urls = session_state.get('image_urls', [])
                photo_to_use = session_state.get('photo_to_use', [])

                # Set layout columns based on whether there are images or URLs
                if uploaded_images or image_urls or photo_to_use:
                    col1, col2 = st.columns([2, 1], gap="medium")
                else:
                    col1, col2 = st.container(), None
                    session_state['image_content'] = []

                with col1:
                    # Display chat interface header and model information if applicable
                    self._display_chat_interface_header()

                    # Option to view or edit the system prompt
                    with st.expander(label=session_state['_']("View or edit system prompt"), expanded=False):
                        self._display_prompt_editor(description)

                    st.markdown("""---""")

                    self._display_chat_buttons()

                    description_to_use = self._get_description_to_use(description)

                    # Displays the existing conversation history
                    conversation_history = session_state['conversation_histories'].get(selected_chatbot, [])
                    images_history = session_state['images_histories'].get(selected_chatbot, [])
                    self._display_conversation(conversation_history, images_history, col1)

                # If col2 is defined, show uploaded images and URLs
                if col2:
                    with col2:
                        self._display_images_column(uploaded_images, image_urls, photo_to_use, selected_chatbot)
                else:
                    # Add empty dictionary if there are no images to keep up with the placing of images in the
                    # conversation
                    session_state['images_histories'][selected_chatbot].append({})

                # Handles the user's input and interaction with the LLM
                self._handle_user_input(description_to_use,
                                        session_state['images_histories'][selected_chatbot][-1],
                                        col1)

                # Adds empty lines to the main app area to avoid hiding text behind chat buttons
                # (might need adjustments)
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
            else:
                st.error(session_state['_']("The System Prompt should be a string and not empty."))


class AIClient:

    def __init__(self):
        if session_state['model_selection'] == 'OpenAI':

            @st.cache_resource
            def load_openai_data():
                return OpenAI()

            self.client = load_openai_data()
        else:
            pass
            # Add here client initialization for local model
            # For example using OpenLLM as inference server
            # session_state["inference_server_url"] = "http://localhost:3000/v1"
            # self.client = OpenAI(base_url=session_state['inference_server_url'])
            # models = session_state["client"].models.list()
            # self.model = models.data[0].id

    @staticmethod
    def get_accessible_models() -> list:
        """
        Get accessible models for the current user
        """
        # Load accessible user models from environment variables
        default_models = json.loads(os.environ['OPENAI_DEFAULT_MODEL'])
        if 'accessible_models' not in session_state and {'USER_ROLES', 'MODELS_PER_ROLE'} <= os.environ.keys():
            user_roles = json.loads(os.environ['USER_ROLES'])
            user_role = user_roles.get(session_state.get('username'))

            models_per_role = json.loads(os.environ['MODELS_PER_ROLE'])
            session_state['accessible_models'] = models_per_role.get(user_role, default_models['models'])

        # Get default model if no configurations found
        if 'accessible_models' not in session_state:
            session_state['accessible_models'] = default_models['models']

        return session_state['accessible_models']

    @staticmethod
    def get_selected_model():
        """
        Get the selected model for the current user
        """
        accessible_models = AIClient.get_accessible_models()

        # Get selected model from dropdown
        if 'selected_model' in session_state and session_state['selected_model'] in accessible_models:
            return session_state['selected_model']

        return os.getenv('OPENAI_DEFAULT_MODEL')

    @staticmethod
    def _generate_response(stream):
        """
        Extracts the content from the stream of responses from the OpenAI API.
        Parameters:
            stream: The stream of responses from the OpenAI API.
        """
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta:
                chunk_content = chunk.choices[0].delta.content
                if isinstance(chunk_content, str):
                    yield chunk_content
                else:
                    continue

    def _concatenate_partial_response(self):
        """
        Concatenates the partial response into a single string.
        """
        # The concatenated response.
        str_response = ""
        for i in self.partial_response:
            if isinstance(i, str):
                str_response += i

        replacements = {
            r'\\\s*\(': r'$',
            r'\\\s*\)': r'$',
            r'\\\s*\[': r'$$',
            r'\\\s*\]': r'$$'
        }

        # Perform the replacements
        for pattern, replacement in replacements.items():
            str_response = re.sub(pattern, replacement, str_response)

        st.markdown(str_response)

        self.partial_response = []
        self.response += str_response

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
            self.response = ""
            # true if the response contains a special text like code block or math expression
            self.special_text = False
            with st.chat_message("assistant"):
                with st.spinner(session_state['_']("Generating response...")):
                    stream = self.client.chat.completions.create(
                        model=session_state['selected_model'],
                        messages=messages,
                        stream=True,
                    )
                    self.partial_response = []

                    gen_stream = self._generate_response(stream)
                    for chunk_content in gen_stream:
                        # check if the chunk is a code block
                        # check if the chunk is a code block
                        if chunk_content == '```':
                            self._concatenate_partial_response()
                            self.partial_response.append(chunk_content)
                            self.special_text = True
                            while self.special_text:
                                try:
                                    chunk_content = next(gen_stream)
                                    self.partial_response.append(chunk_content)
                                    if chunk_content == "`\n\n":
                                        # show partial response to the user and keep it  for later use
                                        self._concatenate_partial_response()
                                        self.special_text = False
                                except StopIteration:
                                    break

                        else:
                            # If the chunk is not a code or math block, append it to the partial response
                            self.partial_response.append(chunk_content)
                            if chunk_content:
                                if '\n' in chunk_content:
                                    self._concatenate_partial_response()

                # If there is a partial response left, concatenate it and render it
                if self.partial_response:
                    self._concatenate_partial_response()

            return self.response

        except Exception as e:
            print(f"An error occurred while fetching the OpenAI response: {e}")
        # Return a default error message
        return session_state['_']("Sorry, I couldn't process that request.")

    @staticmethod
    def _prepare_full_prompt_and_messages(user_prompt, description_to_use):
        """
        Prepares the full prompt and messages combining user input and additional descriptions, including image content
        in the specified format.

        Parameters:
        - user_prompt (str): The original prompt from the user.
        - description_to_use (str): Additional context or instructions to provide to the model.

        Returns:
        - list: List of dictionaries containing messages for the chat completion.
        """

        if session_state["model_selection"] == 'OpenAI':
            messages = [{
                'role': "system",
                'content': description_to_use
            }]
        else:  # Add here conditions for different types of models
            messages = []

        # Add the history of the conversation, ignore the system prompt
        for speaker, message, __ in session_state['conversation_histories'][
            session_state['selected_chatbot_path_serialized']]:
            role = 'user' if speaker == session_state['USER'] else 'assistant'
            messages.append({'role': role, 'content': message})

        # Combine user prompt and image content
        user_message_content = [{"type": "text", "text": user_prompt}]
        user_message_content.extend(session_state['image_content'])

        # Building the messages with the "system" message based on expertise area
        if session_state["model_selection"] == 'OpenAI':
            messages.append({"role": "user", "content": user_message_content})
        else:
            # Add the description to the current user message to simulate a system prompt.
            # This is for models that were trained with no system prompt: LLaMA/Mistral/Mixtral.
            # Source: https://github.com/huggingface/transformers/issues/28366
            # Maybe find a better solution or avoid any system prompt for these models
            messages.append({"role": "user", "content": description_to_use + "\n\n" + user_prompt})

        # This method can be expanded based on how you want to structure the prompt
        # For example, you might prepend the description_to_use before the user_prompt
        return messages
