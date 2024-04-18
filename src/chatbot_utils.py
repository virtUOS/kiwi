import os
import time

import pandas as pd
import streamlit as st
from streamlit import session_state
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SidebarChatManager:

    def __init__(self, sidebar_general_manager, general_manager):
        self.sgm = sidebar_general_manager
        self.gm = general_manager

    def initialize_chat_session_variables(self):
        """
        Initialize chat session variables with default values.

        Sets up the initial state for the chat application to function correctly from the start.
        """
        self.sgm.load_prompts()
        required_keys = {
            'edited_prompts': {},
            'disable_custom': False,
        }

        for key, default_value in required_keys.items():
            if key not in session_state:
                session_state[key] = default_value

    def _show_conversation_controls(self):
        """
        Display buttons for conversation management, including deleting uploading and downloading conversation history,
         in the sidebar.

        This method checks if there is an existing conversation history for the currently selected chatbot path and,
         if so,
        displays options to either delete this history or download it as a CSV file.
        """
        conversation_key = session_state['selected_chatbot_path_serialized']
        with st.sidebar:
            st.markdown("---")
            st.write(session_state['_']("**Options**"))
            col1, col2, col3 = st.columns([1, 1, 1])
            if conversation_key in session_state['conversation_histories'] and session_state[
                    'conversation_histories'][conversation_key]:
                self._delete_conversation_button(col3)
                self._download_conversation_button(col2, conversation_key)
            self._upload_conversation_button(col1, conversation_key)

    def display_chat_top_sidebar_controls(self):
        """
        Display chat top sidebar controls and settings for the chatbot.

        This function performs the following steps:
        1. Display conversation controls.
        """
        self.sgm.load_and_display_prompts()

    def display_chat_middle_sidebar_controls(self):
        """
        Display chat middle sidebar controls and settings for the chatbot.

        This function performs the following steps:
        1. Display conversation controls.
        """
        self._show_conversation_controls()

    @staticmethod
    def _delete_conversation_callback():
        session_state['conversation_histories'][session_state['selected_chatbot_path_serialized']] = []

    def _delete_conversation_button(self, column):
        """
        Render a button in the specified column that allows the user
        to delete the active chatbot's conversation history.

        On button click, this static method removes the conversation history from the session state for the
        current chatbot path and triggers a page rerun to refresh the state.

        :param column: The column in Streamlit where the button should be placed.
        This should be a Streamlit column object.
        """
        column.button("🗑️",
                      on_click=self._delete_conversation_callback,
                      help=session_state['_']("Delete Conversation"))

    @staticmethod
    def _download_conversation_button(container, conversation_key):
        """
        Render a button in the specified column allowing users to download the conversation history as a CSV file.

        This static method creates a DataFrame from the conversation history stored in the session state under the
        given `conversation_key`, converts it to a CSV format, and provides a download button for the generated CSV.

        :param container: The container in Streamlit where the button should be placed.
        This should be a Streamlit column object.
        :param conversation_key: The key that uniquely identifies the conversation in the session state's
        conversation histories.
        """
        conversation_to_download = session_state['conversation_histories'][conversation_key]
        conversation_df = pd.DataFrame(conversation_to_download,
                                       columns=[session_state['_']('Speaker'),
                                                session_state['_']('Message'),
                                                session_state['_']('System prompt')])
        conversation_csv = conversation_df.to_csv(index=False).encode('utf-8')
        container.download_button("⬇️", data=conversation_csv, file_name="conversation.csv", mime="text/csv",
                                  help=session_state['_']("Download Conversation"))

    @staticmethod
    def _process_uploaded_conversation_file(container, conversation_key):
        """
        Handles the processing of a conversation history file uploaded by the user. Validates the uploaded CSV
        file for required columns, updates the session state with the uploaded conversation, and extracts
        the latest 'System prompt' based on user's message from the uploaded conversation for further processing.

        Parameters:
        - container (st.delta_generator.DeltaGenerator): The Streamlit container displaying status messages
          regarding the uploading process.
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
                    container.error(
                        session_state['_']("The uploaded file is missing one or more required columns:"
                                           " 'Speaker', 'Message', 'System prompt'"))
                    return

                # Convert the DataFrame to a list of tuples (converting each row to a tuple)
                conversation_list = conversation_df[required_columns].to_records(index=False).tolist()

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
                container.success(session_state['_']("Successfully uploaded the conversation."))
                time.sleep(2)
            except Exception as e:
                container.error(session_state['_']("Failed to process the uploaded file. Error: "), e)

    @staticmethod
    def _style_language_uploader():
        lang = 'de'
        if 'lang' in st.query_params:
            lang = st.query_params['lang']

        languages = {
            "en": {
                "instructions": "Drag and drop files here",
                "limits": "Limit 200MB per file",
            },
            "de": {
                "instructions": "Dateien hierher ziehen und ablegen",
                "limits": "Limit 200MB pro Datei",
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

    def _upload_conversation_file(self, container, conversation_key):
        """
        Render a file uploader in the specified container allowing users to upload a conversation history CSV file.

        This static method parses the uploaded CSV file to extract the conversation history and updates the
        session state's conversation history for the current conversation_key with the uploaded data.

        Parameters:
        - container: The Streamlit container (e.g., st.sidebar, st) where the uploader should be placed.
        This should be a Streamlit container object.
        - conversation_key: The key that uniquely identifies the conversation in the session state's
        conversation histories.
        """
        self._style_language_uploader()
        container.file_uploader(session_state['_']("**Upload conversation**"),
                                type=['csv'],
                                key='file_uploader_conversation',
                                on_change=self._process_uploaded_conversation_file,
                                args=(container, conversation_key)
                                )

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
        if container.button("⬆️", help=session_state['_']("Upload Conversation")):
            self._upload_conversation_file(st, conversation_key)


class ChatManager:

    def __init__(self, general_manager):
        """
        Initializes the ChatManager instance with the user's identifier.

        Parameters:
        - user: A string identifier for the user, used to differentiate messages in the conversation.
        """
        self.gm = general_manager
        self.client = None

    def set_client(self, client):
        """
        Sets the client for API requests, typically used to interface with chat models.

        Parameters:
        - client: The client instance responsible for handling requests to chat models.
        """
        self.client = client

    def _display_prompt_editor(self, description):
        """
        Displays an editable text area for the current chatbot prompt, allowing the user to make changes.

        Parameters:
        - description: The default chatbot prompt description which can be edited by the user.
        """
        current_chatbot_path_serialized = session_state['selected_chatbot_path_serialized']
        current_edited_prompt = self.gm.get_prompt_to_use(current_chatbot_path_serialized, description)

        st.text_area(session_state['_']("System prompt"),
                     value=current_edited_prompt,
                     on_change=self.gm.update_edited_prompt,
                     args=session_state['selected_chatbot_path'],
                     key='edited_prompt',
                     label_visibility='hidden')

        st.button("🔄", help=session_state['_']("Restore Original Prompt"),
                  on_click=self.gm.restore_prompt,
                  args=session_state['selected_chatbot_path']
                  )

    def _handle_user_input(self, description_to_use):
        """
        Handles the user input: sends the message to OpenAI, prints it, and updates the conversation history.

        Parameters:
        - description_to_use: The current system description or prompt used alongside user messages.
        """
        if user_message := st.chat_input(session_state['USER']):
            current_history = self.gm.get_conversation_history(session_state['selected_chatbot_path_serialized'])

            # Ensures unique submission to prevent re-running on refresh
            if self.gm.is_new_message(current_history, user_message):
                if not current_history:
                    self.gm.create_history(session_state['selected_chatbot_path_serialized'])
                current_history = self.gm.add_conversation_entry(session_state['selected_chatbot_path_serialized'],
                                                                 session_state['USER'],
                                                                 user_message,
                                                                 description_to_use)

                # Print user message immediately after getting entered because we're streaming the chatbot output
                with st.chat_message("user"):
                    st.markdown(user_message)

                # Process and display response
                current_history = self.client.process_response(current_history, user_message, description_to_use)

                self.gm.update_conversation_history(session_state['selected_chatbot_path_serialized'], current_history)
                st.rerun()

    @staticmethod
    def _display_portal_model_info():
        """
        Displays information about the portal and model in use.
        """
        using_text = session_state['_']("You're using the following OpenAI model:")
        remember_text = session_state['_']("Remember **not** to enter any personal information"
                                           " or copyrighted material.")
        model_info = f"{using_text} **{os.getenv('OPENAI_MODEL')}**. {remember_text}"
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
                self._display_portal_model_info()

    def display_chat_interface(self):
        """
        Displays the chat interface, manages the display of conversation history, and handles user input.

        This method sets up the interface for the chat, including fetching and displaying the system prompt,
        updating session states as necessary, and calling methods to display conversation history and handle user input.
        """
        if 'selected_chatbot_path' in session_state and session_state["selected_chatbot_path"]:

            prompt = self.gm.fetch_chatbot_prompt(session_state['selected_chatbot_path'])

            if isinstance(prompt, str) and len(prompt.strip()) > 0:

                # Display chat interface header and model information if applicable
                self._display_chat_interface_header()

                with st.expander(label=session_state['_']("View or edit system prompt"), expanded=False):
                    self._display_prompt_editor(prompt)

                st.markdown("""---""")

                prompt_to_use = self.gm.get_prompt_to_use(session_state['selected_chatbot_path_serialized'],
                                                          prompt)

                # Displays the existing conversation history
                conversation_history = self.gm.get_conversation_history(
                    session_state['selected_chatbot_path_serialized'])

                self.gm.display_conversation(conversation_history)

                # Handles the user's input and interaction with the LLM
                self._handle_user_input(prompt_to_use)
            else:
                st.error(session_state['_']("The System Prompt should be a string and not empty."))
