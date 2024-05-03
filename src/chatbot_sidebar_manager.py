import time
import pandas as pd

import streamlit as st
from streamlit import session_state

import src.streamlit_styling as st_styling


class SidebarChatManager:

    def __init__(self):
        pass

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
        st_styling.style_language_uploader()
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
