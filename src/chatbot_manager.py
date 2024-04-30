import os

import streamlit as st
from streamlit import session_state

import src.utils as utils


class ChatManager:

    def __init__(self):
        """
        Initializes the ChatManager instance with the user's identifier.

        Parameters:
        - user: A string identifier for the user, used to differentiate messages in the conversation.
        """
        super().__init__()
        self.client = None

    def set_client(self, client):
        """
        Sets the client for API requests, typically used to interface with chat models.

        Parameters:
        - client: The client instance responsible for handling requests to chat models.
        """
        self.client = client

    @staticmethod
    def _display_prompt_editor(description):
        """
        Displays an editable text area for the current chatbot prompt, allowing the user to make changes.

        Parameters:
        - description: The default chatbot prompt description which can be edited by the user.
        """
        current_chatbot_path_serialized = session_state['selected_chatbot_path_serialized']
        current_edited_prompt = utils.get_prompt_to_use(current_chatbot_path_serialized, description)

        st.text_area(session_state['_']("System prompt"),
                     value=current_edited_prompt,
                     on_change=utils.update_edited_prompt,
                     args=session_state['selected_chatbot_path'],
                     key='edited_prompt',
                     label_visibility='hidden')

        st.button("🔄", help=session_state['_']("Restore Original Prompt"),
                  on_click=utils.restore_prompt,
                  args=session_state['selected_chatbot_path']
                  )

    def _handle_user_input(self, description_to_use):
        """
        Handles the user input: sends the message to OpenAI, prints it, and updates the conversation history.

        Parameters:
        - description_to_use: The current system description or prompt used alongside user messages.
        """
        if user_message := st.chat_input(session_state['USER']):
            current_history = utils.get_conversation_history(session_state['selected_chatbot_path_serialized'])

            # Ensures unique submission to prevent re-running on refresh
            if utils.is_new_message(current_history, user_message):
                if not current_history:
                    utils.create_history(session_state['selected_chatbot_path_serialized'])
                current_history = utils.add_conversation_entry(session_state['selected_chatbot_path_serialized'],
                                                               session_state['USER'],
                                                               user_message,
                                                               description_to_use)

                # Print user message immediately after getting entered because we're streaming the chatbot output
                with st.chat_message("user"):
                    st.markdown(user_message)

                # Process and display response
                current_history = self.client.process_response(current_history, user_message, description_to_use)

                utils.update_conversation_history(session_state['selected_chatbot_path_serialized'], current_history)
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
        if 'selected_chatbot_path' in session_state and session_state['selected_chatbot_path']:

            # General info on usage
            st.info(session_state['_']("The outputs of the chat assistant may be erroneous - therefore, always "
                                       "check the answers for their accuracy. Remember not to enter any personal "
                                       "information and copyrighted materials."))

            prompt = utils.fetch_chatbot_prompt(session_state['selected_chatbot_path'])

            if isinstance(prompt, str) and len(prompt.strip()) > 0:

                # Display chat interface header and model information if applicable
                self._display_chat_interface_header()

                with st.expander(label=session_state['_']("View or edit system prompt"), expanded=False):
                    self._display_prompt_editor(prompt)

                st.markdown("""---""")

                prompt_to_use = utils.get_prompt_to_use(session_state['selected_chatbot_path_serialized'],
                                                        prompt)

                # Displays the existing conversation history
                conversation_history = utils.get_conversation_history(
                    session_state['selected_chatbot_path_serialized'])

                utils.display_conversation(conversation_history)

                # Handles the user's input and interaction with the LLM
                self._handle_user_input(prompt_to_use)
            else:
                st.error(session_state['_']("The System Prompt should be a string and not empty."))
