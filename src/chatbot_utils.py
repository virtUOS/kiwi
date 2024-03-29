import os
import streamlit as st
from streamlit import session_state
from src import menu_utils
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SidebarChatControls:

    def __init__(self):
        pass

    @staticmethod
    def initialize_chat_session_variables():
        """Initialize essential session variables."""
        required_keys = {
            'prompt_options_chatbot': menu_utils.load_prompts_from_yaml(typ='chat'),
            'edited_prompts': {},
        }

        for key, default_value in required_keys.items():
            if key not in session_state:
                session_state[key] = default_value


class ChatManager:

    def __init__(self, user, general_manager):
        self.gm = general_manager
        self.client = None
        session_state['USER'] = user

    def set_client(self, client):
        self.client = client

    @staticmethod
    def update_edited_prompt():
        session_state['edited_prompts'][session_state[
            'selected_chatbot_path_serialized']] = session_state['edited_prompt']

    @staticmethod
    def restore_prompt():
        if session_state['selected_chatbot_path_serialized'] in session_state['edited_prompts']:
            del session_state['edited_prompts'][session_state['selected_chatbot_path_serialized']]

    def _display_prompt_editor(self, description):
        """Allows editing of the chatbot prompt."""
        current_chatbot_path_serialized = session_state['selected_chatbot_path_serialized']
        current_edited_prompt = session_state['edited_prompts'].get(current_chatbot_path_serialized, description)

        st.text_area(session_state['_']("System prompt"),
                     value=current_edited_prompt,
                     on_change=self.update_edited_prompt,
                     key='edited_prompt',
                     label_visibility='hidden')

        st.button("🔄", help=session_state['_']("Restore Original Prompt"), on_click=self.restore_prompt)

    @staticmethod
    def _get_description_to_use(default_description):
        """Retrieves the prompt description to use, whether edited or default."""
        return session_state['edited_prompts'].get(session_state[
                                                       'selected_chatbot_path_serialized'], default_description)

    def _handle_user_input(self, description_to_use):
        """Handles user input, sending it to OpenAI and displaying the response."""
        if user_message := st.chat_input(session_state['USER']):

            # Get the current conversation history
            current_history = self.gm.get_conversation_history(session_state[
                                                                   'selected_chatbot_path_serialized'])

            # Prevent re-running the query on refresh or navigating back by checking
            # if the last stored message is not from the User or the history is empty
            if (not current_history or current_history[-1][0] != session_state['USER']
                    or current_history[-1][1] != user_message):

                # Print user message immediately after getting entered because we're streaming the chatbot output
                with st.chat_message("user"):
                    st.markdown(user_message)

                response = self.client.get_response(user_message, description_to_use)

                # Update the conversation history in the session state
                self.gm.add_conversation_entry(chatbot=session_state['selected_chatbot_path_serialized'],
                                               speaker='User',
                                               message=user_message)
                self.gm.add_conversation_entry(chatbot=session_state['selected_chatbot_path_serialized'],
                                               speaker='Assistant',
                                               message=response)
                st.rerun()

    def display_chat_interface(self):
        """Displays the chat interface and manages conversation display and input."""
        if 'selected_chatbot_path' in session_state and session_state["selected_chatbot_path"]:
            selected_chatbot_path = session_state["selected_chatbot_path"]

            if isinstance(menu_utils.get_final_description(selected_chatbot_path,
                                                           session_state["prompt_options_chatbot"]), str):

                description = menu_utils.get_final_description(selected_chatbot_path,
                                                               session_state["prompt_options_chatbot"])

                # Get the existing conversation history if there's any
                conversation_history = self.gm.get_conversation_history(
                    session_state['selected_chatbot_path_serialized'])

                # If there's no conversation history, display introduction message
                if not conversation_history:
                    st.header(session_state['_']("How can I help you?"))
                    if session_state['model_selection'] == 'OpenAI':
                        using_text = session_state['_']("You're using the following OpenAI model:")
                        remember_text = session_state['_']("Remember **not** to enter any personal information or "
                                                           "copyrighted material.")
                        st.write(f"{using_text} **{os.getenv('OPENAI_MODEL')}**. {remember_text}")
                        st.write(session_state['_']("Each time you enter information, "
                                                    "a system prompt is sent to the chat model by default."))
                    # Create a key for the new conversation history and initialize empty
                    self.gm.create_empty_history(session_state['selected_chatbot_path_serialized'])

                with st.expander(label=session_state['_']("View or edit system prompt"), expanded=False):
                    self._display_prompt_editor(description)

                st.markdown("""---""")

                description_to_use = self._get_description_to_use(description)

                self.gm.display_conversation(conversation_history)
                self._handle_user_input(description_to_use)
