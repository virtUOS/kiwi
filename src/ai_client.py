import json
import os
import re

from openai import OpenAI

import streamlit as st
from streamlit import session_state


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

        # Add the history of the conversation, ignore the system prompt and images of past messages
        for speaker, message, __, __ in session_state['conversation_histories'][
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
