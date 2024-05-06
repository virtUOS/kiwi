import os

import streamlit as st
from streamlit import session_state

from openai import OpenAI

from typing import Any, Dict, List

from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, VectorStore
from langchain.docstore.document import Document


class AIClient:

    def __init__(self):
        if session_state['model_selection'] == 'OpenAI':

            @st.cache_resource
            def load_openai_data():
                return OpenAI(), os.getenv('OPENAI_MODEL'), os.getenv('OPENAI_API_KEY')

            self.client, self.model, self.openai_key = load_openai_data()

            @st.cache_resource
            def load_langchain_data(model_name):
                llm = ChatOpenAI(model_name=model_name, openai_api_key=os.getenv('OPENAI_API_KEY'))
                question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
                return llm, question_generator

            self.client_lc, self.question_generator = load_langchain_data(self.model)
        else:
            pass
            # Add here client initialization for local model
            # For example using OpenLLM as inference server
            # session_state["inference_server_url"] = "http://localhost:3000/v1"
            # self.client = OpenAI(base_url=session_state['inference_server_url'])
            # models = session_state["client"].models.list()
            # self.model = models.data[0].id

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
                yield chunk_content

    @staticmethod
    def _concatenate_partial_response(partial_response):
        """
        Concatenates the partial response into a single string.

        Parameters:
            partial_response (list): The chunks of the response from the OpenAI API.

        Returns:
            str: The concatenated response.
        """
        str_response = ""
        for i in partial_response:
            if isinstance(i, str):
                str_response += i

        st.markdown(str_response)

        return str_response

    def process_response(self, current_history, user_message, prompt_to_use):
        """
        Submits the user message to OpenAI, retrieves the response, and updates the conversation history.

        Parameters:
        - current_history: The current conversation history.
        - user_message: The message input by the user.
        - prompt_to_use: The current system prompt associated with the user message.

        Returns:
        - current_history: The updated conversation history with the response from the chatbot.
        """
        response = self._get_response(user_message, prompt_to_use)
        # Add AI response to the history
        current_history.append(('Assistant', response, ""))
        return current_history

    def _get_response(self, message, prompt_to_use):
        """
        Sends a prompt to the OpenAI API and returns the API's response.

        Parameters:
            message (str): The user's message or question.
            prompt_to_use (str): Additional context or instructions to provide to the model.

        Returns:
            str: The response from the chatbot.
        """
        try:
            # Prepare the full prompt and messages with context or instructions
            messages = self._prepare_full_prompt_and_messages(message, prompt_to_use)

            # Send the request to the OpenAI API
            # Display assistant response in chat message container
            response = ""
            with st.chat_message("assistant"):
                with st.spinner(session_state['_']("Generating response...")):
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=True,
                    )
                    partial_response = []

                    gen_stream = self._generate_response(stream)
                    for chunk_content in gen_stream:
                        # check if the chunk is a code block
                        if chunk_content == '```':
                            partial_response.append(chunk_content)
                            code_block = True
                            while code_block:
                                try:
                                    chunk_content = next(gen_stream)
                                    partial_response.append(chunk_content)
                                    if chunk_content == "`\n\n":
                                        code_block = False
                                        str_response = self._concatenate_partial_response(partial_response)
                                        partial_response = []
                                        response += str_response

                                except StopIteration:
                                    break

                        else:
                            # If the chunk is not a code block, append it to the partial response
                            partial_response.append(chunk_content)
                            if chunk_content:
                                if '\n' in chunk_content:
                                    str_response = self._concatenate_partial_response(partial_response)
                                    partial_response = []
                                    response += str_response
                # If there is a partial response left, concatenate it and render it
                if partial_response:
                    str_response = self._concatenate_partial_response(partial_response)
                    response += str_response

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

        if session_state["model_selection"] == 'OpenAI':
            messages = [{
                'role': "system",
                'content': description_to_use
            }]
        else:  # Add here conditions for different types of models
            messages = []

        # Add the history of the conversation
        for entry in session_state['conversation_histories'][
                session_state['selected_chatbot_path_serialized']]:
            speaker, message, *__ = entry + (None,)  # Ensure at least 3 elements
            role = 'user' if speaker == session_state['USER'] else 'assistant'
            messages.append({'role': role, 'content': message})

        # Building the messages with the "system" message based on expertise area
        if session_state["model_selection"] == 'OpenAI':
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

    def get_condensed_question(self, user_input: str, chat_history_tuples):
        """
        This function adds context to the user query, by combining it with the chat history
        """
        condensed_question = self.question_generator.predict(
            question=user_input, chat_history=_get_chat_history(chat_history_tuples)
        )
        return condensed_question

    @staticmethod
    def get_sources(vectorstore: VectorStore, query: str) -> List[Document]:
        """This function performs a similarity search between the user query and the
        indexed document, returning the five most relevant document chunks for the given query
        """
        docs = vectorstore.similarity_search(query, k=5)
        return docs

    def get_answer(self, prompt: str, docs: List[Document], query: str, title="") -> Dict[str, Any]:
        """Gets an answer to a question from a list of Documents."""
        doc_chain = load_qa_with_sources_chain(
            llm=self.client_lc,
            chain_type="stuff",
            prompt=prompt,
        )
        answer = doc_chain(
            {"input_documents": docs, "question": query, "title": title}, return_only_outputs=True
        )
        return answer

    def get_vectorstore(self, docs: List[Document]) -> VectorStore:
        """Given as input a document, this function returns the indexed version,
        leveraging the Chroma database"""
        ids = [doc.metadata["source"] for doc in docs]
        unique_ids = set(ids)

        if len(ids) != len(unique_ids):
            raise ValueError("Duplicate IDs found in document sources. Please ensure all IDs are unique.")

        vectorstore = Chroma.from_documents(
            docs,
            OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=self.openai_key),
            # Generate unique ids
            ids=[doc.metadata["source"] for doc in docs],
            collection_name=session_state['collection_name'],
        )
        return vectorstore

    @staticmethod
    def prepare_chunks_for_database(chunks: list) -> List[Document]:
        """Converts a list of chunk data to a list of Documents (chunks of fixed size)
        avoiding duplicates based on a unique source identifier (file name, page, and chunk number).

        Each Document includes the following metadata:
            - page number
            - chunk number
            - file name
            - coordinates (x0, x1, y0, y1)
            - source (unique identifier combining file name, page and chunk number)
        """

        doc_chunks = []
        seen_sources = set()  # Set to track the sources that have been added

        for i, chunk in enumerate(chunks):
            # Generate the source identifier
            source = f"{chunk['file_name'].split('.pdf')[0]}|{chunk['page']}-{chunk['chunk_number']}"

            # Check if this source has already been processed
            if source not in seen_sources:
                doc = Document(
                    page_content=chunk['text'],
                    metadata={'page': chunk['page'],
                              'chunk': chunk['chunk_number'],
                              'file': chunk['file_name'],
                              'x0': chunk['x0'],
                              'x1': chunk['x1'],
                              'y0': chunk['y0'],
                              'y1': chunk['y1'],
                              'source': source
                              }
                )
                doc_chunks.append(doc)
                seen_sources.add(source)  # Mark this source as seen

        return doc_chunks
