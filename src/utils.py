import os
import pandas as pd
import streamlit as st
from streamlit import session_state
from streamlit_option_menu import option_menu
from streamlit_cookies_manager import EncryptedCookieManager
import extra_streamlit_components as stx

from typing import Any, Dict, List

from langchain.chains import LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, VectorStore
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src import menu_utils
from dotenv import load_dotenv

from src.language_utils import initialize_language

# Load environment variables
load_dotenv()


class SidebarManager:

    def __init__(self):
        # Set the language
        initialize_language()
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
            'conversation_histories': {},
            'selected_chatbot_path': [],
            'selected_chatbot_path_serialized': "",
            'chosen_tab_id': "1",
            'current_tab_id': "1",
        }

        for key, default_value in required_keys.items():
            if key not in session_state:
                session_state[key] = default_value

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
    def load_prompts(prompt_options, typ, prompt_key=None):
        language = st.query_params.get('lang', False)
        # Use German language as default
        if not language:
            language = "de"

        """Load chat prompts based on language."""
        session_state[prompt_options] = menu_utils.load_prompts_from_yaml(typ=typ,
                                                                          language=language,
                                                                          prompt_key=prompt_key)

    def logout_and_redirect(self):
        # Set cookies and session variables before redirecting.
        self.cookies["session"] = 'out'
        session_state["password_correct"] = False
        session_state['credentials_checked'] = False
        st.switch_page('start.py')

    def _display_chatbots_menu(self, options, path=[]):
        """Display sidebar menu for chatbot selection."""
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

    def display_general_sidebar_controls(self, display_chatbot_menu=True):
        """Display controls and settings in the sidebar."""

        # First load prompts
        self.load_prompts(prompt_options='prompt_options', typ='chat')

        if display_chatbot_menu:
            # Display the menu for chatbots
            self._display_chatbots_menu(session_state['prompt_options'])

            # Store the serialized path in session state if different from current path
            # Do this here to handle conversation controls showing up when changing
            # chatbot and the conversation exists
            if menu_utils.path_changed(session_state['selected_chatbot_path'], session_state[
                    'selected_chatbot_path_serialized']):
                session_state['selected_chatbot_path_serialized'] = '/'.join(
                    session_state['selected_chatbot_path'])  # Update the serialized path in session state

        with st.sidebar:
            if session_state['model_selection'] == 'OpenAI':
                model_text = session_state['_']("Model:")
                st.write(f"{model_text} {os.getenv('OPENAI_MODEL')}")

    def display_general_sidebar_bottom_controls(self):
        with st.sidebar:
            # Show the conversation controls only if there's a conversation
            if session_state['selected_chatbot_path_serialized'] in session_state[
                'conversation_histories'] and session_state['conversation_histories'][session_state[
                    'selected_chatbot_path_serialized']]:

                st.markdown("""---""")

                st.write(session_state['_']("**Options**"))
                col1, col2 = st.columns([1, 5])
                if col1.button("🗑️", help=session_state['_']("Delete the Conversation")):
                    # Clears the current chatbot's conversation history
                    session_state['conversation_histories'][session_state['selected_chatbot_path_serialized']] = [
                    ]
                    st.rerun()

                conversation_to_download = session_state['conversation_histories'][
                    session_state['selected_chatbot_path_serialized']]

                conversation_df = pd.DataFrame(conversation_to_download,
                                               columns=[session_state['_']('Speaker'), session_state['_']('Message')])
                conversation_csv = conversation_df.to_csv(index=False).encode('utf-8')
                col2.download_button("📂",
                                     data=conversation_csv,
                                     file_name="conversation.csv",
                                     mime="text/csv",
                                     help=session_state['_']("Download the Conversation"))

            st.markdown("""---""")

            if st.button(session_state['_']('Logout')):
                self.logout_and_redirect()

            st.write(f"Version: *Beta*")


class GeneralManager:

    def __init__(self):
        pass

    @staticmethod
    def display_pages_tabs(default_id):

        st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)

        page_text = "Chatbot 🤖"
        if session_state['current_tab_id'] == '2':
            page_text = "Chat with Documents 📖"
        elif session_state['current_tab_id'] == '3':
            page_text = "Chat with Videos 📹"

        with st.expander(f"**{page_text}**"):
            session_state['chosen_tab_id'] = stx.tab_bar(data=[
                stx.TabBarItemData(id=1, title="Chatbot", description=""),
                stx.TabBarItemData(id=2, title="Chat with Documents", description=""),
                stx.TabBarItemData(id=3, title="Chat with Videos", description=""),
            ], default=default_id)

        if 'current_tab_id' not in session_state:
            session_state['current_tab_id'] = session_state['chosen_tab_id']
        elif session_state['chosen_tab_id'] == '1' and session_state['current_tab_id'] != '1':
            session_state['current_tab_id'] = '1'
            st.switch_page("pages/chatbot_app.py")
        elif session_state['chosen_tab_id'] == '2' and session_state['current_tab_id'] != '2':
            session_state['current_tab_id'] = '2'
            st.switch_page("pages/docs_app.py")
        elif session_state['chosen_tab_id'] == '3' and session_state['current_tab_id'] != '3':
            session_state['current_tab_id'] = '3'
            st.switch_page("pages/videos_app.py")

    @staticmethod
    def add_conversation_entry(chatbot, speaker, message):
        session_state['conversation_histories'][chatbot].append((speaker, message))

    @staticmethod
    def get_conversation_history(chatbot):
        return session_state['conversation_histories'].get(chatbot, [])

    @staticmethod
    def has_conversation_history(chatbot):
        if chatbot == 'docs':
            if (chatbot in session_state['conversation_histories']
                    and session_state['uploaded_pdf_files']):
                return True
        return False

    @staticmethod
    def create_empty_history(chatbot):
        # If no conversation in history for this chatbot, then create an empty one
        session_state['conversation_histories'][chatbot] = [
        ]

    @staticmethod
    def display_conversation(conversation_history, container=None):
        """Displays the conversation history for the selected path."""
        if conversation_history:
            for speaker, message in conversation_history:
                if speaker == session_state['USER']:
                    if container:
                        with container:
                            st.chat_message("user").write(message)
                    else:
                        st.chat_message("user").write(message)
                else:
                    if container:
                        with container:
                            st.chat_message("assistant").write(message)
                    else:
                        st.chat_message("assistant").write(message)

    @staticmethod
    def change_chatbot_style():
        # Set style of chat input so that it shows up at the bottom of the column
        chat_input_style = f"""
        <style>
            .stChatInput {{
              position: fixed;
              bottom: 3rem;
            }}
        </style>
        """
        st.markdown(chat_input_style, unsafe_allow_html=True)

    @staticmethod
    def generate_chat_history_tuples(conversation):
        # Generate tuples with the chat history
        # We iterate with a step of 2 and always take
        # the current and next item assuming USER follows by Assistant
        if conversation:
            chat_history_tuples = [
                (conversation[i][1], conversation[i + 1][1])
                # Extract just the messages, ignoring the labels
                for i in range(0, len(conversation) - 1, 2)
                # We use len(conversation)-1 to avoid out-of-range errors
            ]
            return chat_history_tuples
        return []


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
                return question_generator

            self.question_generator = load_langchain_data(self.model)
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
        for speaker, message in session_state['conversation_histories'][
                session_state['selected_chatbot_path_serialized']]:
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
            llm=self.client,
            chain_type="stuff",
            prompt=prompt,
        )
        answer = doc_chain(
            {"input_documents": docs, "question": query, "title": title}, return_only_outputs=True
        )
        return answer

    def get_embeddings(self, docs: List[Document]) -> VectorStore:
        """Given as input a document, this function returns the indexed version,
        leveraging the Chroma database"""
        vectorstore = Chroma.from_documents(
            docs,
            OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=self.openai_key),
            # Generate unique ids
            ids=[doc.metadata["source"] for doc in docs]
        )
        return vectorstore

    @staticmethod
    def text_split(text: str, filename: str, chunk_size=300) -> List[Document]:
        """Converts a string to a list of Documents (chunks of fixed size)
        and returns this along with the following metadata:
            - page number
            - chunk number
            - source (concatenation of page & chunk number)
        """
        if 'chunk_size' not in session_state:
            session_state['chunk_size'] = chunk_size
        if isinstance(text, str):
            text = [text]
        page_docs = [Document(page_content=page) for page in text]

        # Add page numbers as metadata
        for i, doc in enumerate(page_docs):
            doc.metadata["page"] = i + 1

        doc_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=session_state['chunk_size'],
            chunk_overlap=session_state['chunk_size'] * 0.05,  # Overlap is 5% of the chunk size
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )

        for doc in page_docs:
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i, "file": filename}
                )
                # Add sources a metadata
                doc.metadata["source"] = (f"{doc.metadata['file'].split('.pdf')[0]}|"
                                          f"{doc.metadata['page']}-{doc.metadata['chunk']}")
                doc_chunks.append(doc)

        return doc_chunks





