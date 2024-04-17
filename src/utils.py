import os
import re
import streamlit as st
from streamlit import session_state
from streamlit_option_menu import option_menu
from streamlit_cookies_manager import EncryptedCookieManager
# import extra_streamlit_components as stx
from streamlit_navigation_bar import st_navbar

from typing import Any, Dict, List

from openai import OpenAI
from langchain.chains import LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, VectorStore
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from src import menu_utils
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SidebarManager:

    def __init__(self):
        self.cookies = None

    @staticmethod
    def set_user(user):
        session_state['USER'] = user

    @staticmethod
    def get_language():
        return st.query_params.get('lang', False)

    def initialize_cookies(self):
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

        self.cookies = cookies

    def verify_user_session(self):
        """
        Verify the user's session validity. If cookies indicate the session is not yet initiated,
        then redirect the user to the start page.

        This function ensures that unauthorized users are not able to access application sections
        that require a valid session.prompt_op
        """
        if self.cookies and self.cookies.get("session") != 'in':
            st.switch_page("start.py")

    @staticmethod
    def initialize_session_variables():
        """
        Initialize essential session variables with default values.

        Sets up the initial state for model selection, chatbot paths, conversation histories,
        prompt options, etc., essential for the application to function correctly from the start.
        """
        required_keys = {
            'model_selection': "OpenAI",
            'conversation_histories': {},
            'selected_chatbot_path': [],
            'selected_chatbot_path_serialized': "",
            'chosen_page_index': "1",
            'current_page_index': "1",
            'typ': "chat",
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

    def load_prompts(self):
        """
        Load chat prompts based on the selected or default language.

        This method enables dynamic loading of chat prompts to support multi-lingual chatbot interactions.
        If no language is specified in the query parameters, German ('de') is used as the default language.
        """
        language = self.get_language()
        # Use German language as default
        if not language:
            language = "de"

        session_state[f"prompt_options_{session_state['typ']}"] = menu_utils.load_prompts_from_yaml(
            typ=session_state['typ'], language=language)

    def logout_and_redirect(self):
        """
        Perform logout operations and redirect the user to the start page.

        This involves resetting cookies and session variables that signify the user's logged-in state,
        thereby securely logging out the user.
        """
        self.cookies["session"] = 'out'
        session_state["password_correct"] = False
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

    def load_and_display_prompts(self):
        """
        Load chat prompts from the session state and display them in the chatbots menu in the sidebar.

        This method first loads the prompts by calling the `load_prompts` method and
        then displays the chatbots menu with the loaded prompts using the `_display_chatbots_menu` method.
        """
        self.load_prompts()
        self._display_chatbots_menu(session_state[f"prompt_options_{session_state['typ']}"])

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

    @staticmethod
    def _display_model_information():
        """
        Display OpenAI model information in the sidebar if the OpenAI model is the current selection.

        In the sidebar, this static method shows the model name and version pulled from environment variables
        if 'OpenAI' is selected as the model in the session state. The model information helps users
        identify the active model configuration.
        """
        with st.sidebar:
            if session_state['model_selection'] == 'OpenAI':
                model_text = session_state['_']("Model:")
                st.write(f"{model_text} {os.getenv('OPENAI_MODEL')}")

    def display_general_sidebar_controls(self):
        """
        Display sidebar controls and settings for the page.

        This function performs the following steps:
        1. Checks for changes in the selected chatbot path to update the session state accordingly.
        2. Displays model information for the selected model, e.g., OpenAI.
        """
        self._update_path_in_session_state()

        self._display_model_information()

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
                ... (rest of CSS)
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
            st.write(f"Version: *Beta*")

    def display_general_sidebar_bottom_controls(self):
        """
        Display bottom sidebar controls and settings for page.

        This function performs the following steps:
        1. Adds custom CSS to stylize the sidebar.
        2. Offers a logout option.
        """
        self._add_custom_css()

        self._logout_option()


class GeneralManager:

    def __init__(self, sidebar_general_manager):
        self.sgm = sidebar_general_manager

    def display_pages_tabs(self, default_id):

        # We need to do this because we can't initialize language earlier due to affecting the navbar positioning
        chat_with_documents_menu_name = "Chat mit Dokumenten 📖"
        current_language = self.sgm.get_language()
        if current_language:
            if current_language == 'en':
                chat_with_documents_menu_name = "Chat with Documents 📖"

        pages = ["Chatbot 🤖", chat_with_documents_menu_name]

        page = st_navbar(pages, selected=pages[default_id], options={"use_padding": False})

        session_state['chosen_page_index'] = pages.index(page)

        if session_state['chosen_page_index'] == session_state['current_page_index']:
            session_state['current_page_index'] = session_state['chosen_page_index']
        elif session_state['chosen_page_index'] == 0 and session_state['current_page_index'] != 0:
            session_state['current_page_index'] = 0
            session_state['typ'] = 'chat'
            st.switch_page("pages/chatbot_app.py")
        elif session_state['chosen_page_index'] == 1 and session_state['current_page_index'] != 1:
            session_state['current_page_index'] = 1
            session_state['typ'] = 'docs'
            st.switch_page("pages/docs_app.py")

    @staticmethod
    def update_edited_prompt(chatbot):
        """
        Updates the edited prompt in the session state to reflect changes made by the user.

        Parameters:
        - chatbot: The chatbot for which the prompt will be updated.
        """
        session_state['edited_prompts'][chatbot] = session_state['edited_prompt']

    @staticmethod
    def restore_prompt(chatbot):
        """
        Restores the original chatbot prompt by removing any user-made edits from the session state.

        Parameters:
        - chatbot: The chatbot for which the prompt will be restored.
        """
        if chatbot in session_state['edited_prompts']:
            del session_state['edited_prompts'][chatbot]

    @staticmethod
    def fetch_chatbot_prompt(chatbot):
        """
        Fetches the prompt for the current chatbot based on the selected path.

        Parameters:
        - chatbot: The chatbot for which the prompt will be fetched.

        Returns:
        - A string containing the prompt of the current chatbot.
        """
        return menu_utils.get_final_prompt(chatbot,
                                           session_state[f"prompt_options_{session_state['typ']}"])

    @staticmethod
    def get_prompt_to_use(chatbot, description):
        """
        Retrieves and returns the current prompt description for the chatbot, considering any user edits.

        Parameters:
        - chatbot: The chatbot for which the prompt will be returned.
        - default_description: The default description provided for the chatbot's prompt.

        Returns:
        - The current (possibly edited) description to be used as the prompt.
        """
        return session_state['edited_prompts'].get(chatbot, description)

    @staticmethod
    def update_conversation_history(chatbot, current_history):
        """
        Updates the session state with the current conversation history.

        This method ensures that the conversation history for the currently selected chatbot path in the session state
        is updated to reflect any new messages or responses that have been added during the interaction.

        Parameters:
        - chatbot: The chatbot for which the conversation will be updated.
        - current_history: The current conversation history, which is a list of tuples. Each tuple contains
          the speaker ('USER' or 'Assistant'), the message, and optionally, the system description or prompt
          accompanying user messages (for user messages only).
        """
        session_state['conversation_histories'][chatbot] = current_history

    @staticmethod
    def add_conversation_entry(chatbot, speaker, message, prompt=""):
        session_state['conversation_histories'][chatbot].append((speaker, message, prompt))
        return session_state['conversation_histories'][chatbot]

    @staticmethod
    def get_conversation_history(chatbot):
        return session_state['conversation_histories'].get(chatbot, [])

    @staticmethod
    def has_conversation_history(chatbot):
        if chatbot == 'docs':
            if (chatbot in session_state['conversation_histories']
                    and session_state['uploaded_pdf_files']):
                return True
        else:
            if (chatbot in session_state['conversation_histories']
                    and session_state['conversation_histories'][chatbot]):
                return True
        return False

    @staticmethod
    def is_new_message(current_history, user_message):
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
    def create_history(chatbot):
        # If no conversation in history for this chatbot, then create an empty one
        session_state['conversation_histories'][chatbot] = [
        ]

    @staticmethod
    def display_conversation(conversation_history, container=None):
        """
        Displays the conversation history between the user and the assistant within the given container or globally.

        Parameters:
        - conversation_history: A list containing tuples of (speaker, message, system prompt)
         representing the conversation. The system prompt is optional.
        - container: The Streamlit container (e.g., column, expander) where the messages should be displayed.
          If None, messages will be displayed in the main app area.
        """
        chat_message_container = container if container else st
        for entry in conversation_history:
            speaker, message, *__ = entry + (None,)  # Ensure at least 3 elements
            if speaker == session_state['USER']:
                chat_message_container.chat_message("user").write(message)
            elif speaker == "Assistant":
                chat_message_container.chat_message("assistant").write(message)

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

    @staticmethod
    def process_ai_response_for_suggestion_queries(model_output):
        # Ensure model_output is a single string for processing
        model_output_string = ' '.join(model_output) if isinstance(model_output, list) else model_output

        # Remove timestamps
        clean_output = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3}', '', model_output_string)

        # Removing content inside square brackets including the brackets themselves
        clean_output = re.sub(r'\[.*?\]', '', clean_output)

        # Splitting based on '?' to capture questions, adding '?' back to each split segment
        queries = [query + '?' for query in re.split(r'\?+\s*', clean_output) if query.strip()]

        cleaned_queries = []
        for query in queries:
            # Removing anything that is not a letter at the start of each query
            query_clean = re.sub(r'^[^A-Za-z]+', '', query)  # Adjust regex as needed
            # Attempt to more robustly remove quotes around the question
            query_clean = re.sub(r'(^\s*"\s*)|(\s*"$)', '', query_clean)  # Improved to target quotes more effectively
            # Remove numerical patterns and excessive whitespace
            query_clean = re.sub(r'\s*\d+[.)>\-]+\s*$', '', query_clean).strip()
            # Replace multiple spaces with a single space
            query_clean = re.sub(r'\s+', ' ', query_clean)
            if query_clean:
                cleaned_queries.append(query_clean)

        # Sort queries by length (descending) to prefer longer queries and select the first three
        if len(cleaned_queries) > 3:
            cleaned_queries = sorted(cleaned_queries, key=len, reverse=True)[:3]
        elif len(cleaned_queries) < 3:
            cleaned_queries = []

        return cleaned_queries


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

    def get_vectorstore(self, docs: List[Document], collection=None) -> VectorStore:
        """Given as input a document, this function returns the indexed version,
        leveraging the Chroma database"""
        vectorstore = Chroma.from_documents(
            docs,
            OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=self.openai_key),
            # Generate unique ids
            ids=[doc.metadata["source"] for doc in docs],
            collection_name=collection,
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


class SummarizationManager:

    def __init__(self):
        pass

    @staticmethod
    def _generate_prompt_from_template(template):
        return PromptTemplate.from_template(template)

    @staticmethod
    def _text_splitter(docs):
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=0
        )
        split_docs = text_splitter.split_documents(docs)
        return split_docs

    @staticmethod
    def _create_reduce_documents_chain(llm, reduce_prompt):
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        # Combines and iteratively reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=4000,
        )

        return reduce_documents_chain

    @staticmethod
    def _create_map_reduce_documents_chain(llm, map_prompt, reduce_documents_chain):
        map_chain = LLMChain(llm=llm, prompt=map_prompt)

        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )

        return map_reduce_chain

    @staticmethod
    def _create_map_reduce_documents_chain_with_intermediate_steps(llm, map_prompt, reduce_documents_chain):
        map_chain = LLMChain(llm=llm, prompt=map_prompt)

        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=True,
        )

        return map_reduce_chain

    @staticmethod
    def _turn_text_input_to_documents(text_input):
        docs_list = []
        for inp in text_input:
            docs_list.append(Document(inp))
        return docs_list

    def get_data_from_documents(self, docs, model_name, map_prompt_template, reduce_prompt_template):
        llm = ChatOpenAI(temperature=0, model_name=model_name)

        map_prompt = self._generate_prompt_from_template(map_prompt_template)
        reduce_prompt = self._generate_prompt_from_template(reduce_prompt_template)

        reduce_documents_chain = self._create_reduce_documents_chain(llm, reduce_prompt)

        map_reduce_chain = self._create_map_reduce_documents_chain(llm, map_prompt, reduce_documents_chain)

        docs_list = self._turn_text_input_to_documents(docs)
        split_docs = self._text_splitter(docs_list)

        response = map_reduce_chain.invoke(split_docs)

        return response['output_text']

    def get_data_from_video(self,
                            docs,
                            model_name,
                            map_prompt_template,
                            reduce_summary_prompt_template,
                            reduce_topics_prompt_template):
        llm = ChatOpenAI(temperature=0, model_name=model_name)

        map_prompt = self._generate_prompt_from_template(map_prompt_template)
        reduce_summary_prompt = self._generate_prompt_from_template(reduce_summary_prompt_template)
        reduce_topics_prompt = self._generate_prompt_from_template(reduce_topics_prompt_template)

        reduce_summary_chain = self._create_reduce_documents_chain(llm, reduce_summary_prompt)
        reduce_topics_chain = self._create_reduce_documents_chain(llm, reduce_topics_prompt)

        map_summary_chain = self._create_map_reduce_documents_chain_with_intermediate_steps(llm,
                                                                                            map_prompt,
                                                                                            reduce_summary_chain)
        # map_topics_chain = self._create_map_reduce_documents_chain(llm, map_prompt, reduce_topics_chain)

        docs_list = self._turn_text_input_to_documents(docs)
        split_docs = self._text_splitter(docs_list)

        summary = map_summary_chain.invoke(split_docs)
        list_of_intermediate_documents = [Document(topic) for topic in summary['intermediate_steps']]
        topics = reduce_topics_chain.invoke(list_of_intermediate_documents)

        return summary['output_text'], topics['output_text']
