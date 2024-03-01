import os
import base64
import re
import pandas as pd

from dotenv import load_dotenv
from io import BytesIO
from pypdf import PdfReader
import fitz
from pdf2image import convert_from_bytes
from PIL import Image

from typing import Any, Dict, List

import streamlit as st
from streamlit import session_state
from streamlit_option_menu import option_menu
from streamlit_cookies_manager import EncryptedCookieManager
from streamlit_pdf_viewer import pdf_viewer
from streamlit_dimensions import st_dimensions

from src import menu_utils
from src import document_prompt

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, VectorStore
from langchain.chains import LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from src.document_prompt import STUFF_PROMPT

# Load environment variables
load_dotenv()


class SidebarManager:

    def __init__(self):
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
            'selected_chatbot_path': [],
            'conversation_histories': {},
            'selected_chatbot_path_serialized': "",
            'prompt_options': menu_utils.load_prompts_from_yaml(),
            'edited_prompts': {},
            'disable_custom': False,
            'uploaded_pdf_files': []
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
    def load_prompts():
        language = st.query_params.get('lang', False)
        # Use German language as default
        if not language:
            language = "de"

        """Load chat prompts based on language."""
        session_state["prompt_options"] = menu_utils.load_prompts_from_yaml(language=language)

    @staticmethod
    def check_amount_of_uploaded_files_and_set_variables():
        max_files = 5
        session_state['uploaded_pdf_files'] = []
        session_state['uploaded_pdf_files'] = []

        if len(session_state['pdf_files']) > max_files:
            st.sidebar.warning(f"Maximum number of files reached. Only the first {max_files} files will be processed.")

        # If there are uploaded files then use class list variable to store them
        if session_state['pdf_files']:
            for i, file in enumerate(session_state['pdf_files']):
                # Only append the maximum number of files allowed
                if i < max_files:
                    session_state['uploaded_pdf_files'].append(file)
                else:
                    break

    def _display_chatbots_menu(self, options, path=[]):
        """Display sidebar menu for chatbot selection."""
        if isinstance(options, dict) and options:  # Verify options is a dictionary and not empty
            next_level = list(options.keys())

            with st.sidebar:
                choice = option_menu(session_state['_']("Chat Menu"), next_level,
                                     icons=['chat-dots'] * len(next_level),
                                     menu_icon="cast",
                                     default_index=0)

            if choice:
                new_path = path + [choice]
                session_state['selected_chatbot_path'] = new_path
                self._display_chatbots_menu(options.get(choice, {}), new_path)

    def display_sidebar_controls(self):
        """Display controls and settings in the sidebar."""

        # First load prompts
        self.load_prompts()

        # Display the menu for chatbots
        self._display_chatbots_menu(session_state['prompt_options'])

        # Store the serialized path in session state if different from current path
        # Do this here to handle conversation controls showing up when changing
        # chatbot and the conversation exists
        if menu_utils.path_changed(session_state['selected_chatbot_path'],
                                   session_state['selected_chatbot_path_serialized']):
            session_state['selected_chatbot_path_serialized'] = '/'.join(
                session_state['selected_chatbot_path'])  # Update the serialized path in session state

        # Custom prompt selection
        with st.sidebar:
            if session_state['model_selection'] == 'OpenAI':
                model_text = session_state['_']("Model:")
                st.write(f"{model_text} {os.getenv('OPENAI_MODEL')}")

            # Show the conversation controls only if there's a conversation
            if session_state['selected_chatbot_path_serialized'] in session_state['conversation_histories'] and \
                    session_state[
                        'conversation_histories'][session_state['selected_chatbot_path_serialized']]:

                st.markdown("""---""")

                st.write(session_state['_']("**Conversation Controls**"))
                col1, col2 = st.columns([1, 5])
                if col1.button("🗑️", help=session_state['_']("Delete the Current Conversation")):
                    # Clears the current chatbot's conversation history
                    session_state['conversation_histories'][session_state['selected_chatbot_path_serialized']] = [
                    ]
                    st.rerun()

                conversation_to_download = session_state['conversation_histories'][
                    session_state['selected_chatbot_path_serialized']]

                conversation_df = pd.DataFrame(conversation_to_download,
                                               columns=[session_state['_']('Speaker'), session_state['_']('Message')])
                conversation_csv = conversation_df.to_csv(index=False).encode('utf-8')
                col2.download_button("💽",
                                     data=conversation_csv,
                                     file_name="conversation.csv",
                                     mime="text/csv",
                                     help=session_state['_']("Download the Current Conversation"))

            color = st.get_option('theme.secondaryBackgroundColor')

            css = f'''
            [data-testid="stSidebarNav"] {{
                position:absolute;
                bottom: 0;
                z-index: 1;
                background: {color};
            }}
            [data-testid="stSidebarNav"] > ul {{
                padding-top: 2rem;
            }}
            [data-testid="stSidebarNav"] > div {{
                position:absolute;
                top: 0;
            }}
            [data-testid="stSidebarNav"] > div > svg {{
                transform: rotate(180deg) !important;
            }}
            [data-testid="stSidebarNav"] + div {{
                overflow: scroll;
                max-height: 66vh;
            }}
            '''

            st.markdown("""---""")

            st.file_uploader("Upload PDF file", type=['pdf'], accept_multiple_files=True,
                             key='pdf_files', on_change=self.check_amount_of_uploaded_files_and_set_variables)

            st.markdown("""---""")

            if st.button(session_state['_']('Logout')):
                # Set cookies and password session variables before redirecting.
                # Delete credentials session variable
                self.cookies["session"] = 'out'
                session_state["password_correct"] = False
                del session_state['credentials_checked']
                st.switch_page('start.py')

            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


class DocsManager:

    def __init__(self, user):
        self.client = None
        session_state['USER'] = user
        # Columns to display pdf and chat
        self.column_uploaded_files, self.column_pdf, self.column_chat = st.columns([0.10, 0.425, 0.425],
                                                                                   gap="medium")
        # The vector store
        self.vector_store = None
        # Current document
        self.doc_binary_data = None
        # Annotation for current documents
        self.annotations = []
        # Empty container to display pdf
        self.empty_container = st.empty()
        # Get the screen width of the device to display thumbnails
        self.main_dim = st_dimensions(key="main")

    def set_client(self, client):
        self.client = client

    @staticmethod
    def update_edited_prompt():
        session_state['edited_prompts'][session_state['selected_chatbot_path_serialized']] = session_state[
            'edited_prompt']

    def _display_prompt_editor(self, description):
        """Allows editing of the chatbot prompt."""
        current_chatbot_path_serialized = session_state['selected_chatbot_path_serialized']
        current_edited_prompt = session_state['edited_prompts'].get(current_chatbot_path_serialized, description)

        st.text_area(session_state['_']("Edit Prompt"),
                     value=current_edited_prompt,
                     help=session_state['_']("Edit the system prompt for more customized responses."),
                     on_change=self.update_edited_prompt,
                     key='edited_prompt')

        if st.button("🔄", help=session_state['_']("Restore Original Prompt")):
            if current_chatbot_path_serialized in session_state['edited_prompts']:
                del session_state['edited_prompts'][current_chatbot_path_serialized]
                st.rerun()

    @staticmethod
    def _get_description_to_use(default_description):
        """Retrieves the prompt description to use, whether edited or default."""
        return session_state['edited_prompts'].get(session_state['selected_chatbot_path_serialized'],
                                                   default_description)

    @staticmethod
    def _display_conversation():
        """Displays the conversation history for the selected path."""
        conversation_history = session_state['conversation_histories'].get(
            session_state['selected_chatbot_path_serialized'], [])

        if conversation_history:
            for speaker, message in conversation_history:
                if speaker == session_state['USER']:
                    st.chat_message("user").write(message)
                else:
                    st.chat_message("assistant").write(message)

    def _handle_user_input(self, description_to_use):
        """Handles user input, sending it to OpenAI and displaying the response."""
        if user_message := st.chat_input(session_state['USER']):
            # Ensure there's a key for this conversation in the history
            if session_state['selected_chatbot_path_serialized'] not in session_state['conversation_histories']:
                session_state['conversation_histories'][session_state['selected_chatbot_path_serialized']] = []

            # Get the current conversation history
            current_history = session_state['conversation_histories'][session_state['selected_chatbot_path_serialized']]

            # Prevent re-running the query on refresh or navigating back by checking
            # if the last stored message is not from the User or the history is empty
            if not current_history or current_history[-1][0] != session_state['USER'] or current_history[-1][
                1] != user_message:
                # Add user message to the conversation
                current_history.append((session_state['USER'], user_message))

                # Print user message immediately after getting entered because we're streaming the chatbot output
                with st.chat_message("user"):
                    st.markdown(user_message)

                response = self.client.get_response(user_message, description_to_use)

                # Add AI response to the conversation
                current_history.append(('Assistant', response))

                # Update the conversation history in the session state
                session_state['conversation_histories'][
                    session_state['selected_chatbot_path_serialized']] = current_history

                st.rerun()

    @staticmethod
    def parse_pdf(file: BytesIO) -> List[str]:
        """PDF parser"""

        pdf = PdfReader(file)
        output = []
        for page in pdf.pages:
            text = page.extract_text()
            # Merge hyphenated words
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
            # Fix newlines in the middle of sentences
            text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
            # Remove multiple newlines
            text = re.sub(r"\n\s*\n", "\n\n", text)

            output.append(text)

        return output

    @staticmethod
    def text_split(text: str, filename: str) -> List[Document]:
        """Converts a string to a list of Documents (chunks of fixed size)
        and returns this along with the following metadata:
            - page number
            - chunk number
            - source (concatenation of page & chunk number)
        """

        if isinstance(text, str):
            text = [text]
        page_docs = [Document(page_content=page) for page in text]

        # Add page numbers as metadata
        for i, doc in enumerate(page_docs):
            doc.metadata["page"] = i + 1

        doc_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
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
                doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
                doc_chunks.append(doc)

        return doc_chunks

    @staticmethod
    def get_embeddings(docs: List[Document]) -> VectorStore:
        """Given as input a document, this function returns the indexed version,
        leveraging the Chroma database"""

        for doc in docs:
            print(doc.metadata)

        vectorstore = Chroma.from_documents(
            docs,
            OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY')),
            ids=[doc.metadata["source"] for doc in docs]
        )

        return vectorstore

    @staticmethod
    def get_condensed_question(user_input: str, chat_history_tuples, model_name: str):
        """
        This function adds context to the user query, by combining it with the chat history
        """

        llm = ChatOpenAI(model_name=model_name, openai_api_key=os.getenv('OPENAI_API_KEY'))
        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

        condensed_question = question_generator.predict(
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

    @staticmethod
    def get_answer(docs: List[Document], query: str) -> Dict[str, Any]:
        """Gets an answer to a question from a list of Documents."""

        llm = OpenAI()
        doc_chain = load_qa_with_sources_chain(
            llm=llm,
            chain_type="stuff",
            prompt=STUFF_PROMPT,
        )

        answer = doc_chain(
            {"input_documents": docs, "question": query}, return_only_outputs=True
        )

        return answer

    def display_thumbnails(self):

        # Get dimensions of thumbnails relative to screen size
        thumbnail_dim = int(self.main_dim['width'] * 0.05)
        thumbnail_size = (thumbnail_dim, thumbnail_dim)

        # Avoid repeating files by adding them to a set
        files_set = {file.name for file in session_state['uploaded_pdf_files']}

        # To display images horizontally, calculate the number of columns
        num_files = len(files_set)

        if num_files > 0:
            with self.column_uploaded_files:
                st.write(f"**{num_files} files uploaded**")

            thumbnail_images = {}
            for file in session_state['uploaded_pdf_files']:
                filename = file.name

                # Do this in case the user uploads
                # more files to avoid repeated images
                if filename not in thumbnail_images.keys():

                    # Make sure the file pointer is at the beginning
                    file.seek(0)
                    file_bytes = file.read()  # Read the file content into memory

                    # Convert from bytes directly, handling the byte stream. Take the first image
                    image = convert_from_bytes(file_bytes, first_page=1, last_page=1)[0]

                    if image:  # Check if any images were successfully created
                        # Resize image to thumbnail size
                        thumbnail = image.resize(thumbnail_size, Image.Resampling.BILINEAR)
                        # Use the first image as a thumbnail
                        thumbnail_images[filename] = thumbnail
                        with self.column_uploaded_files:
                            st.image(thumbnail)
                            st.write(f"File name: {filename}")

    def display_pdf(self):
        with self.column_pdf:
            pdf_viewer(input=self.doc_binary_data, annotations=self.annotations, height=1000,
                       width=int(self.main_dim['width'] * 0.4))

    def load_doc_to_display(self):

        self.container_pdf = st.container()

        if session_state['uploaded_pdf_files']:
            self.display_thumbnails()

            self.doc_binary_data = session_state['uploaded_pdf_files'][0].getvalue()

            filename = session_state['uploaded_pdf_files'][0].name

            base64_pdf = base64.b64encode(self.doc_binary_data).decode('utf-8')

            # self.doc_binary_data = self.doc_binary_data.decode('utf-8')

            # Embed PDF in HTML
            pdf_display = (F'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                           F'width="100%" height="1000" type="application/pdf"></iframe>')

            # Display file
            # pdf_container.markdown(pdf_display, unsafe_allow_html=True)

            # self.display_pdf()

            # Parse the pdf
            data_dict = self.parse_pdf(session_state['uploaded_pdf_files'][0])

            # Get text data
            text = self.text_split(data_dict, filename)

            self.vector_store = self.get_embeddings(text)

    @staticmethod
    def get_doc(binary_data):
        if binary_data:
            doc = fitz.open(filetype='pdf', stream=binary_data)
        else:
            doc = None
        return doc

    def display_chat_interface(self):
        # Variable to store the current conversation and simplify some code
        conversation = []

        # If there's already a conversation in the history for this chatbot, display it.
        if (session_state['selected_chatbot_path_serialized'] in session_state['conversation_histories']
                and session_state['uploaded_pdf_files']):
            conversation = session_state['conversation_histories'][session_state['selected_chatbot_path_serialized']]
            # Display conversation
            for speaker, message in conversation:

                with self.column_chat:
                    if speaker == session_state['USER']:
                        st.chat_message("user").write(message)
                    else:
                        st.chat_message("assistant").write(message)
        else:
            # If no conversation in history for this chatbot, then create an empty one
            session_state['conversation_histories'][session_state['selected_chatbot_path_serialized']] = [
            ]

        if session_state['selected_chatbot_path']:

            # Store the serialized path in session state if different from current path
            if menu_utils.path_changed(session_state['selected_chatbot_path'],
                                       session_state['selected_chatbot_path_serialized']):
                session_state['selected_chatbot_path_serialized'] = '/'.join(
                    session_state['selected_chatbot_path'])  # Update the serialized path in session state

            doc_prompt = document_prompt.STUFF_PROMPT
            # Checking if the prompt exists
            if doc_prompt:

                with self.column_chat:
                    # Interface to chat with selected expert
                    st.title(f"What do you want to know about your documents? :robot_face:")

                with self.column_chat:

                    # container_chat = self.column_chat.container(border=True)

                    # with container_chat:

                    sources_to_highlight = []
                    # Accept user input
                    if user_message := st.chat_input("Ask something about the PDFs",
                                                     disabled=not session_state['uploaded_pdf_files']):

                        # Prevent re-running the query on refresh or navigating back to the page by checking
                        # if the last stored message is not from the User
                        if not conversation or conversation[-1] != (session_state['USER'], user_message):
                            # Add user message to the conversation
                            session_state['conversation_histories'][session_state[
                                'selected_chatbot_path_serialized']].append((session_state['USER'], user_message))

                            # Print user message immediately after
                            # getting entered because we're streaming the chatbot output
                            with st.chat_message("user"):
                                st.markdown(user_message)

                            # Generate tuples with the chat history
                            # We iterate with a step of 2 and always take
                            # the current and next item assuming USER follows by Assistant
                            chat_history_tuples = [
                                (conversation[i][1], conversation[i + 1][1])
                                # Extract just the messages, ignoring the labels
                                for i in range(0, len(conversation) - 1, 2)
                                # We use len(conversation)-1 to avoid out-of-range errors
                            ]

                            doc = self.get_doc(self.doc_binary_data)
                            with st.spinner():
                                condensed_question = self.get_condensed_question(
                                    user_message, chat_history_tuples, os.getenv('OPENAI_MODEL')
                                )

                                sources = self.get_sources(self.vector_store, condensed_question)
                                all_output = self.get_answer(sources, condensed_question)

                                ai_response = all_output['output_text']

                                # Print user message immediately after getting entered
                                # because we're streaming the chatbot output
                                with st.chat_message("assistant"):
                                    st.markdown(ai_response)
                                    sources_to_display = []
                                    sources_to_highlight = []
                                    for source in sources:
                                        # Collect the references to relevant sources and their first 150 characters
                                        source_reference = source.metadata['source']
                                        # Only add it if the source appears referenced in the response
                                        if source_reference in ai_response:
                                            content = source.page_content
                                            sources_to_display.append(f"{source_reference} - {content[:150]}")
                                            page_number = int(source_reference.split('-')[0])
                                            page = doc[page_number - 1]
                                            rects = page.search_for(content)
                                            for rect in rects:
                                                sources_to_highlight.append({'page': page_number,
                                                                             'x': rect.x0,
                                                                             'y': rect.y0,
                                                                             'width': rect.width,
                                                                             'height': rect.height,
                                                                             'color': "red"})

                                    # Display the sources list if there are relevant sources
                                    if sources_to_display:
                                        st.markdown("Sources:")
                                        for source in sources_to_display:
                                            st.markdown(source)

                                # base64_pdf = base64.b64encode(doc).decode('utf-8')

                                # Embed PDF in HTML
                                # pdf_display = (F'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                                #               F'width="100%" height="1000" type="application/pdf"></iframe>')

                                # pdf_container.markdown(pdf_display)

                            # Get response from OpenAI API
                            # ai_response = get_openai_response(user_message, description_to_use)

                            # Add AI response to the conversation
                            st.session_state['conversation_histories'][st.session_state[
                                'selected_chatbot_path_serialized']].append(('Assistant', ai_response))

        self.annotations = sources_to_highlight

        if self.doc_binary_data:
            self.display_pdf()


class AIClient:

    def __init__(self):
        if session_state['model_selection'] == 'OpenAI':

            @st.cache_resource
            def load_openai_data():
                return OpenAI(), os.getenv('OPENAI_MODEL')

            self.client, self.model = load_openai_data()
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
                with st.spinner(session_state['_']("Writing response...")):
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
