import os
import base64
import re

from dotenv import load_dotenv
from io import BytesIO
from pypdf import PdfReader
import fitz
from pdf2image import convert_from_bytes
from PIL import Image

from typing import Any, Dict, List

import streamlit as st
from streamlit import session_state
from streamlit_pdf_viewer import pdf_viewer
from streamlit_dimensions import st_dimensions

from src import menu_utils

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, VectorStore
from langchain.chains import LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# Load environment variables
load_dotenv()


class SidebarDocsControls:

    def __init__(self):
        pass

    @staticmethod
    def initialize_docs_session_variables():
        """Initialize essential session variables."""
        required_keys = {
            'prompt_options_doc': menu_utils.load_prompts_from_yaml(typ='doc', prompt_key='document_assistance'),
            'uploaded_pdf_files': [],
            'selected_file_name': None,
            'sources_to_highlight': {},
            'sources_to_display': {}
        }

        for key, default_value in required_keys.items():
            if key not in session_state:
                session_state[key] = default_value

    @staticmethod
    def check_amount_of_uploaded_files_and_set_variables():
        max_files = 5
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

    def display_docs_sidebar_controls(self):
        with st.sidebar:
            st.markdown("""---""")

            st.file_uploader("Upload PDF file(s)", type=['pdf'], accept_multiple_files=True,
                             key='pdf_files', on_change=self.check_amount_of_uploaded_files_and_set_variables)


class DocsManager:

    def __init__(self, user):
        self.client = None
        session_state['USER'] = user
        # Columns to display pdf and chat
        self.column_uploaded_files, self.column_pdf, self.column_chat = st.columns([0.10, 0.425, 0.425],
                                                                                   gap="medium")
        # The vector store
        self.vector_store = None
        # Current documents list
        self.doc_binary_data = {}
        # Annotations for current documents
        self.annotations = []
        # Empty container to display pdf
        self.empty_container = st.empty()
        # Get the main component width of streamlit to display thumbnails and pdf
        self.main_dim = st_dimensions(key="main")

    def set_client(self, client):
        self.client = client

    def _handle_user_input(self, description_to_use):
        """Handles user input, sending it to OpenAI and displaying the response."""
        if user_message := st.chat_input(session_state['USER']):
            # Ensure there's a key for this conversation in the history
            if 'docs' not in session_state['conversation_histories']:
                self._create_empty_history()

            # Get the current conversation history
            current_history = session_state['conversation_histories']['docs']

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
                session_state['conversation_histories']['docs'] = current_history

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

    @staticmethod
    def get_embeddings(docs: List[Document]) -> VectorStore:
        """Given as input a document, this function returns the indexed version,
        leveraging the Chroma database"""

        vectorstore = Chroma.from_documents(
            docs,
            OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv('OPENAI_API_KEY')),
            # Generate unique ids
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
            prompt=session_state['prompt_options_doc'],
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

        # Extract file names without the '.pdf' extension
        file_names_without_extension = [os.path.splitext(file_name)[0] for file_name in files_set]

        # Populate the dropdown box with the file names without extension
        self.column_uploaded_files.selectbox("Select a file to display:", file_names_without_extension,
                                             key='selected_file_name')

        self.column_uploaded_files.number_input(label='Input chunk size',
                                                min_value=1,
                                                max_value=4000,
                                                value=300,
                                                key='chunk_size',
                                                help="The chunk size specifies the amount of characters each chunk"
                                                     " of text has. A small chunk size can be better to search for"
                                                     " more specific information, like definitions, while a bigger"
                                                     " chunk size can help search for information that is spread"
                                                     " more widely on the text. Be aware that the chunk size is"
                                                     " limited by the amount of context a model allows at a time"
                                                     " (usually around 4000 for mid-sized models)")

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
                            if os.path.splitext(filename)[0] == st.session_state['selected_file_name']:
                                st.write(f"File: **{filename}** ✅")
                            else:
                                st.write(f"File: {filename}")

    def display_pdf(self):
        with self.column_pdf:
            if st.session_state['selected_file_name']:
                pdf_viewer(input=self.doc_binary_data[st.session_state['selected_file_name']],
                           annotations=self.annotations, height=1000,
                           width=int(self.main_dim['width'] * 0.4))

    def load_doc_to_display(self):

        if session_state['uploaded_pdf_files']:
            self.display_thumbnails()

            text = []

            for file in session_state['uploaded_pdf_files']:
                filename = file.name

                file_stem = os.path.splitext(filename)[0]

                self.doc_binary_data[file_stem] = file.getvalue()

                base64_pdf = base64.b64encode(self.doc_binary_data[file_stem]).decode('utf-8')

                # self.doc_binary_data = self.doc_binary_data.decode('utf-8')

                # Embed PDF in HTML
                pdf_display = (F'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                               F'width="100%" height="1000" type="application/pdf"></iframe>')

                # Display file
                # pdf_container.markdown(pdf_display, unsafe_allow_html=True)

                # self.display_pdf()

                # Parse the pdf
                data_dict = self.parse_pdf(file)

                # Get text data
                text.extend(self.text_split(data_dict, filename))

            self.vector_store = self.get_embeddings(text)
        else:
            with self.column_pdf:
                st.header("Please upload your documents on the sidebar.")

    @staticmethod
    def get_doc(binary_data):
        if binary_data:
            doc = fitz.open(filetype='pdf', stream=binary_data)
        else:
            doc = None
        return doc

    def display_sources(self, file):
        files_with_sources = ', '.join(list(st.session_state['sources_to_display']))
        with self.column_pdf:
            st.write(f"Files with sources: {files_with_sources}")
            st.header(f"Sources for file {file}:")
            for i, source in enumerate(st.session_state['sources_to_display'][file]):
                st.markdown(f"{i + 1}) {source}")

    def display_chat_title(self):
        with self.column_chat:
            # Interface to chat with selected expert
            st.title(f"What do you want to know about your documents?")

    @staticmethod
    def _has_conversation_history():
        if ('docs' in session_state['conversation_histories']
                and session_state['uploaded_pdf_files']):
            return True
        return False

    @staticmethod
    def _create_empty_history():
        # If no conversation in history for this chatbot, then create an empty one
        session_state['conversation_histories']['docs'] = [
        ]
        return session_state['conversation_histories']['docs']

    def _display_conversation(self):
        """Displays the conversation history for the selected path."""
        conversation_history = session_state['conversation_histories'].get(
            'docs', [])

        if conversation_history:
            for speaker, message in conversation_history:
                if speaker == session_state['USER']:
                    with self.column_chat:
                        st.chat_message("user").write(message)
                else:
                    with self.column_chat:
                        st.chat_message("assistant").write(message)

        return conversation_history

    def _collect_user_input(self, conversation, repeat=False):
        with self.column_chat:

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

            # Accept user input
            if user_message := st.chat_input("Ask something about your PDFs",
                                             disabled=not session_state['uploaded_pdf_files']):

                # Prevent re-running the query on refresh or navigating back to the page by checking
                # if the last stored message is not from the User
                if not conversation or conversation[-1] != (session_state['USER'], user_message) or repeat:
                    st.session_state['sources_to_highlight'] = {}
                    st.session_state['sources_to_display'] = {}
                    return user_message

        return None

    def _user_message_processing(self, conversation, user_message):
        if user_message:
            # Add user message to the conversation
            session_state['conversation_histories']['docs'].append((session_state['USER'], user_message))

            # Generate tuples with the chat history
            # We iterate with a step of 2 and always take
            # the current and next item assuming USER follows by Assistant
            chat_history_tuples = [
                (conversation[i][1], conversation[i + 1][1])
                # Extract just the messages, ignoring the labels
                for i in range(0, len(conversation) - 1, 2)
                # We use len(conversation)-1 to avoid out-of-range errors
            ]

            with self.column_chat:
                with st.spinner():
                    condensed_question = self.get_condensed_question(
                        user_message, chat_history_tuples, os.getenv('OPENAI_MODEL')
                    )

                    sources = self.get_sources(self.vector_store, condensed_question)
                    all_output = self.get_answer(sources, condensed_question)

                    ai_response = all_output['output_text']

                    with st.chat_message("assistant"):
                        st.markdown(ai_response)
                        # Store docs to extract all sources the first time
                        docs = {}
                        for source in sources:
                            # Collect the references to relevant sources and their first 150 characters
                            source_reference = source.metadata['source']
                            # Only add it if the source appears referenced in the response
                            if source_reference in ai_response:
                                content = source.page_content

                                page_data = source_reference.split('|')
                                page_file = page_data[0]
                                page_source = page_data[1]
                                # Without this condition, it can lead to error due to deleting a
                                # file from uploads that was already inserted in the VD.
                                # Not ideal because it finds the source in the VD and
                                # references it in the answer
                                if page_file in self.doc_binary_data:
                                    if page_file not in docs:
                                        docs[page_file] = self.get_doc(self.doc_binary_data[page_file])
                                        st.session_state['sources_to_highlight'][page_file] = []
                                        st.session_state['sources_to_display'][page_file] = []
                                    st.session_state['sources_to_display'][page_file].append(
                                        f"{source_reference} - {content[:150]}")
                                    page_number = int(page_source.split('-')[0])
                                    page = docs[page_file][page_number - 1]
                                    rects = page.search_for(
                                        content)  # Maybe use flags here for better search??
                                    for rect in rects:
                                        st.session_state['sources_to_highlight'][page_file].append(
                                            {'page': page_number,
                                             'x': rect.x0,
                                             'y': rect.y0,
                                             'width': rect.width,
                                             'height': rect.height,
                                             'color': "red"})

            # Add AI response to the conversation
            st.session_state['conversation_histories']['docs'].append(('Assistant', ai_response))
        pass

    def display_chat_interface(self):

        if session_state['selected_chatbot_path']:

            doc_prompt = session_state['prompt_options_doc']
            # Checking if the prompt exists
            if doc_prompt:

                self.display_chat_title()

                # If there's already a conversation in the history for this chatbot, display it.
                if self._has_conversation_history():
                    conversation = self._display_conversation()
                else:
                    conversation = self._create_empty_history()

                file_stem = st.session_state['selected_file_name']

                user_message = self._collect_user_input(conversation)

                # Print user message immediately after
                # getting entered because we're streaming the chatbot output
                with self.column_chat:
                    if user_message:
                        with st.chat_message("user"):
                            st.markdown(user_message)

                self._user_message_processing(conversation, user_message)

                if file_stem in st.session_state['sources_to_highlight']:
                    self.annotations = st.session_state['sources_to_highlight'][file_stem]
                else:
                    self.annotations = []

                if file_stem in self.doc_binary_data:
                    self.display_pdf()
                    if file_stem in st.session_state['sources_to_display']:
                        self.display_sources(file_stem)

            else:
                st.warning("No defined prompt was found. Please define a prompt before using the chat.")
