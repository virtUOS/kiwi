import os
import base64
import re

from dotenv import load_dotenv
from io import BytesIO
from pypdf import PdfReader
import fitz
from pdf2image import convert_from_bytes
from PIL import Image

from typing import List

import streamlit as st
from streamlit import session_state
from streamlit_pdf_viewer import pdf_viewer
from streamlit_dimensions import st_dimensions

# Load environment variables
load_dotenv()


class SidebarDocsControls:

    def __init__(self, sidebar_general_manager):
        self.sgm = sidebar_general_manager

    def initialize_docs_session_variables(self):
        """Initialize essential session variables."""
        self.sgm.load_prompts(prompt_options='prompt_options_docs', typ='doc', prompt_key='document_assistance')
        required_keys = {
            'uploaded_pdf_files': [],
            'selected_file_name': None,
            'sources_to_highlight': {},
            'sources_to_display': {},

            # Current documents list in binary
            'doc_binary_data': {},

            # Current vector store
            'vector_store_docs': None,
        }

        for key, default_value in required_keys.items():
            if key not in session_state:
                session_state[key] = default_value

    @staticmethod
    def _check_amount_of_uploaded_files_and_set_variables():

        if session_state['vector_store_docs']:
            session_state['vector_store_docs'].delete_collection()

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
                             key='pdf_files', on_change=self._check_amount_of_uploaded_files_and_set_variables)


class DocsManager:

    def __init__(self, user, general_manager):
        self.gm = general_manager
        self.client = None
        session_state['USER'] = user
        # Annotations for current documents
        self.annotations = []
        # Empty container to display pdf
        self.empty_container = st.empty()
        # Get the main component width of streamlit to display thumbnails and pdf
        self.main_dim = st_dimensions(key="main")

    def set_client(self, client):
        self.client = client

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

    def display_thumbnails(self):

        # Get dimensions of thumbnails relative to screen size
        thumbnail_dim = int(self.main_dim['width'] * 0.05)
        thumbnail_size = (thumbnail_dim, thumbnail_dim)

        # Avoid repeating files by adding them to a set
        files_set = {file.name for file in session_state['uploaded_pdf_files']}

        # Extract file names without the '.pdf' extension
        file_names_without_extension = [os.path.splitext(file_name)[0] for file_name in files_set]

        # Populate the dropdown box with the file names without extension
        self.column_uploaded_files.selectbox(label="Select a file to display:", options=file_names_without_extension,
                                             key='selected_file_name', index=None)

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

    def _display_pdf(self, annotations):
        with self.column_pdf:
            if st.session_state['selected_file_name']:
                pdf_viewer(input=session_state['doc_binary_data'][st.session_state['selected_file_name']],
                           annotations=annotations, height=1000,
                           width=int(self.main_dim['width'] * 0.4))

    @st.cache_data
    def _process_uploaded_files(_self, uploaded_pdf_files):
        # When the uploaded files change reinitialize variables
        text = []
        session_state['doc_binary_data'] = {}
        for file in uploaded_pdf_files:
            filename = file.name

            file_stem = os.path.splitext(filename)[0]

            session_state['doc_binary_data'][file_stem] = file.getvalue()

            base64_pdf = base64.b64encode(session_state['doc_binary_data'][file_stem]).decode('utf-8')

            # self.doc_binary_data = self.doc_binary_data.decode('utf-8')

            # Embed PDF in HTML
            pdf_display = (F'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                           F'width="100%" height="1000" type="application/pdf"></iframe>')

            # Display file
            # pdf_container.markdown(pdf_display, unsafe_allow_html=True)

            # self.display_pdf()

            # Parse the pdf
            pdf_text = _self.parse_pdf(file)

            # Get text data
            text.extend(_self.client.text_split(pdf_text, filename))

        session_state['vector_store_docs'] = _self.client.get_vectorstore(text, collection="docs")

    @staticmethod
    def get_doc(binary_data):
        if binary_data:
            doc = fitz.open(filetype='pdf', stream=binary_data)
        else:
            doc = None
        return doc

    def _display_sources(self, file):
        files_with_sources = ', '.join(list(session_state['sources_to_display']))
        with self.column_pdf:
            st.write(f"Files with sources: {files_with_sources}")
            st.header(f"Sources for file {file}:")
            for i, source in enumerate(session_state['sources_to_display'][file]):
                st.markdown(f"{i + 1}) {source}")

    def _display_chat_title(self):
        # If no conversations yet then display chat title message
        if not self.gm.has_conversation_history('docs'):
            with self.column_chat:
                # Interface to chat with selected expert
                st.title(f"What do you want to know about your documents?")

    def _collect_user_input(self, conversation, repeat=False):
        with self.column_chat:

            # Changes the style of the chat input to appear at the bottom of the column
            self.gm.change_chatbot_style()

            # Accept user input
            if user_message := st.chat_input("Ask something about your PDFs",
                                             disabled=not session_state['uploaded_pdf_files']):

                # Prevent re-running the query on refresh or navigating back to the page by checking
                # if the last stored message is not from the User
                if not conversation or conversation[-1] != (session_state['USER'], user_message) or repeat:
                    session_state['sources_to_highlight'] = {}
                    session_state['sources_to_display'] = {}
                    return user_message

        return None

    def _user_message_processing(self, conversation, user_message):
        if user_message:
            # Add user message to the conversation
            self.gm.add_conversation_entry(chatbot='docs', speaker=session_state['USER'], message=user_message)

            chat_history_tuples = self.gm.generate_chat_history_tuples(conversation)

            with self.column_chat:
                with st.spinner():
                    condensed_question = self.client.get_condensed_question(
                        user_message, chat_history_tuples
                    )

                    sources = self.client.get_sources(session_state['vector_store_docs'], condensed_question)
                    all_output = self.client.get_answer(session_state['prompt_options_docs'],
                                                        sources,
                                                        condensed_question)

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
                                if page_file in session_state['doc_binary_data']:
                                    if page_file not in docs:
                                        docs[page_file] = self.get_doc(session_state['doc_binary_data'][page_file])
                                        session_state['sources_to_highlight'][page_file] = []
                                        session_state['sources_to_display'][page_file] = []
                                    session_state['sources_to_display'][page_file].append(
                                        f"{source_reference} - {content[:150]}")
                                    page_number = int(page_source.split('-')[0])
                                    page = docs[page_file][page_number - 1]
                                    rects = page.search_for(
                                        content)  # Maybe use flags here for better search??
                                    for rect in rects:
                                        session_state['sources_to_highlight'][page_file].append(
                                            {'page': page_number,
                                             'x': rect.x0,
                                             'y': rect.y0,
                                             'width': rect.width,
                                             'height': rect.height,
                                             'color': "red"})

            # Add AI response to the conversation
            self.gm.add_conversation_entry(chatbot='docs', speaker='Assistant', message=ai_response)

    def _handle_user_input(self, conversation_history):
        user_message = self._collect_user_input(conversation_history)

        # Print user message immediately after
        # getting entered because we're streaming the chatbot output
        with self.column_chat:
            if user_message:
                with st.chat_message("user"):
                    st.markdown(user_message)

        self._user_message_processing(conversation_history, user_message)

    def _handle_and_display_sources(self):
        file_stem = session_state['selected_file_name']
        if file_stem in session_state['sources_to_highlight']:
            self.annotations = session_state['sources_to_highlight'][file_stem]
        else:
            self.annotations = []

        if file_stem in session_state['doc_binary_data']:
            self._display_pdf(self.annotations)
            if file_stem in session_state['sources_to_display']:
                self._display_sources(file_stem)

    def _load_doc_to_display(self):
        if session_state['uploaded_pdf_files']:
            self.display_thumbnails()
            self._process_uploaded_files(session_state['uploaded_pdf_files'])
        else:
            with self.column_pdf:
                st.header("Please upload your documents on the sidebar.")

    def display_chat_interface(self):

        # Columns to display pdf and chat
        self.column_uploaded_files, self.column_pdf, self.column_chat = st.columns([0.10, 0.425, 0.425],
                                                                                   gap="medium")

        self._load_doc_to_display()

        if session_state['selected_chatbot_path']:

            doc_prompt = session_state['prompt_options_docs']
            # Checking if the prompt exists
            if doc_prompt:

                self._display_chat_title()

                conversation_history = self.gm.get_conversation_history('docs')

                # If there's already a conversation in the history for this chatbot, display it.
                if conversation_history:
                    self.gm.display_conversation(conversation_history=conversation_history, container=self.column_chat)
                else:
                    # Create empty conversation for this chatbot in case there's no key for it yet
                    self.gm.create_empty_history(chatbot='docs')

                self._handle_user_input(conversation_history)
                self._handle_and_display_sources()

            else:
                st.warning("No defined prompt was found. Please define a prompt before using the chat.")
