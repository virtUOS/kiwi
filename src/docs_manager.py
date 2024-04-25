import os
from pdf2image import convert_from_bytes
from PIL import Image

import streamlit as st
from streamlit import session_state
from streamlit_pdf_viewer import pdf_viewer
from streamlit_dimensions import st_dimensions
import streamlit_pills as stp

import src.utils as utils
import src.docs_utils as docs_utils
from src.summarization_manager import SummarizationManager


class DocsManager:

    def __init__(self):
        """
        Initializes the DocsManager with an optional client and sets up the
        necessary properties for document management and summarization.
        """
        self.client = None
        self.annotations = []  # Annotations for current document
        self.sum_man = SummarizationManager()
        self.main_dim = None

    def set_client(self, client):
        """
        Assigns a client to the DocsManager.

        Parameters:
        - client: The client object to be used by DocsManager.
        """
        self.client = client

    def set_window_dimensions(self):
        """
        Fetches and sets the main component width from the UI framework to
        manage how document thumbnails and PDFs are displayed.
        """
        self.main_dim = st_dimensions(key="main")

    @staticmethod
    def _populate_thumbnails(files_set, thumbnail_size, selected_file_name):
        """
        Populates the session column with thumbnails of uploaded PDF files.
        """
        if not files_set:
            st.session_state['column_uploaded_files'].write("**No files uploaded**")
            return

        st.session_state['column_uploaded_files'].write(f"**{len(files_set)} files uploaded**")

        thumbnail_images = {}  # Storing thumbnails to avoid redundancy
        for file in st.session_state['uploaded_pdf_files']:
            if file.name not in thumbnail_images:
                file.seek(0)  # Reset file pointer
                image = convert_from_bytes(file.read(), first_page=1, last_page=1)[0]

                if image:  # Verification and resizing
                    thumbnail = image.resize(thumbnail_size, Image.Resampling.BILINEAR)
                    thumbnail_images[file.name] = thumbnail

                    with st.session_state['column_uploaded_files']:
                        st.image(thumbnail)
                        display_status = "✅" if os.path.splitext(file.name)[0] == selected_file_name else ""
                        st.write(f"File: **{file.name}** {display_status}")

    def display_thumbnails(self):
        """
        Displays thumbnails for uploaded PDF files, allowing users to select
        a file to view and set a chunk size for text selection.
        """
        if 'uploaded_pdf_files' not in session_state or 'column_uploaded_files' not in session_state:
            st.warning("Uploaded PDF files or UI column for displaying files not found in session state.")
            return

        # Calculating thumbnail dimensions and size relative to screen width
        thumbnail_dim = int(self.main_dim['width'] * 0.05) if self.main_dim else 100
        thumbnail_size = (thumbnail_dim, thumbnail_dim)

        files_set = {file.name for file in st.session_state['uploaded_pdf_files']}
        file_names_without_extension = [os.path.splitext(file_name)[0] for file_name in files_set]

        selected_file_name = st.session_state['column_uploaded_files'].selectbox(
            label="Select a file to display:",
            options=file_names_without_extension,
            key='selected_file_name', index=None  # Setting a default index
        )

        st.session_state['column_uploaded_files'].number_input(
            label='Input chunk size',
            min_value=100, max_value=4000, value=300, key='chunk_size',
            help=("The chunk size specifies the amount of characters each "
                  "chunk of text contains. Adjust based on the level of detail needed.")
        )
        self._populate_thumbnails(files_set, thumbnail_size, selected_file_name)

    def _display_pdf(self, annotations):
        """
        Displays a selected PDF file with annotated sections.

        Parameters:
        - annotations: A list of annotations to apply to the displayed PDF.
        """
        with session_state['column_pdf']:
            pdf_viewer(input=session_state['doc_binary_data'][st.session_state['selected_file_name']],
                       annotations=annotations,
                       annotation_outline_size=2,
                       height=1000,
                       width=int(self.main_dim['width'] * 0.4) if self.main_dim else 400)

    @st.cache_data
    def _process_uploaded_files(_self, uploaded_pdf_files):
        # When the uploaded files change reinitialize variables
        text = []
        session_state['doc_binary_data'] = {}
        for file in uploaded_pdf_files:
            file_name = file.name

            file_stem = os.path.splitext(file_name)[0]

            session_state['doc_binary_data'][file_stem] = file.getvalue()

            # Parse the pdf
            session_state['doc_chunk_data'][file_stem], session_state[
                'doc_text_data'][file_stem] = docs_utils.parse_pdf(
                session_state['doc_binary_data'][file_stem], file_name)

            # Get text data
            text.extend(_self.client.prepare_chunks_for_database(session_state['doc_chunk_data'][file_stem]))

        session_state['collection_name'] = f"docs-{session_state['username']}"
        session_state['vector_store_docs'] = _self.client.get_vectorstore(text)

    @staticmethod
    def _display_sources(file):
        files_with_sources = ', '.join(list(session_state['sources_to_display']))
        with session_state['column_pdf']:
            st.write(f"Files with sources: {files_with_sources}")
            st.header(f"Sources for file {file}:")
            for i, source in enumerate(session_state['sources_to_display'][file]):
                st.markdown(f"{i + 1}) {source}")

    @staticmethod
    def _display_chat_title():
        # If no conversations yet then display chat title message
        if not utils.has_conversation_history('docs'):
            with session_state['column_chat']:
                # Interface to chat with selected expert
                st.title(f"What do you want to know about your documents?")

    @staticmethod
    def _collect_user_input(conversation, repeat=False):
        with session_state['column_chat']:

            # Changes the style of the chat input to appear at the bottom of the column
            utils.change_chatbot_style()

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

    def _process_query_and_get_answer_for_suggestions(self, user_message):
        condensed_question = self.client.get_condensed_question(
            user_message, []
        )

        sources = self.client.get_sources(session_state['vector_store_docs'], condensed_question)
        all_output = self.client.get_answer(session_state['prompt_options_docs'][0],
                                            sources,
                                            condensed_question)

        ai_response = all_output['output_text']

        return ai_response

    def _user_message_processing(self, conversation, user_message):
        if user_message:
            # Add user message to the conversation
            utils.add_conversation_entry(chatbot='docs', speaker=session_state['USER'], message=user_message)

            chat_history_tuples = utils.generate_chat_history_tuples(conversation)

            # The sources to display change with every new user query
            session_state['sources_to_display'] = {}

            with session_state['column_chat']:
                with st.spinner():
                    condensed_question = self.client.get_condensed_question(
                        user_message, chat_history_tuples
                    )

                    sources = self.client.get_sources(session_state['vector_store_docs'], condensed_question)
                    all_output = self.client.get_answer(session_state['prompt_options_docs'][0],
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
                                        docs[page_file] = docs_utils.get_doc(session_state['doc_binary_data'][page_file])
                                        session_state['sources_to_highlight'][page_file] = []
                                        session_state['sources_to_display'][page_file] = []
                                    session_state['sources_to_display'][page_file].append(
                                        f"{source_reference} - {content[:150]}")
                                    page_number = int(page_source.split('-')[0])

                                    session_state['sources_to_highlight'][page_file].append(
                                        {'page': page_number,
                                         'x': source.metadata['x0'],
                                         'y': source.metadata['y0'],
                                         'width': source.metadata['x1'] - source.metadata['x0'],
                                         'height': source.metadata['y1'] - source.metadata['y0'],
                                         'color': "red",
                                         })

            # Add AI response to the conversation
            utils.add_conversation_entry(chatbot='docs', speaker='Assistant', message=ai_response)
            st.rerun()

    def _handle_user_input(self, conversation_history, predefined_prompt_selected):
        if predefined_prompt_selected:
            user_message = predefined_prompt_selected
        else:
            user_message = self._collect_user_input(conversation_history)

        # Print user message immediately after
        # getting entered because we're streaming the chatbot output
        with session_state['column_chat']:
            if user_message:
                with st.chat_message("user"):
                    st.markdown(user_message)

        self._user_message_processing(conversation_history, user_message)

    @staticmethod
    def _display_summary(file_name):
        with session_state['column_chat']:
            with st.expander("Summary"):
                st.markdown(session_state[f'summary_{file_name}'])

    def _generate_summary(self):
        with session_state['column_chat']:
            with st.spinner("Generating summary..."):
                summary_reduce_prompt = session_state['prompt_options_docs'][1]
                summary_map_prompt = session_state['prompt_options_docs'][2]
                file_name = session_state['selected_file_name']
                session_state[f'summary_{file_name}'] = self.sum_man.get_data_from_documents(
                    docs=session_state['doc_text_data'][
                        session_state['selected_file_name']],
                    model_name=os.getenv('OPENAI_MODEL_EXTRA'),
                    map_prompt_template=summary_map_prompt['template'],
                    reduce_prompt_template=summary_reduce_prompt['template'])

    def _handle_and_display_sources(self):
        file_stem = session_state['selected_file_name']
        if file_stem in session_state['sources_to_highlight']:
            self.annotations = session_state['sources_to_highlight'][file_stem]
        else:
            self.annotations = []

        if file_stem in session_state['doc_binary_data']:
            if st.session_state['selected_file_name']:
                self._display_pdf(self.annotations)
                if file_stem in session_state['sources_to_display']:
                    self._display_sources(file_stem)

    def _load_doc_to_display(self):
        if session_state['uploaded_pdf_files']:
            self.display_thumbnails()
            self._process_uploaded_files(session_state['uploaded_pdf_files'])
        else:
            with session_state['column_pdf']:
                st.header("Please upload your documents on the sidebar.")

    def _generate_suggestions(self, file_name):
        session_state[f'suggestions_{file_name}'] = []
        session_state[f'icons_{file_name}'] = []
        with session_state['column_chat']:
            with st.spinner(f"Generating suggestions..."):
                ai_response = self._process_query_and_get_answer_for_suggestions(
                    session_state['prompt_options_docs'][-1]['queries'])
                print(ai_response)
                queries = utils.process_ai_response_for_suggestion_queries(ai_response)
                print(queries)
                for query in queries:
                    session_state[f'suggestions_{file_name}'].append(query)
                    session_state[f'icons_{file_name}'].append(session_state['prompt_options_docs'][-1]['icon'])

    @staticmethod
    def _display_suggestions(file_name):
        session_state['pills_index'] = None
        with session_state['column_chat']:
            with st.expander("Query suggestions:"):
                predefined_prompt_selected = stp.pills("", session_state[f'suggestions_{file_name}'],
                                                       session_state[f'icons_{file_name}'],
                                                       index=session_state['pills_index'])

        # Get rid of the suggestion now that it was chosen
        if predefined_prompt_selected:
            index_to_eliminate = session_state[f'suggestions_{file_name}'].index(predefined_prompt_selected)
            session_state[f'suggestions_{file_name}'].pop(index_to_eliminate)
            session_state[f'icons_{file_name}'].pop(index_to_eliminate)
        return predefined_prompt_selected

    def display_chat_interface(self):

        # Columns to display pdf and chat
        (session_state['column_uploaded_files'],
         session_state['column_pdf'],
         session_state['column_chat']) = st.columns([0.10, 0.425, 0.425], gap="medium")

        self._load_doc_to_display()

        if session_state['selected_chatbot_path']:

            doc_prompt = session_state['prompt_options_docs'][0]
            # Checking if the prompt exists
            if doc_prompt:

                self._display_chat_title()

                file_name = session_state['selected_file_name']
                # Check if there's no summary yet for this file and generate the summary
                if f'summary_{file_name}' not in session_state and session_state['selected_file_name'] is not None:
                    self._generate_summary()
                # If there's already a summary for this file, display it
                if f'summary_{file_name}' in session_state:
                    self._display_summary(file_name)

                predefined_prompt_selected = None
                if f'suggestions_{file_name}' not in session_state and session_state['selected_file_name'] is not None:
                    self._generate_suggestions(file_name)
                if f'suggestions_{file_name}' in session_state and session_state[f'suggestions_{file_name}']:
                    predefined_prompt_selected = self._display_suggestions(file_name)

                print("PPS", predefined_prompt_selected)

                conversation_history = utils.get_conversation_history('docs')

                # If there's already a conversation in the history for this chatbot, display it.
                if conversation_history:
                    utils.display_conversation(conversation_history=conversation_history,
                                               container=session_state['column_chat'])
                else:
                    # Create empty conversation for this chatbot in case there's no key for it yet
                    utils.create_history(chatbot='docs')

                self._handle_user_input(conversation_history, predefined_prompt_selected)
                self._handle_and_display_sources()

            else:
                st.warning("No defined prompt was found. Please define a prompt before using the chat.")
