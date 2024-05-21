import os
import multiprocessing
from pdf2image import convert_from_bytes
from PIL import Image

import streamlit as st
from streamlit import session_state
from streamlit_pdf_viewer import pdf_viewer
from streamlit_dimensions import st_dimensions
import streamlit_pills as stp

import src.streamlit_styling as st_styling
import src.utils as utils
import src.docs_utils as docs_utils
from src.summarization_manager import SummarizationManager

from multiprocessing import Queue


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
    def _populate_thumbnails(files_set, thumbnail_size):
        """
        Populates the session column with thumbnails of uploaded PDF files.
        """
        if not files_set:
            st.session_state['column_uploaded_files'].write("**No files uploaded**")
            return

        st.session_state['column_uploaded_files'].write(f"**{len(files_set)} files uploaded**")

        thumbnail_images = {}  # Storing thumbnails to avoid redundancy
        for file_id, file in st.session_state['uploaded_pdf_files'].items():
            if file.name not in thumbnail_images:
                file.seek(0)  # Reset file pointer
                image = convert_from_bytes(file.read(), first_page=1, last_page=1)[0]

                if image:  # Verification and resizing
                    thumbnail = image.resize(thumbnail_size, Image.Resampling.BILINEAR)
                    thumbnail_images[file.name] = thumbnail

                    with (st.session_state['column_uploaded_files']):
                        st.image(thumbnail)
                        display_status = "✅" if os.path.splitext(file.name)[0] == session_state[
                            'selected_file_name'] else ""
                        st.write(f"{file_id}: **{file.name}** {display_status}")

    def display_thumbnails(self):
        """
        Displays thumbnails for uploaded PDF files, allowing users to select
        a file to view and set a chunk size for text selection.
        """
        if 'uploaded_pdf_files' not in session_state or 'column_uploaded_files' not in session_state:
            st.warning(session_state['_']("Uploaded PDF files or UI column for displaying files "
                                          "not found in session state."))
            return

        # Calculating thumbnail dimensions and size relative to screen width
        thumbnail_dim = int(self.main_dim['width'] * 0.05) if self.main_dim else 100
        thumbnail_size = (thumbnail_dim, thumbnail_dim)

        files_set = {file.name for file_id, file in session_state['uploaded_pdf_files'].items()}
        file_names_without_extension = [os.path.splitext(file_name)[0] for file_name in files_set]

        session_state['column_uploaded_files'].selectbox(
            label=session_state['_']("Select a file to display:"),
            options=file_names_without_extension,
            key='selected_file_name', index=None  # Setting a default index
        )

        session_state['selected_file_id'] = next((file_id for file_id, file in
                                                  session_state['uploaded_pdf_files'].items() if
                                                  file.name.split('.')[0] == session_state['selected_file_name']), None)

        self._populate_thumbnails(files_set, thumbnail_size)

    def _display_pdf(self, annotations):
        """
        Displays a selected PDF file with annotated sections.

        Parameters:
        - annotations: A list of annotations to apply to the displayed PDF.
        """
        with session_state['column_pdf']:
            pdf_viewer(input=session_state['doc_binary_data'][st.session_state['selected_file_id']],
                       annotations=annotations,
                       annotation_outline_size=2,
                       height=1000,
                       width=int(self.main_dim['width'] * 0.4) if self.main_dim else 400)

    def _process_uploaded_files(_self):
        """
        Processes and stores the content of uploaded PDF files into a session state.
        Updates the session state with metadata, binary data, chunk data, and text data for each uploaded PDF.
        """
        text = []
        session_state['doc_binary_data'] = {}
        for file_id, file in session_state['uploaded_pdf_files'].items():
            # Store binary data of the file
            session_state['doc_binary_data'][file_id] = file.getvalue()

            # Parse the PDF and update session state with chunk and text data
            session_state['doc_chunk_data'][file_id], session_state['doc_text_data'][file_id] = \
                docs_utils.parse_pdf(session_state['doc_binary_data'][file_id], file_id)

            # Prepare text data for database
            text.extend(_self.client.prepare_chunks_for_database(session_state['doc_chunk_data'][file_id]))

        # Update collection name based on username and initialize vector store for documents
        session_state['collection_name'] = f"docs-{session_state['username']}"
        session_state['vector_store_docs'] = _self.client.get_vectorstore(text)

    @staticmethod
    def _display_chat_title():
        """
        Displays a title in the chat column if no conversation history is found.
        """
        if not utils.has_conversation_history('docs'):
            with session_state['column_chat']:
                st.title(session_state['_']("What do you want to know about your documents?"))

    @staticmethod
    def _collect_user_input(conversation, repeat=False):
        """
        Collects user input from a chat interface, avoiding repetition of queries on page refresh.

        Parameters:
        conversation : list
            The current conversation history.
        repeat : bool, optional
            A flag to allow repeated queries, by default False.

        Returns:
        str or None
            The user's message if new or None if no new input or disabled due to missing files.
        """
        with session_state['column_chat']:
            st_styling.change_chatbot_style()  # Adjusts chat interface styling

            if user_message := st.chat_input(session_state['_']("Ask something about your PDFs"),
                                             disabled=not session_state['uploaded_pdf_files']):
                if not conversation or conversation[-1] != ('USER', user_message) or repeat:
                    session_state['sources_to_highlight'] = {}
                    session_state['sources_to_display'] = {}
                    return user_message
        return None

    def _user_message_processing(self, conversation, user_message):
        """
        Processes the user's message, updates the conversation history, and generates an AI response.

        This method adds the user's message to the conversation history, generates a condensed question based on
        the message and the conversation history, retrieves relevant sources, generates an AI response, and updates
        the session state to display sources and highlights relevant to the user's query.

        Parameters:
        conversation : list
            The current conversation history between the user and the assistant.
        user_message : str
            The latest message from the user.

        Side Effects:
        Updates the conversation history and session state with the AI's response, sources to highlight,
        and sources to display.
        """
        if user_message:
            # Update conversation history with the user's message
            utils.add_conversation_entry(chatbot='docs', speaker=session_state['USER'], message=user_message)

            # Generate tuples representing the chat history for processing
            chat_history_tuples = utils.generate_chat_history_tuples(conversation)

            # Reset sources to display with each new user query
            session_state['sources_to_display'] = {}

            with session_state['column_chat']:
                with st.spinner(session_state['_']("Generating response...")):
                    # Get condensed question, sources, and AI response
                    condensed_question = self.client.get_condensed_question(user_message, chat_history_tuples)
                    sources = self.client.get_sources(session_state['vector_store_docs'], condensed_question)
                    all_output = self.client.get_answer(session_state['prompt_options_docs'][0], sources,
                                                        condensed_question)
                    ai_response = all_output['output_text']

                    # Display AI's response in the chat column
                    with st.chat_message("assistant"):
                        st.markdown(ai_response)

                        # Process sources to highlight and display
                        self._process_and_display_sources(ai_response, sources)

            # Update conversation history with AI's response and rerun to reflect updates
            utils.add_conversation_entry(chatbot='docs', speaker='Assistant', message=ai_response)
            st.rerun()

    def _process_and_display_sources(self, ai_response, sources):
        """
        Processes sources related to the AI's response for displaying and highlighting in the document.

        Parameters:
        ai_response : str
            The response generated by the AI based on the user's query.
        sources : list
            A list of sources related to the AI's response.

        Side Effects:
        Updates session state with the information necessary to display and highlight sources in the document.
        """
        docs = {}
        for source in sources:
            source_reference = source.metadata['source']
            if source_reference in ai_response:
                content = source.page_content
                page_data = source_reference.split('|')
                page_file = page_data[0]

                if page_file in session_state['doc_binary_data']:
                    if page_file not in docs:
                        docs[page_file] = docs_utils.open_pdf(session_state['doc_binary_data'][page_file])
                        session_state['sources_to_highlight'][page_file] = []
                        session_state['sources_to_display'][page_file] = []

                    # Prepare source display and highlighting metadata
                    session_state['sources_to_display'][page_file].append(f"{source_reference} - {content[:150]}")
                    page_number, page_source = self._parse_page_metadata(page_data[1])
                    session_state['sources_to_highlight'][page_file].append(
                        self._prepare_source_highlight(source, page_number))

    @staticmethod
    def _parse_page_metadata(page_source):
        """
        Parses page source metadata into a usable format.

        Parameters:
        page_source : str
            The source information of the page in 'page_number-source_type' format.

        Returns:
        Tuple[int, str]
            A tuple containing the page number as an integer and the source type as a string.
        """
        page_number = int(page_source.split('-')[0])
        return page_number, page_source

    @staticmethod
    def _prepare_source_highlight(source, page_number):
        """
        Prepares the highlight metadata for a given source.

        Parameters:
        source : object
            The source object containing metadata for the highlight.
        page_number : int
            The page number where the highlight will be applied.

        Returns:
        dict
            A dictionary with the highlight metadata including page number, dimensions, and color.
        """
        return {
            'page': page_number,
            'x': source.metadata['x0'],
            'y': source.metadata['y0'],
            'width': source.metadata['x1'] - source.metadata['x0'],
            'height': source.metadata['y1'] - source.metadata['y0'],
            'color': "red",
        }

    def _handle_user_input(self, conversation_history, predefined_prompt_selected):
        """
        Processes the user's input either from a predefined prompt or collected interactively,
        and updates the conversation accordingly.

        Parameters:
        conversation_history : list
            The current conversation history.
        predefined_prompt_selected : str
            The message from a predefined prompt selected by the user, if any.
        """
        # Determine the source of user message
        user_message = predefined_prompt_selected if predefined_prompt_selected \
            else self._collect_user_input(conversation_history)

        # Display the user's message immediately for streaming output
        if user_message:
            with session_state['column_chat']:
                with st.chat_message("user"):
                    st.markdown(user_message)

        # Process the user's message
        self._user_message_processing(conversation_history, user_message)

    def _background_summary_task(self,
                                 doc_text_data,
                                 model_name,
                                 map_prompt_template,
                                 reduce_prompt_template,
                                 result_queue):
        """
        Executes the summarization process in the background for the given document text data.
        The function is designed to run as a background task to not block the main Streamlit thread.
        Once the summary is generated, it is put into the provided thread-safe queue.

        Parameters:
        doc_text_data : str
            The text content of the document for which the summary is to be generated.
        model_name : str
            The name of the model to use for generating the document summary.
        map_prompt_template : str
            The map prompt template to be used in the summarization process.
        reduce_prompt_template : str
            The reduce prompt template to be used in the summarization process.
        result_queue : Queue
            A thread-safe queue where the generated summary result will be put once the task completes.
        """
        try:
            summary = self.sum_man.get_data_from_documents(
                docs=doc_text_data,
                model_name=model_name,
                map_prompt_template=map_prompt_template,
                reduce_prompt_template=reduce_prompt_template)
            result_queue.put(summary)
        except Exception as e:
            print(f"Error in background task: {e}")  # Ensure any errors are logged

    def _generate_summary_background(self, doc_text_data, summary_queue):
        """
        Initiates the asynchronous generation of a document summary by triggering a background task.
        This method aims to offload intensive summarization processing to a background thread, allowing
        the Streamlit application to remain responsive. The completion of the task is communicated through
        a thread-safe queue, specifically instantiated for each document based on its filename.

        The method sets up necessary parameters for the summarization, including model name and summarization
        templates, and uses these parameters to start the background task. The result of the summarization,
        once completed, will be placed into a dedicated queue identified by the file name, from which it can
        be retrieved and displayed in the Streamlit UI.

        Parameters:
        doc_text_data : str
            The textual content of the document for which a summary is requested. This data will be processed
            by the background task to generate the summary.
        summary_queue : Queue
            A thread-safe queue through which the generated summary will be communicated back to the application.
            This queue is used to safely pass the summary data from the background thread to the Streamlit main thread.
        """
        # Define templates outside the thread
        reduce_prompt_template = session_state.get('prompt_options_docs', [{}])[1].get('template', '')
        map_prompt_template = session_state.get('prompt_options_docs', [{}])[2].get('template', '')

        # Start the background task
        process = multiprocessing.Process(target=self._background_summary_task, args=(
            doc_text_data,
            os.getenv('OPENAI_MODEL_EXTRA'),
            map_prompt_template,
            reduce_prompt_template,
            summary_queue))
        process.start()

    def _background_suggestions_task(self, user_message, prompt, vector_store, result_queue):
        """
        A background task function designed to process the user's query and generate AI-based suggestions
        related to the content of uploaded documents. The function condenses the query, retrieves relevant
        sources, and utilizes an AI model to generate suggestions, which are then placed in the provided
        result queue for retrieval and display.

        Parameters:
        user_message : str
            The user's query regarding the document's content, serving as input for generating suggestions.
        prompt : str
            The prompt directive used to guide the AI model in generating relevant suggestions.
        vector_store : dict or object
            A data structure or object that contains vectorized representations of document content, used
            for retrieving relevant sources related to the user's query.
        result_queue : Queue
            A thread-safe queue where the task places the generated suggestions upon completion, enabling
            their retrieval and display in the Streamlit interface.
        """
        # Condense the user's query to its most relevant form
        condensed_question = self.client.get_condensed_question(user_message, [])

        # Retrieve sources from the document that are relevant to the condensed question
        sources = self.client.get_sources(vector_store, condensed_question)

        # Get the AI-generated answer based on the prompt options, sources, and condensed question
        all_output = self.client.get_answer(
            prompt,
            sources,
            condensed_question
        )

        # Extract the AI-generated text response
        ai_response = all_output['output_text']

        processed_response = utils.process_ai_response_for_suggestion_queries(ai_response)

        # Check if not suggestions were generated for later display purposes
        if not processed_response:
            processed_response = "EMPTY"

        result_queue.put(processed_response)

    def _generate_suggestions_background(self, suggestions_queue):
        """
        Triggers the background generation of query suggestions based on the document's content. This function
        initiates a background task that asynchronously generates suggestions, allowing the Streamlit application
        to stay responsive. The generated suggestions are communicated back via the provided thread-safe queue.

        Parameters:
        suggestions_queue : Queue
            A thread-safe queue through which the generated suggestions are communicated back to the Streamlit
            application. The suggestions are placed in this queue by the background task upon completion.
        """
        # Start the background task
        process = multiprocessing.Process(target=self._background_suggestions_task, args=(
            session_state['prompt_options_docs'][-1]['queries'],
            session_state['prompt_options_docs'][0],
            session_state['vector_store_docs'],
            suggestions_queue))
        process.start()

    def _handle_and_display_annotations_and_sources(self):
        """
        Handles the display of annotations and sources to the user based on the current file selection.
        """
        file_stem = session_state['selected_file_name']
        file_id = session_state['selected_file_id']

        # Prepare annotations for the selected file
        self.annotations = session_state['sources_to_highlight'].get(file_id, [])

        # Display the PDF with optional annotations and list of sources if available
        if file_id in session_state['doc_binary_data'] and file_stem:
            self._display_pdf(self.annotations)

    def _load_doc_to_display(self):
        """
        Loads the document to be displayed. If documents are uploaded, it displays thumbnails
        and processes the files. Otherwise, it prompts the user to upload documents.
        """
        if session_state['uploaded_pdf_files']:
            self.display_thumbnails()
            self._process_uploaded_files()
        else:
            with session_state['column_pdf']:
                st.header(session_state['_']("Please upload your documents on the sidebar."))

    @staticmethod
    def _reset_suggestions(file_name):
        session_state.pop(f'suggestions_{file_name}')

    def _display_suggestions_if_ready(self, file_name):
        """
        Displays generated query suggestions for a specific document if they are available within the session state.
        If the suggestions have not yet been generated or are still processing, this method displays a loading message
        in the chat column to inform the user that the suggestion generation is in progress. This approach allows
        users to be aware of ongoing processes and ensures that the UI remains informative and interactive.

        The method checks the session state for the readiness of the document's suggestions. The suggestions' availability
        is indicated by the presence of an entry in the session state named after the document's filename with a prefix 'suggestions_'.
        If suggestions are ready, they will be displayed within an expandable section in the chat column, providing users with
        interactive options to click on the suggestions that interest them.

        Parameters:
        file_name : str
            A unique identifier for the document whose suggestions are being checked. This identifier is used to look up the
            suggestions within the session state. If the suggestions are available, they are displayed using interactive pills;
            otherwise, a loading message is shown to indicate that the suggestion generation process is still in progress.

        Returns:
        str or None
            The selected query suggestion by the user, if any, otherwise None.
        """
        predefined_prompt_selected = None
        with session_state['column_chat']:
            column_suggestions, column_button = st.columns([9, 1])
        # When summary is ready, display it
        if session_state.get(f'suggestions_{file_name}', False):
            if session_state[f'suggestions_{file_name}'] != "EMPTY":
                with session_state['column_chat']:
                    with column_suggestions:
                        with st.expander(session_state['_']("Query suggestions")):
                            predefined_prompt_selected = stp.pills("", session_state[f'suggestions_{file_name}'],
                                                                   session_state[f'icons_{file_name}'],
                                                                   index=session_state.get('pills_index'))

                    # Remove the chosen suggestion from the list after selection
                    if predefined_prompt_selected:
                        index_to_eliminate = session_state[f'suggestions_{file_name}'].index(predefined_prompt_selected)
                        session_state[f'suggestions_{file_name}'].pop(index_to_eliminate)
                        session_state[f'icons_{file_name}'].pop(index_to_eliminate)

            elif session_state[f'suggestions_{file_name}'] == "EMPTY":
                # Keep displaying a loading message until the summary is ready
                with session_state['column_chat']:
                    with column_suggestions:
                        st.info(session_state['_']("Couldn't generate query suggestions."))

            with session_state['column_chat']:
                with column_button:
                    st.button("🔄", help=session_state['_']("Re-generate suggestions"),
                              on_click=self._reset_suggestions,
                              args=(file_name,)
                              )
        else:
            # Keep displaying a loading message until the summary is ready
            with session_state['column_chat']:
                generating_suggestions_text = session_state['_']("Generating query suggestions for document")
                st.info(f"{generating_suggestions_text} {file_name}...")
        return predefined_prompt_selected

    @staticmethod
    def _reset_summary(file_name):
        session_state.pop(f'summary_{file_name}')

    def _display_summary_if_ready(self, file_name):
        """
        Displays the generated summary for a specific document if it is available within the session state.
        If the summary has not yet been generated or is still processing, this method displays a loading message
        in the chat column to inform the user that the summary generation is in progress. This approach allows
        users to be aware of ongoing processes and ensures that the UI remains informative and interactive.

        The method relies on the Stream_state to check the readiness of the document's summary. The summary's availability
        is indicated by the presence of an entry in the session state named after the document's filename. If the summary
        is ready, it will be displayed within an expandable section in the chat column, allowing users to view the summary
        at their convenience.

        Parameters:
        file_name : str
            A unique identifier for the document whose summary is being checked. This identifier is used to look up the
            summary within the session state. If the summary is available, it is displayed; otherwise, a loading message
            is shown to indicate that the summary generation is still in progress.
        """
        with session_state['column_chat']:
            column_summary, column_button = st.columns([9, 1])
        # When summary is ready, display it
        if session_state.get(f'summary_{file_name}', False):
            with session_state['column_chat']:
                with column_summary:
                    with st.expander(session_state['_']("Summary")):
                        st.markdown(session_state[f'summary_{file_name}'])
                with column_button:
                    st.button("🔄", help=session_state['_']("Re-generate summary"),
                              on_click=self._reset_suggestions,
                              args=(file_name,)
                              )
        else:
            # Keep displaying a loading message until the summary is ready
            with session_state['column_chat']:
                generating_summary_text = session_state['_']("Generating summary for document")
                st.info(f"{generating_summary_text} {file_name}...")

    def display_chat_interface(self):
        """
        Displays the chat interface, handling document display, summary generation, suggestions,
        and user query processing dynamically based on user interaction and document content.
        """
        # Initialize the columns for displaying documents and chat interface
        (session_state['column_uploaded_files'],
         session_state['column_pdf'],
         session_state['column_chat']) = st.columns([0.1, 0.425, 0.425], gap="medium")

        self._load_doc_to_display()

        if session_state['selected_chatbot_path']:
            doc_prompt = session_state['prompt_options_docs'][0]
            predefined_prompt_selected = None
            if doc_prompt:
                self._display_chat_title()

                file_name = session_state['selected_file_name']
                file_id = session_state['selected_file_id']
                if file_name is not None:
                    if f'summary_{file_name}' not in session_state:
                        session_state[f'summary_{file_name}'] = None  # Create the variable to avoid re-running thread
                        session_state[f'result_summary_queue_{file_name}'] = Queue()
                        self._generate_summary_background(session_state['doc_text_data'][file_id],
                                                          session_state[f'result_summary_queue_{file_name}'])

                    # Check if there are results in the queue
                    if (f'result_summary_queue_{file_name}' in session_state and
                            not session_state[f'result_summary_queue_{file_name}'].empty() and
                            not session_state[f'summary_{file_name}']):
                        # Retrieve the result from the queue
                        summary = session_state[f'result_summary_queue_{file_name}'].get()

                        # Safely update session_state in the main thread
                        session_state[f'summary_{file_name}'] = summary

                    self._display_summary_if_ready(file_name)

                    # Generate and display suggestions if not already present for the file
                    if f'suggestions_{file_name}' not in session_state:
                        session_state[f'suggestions_{file_name}'] = []  # Create the variable to avoid re-running thread
                        session_state[f'icons_{file_name}'] = []
                        session_state[f'result_suggestions_queue_{file_name}'] = Queue()
                        self._generate_suggestions_background(session_state[f'result_suggestions_queue_{file_name}'])
                    if (f'result_suggestions_queue_{file_name}' in session_state and
                            not session_state[f'result_suggestions_queue_{file_name}'].empty() and
                            not session_state[f'suggestions_{file_name}']):
                        # Retrieve the result from the queue
                        queries = session_state[f'result_suggestions_queue_{file_name}'].get()

                        if queries != "EMPTY":
                            for query in queries:
                                session_state[f'suggestions_{file_name}'].append(query)
                                session_state[f'icons_{file_name}'].append(session_state[
                                                                               'prompt_options_docs'][-1]['icon'])
                        else:
                            session_state[f'suggestions_{file_name}'] = queries

                    predefined_prompt_selected = self._display_suggestions_if_ready(file_name)

                conversation_history = utils.get_conversation_history('docs')

                # Display conversation history or initialize it if absent
                if conversation_history:
                    utils.display_conversation(conversation_history=conversation_history,
                                               container=session_state['column_chat'])
                else:
                    utils.create_history(chatbot='docs')

                self._handle_user_input(conversation_history, predefined_prompt_selected)
                self._handle_and_display_annotations_and_sources()

            else:
                st.warning(session_state['_']("No defined prompt was found. "
                                              "Please define a prompt before using the chat."))
