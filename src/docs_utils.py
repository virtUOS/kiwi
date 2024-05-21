import fitz
from io import BytesIO

import streamlit as st
from streamlit import session_state

import src.utils as utils

# Variables
MAX_FILES = 5


def initialize_docs_session_variables(typ, language):
    """Initialize essential session variables."""

    session_state['typ'] = typ
    session_state[f"prompt_options_{session_state['typ']}"] = utils.load_prompts(session_state['typ'], language)
    required_keys = {
        'uploaded_pdf_files': {},
        'selected_file_name': None,
        'sources_to_highlight': {},
        'sources_to_display': {},

        # Current documents data
        'doc_text_data': {},
        'doc_chunk_data': {},
        'doc_binary_data': {},

        # Current vector store
        'vector_store_docs': None,
    }

    for key, default_value in required_keys.items():
        if key not in session_state:
            session_state[key] = default_value


def reset_docs_variables():
    if session_state['vector_store_docs']:
        session_state['vector_store_docs'].delete_collection()

    required_keys = {
        'uploaded_pdf_files': {},
        'selected_file_name': None,
        'sources_to_highlight': {},
        'sources_to_display': {},

        # Current documents list in binary
        'doc_text_data': {},
        'doc_chunk_data': {},
        'doc_binary_data': {},

        # Current vector store
        'vector_store_docs': None,
    }

    for key, default_value in required_keys.items():
        session_state[key] = default_value

    utils.create_history('docs')


def open_pdf(binary_data):
    if binary_data:
        doc = fitz.open(filetype='pdf', stream=binary_data)
    else:
        doc = None
    return doc


def get_chunk_size(chunk_size=300):
    if 'chunk_size' not in session_state:
        session_state['chunk_size'] = chunk_size
    return session_state['chunk_size']


def get_overlap_size(overlap_size=20):
    if 'overlap_size' not in session_state:
        session_state['overlap_size'] = overlap_size
    return session_state['overlap_size']


def chunk_text_with_bbox_and_overlap(doc, file_name):
    chunks_info = []
    pages_text = []
    chunk_counter = 1  # Initialize chunk counter
    char_limit = get_chunk_size()
    overlap = get_overlap_size()

    for page_number, page in enumerate(doc, start=1):
        words = page.get_text("words")  # Extract words and positions
        chunk_text = ''
        chunk_bbox = [float('inf'), float('inf'), 0, 0]  # Init with extreme values

        for i, word in enumerate(words):
            x0, y0, x1, y1, word_text, _, _, _ = word
            proposed_chunk = f"{chunk_text} {word_text}".strip()

            # If adding the next word exceeds the char limit and chunk already contains some text
            if len(proposed_chunk) > char_limit and chunk_text:
                source = format_chunk_info(page_number, chunk_text, chunk_bbox, file_name, chunk_counter)
                chunks_info.append(source)
                chunk_counter += 1
                chunk_text = ' '.join(
                    proposed_chunk.split(' ')[-overlap:])  # Keep last 'overlap' words for next chunk
                chunk_bbox = [x0, y0, x1, y1]
            else:
                chunk_text = proposed_chunk
                chunk_bbox = [
                    min(chunk_bbox[0], x0),
                    min(chunk_bbox[1], y0),
                    max(chunk_bbox[2], x1),
                    max(chunk_bbox[3], y1)
                ]

        # Append last chunk if not empty
        if chunk_text:
            source = format_chunk_info(page_number, chunk_text, chunk_bbox, file_name, chunk_counter)
            chunks_info.append(source)
            pages_text.append(chunk_text)

    return chunks_info, pages_text


def format_chunk_info(page_number, chunk_text, chunk_bbox, file_name, chunk_counter):
    x0, y0, x1, y1 = chunk_bbox
    return {
        'file_name': file_name,
        'chunk_number': chunk_counter,
        'page': page_number,
        'x0': x0,
        'y0': y0,
        'x1': x1,
        'y1': y1,
        'text': chunk_text  # Including text for reference
    }


def parse_pdf(file: BytesIO, file_name):
    doc = open_pdf(file)
    return chunk_text_with_bbox_and_overlap(doc, file_name)


def check_amount_of_uploaded_files_and_set_variables():
    reset_docs_variables()

    if len(session_state['pdf_files']) > MAX_FILES:
        maximum_text = session_state['_']("Maximum number of files reached")
        file_processed_text = session_state['_']("Number of files which will be processed")
        st.sidebar.warning(
            f"{maximum_text}. {file_processed_text}: {MAX_FILES}")

    # Variable to count files to avoid going over the MAX_FILES limit
    count_files = 0
    # If there are uploaded files then use class list variable to store them
    if session_state['pdf_files']:
        for file in session_state['pdf_files']:
            # Extract the names of already uploaded files
            uploaded_file_names = [f.name for f_id, f in session_state['uploaded_pdf_files'].items()]
            # Check if the file name is not in the list of already uploaded file names
            if file.name not in uploaded_file_names:  # Avoid duplicate files by name
                # Only append the maximum number of files allowed
                if count_files < MAX_FILES:
                    count_files += 1
                    file_text = session_state['_']("File")
                    file_text += f" {count_files}"
                    session_state['uploaded_pdf_files'][file_text] = file
                else:
                    break
