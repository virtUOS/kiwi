import streamlit as st
from streamlit import session_state

import src.streamlit_styling as st_styling
import src.docs_utils as docs_utils


class SidebarDocsManager:

    def __init__(self):
        super().__init__()

    @staticmethod
    def display_docs_sidebar_controls():
        with st.sidebar:
            st.markdown("""---""")
            st.write(session_state['_']("**Options**"))
            st_styling.style_language_uploader(lang=session_state['selected_language'])
            st.file_uploader(session_state['_']("**Upload PDF file(s)**"), type=['pdf'], accept_multiple_files=True,
                             key='pdf_files', on_change=docs_utils.check_amount_of_uploaded_files_and_set_variables)

            with st.expander(session_state['_']("Advanced Options")):
                st.number_input(
                    label=session_state['_']("Input chunk size"),
                    min_value=100, max_value=4000, value=300, key='chunk_size',
                    help=(
                        session_state['_'](
                            "The chunk size specifies the number of characters in each chunk of text. For example, "
                            "with a chunk size of 300, the text will be divided into pieces of approximately 300 "
                            "characters each. Adjust this based on the level of detail needed.\n\n"
                            "**Why it's important**: A larger chunk size retains more context but might "
                            "be less detailed. A smaller chunk size provides finer detail but may lose some "
                            "context across chunks."
                        )
                    )
                )

                st.number_input(
                    label=session_state['_']("Overlap size"),
                    min_value=0, max_value=3000, value=20, key='overlap_size',
                    help=(
                        session_state['_'](
                            "The overlap size specifies the number of characters that will overlap between "
                            "consecutive chunks. For example, with an overlap size of 20, the last 20 characters "
                            "of one chunk will be the first 20 characters of the next chunk.\n\n**Why it's important**:"
                            " The overlap helps maintain context between chunks. Without adequate overlap, "
                            "some important details may be lost between chunks. Too much overlap "
                            "can lead to redundant information."
                        )
                    )
                )



