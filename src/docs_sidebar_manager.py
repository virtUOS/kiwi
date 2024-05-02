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
            st_styling.style_language_uploader()
            st.file_uploader("Upload PDF file(s)", type=['pdf'], accept_multiple_files=True,
                             key='pdf_files', on_change=docs_utils.check_amount_of_uploaded_files_and_set_variables)

            with st.expander(session_state['_']("Advanced Options")):
                st.number_input(
                    label='Input chunk size',
                    min_value=100, max_value=4000, value=300, key='chunk_size',
                    help=(session_state['_']("The chunk size specifies the amount of characters each "
                          "chunk of text contains. Adjust based on the level of detail needed."))
                )

                st.number_input(
                    label=session_state['_']('Overlap size'),
                    min_value=0, max_value=3000, value=20, key='overlap_size',
                    help=session_state['_']("The overlap size specifies the character count that will overlap "
                                            "between consecutive chunks. It helps in maintaining context "
                                            "connectivity across chunks. Adjust based on the required continuity.")
                )



