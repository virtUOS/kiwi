import streamlit as st

import src.docs_utils as docs_utils


class SidebarDocsManager:

    def __init__(self):
        super().__init__()

    @staticmethod
    def display_docs_sidebar_controls():
        with st.sidebar:
            st.markdown("""---""")
            st.file_uploader("Upload PDF file(s)", type=['pdf'], accept_multiple_files=True,
                             key='pdf_files', on_change=docs_utils.check_amount_of_uploaded_files_and_set_variables)
