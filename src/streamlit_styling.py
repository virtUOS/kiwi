import streamlit as st


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


def style_language_uploader(lang):

    languages = {
        "English": {
            "instructions": "Drag and drop files here",
            "limits": "Limit 200MB per file",
        },
        "Deutsch": {
            "instructions": "Dateien hierher ziehen und ablegen",
            "limits": "Limit 200MB pro Datei",
        },
    }

    hide_label = (
        """
    <style>
        div[data-testid="stFileUploaderDropzoneInstructions"]>div>span {
           visibility:hidden;
        }
        div[data-testid="stFileUploaderDropzoneInstructions"]>div>span::after {
           content:"INSTRUCTIONS_TEXT";
           visibility:visible;
           display:block;
        }
         div[data-testid="stFileUploaderDropzoneInstructions"]>div>small {
           visibility:hidden;
        }
        div[data-testid="stFileUploaderDropzoneInstructions"]>div>small::before {
           content:"FILE_LIMITS";
           visibility:visible;
           display:block;
        }
    </style>
    """.replace("INSTRUCTIONS_TEXT", languages.get(lang).get("instructions"))
        .replace("FILE_LIMITS", languages.get(lang).get("limits"))
    )

    st.markdown(hide_label, unsafe_allow_html=True)


def add_custom_css():
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
