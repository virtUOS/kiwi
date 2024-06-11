import streamlit as st


def apply_styling():
    """
    Apply custom styling to various Streamlit components.

    This function calls other styling functions to customize the appearance
    of specific Streamlit components within the app.
    """
    style_logo()
    style_language_uploader()
    style_text_input()


def style_logo():
    """
    Gets rid of the full screen option for images. It's mainly applied for the logo image.

    This function hides the button that allows viewing the images in full screen.
    It injects custom CSS into the Streamlit app to achieve this.
    """
    hide_img_fs = '''
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    '''
    st.markdown(hide_img_fs, unsafe_allow_html=True)


def style_language_uploader():
    """
    Styles the file uploader component based on the selected language.

    This function customizes the instructions and limits text shown in the file
    uploader based on the user's language preference.
    It supports English and German language instructions.

    The style is applied using specific Streamlit DOM elements.

    It uses a simple data structure to switch between languages and generate the appropriate HTML
    and CSS modifications.

    Supported languages:
    - English ("en")
    - German ("de")

    Changes the visibility of default text and replaces it with translated instructions and file limits.
    """
    lang = 'de'
    if 'lang' in st.query_params:
        lang = st.query_params['lang']

    languages = {
        "en": {
            "instructions": "Drag and drop files here",
            "limits": "Limit 20MB per file",
        },
        "de": {
            "instructions": "Dateien hierher ziehen und ablegen",
            "limits": "Limit 20MB pro Datei",
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


def style_text_input():
    """
    Hides the default submit text in the Streamlit text area component.

    This method injects custom CSS into the Streamlit app to hide the default submit text
    that appears in the text area component.
    This enhances the user interface by removing unnecessary visual elements.
    """
    hide_submit_text = """
    <style>
    div[data-testid="InputInstructions"] > span:nth-child(1) {
        visibility: hidden;
    }
    </style>
    """
    st.markdown(hide_submit_text, unsafe_allow_html=True)
