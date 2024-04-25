import streamlit as st
import gettext
from streamlit import session_state


def translate():
    """
    Translate the text to German.
    return: The translation function using the German language.
    """
    de = gettext.translation('base', localedir='locale', languages=['de'])
    de.install()
    _ = de.gettext
    return _


def set_language(language) -> None:
    """
    Add the language to the query parameters based on the selected language.
    param language: The selected language.
    """
    if language == 'en':
        st.query_params["lang"] = "en"
    elif language == 'de':
        st.query_params["lang"] = "de"


def change_language():
    if session_state["selected_language"] == 'English':
        set_language(language='en')
        session_state['_'] = gettext.gettext
    else:
        set_language(language='de')
        session_state['_'] = translate()


class LanguageManager:

    def __init__(self):
        self.languages = {"English": "en", "Deutsch": "de"}

    @staticmethod
    def initialize_language():
        # If no language is chosen yet set it to German
        if 'selected_language' not in st.session_state or 'lang' not in st.query_params:
            session_state['_'] = translate()
            st.query_params['lang'] = 'de'

    def language_controls(self):
        with st.sidebar:
            st.radio(
                "Language",
                options=self.languages,
                horizontal=True,
                key="selected_language",
                on_change=change_language,
                index=1,
                label_visibility='hidden'
            )
