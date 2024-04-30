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


def change_language():
    if session_state["selected_language"] == 'English':
        st.query_params["lang"] = "en"
        session_state['_'] = gettext.gettext
    else:
        st.query_params["lang"] = "de"
        session_state['_'] = translate()


class LanguageManager:

    def __init__(self):
        self.languages = {"English": "en", "Deutsch": "de"}

    @staticmethod
    def initialize_language():
        # If no language is chosen yet set it to German
        if 'selected_language' not in session_state and 'lang' not in st.query_params:
            session_state['_'] = translate()
            st.query_params['lang'] = 'de'

    @staticmethod
    def get_language():
        return st.query_params.get('lang', False)

    def language_controls(self):
        index = 1

        # In case there's a language set maintain it between pages
        if 'selected_language' in session_state:
            current_language = session_state['selected_language']
            for k, v in self.languages.items():
                if k == current_language:
                    index = list(self.languages).index(k)

        with st.sidebar:
            st.radio(
                "Language",
                options=self.languages,
                horizontal=True,
                key="selected_language",
                on_change=change_language,
                index=index,
                label_visibility='hidden'
            )
