import streamlit as st
import gettext
from streamlit import session_state as ss
from src.chatbot_utils import SidebarManager


def initialize_language():
    languages = {"English": "en", "Deutsch": "de"}

    def change_language():
        if ss["selected_language"] == 'English':
            set_language(language='en')
            ss['_'] = gettext.gettext
        else:
            set_language(language='de')
            ss['_'] = translate()

    if 'selected_language' not in st.session_state:
        set_language(language='de')
        ss['_'] = translate()

    st.radio(
        "Language",
        options=languages,
        horizontal=True,
        key="selected_language",
        on_change=change_language,
        index=1,
        label_visibility='hidden'
    )

    SidebarManager.load_prompts()


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


def get_translate():
    """
    Get the translation function based on the selected language or query parameters.
    return: The translation function.
    """
    if "selected_language" in st.session_state:
        if st.session_state["selected_language"] == 'en':
            set_language(language='en')
            _ = translate()
        else:
            set_language(language='de')
            _ = gettext.gettext
    elif st.query_params.get('lang', None):
        if st.query_params["lang"] == "de":
            st.session_state["selected_language"] = 'de'
            _ = translate()
        elif st.query_params["lang"] == "en":
            st.session_state["selected_language"] = 'en'
            _ = gettext.gettext
    else:
        st.session_state["selected_language"] = 'de'
        set_language(language='de')
        _ = gettext.gettext
    return _
