import streamlit as st
import gettext
from streamlit import session_state as ss


def initialize_language():
    languages = {"English": "en", "Deutsch": "de"}

    def change_language():
        # ss['changed_language'] = True
        if ss["selected_language"] == 'English':
            set_language(language='en')
            ss['_'] = gettext.gettext
        else:
            set_language(language='de')
            ss['_'] = translate()

    # If no language is chosen yet set it to German
    if 'selected_language' not in st.session_state or 'lang' not in st.query_params:
        ss['_'] = translate()
        st.query_params['lang'] = 'de'

    st.radio(
        "Language",
        options=languages,
        horizontal=True,
        key="selected_language",
        on_change=change_language,
        index=1,
        label_visibility='hidden'
    )

    # When changing the language force a rerun to reflect all changes
#    if ss['changed_language']:
#        ss['changed_language'] = False
#        st.rerun()


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
