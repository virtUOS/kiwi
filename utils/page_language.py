import streamlit as st
import gettext

languages = {
    'English': 'en',
    'German': 'de',
}



def translate() :
    # translation
    de = gettext.translation('base', localedir='locale', languages=['de'])
    de.install()
    _ = de.gettext
    return _




def set_default_language() -> None:
    if "lang" not in st.experimental_get_query_params():
        st.experimental_set_query_params(lang="en")
        st.experimental_rerun()


def set_language() -> None:
    if "selected_language" in st.session_state:
        st.experimental_set_query_params(
            lang=languages.get(st.session_state["selected_language"])
        )