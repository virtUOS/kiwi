import streamlit as st
import gettext

# languages = {
#     'English': 'en',
#     'Deutsch': 'de',
# }

languages = {
    'English': '🇬🇧',
    'Deutsch': '🇩🇪'

}


def translate() :
    # translation
    de = gettext.translation('base', localedir='locale', languages=['de'])
    de.install()
    _ = de.gettext
    return _




# def set_default_language() -> None:
#     if not st.query_params.get("lang", ''):
#         st.query_params["lang"] = "en"
        # st.experimental_rerun()

# def set_default_language() -> None:
#     if "lang" not in st.experimental_get_query_params():
#         st.experimental_set_query_params(lang="en")
#         st.experimental_rerun()
#

def set_language(language) -> None:
    """ Add the language to the query parameters based on the selected language. """
    if language == 'en':
            st.query_params["lang"] = "en"
    elif language == 'de':
            st.query_params["lang"] = "de"
