import streamlit as st
import gettext
from streamlit import session_state


def translate():
    """
    Instantiate a translation function for the German language.

    Retrieves the German translation object and sets the global translation function (_)
    to perform text content translation to German.

    Returns:
        function: The gettext translation function tailored for the German language.
    """
    de = gettext.translation('base', localedir='locale', languages=['de'])
    de.install()
    _ = de.gettext
    return _


def update_language_in_session(translation_func):
    """
    Updates the translation function in the session state.

    This function ensures that the text displayed in the application is translated
    according to the currently selected language by updating the translation function
    used throughout the application dynamically.

    Parameters:
        translation_func (function): The translation function to be used for text content.
    """
    session_state['_'] = translation_func


class LanguageManager:
    """
    Manages the language selection and setup for the application by providing functionality
    to initialize the default language, change languages, and render language selection controls.
    """

    def __init__(self):
        """
        Initializes the LanguageManager with supported languages.
        """
        self.languages = {"English": "en", "Deutsch": "de"}
        self.default_index = 1  # Stores the current language's index for optimization

    @staticmethod
    def initialize_language():
        """
        Initializes the application language settings.

        Sets the application's language to German (de) by default if no language
        selection has been explicitly made by the user. This default setting ensures
        that the application has a consistent language setting upon initial use.
        """
        # If no language is chosen yet set it to German
        if 'selected_language' not in session_state:
            session_state['_'] = translate()

    def get_language(self):
        """
        Retrieves the current language code from the session state based on the user's selection.

        Returns:
            str: The current language code if set (e.g., 'en', 'de'), with 'de' as the default value.
        """
        selected_language = session_state.get('selected_language')
        if not selected_language:
            return "de"

        return self.languages.get(selected_language)

    @staticmethod
    def change_language():
        """
        Changes the application's language based on the user's selection, updates the session state,
        and ensures that the translation function and current language index in the session state
        reflect the user's choice efficiently.

        This method accomplishes the following:
        1. Directly retrieves the current language selection from the session state.
        2. Checks if the selected language is indeed different from the current language.
           If not, the method exits, avoiding unnecessary updates.
        3. Depending on the selected language, updates the translation function appropriately.
        4. Updates the current language index in the session state if it has changed.
           This ensures that future UI elements and text translations align with the newly selected language.
        """
        # Directly fetch the currently selected language and compare it against the current session state.
        selected_language = session_state.get('selected_language')

        # Determine and set the appropriate translation function based on the selection.
        if selected_language == 'English':
            update_language_in_session(gettext.gettext)
        else:
            update_language_in_session(translate())

    def language_controls(self):
        """
        Creates language selection controls in the application sidebar.

        Renders a radio button interface for language selection in the sidebar, allowing the user
        to choose between supported languages. This selection automatically updates the application's
        displayed language through a callback on change.
        """

        with st.sidebar:
            st.radio(
                "Language",
                options=list(self.languages),
                horizontal=True,
                index=self.default_index,
                key='selected_language',
                on_change=self.change_language,
                label_visibility='hidden'
            )

