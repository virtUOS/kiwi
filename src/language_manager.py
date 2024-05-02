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
        self.current_index = 1  # Stores the current language's index for optimization

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
        selected_language = st.session_state.get('selected_language')
        if not selected_language:
            return "de"

        return self.languages.get(selected_language)

    def change_language(self):
        """
        Changes the application language based on user selection and updates session state accordingly.

        This method updates both the translation function and the current language index in the session
        state to reflect the user's choice. Also handles updating the language selection control in the UI.
        """
        print("On change", session_state['selected_language'], session_state['language_index'])
        selected_language = session_state['selected_language']
        if selected_language == 'English':
            update_language_in_session(gettext.gettext)
        else:
            update_language_in_session(translate())

        # Update the index in session state when it changes
        if selected_language != list(self.languages.keys())[session_state['language_index']]:
            session_state['language_index'] = list(self.languages.keys()).index(selected_language)
        print("On change", session_state['selected_language'], session_state['language_index'])

    def language_controls(self):
        """
        Creates language selection controls in the application sidebar.

        Renders a radio button interface for language selection in the sidebar, allowing the user
        to choose between supported languages. This selection automatically updates the application's
        displayed language through a callback on change.
        """
        # Use cached index if available, else find it
        if 'language_index' not in session_state:
            current_language = session_state.get('selected_language')
            if current_language and current_language in self.languages:
                self.current_index = list(self.languages.keys()).index(current_language)
            session_state['language_index'] = self.current_index

        with st.sidebar:
            st.radio(
                "Language",
                options=list(self.languages),
                horizontal=True,
                index=session_state['language_index'],
                key='selected_language',
                on_change=self.change_language,
                label_visibility='hidden'
            )

        print("Main",session_state['selected_language'],"\n")
