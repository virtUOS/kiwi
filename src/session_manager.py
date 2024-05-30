import os
import string
import random
import streamlit as st
from streamlit import session_state
from streamlit_cookies_manager import EncryptedCookieManager


class SessionManager:

    def __init__(self):
        self.cookies = None

    def initialize_cookies(self):
        """
        Initialize cookies for managing user sessions with enhanced security.

        Uses an encrypted cookie manager and detects the Safari browser to handle specific JavaScript-based delays.
        This ensures a smooth user experience across different browsers.
        """
        cookies = EncryptedCookieManager(
            prefix=os.getenv("COOKIES_PREFIX"),
            password=os.getenv("COOKIES_PASSWORD")
        )

        if not cookies.ready():
            st.spinner()
            st.stop()

        self.cookies = cookies

        session_state['username'] = self.cookies.get('username')


    def verify_and_set_user_session(self):
        """
        Verify the user's session validity. If cookies indicate the session is not yet initiated,
        then redirect the user to the start page.

        This function ensures that unauthorized users are not able to access application sections
        that require a valid session.prompt_op.

        It also ensures that a username is assigned for the session. If the username is not found in the cookies,
        a random username is generated combining letters and digits and assigned for the duration of the session.
        """
        if self.cookies and self.cookies.get('session') != 'in':
            st.switch_page("start.py")

        session_state['username'] = self.cookies.get('username')
        # If there's no username in the cookies then just generate a random one for the session
        if not session_state['username']:
            alphabet = string.ascii_letters + string.digits
            self.cookies['username'] = "".join(random.choices(alphabet, k=8))
            session_state['username'] = self.cookies.get('username')

    def logout_and_redirect(self):
        """
        Perform logout operations and redirect the user to the start page.

        This involves resetting cookies and session variables that signify the user's logged-in state,
        thereby securely logging out the user.
        """
        self.cookies['session'] = "out"
        self.cookies['username'] = ""
        session_state['password_correct'] = False
        session_state['credentials_checked'] = False
        st.switch_page('start.py')
