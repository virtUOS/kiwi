import base64
from PIL import Image
from io import BytesIO

from streamlit import session_state
from streamlit_float import *
from src import menu_utils
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


class ChatManager:

    def __init__(self, user, advanced_model):
        """
        Initializes the ChatManager instance with the user's identifier.

        Parameters:
        - user: A string identifier for the user, used to differentiate messages in the conversation.
        """
        self.client = None
        session_state['USER'] = user
        self.advanced_model = advanced_model

    def set_client(self, client):
        """
        Sets the client for API requests, typically used to interface with chat models.

        Parameters:
        - client: The client instance responsible for handling requests to chat models.
        """
        self.client = client

    @staticmethod
    def _upload_images_callback():
        """
        Static callback method to handle image upload events.

        This method sets the 'new_images' flag in the session state to True.
        It serves as the callback for the file uploader to indicate that new images have been uploaded.
        """
        session_state['new_images'] = True

    @staticmethod
    def _clear_photo_callback():
        """
        Callback function to clear the photo to use.

        This method sets the 'photo_to_use' session state to an empty list, effectively clearing
        any photo that was selected for use within the session.
        """
        session_state['photo_to_use'] = []

    @staticmethod
    def _activate_camera_callback():
        """
        Toggle the camera activation within the session state.

        This callback method toggles the 'activate_camera' state between True and False,
        and clears the 'photo_to_use' state.
        """
        if not session_state['activate_camera']:
            session_state['activate_camera'] = True
        else:
            session_state['activate_camera'] = False
            session_state['photo_to_use'] = []

    @staticmethod
    def _fetch_chatbot_description():
        """
        Fetches the description for the current chatbot based on the selected path.

        Returns:
        - A string containing the description of the current chatbot.
        """
        return menu_utils.get_final_description(session_state["selected_chatbot_path"], session_state["prompt_options"])

    @staticmethod
    def _get_description_to_use(default_description):
        """
        Retrieves and returns the current prompt description for the chatbot, considering any user edits.

        Parameters:
        - default_description: The default description provided for the chatbot's prompt.

        Returns:
        - The current (possibly edited) description to be used as the prompt.
        """
        return session_state['edited_prompts'].get(session_state[
                                                       'selected_chatbot_path_serialized'], default_description)

    @staticmethod
    def _is_new_message(current_history, user_message):
        """
        Checks if the last message in history differs from the newly submitted user message.

        Parameters:
        - current_history: The current conversation history.
        - user_message: The newly submitted user message.

        Returns:
        - A boolean indicating whether this is a new unique message submission.
        """
        if (not current_history or current_history[-1][0] != session_state['USER']
                or current_history[-1][1] != user_message):
            return True
        return False

    @staticmethod
    def _get_current_conversation_history():
        """
        Retrieves the current conversation history for the selected chatbot path from the session state.

        Returns:
        - A list representing the conversation history.
        """
        serialized_path = session_state['selected_chatbot_path_serialized']
        if serialized_path not in session_state['conversation_histories']:
            session_state['conversation_histories'][serialized_path] = []
        return session_state['conversation_histories'][serialized_path]

    @staticmethod
    def _append_user_message_to_history(current_history, user_message, query_images, description_to_use):
        """
        Appends the user's message to the conversation history in the session state.

        Parameters:
        - current_history: The current conversation history.
        - user_message: The message input by the user.
        - query_images: Images sent alongside the user message.
        - description_to_use: The system description or prompt that should accompany user messages.
        """
        # Adding user message and description to the current conversation
        current_history.append((session_state['USER'], user_message, description_to_use, query_images))

    @staticmethod
    def _update_conversation_history(current_history):
        """
        Updates the session state with the current conversation history.

        This method ensures that the conversation history for the currently selected chatbot path in the session state
        is updated to reflect any new messages or responses that have been added during the interaction.

        Parameters:
        - current_history: The current conversation history, which is a list of tuples. Each tuple contains
          the speaker ('USER' or 'Assistant'), the message, and optionally, the system description or prompt
          accompanying user messages (for user messages only).
        """
        session_state['conversation_histories'][session_state['selected_chatbot_path_serialized']] = current_history

    @staticmethod
    def _update_edited_prompt():
        """
        Updates the edited prompt in the session state to reflect changes made by the user.
        """
        session_state['edited_prompts'][session_state[
            'selected_chatbot_path_serialized']] = session_state['edited_prompt']

    @staticmethod
    def _restore_prompt():
        """
        Restores the original chatbot prompt by removing any user-made edits from the session state.
        """
        if session_state['selected_chatbot_path_serialized'] in session_state['edited_prompts']:
            del session_state['edited_prompts'][session_state['selected_chatbot_path_serialized']]

    @staticmethod
    def _resize_image_and_get_base64(image, thumbnail_size):
        """
        Resize an image and convert it to a Base64 string.

        Parameters:
        - image: The image to be resized.
        - thumbnail_size (tuple): The desired dimensions for the thumbnail as a tuple (width, height).

        Returns:
        - img_str (str): The Base64 string representation of the resized image.
        - resized_img (PIL.Image.Image): The resized image object.

        Notes:
        - This method opens the provided image, resizes it to the specified dimensions,
          and converts it to Base64 format. If the image is in RGBA mode, it is converted
          to RGB format before resizing.
        """
        img = Image.open(image)
        resized_img = img.resize(thumbnail_size, Image.Resampling.BILINEAR)

        # Convert RGBA to RGB if necessary
        if resized_img.mode == 'RGBA':
            resized_img = resized_img.convert('RGB')

        buffer = BytesIO()
        resized_img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str, resized_img

    def _process_response(self, current_history, user_message, description_to_use):
        """
        Submits the user message to OpenAI, retrieves the response, and updates the conversation history.

        Parameters:
        - current_history: The current conversation history.
        - user_message: The message input by the user.
        - description_to_use: The current system description or prompt associated with the user message.
        """
        response = self.client.get_response(user_message, description_to_use)
        # Add AI response to the history
        current_history.append(('Assistant', response, "", ""))

    def _handle_user_input(self, description_to_use, images, current_conversation_history, container_chat):
        """
        Handles the user input: sends the message to OpenAI, prints it, and updates the conversation history.

        This method is responsible for capturing the user input from the chat interface, sending it to the
        language model for a response, displaying the user message immediately, and updating the conversation
        history to include both the user message and the model's response. It ensures that the user input is
        processed only once per new message.

        Parameters:
        - description_to_use (str): The current system description or prompt used alongside user messages.
        - container_chat (streamlit.container): The Streamlit container where the chat messages are displayed.
        """
        if user_message := st.chat_input(session_state['USER']):

            with container_chat:
                # Ensures unique submission to prevent re-running on refresh
                if self._is_new_message(current_conversation_history, user_message):
                    self._append_user_message_to_history(current_conversation_history,
                                                         user_message,
                                                         images,
                                                         description_to_use)

                    # Print user message immediately after getting entered because we're streaming the chatbot output
                    with st.chat_message("user"):
                        st.markdown(user_message)
                        self._display_images_inside_message(images)

                    # Process and display response
                    self._update_conversation_history(current_conversation_history)
                    self._process_response(current_conversation_history, user_message, description_to_use)
                    st.rerun()

    def _display_prompt_editor(self, description):
        """
        Displays an editable text area for the current chatbot prompt, allowing the user to make changes.

        Parameters:
        - description: The default chatbot prompt description which can be edited by the user.
        """
        current_chatbot_path_serialized = session_state['selected_chatbot_path_serialized']
        current_edited_prompt = session_state['edited_prompts'].get(current_chatbot_path_serialized, description)

        st.text_area(session_state['_']("System prompt"),
                     value=current_edited_prompt,
                     on_change=self._update_edited_prompt,
                     key='edited_prompt',
                     label_visibility='hidden')

        st.button("🔄", help=session_state['_']("Restore Original Prompt"), on_click=self._restore_prompt)

    @staticmethod
    def _display_images_inside_message(images, user_message_container=None):
        """
        Displays images associated with a user's message within an expandable section.

        Parameters:
        - images (dict): A dictionary mapping image names to image objects or URLs.
            Example: {"image1": <image_object_or_url>, "image2": <image_object_or_url>}
        - user_message_container (streamlit.delta_generator.DeltaGenerator): The container where the images are
        to be displayed. If None, will use the main app area.
        """
        # Check if there are images associated with this message
        if images:
            user_message_container = user_message_container if user_message_container else st
            with user_message_container.expander(session_state['_']("Images")):
                for i, (name, image) in enumerate(images.items()):
                    st.write(f"{i + 1} - {name}")
                    if isinstance(image, str):  # It's an image URL
                        st.write(image)
                    else:  # It's an image object
                        st.image(image)

    def _display_conversation(self, conversation_history, container=None):
        """Displays the conversation history between the user and the assistant within the given container or globally.

        Parameters:
        - conversation_history: A list containing tuples of (speaker, message) representing the conversation.
        - container: The Streamlit container (e.g., column, expander) where the messages should be displayed.
        If None, messages will be displayed in the main app area.
        """
        for speaker, message, __, images in conversation_history:
            chat_message_container = container if container else st
            if speaker == session_state['USER']:
                user_message_container = chat_message_container.chat_message("user")
                user_message_container.write(message)
                self._display_images_inside_message(images, user_message_container)

            elif speaker == "Assistant":
                chat_message_container.chat_message("assistant").write(message)

    @staticmethod
    def _display_openai_model_info():
        """
        Displays information about the current OpenAI model in use.
        """
        using_text = session_state['_']("You're using the following OpenAI model:")
        model_info = f"{using_text} **{session_state['selected_model']}**."
        st.write(model_info)
        st.write(session_state['_']("Each time you enter information,"
                                    " a system prompt is sent to the chat model by default."))

    def _display_chat_interface_header(self):
        """
        Displays headers and model information on the chat interface based on the user's selection.
        """
        st.markdown("""---""")  # Separator for layout
        if (session_state['selected_chatbot_path_serialized'] not in session_state['conversation_histories']
                or not session_state['conversation_histories'][session_state['selected_chatbot_path_serialized']]):
            st.header(session_state['_']("How can I help you?"))
            if session_state['model_selection'] == 'OpenAI':
                self._display_openai_model_info()

    def _display_images_column(self, uploaded_images, image_urls, photo_to_use):
        """
        Displays uploaded images, image URLs, and user photo in a specified column.

        Parameters:
        - uploaded_images (list): List of uploaded images.
        - image_urls (list): List of image URLs.
        - photo_to_use (UploadedFile): Photo captured via camera input.

        Returns:
        - images_dict (dict): A dictionary containing all images to populate alongside
          the conversation history.
        """
        thumbnail_dim = 100
        thumbnail_size = (thumbnail_dim, thumbnail_dim)

        # Create a new dictionary to hold all images for populating alongside the conversation history
        images_dict = {}

        if uploaded_images:
            st.markdown(session_state['_']("### Uploaded Images"))
            for image in uploaded_images:
                image64, resized_image = self._resize_image_and_get_base64(image, thumbnail_size)
                images_dict[image.name] = resized_image
                session_state['image_content'].append({
                    'type': "image_url",
                    'image_url': {"url": f"data:image/jpeg;base64,{image64}"}
                })
                st.image(image)

        if image_urls:
            st.markdown(session_state['_']("### Image URLs"))
            for i, url in enumerate(image_urls):
                images_dict[f"url-{i}"] = url
                session_state['image_content'].append({
                    'type': "image_url",
                    'image_url': {"url": url}
                })
                st.write(url)

        if photo_to_use:
            st.markdown(session_state['_']("### Photo"))
            photo64, resized_photo = self._resize_image_and_get_base64(photo_to_use, thumbnail_size)
            images_dict['photo'] = resized_photo
            session_state['image_content'].append({
                'type': "image_url",
                'image_url': {"url": f"data:image/jpeg;base64,{photo64}"}
            })
            st.image(photo_to_use)
            st.button(session_state['_']("Clear photo 🧹"), on_click=self._clear_photo_callback)

        # Return dictionary of images for the chat history
        return images_dict

    @staticmethod
    def _display_camera():
        """
        Renders the camera input widget and displays the captured photo.

        This method creates a camera input widget within a centered column.
        If the photo is captured, it provides an option to use the photo and clears
        existing photos if the 'Use photo' button is pressed.
        """
        col3, col4, col5 = st.columns([1, 1, 1])
        with col4:
            st.camera_input(
                session_state['_']("Take a photo"),
                key='your_photo',
                label_visibility="hidden")
            if session_state['your_photo']:
                st.button("Use photo",
                          key='use_photo_button',
                          use_container_width=True,)
            float_parent(f"bottom: 20rem; background-color: var(--default-backgroundColor); padding-top: 1rem;")
            if session_state['your_photo']:
                if session_state['use_photo_button']:
                    session_state['photo_to_use'] = session_state['your_photo']
                    session_state['activate_camera'] = False
                    st.rerun()

    def _display_chat_buttons(self):
        """
        Displays a set of interactive buttons within the chat interface, allowing users to:
        - Activate or deactivate the camera.
        - Upload images.
        - Delete the conversation history.
        """
        container_camera = st.container()
        container_images_controls = st.container()

        if 'selected_model' in session_state and session_state['selected_model'] == self.advanced_model:
            with container_camera:
                float_parent("margin-left: 0rem; bottom: 6.9rem;background-color: var(--default-backgroundColor); "
                             "padding-top: 0.9rem;")

                st.button("📷",
                          key='activate_camera_key',
                          on_click=self._activate_camera_callback,
                          help=session_state['_']("Activate camera"))

            with container_images_controls:
                float_parent("margin-left: 6rem; bottom: 6.9rem;background-color: var(--default-backgroundColor); "
                             "padding-top: 0.9rem;")

                with st.popover("🖼️", help=session_state['_']("Images")):

                    session_state['uploaded_images'] = st.file_uploader(session_state['_']("Upload Images"),
                                                                        type=['png', 'jpeg', 'jpg', 'gif', 'webp'],
                                                                        accept_multiple_files=True,
                                                                        key=session_state['images_key'],
                                                                        on_change=self._upload_images_callback)

                    if session_state['new_images']:
                        session_state['new_images'] = False
                        st.rerun()

                    urls = st.text_area(session_state['_']("Enter Image URLs (one per line)"))

                    if st.button(session_state['_']("Add URLs")):
                        # Process the URLs
                        url_list = urls.strip().split('\n')
                        session_state['image_urls'].extend([url.strip() for url in url_list if url.strip()])

    def display_chat_interface(self):
        """
        Displays the chat interface, manages the display of conversation history, and handles user input.

        This method sets up the interface for the chat, including:
        - Fetching and displaying the system prompt.
        - Updating session states as necessary.
        - Displaying conversation history if any.
        - Handling user input.
        - Displaying uploaded images and URLs if provided.

        It also provides an option to view or edit the system prompt through an expander in the layout.
        """
        if 'selected_chatbot_path' in session_state and session_state["selected_chatbot_path"]:
            current_history = self._get_current_conversation_history()
            description = self._fetch_chatbot_description()

            if isinstance(description, str) and description.strip():
                self._display_chat_buttons()

                if session_state.get('activate_camera', False):
                    self._display_camera()

                # Initialize variables for uploaded content
                uploaded_images = session_state.get('uploaded_images', [])
                image_urls = session_state.get('image_urls', [])
                photo_to_use = session_state.get('photo_to_use', [])

                # Set layout columns based on whether there are images or URLs
                if uploaded_images or image_urls or photo_to_use:
                    col1, col2 = st.columns([2, 1], gap="medium")
                else:
                    col1, col2 = st.container(), None

                # For each interaction, we want to always refill the image content
                # or leave it empty in case of no images so we reinitialize it every time.
                session_state['image_content'] = []

                with col1:
                    # Display chat interface header and model information if applicable
                    self._display_chat_interface_header()

                    # Option to view or edit the system prompt
                    with st.expander(label=session_state['_']("View or edit system prompt"), expanded=False):
                        self._display_prompt_editor(description)

                    st.markdown("""---""")

                    description_to_use = self._get_description_to_use(description)

                    # Displays the existing conversation history
                    self._display_conversation(current_history, col1)

                # If col2 is defined, show uploaded images and URLs
                images_dict = {}
                if col2:
                    with col2:
                        images_dict = self._display_images_column(uploaded_images,
                                                                  image_urls,
                                                                  photo_to_use)

                # Handles the user's input and interaction with the LLM
                self._handle_user_input(description_to_use,
                                        images_dict,
                                        current_history,
                                        col1)

                # Adds empty lines to the main app area to avoid hiding text behind chat buttons
                # (might need adjustments)
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
            else:
                st.error(session_state['_']("The System Prompt should be a string and not empty."))
