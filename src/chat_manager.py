import base64
import requests

from PIL import Image
from io import BytesIO

from streamlit import session_state
from streamlit_float import *
from streamlit_image_gallery import streamlit_image_gallery

from src import menu_utils
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ChatManager:

    def __init__(self, user, advanced_model, sidebar_manager):
        """
        Initializes the ChatManager instance with the user's identifier.

        Parameters:
        - user: A string identifier for the user, used to differentiate messages in the conversation.
        """
        self.client = None
        session_state['USER'] = user
        self.advanced_model = advanced_model
        self.sbm = sidebar_manager

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

        This method sets the 'images_state' flag in the session state to the corresponding value.
        It serves as the callback for the images' widgets to indicate that new images have been uploaded.
        """
        # If the user already interacted with images we want to reset all of them
        if session_state['images_state'] == 1 or session_state['images_state'] == -1:
            session_state['images_state'] = 0
            session_state['current_image_urls'] = []
            session_state['current_uploaded_images'] = []
            session_state['current_photo_to_use'] = []

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
        # Only add images on the first message after uploading images
        if session_state['images_state'] == 0:
            current_history.append((session_state['USER'], user_message, description_to_use, query_images))
        else:
            current_history.append((session_state['USER'], user_message, description_to_use, {}))

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
    def _convert_image_to_base64(image):
        """
        Convert an image to a Base64 string.

        Parameters:
        - image: The image to be converted. This can be a URL (HTTP/HTTPS) or a Data URL.

        Returns:
        - img_str (str): The Base64 string representation of the image.
        """
        if isinstance(image, str):
            if image.startswith('data:image'):
                # Handle Data URL
                img_str = image.split(',', 1)[1]
                img_data = base64.b64decode(img_str)
            else:
                # Handle regular URL
                img_data = requests.get(image).content

            # Open image from bytes
            img = Image.open(BytesIO(img_data))
        else:
            img = Image.open(image)

        # TODO: Should we send the reduced image to the model instead of the original-sized one?
        #  The advantage of the reduced ones are faster answers and it seems to work better
        #  for the model to see some details on reduced images that are lost on original-sized ones (ex. illusions)
        # resized_img = img.resize(thumbnail_size, Image.Resampling.BILINEAR)

        # Convert RGBA to RGB if necessary
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        # Save the image to a bytes buffer
        buffer = BytesIO()
        img.save(buffer, format="JPEG")

        # Get the base64 encoded string
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{img_str}"

        return image_data_url

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

    def _handle_user_input(self, description_to_use, images, current_conversation_history):
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
            # Ensures unique submission to prevent re-running on refresh
            if self._is_new_message(current_conversation_history, user_message):
                self._append_user_message_to_history(current_conversation_history,
                                                     user_message,
                                                     images,
                                                     description_to_use)

                # Print user message immediately after getting entered because we're streaming the chatbot output
                with st.chat_message("user"):
                    st.markdown(user_message)

                # Process and display response
                self._update_conversation_history(current_conversation_history)
                self._process_response(current_conversation_history, user_message, description_to_use)
                # Set images' widgets to reset when uploading new images after the user interacted
                if images:
                    self.sbm.reset_images_widgets()
                    session_state['images_state'] = 1  # Now we know the user has interacted with the images
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
    def _display_images_inside_message(images):
        """
        Displays images associated with a user's message within an expandable section.

        Parameters:
        - images (dict): A dictionary mapping image names to image objects or URLs.
            Example: {"image1": <image_object_or_url>, "image2": <image_object_or_url>}
        to be displayed. If None, will use the main app area.
        """
        # Check if there are images associated with this message
        if images:
            images_list = []
            for i, (name, image) in enumerate(images.items()):
                images_list.append({
                    'title': f"{i + 1} - {name}",
                    'src': image
                })

            col1, _ = st.columns(2)
            with col1:
                with st.chat_message("user"):
                    streamlit_image_gallery(images_list, max_width='100%')

    def _display_conversation(self, conversation_history):
        """Displays the conversation history between the user and the assistant within the given container or globally.

        Parameters:
        - conversation_history: A list containing tuples of (speaker, message) representing the conversation.
        If None, messages will be displayed in the main app area.
        """
        for speaker, message, _, images in conversation_history:
            if speaker == session_state['USER']:
                self._display_images_inside_message(images)
                with st.chat_message("user"):
                    st.write(message)

            elif speaker == "Assistant":
                with st.chat_message("assistant"):
                    st.write(message)

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

    def _fill_up_image_content(self, uploaded_images, image_urls, photo_to_use):
        """
        Fills image content with uploaded images, image URLs, and user photo.

        Parameters:
        - uploaded_images (list): List of uploaded images.
        - image_urls (list): List of image URLs.
        - photo_to_use (UploadedFile): Photo captured via camera input.

        Returns:
        - images_dict (dict): A dictionary containing all images to populate alongside
          the conversation history.
        """
        # Create a new dictionary to hold all images for populating alongside the conversation history
        images_dict = {}

        if uploaded_images:
            for image in uploaded_images:
                image64 = self._convert_image_to_base64(image)
                images_dict[image.name] = image64
                session_state['image_content'].append({
                    'type': "image_url",
                    'image_url': {"url": image64}
                })

        if image_urls:
            for i, url in enumerate(image_urls):
                url64 = self._convert_image_to_base64(url)
                images_dict[f"url-{i}"] = url64
                session_state['image_content'].append({
                    'type': "image_url",
                    'image_url': {"url": url64}
                })

        if photo_to_use:
            photo64 = self._convert_image_to_base64(photo_to_use)
            images_dict['photo'] = photo64
            session_state['image_content'].append({
                'type': "image_url",
                'image_url': {"url": photo64}
            })

        # Return dictionary of images for the chat history
        return images_dict

    def _display_camera(self):
        """
        Renders the camera input widget and displays the captured photo.

        This method creates a camera input widget within a centered column.
        If the photo is captured, it provides an option to use the photo and clears
        existing photos if the 'Use photo' button is pressed.
        """
        col3, col4, col5 = st.columns([1, 1, 1])
        with col4:
            float_parent(f"bottom: 20rem; background-color: var(--default-backgroundColor); padding-top: 1rem;")
            st.camera_input(
                session_state['_']("Take a photo"),
                key='your_photo',
                label_visibility="hidden")
            if session_state['your_photo']:
                st.button("Use photo",
                          key='use_photo_button',
                          use_container_width=True,
                          on_click=self._upload_images_callback)

        if session_state['your_photo']:
            if session_state['use_photo_button']:
                session_state['photo_to_use'] = session_state['your_photo']
                session_state['activate_camera'] = False
                session_state['toggle_camera_label'] = "Activate camera"
                st.rerun()

    @staticmethod
    def _count_images():
        if session_state['uploaded_images']:
            counter_images = len(session_state['uploaded_images'])
        else:
            counter_images = len(session_state['current_uploaded_images'])
        if session_state['image_urls']:
            counter_images += len(session_state['image_urls'])
        else:
            counter_images += len(session_state['current_image_urls'])
        if session_state['photo_to_use'] or session_state['current_photo_to_use']:
            counter_images += 1
        return counter_images

    def _display_chat_buttons(self):
        """
        Displays a set of interactive buttons within the chat interface, allowing users to:
        - Activate or deactivate the camera.
        - Upload images.
        - Delete the conversation history.
        """
        container_camera = st.container()
        container_images_controls = st.container()
        container_clear_images_button = st.container()

        if 'selected_model' in session_state and session_state['selected_model'] == self.advanced_model:
            with container_camera:
                float_parent("margin-left: 0rem; bottom: 6.9rem;background-color: var(--default-backgroundColor); "
                             "padding-top: 0.9rem;")

                st.button("📷",
                          key='activate_camera_key',
                          on_click=self._activate_camera_callback,
                          help=session_state['_'](session_state['toggle_camera_label'])
                          )

            with container_images_controls:
                float_parent("margin-left: 3.3rem; bottom: 6.9rem;background-color: var(--default-backgroundColor); "
                             "padding-top: 0.9rem;")

                with st.popover("🖼️", help=session_state['_']("Images")):
                    session_state['uploaded_images'] = st.file_uploader(session_state['_']("Upload Images"),
                                                                        type=['png', 'jpeg', 'jpg', 'gif', 'webp'],
                                                                        accept_multiple_files=True,
                                                                        key=session_state['images_key'],
                                                                        on_change=self._upload_images_callback)

                    urls = st.text_area(session_state['_']("Enter Image URLs (one per line)"),
                                        key=session_state['urls_key'])

                    col1, col2 = st.columns([1, 1])

                    if col1.button(session_state['_']("Add URLs"), on_click=self._upload_images_callback):
                        # Process the URLs
                        url_list = urls.strip().split('\n')
                        session_state['image_urls'].extend([url.strip() for url in url_list if url.strip()])

            if session_state['images_state'] >= 0:
                counter_images = self._count_images()
                if counter_images == 1:
                    clear_images_label = session_state['_']("Clear image 🧹")
                else:
                    clear_images_label = session_state['_']("Clear images 🧹")

                with container_clear_images_button:
                    float_parent(
                        "margin-left: 8rem; bottom: 6.9rem;background-color: var(--default-backgroundColor); "
                        "padding-top: 0.9rem;")
                    st.button(clear_images_label + f": {counter_images}", on_click=self.sbm.clear_images_callback)

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
                if session_state.get('activate_camera', False):
                    session_state['toggle_camera_label'] = "Deactivate camera"
                    self._display_camera()
                else:
                    session_state['toggle_camera_label'] = "Activate camera"

                # Display buttons above chat widget. Put it before checking for uploaded images
                # to help app flow
                self._display_chat_buttons()

                # Initialize variables for uploaded content
                uploaded_images = session_state.get('uploaded_images', [])
                if uploaded_images:
                    session_state['current_uploaded_images'] = uploaded_images

                image_urls = session_state.get('image_urls', [])
                if image_urls:
                    session_state['current_image_urls'] = image_urls

                photo_to_use = session_state.get('photo_to_use', [])
                if photo_to_use:
                    session_state['current_photo_to_use'] = photo_to_use

                # For each interaction, we want to always refill the image content
                # or leave it empty in case of no images so we reinitialize it every time.
                session_state['image_content'] = []

                # Display chat interface header and model information if applicable
                self._display_chat_interface_header()

                # Option to view or edit the system prompt
                with st.expander(label=session_state['_']("View or edit system prompt"), expanded=False):
                    self._display_prompt_editor(description)

                st.markdown("""---""")

                description_to_use = self._get_description_to_use(description)

                # Displays the existing conversation history
                self._display_conversation(current_history)

                # If there are images, fill up image info and show them
                images_dict = {}
                if session_state['images_state'] >= 0:
                    images_dict = self._fill_up_image_content(session_state['current_uploaded_images'],
                                                              session_state['current_image_urls'],
                                                              session_state['current_photo_to_use'])
                    # We want to display the images just once in the chat area
                    if session_state['images_state'] == 0:
                        self._display_images_inside_message(images_dict)

                # Handles the user's input and interaction with the LLM
                self._handle_user_input(description_to_use,
                                        images_dict,
                                        current_history)

                # Adds empty lines to the main app area to avoid hiding text behind chat buttons
                # (might need adjustments)
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
            else:
                st.error(session_state['_']("The System Prompt should be a string and not empty."))
