import os
import re
from urllib.parse import urlparse, parse_qs
import requests
from bs4 import BeautifulSoup
from pydub import AudioSegment

import yt_dlp
from pathlib import Path

from dotenv import load_dotenv

import streamlit as st
from streamlit import session_state
import streamlit_pills as stp
from streamlit_dimensions import st_dimensions

from src import videos_config

# Load environment variables
load_dotenv()


class SidebarVideosControls:

    def __init__(self, sidebar_general_manager, general_manager):
        self.sgm = sidebar_general_manager
        self.gm = general_manager
        self.client = None

    def set_client(self, client):
        self.client = client.client

    def initialize_videos_session_variables(self):
        """Initialize essential session variables."""
        self.sgm.load_prompts(prompt_options='prompt_options_videos', typ='vid', prompt_key='video_assistance')
        required_keys = {
            'current_video': None,
            'previous_video': None,
            'youtube': False,
            'selected_subtitles_file': None,
            'previous_selected_subtitles_file': None,
            'subtitles_file_list': None,
            'selected_subtitles_text': "",
            'input_disabled': False,
            'pills_index': None,
            'subtitles_files_folder': videos_config.TRANSCRIPTIONS_DIR,
            'video_title': None,
            'summary': None,
            'topics': None,
            'artist': None,
            'lyrics': None,
            'suggestions_list': [],
            'icons_list': [],
            'suggested': False,
            'changed_video': True,

            # Current vector store
            'vector_store_transcription': None,
        }

        for key, default_value in required_keys.items():
            if key not in session_state:
                session_state[key] = default_value

    def _reset_videos_session_variables(self):
        """Initialize essential session variables."""

        if session_state['vector_store_transcription']:
            session_state['vector_store_transcription'].delete_collection()

        required_keys = {
            'selected_subtitles_file': None,
            'subtitles_file_list': None,
            'selected_subtitles_text': "",
            'input_disabled': False,
            'pills_index': None,
            'summary': None,
            'topics': None,
            'artist': None,
            'lyrics': None,
            'suggestions_list': [],
            'icons_list': [],
            'suggested': False,
            'changed_video': True,

            # Current vector store
            'vector_store_transcription': None,
        }

        for key, default_value in required_keys.items():
            session_state[key] = default_value

        self.gm.create_empty_history('videos')

    def _reset_transcription_session_variables(self):
        """Initialize essential session variables."""

        if session_state['vector_store_transcription']:
            session_state['vector_store_transcription'].delete_collection()

        required_keys = {
            'selected_subtitles_text': "",
            'input_disabled': False,
            'pills_index': None,
            'summary': None,
            'topics': None,
            'artist': None,
            'lyrics': None,
            'suggestions_list': [],
            'icons_list': [],
            'suggested': False,
            'changed_video': True,

            # Current vector store
            'vector_store_transcription': None,
        }

        for key, default_value in required_keys.items():
            session_state[key] = default_value

        self.gm.create_empty_history('videos')

    @staticmethod
    def _extract_video_id(url):
        query_params = parse_qs(urlparse(url).query)
        video_id = query_params.get("v", [None])[0]
        return video_id

    def _get_video_id(self):
        return self._extract_video_id(session_state['current_video'])

    @staticmethod
    def _get_audio_file(video_id):
        return Path(videos_config.AUDIOS_DIR, video_id + '.m4a')

    @staticmethod
    def _save_transcription_file(video_id, transcription):
        with open(f"{Path(videos_config.TRANSCRIPTIONS_DIR, video_id)}.vtt", "w") as f:
            f.write(transcription)

    @staticmethod
    def _open_transcription_file(file):
        with open(Path(videos_config.TRANSCRIPTIONS_DIR, file), 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file.readlines()]
            text = ' '.join(lines)
        return text, lines

    @staticmethod
    def _split_audio_file(audio_file_path, target_size=26000000):
        """Split audio file into smaller segments.

        Args:
        - audio_file_path: The path to the audio file to split.
        - target_size: Target maximum size in bytes for each segment. Default is set close to OpenAI's limit.

        Returns:
        - A list of paths to the audio segments.
        """
        audio = AudioSegment.from_file(audio_file_path)
        # Estimate duration per segment to keep segment size under target size
        segment_duration_ms = (len(audio) / os.path.getsize(audio_file_path)) * target_size
        segments = []
        for i in range(0, len(audio), int(segment_duration_ms)):
            segment = audio[i:i + int(segment_duration_ms)]
            segment_path = f"{audio_file_path.stem}_part{i // int(segment_duration_ms)}.mp3"  # saving as mp3
            segment.export(segment_path, format="mp3")
            if os.path.getsize(segment_path) < target_size:  # Ensure the segment size is within the limit
                segments.append(segment_path)
        return segments

    def _generate_transcription(self, video_id):
        audio_file_path = Path(videos_config.AUDIOS_DIR, f"{video_id}.m4a")
        if audio_file_path.is_file():
            try:
                with st.spinner(f'Generating transcription for YouTube video: {video_id}'):
                    # Check file size first
                    if os.path.getsize(audio_file_path) > 26000000:
                        segment_paths = self._split_audio_file(audio_file_path)
                        transcriptions = []
                        if session_state['video_title']:
                            prompt = f"The title of this video is: {session_state['video_title']}"
                        else:
                            prompt = ""
                        for segment_path in segment_paths:
                            with open(segment_path, "rb") as segment_file:
                                transcription = self.client.audio.transcriptions.create(
                                    model="whisper-1",
                                    file=segment_file,
                                    response_format='vtt',
                                    prompt=prompt
                                )
                                transcriptions.append(transcription)
                                os.remove(segment_path)  # Clean up temporary files
                        full_transcription = " ".join(
                            transcriptions)  # Assuming concatenation method as simple string join
                        self._save_transcription_file(video_id, full_transcription)
                    else:
                        with open(audio_file_path, "rb") as audio_file:
                            transcription = self.client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file,
                                response_format='vtt'
                            )
                            self._save_transcription_file(video_id, transcription)
            except Exception as e:
                st.error(f"Failed to generate the transcription: {str(e)}")
        else:
            st.error(f"Audio file for YouTube video {video_id} does not exist.")

    def _process_video(self):

        if session_state['youtube']:

            video_id = self._get_video_id()

            # First check if an audio file for the video already exists
            audio_file = self._get_audio_file(video_id)
            # If it doesn't exist, download the youtube audio
            if not audio_file.is_file():
                ydl_opts = {
                    'format': 'm4a/bestaudio/best',
                    'paths': {'home': videos_config.AUDIOS_DIR},
                    'outtmpl': {'default': '%(id)s.%(ext)s'},
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'm4a',
                    }]
                }
                with st.spinner(f'Getting audio for YouTube file {video_id}'):
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        error_code = ydl.download([session_state['current_video']])
                        if error_code != 0:
                            raise Exception('Failed to download video')

            self._generate_transcription(video_id)

    @staticmethod
    def _extract_video_youtube_title(youtube_video):
        r = requests.get(youtube_video)
        soup = BeautifulSoup(r.text, features="lxml")

        link = soup.find_all(name="title")[0]
        title = link.text
        title = title.replace("<title>", "")
        title = title.replace("</title>", "")
        title = title.replace("- YouTube", "")
        title = title.strip()
        return title


    def display_videos_sidebar_controls(self):
        with st.sidebar:

            with st.expander("Video controls"):
                # File uploader for video
                uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"],
                                                  accept_multiple_files=False)

                youtube_video = st.text_input("Or provide a YouTube URL: ")

                video_name = None
                if not youtube_video and uploaded_video:
                    session_state['current_video'] = uploaded_video
                    video_name = uploaded_video
                    session_state['video_title'] = None
                    # Control if it's a YouTube video for subtitles
                    session_state['youtube'] = False
                elif youtube_video:
                    video_name = Path(self._extract_video_id(youtube_video))
                    session_state['current_video'] = youtube_video
                    session_state['video_title'] = self._extract_video_youtube_title(youtube_video)
                    # Control if it's a YouTube video for subtitles
                    session_state['youtube'] = True

                if session_state['previous_video'] != session_state['current_video']:
                    # Reset some variables set up in the previous video
                    self._reset_videos_session_variables()
                    session_state['previous_video'] = session_state['current_video']

                # If there's a video
                if video_name:
                    session_state['file_stem'] = video_name.name.split('.')[0].split('_')[0]

                    session_state['subtitles_file_list'] = [
                        file for file in os.listdir(session_state['subtitles_files_folder']) if
                        file.endswith(".vtt") and file.startswith(session_state['file_stem'])]

                    if session_state['subtitles_file_list']:
                        st.selectbox("Select an subtitles file to display the transcription",
                                     session_state['subtitles_file_list'],
                                     key='selected_subtitles_file',
                                     index=None)

                        if (session_state['previous_selected_subtitles_file']
                                != session_state['selected_subtitles_file'] and
                                session_state['selected_subtitles_file']):
                            self._reset_transcription_session_variables()
                            _, session_state['selected_subtitles_text'] = self._open_transcription_file(
                                session_state['selected_subtitles_file'])
                            session_state['previous_selected_subtitles_file'] = session_state['selected_subtitles_file']
                    else:
                        st.button("Generate_transcription", on_click=self._process_video)


class VideosManager:

    def __init__(self, user, general_manager):
        self.gm = general_manager
        self.client = None
        session_state['USER'] = user

        # Get the main component width of streamlit to display thumbnails and pdf
        self.main_dim = st_dimensions(key="main")

    def set_client(self, client):
        self.client = client

    @staticmethod
    def _get_subtitles():
        subtitles_dict = {}
        # Subtitles don't work on youtube videos
        if not session_state['youtube']:
            english_srt_file = Path(st.session_state.srt_files_folder, st.session_state.file_stem + "_en.srt")
            if english_srt_file.is_file():
                subtitles_dict['English'] = str(english_srt_file)
            german_srt_file = Path(st.session_state.srt_files_folder, st.session_state.file_stem + "_de.srt")
            if german_srt_file.is_file():
                subtitles_dict['German'] = str(german_srt_file)
            spanish_srt_file = Path(st.session_state.srt_files_folder, st.session_state.file_stem + "_es.srt")
            if spanish_srt_file.is_file():
                subtitles_dict['Spanish'] = str(spanish_srt_file)
        return subtitles_dict

    def _display_video(self):
        # Display the current video
        if session_state['current_video'] is not None:
            subtitles_dict = self._get_subtitles()
            with session_state['column_video'].container():
                st.video(session_state['current_video'], subtitles=subtitles_dict)
                if session_state['video_title']:
                    st.write(f"**{session_state['video_title']}**")

    @staticmethod
    def _display_suggestions():
        predefined_prompt_selected = None
        if session_state['suggestions_list']:
            session_state['pills_index'] = None
            with session_state['column_chat']:
                with st.expander("Query suggestions:"):
                    predefined_prompt_selected = stp.pills("", session_state['suggestions_list'],
                                                           session_state['icons_list'],
                                                           index=session_state['pills_index'])

            # Get rid of the suggestion now that it was chosen
            if predefined_prompt_selected:
                index_to_eliminate = session_state['suggestions_list'].index(predefined_prompt_selected)
                session_state['suggestions_list'].pop(index_to_eliminate)
                session_state['icons_list'].pop(index_to_eliminate)
        return predefined_prompt_selected

    def _display_chat(self):
        user_message = None
        conversation_history = None
        if session_state['current_video']:

            predefined_prompt_selected = self._display_suggestions()

            conversation_history = self.gm.get_conversation_history('videos')

            if conversation_history:
                self.gm.display_conversation(conversation_history, container=session_state['column_chat'])

            # Changes the style of the chat input to appear at the bottom of the column
            self.gm.change_chatbot_style()

            user_message = self._handle_user_input(predefined_prompt_selected)

            if user_message:
                self.gm.add_conversation_entry(chatbot='videos', speaker=session_state['USER'], message=user_message)

        return user_message, conversation_history

    @staticmethod
    def _handle_user_input(predefined_prompt):

        with session_state['column_chat']:
            # Accept user input
            user_input = st.chat_input("Ask anything to your Video",
                                       disabled=not session_state['current_video'])
        user_message = None
        if predefined_prompt:
            user_message = predefined_prompt
        elif user_input:
            user_message = user_input

        if user_message:
            with session_state['column_chat']:
                with st.chat_message("user"):
                    st.markdown(user_message)

        return user_message

    @staticmethod
    def _color_specific_words(line, target_words):
        # Split the line
        words = line.split()
        # Apply styles to specific words
        colored_line = ""
        for word in words:
            cleaned_word = word.strip(".!?:\"';()")

            if cleaned_word.lower() == target_words[0].lower():
                colored_line += f"<span style='color: blue; '>{word}</span> "
            elif cleaned_word.lower() == target_words[1].lower():
                colored_line += f"<span style='color: red; '>{word}</span> "
            else:
                colored_line += f"{word} "

        return colored_line

    @st.cache_data
    def _prepare_transcription(_self, transcription):
        # Initialize an empty string to store colored lines
        colored_lines = ""
        for line in transcription:
            words = ["SPEAKER_01", "SPEAKER_02"]
            colored_line = _self._color_specific_words(line, words)
            colored_lines += colored_line + "<br>"
        # Create a scrollable markdown section
        scrollable_code = f"""
        <div style='height: 300px; overflow-y: auto;'>
        {colored_lines}
        </div>
        """
        return scrollable_code

    def _process_transcription(self, transcription):
        if session_state['changed_video']:
            # When the uploaded files change reinitialize variables
            # Get text data
            text = self.client.text_split(transcription, session_state['video_title'])
            session_state['vector_store_transcription'] = self.client.get_vectorstore(text, collection="videos")

    @staticmethod
    def _prepare_title():
        if session_state['video_title']:  # If a title is provided
            # With title
            return "This inquiry pertains to the video titled \"{}\".".format(session_state['video_title'])
        return ""

    def _process_user_message_and_get_answer(self, user_message, conversation_history=None):
        chat_history_tuples = self.gm.generate_chat_history_tuples(conversation_history)
        with session_state['column_chat']:
            with st.spinner():
                condensed_question = self.client.get_condensed_question(
                    user_message, chat_history_tuples
                )

                title = self._prepare_title()
                sources = self.client.get_sources(session_state['vector_store_transcription'], condensed_question)
                all_output = self.client.get_answer(session_state['prompt_options_videos'][-1],
                                                    sources,
                                                    condensed_question,
                                                    title)

                ai_response = all_output['output_text']

                return ai_response

    @staticmethod
    def _display_ai_response(ai_response, container):
        if container:
            with container:
                with st.chat_message("assistant"):
                    st.markdown(ai_response)

    @staticmethod
    def _display_ai_info_box(info, typ):
        st.text_area(typ, info, height=300)

    def _generate_and_display_info(self, info):
        if session_state['changed_video']:
            with st.spinner(f"Generating {info}"):
                for i, prompt in enumerate(session_state['prompt_options_videos'][:-2]):
                    if info in prompt.keys():
                        session_state[info] = self._process_user_message_and_get_answer(
                            session_state['prompt_options_videos'][i][info])
                        break

    @staticmethod
    def _process_ai_response_for_suggestion_queries(model_output):
        # Ensure model_output is a single string for processing
        model_output_string = ' '.join(model_output) if isinstance(model_output, list) else model_output

        # Remove timestamps
        clean_output = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3}', '', model_output_string)

        # Removing content inside square brackets including the brackets themselves
        clean_output = re.sub(r'\[.*?\]', '', clean_output)

        # Splitting based on '?' to capture questions, adding '?' back to each split segment
        queries = [query + '?' for query in re.split(r'\?+\s*', clean_output) if query.strip()]

        cleaned_queries = []
        for query in queries:
            # Removing anything that is not a letter at the start of each query
            query_clean = re.sub(r'^[^A-Za-z]+', '', query)  # Adjust regex as needed
            # Attempt to more robustly remove quotes around the question
            query_clean = re.sub(r'(^\s*"\s*)|(\s*"$)', '', query_clean)  # Improved to target quotes more effectively
            # Remove numerical patterns and excessive whitespace
            query_clean = re.sub(r'\s*\d+[.)>\-]+\s*$', '', query_clean).strip()
            # Replace multiple spaces with a single space
            query_clean = re.sub(r'\s+', ' ', query_clean)
            if query_clean:
                cleaned_queries.append(query_clean)

        # Sort queries by length (descending) to prefer longer queries and select the first three
        if len(cleaned_queries) > 3:
            cleaned_queries = sorted(cleaned_queries, key=len, reverse=True)[:3]
        elif len(cleaned_queries) < 3:
            cleaned_queries = []

        print(cleaned_queries)

        return cleaned_queries

    def _generate_suggestions(self):
        # Just run it the first time when a new transcription is loaded
        if session_state['changed_video']:
            with st.spinner(f"Generating suggestions..."):
                ai_response = self._process_user_message_and_get_answer(
                    session_state['prompt_options_videos'][-2]['queries'])
                queries = self._process_ai_response_for_suggestion_queries(ai_response)
                for query in queries:
                    session_state['suggestions_list'].append(query)
                    session_state['icons_list'].append(session_state['prompt_options_videos'][-2]['icon'])

    def display_video_interface(self):
        # Use st.columns to create two columns
        (session_state['column_video'],
         mid,
         session_state['column_info1'],
         mid2,
         session_state['column_info2']) = st.columns([20, 1, 10, 1, 10])
        self._display_video()

        # Only load this if there's a transcription
        if session_state['selected_subtitles_file']:
            session_state['column_chat'], session_state['column_transcription'] = st.columns([0.5, 0.5], gap="medium")
            user_message, conversation_history = self._display_chat()
            scrollable_code = self._prepare_transcription(session_state['selected_subtitles_text'])
            with session_state['column_transcription']:
                st.markdown("### Transcription:")
                st.markdown(scrollable_code, unsafe_allow_html=True)
            self._process_transcription(' '.join(session_state['selected_subtitles_text']))
            with session_state['column_info1']:
                self._generate_and_display_info("summary")
                self._display_ai_info_box(session_state["summary"], "Summary")
            with session_state['column_info2']:
                self._generate_and_display_info("topics")
                self._display_ai_info_box(session_state["topics"], "Topics")
            self._generate_suggestions()

            # After processing everything when a video is changed, rerun to reflect changes
            if session_state['changed_video']:
                session_state['changed_video'] = False
                st.rerun()

            if user_message:
                ai_response = self._process_user_message_and_get_answer(user_message=user_message,
                                                                        conversation_history=conversation_history)
                self._display_ai_response(ai_response, session_state['column_chat'])
                self.gm.add_conversation_entry('videos', 'Assistant', ai_response)
                st.rerun()
