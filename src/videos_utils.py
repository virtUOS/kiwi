import os
import base64
from urllib.parse import urlparse, parse_qs
import requests
from bs4 import BeautifulSoup
from pydub import AudioSegment

import yt_dlp
from pathlib import Path

from dotenv import load_dotenv
import fitz

from typing import Any, Dict, List

import streamlit as st
from streamlit import session_state
import streamlit_pills as stp
from streamlit_dimensions import st_dimensions

from src import menu_utils
from src import videos_config

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, VectorStore
from langchain.chains import LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# Load environment variables
load_dotenv()


class SidebarVideosControls:

    def __init__(self, sidebar_general_manager):
        self.sgm = sidebar_general_manager
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
            'subtitles_file_list': None,
            'selected_subtitles_text': "",
            'previous_question': "",
            'question': None,
            'input_disabled': False,
            'pills_index': None,
            'subtitles_files_folder': videos_config.TRANSCRIPTIONS_DIR,
            'video_title': None,
            'summary': None,
            'topics': None,

            # Current vector store
            'vector_store_transcription': None,
        }

        for key, default_value in required_keys.items():
            if key not in session_state:
                session_state[key] = default_value

        self._fill_suggestions_pills_list()

    def _reset_videos_session_variables(self):
        """Initialize essential session variables."""
        required_keys = {
            'selected_subtitles_file': None,
            'subtitles_file_list': None,
            'selected_subtitles_text': "",
            'previous_question': "",
            'question': None,
            'input_disabled': False,
            'pills_index': None,
            'summary': None,
            'topics': None,

            # Current vector store
            'vector_store_transcription': None,
        }

        for key, default_value in required_keys.items():
            session_state[key] = default_value

        self._fill_suggestions_pills_list()

    def _reset_transcription_session_variables(self):
        """Initialize essential session variables."""
        required_keys = {
            'selected_subtitles_text': "",
            'input_disabled': False,
            'pills_index': None,
            'summary': None,
            'topics': None,

            # Current vector store
            'vector_store_transcription': None,
        }

        for key, default_value in required_keys.items():
            session_state[key] = default_value

        self._fill_suggestions_pills_list()

    @staticmethod
    def _fill_suggestions_pills_list():
        # If the list of suggestions doesn't exist or is empty, fill it up
        if 'suggestions_list' not in session_state or not session_state['suggestions_list']:
            session_state['suggestions_list'] = []
            session_state['icons_list'] = []

            # The last prompt in the list of options is the template prompt
            for prompt in session_state['prompt_options_videos'][:-1]:
                session_state['suggestions_list'].append(prompt['prompt'])
                session_state['icons_list'].append(prompt['icon'])

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
                        for segment_path in segment_paths:
                            with open(segment_path, "rb") as segment_file:
                                transcription = self.client.audio.transcriptions.create(
                                    model="whisper-1",
                                    file=segment_file,
                                    response_format='vtt'
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

            # audio = transcribe.extract_audio(audio_file)
            # transcribe.transcribe(audio, audio_file, videos_config.WHISPER_MODEL_SIZE, videos_config.DEVICE,
            #                      videos_config.QUANT, videos_config.LANGUAGE, videos_config.DIARIZE)

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
                        selected_subtitles_file = st.selectbox("Select an subtitles file to display the transcription",
                                                               session_state['subtitles_file_list'])

                        if selected_subtitles_file != session_state['selected_subtitles_file']:
                            self._reset_transcription_session_variables()
                            session_state['selected_subtitles_file'] = selected_subtitles_file
                            _, session_state['selected_subtitles_text'] = self._open_transcription_file(
                                selected_subtitles_file)
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
            else:
                self.gm.create_empty_history('videos')

            # Changes the style of the chat input to appear at the bottom of the column
            self.gm.change_chatbot_style()

            user_message = self._handle_user_input(predefined_prompt_selected)

            if user_message:
                self.gm.add_conversation_entry(chatbot='videos', speaker='User', message=user_message)

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

    def _display_transcription(self):
        # Initialize an empty string to store colored lines
        colored_lines = ""
        with session_state['column_transcription']:
            st.markdown("### Transcription:")
            for line in session_state['selected_subtitles_text']:
                words = ["SPEAKER_01", "SPEAKER_02"]
                colored_line = self._color_specific_words(line, words)
                colored_lines += colored_line + "<br>"
            # Create a scrollable markdown section
            scrollable_code = f"""
            <div style='height: 300px; overflow-y: auto;'>
            {colored_lines}
            </div>
            """
            st.markdown(scrollable_code, unsafe_allow_html=True)

    @st.cache_data
    def _process_transcription(_self, transcription):
        # When the uploaded files change reinitialize variables
        text = []
        # Get text data
        text.extend(_self.client.text_split(transcription, session_state['video_title']))
        session_state['vector_store_transcription'] = _self.client.get_embeddings(text)

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
                all_output = self.client.get_answer(session_state['prompt_options_videos'],
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
    def _display_ai_info_box(info, typ, container):
        if container:
            with container:
                st.text_area(typ, info)

    def _generate_and_display_summary_and_topics(self):
        if not session_state['summary']:
            session_state['summary'] = self._process_user_message_and_get_answer(
                session_state['prompt_options_videos'][0])
            self._display_ai_info_box(session_state['summary'], 'Summary', session_state['column_info1'])

        if not session_state['topics']:
            session_state['topics'] = self._process_user_message_and_get_answer(
                session_state['prompt_options_videos'][1])
            self._display_ai_info_box(session_state['topics'], 'Topics', session_state['column_info2'])

    def display_video_interface(self):
        # Use st.columns to create two columns
        (session_state['column_video'],
         mid,
         session_state['column_info1'],
         mid2,
         session_state['column_info2']) = st.columns([20, 1, 10, 1, 10])
        self._display_video()

        # Only load this if there's a transcription
        if session_state['selected_subtitles_text']:
            session_state['column_chat'], session_state['column_transcription'] = st.columns([0.5, 0.5], gap="medium")
            user_message, conversation_history = self._display_chat()
            self._display_transcription()
            self._process_transcription(' '.join(session_state['selected_subtitles_text']))
            self._generate_and_display_summary_and_topics()
            if user_message:
                print(conversation_history)
                ai_response = self._process_user_message_and_get_answer(user_message=user_message,
                                                                        conversation_history=conversation_history)
                self._display_ai_response(ai_response, session_state['column_chat'])
