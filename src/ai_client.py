import os
import json

import streamlit as st
from streamlit import session_state

from openai import OpenAI

from typing import Any, Dict, List

from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, VectorStore
from langchain.docstore.document import Document


class AIClient:

    def __init__(self):
        if session_state['model_selection'] == 'OpenAI':

            @st.cache_resource
            def load_openai_data():
                return OpenAI(), session_state['selected_model'], os.getenv('OPENAI_API_KEY')

            self.client, self.model, self.openai_key = load_openai_data()

            @st.cache_resource
            def load_langchain_data(model_name):
                llm = ChatOpenAI(model_name=model_name, openai_api_key=os.getenv('OPENAI_API_KEY'))
                question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
                return llm, question_generator

            self.client_lc, self.question_generator = load_langchain_data(self.model)
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large",
                                               openai_api_key=self.openai_key)  # Adjust model as required

        else:
            pass
            # Add here client initialization for local model
            # For example using OpenLLM as inference server
            # session_state["inference_server_url"] = "http://localhost:3000/v1"
            # self.client = OpenAI(base_url=session_state['inference_server_url'])
            # models = session_state["client"].models.list()
            # self.model = models.data[0].id

        self.vectorstore = None
        self.persistent_vdb_directory = os.getenv('VDB_PATH')

    @staticmethod
    def set_accessible_models():
        """
        Get accessible models for the current user
        """
        # Load accessible user models from environment variables
        default_models = json.loads(os.environ['OPENAI_DEFAULT_MODEL'])
        default_extra_models = json.loads(os.environ['OPENAI_MODEL_EXTRA'])
        if 'accessible_models' not in session_state and {'USER_ROLES', 'MODELS_PER_ROLE'} <= os.environ.keys():
            user_roles = json.loads(os.environ['USER_ROLES'])
            user_role = user_roles.get(session_state.get('username'))

            models_per_role = json.loads(os.environ['MODELS_PER_ROLE'])
            session_state['accessible_models'] = models_per_role.get(user_role, default_models['models'])

        # Get default model if no configurations found
        if 'accessible_models' not in session_state:
            session_state['accessible_models'] = default_models['models']

        # Get default model if no configurations found
        if 'accessible_models_extra' not in session_state:
            session_state['accessible_models_extra'] = default_extra_models['models']

    def get_condensed_question(self, user_input: str, chat_history_tuples):
        """
        This function adds context to the user query, by combining it with the chat history
        """
        condensed_question = self.question_generator.predict(
            question=user_input, chat_history=_get_chat_history(chat_history_tuples)
        )
        return condensed_question

    def get_sources(self, query: str) -> List[Document]:
        """This function performs a similarity search between the user query and the
        indexed document, returning the five most relevant document chunks for the given query
        """
        self.load_vector_store()
        docs = self.vectorstore.similarity_search(query, k=5)
        return docs

    def get_answer(self, prompt: str, docs: List[Document], query: str, title="") -> Dict[str, Any]:
        """Gets an answer to a question from a list of Documents."""
        doc_chain = load_qa_with_sources_chain(
            llm=self.client_lc,
            chain_type="stuff",
            prompt=prompt,
        )
        answer = doc_chain(
            {"input_documents": docs, "question": query, "title": title}, return_only_outputs=True
        )
        return answer

    def delete_collection(self):
        """Deletes a specific collection from the persistent vector store."""
        if 'collection_name' in session_state:
            if self.vectorstore is None:
                self.load_vector_store()

            try:
                self.vectorstore.delete_collection(session_state['collection_name'])
                print(f"Deleted the collection: {session_state['collection_name']}")
                self.vectorstore = None  # Reset the vector store attribute if the deleted collection is the current one
            except Exception as e:
                print(f"Error deleting collection {session_state['collection_name']}: {e}")

    def set_vector_store(self, docs: List[Document]):
        """Sets the vector store as an attribute of the class instance."""
        self.vectorstore = self._get_vector_store(docs)

    def load_vector_store(self):
        """Loads the existing vector store from the persistence directory."""
        self.vectorstore = Chroma(
            collection_name=session_state['collection_name'],
            persist_directory=self.persistent_vdb_directory,
            embedding_function=self.embeddings
        )

    def _get_vector_store(self, docs: List[Document]) -> VectorStore:
        """Given as input a document, this function returns the indexed version,
        leveraging the Chroma database"""
        ids = [doc.metadata["source"] for doc in docs]
        unique_ids = set(ids)

        if len(ids) != len(unique_ids):
            raise ValueError("Duplicate IDs found in document sources. Please ensure all IDs are unique.")

        # Ensure the persistence directory exists
        os.makedirs(self.persistent_vdb_directory, exist_ok=True)

        vectorstore = Chroma.from_documents(
            docs,
            self.embeddings,
            # Generate unique ids
            ids=[doc.metadata["source"] for doc in docs],
            collection_name=session_state['collection_name'],
            persist_directory=self.persistent_vdb_directory  # Specify the directory for persistence
        )
        return vectorstore

    @staticmethod
    def prepare_chunks_for_database(chunks: list) -> List[Document]:
        """Converts a list of chunk data to a list of Documents (chunks of fixed size)
        avoiding duplicates based on a unique source identifier (file name, page, and chunk number).

        Each Document includes the following metadata:
            - page number
            - chunk number
            - file name
            - coordinates (x0, x1, y0, y1)
            - source (unique identifier combining file name, page and chunk number)
        """

        doc_chunks = []
        seen_sources = set()  # Set to track the sources that have been added

        for i, chunk in enumerate(chunks):
            # Generate the source identifier
            source = f"{chunk['file_name'].split('.pdf')[0]}|{chunk['page']}-{chunk['chunk_number']}"

            # Check if this source has already been processed
            if source not in seen_sources:
                doc = Document(
                    page_content=chunk['text'],
                    metadata={'page': chunk['page'],
                              'chunk': chunk['chunk_number'],
                              'file': chunk['file_name'],
                              'x0': chunk['x0'],
                              'x1': chunk['x1'],
                              'y0': chunk['y0'],
                              'y1': chunk['y1'],
                              'source': source
                              }
                )
                doc_chunks.append(doc)
                seen_sources.add(source)  # Mark this source as seen

        return doc_chunks
