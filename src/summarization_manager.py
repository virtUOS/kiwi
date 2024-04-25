from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


class SummarizationManager:

    def __init__(self):
        pass

    @staticmethod
    def _generate_prompt_from_template(template):
        return PromptTemplate.from_template(template)

    @staticmethod
    def _text_splitter(docs, chunk_size=1000):
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=0
        )
        split_docs = text_splitter.split_documents(docs)
        return split_docs

    @staticmethod
    def _create_reduce_documents_chain(llm, reduce_prompt):
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        # Combines and iteratively reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=4000,
        )

        return reduce_documents_chain

    @staticmethod
    def _create_map_reduce_documents_chain(llm, map_prompt, reduce_documents_chain):
        map_chain = LLMChain(llm=llm, prompt=map_prompt)

        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
        )

        return map_reduce_chain

    @staticmethod
    def _create_map_reduce_documents_chain_with_intermediate_steps(llm, map_prompt, reduce_documents_chain):
        map_chain = LLMChain(llm=llm, prompt=map_prompt)

        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="docs",
            # Return the results of the map steps in the output
            return_intermediate_steps=True,
        )

        return map_reduce_chain

    @staticmethod
    def _turn_text_input_to_documents(text_input):
        docs_list = []
        for inp in text_input:
            docs_list.append(Document(inp))
        return docs_list

    def get_data_from_documents(self, docs, model_name, map_prompt_template, reduce_prompt_template):
        llm = ChatOpenAI(temperature=0, model_name=model_name)

        map_prompt = self._generate_prompt_from_template(map_prompt_template)
        reduce_prompt = self._generate_prompt_from_template(reduce_prompt_template)

        reduce_documents_chain = self._create_reduce_documents_chain(llm, reduce_prompt)

        map_reduce_chain = self._create_map_reduce_documents_chain(llm, map_prompt, reduce_documents_chain)

        docs_list = self._turn_text_input_to_documents(docs)
        split_docs = self._text_splitter(docs_list)

        response = map_reduce_chain.invoke(split_docs)

        return response['output_text']

    def get_data_from_video(self,
                            docs,
                            model_name,
                            map_prompt_template,
                            reduce_summary_prompt_template,
                            reduce_topics_prompt_template):
        llm = ChatOpenAI(temperature=0, model_name=model_name)

        map_prompt = self._generate_prompt_from_template(map_prompt_template)
        reduce_summary_prompt = self._generate_prompt_from_template(reduce_summary_prompt_template)
        reduce_topics_prompt = self._generate_prompt_from_template(reduce_topics_prompt_template)

        reduce_summary_chain = self._create_reduce_documents_chain(llm, reduce_summary_prompt)
        reduce_topics_chain = self._create_reduce_documents_chain(llm, reduce_topics_prompt)

        map_summary_chain = self._create_map_reduce_documents_chain_with_intermediate_steps(llm,
                                                                                            map_prompt,
                                                                                            reduce_summary_chain)
        # map_topics_chain = self._create_map_reduce_documents_chain(llm, map_prompt, reduce_topics_chain)

        docs_list = self._turn_text_input_to_documents(docs)
        split_docs = self._text_splitter(docs_list)

        summary = map_summary_chain.invoke(split_docs)
        list_of_intermediate_documents = [Document(topic) for topic in summary['intermediate_steps']]
        topics = reduce_topics_chain.invoke(list_of_intermediate_documents)

        return summary['output_text'], topics['output_text']
