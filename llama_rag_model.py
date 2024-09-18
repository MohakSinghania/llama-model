import re
import uuid
import boto3
import tempfile
import constants
from io import BytesIO
from pprint import pprint
from typing import Dict, TypedDict, Any
from langgraph.graph import END, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain.globals import set_verbose, set_debug
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from database import PDFDataDatabase, VectorStorePostgresVector
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

set_debug(True)
set_verbose(True)

# LLM

local_llm = constants.LOCAL_LLM
db = PDFDataDatabase()


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


class llama_model:
    def __init__(self) -> None:
        self.local_llm = constants.LOCAL_LLM
        self.pdf_directory_class = constants.PDF_DIRECTORY_CLASS
        self.pdf_directory_all = constants.PDF_DIRECTORY_ALL
        self.pdf_directory_school = constants.PDF_DIRECTORY_SCHOOL
        self.pdf_directory_college = constants.PDF_DIRECTORY_COLLEGE
        self.embedding = HuggingFaceEmbeddings(model_name=constants.HUGGINGFACE_MODEL)
        self.s3_client = boto3.client(
                            's3',
                            aws_access_key_id=constants.ACCESS_KEY,
                            aws_secret_access_key=constants.SECRET_KEY
                        )

    def _delete_s3_file(self, s3_file_key):
        # Delete the file from the S3 bucket
        try:
            bucket_name = constants.BUCKETNAME
            self.s3_client.delete_object(Bucket=bucket_name, Key=s3_file_key)
            return True
        except Exception:
            return False

    def _get_docs_split(self, pdf_files) -> Any:
        docs_list = []
        # for file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_file.write(pdf_files.getvalue())
            temp_file_path = temp_file.name
            loaded_docs = PyMuPDFLoader(temp_file_path).load()

        # Ensure that loaded_docs is processed correctly
        for doc in loaded_docs:
            if isinstance(doc, tuple):
                # Assuming the tuple structure is like (page_number, page_content)
                page_number, page_content = doc
                docs_list.append({'page_content': page_content})
            else:
                # If it's already in the desired format, just add it
                docs_list.append(doc)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=5)
        doc_split = text_splitter.split_documents(docs_list)
        return doc_split

    def _pdf_file_save(self, teacher_id, pdf_file) -> Dict:
        """Saves the PDF file in S3 Bucket if it does not already exist."""
        bucket_name = constants.BUCKETNAME
        pdf_file_path = f"{self.pdf_directory_all}{pdf_file.filename}"
        pdf_uuid = uuid.uuid4()

        try:
            self.s3_client.upload_fileobj(pdf_file, bucket_name, pdf_file_path)
            print(f"File uploaded successfully to {pdf_file_path}")
        except Exception as e:
            print(f"Failed to upload file: {e}")

        db.connect()
        db.insert_or_update_data(
            pdf_id=pdf_uuid,
            upload_by=teacher_id,  # Example teacher ID
            pdf_file_name=pdf_file.filename,
            pdf_path=pdf_file_path,
        )
        db.close_connection()
        return {'pdf_id': pdf_uuid, "pdf_path": pdf_file_path}

    def _pdf_file_process(self, file_name, pdf_file_path, teacher_id=None) -> Dict:
        """Saves the PDF file in S3 Bucket if it does not already exist."""
        pdf_uuid = uuid.uuid4()

        db.connect()
        db.insert_or_update_data_class(
            pdf_id=pdf_uuid,
            upload_by=teacher_id,  # Example teacher ID
            pdf_file_name=file_name,
            pdf_path=pdf_file_path,
        )
        db.close_connection()
        return {'pdf_id': pdf_uuid, "pdf_path": pdf_file_path}

    def _pdf_file_save_class(self, teacher_id, pdf_file, class_name):
        """Saves the PDF file in S3 Bucket if it does not already exist."""
        bucket_name = constants.BUCKETNAME
        file_name = f"{class_name}_{pdf_file.filename}"
        pdf_file_path = f"{self.pdf_directory_class}{class_name}/{file_name}"
        pdf_uuid = uuid.uuid4()

        try:
            self.s3_client.upload_fileobj(pdf_file, bucket_name, pdf_file_path)
            print(f"File uploaded successfully to {pdf_file_path}")
        except Exception as e:
            print(f"Failed to upload file: {e}")

        db.connect()
        db.insert_or_update_data_class(
            pdf_id=pdf_uuid,
            upload_by=teacher_id,  # Example teacher ID
            pdf_file_name=file_name,
            pdf_path=pdf_file_path,
            class_name=class_name,
        )
        db.close_connection()
        return {'pdf_id': pdf_uuid, "pdf_path": pdf_file_path}

    def _pdf_file_save_selection(
                                self, teacher_id, pdf_file, s_c_ce_type, board_type=None, state_board=None, class_name=None, college_name=None,
                                course_name=None, subject_name=None, stream_type=None
                                ) -> Dict:
        """Saves the PDF file in S3 Bucket if it does not already exist."""
        bucket_name = constants.BUCKETNAME
        if s_c_ce_type == "school":
            if board_type != 'state_board':
                if stream_type is None:
                    file_name = f"{s_c_ce_type}_{board_type}_{class_name}_{pdf_file.filename}"
                    pdf_file_path = f"{self.pdf_directory_school}{board_type}/{class_name}/{file_name}"
                else:
                    file_name = f"{s_c_ce_type}_{board_type}_{stream_type}_{class_name}_{pdf_file.filename}"
                    pdf_file_path = f"{self.pdf_directory_school}{board_type}/{stream_type}/{class_name}/{file_name}"
            else:
                if stream_type is None:
                    file_name = f"{s_c_ce_type}_{board_type}_{state_board}_{class_name}_{pdf_file.filename}"
                    pdf_file_path = f"{self.pdf_directory_school}{board_type}/{state_board}/{class_name}/{file_name}"
                else:
                    file_name = f"{s_c_ce_type}_{board_type}_{state_board}_{stream_type}_{class_name}_{pdf_file.filename}"
                    pdf_file_path = f"{self.pdf_directory_school}{board_type}/{state_board}/{stream_type}/{class_name}/{file_name}"

        elif s_c_ce_type == "college":
            file_name = f"{s_c_ce_type}_{college_name}_{course_name}_{stream_type}_{subject_name}_{pdf_file.filename}"
            pdf_file_path = f"{self.pdf_directory_college}{college_name}/{course_name}/{stream_type}/{subject_name}/{file_name}"
        else:
            pass

        pdf_uuid = uuid.uuid4()
        try:
            self.s3_client.upload_fileobj(pdf_file, bucket_name, pdf_file_path)
            print(f"File uploaded successfully to {pdf_file_path}")
        except Exception as e:
            print(f"Failed to upload file: {e}")
        data = {
            "pdf_id": pdf_uuid,
            "upload_by": teacher_id,
            "pdf_file_name": file_name,
            "pdf_path": pdf_file_path,
            "school_college_ce": s_c_ce_type,
            "board_type": board_type,
            "state_board": state_board,
            "class_name": class_name,
            "stream_name": stream_type,
            "college_name": college_name,
            "course_name": course_name,
            "subject_name": subject_name,
            "competitve_exam_name": None
        }
        db.connect()
        db.insert_or_update_data_selection(data)
        db.close_connection()
        return {'pdf_id': pdf_uuid, "pdf_path": pdf_file_path}

    def _create_embedding_all(self, pdf_id, pdf_path, class_name=None) -> dict:
        bucket_name = constants.BUCKETNAME

        response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=pdf_path)
        if 'Contents' in response:
            if response['Contents'][0]['Key'].endswith('.pdf'):
                pdf_file = self.s3_client.get_object(Bucket=bucket_name, Key=response['Contents'][0]['Key'])['Body'].read()
                pdf_files = (BytesIO(pdf_file))

        doc_split = self._get_docs_split(pdf_files)
        try:
            # Create a vector store
            if class_name is None:
                vector_store = VectorStorePostgresVector("all_pdf_files", self.embedding)
                document_present_or_not = vector_store.check_if_record_exist(pdf_id)
                if not document_present_or_not['is_rec_exist']:
                    pdf_id = str(pdf_id)
                    vector_store.store_docs_to_collection(pdf_id, doc_split, pdf_path)
                    return {'message': 'PDF Uploaded Successfully', 'status': 201}
                else:
                    return {'message': 'PDF is already Uploaded', 'status': 203}
            else:
                vector_store = VectorStorePostgresVector(class_name, self.embedding)
                document_present_or_not = vector_store.check_if_record_exist(pdf_id)
                if not document_present_or_not['is_rec_exist']:
                    pdf_id = str(pdf_id)
                    vector_store.store_docs_to_collection(pdf_id, doc_split, pdf_path)
                    return {'message': 'PDF Uploaded Successfully', 'status': 201}
                else:
                    return {'message': 'PDF is already Uploaded', 'status': 203}
        except Exception:
            return {'message': 'Failed to Upload the PDF', 'status': 401}

    def _create_embedding_selection(
                                    self, pdf_id, pdf_path, s_c_ce_type, board_type=None, state_board=None, class_name=None, college_name=None,
                                    course_name=None, subject_name=None, stream_type=None
                                    ) -> dict:
        bucket_name = constants.BUCKETNAME

        response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=pdf_path)
        if 'Contents' in response:
            if response['Contents'][0]['Key'].endswith('.pdf'):
                pdf_file = self.s3_client.get_object(Bucket=bucket_name, Key=response['Contents'][0]['Key'])['Body'].read()
                pdf_files = (BytesIO(pdf_file))

        doc_split = self._get_docs_split(pdf_files)
        try:
            if s_c_ce_type == 'school':
                if board_type != 'state_board':
                    if stream_type is None:
                        vector_store = VectorStorePostgresVector(f"{board_type}_{class_name}", self.embedding)
                        document_present_or_not = vector_store.check_if_record_exist(pdf_id)
                        if not document_present_or_not['is_rec_exist']:
                            pdf_id = str(pdf_id)
                            vector_store.store_docs_to_collection(pdf_id, doc_split, pdf_path)
                            return {'message': 'PDF Uploaded Successfully', 'status': 201}
                        else:
                            return {'message': 'PDF is already Uploaded', 'status': 203}
                    else:
                        vector_store = VectorStorePostgresVector(f"{board_type}_{stream_type}_{class_name}", self.embedding)
                        document_present_or_not = vector_store.check_if_record_exist(pdf_id)
                        if not document_present_or_not['is_rec_exist']:
                            pdf_id = str(pdf_id)
                            vector_store.store_docs_to_collection(pdf_id, doc_split, pdf_path)
                            return {'message': 'PDF Uploaded Successfully', 'status': 201}
                        else:
                            return {'message': 'PDF is already Uploaded', 'status': 203}
                else:
                    if stream_type is None:
                        vector_store = VectorStorePostgresVector(f"{board_type}_{state_board}_{class_name}", self.embedding)
                        document_present_or_not = vector_store.check_if_record_exist(pdf_id)
                        if not document_present_or_not['is_rec_exist']:
                            pdf_id = str(pdf_id)
                            vector_store.store_docs_to_collection(pdf_id, doc_split, pdf_path)
                            return {'message': 'PDF Uploaded Successfully', 'status': 201}
                        else:
                            return {'message': 'PDF is already Uploaded', 'status': 203}
                    else:
                        vector_store = VectorStorePostgresVector(f"{board_type}_{state_board}_{stream_type}_{class_name}", self.embedding)
                        document_present_or_not = vector_store.check_if_record_exist(pdf_id)
                        if not document_present_or_not['is_rec_exist']:
                            pdf_id = str(pdf_id)
                            vector_store.store_docs_to_collection(pdf_id, doc_split, pdf_path)
                            return {'message': 'PDF Uploaded Successfully', 'status': 201}
                        else:
                            return {'message': 'PDF is already Uploaded', 'status': 203}
            elif s_c_ce_type == 'college':
                vector_store = VectorStorePostgresVector(f"{college_name}_{course_name}_{stream_type}_{subject_name}", self.embedding)
                document_present_or_not = vector_store.check_if_record_exist(pdf_id)
                if not document_present_or_not['is_rec_exist']:
                    pdf_id = str(pdf_id)
                    vector_store.store_docs_to_collection(pdf_id, doc_split, pdf_path)
                    return {'message': 'PDF Uploaded Successfully', 'status': 201}
                else:
                    return {'message': 'PDF is already Uploaded', 'status': 203}
            else:
                pass
        except Exception:
            return {'message': 'Failed to Upload the PDF', 'status': 401}

    def _vectorstore_retriever(self, class_name=None):
        if class_name is not None:
            vector_store = VectorStorePostgresVector(class_name, self.embedding)
            return vector_store.get_or_create_collection().as_retriever()
        else:
            vector_store = VectorStorePostgresVector("all_pdf_files", self.embedding)
            return vector_store.get_or_create_collection().as_retriever()

    def _vectorstore_retriever_selection(self, data):
        if data['school_college_ce'] == 'school':
            if data['board'] != 'state_board':
                if data['stream'] is None:
                    vector_store = VectorStorePostgresVector(f"{data['board']}_{data['class_name']}", self.embedding)
                    return vector_store.get_or_create_collection().as_retriever()
                else:
                    vector_store = VectorStorePostgresVector(f"{data['board']}_{data['stream']}_{data['class_name']}", self.embedding)
                    return vector_store.get_or_create_collection().as_retriever()
            else:
                if data['stream'] is None:
                    vector_store = VectorStorePostgresVector(f"{data['board']}_{data['state_board']}_{data['class_name']}", self.embedding)
                    return vector_store.get_or_create_collection().as_retriever()
                else:
                    vector_store = VectorStorePostgresVector(f"{data['board']}_{data['state_board']}_{data['stream']}_{data['class_name']}",
                                                             self.embedding)
                    return vector_store.get_or_create_collection().as_retriever()
        elif data['school_college_ce'] == 'college':
            vector_store = VectorStorePostgresVector(f"{data['college_name']}_{data['course_name']}_{data['stream']}_{data['subject']}",
                                                     self.embedding)
            return vector_store.get_or_create_collection().as_retriever()

        else:
            pass

    def _get_answer_to_query(self, query, class_name=None):
        try:
            llm_format = ChatOllama(model=local_llm, format="json", temperature=0)
            llm_without_format = ChatOllama(model=local_llm, temperature=0)

            if class_name is not None:
                retriever = self._vectorstore_retriever(class_name)
            else:
                retriever = self._vectorstore_retriever()

            # Nodes
            def retrieve(state):
                """
                Retrieve documents

                Args:
                    state (dict): The current graph state

                Returns:
                    state (dict): New key added to state, documents, that contains retrieved documents
                """
                print("---RETRIEVE---")
                state_dict = state["keys"]
                question = state_dict["question"]
                documents = retriever.get_relevant_documents(question)
                return {"keys": {"documents": documents, "question": question}}

            def generate(state):
                """
                Generate answer

                Args:
                    state (dict): The current graph state

                Returns:
                    state (dict): New key added to state, generation, that contains generation
                """
                print("---GENERATE---")
                state_dict = state["keys"]
                question = state_dict["question"]
                documents = state_dict["documents"]

                # Prompt
                prompt = PromptTemplate(
                        template="""You are an assistant for question-answering tasks. \n
                        Treat as a Question , regardless of punctuation. \n
                        Use the following pieces of retrieved context to answer the question. \n
                        If the Question does not belongs to the Context, just say "FALLBACK",
                        don't give information based on your own Knowledge base,
                        just say or provide answer as "FALLBACK". \n
                        if the Context is a empty list then also say or provide answer as "FALLBACK". \n

                        Question: {question}
                        Context: {context}
                        Answer:
                        """,
                        input_variables=["question", "document"],
                    )

                # Post-processing
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                # Chain
                rag_chain = prompt | llm_without_format | StrOutputParser()

                # Run
                generation = rag_chain.invoke({"context": documents, "question": question})
                return {
                    "keys": {"documents": documents, "question": question, "generation": generation}
                }

            def grade_documents(state):
                """
                Determines whether the retrieved documents are relevant to the question.

                Args:
                    state (dict): The current graph state

                Returns:
                    state (dict): Updates documents key with relevant documents
                """

                print("---CHECK RELEVANCE---")
                state_dict = state["keys"]
                question = state_dict["question"]
                documents = state_dict["documents"]

                prompt = PromptTemplate(
                    template="""You are a grader assessing the relevance of a retrieved
                    document to a user question. \n
                    Here is the retrieved document: \n\n {context} \n\n
                    Here is the user question: {question} \n
                    If the document contains keywords related to the user question,
                    grade it as relevant. \n
                    If the document does not contain Maximum keywords or is a empty list,
                    grade it as irrelevant. \n
                    It does not need to be a stringent test. The goal is to filter out
                    erroneous retrievals. \n
                    Give a binary score of 'yes' or 'no' score to indicate whether the document
                    is relevant to the question. \n
                    Provide the binary score as a JSON with a single key 'score' and no preamble
                    or explanation.
                    """,
                    input_variables=["question", "context"],
                )

                chain = prompt | llm_format | JsonOutputParser()

                # Score
                filtered_docs = []
                for d in documents:
                    score = chain.invoke(
                        {
                            "question": question,
                            "context": d.page_content,
                        }
                    )
                    grade = score["score"]
                    if grade == "yes":
                        print("---GRADE: DOCUMENT RELEVANT---")
                        filtered_docs.append(d)
                    else:
                        continue

                return {
                    "keys": {
                        "documents": filtered_docs,
                        "question": question,
                    }
                }

            def transform_query(state):
                """
                Transform the query to produce a better question.

                Args:
                    state (dict): The current graph state

                Returns:
                    state (dict): Updates question key with a re-phrased question
                """

                print("---TRANSFORM QUERY---")
                state_dict = state["keys"]
                question = state_dict["question"]
                # documents = state_dict["documents"]

                # Create a prompt template with format instructions and the query
                prompt = PromptTemplate(
                    template="""Treat as a Question , regardless of punctuation. \n
                    Here is the initial question:
                    \n ------- \n
                    {question}
                    \n ------- \n
                    Provide an improved question without any premable, only respond with the
                    updated question: """,
                    input_variables=["question"],
                )
                # Prompt
                chain = prompt | llm_without_format | StrOutputParser()

                better_question = chain.invoke({"question": question})

                return {
                    "keys": {"question": better_question}
                }

            def decide_to_generate(state):
                """
                Determines whether to generate an answer or re-generate a question for web search.

                Args:
                    state (dict): The current state of the agent, including all keys.

                Returns:
                    str: Next node to call
                """

                print("---DECIDE TO GENERATE---")
                return "generate"

            # Conditional edge

            def grade_generation_v_documents_and_question(state):
                """
                Determines whether the generation is grounded in the document and answers question.

                Args:
                    state (dict): The current graph state

                Returns:
                    str: Decision for next node to call
                """
                state_dict = state["keys"]
                question = state_dict["question"]
                documents = state_dict["documents"]
                generation = state_dict["generation"]

                prompt_hallucination = PromptTemplate(
                    template="""You are a grader assessing whether
                    an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate
                    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
                    single key 'score' and no preamble or explanation.

                    Here are the facts:
                    {documents}

                    Here is the answer:
                    {generation}
                    """,
                    input_variables=["generation", "documents"],
                    )

                hallucination_grader = prompt_hallucination | llm_format | JsonOutputParser()
                score = hallucination_grader.invoke({"documents": documents, "generation": generation})

                # Check hallucination
                try:
                    grade = score['score']
                    if grade == "yes":
                        prompt_resolve = PromptTemplate(
                            template="""You are a grader assessing whether an answer is useful to resolve a question.
                            Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.
                            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

                            Here is the answer:
                            {generation}

                            Here is the question: {question}
                            """,
                            input_variables=["generation", "question"],
                            )

                        answer_grader = prompt_resolve | llm_format | JsonOutputParser()
                        score = answer_grader.invoke({"question": question, "generation": generation})
                        grade = score['score']
                        if grade == "yes":
                            return "useful"
                    else:
                        return "not supported"
                except Exception:
                    return "not supported"

            workflow = StateGraph(GraphState)

            # Define the nodes
            workflow.add_node("retrieve", retrieve)  # retrieve
            workflow.add_node("grade_documents", grade_documents)  # grade documents
            workflow.add_node("generate", generate)  # generatae
            workflow.add_node("transform_query", transform_query)  # transform_query

            workflow.set_entry_point("transform_query")
            workflow.add_edge("transform_query", "retrieve")
            workflow.add_edge("retrieve", "grade_documents")
            workflow.add_conditional_edges(
                "grade_documents",
                decide_to_generate,
                {
                    "generate": "generate",
                },
            )
            workflow.add_conditional_edges(
                "generate",
                grade_generation_v_documents_and_question,
                {
                    "not supported": END,
                    "useful": END,
                },
            )

            # Compile
            app = workflow.compile()
            # Run
            inputs = {
                "keys": {
                    "question": query,
                }
            }
            for output in app.stream(inputs):
                for key, value in output.items():
                    # Node
                    pprint(f"Node '{key}':")

                pprint("\n---\n")
            answer = value["keys"]["generation"]
            fallback_status = bool(re.search(r"FALLBACK", answer, re.I))
            if fallback_status:
                return {'message': 'Query processed successfully', 'status': 404, 'question': query,
                        'answer': "Sorry the provided query does not belong to context"}
            else:
                return {'message': 'Query processed successfully', 'status': 200, 'question': query, 'answer': answer}
        except Exception:
            return {'message': 'Answer is not available in the PDF', 'status': 404, 'question': query, 'answer': answer}

    def _get_answer_to_query_selection(self, query, data):
        try:
            llm_format = ChatOllama(model=local_llm, format="json", temperature=0)
            llm_without_format = ChatOllama(model=local_llm, temperature=0)
            retriever = self._vectorstore_retriever_selection(data)

            # Nodes
            def retrieve(state):
                """
                Retrieve documents

                Args:
                    state (dict): The current graph state

                Returns:
                    state (dict): New key added to state, documents, that contains retrieved documents
                """
                print("---RETRIEVE---")
                state_dict = state["keys"]
                question = state_dict["question"]
                documents = retriever.get_relevant_documents(question)
                return {"keys": {"documents": documents, "question": question}}

            def generate(state):
                """
                Generate answer

                Args:
                    state (dict): The current graph state

                Returns:
                    state (dict): New key added to state, generation, that contains generation
                """
                print("---GENERATE---")
                state_dict = state["keys"]
                question = state_dict["question"]
                documents = state_dict["documents"]

                # Prompt
                prompt = PromptTemplate(
                        template="""You are an assistant for question-answering tasks. \n
                        Treat as a Question , regardless of punctuation. \n
                        Use the following pieces of retrieved context to answer the question. \n
                        If the Question does not belongs to the Context, just say "FALLBACK",
                        don't give information based on your own Knowledge base,
                        just say or provide answer as "FALLBACK". \n
                        if the Context is a empty list then also say or provide answer as "FALLBACK". \n

                        Question: {question}
                        Context: {context}
                        Answer:
                        """,
                        input_variables=["question", "document"],
                    )

                # Post-processing
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                # Chain
                rag_chain = prompt | llm_without_format | StrOutputParser()

                # Run
                generation = rag_chain.invoke({"context": documents, "question": question})
                return {
                    "keys": {"documents": documents, "question": question, "generation": generation}
                }

            def grade_documents(state):
                """
                Determines whether the retrieved documents are relevant to the question.

                Args:
                    state (dict): The current graph state

                Returns:
                    state (dict): Updates documents key with relevant documents
                """

                print("---CHECK RELEVANCE---")
                state_dict = state["keys"]
                question = state_dict["question"]
                documents = state_dict["documents"]

                prompt = PromptTemplate(
                    template="""You are a grader assessing the relevance of a retrieved
                    document to a user question. \n
                    Here is the retrieved document: \n\n {context} \n\n
                    Here is the user question: {question} \n
                    If the document contains keywords related to the user question,
                    grade it as relevant. \n
                    If the document does not contain Maximum keywords or is a empty list,
                    grade it as irrelevant. \n
                    It does not need to be a stringent test. The goal is to filter out
                    erroneous retrievals. \n
                    Give a binary score of 'yes' or 'no' score to indicate whether the document
                    is relevant to the question. \n
                    Provide the binary score as a JSON with a single key 'score' and no preamble
                    or explanation.
                    """,
                    input_variables=["question", "context"],
                )

                chain = prompt | llm_format | JsonOutputParser()

                # Score
                filtered_docs = []
                for d in documents:
                    score = chain.invoke(
                        {
                            "question": question,
                            "context": d.page_content,
                        }
                    )
                    grade = score["score"]
                    if grade == "yes":
                        print("---GRADE: DOCUMENT RELEVANT---")
                        filtered_docs.append(d)
                    else:
                        continue

                return {
                    "keys": {
                        "documents": filtered_docs,
                        "question": question,
                    }
                }

            def transform_query(state):
                """
                Transform the query to produce a better question.

                Args:
                    state (dict): The current graph state

                Returns:
                    state (dict): Updates question key with a re-phrased question
                """

                print("---TRANSFORM QUERY---")
                state_dict = state["keys"]
                question = state_dict["question"]
                # documents = state_dict["documents"]

                # Create a prompt template with format instructions and the query
                prompt = PromptTemplate(
                    template="""Treat as a Question , regardless of punctuation. \n
                    Here is the initial question:
                    \n ------- \n
                    {question}
                    \n ------- \n
                    Provide an improved question without any premable, only respond with the
                    updated question: """,
                    input_variables=["question"],
                )
                # Prompt
                chain = prompt | llm_without_format | StrOutputParser()

                better_question = chain.invoke({"question": question})

                return {
                    "keys": {"question": better_question}
                }

            def decide_to_generate(state):
                """
                Determines whether to generate an answer or re-generate a question for web search.

                Args:
                    state (dict): The current state of the agent, including all keys.

                Returns:
                    str: Next node to call
                """

                print("---DECIDE TO GENERATE---")
                return "generate"

            # Conditional edge

            def grade_generation_v_documents_and_question(state):
                """
                Determines whether the generation is grounded in the document and answers question.

                Args:
                    state (dict): The current graph state

                Returns:
                    str: Decision for next node to call
                """
                state_dict = state["keys"]
                question = state_dict["question"]
                documents = state_dict["documents"]
                generation = state_dict["generation"]

                prompt_hallucination = PromptTemplate(
                    template="""You are a grader assessing whether
                    an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate
                    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
                    single key 'score' and no preamble or explanation.

                    Here are the facts:
                    {documents}

                    Here is the answer:
                    {generation}
                    """,
                    input_variables=["generation", "documents"],
                    )

                hallucination_grader = prompt_hallucination | llm_format | JsonOutputParser()
                score = hallucination_grader.invoke({"documents": documents, "generation": generation})

                # Check hallucination
                try:
                    grade = score['score']
                    if grade == "yes":
                        prompt_resolve = PromptTemplate(
                            template="""You are a grader assessing whether an answer is useful to resolve a question.
                            Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.
                            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

                            Here is the answer:
                            {generation}

                            Here is the question: {question}
                            """,
                            input_variables=["generation", "question"],
                            )

                        answer_grader = prompt_resolve | llm_format | JsonOutputParser()
                        score = answer_grader.invoke({"question": question, "generation": generation})
                        grade = score['score']
                        if grade == "yes":
                            return "useful"
                    else:
                        return "not supported"
                except Exception:
                    return "not supported"

            workflow = StateGraph(GraphState)

            # Define the nodes
            workflow.add_node("retrieve", retrieve)  # retrieve
            workflow.add_node("grade_documents", grade_documents)  # grade documents
            workflow.add_node("generate", generate)  # generatae
            workflow.add_node("transform_query", transform_query)  # transform_query

            workflow.set_entry_point("transform_query")
            workflow.add_edge("transform_query", "retrieve")
            workflow.add_edge("retrieve", "grade_documents")
            workflow.add_conditional_edges(
                "grade_documents",
                decide_to_generate,
                {
                    "generate": "generate",
                },
            )
            workflow.add_conditional_edges(
                "generate",
                grade_generation_v_documents_and_question,
                {
                    "not supported": END,
                    "useful": END,
                },
            )

            # Compile
            app = workflow.compile()
            # Run
            inputs = {
                "keys": {
                    "question": query,
                }
            }
            for output in app.stream(inputs):
                for key, value in output.items():
                    # Node
                    pprint(f"Node '{key}':")

                pprint("\n---\n")
            answer = value["keys"]["generation"]
            fallback_status = bool(re.search(r"FALLBACK", answer, re.I))
            if fallback_status:
                return {'message': 'Query processed successfully', 'status': 404, 'question': query,
                        'answer': "Sorry the provided query does not belong to context"}
            else:
                return {'message': 'Query processed successfully', 'status': 200, 'question': query, 'answer': answer}
        except Exception:
            return {'message': 'Answer is not available in the PDF', 'status': 404, 'question': query, 'answer': answer}

    def _display_files(self, admin_id, class_name):
        if class_name != 'None':
            db.connect()
            filenames = db.get_files_by_class(class_name)
            db.close_connection()
            if filenames != []:
                return {'message': f'The List of files for {class_name}', 'status': 200, 'filenames': filenames}
            else:
                return {'message': f'There is no files for {class_name}', 'status': 404, 'filenames': filenames}
        else:
            db.connect()
            filenames = db.get_files_by_class()
            db.close_connection()
            if filenames != []:
                return {'message': f'The List of files for {class_name}', 'status': 200, 'filenames': filenames}
            else:
                return {'message': f'There is no files for {class_name}', 'status': 404, 'filenames': filenames}

    def _delete_files(self, file_id, file_name, class_name):
        if class_name != 'None':
            vector_store = VectorStorePostgresVector(class_name, self.embedding)
            embeddings_status = vector_store.delete_file_embeddings_from_collection(file_id)
            if not embeddings_status['is_rec_exist']:
                db.connect()
                file_status = db.delete_record(file_id, file_name, class_name)
                db.close_connection()
                status = False
                if file_status['pdf_path'] is not None:
                    status = self._delete_s3_file(file_status['pdf_path'])
                if status:
                    return {'status': 200}
                else:
                    return {'status': 401}
            else:
                return {'status': 401}

        else:
            vector_store = VectorStorePostgresVector("all_pdf_files", self.embedding)
            embeddings_status = vector_store.delete_file_embeddings_from_collection(file_id)
            if not embeddings_status['is_rec_exist']:
                db.connect()
                file_status = db.delete_record(file_id, file_name)
                db.close_connection()
                status = False
                if file_status['pdf_path'] is not None:
                    status = self._delete_s3_file(file_status['pdf_path'])
                if status:
                    return {'status': 200}
                else:
                    return {'status': 401}
            else:
                return {'status': 401}
