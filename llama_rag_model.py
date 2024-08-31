import constants
from dependency import *

set_debug(True)
set_verbose(True)

### LLM

local_llm = constants.LOCAL_LLM


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents 
    """
    question : str
    generation : str
    documents : List[str]


class llama_model:
    def __init__(self) -> None:
        self.local_llm = constants.LOCAL_LLM
        self.pdf_directory = constants.PDF_DIRECTORY
        self.persist_directory = constants.PERSIST_DIRECTORY
        self.huggingface_model = constants.HUGGINGFACE_MODEL
        self.pdf_directory_all = constants.PDF_DIRECTORY_ALL
        self.persist_directory_all = constants.PERSIST_DIRECTORY_ALL

    def _pdf_file_save_class(self, pdf_file, class_name):
        """Saves the PDF file locally if it does not already exist."""
        pdf_path = os.path.join(self.pdf_directory, class_name)
        
        # Ensure the directory exists
        os.makedirs(pdf_path, exist_ok=True)
        
        file_name = f"{class_name}_{pdf_file.filename}"
        pdf_file_path = os.path.join(pdf_path, file_name)
        if not os.path.exists(pdf_file_path):
            pdf_file.save(pdf_file_path)

    def _pdf_file_save(self, pdf_file):
        """Saves the PDF file locally if it does not already exist."""
        pdf_path = self.pdf_directory_all
        
        # Ensure the directory exists
        os.makedirs(pdf_path, exist_ok=True)
        
        pdf_file_path = os.path.join(pdf_path, pdf_file.filename)
        if not os.path.exists(pdf_file_path):
            pdf_file.save(pdf_file_path)

    def _create_embedding(self, class_name) -> dict:
        pdf_directory = os.path.join(self.pdf_directory, class_name)
        
        # Load PDF documents
        pdf_files = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith('.pdf')]
        docs = [PyMuPDFLoader(file).load() for file in pdf_files]
        docs_list = [item for sublist in docs for item in sublist]
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(docs_list)
        try:
            # Create a vector store
            persist_directory = os.path.join(self.persist_directory, class_name)
            if os.path.exists(persist_directory):
                shutil.rmtree(persist_directory)
            os.makedirs(persist_directory, exist_ok=True)
            vectorstore = Chroma.from_documents(
                        documents=doc_splits,
                        collection_name=f"{class_name}",
                        embedding=HuggingFaceEmbeddings(model_name=self.huggingface_model),
                        persist_directory=f"{persist_directory}"
                        )
            return {'message': 'PDF Uploaded Successfully', 'status': 201}
        except:
            return {'message': 'PDF not Uploaded Successfully', 'status': 400}

    def _create_embedding_all(self) -> dict:
        pdf_directory = self.pdf_directory_all
        
        # Load PDF documents
        pdf_files = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith('.pdf')]
        docs = [PyMuPDFLoader(file).load() for file in pdf_files]
        docs_list = [item for sublist in docs for item in sublist]
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(docs_list)
        try:
            # Create a vector store
            persist_directory = os.path.join(self.persist_directory_all, "all_pdf_files")
            if os.path.exists(persist_directory):
                shutil.rmtree(persist_directory)
            os.makedirs(persist_directory, exist_ok=True)
            vectorstore = Chroma.from_documents(
                        documents=doc_splits,
                        collection_name=f"all_pdf_files",
                        embedding=HuggingFaceEmbeddings(model_name=self.huggingface_model),
                        persist_directory=f"{persist_directory}"
                        )
            return {'message': 'PDF Uploaded Successfully', 'status': 201}
        except:
            return {'message': 'PDF not Uploaded Successfully', 'status': 400}

    def _vectorstore_retriever(self, persist_directory, class_name=None):
        if class_name is not None:
            vectorstore = Chroma(
                        collection_name=f"{class_name}",
                        embedding_function=HuggingFaceEmbeddings(model_name=self.huggingface_model),
                        persist_directory=f"{persist_directory}"
                    )
        else:
            vectorstore = Chroma(
                        collection_name=f"all_pdf_files",
                        embedding_function=HuggingFaceEmbeddings(model_name=self.huggingface_model),
                        persist_directory=f"{persist_directory}"
                    )
        return vectorstore.as_retriever()


    def _get_answer_to_query(self, query, class_name=None):
        try:
            llm_format = ChatOllama(model=local_llm, format="json", temperature=0)
            llm_without_format = ChatOllama(model=local_llm, temperature=0)
            
            if class_name is not None:
                persist_directory = os.path.join(self.persist_directory, class_name)
                retriever = self._vectorstore_retriever(persist_directory, class_name)
            else:
                persist_directory = os.path.join(self.persist_directory_all, "all_pdf_files")
                retriever = self._vectorstore_retriever(persist_directory)

            ### Nodes
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
                        template="""You are an assistant for question-answering tasks. 
                        Treat as a Question , regardless of punctuation.
                        Use the following pieces of retrieved context to answer the question. If the Question does not belongs to the Context ,  just say "FALLBACK". 

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
                    "keys": {"question": better_question,}
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
                state_dict = state["keys"]
                question = state_dict["question"]
                filtered_documents = state_dict["documents"]

                return "generate"


            ### Conditional edge

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
                grade = score['score']

                # Check hallucination
                try:
                    if grade == "yes":
                        prompt_resolve = PromptTemplate(
                        template="""You are a grader assessing whether an 
                        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
                        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
                        
                        Here is the answer:
                        {generation} 

                        Here is the question: {question}
                        """,
                        input_variables=["generation", "question"],
                        )

                        answer_grader = prompt_resolve | llm_format | JsonOutputParser()
                        score = answer_grader.invoke({"question": question,"generation": generation})
                        grade = score['score']
                        if grade == "yes":
                            return "useful"
                    else:
                        return "not supported"
                except:
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
            fallback_status = bool(re.search(r"FALLBACK",answer, re.I))
            if fallback_status:
                return {'message': 'Query processed successfully', 'status': 404, 'question':query, 'answer': "Sorry the provided query does not belong to context"}
            else:
                return {'message': 'Query processed successfully', 'status': 200, 'question':query, 'answer': answer}
        except:
            return {'message': 'Answer is not available in the PDF', 'status': 404, 'question':query, 'answer': answer}

    def _display_files(self, class_name):
        pdf_directory = os.path.join(self.pdf_directory, class_name)
        files = os.listdir(pdf_directory)
        pdf_files = [file for file in files if file.endswith('.pdf')]
        if pdf_files != []:
            return {'message': f'The List of files for {class_name}', 'status': 200, 'pdf_files': pdf_files}
        else:
            return {'message': f'There is no files for {class_name}', 'status': 404, 'pdf_files': pdf_files}

    def _delete_files(self, file_name, class_name):
        pdf_directory = os.path.join(self.pdf_directory, class_name)
        file_path = os.path.join(pdf_directory, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
 