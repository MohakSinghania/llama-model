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
    web_search : str
    documents : List[str]


class llama_model:
    def __init__(self) -> None:
        self.local_llm = constants.LOCAL_LLM
        self.pdf_directory = constants.PDF_DIRECTORY
        self.persist_directory = constants.PERSIST_DIRECTORY
        self.huggingface_model = constants.HUGGINGFACE_MODEL
        self.pdf_directory_all = constants.PDF_DIRECTORY_ALL
        self.persist_directory_all = constants.PERSIST_DIRECTORY_ALL


    def _pdf_file_save(self, pdf_file) -> dict:
        """Saves the PDF file locally if it does not already exist."""
        pdf_path = self.pdf_directory_all
        
        # Ensure the directory exists
        os.makedirs(pdf_path, exist_ok=True)
        
        pdf_file_path = os.path.join(pdf_path, pdf_file.name)
        if not os.path.exists(pdf_file_path):
            with open(pdf_file_path, "wb") as f:
                f.write(pdf_file.getbuffer())

    def _pdf_file_save_class(self, pdf_file, class_name):
        """Saves the PDF file locally if it does not already exist."""
        pdf_path = os.path.join(self.pdf_directory, class_name)
        
        # Ensure the directory exists
        os.makedirs(pdf_path, exist_ok=True)
        
        file_name = f"{class_name}_{pdf_file.name}"
        pdf_file_path = os.path.join(pdf_path, file_name)
        if not os.path.exists(pdf_file_path):
            with open(pdf_file_path, "wb") as f:
                f.write(pdf_file.getbuffer())

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

    def _get_answer_to_query(self, query, class_name):
        try:
            persist_directory = os.path.join(self.persist_directory, class_name)
            # LLM
            llm_format = ChatOllama(model=local_llm, format="json", temperature=0)
            llm_without_format = ChatOllama(model=local_llm, temperature=0)

            vectorstore = Chroma(
                        collection_name=f"{class_name}",
                        embedding_function=HuggingFaceEmbeddings(model_name=self.huggingface_model),
                        persist_directory=f"{persist_directory}"
                        )
            retriever = vectorstore.as_retriever()
            
            prompt_grader = PromptTemplate(
                    template="""You are a grader assessing relevance 
                    of a retrieved document to a user question. If the document contains keywords related to the user question, 
                    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
                    
                    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
                    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
                    
                    Here is the retrieved document: 
                    {document}
                    
                    Here is the user question: 
                    {question}
                    """,
                    input_variables=["question", "document"],
                    )
            retrieval_grader = prompt_grader | llm_format | JsonOutputParser()

            prompt_retrieval = PromptTemplate(
                    template="""You are an assistant for question-answering tasks. 
                    Use the following pieces of retrieved context to answer the question. If the Question does not belongs to the Context ,  just say "FALLBACK". 
                    Use three sentences maximum and keep the answer concise:
                    Question: {question} 
                    Context: {context} 
                    Answer: 
                    """,
                    input_variables=["question", "document"],
                    )

            # Chain
            rag_chain = prompt_retrieval | llm_without_format | StrOutputParser()

            ### Hallucination Grader 

            # Prompt
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

            # Prompt
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


            prompt_choice = PromptTemplate(
                        template="""You are an expert at routing a 
                        user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
                        prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
                        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
                        or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
                        no premable or explaination. 
                        
                        Question to route: 
                        {question}""",
                        input_variables=["question"],
                        )

            question_router = prompt_choice | llm_format | JsonOutputParser()

            ### Nodes
            def retrieve(state):
                """
                Retrieve documents from vectorstore

                Args:
                    state (dict): The current graph state

                Returns:
                    state (dict): New key added to state, documents, that contains retrieved documents
                """
                question = state["question"]

                # Retrieval
                documents = retriever.invoke(question)
                return {"documents": documents, "question": question}

            def generate(state):
                """
                Generate answer using RAG on retrieved documents

                Args:
                    state (dict): The current graph state

                Returns:
                    state (dict): New key added to state, generation, that contains LLM generation
                """
                question = state["question"]
                documents = state["documents"]
                
                # RAG generation
                generation = rag_chain.invoke({"context": documents, "question": question})
                return {"documents": documents, "question": question, "generation": generation}

            def grade_documents(state):
                """
                Determines whether the retrieved documents are relevant to the question
                If any document is not relevant, we will set a flag to run web search

                Args:
                    state (dict): The current graph state

                Returns:
                    state (dict): Filtered out irrelevant documents and updated web_search state
                """

                question = state["question"]
                documents = state["documents"]
                
                # Score each doc
                filtered_docs = []
                for d in documents:
                    score = retrieval_grader.invoke({"question": question, "document": d.page_content})
                    grade = score['score']
                    # Document relevant
                    if grade.lower() == "yes":
                        filtered_docs.append(d)
                return {"documents": filtered_docs, "question": question}

            ### Conditional edge
            def route_question(state):
                """
                Route question to web search or RAG.

                Args:
                    state (dict): The current graph state

                Returns:
                    str: Next node to call
                """

                question = state["question"]
                source = question_router.invoke({"question": question})  

                return "vectorstore"

            def decide_to_generate(state):
                """
                Determines whether to generate an answer, or add web search

                Args:
                    state (dict): The current graph state

                Returns:
                    str: Binary decision for next node to call
                """

                question = state["question"]
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

                question = state["question"]
                documents = state["documents"]
                generation = state["generation"]

                score = hallucination_grader.invoke({"documents": documents, "generation": generation})
                grade = score['score']

                # Check hallucination
                if grade == "yes":
                    # Check question-answering
                    score = answer_grader.invoke({"question": question,"generation": generation})
                    grade = score['score']
                    if grade == "yes":
                        return "useful"
                else:
                    return "not supported"

            workflow = StateGraph(GraphState)

            # Define the nodes
            workflow.add_node("retrieve", retrieve) # retrieve
            workflow.add_node("grade_documents", grade_documents) # grade documents
            workflow.add_node("generate", generate) # generatae


            # Build graph
            workflow.set_conditional_entry_point(
                route_question,
                {
                    "vectorstore": "retrieve",
                },
            )

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
            inputs = {"question": f'{query}'}
            for output in app.stream(inputs):
                for key, value in output.items():
                    pprint(f"Finished running: {key}:")
            answer = value["generation"]
            if "FALLBACK" in answer:
                return {'message': 'Query processed successfully', 'status': 404, 'question':query, 'answer': "Sorry the provided query does not belong to context"}
            else:
                return {'message': 'Query processed successfully', 'status': 200, 'question':query, 'answer': answer}
        except:
            return {'message': 'Answer is not available in the PDF', 'status': 404, 'question':query, 'answer': answer}


    def _get_answer_to_query_all(self, query):
        try:
            persist_directory = os.path.join(self.persist_directory_all, "all_pdf_files")
            # LLM
            llm_format = ChatOllama(model=local_llm, format="json", temperature=0)
            llm_without_format = ChatOllama(model=local_llm, temperature=0)

            vectorstore = Chroma(
                        collection_name=f"all_pdf_files",
                        embedding_function=HuggingFaceEmbeddings(model_name=self.huggingface_model),
                        persist_directory=f"{persist_directory}"
                        )
            retriever = vectorstore.as_retriever()
            
            prompt_grader = PromptTemplate(
                    template="""You are a grader assessing relevance 
                    of a retrieved document to a user question. If the document contains keywords related to the user question, 
                    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
                    
                    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
                    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
                    
                    Here is the retrieved document: 
                    {document}
                    
                    Here is the user question: 
                    {question}
                    """,
                    input_variables=["question", "document"],
                    )
            retrieval_grader = prompt_grader | llm_format | JsonOutputParser()
            prompt_retrieval = PromptTemplate(
                    template="""You are an assistant for question-answering tasks. 
                    Use the following pieces of retrieved context to answer the question. If the Question does not belongs to the Context ,  just say "FALLBACK". 
                    Use three sentences maximum and keep the answer concise:
                    Question: {question} 
                    Context: {context} 
                    Answer: 
                    """,
                    input_variables=["question", "document"],
                    )

            # Chain
            rag_chain = prompt_retrieval | llm_without_format | StrOutputParser()

            ### Hallucination Grader 

            # Prompt
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

            # Prompt
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


            prompt_choice = PromptTemplate(
                        template="""You are an expert at routing a 
                        user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
                        prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
                        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
                        or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
                        no premable or explaination. 
                        
                        Question to route: 
                        {question}""",
                        input_variables=["question"],
                        )

            question_router = prompt_choice | llm_format | JsonOutputParser()

            ### Nodes
            def retrieve(state):
                """
                Retrieve documents from vectorstore

                Args:
                    state (dict): The current graph state

                Returns:
                    state (dict): New key added to state, documents, that contains retrieved documents
                """
                question = state["question"]

                # Retrieval
                documents = retriever.invoke(question)

                return {"documents": documents, "question": question}

            def generate(state):
                """
                Generate answer using RAG on retrieved documents

                Args:
                    state (dict): The current graph state

                Returns:
                    state (dict): New key added to state, generation, that contains LLM generation
                """
                question = state["question"]
                documents = state["documents"]
                
                # RAG generation
                generation = rag_chain.invoke({"context": documents, "question": question})
                return {"documents": documents, "question": question, "generation": generation}

            def grade_documents(state):
                """
                Determines whether the retrieved documents are relevant to the question
                If any document is not relevant, we will set a flag to run web search

                Args:
                    state (dict): The current graph state

                Returns:
                    state (dict): Filtered out irrelevant documents and updated web_search state
                """

                question = state["question"]
                documents = state["documents"]
                
                # Score each doc
                filtered_docs = []
                for d in documents:
                    score = retrieval_grader.invoke({"question": question, "document": d.page_content})
                    grade = score['score']
                    # Document relevant
                    if grade.lower() == "yes":
                        filtered_docs.append(d)

                return {"documents": filtered_docs, "question": question}

            ### Conditional edge
            def route_question(state):
                """
                Route question to web search or RAG.

                Args:
                    state (dict): The current graph state

                Returns:
                    str: Next node to call
                """

                question = state["question"]
                source = question_router.invoke({"question": question})  

                return "vectorstore"

            def decide_to_generate(state):
                """
                Determines whether to generate an answer, or add web search

                Args:
                    state (dict): The current graph state

                Returns:
                    str: Binary decision for next node to call
                """

                question = state["question"]
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

                question = state["question"]
                documents = state["documents"]
                generation = state["generation"]

                score = hallucination_grader.invoke({"documents": documents, "generation": generation})

                grade = score['score']

                # Check hallucination
                if grade == "yes":
                    # Check question-answering
                    score = answer_grader.invoke({"question": question,"generation": generation})
                    grade = score['score']
                    if grade == "yes":
                        return "useful"
                else:
                    return "not supported"

            workflow = StateGraph(GraphState)

            # Define the nodes
            workflow.add_node("retrieve", retrieve) # retrieve
            workflow.add_node("grade_documents", grade_documents) # grade documents
            workflow.add_node("generate", generate) # generatae


            # Build graph
            workflow.set_conditional_entry_point(
                route_question,
                {
                    "vectorstore": "retrieve",
                },
            )

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
            inputs = {"question": f'{query}'}
            for output in app.stream(inputs):
                for key, value in output.items():
                    pprint(f"Finished running: {key}:")
            answer = value["generation"]
            if "FALLBACK" in answer:
                return {'message': 'Query processed successfully', 'status': 404, 'question':query, 'answer': "Sorry the provided query does not belong to context"}
            else:
                return {'message': 'Query processed successfully', 'status': 200, 'question':query, 'answer': answer}
        except:
            return {'message': 'Answer is not available in the PDF', 'status': 404, 'question':query, 'answer': answer}