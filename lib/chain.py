import os
from operator import itemgetter

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate

from lib.models import MODELS_MAP
from lib.utils import format_docs, retrieve_answer, load_embeddings
from lib.entities import LLMEvalResult


def create_retriever(llm_name, db_path, docs, collection_name="local-rag"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=60)
    
    splits = text_splitter.split_documents(docs)
    
    embeddings = load_embeddings(llm_name)
    
    if not os.path.exists(db_path):
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=db_path, collection_name=collection_name)
    else:
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings, collection_name=collection_name)
                            
    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    return retriever
      

def create_qa_chain(llm, retriever, prompts_text):
    
    initial_prompt_text = prompts_text["initial_prompt"]
    qa_eval_prompt_text = prompts_text["evaluation_prompt"]
    qa_eval_prompt_with_context_text = prompts_text["evaluation_with_context_prompt"]
    
    initial_prompt = PromptTemplate(
        template=initial_prompt_text,
        input_variables=["question", "context"]
    )
    
    json_parser = JsonOutputParser(pydantic_object=LLMEvalResult)
    qa_eval_prompt = PromptTemplate(
        template=qa_eval_prompt_text,
        input_variables=["question","answer"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()},
    )

    # qa_eval_prompt_with_context = PromptTemplate(
    #     template=qa_eval_prompt_text,
    #     input_variables=["question","answer","context"],
    #     partial_variables={"format_instructions": json_parser.get_format_instructions()},
    # )

    chain = (
        RunnableParallel(context = retriever | format_docs, question = RunnablePassthrough()) |
        RunnableParallel(answer = initial_prompt | llm | retrieve_answer, question = itemgetter("question"), context = itemgetter("context") ) |
        RunnableParallel(input =  qa_eval_prompt, context = itemgetter("context"), answer = itemgetter("answer")) |
        RunnableParallel(evaluation = itemgetter("input") | llm , context = itemgetter("context"), answer = itemgetter("answer") ) | 
        RunnableParallel(output = itemgetter("answer"), evalutation = itemgetter("evaluation") | json_parser,  context = itemgetter("context"))
    )
    return chain

