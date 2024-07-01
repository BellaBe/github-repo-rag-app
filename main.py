import streamlit as st

import argparse
import os
from dotenv import load_dotenv

from langchain.globals import set_debug, set_verbose


from lib.repository import download_github_repo
from lib.loader import load_files
from lib.chain import create_retriever, create_qa_chain
from lib.utils import read_prompt, load_LLM, select_model
from lib.models import MODELS_MAP

# set_debug(True)
set_verbose(True)

load_dotenv()


def main():
    # Prompt user to select the model
    model_name = select_model()
    model_info = MODELS_MAP[model_name]
    
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="GitHub Repo QA CLI Application")
    parser.add_argument("repo_url", type=str, help="URL of the GitHub repository")
    args = parser.parse_args()

    # Extract the repository name from the URL
    repo_url = args.repo_url
    repo_name = repo_url.split("/")[-1].replace(".git", "")

    # Compute the path to the data folder relative to the script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.join(base_dir, "data", repo_name)
    db_dir = os.path.join(base_dir, "data", "db")
    prompt_templates_dir = os.path.join(base_dir, "prompt_templates")

    # Download the GitHub repository
    print(f"Downloading repository from {repo_url}...")
    download_github_repo(repo_url, repo_dir)
    
    # Load prompt templates
    prompts_text = {
        "initial_prompt": read_prompt(os.path.join(prompt_templates_dir, 'initial_prompt.txt')),
        "evaluation_prompt": read_prompt(os.path.join(prompt_templates_dir, 'evaluation_prompt.txt')),
        "evaluation_with_context_prompt": read_prompt(os.path.join(prompt_templates_dir, 'evaluation_with_context_prompt.txt'))
    }

    # Load documents from the repository
    print(f"Loading documents from {repo_dir}...")
    document_chunks = load_files(repository_path=repo_dir)
    print(f"Created chunks length is: {len(document_chunks)}")
    
    # Create model, retriever, and prompt
    print(f"Creating retrieval QA chain using {model_name}...")
    llm = load_LLM(model_name)
    retriever = create_retriever(model_name, db_dir, document_chunks)
    qa_chain = create_qa_chain(llm, retriever, prompts_text)
    
    print("You can start asking questions. Type 'exit' to quit.")
    while True:
        question = input("Question: ")
        if question.lower() == "exit":
            break
        answer = qa_chain.invoke(question)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
    
