import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

MODELS_MAP = {
    "OpenAI GPT-3.5 Turbo": {
        "class": OpenAI,
        "params": {
            "temperature": 0,
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "embedding_class": OpenAIEmbeddings,
        "embedding_params": {
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    },
    "Groq LLaMA3 70b": {
        "class": ChatGroq,
        "params": {
            "model_name": "llama3-70b-8192",
            "groq_api_key": os.getenv("GROQ_API_KEY")
        },
        "embedding_class": HuggingFaceEmbeddings,
        "embedding_params": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2"
        }
    },
    "Groq Mixtral 8x7b": {
        "class": ChatGroq,
        "params": {
            "model_name": "mixtral-8x7b-32768",
            "groq_api_key": os.getenv("GROQ_API_KEY")
        },
        "embedding_class": HuggingFaceEmbeddings,
        "embedding_params": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2"
        }
    }
}

