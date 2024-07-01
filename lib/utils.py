from lib.models import MODELS_MAP

def read_prompt(file_name):
    with open(file_name, 'r') as file:
        return file.read()
        
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
def retrieve_answer(output):
    return output.content
    
def load_LLM(llm_name):
    model_config = MODELS_MAP[llm_name]
    model_class = model_config["class"]
    params = model_config["params"]
    llm = model_class(**params)
    return llm


def load_embeddings(llm_name):
    model_config = MODELS_MAP[llm_name]
    embedding_class = model_config["embedding_class"]
    embedding_params = model_config["embedding_params"]
    embeddings = embedding_class(**embedding_params)
    return embeddings
    
def get_available_models():
    return list(MODELS_MAP.keys())
    
def select_model():
    models = get_available_models()
    print("Available Models:")
    for i, model in enumerate(models):
        print(f"{i + 1}. {model}")
    
    while True:
        try:
            choice = int(input("Select a model by number: ")) - 1
            if 0 <= choice < len(models):
                return models[choice]
            else:
                print("Invalid choice. Please select a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
