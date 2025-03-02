import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from ollama import chat  # Ensure Ollama is installed and running

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
MODEL_NAME = "llava"  # Change to your desired model

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)

def query_rag(query_text: str):
    # Prepare the vector database.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)

    # Format the context and prompt.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Query the model using the new Ollama llava logic.
    response = chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
    response_text = response['message']['content']

    sources = [doc.metadata.get("id", None) for doc, _ in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()