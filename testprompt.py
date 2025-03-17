import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="qwen2.5-coder")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"\n=== RESPONSE ===\n{response_text}\nSources: {sources}\n=== END OF RESPONSE ===\n"

    with open("output.md", "a") as f:
        f.write(f"\n# Query: {query_text}\n{formatted_response}\n")

    print(formatted_response)
    print("Check output.md.")

def main():
    while True:
        query_text = input("Enter your query (or type 'exit' to quit): ")
        if query_text.lower() == "exit":
            print("Exiting...")
            break
        query_rag(query_text)

if __name__ == "__main__":
    main()
