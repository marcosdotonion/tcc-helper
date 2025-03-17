import argparse
import aiohttp
import asyncio
import aiofiles
import json
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma"
LLAMA_CPP_URL = "http://localhost:8080/v1/completions"  # Adjust the URL if needed

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class CustomEmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_query(self, text):
        return self.model.encode(text).tolist()

    def embed_documents(self, texts):
        return [self.model.encode(t).tolist() for t in texts]

#def get_embedding_function():
#    model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and fast
#    return model

def filter_documents_by_author(query_text, db):
    author_name = None
    # Check for author in the query
    query_lower = query_text.lower()
    if "use" in query_lower:
        parts = query_lower.split("use")
        if len(parts) > 1:
            # Extract the text after "use" and clean it up
            author_text = parts[1].strip()
            # Extract the author name (e.g., "michel_foucault")
            author_name = author_text.split()[0].strip()  # Take the first word after "use"

    # Debug line to print detected author(s)
    if author_name:
        print(f"author: {author_name}")
    else:
        print("No specific author detected in the query.")

    # Return the original database (no filtering)
    return db

async def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = CustomEmbeddingFunction()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Filter documents based on author if specified in the query
    filtered_db = filter_documents_by_author(query_text, db)

    # Search the DB with the filtered documents.
    results = filtered_db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    payload = {
        "prompt": prompt,
        "temperature": 0.4,
        "min_p": 0.05,
        "max_tokens": 512
    }

    # Asynchronous API request using aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(LLAMA_CPP_URL, json=payload) as response:
            response_data = await response.json()
            response_text = response_data.get("choices", [{}])[0].get("text", "No response received.")

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"\n=== RESPONSE ===\n{response_text}\nSources: {sources}\n=== END OF RESPONSE ===\n"

    # Asynchronously write to the output file
    async with aiofiles.open("output.md", "a") as f:
        await f.write(f"\n# Query: {query_text}\n{formatted_response}\n")

    print(formatted_response)
    print("Check output.md.")

async def main():
    while True:
        query_text = input("Enter your query (or type 'exit' to quit): ")
        if query_text.lower() == "exit":
            print("Exiting...")
            break
        # Run each query concurrently
        await query_rag(query_text)

if __name__ == "__main__":
    asyncio.run(main())
