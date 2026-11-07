from langchain.tools import tool
from src.rag.vector_db.vector_database import VectorDatabase


@tool
def similarity_search(query: str) -> str:
    """Search the knowledge base for documents semantically similar to the query."""

    vectorstore = VectorDatabase()
    documents = vectorstore.similarity_search(query=query, k=5)

    if not documents:
        return "No relevant documents found."

    results = []

    for i, doc in enumerate(documents, start=1):
        results.append(
            f"Document {i}:\n"
            f"Content: {doc.page_content}\n"
            f"Metadata: {doc.metadata}"
        )

    return "\n\n".join(results)