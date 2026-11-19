from langchain.tools import tool

from rag.vector_db.vector_database import VectorDatabase


def format_documents(documents) -> str:
    """Format retrieved documents for the agent."""
    if not documents:
        return "No relevant documents found."

    return "\n\n".join(
        f"Document {i}:\n"
        f"Content: {doc.page_content}\n"
        f"Metadata: {doc.metadata}"
        for i, doc in enumerate(documents, start=1)
    )


def build_retrieval_tools(vectorstore: VectorDatabase, k: int = 5) -> list:
    """Create the retrieval tools bound to the given vectorstore."""

    @tool
    def similarity_search(query: str) -> str:
        """Retrieve documents by semantic similarity.

        Use for conceptual questions, paraphrased queries,
        and questions where matching meaning is more important
        than matching exact words.
        """
        return format_documents(vectorstore.similarity_search(query=query, k=k))

    @tool
    def bm25_search(query: str) -> str:
        """Retrieve documents using BM25 keyword matching.

        Use when exact technical terms, identifiers, codes,
        document numbers, or specific phrases are important.
        """
        return format_documents(vectorstore.bm25_search(query=query, k=k))

    @tool
    def hybrid_search(query: str) -> str:
        """Retrieve documents using hybrid semantic and keyword search.

        Use for questions combining technical terminology
        with conceptual meaning, or when the best retrieval
        method is uncertain.
        """
        return format_documents(vectorstore.hybrid_search(query=query, k=k))

    return [similarity_search, bm25_search, hybrid_search]