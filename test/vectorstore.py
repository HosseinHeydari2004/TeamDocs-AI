from src.rag.chunker.markdown import MarkdownChunker
from src.rag import LoaderPipeline
from src.rag.cleaners.docs_cleaner import DocumentCleaner
from src.rag.metadata_extraction.extractor import MetadataExtractor
from src.rag.vector_db.vector_database import VectorDatabase
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

loader = LoaderPipeline()
docs = loader.load(
    source=r"D:\AI Source\Projects\list_of_projects\medvision\README.md"
)
cleaner = DocumentCleaner()
docs = cleaner.clean(documents=docs)

llm = init_chat_model(
    model="auto",
    model_provider="openrouter"
)

extractor = MetadataExtractor(llm=llm)
docs = extractor.extract_by_source(documents=docs)
chunker = MarkdownChunker()
docs = chunker.chunk(doc=docs)



store = VectorDatabase()

source_path = docs[0].metadata.get("source") if docs else None

if source_path:
    store.delete_by_source(source_path)

store.add_documents(documents=docs)

question = "who is Seyede Reyhane Khorashadizade?"

k = 5

dense_results = store.similarity_search(question, k=k)
bm25_results = store.bm25_search(question, k=k)
hybrid_results = store.hybrid_search(question, k=k, fetch_k=20)

def show_results(name, documents):
    print(f"\n{'=' * 20} {name} {'=' * 20}")

    for i, doc in enumerate(documents, start=1):
        print(f"\nRank: {i}")
        print(f"Source: {doc.metadata.get('source')}")
        print(doc.page_content[:500])


show_results("Dense Search", dense_results)
show_results("BM25 Search", bm25_results)
show_results("Hybrid Search", hybrid_results)

