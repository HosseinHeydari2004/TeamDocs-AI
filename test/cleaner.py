from src.rag.cleaners.web_cleaner import WebDocumentCleaner

from src.rag import LoaderPipeline

loader = LoaderPipeline()
docs = loader.load(
    source="https://docs.python.org/3.12/tutorial/controlflow.html#"
)

cleaner = WebDocumentCleaner()
docs = cleaner.clean(documents=docs)

print(docs[0].page_content)