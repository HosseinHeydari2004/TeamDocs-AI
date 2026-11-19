from src.rag import LoaderPipeline
from src.rag.cleaners.web_cleaner import WebDocumentCleaner
from src.rag.chunker.web import WebChunker

loader = LoaderPipeline()
docs = loader.load(
    source="https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html"
)
cleaner = WebDocumentCleaner()
docs = cleaner.clean(documents=docs)
chunkr = WebChunker()

docs_chunked = chunkr.chunk(docs=docs)

for doc in docs_chunked:
    print(doc.page_content)
    print("_" * 80)