from src.rag import LoaderPipeline
from src.rag.chunker.docs import DocsChunker

loader = LoaderPipeline()
docs = loader.load(
    source=r"C:\Users\hosse\Downloads\TeamDocs AI – Test Documentation.docx"
)
# for doc in docs:
#     print(doc.metadata)
#     print(doc.page_content)
#     print()
#     print("_" * 80)
chunker = DocsChunker()
chunks = chunker.chunk(doc=docs)
for chunk in chunks:
    print(chunk.metadata)
    print(chunk.page_content)
    print()
    print("_" * 80)