from src.rag import LoaderPipeline
from src.rag.chunker.code import CodeChunker


loader = LoaderPipeline()
docs = loader.load(
    source=r"D:\جزوه و اسلاید های و کتاب ها C++\[FVCP9504] C_Plus_Plus_Programming [Files_And_Codes]\S08\6.cpp"
)
chunker = CodeChunker()

docs_chunked = chunker.chunk(docs=docs)
for doc in docs_chunked:
    print(doc.page_content, end=f"\n{'_' * 80}\n")