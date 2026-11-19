from src.rag.chunker.markdown import MarkdownChunker
from src.rag import LoaderPipeline
from src.rag.cleaners.docs_cleaner import DocumentCleaner
from src.rag.metadata_extraction.extractor import MetadataExtractor
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
docs = chunker.chunk(docs=docs)

for meta, txt in docs[0].metadata.items():
    print(f"{meta}: {txt}")