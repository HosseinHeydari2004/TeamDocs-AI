from langchain.chat_models import init_chat_model
from src.rag import LoaderPipeline
from src.rag.metadata_extraction.extractor import MetadataExtractor
from src.rag.cleaners.web_cleaner import WebDocumentCleaner

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "auto",
    model_provider="openrouter"
)

extractor = MetadataExtractor(llm=model)

loader = LoaderPipeline()

docs = loader.load(
    source=["https://docs.python.org/3.12/tutorial/controlflow.html#", "cleaner.py"]
)
cleaner = WebDocumentCleaner()

docs = cleaner.clean(documents=docs)

docs = extractor.extract(documents=docs)
