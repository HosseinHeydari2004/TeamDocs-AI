from langchain_core.language_models import BaseChatModel
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma


class VectorDatabase:
    def __init__(self):
        self._embeddings = HuggingFaceEmbeddings(
            model_name_or_path="BAAI/bge-small-en-v1.5",
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

