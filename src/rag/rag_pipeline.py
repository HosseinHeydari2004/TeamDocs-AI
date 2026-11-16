"""from __future__ import annotations

from .vector_db.vector_database import VectorDatabase
from .preprossing.preprossor import PreprocessingPipeline
from .rag_agent.tools import build_retrieval_tools
from .rag_agent.prompt import RAG_AGENT_SYSTEM_PROMPT
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.documents import Document


class RagPipeline:
    def __init__(self):
        self._chat_model = ChatOllama(
            model="qwen2.5:7b-instruct-q4_K_M",
            temperature=0
        )
        self._vectorstore: VectorDatabase = VectorDatabase()
        self._preprocessor = PreprocessingPipeline(model=self._chat_model)
        self._agent = create_agent(
            model=self._chat_model,
            system_prompt=RAG_AGENT_SYSTEM_PROMPT,
            tools=build_retrieval_tools(
                vectorstore=self._vectorstore
            )
        )

    def add_documents(self, source: list[Document]) -> None:
        documents = self._preprocessor.run(source=source)
        self._vectorstore.add_documents(documents=documents)

    def run_pipeline(self, query: str):
        response = self._agent.invoke(
            {
                "messages": [HumanMessage(content=query)]
            }
        )
        return response"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

from .vector_db.vector_database import VectorDatabase
from .preprossing.preprossor import PreprocessingPipeline
from .rag_agent.tools import build_retrieval_tools
from .rag_agent.prompt import RAG_AGENT_SYSTEM_PROMPT

from langchain.agents import create_agent
from langchain.messages import HumanMessage

from devmind.llm import get_chat_model


class RagPipeline:
    def __init__(self, collection_name: str = "teamdocs-ai"):
        self._chat_model = get_chat_model()

        self._vectorstore = VectorDatabase(
            collection_name=collection_name
        )

        self._preprocessor = PreprocessingPipeline(
            model=self._chat_model
        )

        self._agent = create_agent(
            model=self._chat_model,
            system_prompt=RAG_AGENT_SYSTEM_PROMPT,
            tools=build_retrieval_tools(
                vectorstore=self._vectorstore
            )
        )

    def add_documents(
        self,
        source: str | Path | Iterable[str | Path],
    ) -> None:
        documents = self._preprocessor.run(source=source)
        self._vectorstore.add_documents(documents=documents)

    def run_pipeline(self, query: str):
        response = self._agent.invoke(
            {
                "messages": [HumanMessage(content=query)]
            }
        )
        return response
