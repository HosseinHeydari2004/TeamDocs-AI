from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from langchain.agents import create_agent
from langchain.messages import HumanMessage

from .preprossing.preprossor import PreprocessingPipeline
from .rag_agent.prompt import RAG_AGENT_SYSTEM_PROMPT
from .rag_agent.tools import build_retrieval_tools
from .vector_db.vector_database import VectorDatabase


class RagPipeline:
    def __init__(self, collection_name: str = "teamdocs-ai", chat_model=None):
        if chat_model is None:
            from devmind.llm import get_chat_model

            chat_model = get_chat_model()
        self._chat_model = chat_model

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
