
from langchain_core.messages import HumanMessage, SystemMessage

from devmind.agents.prompts import REPO_QA_SYSTEM_PROMPT


class RepoQA:
    """Answer repository questions using context provided by the caller."""

    def __init__(self, chat_model):
        """
        Args:
            chat_model: An initialized LangChain chat model.
        """
        self.chat_model = chat_model

    def answer(self, question: str, context: str) -> str:
        """Generate an answer grounded in the supplied repository context."""
        if not question.strip():
            raise ValueError("Question must not be empty.")

        if not context.strip():
            context = "No repository context was provided."

        messages = [
            SystemMessage(content=REPO_QA_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Repository context:\n{context}\n\n"
                    f"User question:\n{question}"
                )
            ),
        ]

        response = self.chat_model.invoke(messages)
        return str(response.content)
