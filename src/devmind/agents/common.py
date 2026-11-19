"""Small helpers shared by the sub-agents and the Main Agent."""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def final_answer(result: dict) -> str:
    """The last AI message that carries text and no pending tool calls."""
    for message in reversed(result.get("messages", [])):
        if getattr(message, "type", "") == "ai" and not getattr(message, "tool_calls", None):
            text = message_text(message.content).strip()
            if text:
                return text
    return ""


def limits(max_tool_calls: int) -> list:
    """Bound a run: surplus tool calls are refused (the model then answers), model calls hard-stop."""
    return [
        ToolCallLimitMiddleware(run_limit=max_tool_calls, exit_behavior="continue"),
        ModelCallLimitMiddleware(run_limit=max_tool_calls + 3, exit_behavior="end"),
    ]
