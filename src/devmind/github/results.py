"""Helpers for reading MCP ``CallToolResult`` objects across SDK versions.

``mcp`` 1.x exposes ``isError``; ``mcp`` 2.x renamed it ``is_error``. Reading only the old name
made every error look like a success, so error text could be stored as file content.
"""
from __future__ import annotations

from typing import Any


def result_is_error(result: Any) -> bool:
    if isinstance(result, BaseException):
        return True
    for name in ("is_error", "isError"):
        value = getattr(result, name, None)
        if isinstance(value, bool):
            return value
    return False


def result_text(result: Any) -> str:
    """The textual payload of a tool result.

    The real server answers a file read with a status line ("successfully downloaded text file ...") followed by an
    embedded resource holding the file. When resources are present only their text is returned, so the status line
    never ends up inside downloaded/indexed files.
    """
    resources: list[str] = []
    texts: list[str] = []
    for item in getattr(result, "content", None) or []:
        resource_text = getattr(getattr(item, "resource", None), "text", None)
        if resource_text:
            resources.append(resource_text)
        elif getattr(item, "text", None):
            texts.append(item.text)
    return "\n".join(resources or texts)
