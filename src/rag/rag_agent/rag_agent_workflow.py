import re


def _text(content) -> str:
    """Normalize message content (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _short(text: str, limit: int = 300) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + " ..."


def print_message(message, step: int) -> None:
    """Print one message of the agent workflow."""
    line = "=" * 70

    if message.type == "human":
        print(f"{line}\n[{step}] USER\n{_text(message.content)}")

    elif message.type == "ai":
        if message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"{line}\n[{step}] AI -> TOOL CALL")
                print(f"  tool : {tool_call['name']}")
                print(f"  args : {tool_call['args']}")
                print(f"  id   : {tool_call['id']}")
        else:
            answer = _text(message.content)
            print(f"{line}\n[{step}] AI -> FINAL ANSWER")
            if answer.strip():
                print(answer)
            else:
                print("  !!! EMPTY ANSWER (model produced no text)")

        usage = getattr(message, "usage_metadata", None)
        if usage:
            print(f"  tokens: {usage}")

    elif message.type == "tool":
        content = _text(message.content)
        docs_count = len(re.findall(r"^Document \d+", content, flags=re.MULTILINE))
        print(f"{line}\n[{step}] TOOL RESULT ({message.name})")
        print(f"  documents : {docs_count}")
        print(f"  length    : {len(content)} chars")
        print(f"  preview   : {_short(content)}")

    else:
        print(f"{line}\n[{step}] {message.type.upper()}\n{_short(_text(message.content))}")


def print_workflow(response) -> None:
    """Print the full workflow of a finished agent run."""
    messages = response["messages"]

    for step, message in enumerate(messages, start=1):
        print_message(message, step)

    tool_calls_count = sum(
        len(message.tool_calls)
        for message in messages
        if message.type == "ai" and message.tool_calls
    )
    print("=" * 70)
    print(f"SUMMARY: {len(messages)} messages, {tool_calls_count} tool call(s)")


def stream_workflow(agent, question: str, config: dict | None = None) -> dict:
    """Run the agent and print each step as soon as it happens."""
    printed = 0
    response = None

    for response in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="values",
    ):
        messages = response["messages"]
        for message in messages[printed:]:
            printed += 1
            print_message(message, printed)

    return response