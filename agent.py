"""A small command-line agent that can call tools to get things done."""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, AuthenticationError, APIStatusError

import tools

SYSTEM_PROMPT = (
    "You are a small command-line assistant with a set of tools. "
    "Use the tools when they help you answer a question or finish a task. "
    "Keep your replies short and to the point."
)

# Stop a single turn from looping on tools forever.
MAX_STEPS = 10


def backend_hint():
    """A clear message pointing at the usual backend setup problems."""
    where = os.getenv("OPENAI_BASE_URL") or "the OpenAI API"
    return (
        f"Could not reach {where}.\n"
        "  - If you are running a local model, make sure the server is up "
        "(for Ollama, run 'ollama serve').\n"
        "  - Check OPENAI_BASE_URL, OPENAI_API_KEY and OPENAI_MODEL in your .env."
    )


def check_backend(client):
    """Fail fast on startup with a clear message if the backend is unreachable."""
    try:
        client.models.list()
    except APIConnectionError:
        sys.exit(backend_hint())
    except AuthenticationError:
        sys.exit("Authentication failed. Check OPENAI_API_KEY in your .env.")
    except APIStatusError:
        # The backend answered, so it is reachable. Some servers do not expose a
        # models endpoint; any real problem will surface on the first turn.
        pass


def stream_response(client, model, messages):
    """Send one request with streaming on, print the reply as it arrives, and
    return the assistant message rebuilt from the streamed chunks.

    The model streams plain text a few characters at a time, and it streams any
    tool calls as fragments, so both are stitched back together here into the
    single message shape the API expects to receive back.
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools.TOOLS,
        stream=True,
    )

    content_parts = []
    # Tool calls arrive in pieces keyed by their position in the list. Each piece
    # carries a bit of the id, the name, or the arguments string.
    tool_calls = {}
    started_reply = False

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if delta.content:
            # Print the "agent >" prefix once, on the first token of text.
            if not started_reply:
                print("\nagent > ", end="", flush=True)
                started_reply = True
            print(delta.content, end="", flush=True)
            content_parts.append(delta.content)

        for piece in delta.tool_calls or []:
            call = tool_calls.setdefault(
                piece.index, {"id": None, "name": None, "arguments": ""}
            )
            if piece.id:
                call["id"] = piece.id
            if piece.function and piece.function.name:
                call["name"] = piece.function.name
            if piece.function and piece.function.arguments:
                call["arguments"] += piece.function.arguments

    if started_reply:
        print()  # finish the streamed line

    content = "".join(content_parts)
    ordered_calls = [tool_calls[index] for index in sorted(tool_calls)]

    message = {"role": "assistant", "content": content or None}
    if ordered_calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for call in ordered_calls
        ]
    return message


def run_turn(client, model, messages):
    for _ in range(MAX_STEPS):
        try:
            message = stream_response(client, model, messages)
        except APIConnectionError:
            print(f"\n{backend_hint()}\n")
            return
        except AuthenticationError:
            print("\nAuthentication failed. Check OPENAI_API_KEY in your .env.\n")
            return
        except APIStatusError as error:
            print(f"\nThe model backend returned an error: {error.message}\n")
            return

        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            print()  # blank line after the reply
            return

        for call in tool_calls:
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            print(f"  [tool] {name} {arguments}")
            result = tools.dispatch(name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": str(result),
                }
            )

    print("\nagent > Stopped after too many tool steps.\n")


def load_history(path):
    """Load a conversation from a JSONL session file.

    Returns the list of messages, or None when the file does not exist yet so the
    caller can start a fresh conversation.
    """
    if not os.path.exists(path):
        return None
    messages = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages or None


def save_history(path, messages):
    """Save the conversation to a JSONL session file, one message per line."""
    with open(path, "w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(json.dumps(message) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="A small command-line agent that can call tools."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Run this single prompt and exit. Omit it for the interactive loop.",
    )
    parser.add_argument(
        "--session",
        metavar="FILE",
        help="Load and save the conversation to a JSONL session file so it can "
        "be resumed later.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Auto-approve tool confirmations (overwriting a file, running a "
        "command) so the agent can be used non-interactively.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    # Let tools approve their own confirmations when running non-interactively.
    tools.AUTO_APPROVE = args.yes
    load_dotenv()
    # base_url lets us point at any OpenAI-compatible backend (a local Ollama
    # server, Groq, and so on). Left unset, it talks to OpenAI directly.
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    check_backend(client)

    # Resume a saved session when one is given, otherwise start with just the
    # system prompt.
    messages = load_history(args.session) if args.session else None
    if messages:
        print(f"Resumed {len(messages)} messages from {args.session}.\n")
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Single-prompt mode: run one prompt straight from the command line and exit,
    # which makes the agent usable from a script instead of only interactively.
    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        run_turn(client, model, messages)
        if args.session:
            save_history(args.session, messages)
        return

    print("mini-agent is ready. Type a request, or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        run_turn(client, model, messages)
        # Save after each turn so a session survives even if the process stops.
        if args.session:
            save_history(args.session, messages)


if __name__ == "__main__":
    main()
