"""A small command-line agent that can call tools to get things done."""

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


def run_turn(client, model, messages):
    for _ in range(MAX_STEPS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools.TOOLS,
            )
        except APIConnectionError:
            print(f"\n{backend_hint()}\n")
            return
        except AuthenticationError:
            print("\nAuthentication failed. Check OPENAI_API_KEY in your .env.\n")
            return
        except APIStatusError as error:
            print(f"\nThe model backend returned an error: {error.message}\n")
            return

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            print(f"\nagent > {message.content}\n")
            return

        for call in message.tool_calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            print(f"  [tool] {name} {arguments}")
            result = tools.dispatch(name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                }
            )

    print("\nagent > Stopped after too many tool steps.\n")


def main():
    load_dotenv()
    # base_url lets us point at any OpenAI-compatible backend (a local Ollama
    # server, Groq, and so on). Left unset, it talks to OpenAI directly.
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    check_backend(client)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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


if __name__ == "__main__":
    main()
