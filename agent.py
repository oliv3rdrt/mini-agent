"""A small command-line agent that can call tools to get things done."""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

import tools

SYSTEM_PROMPT = (
    "You are a small command-line assistant with a set of tools. "
    "Use the tools when they help you answer a question or finish a task. "
    "Keep your replies short and to the point."
)

# Stop a single turn from looping on tools forever.
MAX_STEPS = 10


def run_turn(client, model, messages):
    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools.TOOLS,
        )
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
