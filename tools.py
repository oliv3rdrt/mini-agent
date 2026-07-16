"""Tools the agent can call, plus the schemas the model sees."""

import os
import subprocess
from pathlib import Path

# Cap tool output so a big file or noisy command does not flood the context.
MAX_OUTPUT = 4000


def _root():
    """The sandbox root, if one is set via MINI_AGENT_ROOT, else None."""
    value = os.getenv("MINI_AGENT_ROOT")
    return Path(value).expanduser().resolve() if value else None


def _resolve(path):
    """Resolve a path, keeping it inside the sandbox root when one is set.

    Returns (resolved_path, None) when the path is allowed, or
    (None, message) when it would escape the root.
    """
    p = Path(path).expanduser()
    root = _root()
    if root is None:
        return p, None
    full = (p if p.is_absolute() else root / p).resolve()
    try:
        full.relative_to(root)
    except ValueError:
        return None, f"Path is outside the working directory: {path}"
    return full, None


def read_file(path):
    p, error = _resolve(path)
    if error:
        return error
    if not p.is_file():
        return f"No file at {path}"
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_OUTPUT:
        return text[:MAX_OUTPUT] + "\n... (truncated)"
    return text


def write_file(path, content):
    p, error = _resolve(path)
    if error:
        return error
    # Ask before clobbering a file that is already there. New files write freely.
    if p.exists():
        print(f"\n  {path} already exists.")
        if input("  overwrite it? [y/N] ").strip().lower() != "y":
            return "Write was declined by the user."
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}"


def list_files(directory="."):
    p, error = _resolve(directory)
    if error:
        return error
    if not p.is_dir():
        return f"No directory at {directory}"
    entries = []
    for item in sorted(p.iterdir()):
        entries.append(item.name + ("/" if item.is_dir() else ""))
    return "\n".join(entries) if entries else "(empty)"


def run_command(command):
    # Ask first so nothing touches the shell without a yes.
    print(f"\n  proposed command: {command}")
    if input("  run it? [y/N] ").strip().lower() != "y":
        return "Command was declined by the user."
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=60
    )
    output = (result.stdout + result.stderr).strip()
    if len(output) > MAX_OUTPUT:
        output = output[:MAX_OUTPUT] + "\n... (truncated)"
    return output or f"(exit code {result.returncode}, no output)"


HANDLERS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "run_command": run_command,
}


def dispatch(name, arguments):
    handler = HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    try:
        return handler(**arguments)
    except Exception as error:
        return f"Tool error: {error}"


# These schemas tell the model what each tool does and what arguments to send.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text to a file, creating folders as needed. "
            "Asks the user before overwriting a file that already exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to write to."},
                    "content": {"type": "string", "description": "Text to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files and folders in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to list. Defaults to the current one.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command. The user is asked to confirm first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run."}
                },
                "required": ["command"],
            },
        },
    },
]
