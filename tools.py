"""Tools the agent can call, plus the schemas the model sees."""

import os
import subprocess
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

# Cap tool output so a big file or noisy command does not flood the context.
MAX_OUTPUT = 4000

# Stop reading a web page once it hits this size, before we even trim the text,
# so a huge download cannot tie things up.
MAX_FETCH_BYTES = 2_000_000


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


class _TextExtractor(HTMLParser):
    """Pull the visible text out of an HTML page.

    Script and style blocks are skipped so their contents do not end up in the
    text the model reads.
    """

    _SKIP_TAGS = {"script", "style"}

    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._parts.append(text)

    def text(self):
        return "\n".join(self._parts)


def fetch_url(url):
    # Only fetch over the web, never file:// or other local schemes.
    if not url.lower().startswith(("http://", "https://")):
        return "Only http and https URLs are supported."
    # A user agent keeps some sites from refusing the request outright.
    request = urllib.request.Request(url, headers={"User-Agent": "mini-agent"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            content_type = response.headers.get_content_type()
            raw = response.read(MAX_FETCH_BYTES)
    except (urllib.error.URLError, ValueError, TimeoutError) as error:
        return f"Could not fetch {url}: {error}"

    body = raw.decode(charset, errors="replace")
    # Strip the tags out of HTML; other content types (plain text, JSON) are
    # already readable as they are.
    if "html" in content_type:
        parser = _TextExtractor()
        parser.feed(body)
        text = parser.text()
    else:
        text = body

    text = text.strip()
    if not text:
        return "(no readable text found)"
    # Trim to the same cap as the other tools so a big page cannot flood the context.
    if len(text) > MAX_OUTPUT:
        return text[:MAX_OUTPUT] + "\n... (truncated)"
    return text


HANDLERS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "run_command": run_command,
    "fetch_url": fetch_url,
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
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a web page or text URL and return its readable text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The http or https URL to fetch.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]
