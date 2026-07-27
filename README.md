# mini-agent

A small command-line AI agent. You type a request, the model thinks, and when
it needs to it calls tools to read files, write files, list a directory, or run
a shell command. Results go back to the model and it keeps going until the task
is done.

It talks to any OpenAI-compatible backend, so it runs against a local model for
free or against a hosted API if you prefer.

## How it works

The whole thing is two files:

- `agent.py` runs the chat loop. Each turn it sends the conversation to the
  model. If the model asks for a tool, the agent runs it, feeds the result back,
  and repeats. When the model replies with plain text instead of a tool call,
  that answer streams to the screen as it is generated and the turn ends. A step
  cap keeps a single turn from looping on tools forever.
- `tools.py` holds the four tools and the schemas that describe them to the
  model.

## Setup

```bash
cd mini-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then open `.env` and pick a backend. The three options are already in the file,
you just uncomment the one you want.

### Run it locally for free (Ollama)

No key, no signup, runs on your machine.

```bash
# install Ollama from https://ollama.com, then pull a small model
ollama pull llama3.2
ollama serve
```

In `.env`:

```
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=llama3.2
```

### Or use a hosted API

Point it at OpenAI or any OpenAI-compatible service by setting the key, the
model, and (for anything other than OpenAI) the base URL in `.env`.

## Run

```bash
python3 agent.py
```

Then just talk to it:

```
you > list the files in this folder
you > read agent.py and tell me what the main loop does
you > create a file called notes.txt with a short todo list
```

Type `exit` or `quit` to leave.

## Saving and resuming a session

Pass `--session FILE` to save the conversation and resume it next time:

```bash
python3 agent.py --session chat.jsonl
```

The file is a JSONL log, one message per line, rewritten after every turn. If
the file already exists the conversation picks up where it left off, so you can
stop and come back to it, or keep it around as a record of a past session.

## Tools

| Tool | What it does |
| --- | --- |
| `read_file` | Read a text file |
| `write_file` | Write text to a file, creating folders if needed; asks before overwriting an existing file |
| `list_files` | List the contents of a directory |
| `run_command` | Run a shell command after you confirm with `y` |
| `fetch_url` | Fetch a web page or text URL and return its readable text |

`run_command` always asks before it runs anything, so the agent can never touch
your shell without a yes.

### Keeping the file tools in one directory

By default the file tools can read and write anywhere you can. Set
`MINI_AGENT_ROOT` to a directory to sandbox them: any `read_file`, `write_file`,
or `list_files` path that resolves outside that directory is refused.

```bash
MINI_AGENT_ROOT="$(pwd)" python3 agent.py
```

## Tests

The tools and the dispatch layer have a small test suite.

```bash
pip install -r requirements-dev.txt
pytest
```

## Adding your own tool

1. Write a function in `tools.py`.
2. Add it to the `HANDLERS` dictionary.
3. Add a matching entry to the `TOOLS` list so the model knows it exists.

That is all the loop needs to start using it.
