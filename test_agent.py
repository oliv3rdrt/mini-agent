"""Tests for the agent loop helpers that do not need a live model."""

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

import agent


def _text_chunk(text):
    """A streamed chunk that carries a piece of plain text."""
    delta = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _tool_chunk(index, call_id=None, name=None, arguments=None):
    """A streamed chunk that carries a fragment of a tool call."""
    function = SimpleNamespace(name=name, arguments=arguments)
    piece = SimpleNamespace(index=index, id=call_id, function=function)
    delta = SimpleNamespace(content=None, tool_calls=[piece])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class _FakeStreamClient:
    """A client whose streaming create() replays a fixed list of chunks."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **_kwargs):
        return iter(self._chunks)


class _SequencedClient:
    """A client whose create() returns the next chunk stream on each call, so a
    whole run_turn (a tool call, then a final reply) can be driven."""

    def __init__(self, chunk_lists):
        self._chunk_lists = list(chunk_lists)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **_kwargs):
        return iter(self._chunk_lists.pop(0))


def test_stream_response_joins_streamed_text():
    client = _FakeStreamClient([_text_chunk("Hel"), _text_chunk("lo")])
    message = agent.stream_response(client, "model", [])
    assert message["content"] == "Hello"
    assert "tool_calls" not in message


def test_stream_response_stitches_tool_call_fragments():
    # The name and id come in the first fragment, the arguments across the next.
    chunks = [
        _tool_chunk(0, call_id="call_1", name="read_file", arguments='{"pa'),
        _tool_chunk(0, arguments='th": '),
        _tool_chunk(0, arguments='"x.txt"}'),
    ]
    message = agent.stream_response(_FakeStreamClient(chunks), "model", [])
    assert message["content"] is None
    (call,) = message["tool_calls"]
    assert call["id"] == "call_1"
    assert call["type"] == "function"
    assert call["function"]["name"] == "read_file"
    assert call["function"]["arguments"] == '{"path": "x.txt"}'


def test_stream_response_keeps_multiple_tool_calls_in_order():
    chunks = [
        _tool_chunk(0, call_id="a", name="list_files", arguments="{}"),
        _tool_chunk(1, call_id="b", name="read_file", arguments='{"path": "y"}'),
    ]
    message = agent.stream_response(_FakeStreamClient(chunks), "model", [])
    names = [call["function"]["name"] for call in message["tool_calls"]]
    assert names == ["list_files", "read_file"]


def test_save_and_load_history_round_trip(tmp_path):
    path = tmp_path / "session.jsonl"
    messages = [
        {"role": "system", "content": "hi"},
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "a\nb"},
    ]
    agent.save_history(str(path), messages)
    assert agent.load_history(str(path)) == messages


def test_load_history_missing_file_returns_none(tmp_path):
    assert agent.load_history(str(tmp_path / "nope.jsonl")) is None


def test_save_history_leaves_no_temp_file(tmp_path):
    path = tmp_path / "session.jsonl"
    agent.save_history(str(path), [{"role": "user", "content": "hi"}])
    # The atomic swap should leave only the session file, no leftover temp files.
    assert list(tmp_path.iterdir()) == [path]


def test_save_history_keeps_old_file_when_the_write_fails(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text('{"role": "system", "content": "old"}\n', encoding="utf-8")

    # object() is not JSON serializable, so the write fails partway through.
    with pytest.raises(TypeError):
        agent.save_history(str(path), [{"role": "user", "content": object()}])

    # The original file is untouched and no temporary file is left behind.
    assert path.read_text(encoding="utf-8") == '{"role": "system", "content": "old"}\n'
    assert list(tmp_path.iterdir()) == [path]


def test_parse_args_reads_session_flag():
    args = agent.parse_args(["--session", "s.jsonl"])
    assert args.session == "s.jsonl"


def test_parse_args_reads_single_prompt():
    args = agent.parse_args(["list the files here"])
    assert args.prompt == "list the files here"
    assert args.session is None


def test_parse_args_prompt_defaults_to_none_for_interactive_mode():
    args = agent.parse_args([])
    assert args.prompt is None


def test_parse_args_combines_prompt_and_session():
    args = agent.parse_args(["do a thing", "--session", "s.jsonl"])
    assert args.prompt == "do a thing"
    assert args.session == "s.jsonl"


def test_parse_args_reads_yes_flag():
    assert agent.parse_args(["--yes"]).yes is True
    assert agent.parse_args([]).yes is False


def _request():
    return httpx.Request("POST", "http://localhost")


def test_stream_with_retries_recovers_from_transient_errors(monkeypatch):
    attempts = {"n": 0}

    def flaky(client, model, messages):
        attempts["n"] += 1
        if attempts["n"] < agent.MAX_RETRIES:
            raise APIConnectionError(request=_request())
        return {"role": "assistant", "content": "ok"}

    monkeypatch.setattr(agent, "stream_response", flaky)
    monkeypatch.setattr(agent.time, "sleep", lambda *_: None)

    result = agent.stream_with_retries(None, "model", [])
    assert result == {"role": "assistant", "content": "ok"}
    assert attempts["n"] == agent.MAX_RETRIES


def test_stream_with_retries_does_not_retry_client_errors(monkeypatch):
    response = httpx.Response(400, request=_request())
    attempts = {"n": 0}

    def bad_request(client, model, messages):
        attempts["n"] += 1
        raise APIStatusError("bad request", response=response, body=None)

    monkeypatch.setattr(agent, "stream_response", bad_request)
    monkeypatch.setattr(agent.time, "sleep", lambda *_: None)

    with pytest.raises(APIStatusError):
        agent.stream_with_retries(None, "model", [])
    # A 4xx is a real error, so it is raised on the first try, not retried.
    assert attempts["n"] == 1


def test_run_turn_reports_invalid_tool_arguments(monkeypatch):
    # First response asks for a tool with broken JSON arguments; the second is a
    # plain reply that ends the turn.
    tool_call = [_tool_chunk(0, call_id="c1", name="read_file", arguments="{not json")]
    reply = [_text_chunk("done")]
    client = _SequencedClient([tool_call, reply])

    dispatched = []
    monkeypatch.setattr(
        agent.tools, "dispatch", lambda name, args: dispatched.append((name, args))
    )

    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    agent.run_turn(client, "model", messages)

    # The tool is not run with the bad arguments.
    assert dispatched == []
    # A tool result explaining the invalid arguments is fed back for that call.
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "c1"
    assert "Invalid arguments" in tool_messages[0]["content"]
