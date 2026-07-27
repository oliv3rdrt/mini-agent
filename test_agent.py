"""Tests for the agent loop helpers that do not need a live model."""

from types import SimpleNamespace

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
