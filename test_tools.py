"""Tests for the tools and the dispatch layer."""

import email
import urllib.error
import urllib.request

import tools


class _FakeResponse:
    """Stand-in for the object urllib returns, enough for fetch_url to read."""

    def __init__(self, body, content_type="text/html; charset=utf-8"):
        self._body = body.encode("utf-8")
        self.headers = email.message.Message()
        self.headers["Content-Type"] = content_type

    def read(self, amount=-1):
        if amount is None or amount < 0:
            return self._body
        return self._body[:amount]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_write_then_read_file(tmp_path):
    path = tmp_path / "note.txt"
    message = tools.write_file(str(path), "hello")
    assert "5 characters" in message
    assert tools.read_file(str(path)) == "hello"


def test_write_file_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deep" / "note.txt"
    tools.write_file(str(path), "x")
    assert path.read_text(encoding="utf-8") == "x"


def test_write_file_new_file_does_not_prompt(tmp_path, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("a brand new file should not prompt for overwrite")

    monkeypatch.setattr("builtins.input", refuse)
    path = tmp_path / "new.txt"
    tools.write_file(str(path), "hi")
    assert path.read_text(encoding="utf-8") == "hi"


def test_write_file_overwrite_confirmed(tmp_path, monkeypatch):
    path = tmp_path / "f.txt"
    path.write_text("old", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    tools.write_file(str(path), "new")
    assert path.read_text(encoding="utf-8") == "new"


def test_write_file_overwrite_declined_keeps_original(tmp_path, monkeypatch):
    path = tmp_path / "f.txt"
    path.write_text("old", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    result = tools.write_file(str(path), "new")
    assert path.read_text(encoding="utf-8") == "old"
    assert "declined" in result


def test_read_missing_file(tmp_path):
    result = tools.read_file(str(tmp_path / "nope.txt"))
    assert result.startswith("No file at")


def test_read_file_truncates_long_content(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("a" * (tools.MAX_OUTPUT + 500), encoding="utf-8")
    result = tools.read_file(str(path))
    assert result.endswith("... (truncated)")
    assert len(result) <= tools.MAX_OUTPUT + len("\n... (truncated)")


def test_list_files_sorts_and_marks_directories(tmp_path):
    (tmp_path / "b.txt").write_text("", encoding="utf-8")
    (tmp_path / "a").mkdir()
    assert tools.list_files(str(tmp_path)).splitlines() == ["a/", "b.txt"]


def test_list_files_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert tools.list_files(str(empty)) == "(empty)"


def test_list_files_missing_directory(tmp_path):
    result = tools.list_files(str(tmp_path / "nope"))
    assert result.startswith("No directory at")


def test_dispatch_routes_to_the_handler(tmp_path):
    path = tmp_path / "x.txt"
    tools.dispatch("write_file", {"path": str(path), "content": "hi"})
    assert tools.read_file(str(path)) == "hi"


def test_dispatch_unknown_tool():
    assert tools.dispatch("does_not_exist", {}) == "Unknown tool: does_not_exist"


def test_dispatch_bad_arguments():
    # read_file needs a 'path', so an empty argument map is a caller mistake.
    result = tools.dispatch("read_file", {})
    assert result.startswith("Tool error:")


def test_no_sandbox_by_default(tmp_path, monkeypatch):
    # Without MINI_AGENT_ROOT set, any readable path is allowed.
    monkeypatch.delenv("MINI_AGENT_ROOT", raising=False)
    path = tmp_path / "x.txt"
    path.write_text("ok", encoding="utf-8")
    assert tools.read_file(str(path)) == "ok"


def test_sandbox_allows_paths_inside_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MINI_AGENT_ROOT", str(tmp_path))
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    assert tools.read_file("a.txt") == "hi"


def test_sandbox_blocks_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("MINI_AGENT_ROOT", str(tmp_path))
    result = tools.read_file("../../../etc/hosts")
    assert result.startswith("Path is outside the working directory")


def test_sandbox_blocks_write_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("MINI_AGENT_ROOT", str(tmp_path))
    outside = tmp_path.parent / "escaped.txt"
    result = tools.write_file(str(outside), "nope")
    assert result.startswith("Path is outside the working directory")
    assert not outside.exists()


def test_fetch_url_rejects_non_http_scheme():
    assert tools.fetch_url("file:///etc/passwd") == "Only http and https URLs are supported."


def test_fetch_url_extracts_readable_text_from_html(monkeypatch):
    html = (
        "<html><head><style>.x{color:red}</style></head>"
        "<body><h1>Title</h1><script>var secret = 1;</script>"
        "<p>Hello world</p></body></html>"
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse(html))
    result = tools.fetch_url("https://example.com")
    assert "Title" in result
    assert "Hello world" in result
    # Script and style contents must not leak into the text.
    assert "secret" not in result
    assert "color:red" not in result


def test_fetch_url_returns_plain_text_as_is(monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse("just text", content_type="text/plain; charset=utf-8"),
    )
    assert tools.fetch_url("https://example.com/x.txt") == "just text"


def test_fetch_url_truncates_long_output(monkeypatch):
    body = "<p>" + "a" * (tools.MAX_OUTPUT + 500) + "</p>"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse(body))
    result = tools.fetch_url("https://example.com")
    assert result.endswith("... (truncated)")
    assert len(result) <= tools.MAX_OUTPUT + len("\n... (truncated)")


def test_fetch_url_reports_network_errors(monkeypatch):
    def boom(*_args, **_kwargs):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert tools.fetch_url("https://example.com").startswith("Could not fetch")


def test_fetch_url_dispatches(monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse("hi", content_type="text/plain"),
    )
    assert tools.dispatch("fetch_url", {"url": "https://example.com"}) == "hi"
