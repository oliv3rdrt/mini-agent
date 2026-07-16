"""Tests for the tools and the dispatch layer."""

import tools


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
