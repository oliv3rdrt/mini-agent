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
