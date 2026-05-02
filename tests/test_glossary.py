import pytest
from pathlib import Path

from words import _load_glossary


def test_load_glossary_explicit_path_returns_contents(tmp_path):
    p = tmp_path / "g.md"
    p.write_text("용탕: 용서받은 탕자")
    assert _load_glossary(p) == "용탕: 용서받은 탕자"


def test_load_glossary_explicit_missing_raises(tmp_path):
    missing = tmp_path / "nope.md"
    with pytest.raises(FileNotFoundError):
        _load_glossary(missing)


def test_load_glossary_default_returns_empty_when_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _load_glossary(None) == ""


def test_load_glossary_default_reads_local_glossary_md(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "glossary.md").write_text("# Glossary\n새누리: church")
    assert "새누리: church" in _load_glossary(None)
