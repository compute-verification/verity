"""`research data sql` on a mistyped column names the real tables and columns instead of a bare sqlite error."""
from __future__ import annotations

from research.store.cli import main
from research.store.local import LocalStore


def test_a_missing_column_prints_the_schema_and_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    LocalStore().put_artifact("proof/v1", {"n": 1}, {})
    assert main(["sql", "SELECT id, created_at FROM artifacts"]) == 2
    err = capsys.readouterr().err
    assert "no such column: created_at" in err
    assert "  artifacts(id, kind, bytes, payload_type, manifest_json, local, remote, repo_path" in err
    assert "research data preserved" in err
    assert main(["sql", "SELECT count(*) AS n FROM artifacts"]) == 0
    assert capsys.readouterr().out.split() == ["n", "1"]
