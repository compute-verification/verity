"""`bench.frozen.path`: a path the frozen evidence doesn't hold fails, so a key a move rewrote can't read as an empty
directory (the ledger import would then add no labels and pass)."""
from __future__ import annotations

import pytest

from verity_numerical.bench import frozen


def test_a_path_the_frozen_tree_lacks_fails(tmp_path, monkeypatch):
    (tmp_path / "backends/numerical/reports/ledger").mkdir(parents=True)
    (tmp_path / "backends/numerical/reports/ledger/a.jsonl").write_text("{}\n")
    monkeypatch.setattr(frozen, "root", lambda: tmp_path)
    assert frozen.path("backends/numerical/reports/ledger") == tmp_path / "backends/numerical/reports/ledger"
    assert frozen.path("backends/numerical/reports/ledger/a.jsonl").read_text() == "{}\n"
    with pytest.raises(FileNotFoundError, match="former repository paths"):
        frozen.path("numerical/reports/ledger")
