from __future__ import annotations

import sys

import pytest

from verity_cli import run as R


@pytest.fixture(autouse=True)
def built_verifier(monkeypatch):
    monkeypatch.setattr(R, "build_verifier", lambda: None)


@pytest.fixture(autouse=True)
def no_node_settings(tmp_path, monkeypatch):
    """The adapter's `run.sh settings MODEL -- CMD` (`served_table.SETTINGS`) reads node A's files; here CMD runs under
    no settings."""
    if (table := sys.modules.get("served_table")) is not None:
        stand_in = tmp_path / "settings.sh"
        stand_in.write_text('shift 3\nexec "$@"\n')
        monkeypatch.setattr(table, "SETTINGS", str(stand_in), raising=False)
