"""The control disk relay (infra/control/control_disk_alert.py) calls the steward's notes sync stalled on a detached HEAD, an old
index.lock or a change left unpushed, and not otherwise."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("control_disk_alert", ROOT / "infra" / "control" / "control_disk_alert.py")
A = sys.modules["control_disk_alert"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
NOW = 1_792_430_000


def test_the_notes_sync_stalls_on_a_detached_head_an_old_lock_or_an_unpushed_change():
    assert A.sync_stall(f"{NOW} main 0 0 0") == []
    assert any("detached" in w for w in A.sync_stall(f"{NOW} - 0 0 0"))
    assert any("index.lock" in w for w in A.sync_stall(f"{NOW} main {NOW - 3600} 0 0"))
    assert A.sync_stall(f"{NOW} main {NOW - 60} 0 0") == []
    assert any("unpushed" in w for w in A.sync_stall(f"{NOW} main 0 {NOW - 7200} 0"))
    assert any("uncommitted" in w for w in A.sync_stall(f"{NOW} main 0 0 {NOW - 7200}"))
