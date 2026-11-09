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


def test_the_disk_alerts_at_85_90_and_95_and_a_level_rearms_three_points_under_it():
    """ci (CI-489, 9 Nov): an alarm at 85%; and the disk that fell from 97% to 90% and climbed back alerted nothing at 95%."""
    assert A.step(84, 0) == (None, 0) and A.step(85, 0) == ("alert", 85) and A.step(96, 85) == ("alert", 95)
    assert A.step(93, 95) == (None, 95), "within REARM of 95: no second alert for a wobble"
    assert A.step(90, 95) == (None, 90) and A.step(95, 90) == ("alert", 95)
    assert A.step(79, 90) == ("clear", 0) and A.step(79, 0) == (None, 0)
