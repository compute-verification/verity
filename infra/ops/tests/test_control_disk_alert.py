"""The control disk relay (infra/control/control_disk_alert.py) alerts on the router's heartbeat only once its code is deployed, and
only when the heartbeat (or, before the first one, the deploy) is more than 5 minutes old; and on a stalled notes sync."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("control_disk_alert", ROOT / "infra" / "control" / "control_disk_alert.py")
A = sys.modules["control_disk_alert"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
NOW = 1_792_430_000


def beat(heartbeat: int, deployed: int) -> str | None:
    return A.router_stale(f"{NOW} {heartbeat} {deployed}")


def test_the_router_is_quiet_until_its_code_is_deployed():
    assert beat(0, 0) is None
    assert beat(NOW - 3600, 0) is None


def test_a_fresh_heartbeat_is_ok_and_one_past_five_minutes_alerts():
    assert beat(NOW - 30, NOW - 86400) is None
    assert beat(NOW - 300, NOW - 86400) is None
    assert "6 min old" in beat(NOW - 360, NOW - 86400)


def test_no_heartbeat_alerts_five_minutes_after_the_deploy():
    assert beat(0, NOW - 120) is None
    assert "no heartbeat file" in beat(0, NOW - 600)


def test_the_notes_sync_stalls_on_a_detached_head_or_an_old_lock():
    assert A.sync_stall(f"{NOW} main 0 0 0") == []
    assert any("detached" in w for w in A.sync_stall(f"{NOW} - 0 0 0"))
    assert any("index.lock" in w for w in A.sync_stall(f"{NOW} main {NOW - 3600} 0 0"))
