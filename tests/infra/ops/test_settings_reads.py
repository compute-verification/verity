"""Every read of the environment infra's code makes through research's registry (`research.settings`) names a variable it
declares: an undeclared name would fail only when its line first ran, on the control pod."""
from pathlib import Path

from research import settings

ROOT = Path(__file__).resolve().parents[3]
FILES = [*sorted((ROOT / "infra/ops/verity_ops").glob("*.py")),
         *(ROOT / "infra/control" / f for f in ("agent_watch_relay.py", "control_disk_alert.py", "vm_stop.py", "watchers.py"))]


def test_every_read_through_the_registry_is_declared():
    found = [f"{p.relative_to(ROOT)}:{line}: {name}" for p in FILES for line, name in settings.undeclared(p.read_text())]
    assert found == [], "declare these in tools/research/src/research/settings.py (DECLARED):\n" + "\n".join(found)
