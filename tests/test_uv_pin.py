"""Every machine runs the uv that check pins: `tools/check/pod_setup.sh` installs it on a check pod and check's preflight refuses
any other, so the Nebius nodes' first boot (`vm_setup.sh`) installs the same one.  An unpinned install gave vy-nebius-2 the
latest uv, and every check in its slot was refused at preflight."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_nebius_first_boot_installs_the_uv_check_pins() -> None:
    pod = re.search(r"^UV_VERSION=\$\{UV_VERSION:-([^}]+)\}", (ROOT / "tools/check/pod_setup.sh").read_text(), re.M)
    vm = re.search(r"^UV_VERSION=(\S+)", (ROOT / "tools/research/src/research/pods/nebius/vm_setup.sh").read_text(), re.M)
    assert pod and vm and vm.group(1) == pod.group(1), (pod and pod.group(1), vm and vm.group(1))
