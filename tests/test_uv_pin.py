"""Every machine runs the uv that check pins: `tools/verity/check/pod_setup.sh` installs it on a check pod and check's preflight refuses
any other, so the Nebius nodes' first boot (`vm_setup.sh`) and the jobs' base image install the same one.  An unpinned install
gave vy-nebius-2 the latest uv, and every check in its slot was refused at preflight."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEBIUS = ROOT / "tools/research/src/research/pods/nebius"


def test_the_nebius_first_boot_and_base_image_install_the_uv_check_pins() -> None:
    pod = re.search(r"^UV_VERSION=\$\{UV_VERSION:-([^}]+)\}", (ROOT / "tools/verity/check/pod_setup.sh").read_text(), re.MULTILINE)
    vm = re.search(r"^UV_VERSION=(\S+)", (NEBIUS / "vm_setup.sh").read_text(), re.MULTILINE)
    image = re.search(r"^ARG UV_VERSION=(\S+)", (NEBIUS / "image/Dockerfile").read_text(), re.MULTILINE)
    assert pod and vm and image and vm.group(1) == image.group(1) == pod.group(1), [m and m.group(1) for m in (pod, vm, image)]
