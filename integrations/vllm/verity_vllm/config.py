"""Where the integration's run definitions live.

``manifests/`` (checkpoint pins, semantic profiles) and ``workloads/`` (row workloads) sit beside the package in the
source tree; they are run definitions, not package data.  Library code locates them, and the source tree itself,
through the names below.
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

#: The source tree holding the ``verity_vllm`` package, ``manifests/`` and ``workloads/``.
ROOT = Path(str(resources.files("verity_vllm"))).resolve().parent
MANIFESTS = ROOT / "manifests"
WORKLOADS = ROOT / "workloads"
CHECKPOINTS = MANIFESTS / "checkpoints.json"


def checkpoints_hf_home() -> str | None:
    """The HF cache the checkpoint pins name (``checkpoints.json`` ``hf_home``); None without the manifest."""
    try:
        return json.loads(CHECKPOINTS.read_text()).get("hf_home") or None
    except (OSError, ValueError):
        return None
