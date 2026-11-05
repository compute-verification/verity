"""Every environment variable core's difftests read, declared once (S1 of the quality rollout; the template is
`integrations/vllm/verity_vllm/settings.py`). Each is an opt-in that says which tests run or where one writes what it
found, never a value a comparison depends on. `get(NAME)` is the value, or "" when unset, and refuses a name not declared
in `SETTINGS`."""

from __future__ import annotations

import os

SETTINGS = {
    "VERITY_EVALUATE_FULL_LOGF": "1: run NvLogf_v1 and _v2 on all 2^32 words (about 85 core-hours)",
    "VERITY_SKIP_STORE": "1: skip the tests that fetch from the evidence store (check.py refuses it in a recorded run)",
    "VERITY_VSTAR_STAGE": "inner sessions' shapes, `M,K_LOG,CLAIMS` separated by `;`, whose whole stage V*'s difftest compares",
    "VERITY_VSTAR_STAGE_OUT": "a directory V*'s stage comparison writes its per-stage record to",
    "VERITY_VSTAR_IN_DEPLOYED": "set: stage a recorded session's every statement (the pod's)",
}


def get(name: str) -> str:
    if name not in SETTINGS:
        raise KeyError(f"{name} is not declared in verity_difftest/settings.py (SETTINGS)")
    return os.environ.get(name, "")
