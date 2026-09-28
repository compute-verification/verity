"""``fp4_emulation_1ad1aaa2`` as ``E``, imported under the module name it was written against.

The file is 1ad1aaa2's verbatim (``t1_merged.emulation_gate`` checks its bytes), so it still imports the pinned models as
``verity.ml.tc.models``, which are ``verity.ml.circuits.silicon.models`` now, and puts the tree's former source directory on
``sys.path``, which no longer exists and comes off again.
"""
from __future__ import annotations

import sys
from pathlib import Path

from verity.ml.circuits.silicon import models

sys.modules.setdefault("verity.ml.tc.models", models)

import fp4_emulation_1ad1aaa2 as E  # noqa: E402

_FORMER = str(Path(E.__file__).resolve().parents[2] / "packages" / "verity" / "src")
sys.path[:] = [p for p in sys.path if p != _FORMER]

__all__ = ["E"]
