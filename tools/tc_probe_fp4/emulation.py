"""``fp4_emulation_1ad1aaa2`` as ``E``, imported under the module name it was written against.

The file is 1ad1aaa2's verbatim (``t1_merged.emulation_gate`` checks its bytes), so it still imports the pinned models as
``verity.ml.tc.models``, which are ``verity_catalog.silicon.models`` now.
"""
from __future__ import annotations

import sys

from verity_catalog.silicon import models

sys.modules.setdefault("verity.ml.tc.models", models)

import fp4_emulation_1ad1aaa2 as E  # noqa: E402

__all__ = ["E"]
