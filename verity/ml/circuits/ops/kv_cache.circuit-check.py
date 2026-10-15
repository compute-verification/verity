"""circuit-check's binding of ``KvCacheAppendE4m3_v1``."""
from __future__ import annotations

from typing import Any


def _core_roots() -> list[Any]:
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.kv_cache import KvCacheAppendE4m3

    return [bind(KvCacheAppendE4m3)]
