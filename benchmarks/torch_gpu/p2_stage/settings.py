"""p2_stage's settings: the variables its modules read, declared once (`verity_config.settings`).

It owns ``P2_RECEIPTS_``: a variable with it that nothing here declares is refused when the registry loads.  It doesn't own
``P2_``, which benchmarks/pous' decode reads.  The launchers' own variables (LEASE_MIN, RANDOM_UNITS, SAMPLE, ...) are the
shell's, and sha512_rows' engine reads its own through sha512_rows' registry.
"""
from __future__ import annotations

from verity_config.settings import WHERE, Registry, Setting as S

REGISTRY = Registry(
    __file__, ("P2_RECEIPTS_",),
    S("P2_RECEIPTS_BUILD", str, None, WHERE, "sketch.py's build cache (default ~/.cache/verity/p2_receipts): builds are keyed "
                                             "by a hash of their sources and flags, so where they live changes nothing"))
