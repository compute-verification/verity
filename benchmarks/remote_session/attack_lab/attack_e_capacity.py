"""Attack E (timing and capacity): the verifier's "measured" ledger takes the node's capacity R from the prover.

`benchmarks/remote_session/verifier.py` reads `calibration["read_bytes_per_s"]` from the prover's `ready` message
(`prover.py` sends it) and passes it to `audit.ledger` as `capacity_bytes_per_s` for the "measured" ledger; the
"specified" ledger uses max(datasheet, that). The ledger refuses only a capacity below what the record's receipts
need (`reads > R·span`, "understated"). So a prover that reports any R down to that floor gets a smaller μ_timing and
a smaller γ with the same receipts, the same reads and the same openings. The bytes certified, (1 − γ)·R·span, don't
move: every term but μ_timing is a byte count over R·span, so the report rescales γ as a utilization and earns nothing.

This rebuilds the deployed honest session's record at its real scale (r20261014-193042-3d4f session 0, #2028's
inputs: 85 GiB, five 60 s windows of 951/951/950/951/951 whole-store passes, A = 73.285 ms, 33,273 openings, δ =
2^-40) and runs the repository's `audit.ledger` on it at the specified, the honestly measured and the least accepted
capacity, and just below that.

    python3 attack_e_capacity.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verity.core.protocols.profile import IntegrityProfile, Subset  # noqa: E402
from verity.core.protocols.remote_bandwidth import audit, codec, timing  # noqa: E402

GIB = 85
BLOCKS = GIB * 2 ** 30 // codec.BLOCK_BYTES
PASSES = (951, 951, 950, 951, 951)
WINDOW_NS = 60 * timing.S
ALLOWANCE_NS = 73_285_000
SAMPLES = 33_273
SPEC = 1597e9
MEASURED = 1539.26e9


def record() -> audit.Record:
    windows = []
    for t, p in enumerate(PASSES):
        lengths = (BLOCKS,) * p
        prof = IntegrityProfile(("receipt",), (sum(lengths),), {"receipt": Subset(SAMPLES)}, audit.DELTA,
                                drawn={"receipt": SAMPLES})
        released, deadline = t * WINDOW_NS, (t + 1) * WINDOW_NS + ALLOWANCE_NS
        windows.append(audit.WindowRecord(t, lengths, released, deadline, deadline + 1, prof))
    return audit.Record(b"\0" * 64, b"\0" * 64, BLOCKS, codec.BLOCK_BYTES, tuple(windows))


def wrong(profile: IntegrityProfile) -> float:
    return timing.wrong_bound(profile.sizes[0], SAMPLES, audit.DELTA)


def main() -> int:
    r = record()
    rebuild = audit.Rebuild(r.codec, timing.SQUARINGS_PER_S["p1024"])
    reads = r.receipts * r.block_bytes
    least = max(reads / (r.span_ns / 1e9),
                max(w.receipts * r.block_bytes / (w.span_ns / 1e9) for w in r.windows))
    rows = []
    for name, cap in (("specified", max(SPEC, MEASURED)), ("measured (honest report)", MEASURED),
                      ("least accepted (prover reports reads/span)", math.nextafter(least, math.inf)),
                      ("just below the least", least * (1 - 1e-6))):
        try:
            led = audit.ledger(r, capacity_bytes_per_s=cap, chip_bytes=timing.CHIP_BYTES, rebuild=rebuild, wrong=wrong)
            rows.append({"capacity": name, "R_GBps": cap / 1e9, "gamma": led.gamma, "timing": led.timing,
                         "chip": led.chip, "sample": led.sample, "counting": led.counting, "rebuild": led.rebuild,
                         "worst_window_gamma": led.worst.gamma, "reads_bytes": reads,
                         "certified_bytes": (1 - led.gamma) * cap * r.span_ns / 1e9})
        except ValueError as e:
            rows.append({"capacity": name, "R_GBps": cap / 1e9, "refused": str(e)[:90], "reads_bytes": reads})
    for row in rows:
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
