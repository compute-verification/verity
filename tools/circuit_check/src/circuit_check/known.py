"""The failures `main` already has, each with its cause: `circuit-check --all` and the suite pass on them and fail on anything
else, and on an entry that no longer fails (a fixed one is dropped here)."""
from __future__ import annotations

from typing import Iterable

#: pieces of the C-Flock lowering that differ from their IR primitive, and why; a Definition lowered through one inherits it
KNOWN_PIECES = {
    "F32Add_v1": "NaN payloads: the piece returns the canonical NaN 0x7FC00000, the IR primitive the quieted operand NaN "
                 "(0xFFC00000 + 0xFFC00000 = 0xFFC00000, 0x7F800001 -> 0x7FC00001); backends/flock/tests/test_ir_lowering.py "
                 "compares NaNs as a class",
    "F32Mul_v1": "NaN payloads, as F32Add_v1",
}
_TOPP = ("the top-p keep word raises unless its committed `splits` operand is one of (1, 2, 4, 8, 16, 32) "
         "(verity_vllm.program.registry.sampling.topp_mask_word): partial on a committed, private value")
#: other failures main has, by "<check>/<code> <Definition family>", and why
KNOWN: dict[str, str] = {
    "correspondence/reference-not-total TopPMaskWordx16_v1": _TOPP,
    "correspondence/reference-not-total TopPMaskWordx64_v1": _TOPP,
    "correspondence/reference-not-total TopPMask_v1": _TOPP,
    "correspondence/reference-not-total GumbelTopPTokenSelect_v1": _TOPP,
    "partition/gate-recomputed ScaledMmFp8Block_v1": "every coordinate of a 128-column weight block recomputes the block's scale "
        "product F32Mul(sx[kb], sw[kb]) (fp8.ScaledMmFp8BlockCoordinate): 127 copies across units per block and kb; "
        "word.unit_rule partitions the batch's members one by one and does not see it",
    "partition/committed-unread DeriveRefBf16ToF32_v1": "a Call that is only wiring (Bf16ToF32 per element): its outputs are its "
        "inputs rearranged, so the cut commits inputs no unit reads (verity.ir.partition.validate_unit_cut counts a returned "
        "wiring value as no read)",
}


def key(f: dict) -> str:
    return f"{f['check']}/{f['code']} {f['definition'].split('{')[0]}"


def known(f: dict) -> str | None:
    """Why the failure ``f`` is known, or None when it is new."""
    if key(f) in KNOWN:
        return KNOWN[key(f)]
    if f["code"] == "lowering-mismatch":
        causes = [f["definition"]] if f["definition"] in KNOWN_PIECES else f.get("causes") or []
        if causes and all(c in KNOWN_PIECES for c in causes):
            return "; ".join(f"{c}: {KNOWN_PIECES[c]}" for c in causes)
    return None


def stale(failures: Iterable[dict]) -> list[str]:
    """The entries no failure among ``failures`` (every registered Definition's) matches any more."""
    fs = list(failures)
    keys = {key(f) for f in fs}
    pieces = {f["definition"] for f in fs if f["code"] == "lowering-mismatch"}
    return sorted(set(KNOWN) - keys) + sorted(set(KNOWN_PIECES) - pieces)


__all__ = ["KNOWN", "KNOWN_PIECES", "key", "known", "stale"]
