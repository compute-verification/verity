"""The swap-requests negative of the TP Commit: two requests' rows relabelled with each other's ids."""
from __future__ import annotations


def parse_swap_spec(spec: list | None) -> dict | None:
    """`[ra, rb]` = swap at EVERY step; `[ra, rb, lo, hi]` = swap ONLY on engine steps lo..hi inclusive (the decode-only swap -- prefill rows
    untouched, every swapped row one token, coverage/composition blind; attribution alone must catch it)."""
    if not spec or len(spec) < 2:
        return None
    ra, rb = str(spec[0]), str(spec[1])
    if len(spec) >= 4 and spec[2] is not None and spec[3] is not None:
        return {"a": ra, "b": rb, "lo": int(spec[2]), "hi": int(spec[3]), "mode": "steps"}
    return {"a": ra, "b": rb, "lo": None, "hi": None, "mode": "all"}


def swap_label(label: str, sw: dict | None, step: int | None) -> str:
    """The label a request's rows carry under the swap negative at `step` (None = no step context: the ALL-steps swap applies, a ranged one
    does not -- a ranged swap never touches a step-less consumer)."""
    if not sw:
        return label
    if sw.get("mode") == "steps" and (step is None or not (int(sw["lo"]) <= int(step) <= int(sw["hi"]))):
        return label
    if label == sw["a"]:
        return sw["b"]
    if label == sw["b"]:
        return sw["a"]
    return label
