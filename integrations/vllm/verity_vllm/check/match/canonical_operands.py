"""Canonical operand forms shared by the comparator's record and progression mappings (`program_compare.Prog`)."""

from __future__ import annotations


def weight_field_name(name: str) -> str:
    """Module-path-normalised weight name: the derived wrapper holds the model as `model.` (so `model.model.layers...`),
    the record's hints are relative to the model (`model.layers...`); strip ONE leading `model.` from the wrapper's names
    only when a second `model.` follows or the rest is `lm_head`/`model...`.  The tensor-parallel rank wrapper (derive_step
    --tp: the TP stub holds the request wrapper as `inner`, the same convention as tp_links' `module_path` = `inner.model.<path> (<Class>)`)
    adds one level: `inner.model.model.layers...` -> `model.layers...` (measured: the ONLY operand difference of the first TP2 B1 per-rank
    compare was this prefix on every weight / cos-sin operand, 37,342 of 111,771; the collective and activation wiring was equal)."""
    if name.startswith("inner."):
        name = name[len("inner."):]
    if name.startswith("model.model.") or name.startswith("model.lm_head."):
        return name[len("model."):]
    return name


def as_progression(t: tuple) -> tuple:
    """A record-form canonical tuple as a one-run progression `head + (base, zero delta, 1)`, with the head/base split the
    uniform mappings use: N / N-PAD -> base (s, jj, lo, hi); XREQ -> head (XREQ, kind), base (idx, lo, hi); W / SUPPLIED -> head
    (W | SUPPLIED, name), base (lo, hi); W? / MUX-UNFOLD-FAILED -> base (idx | j, lo, hi); TOK / SEED / SPLITS -> base (lo, hi); a raw
    run -> base (idx, lo, hi)."""
    k = t[0]
    if k in ("N", "N-PAD"):
        base = t[1:5]
    elif k in ("W", "SUPPLIED"):
        base = t[2:4]
    elif k in ("TOK", "SEED", "SPLITS"):
        base = t[1:3]
    elif k == "XREQ":
        base = t[2:5]
    else:   # W?, MUX-UNFOLD-FAILED, raw (space, idx, lo, hi)
        base = t[-3:]
    head = t[:len(t) - len(base)]
    return head + (base, (0,) * len(base), 1)
