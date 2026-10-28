"""A top-p select's split count S in a compared pair of Programs.

`program_compare.Prog._bind_splits` binds each select's `Const32[S]` operand and removes it from the sequence, so neither the
canonical hash nor a DAG hash carries S.  A Program at the registered S carries its own `Const32[S]`, bound the same way
(`sampler_geometry.constant_splits_agree` holds the two sides' values equal per event), and may state its select specialised on S:
`GumbelTopPTokenSelect_v2{V,S}` (`registry.topp_words`: the keep word as word gates at the constant split count, the same six
operands), or `GumbelTopPTokenSelect_v3{V,S}`, v2 without its dead gates (`registry.dead_free`, the registered compiler's form).
The fold never does, so a specialised select compares as the fold's `GumbelTopPTokenSelect_v1{V}` (`as_fold_selects`) with its S
held to the value of the `Const32` its splits operand binds (`hold`).
"""
from __future__ import annotations

import re

#: a specialised select's spec: an optional scope prefix, V and S
SPECIALISED = re.compile(r"((?:[^:]*:)?)GumbelTopPTokenSelect_v[23]\{V=(\d+),S=(\d+)\}")
FOLD_SELECT = "GumbelTopPTokenSelect_v1"


def as_fold_selects(inst: list[dict]) -> dict[int, int]:
    """Rewrite each specialised select's spec to the fold's `GumbelTopPTokenSelect_v1{V}` (in place); {select node k: S}."""
    out = {}
    for r in inst:
        m = SPECIALISED.fullmatch(r["spec"])
        if m:
            r["spec"] = f"{m.group(1)}{FOLD_SELECT}{{V={m.group(2)}}}"
            out[r["k"]] = int(m.group(3))
    return out


def mismatches(P) -> list[int]:
    """The specialised selects of a compared Program whose bound `Const32` is not their S."""
    bound = getattr(P, "splits_values", {})
    return sorted(k for k, S in getattr(P, "split_statics", {}).items() if bound.get(k) != S)


def hold(A, B, res: dict, constants: dict | None) -> None:
    """`program_compare.compare_steps`' result `res` held to both sides' split counts: `splits_values`, the per-event agreement of
    their own `Const32[S]` (`constants`, from `constant_splits_agree`; None when either side binds none), and `split_statics`, each
    side's specialised selects and mismatches (when either has one).  A disagreement makes the canonical comparison unequal."""
    eq = res["canonical_sha256"]["equal"]
    if constants is not None:
        res["splits_values"], eq = constants, eq and constants["equal"]
    sta, stb = getattr(A, "split_statics", {}), getattr(B, "split_statics", {})
    if sta or stb:
        bad = {"a": mismatches(A), "b": mismatches(B)}
        res["split_statics"], eq = {"a": len(sta), "b": len(stb), "mismatch": bad}, eq and not (bad["a"] or bad["b"])
    res["canonical_sha256"]["equal"] = eq
