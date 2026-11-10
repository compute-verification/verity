"""The blockwise FP8 GEMM's scale products as a required value: the committed interior words `Q_word_v1{16,32}` derives for
`ScaledMmFp8BlockSharedScale_v1{K,N,G}` (`TargetProfile.fp8_scale_construction = "shared-per-block"`, the default).  Each weight block's
K/G products `F32Mul_v1(x_s[kb], w_s[nb][kb])` are read by the block's G coordinates (a partial last block's by its N % G), so the
partition commits them: cdiv(N, G) * K/G f32 words per Call, weight block nb major, k block kb minor (`slots` reads them off the
partition and refuses any other set).

They are interior to the Call: `kernel_rows` adds one `scale_products` identity per (step, linear module), ranging over the module's
shared-scale Calls at that step, and the scale-products source (`acquire.sources.scale_products_source`) acquires
`<linear module>/scale_products`, f32 [rows, cdiv(N, G) * K/G], computed from the launch's activation scales and the module's
`weight_scale_inv` by `kernels.fp8_twins.scale_products`.  No kernel changes.  The policy applies to every manifest: a Program without a
shared-scale Call gets no identity and an unchanged header.  The Match record holds no value for it.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from functools import lru_cache
from typing import Any, Callable, Mapping

from verity.ml.definitions.registry import fp8
from verity.ml.definitions.registry import fp8_vocab
from verity.ml.definitions.registry import prims as P

FAMILY = "scale_products"
MEMBER = "scale_products"
#: the shared-scale GEMM Definitions whose Calls commit their scale products
SHARED = frozenset(d.id for d in fp8_vocab.SHARED_SCALE.values())


def blocks(st: Mapping[str, Any]) -> tuple[int, int]:
    """(cdiv(N, G), K/G) of a shared-scale Call's statics: a partial last weight block (N % G rows) has its K/G products too."""
    return fp8.weight_blocks(int(st["N"]), int(st["G"])), int(st["K"]) // int(st["G"])


def words(st: Mapping[str, Any]) -> int:
    nb, kb = blocks(st)
    return nb * kb


#: `Q_word_v1` committed interior words per Call that the source acquires (`word.check_calls`)
TAP_WORDS = {fam: words for fam in SHARED}

POLICY_TEXT = ("scale products: per token row of " + ", ".join(sorted(SHARED)) + " each weight block's committed products "
               "F32Mul_v1(x_s[kb], w_s[nb][kb]), nb major -- `<linear module>/scale_products` (family scale_products, f32 [rows, N/G * K/G])")


@lru_cache(maxsize=None)
def slots(K: int, N: int, G: int) -> tuple[int, ...]:
    """The shared-scale Call's committed interior gates under `Q_word_v1{16,32}` (offsets in its canonical layout, in gate order): word i
    of a row is the value of gate `slots(...)[i]`.  Refuses a partition whose cut fails, that recomputes a gate, or that commits other values."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions import word as W

    g = W.Graph(bind(fp8.ScaledMmFp8BlockSharedScale, K=K, N=N, G=G))
    R = W.units(g, 16, 32)
    if not R["cut"].ok or g.recomputed:
        raise ValueError(f"scale products: the partition of K={K} N={N} G={G} fails {R['cut'].codes} with {len(g.recomputed)} recomputed gates")
    owner, kinds = R["owner"], R["kinds"]
    crossing = set(g.src[owner[g.src] != owner[g.dst]].tolist())
    committed = sorted(x for x in crossing if kinds[owner[x]] == "committed")
    got = [g.prim_names[int(g.prim[x])] for x in committed]
    want = [P.F32Mul.id] * words({"K": K, "N": N, "G": G})
    if got != want:
        raise ValueError(f"scale products: K={K} N={N} G={G} commits {len(got)} interior values ({sorted(set(got))}), the source acquires "
                         f"{len(want)} {P.F32Mul.id}")
    return tuple(int(g.keep[x]) for x in committed)


def kernel_rows(pop: Any, add: Callable[..., None], impl: Callable[[str], str], aliases: Mapping[str, tuple[str, int]]) -> dict[str, Any] | None:
    """Add one `scale_products` identity per (step, linear module): its range is the module's shared-scale Calls at that step,
    cdiv(N, G) * K/G f32 each.  `add` is `request_manifest`'s row writer.  None when the Program has no shared-scale Call (the header
    is then unchanged)."""
    groups: dict[tuple[int, str], dict[str, Any]] = {}
    for c in pop.P.calls:
        if c.family not in SHARED:
            continue
        nb, kb = blocks(c.statics)
        key = (pop.step_of(c.id), pop.program_path(c.id))
        g = groups.setdefault(key, {"spec": c.family, "n_blocks": nb, "k_blocks": kb, "rows": 0})
        if (g["spec"], g["n_blocks"], g["k_blocks"]) != (c.family, nb, kb):
            raise ValueError(f"scale products: module {impl(key[1])} step {key[0]} runs two block geometries ({g}, {c.definition})")
        g["rows"] += 1
    if not groups:
        return None
    rows: Counter = Counter()
    total = 0
    for (step, mod), g in sorted(groups.items()):
        canon, k = aliases.get(mod, (mod, 0))
        w = g["n_blocks"] * g["k_blocks"]
        add(FAMILY, step, k, impl(canon), MEMBER, 0, g["rows"] * w, rows=g["rows"], width=w, dtype="f32", spec=g["spec"],
            n_blocks=g["n_blocks"], k_blocks=g["k_blocks"])
        rows[impl(canon)] += g["rows"]
        total += g["rows"] * w
    return {"words": total, "rows_by_site": dict(sorted(rows.items()))}


def header(q: dict[str, Any], stats: Mapping[str, Any] | None) -> dict[str, Any]:
    """The manifest's query header under the scale-products policy (`q` itself when the Program has no shared-scale Call, `stats` None)."""
    if stats is None:
        return q
    text = f"{q['query_text']}; {POLICY_TEXT}"
    out = dict(q, query_text=text, query_id_digest=hashlib.sha256(text.encode()).hexdigest(),
               required_families=[*q.get("required_families", []), FAMILY])
    out["scale_products"] = {"family": FAMILY, "member": MEMBER, "definitions": sorted(SHARED), "layout": "nb major, kb minor", **stats}
    return out


class Policy:
    """The scale-products policy as `required.request_manifest(taps=(POLICY,))` applies it."""

    name = FAMILY
    tap_words = TAP_WORDS
    kernel_rows = staticmethod(kernel_rows)
    header = staticmethod(header)


POLICY = Policy()


def policy_of(manifest: Mapping[str, Any] | None) -> bool:
    q = (manifest or {}).get("query")
    return isinstance(q, dict) and isinstance(q.get("scale_products"), dict)


def manifest_sites(manifest: Mapping[str, Any] | None) -> dict[str, dict[str, int]]:
    """op_path -> {"n_blocks", "k_blocks"} of every `scale_products` identity of a manifest; a site named with two geometries is refused."""
    out: dict[str, dict[str, int]] = {}
    for r in (manifest or {}).get("identities") or []:
        if r.get("family") != FAMILY:
            continue
        g = {"n_blocks": int(r["n_blocks"]), "k_blocks": int(r["k_blocks"])}
        if out.setdefault(str(r["op_path"]), g) != g:
            raise ValueError(f"scale products: {r['op_path']} is named with two block geometries ({out[str(r['op_path'])]}, {g})")
    return out


__all__ = ["FAMILY", "MEMBER", "SHARED", "TAP_WORDS", "POLICY_TEXT", "POLICY", "Policy", "blocks", "words", "slots", "kernel_rows", "header",
           "policy_of", "manifest_sites"]
