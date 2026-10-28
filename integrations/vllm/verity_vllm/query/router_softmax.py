"""The MoE router's softmax as a required value: the committed interior words `Q_word_v1{16,32}` derives for the selection-order router
(`MoeRouterTopKOrdered_v1` / `MoeRouterTopKOrderedNorm_v1`, `moe_construction = "indexed-read-ordered"`).  Opt-in (`manifest build
--router-softmax`, the row's ROUTER_TAP): with it off nothing here runs, and no manifest, Program, root or verdict changes.

The router computes its softmax once and every selection round, weight and (Norm) epilogue reads it, so the partition commits, per token
row, in the Call's canonical gate order (`layout`): the row max, the E exponentials, the reciprocal of their sum and the E probabilities
(2E + 2 words); the Norm router adds the TOPK selected probabilities and the epilogue's scale.  `slots` reads those gates off the
partition itself and refuses any other set.  They are interior to the Call, so no Program Value carries them: `kernel_rows` adds one
`router_softmax` identity per (step, MoE layer), ranging over the layer's router Calls at that step, `words` f32 each, and the
router-softmax tap acquires it (`acquire.sources.router_softmax_source`, vLLM's `topkGating` with its softmax stored) as
`moe/L<k>/router_softmax`, one row of `words` per token in the router input's row order.  The Match record holds no value for it.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from functools import lru_cache
from typing import Any, Callable, Mapping

FAMILY = "router_softmax"
MEMBER = "router_softmax"
#: the selection-order router Definitions -> whether the Call renormalizes (the Norm epilogue)
ROUTERS: dict[str, bool] = {"MoeRouterTopKOrdered_v1": False, "MoeRouterTopKOrderedNorm_v1": True}
#: (name, (activation family, primitive)) of each committed value, in the Call's canonical gate order
#: (the exponentials and the reciprocal are their NaN-canonical selects, `registry.moe._ptx_nan`)
_CLASSES = (("max", ("MoeRouterProbs_v1", "F32Fmaxf_v1")), ("ex", ("MoeRouterProbs_v1", "SelectF32_v1")),
            ("rcp", ("MoeRouterProbs_v1", "SelectF32_v1")), ("p", ("MoeRouterProbs_v1", "SelectF32_v1")),
            ("sel", ("MoeRouterPick_v1", "SelectF32_v1")), ("scale", ("MoeRouterTopKOrderedNorm_v1", "F32Div_v1")))

#: `Q_word_v1` committed interior words per Call that the tap acquires, as a function of the router's statics (`word.check_calls`)
TAP_WORDS = {fam: (lambda st, r=renorm: words(int(st["E"]), int(st["TOPK"]), r)) for fam, renorm in ROUTERS.items()}

POLICY_TEXT = ("router softmax: per token row of " + ", ".join(sorted(ROUTERS)) + " the committed softmax -- max, E exponentials, "
               "1/sum, E probabilities (+ TOPK selected probabilities and the scale under renormalize) -- `moe/L<k>/router_softmax` (family "
               "router_softmax, router-softmax tap)")


def layout(E: int, TOPK: int, renormalize: bool) -> tuple[tuple[str, int], ...]:
    """(value, count) of one row's words, in order: max, ex[E], rcp, p[E] (+ sel[TOPK], scale)."""
    counts = {"max": 1, "ex": E, "rcp": 1, "p": E, "sel": TOPK if renormalize else 0, "scale": 1 if renormalize else 0}
    return tuple((name, counts[name]) for name, _cls in _CLASSES if counts[name])


def words(E: int, TOPK: int, renormalize: bool) -> int:
    return sum(n for _v, n in layout(E, TOPK, renormalize))


def offsets(E: int, TOPK: int, renormalize: bool) -> dict[str, int]:
    """The first word of each value in a row."""
    out, at = {}, 0
    for name, n in layout(E, TOPK, renormalize):
        out[name] = at
        at += n
    return out


def router_fn(E: int, TOPK: int, VPT: int, renormalize: bool) -> Any:
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import moe

    return bind(moe.MoeRouterTopKOrderedNorm if renormalize else moe.MoeRouterTopKOrdered, E=E, TOPK=TOPK, VPT=VPT)


@lru_cache(maxsize=None)
def slots(E: int, TOPK: int, VPT: int, renormalize: bool) -> tuple[int, ...]:
    """The router Call's committed interior gates under `Q_word_v1{16,32}`, as offsets in the Call's canonical layout, in gate order:
    word i of a tapped row is the value of gate `slots(...)[i]`.  Refuses a partition whose cut fails, that recomputes a gate, or whose
    committed interior values are not `layout`'s, class by class."""
    from verity.ml.definitions import word as W

    G = W.Graph(router_fn(E, TOPK, VPT, renormalize))
    R = W.units(G, 16, 32)
    if not R["cut"].ok or G.recomputed:
        raise ValueError(f"router softmax: the partition of E={E} TOPK={TOPK} renormalize={renormalize} fails {R['cut'].codes} "
                         f"with {len(G.recomputed)} recomputed gates")
    owner, kinds = R["owner"], R["kinds"]
    crossing = set(G.src[owner[G.src] != owner[G.dst]].tolist())
    committed = sorted(g for g in crossing if kinds[owner[g]] == "committed")
    got = [(G.scope_names[int(G.scope[g])].split("{")[0], G.prim_names[int(G.prim[g])]) for g in committed]
    cls = dict(_CLASSES)
    want = [cls[name] for name, n in layout(E, TOPK, renormalize) for _ in range(n)]
    if got != want:
        first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        raise ValueError(f"router softmax: E={E} TOPK={TOPK} renormalize={renormalize} commits {len(got)} interior values, the tap writes "
                         f"{len(want)}; first difference at word {first}: {got[first:first + 1]} vs {want[first:first + 1]}")
    return tuple(int(G.keep[g]) for g in committed)


def kernel_rows(pop: Any, add: Callable[..., None], impl: Callable[[str], str], aliases: Mapping[str, tuple[str, int]] | None = None) -> dict[str, Any]:
    """Add one `router_softmax` identity per (step, MoE layer): its range is the layer's router Calls at that step, `words` f32 each.
    `add` is `request_manifest`'s row writer (`aliases` is unused: MoE identities are named by layer).  Returns the words and rows by
    Definition family."""
    from verity_vllm.query.manifest import format as fmt

    groups: dict[tuple[int, str], dict[str, Any]] = {}
    for c in pop.P.calls:
        if c.family not in ROUTERS:
            continue
        st = c.statics
        E, TOPK, VPT = int(st["E"]), int(st["TOPK"]), int(st["VPT"])
        mod = fmt.split_name(c.name or "")[0]
        g = groups.setdefault((pop.step_of(c.id), mod), {"family": c.family, "E": E, "TOPK": TOPK, "VPT": VPT, "rows": 0})
        if (g["family"], g["E"], g["TOPK"], g["VPT"]) != (c.family, E, TOPK, VPT):
            raise ValueError(f"router softmax: module {impl(mod)} step {pop.step_of(c.id)} runs two routers ({g['family']}, {c.family})")
        g["rows"] += 1
    total: Counter = Counter()
    rows: Counter = Counter()
    for (step, mod), g in sorted(groups.items()):
        renorm = ROUTERS[g["family"]]
        n = words(g["E"], g["TOPK"], renorm)
        add(FAMILY, step, 0, f"moe/L{fmt.layer_of(mod)}", MEMBER, 0, g["rows"] * n, rows=g["rows"], width=n, dtype="f32", spec=g["family"],
            moe_module=impl(mod), experts=g["E"], topk=g["TOPK"], renormalize=renorm)
        total[g["family"]] += g["rows"] * n
        rows[g["family"]] += g["rows"]
    return {"words": sum(total.values()), "words_by_family": dict(sorted(total.items())), "rows_by_family": dict(sorted(rows.items()))}


def header(q: dict[str, Any], stats: Mapping[str, Any] | None) -> dict[str, Any]:
    """The manifest's query header under the router-softmax policy (`q` itself when the policy is off, `stats` None)."""
    if stats is None:
        return q
    text = f"{q['query_text']}; {POLICY_TEXT}"
    out = dict(q, query_text=text, query_id_digest=hashlib.sha256(text.encode()).hexdigest(),
               required_families=[*q.get("required_families", []), FAMILY])
    out["router_softmax"] = {"family": FAMILY, "member": MEMBER, "routers": dict(ROUTERS), "layout": "max, ex[E], rcp, p[E] (+ sel[TOPK], scale)",
                             **stats}
    return out


class Policy:
    """The router-softmax policy as `required.request_manifest(taps=(POLICY,))` applies it."""

    name = FAMILY
    tap_words = TAP_WORDS
    kernel_rows = staticmethod(kernel_rows)
    header = staticmethod(header)


POLICY = Policy()


def policy_of(manifest: Mapping[str, Any] | None) -> bool:
    """Whether a manifest was built under the router-softmax policy (its query header says so)."""
    q = (manifest or {}).get("query")
    return isinstance(q, dict) and isinstance(q.get("router_softmax"), dict)


def manifest_layers(manifest: Mapping[str, Any] | None) -> dict[int, dict[str, Any]]:
    """MoE layer -> {"module", "experts", "topk", "renormalize", "width"} of every `router_softmax` identity of a manifest; a layer named
    with two geometries is refused."""
    out: dict[int, dict[str, Any]] = {}
    for r in (manifest or {}).get("identities") or []:
        if r.get("family") != FAMILY:
            continue
        layer = int(str(r["op_path"]).rsplit("/L", 1)[1])
        g = {"module": r.get("moe_module"), "experts": int(r["experts"]), "topk": int(r["topk"]), "renormalize": bool(r["renormalize"]),
             "width": int(r["width"])}
        if out.setdefault(layer, g) != g:
            raise ValueError(f"router softmax: layer {layer} is named with two geometries ({out[layer]}, {g})")
    return out


def words_per_token(manifest: Mapping[str, Any] | None) -> int:
    """The softmax words one token row commits across the manifest's MoE layers: the source's arena bound."""
    return sum(g["width"] for g in manifest_layers(manifest).values())


__all__ = ["FAMILY", "MEMBER", "ROUTERS", "TAP_WORDS", "POLICY_TEXT", "POLICY", "Policy", "layout", "words", "offsets", "router_fn", "slots",
           "kernel_rows", "header", "policy_of", "manifest_layers", "words_per_token"]
