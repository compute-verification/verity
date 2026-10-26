"""The guarded max and `max * scale` as committed values of every attention Call.  Opt-in (`manifest build --guarded-max`, the row's
GUARDED_MAX_TAP): with it off nothing here runs, and no manifest, Program, root or verdict changes.

In each key block after a row's first, the FA2 / FA3 online softmax reads `GuardNegInfZero(m)` (the running max, -inf read as 0) in two
places, the block's rescale factor and `max * scale`; and every block's exp2 units read `max_scaled = F32MulFtz(m_use, scale_log2)`.  The
no-recompute partition commits both.  The guarded-max tap builds (`ops/pod_fa2_tap.sh` FA2_TAP_ROW_GUARD=1, `ops/pod_fa3_tap.sh`
FA3_TAP_ROW_GUARD=1; `verity_tap_row3() == 1`, `verity_tap_src() & 64`) write the guarded max into ROW word 3
(`commit.hidden_stream.ROW_GUARD_FIELD`), which the default builds never write, and `max_scaled` into the MS class appended after FIN (one
word per slab, row and key block).  The MS class lengthens every hidden-stream identity, so the manifest's identities and digest move under
the policy (`with_ms`); the Commit refuses a tap build that does not match the manifest (`acquire.sources.hidden_source.make_hidden_source`).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable, Mapping

#: `verity_tap_row3()` of the tap build the policy needs (0 = ROW word 3 never written)
TAP_ROW3 = 1

POLICY_TEXT = ("guarded max: in every attention Call, GuardNegInfZero(m) of each key block after a query row's first is committed as word 3 "
               "of that (row, head, block)'s ROW entry in the attention hidden stream, and max * scale of every key block as that "
               "(row, head, block)'s word of the stream's MS class (the guarded-max tap build)")


def _attention_calls(P: Any) -> Iterable[tuple[str, int, int, int, int]]:
    """(family, NH, T, BN, MASKED_FROM) of every attention Call (`BN` = 64 where the Definition has no such static; `MASKED_FROM` = 0,
    every block guarded, except the per-iteration Check_inf chains, FA3's `Attention_v4` and FA2's `Attention_v5` / `_v18`)."""
    from verity_vllm.query.required import FLASH_ATTENTION_LAUNCHES

    for c in P.calls:
        if c.family not in FLASH_ATTENTION_LAUNCHES:
            continue
        st = c.statics
        if "T" not in st or "NH" not in st:
            raise ValueError(f"guarded max: {c.definition} has no (T, NH) statics; tensor-parallel attention is not covered")
        yield c.family, int(st["NH"]), int(st["T"]), int(st.get("BN", 64)), int(st.get("MASKED_FROM", 0))


def attention_words(P: Any) -> dict[str, int]:
    """The guarded maxes the query commits, by attention Definition family: NH x (ceil(T / BN) - 1) per Call, one per head and key block
    after the query row's first (under `Attention_v4` / `Attention_v5`, only the blocks at or above MASKED_FROM: an unguarded block reads the running max
    itself, ROW word 0)."""
    out: Counter = Counter()
    for fam, NH, T, BN, MF in _attention_calls(P):
        out[fam] += NH * max(0, -(-T // BN) - 1 - MF)
    return dict(sorted(out.items()))


def max_scaled_words(P: Any) -> dict[str, int]:
    """The `max_scaled` values the query commits, by attention Definition family: per Call, NH x the key blocks with more than one visible
    key (a block's single exp2 unit whose FMA reads `max_scaled` absorbs it); the first block included."""
    out: Counter = Counter()
    for fam, NH, T, BN, _mf in _attention_calls(P):
        nb = -(-T // BN)
        out[fam] += NH * (nb - (T - (nb - 1) * BN == 1))
    return dict(sorted(out.items()))


def _tally(words: Mapping[str, int]) -> dict[str, Any]:
    return {"words": sum(words.values()), "words_by_family": dict(sorted(words.items()))}


def header(q: dict[str, Any], words: Mapping[str, int], ms_words: Mapping[str, int]) -> dict[str, Any]:
    """The manifest's query header under the guarded-max policy; `words` is `attention_words` and `ms_words` `max_scaled_words`, summed
    over the request Programs."""
    text = f"{q['query_text']}; {POLICY_TEXT}"
    return dict(q, query_text=text, query_id_digest=hashlib.sha256(text.encode()).hexdigest(),
                guarded_max={"stream_family": q.get("attention_hidden_family"), "row_field": 3, "tap_row3": TAP_ROW3, **_tally(words),
                             "max_scaled": {"stream_class": "MS", "src_bit": 64, **_tally(ms_words)}})


def merged(q: dict[str, Any], parts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """A global manifest's query header `q` (its first component's) with the guarded maxes and `max_scaled` values of every component
    header in `parts` summed."""
    words: Counter = Counter()
    ms: Counter = Counter()
    for p in parts:
        words.update(p["guarded_max"]["words_by_family"])
        ms.update(p["guarded_max"]["max_scaled"]["words_by_family"])
    g = q["guarded_max"]
    return dict(q, guarded_max=dict(g, **_tally(words), max_scaled=dict(g["max_scaled"], **_tally(ms))))


def with_ms(man: dict[str, Any]) -> dict[str, Any]:
    """`man` with the MS class in every attention hidden-stream identity: its element range grows by HB x M x NB words (the class follows
    FIN), its geometry says `ms`, and the manifest digest is recomputed.  Identities that already say `ms` are kept."""
    from verity_vllm.query.manifest import format as fmt

    fam = man["query"].get("attention_hidden_family")
    ids = []
    for r in man["identities"]:
        g = r.get("geometry")
        if r.get("family") != fam or not isinstance(g, dict) or g.get("ms"):
            ids.append(r)
            continue
        dims = [int(g[k]) for k in ("HB", "M", "NB", "BN", "D")]
        if [int(x) for x in r["element_range"]] != [0, fmt.mat_total_words(*dims)]:
            raise ValueError(f"guarded max: {r['op_path']}/{r['output_member']} element range {r['element_range']} is not its geometry's stream")
        ids.append(dict(r, element_range=[0, fmt.mat_total_words(*dims, ms=True)], geometry=dict(g, ms=True)))
    return dict(man, identities=ids, manifest_digest=fmt.manifest_digest(ids))


def policy_of(manifest: Mapping[str, Any] | None) -> bool:
    """Whether a manifest was built under the guarded-max policy (its query header says so)."""
    q = (manifest or {}).get("query")
    return isinstance(q, dict) and isinstance(q.get("guarded_max"), dict)


__all__ = ["TAP_ROW3", "POLICY_TEXT", "attention_words", "max_scaled_words", "header", "merged", "with_ms", "policy_of"]
