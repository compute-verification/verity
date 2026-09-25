"""FA-tap exactness: the tap build the hidden sources launch in place of vLLM's FlashAttention computes exactly vLLM's result.

`acquire.hidden_source` replaces `_vllm_fa2_C.varlen_fwd` / `_vllm_fa3_C.fwd` with a tap build (`ops/pod_fa2_tap.sh` ->
`verity_fa2_matReq.so`, stream layout v8; `ops/pod_fa3_tap.sh` -> `verity_fa3_matReq.so`, layout v9) armed per launch with the tile it
declares.  One GPU process checks, for every head dim the build compiles and every case of `cases` (random bf16 varlen paged-KV inputs,
page 16, num_splits 1, softmax scale D^-0.5, causal as vLLM passes it):

  (a) `out` / `softmax_lse` of the armed tap launch are bit-identical to the installed vLLM kernel's and to the same build with the
      tap off;
  (b) the kernel accepted the declared tile (`verity_tap_skipped` unchanged);
  (c) the stream is closed: every word of the positions the launch visits (rows below the sequence's row count, key blocks below
      n_block_max, the planes the build writes) is written and no other word is (the fill pattern stays);
  (d) it fails closed: a declaration that is not the launched kernel's (wrong kBlockN, wrong head_dim; FA2: the swapped geometry on an
      un-swapped windowed decode; FA3: an M cap below the packed rows) is skipped (counter + 1), writes nothing, and the output still
      equals the installed kernel's.

This module is the property and its record: the cases and negatives, the geometry and tile `hidden_source` declares (`fa2_swapped`,
`fa2_kblock_n`, `fa3_tile`), the words each launch must write, and `fa_tap_exactness.json` (environment, both .so sha256, per-case and
negative results, `ok`) whose `digest` is the sha256 of the canonical JSON of every other field; runs cite the record by that digest.
`verify_record` re-derives the digest and the verdict without a GPU.

The GPU run is `tests/properties/fa_tap_exactness_gpu.py`: it seeds the inputs, records the launch arguments `flash_attn_varlen_func`
passes to the installed op and parses its arguments, which library code may not do until the challenge module, `engine/hooks.py` and
`pipeline/cli.py` exist (lints P3, P9, P6).
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterator

import numpy as np

from verity_vllm.acquire.sources.hidden_source import KBLOCK_M, fa2_kblock_n, fa2_swapped, fa3_tile
from verity_vllm.commit.hidden_stream import SRC_BITS, StreamLayout
from verity_vllm.correspondence.runtime import canonical_json


SCHEMA = "verity-vllm/fa-tap-exactness/v1"
RECORD = "fa_tap_exactness.json"
PAGE = 16
#: an f32 NaN payload no tap word takes (the kernels write finite, -inf and +0 words)
FILL = 0x7FC00001
INSTALLED = {2: ("_vllm_fa2_C", "varlen_fwd"), 3: ("_vllm_fa3_C", "fwd")}
#: larger than every key length below, so no launch has a key block left of its window (n_block_min stays 0)
WINDOW = (4095, 0)
DECODE_K = [1, 17, 64, 65, 128, 129, 1000, 1031, 2047]
QUICK = ("prefill_short", "prefill_256", "decode_gqa", "g2_decode_gqa_softcap_window", "smallq_onewg")


def _case(name: str, H: int, Hk: int, seqlens_q: list[int], seqlens_k: list[int], softcap: float = 0.0,
          window: tuple[int, int] = (-1, -1)) -> dict[str, Any]:
    return {"name": name, "H": H, "Hk": Hk, "seqlens_q": list(seqlens_q), "seqlens_k": list(seqlens_k), "softcap": float(softcap),
            "window": list(window)}


def cases(fa_version: int, quick: bool = False) -> list[dict[str, Any]]:
    """The launches checked at every head dim: lengths on both sides of the key-block and M-tile edges in mixed batches, chunked prefill
    (seqlen_q < seqlen_k, bottom-right causal), GQA and MHA decode.  FA2 adds the softcap and sliding-window kernel instantiations and the
    windowed GQA decode the kernel does not swap; FA3 (which disarms the tap for softcap and windows) adds the one-MMA-warpgroup tile
    with several rows per sequence and a chunked tail whose first rows see leading all-masked key blocks."""
    cs = [_case("prefill_short", 4, 2, [1, 17, 63, 64, 65], [1, 17, 63, 64, 65]),
          _case("prefill_mid", 4, 2, [127, 128, 129, 255], [127, 128, 129, 255]),
          _case("prefill_256", 4, 2, [256, 257, 5], [256, 257, 5]),
          _case("prefill_long", 2, 2, [1000, 1031], [1000, 1031]),
          _case("chunked", 4, 2, [17, 64, 1], [300, 1000, 129]),
          _case("decode_gqa", 8, 2, [1] * 9, DECODE_K),
          _case("decode_mha", 2, 2, [1] * 9, DECODE_K)]
    if fa_version == 2:
        g2 = [1, 17, 64, 65, 255, 257]
        g2_decode = [1, 17, 64, 65, 129, 1000, 1031]
        cs += [_case("g2_prefill_softcap", 8, 4, g2, g2, softcap=50.0),
               _case("g2_prefill_softcap_window", 8, 4, g2, g2, softcap=50.0, window=WINDOW),
               _case("g2_prefill_long_window", 4, 2, [1000, 1031], [1000, 1031], softcap=50.0, window=WINDOW),
               _case("g2_chunked_softcap_window", 8, 4, [17, 64, 1], [300, 1000, 129], softcap=50.0, window=WINDOW),
               _case("g2_decode_gqa_softcap", 8, 4, [1] * 7, g2_decode, softcap=50.0),
               _case("g2_decode_gqa_softcap_window", 8, 4, [1] * 7, g2_decode, softcap=50.0, window=WINDOW),
               _case("g2_decode_mha_window", 4, 4, [1] * 5, [1, 64, 65, 1000, 1031], window=WINDOW)]
    else:
        cs += [_case("smallq_onewg", 4, 2, [3, 5, 8, 1], [3, 700, 8, 129]),
               _case("chunked_tail", 4, 2, [64, 40], [164, 290])]
    return [c for c in cs if c["name"] in QUICK] if quick else cs


def geometry(fa_version: int, c: dict[str, Any], D: int, swapped: bool | None = None) -> dict[str, Any]:
    """The stream geometry `hidden_source`'s wrapper declares for case `c` at head dim `D`: HB slabs of M rows, NB key blocks, the tile
    (kBM, BN) = (kBlockM, kBlockN).  FA2: a slab is (sequence, head), or (sequence, KV head) with the group's query heads as rows when the
    kernel swaps (`fa2_swapped`; `swapped` overrides it); FA3: a slab is (sequence, KV head) and the rows are PackGQA-packed (query row,
    head in group) pairs, `g` per query row."""
    b, H, Hk = len(c["seqlens_q"]), c["H"], c["Hk"]
    max_q, max_k = max(c["seqlens_q"]), max(c["seqlens_k"])
    if fa_version == 2:
        if swapped is None:
            swapped = fa2_swapped(max_q, H, Hk, D, c["window"][0], c["window"][1])
        kBM, BN, g = KBLOCK_M, fa2_kblock_n(D, 1), 1
        h, M = (Hk, H // Hk) if swapped else (H, max_q)
    else:
        swapped, g = False, H // Hk
        kBM, BN = fa3_tile(D, max_q, g, 1)
        h, M = Hk, max_q * g
    return {"HB": b * h, "h": h, "M": M, "NB": -(-max_k // BN), "BN": BN, "kBM": kBM, "D": D, "g": g, "swapped": bool(swapped)}


def visited_rows(c: dict[str, Any], geo: dict[str, Any]) -> Iterator[tuple[int, int, int]]:
    """(slab, row, n_block_max) of every stream row the launch writes.  The kernels visit key blocks n_block_max - 1 .. 0; under the
    causal mask n_block_max is the M tile's, bottom-right aligned: the tile's last query row sees keys below (row + 1) + sk - sq (FA3
    divides packed rows by g, block.h get_n_block_min_max).  The swapped FA2 decode is not causal."""
    BN, kBM, g = geo["BN"], geo["kBM"], geo["g"]
    for seq, (sq, sk) in enumerate(zip(c["seqlens_q"], c["seqlens_k"])):
        blocks = -(-sk // BN)
        for hh in range(geo["h"]):
            slab = seq * geo["h"] + hh
            for r in range(geo["M"] if geo["swapped"] else sq * g):
                if geo["swapped"]:
                    yield slab, r, blocks
                    continue
                last_q = ((r // kBM + 1) * kBM - 1) // g + 1
                yield slab, r, max(0, min(blocks, -(-(last_q + sk - sq) // BN)))


def expected_written(layout: StreamLayout, src: int, rows: Iterator[tuple[int, int, int]]) -> np.ndarray:
    """Mask over the padded stream words: True where the launch must write (the planes of `src` at the visited rows), False elsewhere.
    Word 3 of a ROW entry is never written."""
    HB, M, N, NB, D = layout.HB, layout.M, layout.N, layout.NB, layout.D
    o = layout.offsets
    w = np.zeros(layout.padded_words(), dtype=bool)
    S, P = w[o["S"]:o["P"]].reshape(HB, M, N), w[o["P"]:o["RP"]].reshape(HB, M, N)
    RP, ROW = w[o["RP"]:o["ROW"]].reshape(HB, M, N // 2), w[o["ROW"]:o["O"]].reshape(HB, M, NB, 8)
    O, FIN = w[o["O"]:o["FIN"]].reshape(HB, M, NB, D), w[o["FIN"]:o["total"]].reshape(HB, M, 4)
    for slab, r, nbm in rows:
        cols = nbm * layout.BN
        if src & SRC_BITS["S"]:
            S[slab, r, :cols] = True
        if src & SRC_BITS["P"]:
            P[slab, r, :cols] = True
        if src & SRC_BITS["RP"]:
            RP[slab, r, :cols // 2] = True
        if src & SRC_BITS["ROW"]:
            ROW[slab, r, :nbm, [0, 1, 2, 4, 5, 6, 7]] = True
        if src & SRC_BITS["O"]:
            O[slab, r, :nbm, :] = True
        if src & SRC_BITS["FIN"]:
            FIN[slab, r, :] = True
    return w


def negatives(fa_version: int, D: int) -> list[tuple[dict[str, Any], dict[str, int], str]]:
    """(case, declared geometry, why) of every declaration the kernel must refuse at head dim `D`."""
    by_name = {c["name"]: c for c in cases(fa_version)}
    out = []
    for c in ([by_name["prefill_short"]] if fa_version == 2 else [by_name["prefill_short"], by_name["decode_gqa"]]):
        geo = geometry(fa_version, c, D)
        decl = {k: geo[k] for k in ("HB", "M", "NB", "BN", "D")}
        wrong_bn = (64 if geo["BN"] == 128 else 128) if fa_version == 2 else (128 if geo["BN"] != 128 else 192)
        out.append((c, {**decl, "D": 64 if D != 64 else 128}, "wrong head_dim"))
        out.append((c, {**decl, "BN": wrong_bn}, "wrong kBlockN"))
    if fa_version == 2:
        c = by_name["g2_decode_gqa_softcap_window"]
        geo = geometry(2, c, D, swapped=True)
        out.append((c, {k: geo[k] for k in ("HB", "M", "NB", "BN", "D")}, "swapped geometry declared for an un-swapped windowed decode"))
    else:
        geo = geometry(3, by_name["decode_gqa"], D)
        out.append((by_name["decode_gqa"], {**{k: geo[k] for k in ("HB", "M", "NB", "BN", "D")}, "M": geo["M"] - 1}, "M cap below the packed rows"))
    return out


# ------------------------------------------------------------------------------------------------------------------------ record
def case_ok(e: dict[str, Any]) -> bool:
    return (not e.get("error") and e.get("skipped") == 0 and e.get("out_lse_eq_installed") is True and e.get("out_lse_eq_untapped") is True
            and e.get("spill_outside_layout") == 0 and e.get("expected_but_unwritten") == 0)


def negative_ok(e: dict[str, Any]) -> bool:
    return not e.get("error") and e.get("skipped") == 1 and e.get("buffer_untouched") is True and e.get("out_eq_installed") is True


def _verdict(doc: dict[str, Any]) -> bool:
    cs, ns = doc.get("cases") or [], doc.get("negatives") or []
    return bool(cs) and bool(ns) and all(map(case_ok, cs)) and all(map(negative_ok, ns))


def _digest(doc: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json({k: v for k, v in doc.items() if k != "digest"})).hexdigest()


def seal(doc: dict[str, Any]) -> dict[str, Any]:
    """`doc` with `ok` (at least one case and one negative, every one ok) and its `digest`."""
    body = {k: v for k, v in doc.items() if k != "digest"}
    body["ok"] = _verdict(body)
    return {**body, "digest": _digest(body)}


def verify_record(doc: dict[str, Any]) -> list[str]:
    """What is wrong with a record: its schema, its digest, and every `ok` re-derived from the fields beside it (empty = intact and
    consistent)."""
    problems = [] if doc.get("schema") == SCHEMA else [f"schema {doc.get('schema')!r}, not {SCHEMA}"]
    if doc.get("digest") != _digest(doc):
        problems.append("digest does not match the record")
    for kind, rule in (("cases", case_ok), ("negatives", negative_ok)):
        for e in doc.get(kind) or []:
            if bool(e.get("ok")) != rule(e):
                problems.append(f"{kind} {e.get('case')} hd{e.get('D')}: ok={e.get('ok')} but its fields say {rule(e)}")
    if bool(doc.get("ok")) != _verdict(doc):
        problems.append(f"ok={doc.get('ok')} but the entries say {_verdict(doc)}")
    return problems
