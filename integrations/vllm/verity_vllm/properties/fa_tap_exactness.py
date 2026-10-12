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
      equals the installed kernel's;
  (e) a guarded-max build (`verity_tap_row3() == 1`, `query.guarded_max`) against a record of the default build (`baseline`): every stream
      word of the six classes but ROW word 3 equals the default build's (`masked_digest`), and word 3 is the IR's `GuardNegInfZero_v1` of
      the entry's row_max at every key block after a row's first (the guard's unit reads that committed word) and 0 at the first
      (`check_row3`; the IR reference on a sample, special values included, and its vectorised twin on every word);
  (f) the same build's MS class (`verity_tap_src() & 64`): at every visited key block the word is the IR's
      `F32MulFtz_v1(GuardNegInfZero_v1(row_max), scale_log2)` of the ROW entry's row_max, the value the block's exp2 units read
      (`check_ms`; twin on every word, the reference on a sample);
  (g) FA3 under the construction `check-inf-per-iteration` (`registry.fa3_check_inf`, `TargetProfile.fa3_construction`): the MS words
      against `AttnBlock_v4`'s value -- the running max guarded only at the key blocks the kernel runs with `Check_inf`, the tile's
      first and causal-masked ones (`masked_from`) -- and every output row against `Attention_v4` evaluated on the launch's inputs
      (`out_vs_ir`: the twin, and the reference evaluator on rows the twin does not cover).

Every stream word is `FILL` before the launch.  The `edge_rows` cases give some query rows the scores -inf (so the max and the guard see
-inf everywhere), +inf or NaN throughout (`SPECIAL`).

This module is the property and its record: the cases and negatives, the geometry and tile `hidden_source` declares (`fa2_swapped`,
`fa2_kblock_n`, `fa3_tile`), the words each launch must write, and `fa_tap_exactness.json` (environment, both .so sha256, per-case and
negative results, `ok`) whose `digest` is the sha256 of the canonical JSON of every other field; runs cite the record by that digest.
`verify_record` re-derives the digest and the verdict without a GPU.

The GPU run is `tests/properties/fa_tap_exactness_gpu.py`: it seeds the inputs, records the launch arguments `flash_attn_varlen_func`
passes to the installed op and parses its arguments, which library code may not do until the challenge module, `engine/hooks.py` and
`pipeline/cli.py` exist (lints P3, P9, P6).
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Iterator

import numpy as np

from verity_vllm.acquire.sources.hidden_source import KBLOCK_M, fa2_kblock_n, fa2_swapped, fa3_tile
from verity_vllm.commit.hidden_stream import ROW_GUARD_FIELD, SRC_BITS, StreamLayout
from verity_vllm.properties.record import digest_of, seal as seal_record


SCHEMA = "verity-vllm/fa-tap-exactness/v1"
RECORD = "fa_tap_exactness.json"
PAGE = 16
#: an f32 NaN payload no tap word takes (the kernels write finite, -inf and +0 words)
FILL = 0x7FC00001
#: the attention constructions an FA3 record is checked under (`TargetProfile.FA3_CONSTRUCTIONS`); the default is the record's
CONSTRUCTIONS = ("guard-every-block", "check-inf-per-iteration")
INSTALLED = {2: ("_vllm_fa2_C", "varlen_fwd"), 3: ("_vllm_fa3_C", "fwd")}
#: larger than every key length below, so no launch has a key block left of its window (n_block_min stays 0)
WINDOW = (4095, 0)
DECODE_K = [1, 17, 64, 65, 128, 129, 1000, 1031, 2047]
QUICK = ("prefill_short", "prefill_256", "decode_gqa", "g2_decode_gqa_softcap_window", "smallq_onewg", "edge_rows")
#: the edge case's sequences: a 130-row prefill (one and two key blocks) and three decodes (tokens 130, 131, 132) over 257, 300 and 1 keys
EDGE = ([130, 1, 1, 1], [130, 257, 300, 1])
#: what every score of a query token's rows is in the edge cases (the GPU run sets q[t, :, 0] = +inf / -inf / NaN against k[..., 0] < 0;
#: the softcap maps +-inf to +-CAP)
SPECIAL = {"-inf": [0, 64, 129, 130], "+inf": [70], "nan": [128, 131]}


def _case(name: str, H: int, Hk: int, seqlens_q: list[int], seqlens_k: list[int], softcap: float = 0.0,
          window: tuple[int, int] = (-1, -1), special: dict[str, list[int]] | None = None) -> dict[str, Any]:
    return {"name": name, "H": H, "Hk": Hk, "seqlens_q": list(seqlens_q), "seqlens_k": list(seqlens_k), "softcap": float(softcap),
            "window": list(window), **({"special": special} if special else {})}


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
          _case("decode_mha", 2, 2, [1] * 9, DECODE_K),
          _case("edge_rows", 4, 2, *EDGE, special=SPECIAL)]
    if fa_version == 2:
        g2 = [1, 17, 64, 65, 255, 257]
        g2_decode = [1, 17, 64, 65, 129, 1000, 1031]
        cs += [_case("g2_prefill_softcap", 8, 4, g2, g2, softcap=50.0),
               _case("g2_prefill_softcap_window", 8, 4, g2, g2, softcap=50.0, window=WINDOW),
               _case("g2_prefill_long_window", 4, 2, [1000, 1031], [1000, 1031], softcap=50.0, window=WINDOW),
               _case("g2_chunked_softcap_window", 8, 4, [17, 64, 1], [300, 1000, 129], softcap=50.0, window=WINDOW),
               _case("g2_decode_gqa_softcap", 8, 4, [1] * 7, g2_decode, softcap=50.0),
               _case("g2_decode_gqa_softcap_window", 8, 4, [1] * 7, g2_decode, softcap=50.0, window=WINDOW),
               _case("g2_decode_mha_window", 4, 4, [1] * 5, [1, 64, 65, 1000, 1031], window=WINDOW),
               _case("g2_edge_rows_softcap", 8, 4, *EDGE, softcap=50.0, special=SPECIAL)]
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


def masked_from(c: dict[str, Any], geo: dict[str, Any]) -> dict[tuple[int, int], int]:
    """FA3: (slab, packed row) -> the row's tile's `n_block_min_causal_local_mask` (`attention_vocab.fa3_masked_from`), the lowest key block
    the kernel runs with `Check_inf`: tile `m_block = row // kBM` starts at query row `m_block * kBM // g`, and the blocks at or above
    `(m_idx_min + sk - sq) // BN` are its first and causal-masked iterations."""
    BN, kBM, g, out = geo["BN"], geo["kBM"], geo["g"], {}
    for seq, (sq, sk) in enumerate(zip(c["seqlens_q"], c["seqlens_k"])):
        for hh in range(geo["h"]):
            for r in range(sq * g):
                out[(seq * geo["h"] + hh, r)] = ((r // kBM) * kBM // g + sk - sq) // BN
    return out


def expected_written(layout: StreamLayout, src: int, rows: Iterator[tuple[int, int, int]], row3: int = 0) -> np.ndarray:
    """Mask over the padded stream words: True where the launch must write (the planes of `src` at the visited rows), False elsewhere.
    Word 3 of a ROW entry is written only by a guarded-max build (`row3` 1), at every visited block; the MS class (`layout.ms`) likewise."""
    HB, M, N, NB, D = layout.HB, layout.M, layout.N, layout.NB, layout.D
    o = layout.offsets
    w = np.zeros(layout.padded_words(), dtype=bool)
    S, P = w[o["S"]:o["P"]].reshape(HB, M, N), w[o["P"]:o["RP"]].reshape(HB, M, N)
    RP, ROW = w[o["RP"]:o["ROW"]].reshape(HB, M, N // 2), w[o["ROW"]:o["O"]].reshape(HB, M, NB, 8)
    O, FIN = w[o["O"]:o["FIN"]].reshape(HB, M, NB, D), w[o["FIN"]:o["FIN"] + HB * M * 4].reshape(HB, M, 4)
    MS = w[o["MS"]:o["total"]].reshape(HB, M, NB) if layout.ms and src & SRC_BITS["MS"] else None
    for slab, r, nbm in rows:
        cols = nbm * layout.BN
        if src & SRC_BITS["S"]:
            S[slab, r, :cols] = True
        if src & SRC_BITS["P"]:
            P[slab, r, :cols] = True
        if src & SRC_BITS["RP"]:
            RP[slab, r, :cols // 2] = True
        if src & SRC_BITS["ROW"]:
            ROW[slab, r, :nbm, [0, 1, 2, 4, 5, 6, 7] + ([ROW_GUARD_FIELD] if row3 else [])] = True
        if src & SRC_BITS["O"]:
            O[slab, r, :nbm, :] = True
        if src & SRC_BITS["FIN"]:
            FIN[slab, r, :] = True
        if MS is not None:
            MS[slab, r, :nbm] = True
    return w


def _row_plane(words: np.ndarray, layout: StreamLayout) -> np.ndarray:
    o = layout.offsets
    return words[o["ROW"]:o["O"]].reshape(layout.HB, layout.M, layout.NB, 8)


def masked_digest(words: np.ndarray, layout: StreamLayout) -> str:
    """sha256 of the six classes' padded stream (u32 words, the padding `FILL`) with word 3 of every ROW entry replaced by `FILL`: equal
    for a guarded-max build and the default build on the same launch iff they write the same six classes but for that word (the MS
    class follows FIN and is not in it)."""
    six = dataclasses.replace(layout, ms=False)
    w = np.full(six.padded_words(), FILL, dtype=np.uint32)
    w[:six.total_words] = np.asarray(words, dtype=np.uint32)[:six.total_words]
    _row_plane(w, six)[..., ROW_GUARD_FIELD] = FILL
    return hashlib.sha256(w.tobytes()).hexdigest()


def ir_guard(m: int) -> int:
    """The IR model's `GuardNegInfZero_v1(m)` by the reference evaluator."""
    from verity.ml.circuits.evaluation.reference import evaluate
    from verity_vllm.program.registry import prims

    return int(evaluate(prims.GuardNegInfZero, int(m))[0])


def check_row3(words: np.ndarray, layout: StreamLayout, rows: Iterator[tuple[int, int, int]], ir_sample: int = 64) -> dict[str, Any]:
    """ROW word 3 of a guarded-max build at every visited (slab, row, block): the IR's `GuardNegInfZero_v1` of the entry's row_max (word 0)
    at every block after the row's first (the kernel visits blocks n_block_max - 1 .. 0), 0 at the first.  All words by the twin, and
    `ir_sample` of them (every distinct special row_max among them) by the reference evaluator."""
    from verity_vllm.program.kernels.derived_rows import guard_neg_inf_zero

    R = _row_plane(np.asarray(words, dtype=np.uint32), layout)
    later: list[tuple[int, int, int]] = []
    first: list[tuple[int, int, int]] = []
    for slab, r, nbm in rows:
        if nbm:
            first.append((slab, r, nbm - 1))
            later += [(slab, r, b) for b in range(nbm - 1)]
    rec: dict[str, Any] = {"guard_words": len(later), "guard_mismatch": 0, "first_block_nonzero": 0, "ir_sampled": 0, "ir_mismatch": 0}
    if first:
        s, r, b = np.array(first).T
        rec["first_block_nonzero"] = int(np.count_nonzero(R[s, r, b, ROW_GUARD_FIELD]))
    if later:
        s, r, b = np.array(later).T
        m, g = R[s, r, b, 0], R[s, r, b, ROW_GUARD_FIELD]
        bad = np.flatnonzero(g != guard_neg_inf_zero(m))
        rec["guard_mismatch"] = int(bad.size)
        if bad.size:
            i = int(bad[0])
            rec["first_mismatch"] = {"slab": int(s[i]), "row": int(r[i]), "block": int(b[i]), "row_max": int(m[i]), "word3": int(g[i])}
        special = np.flatnonzero(((m & 0x7F800000) == 0x7F800000) | (m == 0) | (m == 0x80000000))
        pick = np.unique(np.concatenate([special[np.unique(m[special], return_index=True)[1]],
                                         np.linspace(0, m.size - 1, min(ir_sample, m.size)).astype(int)]))
        rec["ir_sampled"] = int(pick.size)
        rec["ir_mismatch"] = sum(int(g[i]) != ir_guard(int(m[i])) for i in pick)
    return rec


def scale_log2_bits(c: dict[str, Any], D: int) -> int:
    """The f32 bits of the kernel's `scale_softmax_log2` for case `c` at head dim `D`, the IR's constant: softmax_scale * log2(e), or
    softcap * log2(e) with a softcap."""
    from verity_vllm.program.registry import b1, dense

    return dense.softcap_scale_log2_bits(c["softcap"]) if c["softcap"] else b1._scale_log2_bits(D)


def ir_max_scaled(m: int, scale_log2: int, check: bool = True) -> int:
    """The IR model's `F32MulFtz_v1(GuardNegInfZero_v1(m), scale_log2)` by the reference evaluator (`check` False: `AttnBlock_v4`'s
    unguarded iteration, `F32MulFtz_v1(m, scale_log2)`)."""
    from verity.ml.circuits.evaluation.reference import evaluate
    from verity_vllm.program.registry import prims

    return int(evaluate(prims.F32MulFtz, ir_guard(m) if check else int(m), int(scale_log2))[0])


def check_ms(words: np.ndarray, layout: StreamLayout, rows: Iterator[tuple[int, int, int]], scale_log2: int, ir_sample: int = 64,
             masked_from: dict[tuple[int, int], int] | None = None) -> dict[str, Any]:
    """The MS class at every visited (slab, row, block): the IR's `F32MulFtz_v1(GuardNegInfZero_v1(row_max), scale_log2)` of the ROW entry's
    row_max (word 0, the running max after the block).  With `masked_from` (FA3 under `check-inf-per-iteration`) the guard applies only at
    blocks >= the row's value, as `AttnBlock_v4{CHECK}`.  All words by the twin, `ir_sample` of them (every distinct special row_max among
    them) by the reference evaluator; mismatches whose row_max is -inf are counted apart."""
    from verity_vllm.program.kernels.derived_rows import guard_neg_inf_zero, mul_ftz

    w = np.asarray(words, dtype=np.uint32)
    R, o = _row_plane(w, layout), layout.offsets
    MS = w[o["MS"]:o["total"]].reshape(layout.HB, layout.M, layout.NB)
    at = [(slab, r, b) for slab, r, nbm in rows for b in range(nbm)]
    rec: dict[str, Any] = {"ms_words": len(at), "ms_mismatch": 0, "ms_mismatch_neg_inf_max": 0, "ir_sampled": 0, "ir_mismatch": 0}
    if masked_from is not None:
        rec["unguarded_words"] = sum(b < masked_from[(slab, r)] for slab, r, b in at)
    if not at:
        return rec
    s, r, b = np.array(at).T
    m, v = R[s, r, b, 0], MS[s, r, b]
    check = np.ones(len(at), dtype=bool) if masked_from is None else np.array([b_ >= masked_from[(s_, r_)] for s_, r_, b_ in at])
    bad = np.flatnonzero(v != mul_ftz(np.where(check, guard_neg_inf_zero(m), m), np.uint32(scale_log2)))
    rec["ms_mismatch"], rec["ms_mismatch_neg_inf_max"] = int(bad.size), int(np.count_nonzero(m[bad] == np.uint32(0xFF800000)))
    if bad.size:
        i = int(bad[0])
        rec["first_mismatch"] = {"slab": int(s[i]), "row": int(r[i]), "block": int(b[i]), "row_max": int(m[i]), "ms": int(v[i])}
    special = np.flatnonzero(((m & 0x7F800000) == 0x7F800000) | (m == 0) | (m == 0x80000000))
    pick = np.unique(np.concatenate([special[np.unique(m[special], return_index=True)[1]],
                                     np.linspace(0, m.size - 1, min(ir_sample, m.size)).astype(int)]))
    rec["ir_sampled"] = int(pick.size)
    rec["ir_mismatch"] = sum(int(v[i]) != ir_max_scaled(int(m[i]), scale_log2, bool(check[i])) for i in pick)
    return rec


def out_ok(e: dict[str, Any]) -> bool:
    """An FA3 case under `check-inf-per-iteration`: every output row equals `Attention_v4` on the launch's inputs."""
    r = e.get("out_vs_ir") or {}
    return r.get("rows", 0) > 0 and r.get("mismatch") == 0 and r.get("unevaluated") == 0


def ms_ok(e: dict[str, Any]) -> bool:
    """A guarded-max case's MS class: every visited word is the IR's, on the twin and on the reference sample."""
    r = e.get("ms") or {}
    return r.get("ms_mismatch") == 0 and r.get("ir_mismatch") == 0 and (r.get("ir_sampled", 0) > 0 or r.get("ms_words") == 0)


def row3_ok(e: dict[str, Any]) -> bool:
    """A guarded-max case: word 3 is the guard everywhere, 0 at first blocks, the IR agrees on a sample (when any row visits a block after its
    first), and every other word equals the baseline (default) build's."""
    r = e.get("row3") or {}
    return (r.get("guard_mismatch") == 0 and r.get("first_block_nonzero") == 0 and r.get("ir_mismatch") == 0
            and (r.get("ir_sampled", 0) > 0 or r.get("guard_words") == 0) and r.get("stream_eq_baseline") is True)


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
            and e.get("spill_outside_layout") == 0 and e.get("expected_but_unwritten") == 0 and ("row3" not in e or row3_ok(e))
            and ("ms" not in e or ms_ok(e)) and ("out_vs_ir" not in e or out_ok(e)))


def negative_ok(e: dict[str, Any]) -> bool:
    return not e.get("error") and e.get("skipped") == 1 and e.get("buffer_untouched") is True and e.get("out_eq_installed") is True


def _verdict(doc: dict[str, Any]) -> bool:
    """Every case and negative ok; a guarded-max build's record also names its baseline and checks word 3 (and, when the build writes the
    MS class, that class) in every case."""
    cs, ns = doc.get("cases") or [], doc.get("negatives") or []
    tap = doc.get("tap") or {}
    guarded, ms = bool(tap.get("row3")), bool(int(tap.get("src") or 0) & SRC_BITS["MS"])
    per_iteration = doc.get("construction", CONSTRUCTIONS[0]) == "check-inf-per-iteration"
    return (bool(cs) and bool(ns) and all(map(case_ok, cs)) and all(map(negative_ok, ns))
            and (not guarded or (bool((doc.get("baseline") or {}).get("digest")) and all("row3" in c for c in cs)))
            and (not ms or all("ms" in c for c in cs)) and (not per_iteration or all("out_vs_ir" in c for c in cs)))


def seal(doc: dict[str, Any]) -> dict[str, Any]:
    """`doc` with `ok` (at least one case and one negative, every one ok) and its `digest`."""
    return seal_record(doc, _verdict(doc))


def verify_record(doc: dict[str, Any]) -> list[str]:
    """What is wrong with a record: its schema, its digest, and every `ok` re-derived from the fields beside it (empty = intact and
    consistent)."""
    problems = [] if doc.get("schema") == SCHEMA else [f"schema {doc.get('schema')!r}, not {SCHEMA}"]
    if doc.get("digest") != digest_of(doc):
        problems.append("digest does not match the record")
    for kind, rule in (("cases", case_ok), ("negatives", negative_ok)):
        for e in doc.get(kind) or []:
            if bool(e.get("ok")) != rule(e):
                problems.append(f"{kind} {e.get('case')} hd{e.get('D')}: ok={e.get('ok')} but its fields say {rule(e)}")
    if bool(doc.get("ok")) != _verdict(doc):
        problems.append(f"ok={doc.get('ok')} but the entries say {_verdict(doc)}")
    return problems
