#!/usr/bin/env python
"""R13 lane hopper (2b): exactness test of the FA3 (sm90) M1 hidden-value tap `verity_fa3_matReq.so` (`pod_fa3_tap.sh`, layout v9).

The FA2 test (`fa2_tap_xcheck.py`, lane fa2) transposed to FlashAttention-3 on the H100; the two phases and every verdict are the same:

  dump    (venv-cu129, GPU)  per head dim (64, 128) and case: random bf16 vLLM-style varlen paged inputs -> the TAPPED `fwd` with the
          M1 stream armed for the declared tile (kBlockN, head_dim) -> (a) verity_tap_skipped == 0, (b) out / softmax_lse bit-identical
          to the installed `_vllm_fa3_C.fwd` and to the same .so untapped, (c) nothing written outside the layout's visited positions,
          everything inside written; planes + inputs -> <out>/<case>.npz.  Fail-closed negatives: a declaration that is not the
          launched kernel's tile (wrong kBlockN / head_dim) is SKIPPED (counter +1, buffer untouched).
  xcheck  (venv-py312, CPU)  for sampled (sequence, KV head, packed row) triples: the Verity-semantics oracle for the FA3 chain --
          `fa2_attn_oracle.attention_head` re-instantiated in the Hopper binding environment of `verity_vllm.registry.hopper`
          (HopperBF16WgmmaDot16 for every k16 step, Fa3InvSum for the epilogue, kBlockN of the launched tile), i.e. `AttnBlockFA3` /
          `AttentionHeadFA3` word for word -- on the dump's own inputs; every tapped word (S masked, P, bf16 P, row max, rescale, lane
          sums, O after each block, sum, 1/sum) and the bf16 output row must be bit-equal.  Negative control: injected faults reported.

What is FA3-specific (and where the FA2 test's assumptions are replaced):
  * PackGQA: the kernel's rows are packed (query row, q-head-in-group) pairs m = q_row * g + head_in_group (g = H / Hk); the slab is
    (sequence, KV head); M = max_seqlen_q * g.  The decode "swapped" kernel of FA2 does not exist: decode is PackGQA with seqlen_q = 1.
  * the tile is (kBlockM, kBlockN) = tile_size_fwd_sm90 (hdim 64: 192 x 128, or 64 x 192 under one MMA warpgroup, i.e.
    max_seqlen_q * g <= 64; hdim 128: 128 x 128 or 64 x 128) -- `fa3_tile`, cross-checked with `verity_vllm.registry.targets.fa3_kblock_n`;
  * causal n_block_max per M tile follows block.h::get_n_block_min_max with the PackGQA divide: m_idx_max = (((m_block+1)*kBlockM - 1)
    // g) + 1, n_block_max = min(ceil(sk / BN), ceil((m_idx_max + sk - sq) / BN)); rows of the tile below its last query row see
    leading all-masked blocks (checked as the neutral state, as FA2).
  * causal is always on (vLLM passes causal=True for decode too; with seqlen_q = 1 it is the full key range).

  PY=/workspace/venv-cu129/bin/python; PY312=/workspace/venv-py312/bin/python; cd /workspace/cp/veritor-hopper
  $PY    verity_capture/sweep/fa3_tap_xcheck.py dump   --so /workspace/cp/fa2/build/fa3_matReq/verity_fa3_matReq.so --hdims 64,128 --out /workspace/cp/fa2/xcheck_fa3
  $PY312 verity_capture/sweep/fa3_tap_xcheck.py xcheck --dumps /workspace/cp/fa2/xcheck_fa3 --out /workspace/cp/fa2/xcheck_fa3/xcheck.json [--rows 4]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (REPO, os.path.join(REPO, "vllm-poc"), os.path.join(REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

_spec = importlib.util.spec_from_file_location("fa2_tap_xcheck", os.path.join(HERE, "fa2_tap_xcheck.py"))
X = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(X)          # shared: make_case, pick_rows, f32w, eq, FILL, NEG_INF
PAGE, FILL, NEG_INF, NUM_SPLITS = X.PAGE, X.FILL, X.NEG_INF, X.NUM_SPLITS
LAYOUT_VERSION = 9


def fa3_tile(D: int, max_seqlen_q: int, g: int) -> tuple[int, int]:
    """(kBlockM, kBlockN) of the FA3 forward vLLM launches (bf16, causal, paged KV without TMA, PackGQA), hopper/tile_size.h::
    tile_size_fwd_sm90 with heuristics.h::use_one_mma_wg (seqlen_q * qheads_per_kvhead <= 64, seqlen_q = max_seqlen_q under varlen)."""
    one_wg = max_seqlen_q * g <= 64
    if D == 64:
        return (64, 192) if one_wg else (192, 128)
    if D == 128:
        return (64, 128) if one_wg else (128, 128)
    raise ValueError(f"FA3 head dim {D}: not modelled")


def cases_for(D: int, quick: bool) -> list[dict]:
    cs = X.cases_for(D, quick)
    # one-MMA-warpgroup tile with several causal rows per sequence (speculative-decode-like shapes: max_seqlen_q * g <= 64)
    cs.append(dict(name="smallq_onewg", kind="prefill", H=4, Hk=2, seqlens_q=[3, 5, 8, 1], seqlens_k=[3, 700, 8, 129], causal=True, D=D))
    # chunked prefill whose first rows see a leading all-masked key block under the TILE's causal n_block_max (partial-domain rows on
    # every tile shape: hdim 64 192x128 and hdim 128 128x128 -- with sq == sk and g = 2 the 128-row tile never has one)
    cs.append(dict(name="chunked_tail", kind="prefill", H=4, Hk=2, seqlens_q=[64, 40], seqlens_k=[164, 290], causal=True, D=D))
    return cs


# ------------------------------------------------------------------------------------------------------------------ dump (GPU)
def _installed_so() -> str:
    import vllm.vllm_flash_attn as m
    d = os.path.dirname(m.__file__)
    so = glob.glob(os.path.join(d, "_vllm_fa3_C*.so"))
    assert so, d
    return so[0]


def call(ns, t):
    """torch.ops.<ns>.fwd with vLLM's argument list (vllm_flash_attn/flash_attn_interface.py, fa_version 3): paged KV through
    page_table + seqused_k, no descales (bf16), causal, num_splits 1, pack_gqa left to the kernel (varlen -> always packed)."""
    import torch
    op = getattr(torch.ops, ns).fwd
    r = op(t["q"], t["k"], t["v"], None, None, None, None, t["cu_seqlens_q"], None, None, None, t["seqused_k"], t["max_seqlen_q"],
           t["max_seqlen_k"], t["block_table"], None, None, None, None, None, None, None, None, t["scale"], t["causal"], -1, -1, 0.0,
           True, None, NUM_SPLITS, None, 0, None, 1, 0, None)
    return r[0], r[1]


def geometry(t):
    b = int(t["cu_seqlens_q"].numel()) - 1
    H, Hk, D = t["H"], t["Hk"], t["D"]
    g = H // Hk
    kBM, BN = fa3_tile(D, int(t["max_seqlen_q"]), g)
    M = int(t["max_seqlen_q"]) * g
    NB = (int(t["max_seqlen_k"]) + BN - 1) // BN
    return dict(b=b, h=Hk, g=g, M=M, NB=NB, BN=BN, kBM=kBM, D=D, HB=b * Hk)


def n_block_max(sq: int, sk: int, m_block: int, kBM: int, BN: int, g: int, causal: bool) -> int:
    nb = (sk + BN - 1) // BN
    if causal:
        m_idx_max = ((m_block + 1) * kBM - 1) // g + 1
        nb = min(nb, (m_idx_max + sk - sq + BN - 1) // BN)
    return max(nb, 0)


def expected_written_mask(layout, g_, seqlens_q, seqlens_k, causal, src):
    from verity_capture.experimental.fa2_commit.hidden_stream import SRC_BITS
    HB, M, NB, N, D = layout.HB, layout.M, layout.NB, layout.N, layout.D
    o = layout.offsets
    w = np.zeros(layout.padded_words(), dtype=bool)
    S = w[o["S"]:o["P"]].reshape(HB, M, N); P = w[o["P"]:o["RP"]].reshape(HB, M, N); RP = w[o["RP"]:o["ROW"]].reshape(HB, M, N // 2)
    ROW = w[o["ROW"]:o["O"]].reshape(HB, M, NB, 8); O = w[o["O"]:o["FIN"]].reshape(HB, M, NB, D); FIN = w[o["FIN"]:o["total"]].reshape(HB, M, 4)
    for seq in range(g_["b"]):
        sq, sk = seqlens_q[seq], seqlens_k[seq]
        for kvh in range(g_["h"]):
            slab = seq * g_["h"] + kvh
            for r in range(sq * g_["g"]):
                nbm = n_block_max(sq, sk, r // g_["kBM"], g_["kBM"], g_["BN"], g_["g"], causal)
                if src & SRC_BITS["S"]: S[slab, r, :nbm * g_["BN"]] = True
                if src & SRC_BITS["P"]: P[slab, r, :nbm * g_["BN"]] = True
                if src & SRC_BITS["RP"]: RP[slab, r, :nbm * g_["BN"] // 2] = True
                if src & SRC_BITS["ROW"]: ROW[slab, r, :nbm, [0, 1, 2, 4, 5, 6, 7]] = True
                if src & SRC_BITS["O"]: O[slab, r, :nbm, :] = True
                if src & SRC_BITS["FIN"]: FIN[slab, r, :] = True
    return w


def cmd_dump(a):
    import torch
    from verity_capture.experimental.fa2_commit.hidden_stream import StreamLayout
    os.makedirs(a.out, exist_ok=True)
    torch.ops.load_library(_installed_so())
    torch.ops.load_library(a.so)
    ns = os.path.basename(a.so)[:-3]
    tap = getattr(torch.ops, ns)
    assert hasattr(tap, "verity_set_tap_v9"), f"{a.so}: not a v9 FA3 tap build"
    mode, src, hd_mask, ver = int(tap.verity_tap_mode()), int(tap.verity_tap_src()), int(tap.verity_tap_hdims()), int(tap.verity_tap_layout_version())
    assert mode == 3 and ver == LAYOUT_VERSION, (mode, ver)
    hdims = [int(h) for h in a.hdims.split(",")]
    summary = {"so": a.so, "ns": ns, "src": src, "hdims_mask": hd_mask, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "installed_so": _installed_so(), "cases": [], "negatives": [], "ok": True}
    try:
        from verity_vllm.registry.targets import fa3_kblock_n
    except Exception:  # noqa: BLE001  (venv-cu129 may lack the 3.12-only registry)
        fa3_kblock_n = None
    for D in hdims:
        assert hd_mask & (1 << (D // 32)), f"hd {D} not compiled ({hd_mask})"
        for ci, c in enumerate(cases_for(D, a.quick)):
            t = X.make_case(c, seed=1000 * D + ci)
            g = geometry(t)
            BN = g["BN"]
            if fa3_kblock_n is not None:
                assert fa3_kblock_n(D, int(t["max_seqlen_q"]), g["g"]) == BN, (D, t["max_seqlen_q"], g["g"], BN)
            L = StreamLayout(g["HB"], g["M"], g["NB"], BN, D)
            buf = torch.full((L.padded_words(),), FILL, dtype=torch.int32, device="cuda")
            sk0 = int(tap.verity_tap_skipped())
            tap.verity_set_tap_v9(3, buf, g["HB"], g["M"], g["NB"], BN, D, 7)
            out_t, lse_t = call(ns, t)
            torch.cuda.synchronize()
            tap.verity_set_tap_off()
            skipped = int(tap.verity_tap_skipped()) - sk0
            out_u, lse_u = call(ns, t)                        # same .so, tap off
            out_i, lse_i = call("_vllm_fa3_C", t)             # the installed vLLM kernel
            torch.cuda.synchronize()
            eq_u = bool(torch.equal(out_t.view(torch.int16), out_u.view(torch.int16)) and torch.equal(lse_t.view(torch.int32), lse_u.view(torch.int32)))
            eq_i = bool(torch.equal(out_t.view(torch.int16), out_i.view(torch.int16)) and torch.equal(lse_t.view(torch.int32), lse_i.view(torch.int32)))
            words = buf.cpu().numpy().view(np.uint32)
            exp = expected_written_mask(L, g, c["seqlens_q"], c["seqlens_k"], c["causal"], src)
            written = words != FILL
            spill = int(np.count_nonzero(written & ~exp))
            unwritten = int(np.count_nonzero(exp & ~written))
            rec = {"case": c["name"], "D": D, "BN": BN, "kBM": g["kBM"], "kind": c["kind"], "H": c["H"], "Hk": c["Hk"], "seqlens_q": c["seqlens_q"],
                   "seqlens_k": c["seqlens_k"], "HB": g["HB"], "M": g["M"], "NB": g["NB"], "words": int(L.total_words), "padded_words": int(L.padded_words()),
                   "expected_written": int(exp.sum()), "written": int(written.sum()), "spill_outside_layout": spill, "expected_but_unwritten": unwritten,
                   "skipped": skipped, "out_lse_eq_untapped": eq_u, "out_lse_eq_installed": eq_i}
            rec["ok"] = skipped == 0 and eq_u and eq_i and spill == 0 and unwritten == 0
            summary["ok"] &= rec["ok"]
            summary["cases"].append(rec)
            print(f"[dump] hd{D} {c['name']:14s} tile={g['kBM']}x{BN} HB={g['HB']} M={g['M']} NB={g['NB']} words={L.total_words} skipped={skipped} "
                  f"out/lse==untapped {eq_u} ==installed {eq_i} spill={spill} unwritten={unwritten} -> {'OK' if rec['ok'] else 'FAIL'}", flush=True)
            np.savez_compressed(os.path.join(a.out, f"hd{D}_{c['name']}.npz"), stream=words, src=src, D=D, BN=BN, kBM=g["kBM"], HB=g["HB"], M=g["M"], NB=g["NB"],
                                H=c["H"], Hk=c["Hk"], causal=c["causal"], seqlens_q=np.array(c["seqlens_q"]), seqlens_k=np.array(c["seqlens_k"]),
                                q=t["q"].view(torch.int16).cpu().numpy().view(np.uint16), k=t["k"].view(torch.int16).cpu().numpy().view(np.uint16),
                                v=t["v"].view(torch.int16).cpu().numpy().view(np.uint16), block_table=t["block_table"].cpu().numpy(),
                                out=out_t.view(torch.int16).cpu().numpy().view(np.uint16), lse=lse_t.float().cpu().numpy(), scale=np.float32(t["scale"]))
        # fail-closed negatives: a declaration that is not this launch's tile must be skipped and write nothing (prefill and decode tiles)
        for c in (cases_for(D, True)[0], [k for k in cases_for(D, False) if k["name"] == "decode_gqa"][0]):
            t = X.make_case(c, seed=7 * D)
            g = geometry(t)
            BN = g["BN"]
            for (bn_d, d_d, why) in ((BN, 64 if D != 64 else 128, "wrong head_dim"), (128 if BN != 128 else 192, D, "wrong kBlockN")):
                Lw = StreamLayout(g["HB"], g["M"], g["NB"], bn_d, d_d)
                buf = torch.full((max(Lw.padded_words(), StreamLayout(g["HB"], g["M"], g["NB"], BN, D).padded_words()),), FILL, dtype=torch.int32, device="cuda")
                sk0 = int(tap.verity_tap_skipped())
                tap.verity_set_tap_v9(3, buf, g["HB"], g["M"], g["NB"], bn_d, d_d, 9)
                out_w, lse_w = call(ns, t)
                torch.cuda.synchronize()
                tap.verity_set_tap_off()
                skipped = int(tap.verity_tap_skipped()) - sk0
                untouched = bool((buf == FILL).all().item())
                out_i, lse_i = call("_vllm_fa3_C", t)
                eq_i = bool(torch.equal(out_w.view(torch.int16), out_i.view(torch.int16)))
                neg = {"D": D, "case": c["name"], "declared": [bn_d, d_d], "kernel_tile": [BN, D], "why": why, "skipped": skipped, "buffer_untouched": untouched,
                       "out_eq_installed": eq_i, "ok": skipped == 1 and untouched and eq_i}
                summary["ok"] &= neg["ok"]
                summary["negatives"].append(neg)
                print(f"[dump] hd{D} negative {c['name']} {why}: declared ({bn_d},{d_d}) vs kernel ({BN},{D}): skipped={skipped} untouched={untouched} -> {'OK' if neg['ok'] else 'FAIL'}", flush=True)
        # caps too small (M does not cover the packed rows) -> skipped, untouched
        c = [k for k in cases_for(D, False) if k["name"] == "decode_gqa"][0]
        t = X.make_case(c, seed=11 * D)
        g = geometry(t)
        Ls = StreamLayout(g["HB"], g["M"] - 1, g["NB"], g["BN"], D)
        buf = torch.full((StreamLayout(g["HB"], g["M"], g["NB"], g["BN"], D).padded_words(),), FILL, dtype=torch.int32, device="cuda")
        sk0 = int(tap.verity_tap_skipped())
        tap.verity_set_tap_v9(3, buf, g["HB"], g["M"] - 1, g["NB"], g["BN"], D, 9)
        call(ns, t); torch.cuda.synchronize(); tap.verity_set_tap_off()
        skipped = int(tap.verity_tap_skipped()) - sk0
        untouched = bool((buf == FILL).all().item())
        neg = {"D": D, "case": c["name"], "why": "M cap below packed rows", "declared_M": g["M"] - 1, "packed_rows": g["M"], "skipped": skipped,
               "buffer_untouched": untouched, "ok": skipped == 1 and untouched}
        summary["ok"] &= neg["ok"]; summary["negatives"].append(neg)
        print(f"[dump] hd{D} negative M cap {g['M'] - 1} < packed rows {g['M']}: skipped={skipped} untouched={untouched} -> {'OK' if neg['ok'] else 'FAIL'}", flush=True)
    json.dump(summary, open(os.path.join(a.out, "dump_summary.json"), "w"), indent=1)
    print("DUMP", "OK" if summary["ok"] else "FAIL", os.path.join(a.out, "dump_summary.json"))
    sys.exit(0 if summary["ok"] else 1)


# ------------------------------------------------------------------------------------------------------------------ xcheck (oracle)
def fa3_oracle():
    """`fa2_attn_oracle` re-instantiated in the Hopper binding environment (verity_vllm.registry.hopper: HopperBF16WgmmaDot16 for the k16
    step, Fa3InvSum for the epilogue) -- the same code objects as the FA2 oracle, the FA3-bound functions: AttnBlockFA3 / AttentionHeadFA3."""
    from verity_vllm.registry import hopper
    spec = importlib.util.spec_from_file_location("fa2_attn_oracle", os.path.join(HERE, "fa2_attn_oracle.py"))
    O2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(O2)
    env = dict(vars(O2))
    env["P"] = hopper._BoundPrims()
    env["mma16"] = lambda acc, a, b: hopper.HopperBF16WgmmaDot16.evaluate(acc, *a, *b)
    for name in ("dot_bf16", "attn_block", "attention_head"):
        fn = getattr(O2, name)
        env[name] = types.FunctionType(fn.__code__, env, name, fn.__defaults__, fn.__closure__)
    O = types.SimpleNamespace(**env)
    O.hopper = hopper
    return O


def planes(z):
    from verity_capture.experimental.fa2_commit.hidden_stream import StreamLayout
    L = StreamLayout(int(z["HB"]), int(z["M"]), int(z["NB"]), int(z["BN"]), int(z["D"]))
    d = z["stream"]; o = L.offsets; HB, M, N, NB, D = L.HB, L.M, L.N, L.NB, L.D
    return L, dict(S=d[o["S"]:o["P"]].reshape(HB, M, N), P=d[o["P"]:o["RP"]].reshape(HB, M, N), RP=d[o["RP"]:o["ROW"]].reshape(HB, M, N // 2),
                   ROW=d[o["ROW"]:o["O"]].reshape(HB, M, NB, 8), O=d[o["O"]:o["FIN"]].reshape(HB, M, NB, D), FIN=d[o["FIN"]:o["total"]].reshape(HB, M, 4))


def xcheck_file(path: str, O, nrows: int, seed: int = 0, perturb: bool = False) -> dict:
    z = dict(np.load(path))
    L, pl = planes(z)
    D, BN, HB, M, NB, kBM = L.D, L.BN, L.HB, L.M, L.NB, int(z["kBM"])
    src = int(z["src"]); causal = bool(z["causal"])
    H, Hk = int(z["H"]), int(z["Hk"]); g = H // Hk
    seqlens_q, seqlens_k = [int(x) for x in z["seqlens_q"]], [int(x) for x in z["seqlens_k"]]
    cu_q = np.concatenate([[0], np.cumsum(seqlens_q)])
    q, k, v, bt, out = z["q"], z["k"], z["v"], z["block_table"], z["out"]
    if perturb:                                                 # negative control: five faults a vacuous comparator would miss
        pl["P"][0, 0, 0] ^= 1
        pl["O"][0, 0, 0, min(3, D - 1)] ^= 0x00010000
        r = pl["ROW"][0, 0, 0]; r[4], r[5] = r[5], r[4]
        pl["S"][0, 0, min(2, BN - 1)] = 0
        out = out.copy(); out.reshape(-1)[0] ^= 1
        path += "#negative-control"
    sl2 = O.scale_log2_bits(D)
    assert X.f32w(z["scale"])[0] == O.P.bits_of(np.float32(D ** -0.5))
    res = {"file": path, "D": D, "BN": BN, "kBM": kBM, "HB": HB, "M": M, "NB": NB, "g": g, "causal": causal, "rows_compared": 0,
           "rows_partial_domain": 0, "blocks_compared": 0, "leading_masked_blocks_checked": 0, "words_compared": 0, "mismatches": 0,
           "mismatch_examples": [], "per_key_words": {}, "per_key_bad": {}, "oracle_s": 0.0}
    rng = np.random.default_rng(seed)

    def cmp(tag, key, ref, got, zero_ok=False):
        for i, (x, y) in enumerate(zip(ref, got)):
            res["per_key_words"][key] = res["per_key_words"].get(key, 0) + 1
            res["words_compared"] += 1
            if not X.eq(int(x), int(y), zero_ok):
                res["mismatches"] += 1
                res["per_key_bad"][key] = res["per_key_bad"].get(key, 0) + 1
                if len(res["mismatch_examples"]) < 20:
                    res["mismatch_examples"].append(f"{tag}/{key}[{i}] oracle {int(x):08x} kernel {int(y):08x}")

    for seq in range(len(seqlens_q)):
        sq, sk = seqlens_q[seq], seqlens_k[seq]
        packed_rows = sq * g
        for kvh in range(Hk):
            slab = seq * Hk + kvh
            K = [[int(w) for w in k[bt[seq, j // PAGE], j % PAGE, kvh]] for j in range(sk)]
            V = [[int(w) for w in v[bt[seq, j // PAGE], j % PAGE, kvh]] for j in range(sk)]
            rows = X.pick_rows(packed_rows, nrows, rng)
            # make sure the head-in-group dimension is exercised too (rows of the same query row, different heads)
            if g > 1 and packed_rows > nrows and all(r % g == rows[0] % g for r in rows):
                rows[-1] = (rows[0] + 1) % packed_rows
            # the first packed row of every later M tile: the rows with the most leading all-masked blocks (causal n_block_max is the
            # tile's), i.e. the partial-domain path -- and the last row of the tile before it (n_block_max boundary)
            rows += [t for t in range(kBM, packed_rows, kBM)][:3] + [t - 1 for t in range(kBM, packed_rows, kBM)][:2]
            for r in sorted(set(rows)):
                q_row, hig = divmod(r, g)
                qh = kvh * g + hig
                qrow = [int(w) for w in q[cu_q[seq] + q_row, qh]]; out_row = out[cu_q[seq] + q_row, qh]
                tag = f"hd{D}/seq{seq}/slab{slab}/row{r}(q{q_row},h{qh})"
                nbm = n_block_max(sq, sk, r // kBM, kBM, BN, g, causal)
                visited = sorted(range(nbm), key=lambda nb: int(pl["ROW"][slab, r, nb, 2]))
                if src & 8:
                    assert visited == list(range(nbm - 1, -1, -1)), (tag, visited, [int(pl["ROW"][slab, r, nb, 2]) for nb in range(nbm)])
                    assert all(int(pl["ROW"][slab, r, nb, 2]) == FILL for nb in range(nbm, NB)), (tag, "block beyond n_block_max written")
                limit = min(sk, q_row + 1 + sk - sq) if causal else sk
                with_keys = [nb for nb in visited if nb * BN < limit]
                leading = [nb for nb in visited if nb * BN >= limit]
                for nb in leading:                                        # (a) all-masked leading blocks: the neutral state
                    res["leading_masked_blocks_checked"] += 1
                    cols = slice(nb * BN, (nb + 1) * BN)
                    if src & 1: cmp(tag, "masked.s", [NEG_INF] * BN, pl["S"][slab, r, cols])
                    if src & 2: cmp(tag, "masked.p", [0] * BN, pl["P"][slab, r, cols])
                    if src & 8: cmp(tag, "masked.m", [NEG_INF], [pl["ROW"][slab, r, nb, 0]]); cmp(tag, "masked.l", [0] * 4, pl["ROW"][slab, r, nb, 4:8])
                    if src & 16: cmp(tag, "masked.o", [0] * D, pl["O"][slab, r, nb])
                if leading and with_keys and src & 8:
                    cmp(tag, "masked.next_scale", [0], [pl["ROW"][slab, r, with_keys[0], 1]])
                if not with_keys:
                    continue
                if leading:
                    res["rows_partial_domain"] += 1
                t0 = time.time()
                head = O.attention_head(qrow, K, V, sk, D, sl2, BN, limit, len(with_keys))
                res["oracle_s"] += time.time() - t0
                res["rows_compared"] += 1
                assert [b["n_block"] for b in head["blocks"]] == with_keys
                for rec, nb in zip(head["blocks"], with_keys):
                    res["blocks_compared"] += 1
                    cols = slice(nb * BN, (nb + 1) * BN)
                    if src & 1: cmp(tag, "s_masked", rec["s_masked"], pl["S"][slab, r, cols])
                    if src & 2: cmp(tag, "p", rec["p"], pl["P"][slab, r, cols])
                    if src & 4:
                        w = pl["RP"][slab, r, nb * BN // 2:(nb + 1) * BN // 2]
                        got = [int(x) for pair in w for x in (pair & 0xFFFF, pair >> 16)]
                        cmp(tag, "pb", rec["pb"], got)
                    if src & 8:
                        cmp(tag, "m_new", [rec["m_new"]], [pl["ROW"][slab, r, nb, 0]], zero_ok=True)
                        if not rec["is_first"]:
                            cmp(tag, "scores_scale", [rec["scores_scale"]], [pl["ROW"][slab, r, nb, 1]])
                        elif not leading:
                            cmp(tag, "first_scale==1", [0x3F800000], [pl["ROW"][slab, r, nb, 1]])
                        cmp(tag, "l", rec["l"], pl["ROW"][slab, r, nb, 4:8])
                    if src & 16: cmp(tag, "o", rec["o"], pl["O"][slab, r, nb])
                if src & 32:
                    cmp(tag, "sum", [head["sum"]], [pl["FIN"][slab, r, 0]])
                    cmp(tag, "inv_sum", [head["inv_sum"]], [pl["FIN"][slab, r, 1]])
                    cmp(tag, "final_max", [head["blocks"][-1]["m_new"]], [pl["FIN"][slab, r, 2]], zero_ok=True)
                cmp(tag, "out", head["out"], [int(w) for w in out_row])
    res["verdict"] = "BIT-EXACT" if res["mismatches"] == 0 else f"{res['mismatches']} mismatching words"
    return res


def cmd_xcheck(a):
    O = fa3_oracle()
    files = sorted(glob.glob(os.path.join(a.dumps, "hd*.npz")))
    if a.only:
        files = [f for f in files if any(s in os.path.basename(f) for s in a.only.split(","))]
    assert files, a.dumps
    results, bad = [], 0
    for f in files:
        r = xcheck_file(f, O, a.rows)
        results.append(r); bad += r["mismatches"]
        print(f"[xcheck] {os.path.basename(f):28s} tile={r['kBM']}x{r['BN']} rows={r['rows_compared']} (partial-domain {r['rows_partial_domain']}) "
              f"blocks={r['blocks_compared']} leading-masked={r['leading_masked_blocks_checked']} words={r['words_compared']} oracle {r['oracle_s']:.0f}s -> {r['verdict']}", flush=True)
        for ex in r["mismatch_examples"][:8]:
            print("    ", ex)
        json.dump({"partial": True, "results": results}, open(a.out + ".partial", "w"), indent=1)
    neg = xcheck_file(files[0], O, min(a.rows, 2), perturb=True)
    results.append(neg)
    control_ok = neg["mismatches"] >= 5
    print(f"[xcheck] negative control ({os.path.basename(files[0])}): {neg['verdict']} (expected >= 5)")
    per_hd = {}
    for r in results[:-1]:
        d = per_hd.setdefault(str(r["D"]), {"files": 0, "rows": 0, "blocks": 0, "words": 0, "mismatches": 0, "tiles": []})
        d["files"] += 1; d["rows"] += r["rows_compared"]; d["blocks"] += r["blocks_compared"]; d["words"] += r["words_compared"]; d["mismatches"] += r["mismatches"]
        tile = f"{r['kBM']}x{r['BN']}"
        if tile not in d["tiles"]: d["tiles"].append(tile)
    summary = {"verdict": ("BIT-EXACT" if bad == 0 else f"{bad} MISMATCHES") + ("" if control_ok else " (NEGATIVE CONTROL FAILED)"), "per_hd": per_hd,
               "oracle": "fa2_attn_oracle re-instantiated over verity_vllm.registry.hopper (HopperBF16WgmmaDot16, Fa3InvSum) == AttnBlockFA3/AttentionHeadFA3",
               "negative_control_mismatches": neg["mismatches"], "rows_per_file": a.rows, "results": results}
    json.dump(summary, open(a.out, "w"), indent=1)
    if os.path.exists(a.out + ".partial"):
        os.remove(a.out + ".partial")
    print("XCHECK", summary["verdict"], json.dumps(per_hd), "->", a.out)
    sys.exit(1 if (bad or not control_ok) else 0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("--so", required=True); d.add_argument("--hdims", default="64,128"); d.add_argument("--out", required=True)
    d.add_argument("--quick", action="store_true")
    d.set_defaults(fn=cmd_dump)
    x = sub.add_parser("xcheck"); x.add_argument("--dumps", required=True); x.add_argument("--out", required=True); x.add_argument("--rows", type=int, default=4)
    x.add_argument("--only", default=None, help="comma list of substrings of dump file names")
    x.set_defaults(fn=cmd_xcheck)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
