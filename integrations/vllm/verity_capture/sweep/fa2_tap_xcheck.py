#!/usr/bin/env python
"""R13 lane fa2 (C-FA2-HDIM): exactness test of the head-dim-portable FA2 M1 hidden-value tap (`pod_fa2_tap.sh`, tap layout v8).

Two phases (the oracle is torch-free Python 3.12 `verity.ir`; the kernel needs the vLLM venv):

  dump    (venv-cu129, GPU)  for every head dim and case: random bf16 vLLM-style varlen paged inputs -> the TAPPED kernel with the M1
          stream armed for the declared tile (kBlockN, head_dim) -> checks (a) verity_tap_skipped == 0, (b) out / softmax_lse
          bit-identical to the installed `_vllm_fa2_C.varlen_fwd` and to the same .so untapped, (c) every word outside the layout's
          visited positions (rows >= seqlen_q, blocks >= n_block_max, padding) still carries the fill pattern (nothing spills; the
          buffer size / coordinate mapping is the declared one), then writes the planes + inputs to <out>/<case>.npz.
          Also the fail-closed negatives: a tile declaration that is not the launched kernel's (wrong kBlockN or head_dim) must be
          SKIPPED by the kernel (counter +1, buffer untouched); the R9 hd-64 .so must refuse hd 96/128 (legacy_so).
  xcheck  (venv-py312, CPU)  for sampled (sequence, head, row) triples: the Verity-semantics oracle `fa2_attn_oracle.attention_head`
          (`AttentionHead_v3{T, D, BN}` with the registry primitives: FTZ f32, MUFU.EX2/RCP tables, Ampere m16n8k16 model, RNE bf16) on
          the dump's own inputs; every tapped word (S masked, P, row max, rescale, lane sums, O after each block, sum, 1/sum) and the
          bf16 output row must be bit-equal (+0/-0 equal on the max chain; leading all-masked blocks of a causal tile checked as the
          neutral state, as R8 `p0_oracle_xcheck.py`).  Negative control: injected faults must be reported.

  PY=/workspace/venv-cu129/bin/python; PY312=/workspace/venv-py312/bin/python; cd /workspace/cp/veritor
  $PY    verity_capture/sweep/fa2_tap_xcheck.py dump   --so /workspace/cp/fa2/build/matReq/verity_fa2_matReq.so --hdims 64,96,128 --out /workspace/cp/fa2/xcheck
  $PY312 verity_capture/sweep/fa2_tap_xcheck.py xcheck --dumps /workspace/cp/fa2/xcheck --out /workspace/cp/fa2/xcheck/xcheck.json [--rows 6]

Cases per head dim (BI serving shapes: paged KV, page 16, num_splits 1, softmax scale D^-0.5, no alibi; --cases gemma adds softcap 50 /
  sliding window (4095, 0) / un-swapped GQA decode launches, R15 capacity):
  prefill (causal, GQA 4/2):  awkward lengths 1, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000, 1031 in mixed batches;
  chunked prefill:            seqlen_q < seqlen_k (17 new tokens on 300 keys; 64 on 1000) -- bottom-right causal mask;
  decode (seqlen_q 1):        GQA 8/2 (swapped kernel: rows = 4 query heads of a KV group, non-causal) and MHA 2/2 (M = 1) over KV
                              lengths 1, 17, 64, 65, 128, 129, 1000, 1031, 2047 in one batch.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (REPO, os.path.join(REPO, "vllm-poc")):
    if p not in sys.path:
        sys.path.insert(0, p)

PAGE = 16
NUM_SPLITS = 1
KBLOCK_M = 64
FILL = 0x7FC00001          # NaN pattern no kernel value can take (fp32 NaN with payload; the kernel writes finite / -inf / +0 words)
NEG_INF = 0xFF800000


def kblock_n(D: int) -> int:
    from verity_capture.experimental.commit_integ.hidden_source import fa2_kblock_n
    return fa2_kblock_n(D, NUM_SPLITS)


def cases_for(D: int, quick: bool) -> list[dict]:
    cs = [
        dict(name="prefill_short", kind="prefill", H=4, Hk=2, seqlens_q=[1, 17, 63, 64, 65], seqlens_k=[1, 17, 63, 64, 65], causal=True),
        dict(name="prefill_mid", kind="prefill", H=4, Hk=2, seqlens_q=[127, 128, 129, 255], seqlens_k=[127, 128, 129, 255], causal=True),
        dict(name="prefill_256", kind="prefill", H=4, Hk=2, seqlens_q=[256, 257, 5], seqlens_k=[256, 257, 5], causal=True),
        dict(name="prefill_long", kind="prefill", H=2, Hk=2, seqlens_q=[1000, 1031], seqlens_k=[1000, 1031], causal=True),
        dict(name="chunked", kind="prefill", H=4, Hk=2, seqlens_q=[17, 64, 1], seqlens_k=[300, 1000, 129], causal=True),
        dict(name="decode_gqa", kind="decode", H=8, Hk=2, seqlens_q=[1] * 9, seqlens_k=[1, 17, 64, 65, 128, 129, 1000, 1031, 2047], causal=True),
        dict(name="decode_mha", kind="decode", H=2, Hk=2, seqlens_q=[1] * 9, seqlens_k=[1, 17, 64, 65, 128, 129, 1000, 1031, 2047], causal=True),
    ]
    if quick:
        cs = [c for c in cs if c["name"] in ("prefill_short", "prefill_256", "decode_gqa")]
    for c in cs:
        c["D"] = D
        c.setdefault("softcap", 0.0); c.setdefault("window", (-1, -1))
    return cs


def gemma_cases_for(D: int) -> list[dict]:
    """[R15 capacity] Gemma-2-shaped launches (softcap 50 on every layer; sliding window (4095, 0) on every other layer; GQA 8/4 at 2B):
    the softcap (Is_softcap) and local (Is_local) kernel instantiations, and the UN-swapped GQA decode of the sliding layers
    (flash_api.cpp:646: no seqlenq_ngroups swap when a window is set) -- the geometry the host mirror must declare (fa2_swapped)."""
    W = (4095, 0)
    cs = [
        dict(name="g2_prefill_softcap", kind="prefill", H=8, Hk=4, seqlens_q=[1, 17, 64, 65, 255, 257], seqlens_k=[1, 17, 64, 65, 255, 257], causal=True, softcap=50.0, window=(-1, -1)),
        dict(name="g2_prefill_softcap_window", kind="prefill", H=8, Hk=4, seqlens_q=[1, 17, 64, 65, 255, 257], seqlens_k=[1, 17, 64, 65, 255, 257], causal=True, softcap=50.0, window=W),
        dict(name="g2_prefill_long_window", kind="prefill", H=4, Hk=2, seqlens_q=[1000, 1031], seqlens_k=[1000, 1031], causal=True, softcap=50.0, window=W),
        dict(name="g2_chunked_softcap_window", kind="prefill", H=8, Hk=4, seqlens_q=[17, 64, 1], seqlens_k=[300, 1000, 129], causal=True, softcap=50.0, window=W),
        dict(name="g2_decode_gqa_softcap", kind="decode", H=8, Hk=4, seqlens_q=[1] * 7, seqlens_k=[1, 17, 64, 65, 129, 1000, 1031], causal=True, softcap=50.0, window=(-1, -1)),      # swapped
        dict(name="g2_decode_gqa_softcap_window", kind="decode", H=8, Hk=4, seqlens_q=[1] * 7, seqlens_k=[1, 17, 64, 65, 129, 1000, 1031], causal=True, softcap=50.0, window=W),     # NOT swapped
        dict(name="g2_decode_mha_window", kind="decode", H=4, Hk=4, seqlens_q=[1] * 5, seqlens_k=[1, 64, 65, 1000, 1031], causal=True, softcap=0.0, window=W),                        # Phi-3 shape (MHA, window, no softcap)
    ]
    for c in cs:
        c["D"] = D
    return cs


# ------------------------------------------------------------------------------------------------------------------ dump (GPU)
def _installed_so() -> str:
    import vllm.vllm_flash_attn as m
    d = os.path.dirname(m.__file__)
    so = glob.glob(os.path.join(d, "_vllm_fa2_C*.so"))
    assert so, d
    return so[0]


def make_case(c: dict, seed: int, device="cuda"):
    import torch
    g = torch.Generator(device=device).manual_seed(seed)
    b, H, Hk, D = len(c["seqlens_q"]), c["H"], c["Hk"], c["D"]
    total_q, max_k = sum(c["seqlens_q"]), max(c["seqlens_k"])
    bps = (max_k + PAGE - 1) // PAGE
    num_blocks = b * bps + 8
    q = torch.randn(total_q, H, D, device=device, generator=g).to(torch.bfloat16)
    k = torch.randn(num_blocks, PAGE, Hk, D, device=device, generator=g).to(torch.bfloat16)
    v = torch.randn(num_blocks, PAGE, Hk, D, device=device, generator=g).to(torch.bfloat16)
    perm = torch.randperm(num_blocks, device=device, generator=g).to(torch.int32)
    block_table = perm[: b * bps].reshape(b, bps).contiguous()
    cu_q = torch.zeros(b + 1, dtype=torch.int32, device=device)
    cu_q[1:] = torch.cumsum(torch.tensor(c["seqlens_q"], device=device), 0)
    seqused_k = torch.tensor(c["seqlens_k"], dtype=torch.int32, device=device)
    wl, wr = c.get("window", (-1, -1))
    return dict(q=q, k=k, v=v, cu_seqlens_q=cu_q, seqused_k=seqused_k, block_table=block_table, max_seqlen_q=max(c["seqlens_q"]),
                max_seqlen_k=max_k, causal=c["causal"], scale=float(np.float32(D ** -0.5)), H=H, Hk=Hk, D=D,
                softcap=float(c.get("softcap", 0.0)), wl=int(wl), wr=int(wr))


def call(ns, t):
    import torch
    op = getattr(torch.ops, ns).varlen_fwd
    dummy_cu_k = torch.zeros_like(t["cu_seqlens_q"])
    return op(t["q"], t["k"], t["v"], None, t["cu_seqlens_q"], dummy_cu_k, t["seqused_k"], None, t["block_table"], None,
              t["max_seqlen_q"], t["max_seqlen_k"], 0.0, t["scale"], False, t["causal"], t.get("wl", -1), t.get("wr", -1),
              t.get("softcap", 0.0), False, NUM_SPLITS, None)


def geometry(t, force_swapped: bool | None = None):
    from verity_capture.experimental.commit_integ.hidden_source import fa2_swapped
    b = int(t["cu_seqlens_q"].numel()) - 1
    H, Hk, D = t["H"], t["Hk"], t["D"]
    swapped = fa2_swapped(t["max_seqlen_q"], H, Hk, D, t.get("wl", -1), t.get("wr", -1)) if force_swapped is None else force_swapped
    h = Hk if swapped else H
    M = (H // Hk) if swapped else int(t["max_seqlen_q"])
    BN = kblock_n(D)
    NB = (int(t["max_seqlen_k"]) + BN - 1) // BN
    return dict(b=b, h=h, M=M, NB=NB, BN=BN, D=D, HB=b * h, swapped=swapped)


def n_block_max(sq: int, sk: int, m_block: int, BN: int, causal: bool) -> int:
    nb = (sk + BN - 1) // BN
    if causal:
        nb = min(nb, ((m_block + 1) * KBLOCK_M + sk - sq + BN - 1) // BN)
    return max(nb, 0)


def expected_written_mask(layout, g, seqlens_q, seqlens_k, causal, src):
    """Boolean mask over the stream words the kernel is expected to write for this launch (everything else must keep the fill)."""
    from verity_capture.experimental.fa2_commit.hidden_stream import SRC_BITS
    HB, M, NB, N, D = layout.HB, layout.M, layout.NB, layout.N, layout.D
    o = layout.offsets
    w = np.zeros(layout.padded_words(), dtype=bool)
    S = w[o["S"]:o["P"]].reshape(HB, M, N); P = w[o["P"]:o["RP"]].reshape(HB, M, N); RP = w[o["RP"]:o["ROW"]].reshape(HB, M, N // 2)
    ROW = w[o["ROW"]:o["O"]].reshape(HB, M, NB, 8); O = w[o["O"]:o["FIN"]].reshape(HB, M, NB, D); FIN = w[o["FIN"]:o["total"]].reshape(HB, M, 4)
    for seq in range(g["b"]):
        sq = M if g["swapped"] else seqlens_q[seq]
        sk = seqlens_k[seq]
        for hh in range(g["h"]):
            slab = seq * g["h"] + hh
            for r in range(sq):
                nbm = n_block_max(sq, sk, r // KBLOCK_M, g["BN"], causal and not g["swapped"])
                if src & SRC_BITS["S"]: S[slab, r, :nbm * g["BN"]] = True
                if src & SRC_BITS["P"]: P[slab, r, :nbm * g["BN"]] = True
                if src & SRC_BITS["RP"]: RP[slab, r, :nbm * g["BN"] // 2] = True
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
    assert hasattr(tap, "verity_set_tap_v8"), f"{a.so}: not a v8 (head-dim-portable) tap build"
    mode, src, hd_mask, ver = int(tap.verity_tap_mode()), int(tap.verity_tap_src()), int(tap.verity_tap_hdims()), int(tap.verity_tap_layout_version())
    assert mode == 3 and ver == 8, (mode, ver)
    hdims = [int(h) for h in a.hdims.split(",")]
    summary = {"so": a.so, "ns": ns, "src": src, "hdims_mask": hd_mask, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "cases": [], "negatives": [], "ok": True}
    for D in hdims:
        assert hd_mask & (1 << (D // 32)), f"hd {D} not compiled ({hd_mask})"
        BN = kblock_n(D)
        cases = ([] if a.cases == "gemma" else cases_for(D, a.quick)) + (gemma_cases_for(D) if a.cases in ("gemma", "all") else [])
        for ci, c in enumerate(cases):
            t = make_case(c, seed=1000 * D + ci)
            g = geometry(t)
            L = StreamLayout(g["HB"], g["M"], g["NB"], BN, D)
            buf = torch.full((L.padded_words(),), FILL, dtype=torch.int32, device="cuda")
            sk0 = int(tap.verity_tap_skipped())
            tap.verity_set_tap_v8(3, buf, g["HB"], g["M"], g["NB"], BN, D, 7)
            out_t, lse_t = call(ns, t)
            torch.cuda.synchronize()
            tap.verity_set_tap_v8(0, None, 0, 0, 0, 0, 0, 0)
            skipped = int(tap.verity_tap_skipped()) - sk0
            out_u, lse_u = call(ns, t)                        # same .so, tap off
            out_i, lse_i = call("_vllm_fa2_C", t)             # the installed vLLM kernel
            torch.cuda.synchronize()
            eq_u = bool(torch.equal(out_t.view(torch.int16), out_u.view(torch.int16)) and torch.equal(lse_t.view(torch.int32), lse_u.view(torch.int32)))
            eq_i = bool(torch.equal(out_t.view(torch.int16), out_i.view(torch.int16)) and torch.equal(lse_t.view(torch.int32), lse_i.view(torch.int32)))
            words = buf.cpu().numpy().view(np.uint32)
            exp = expected_written_mask(L, g, c["seqlens_q"], c["seqlens_k"], c["causal"], src)
            written = words != FILL
            spill = int(np.count_nonzero(written & ~exp))
            unwritten = int(np.count_nonzero(exp & ~written))
            rec = {"case": c["name"], "D": D, "BN": BN, "kind": c["kind"], "H": c["H"], "Hk": c["Hk"], "seqlens_q": c["seqlens_q"], "seqlens_k": c["seqlens_k"],
                   "softcap": t["softcap"], "window": [t["wl"], t["wr"]],
                   "HB": g["HB"], "M": g["M"], "NB": g["NB"], "swapped": g["swapped"], "words": int(L.total_words), "padded_words": int(L.padded_words()),
                   "expected_written": int(exp.sum()), "written": int(written.sum()), "spill_outside_layout": spill, "expected_but_unwritten": unwritten,
                   "skipped": skipped, "out_lse_eq_untapped": eq_u, "out_lse_eq_installed": eq_i}
            rec["ok"] = skipped == 0 and eq_u and eq_i and spill == 0 and unwritten == 0
            summary["ok"] &= rec["ok"]
            summary["cases"].append(rec)
            print(f"[dump] hd{D} {c['name']:28s} BN={BN} HB={g['HB']} M={g['M']} NB={g['NB']} softcap={t['softcap']} window={t['wl']},{t['wr']} "
                  f"swapped={g['swapped']} words={L.total_words} skipped={skipped} "
                  f"out/lse==untapped {eq_u} ==installed {eq_i} spill={spill} unwritten={unwritten} -> {'OK' if rec['ok'] else 'FAIL'}", flush=True)
            np.savez_compressed(os.path.join(a.out, f"hd{D}_{c['name']}.npz"), stream=words, src=src, D=D, BN=BN, HB=g["HB"], M=g["M"], NB=g["NB"],
                                H=c["H"], Hk=c["Hk"], swapped=g["swapped"], causal=c["causal"], seqlens_q=np.array(c["seqlens_q"]), seqlens_k=np.array(c["seqlens_k"]),
                                softcap=np.float32(t["softcap"]), wl=np.int64(t["wl"]), wr=np.int64(t["wr"]),
                                q=t["q"].view(torch.int16).cpu().numpy().view(np.uint16), k=t["k"].view(torch.int16).cpu().numpy().view(np.uint16),
                                v=t["v"].view(torch.int16).cpu().numpy().view(np.uint16), block_table=t["block_table"].cpu().numpy(),
                                out=out_t.view(torch.int16).cpu().numpy().view(np.uint16), lse=lse_t.float().cpu().numpy(), scale=np.float32(t["scale"]))
        # fail-closed negatives for this head dim: a declaration that is not this kernel's tile must be skipped and write nothing
        c = cases_for(D, True)[0]
        t = make_case(c, seed=7 * D)
        g = geometry(t)
        for (bn_d, d_d, why) in ((BN, 64 if D != 64 else 128, "wrong head_dim"), (64 if BN == 128 else 128, D, "wrong kBlockN")):
            Lw = StreamLayout(g["HB"], g["M"], g["NB"], bn_d, d_d)
            buf = torch.full((max(Lw.padded_words(), StreamLayout(g["HB"], g["M"], g["NB"], BN, D).padded_words()),), FILL, dtype=torch.int32, device="cuda")
            sk0 = int(tap.verity_tap_skipped())
            tap.verity_set_tap_v8(3, buf, g["HB"], g["M"], g["NB"], bn_d, d_d, 9)
            out_w, lse_w = call(ns, t)
            torch.cuda.synchronize()
            tap.verity_set_tap_v8(0, None, 0, 0, 0, 0, 0, 0)
            skipped = int(tap.verity_tap_skipped()) - sk0
            untouched = bool((buf == FILL).all().item())
            out_i, lse_i = call("_vllm_fa2_C", t)
            eq_i = bool(torch.equal(out_w.view(torch.int16), out_i.view(torch.int16)))
            neg = {"D": D, "declared": [bn_d, d_d], "kernel_tile": [BN, D], "why": why, "skipped": skipped, "buffer_untouched": untouched, "out_eq_installed": eq_i,
                   "ok": skipped == 1 and untouched and eq_i}
            summary["ok"] &= neg["ok"]
            summary["negatives"].append(neg)
            print(f"[dump] hd{D} negative {why}: declared ({bn_d},{d_d}) vs kernel ({BN},{D}): skipped={skipped} untouched={untouched} -> {'OK' if neg['ok'] else 'FAIL'}", flush=True)
        if a.cases in ("gemma", "all"):
            # [R15 capacity] the R13 mirror's geometry for a WINDOWED GQA decode (declared swapped: b*Hk slabs x H/Hk rows) is not the
            # launched kernel's (un-swapped: b*H slabs x 1 row, flash_api.cpp:646) -> the arm check (b*h <= vt.H) must SKIP it and write
            # nothing (X-03 fails the run by name); never a mis-addressed layout of the right size.
            c = [x for x in gemma_cases_for(D) if x["name"] == "g2_decode_gqa_softcap_window"][0]
            t = make_case(c, seed=11 * D)
            g_ok, g_bad = geometry(t), geometry(t, force_swapped=True)
            assert not g_ok["swapped"] and g_bad["swapped"] and g_ok["HB"] * g_ok["M"] == g_bad["HB"] * g_bad["M"]
            Lb = StreamLayout(g_bad["HB"], g_bad["M"], g_bad["NB"], BN, D)
            buf = torch.full((Lb.padded_words(),), FILL, dtype=torch.int32, device="cuda")
            sk0 = int(tap.verity_tap_skipped())
            tap.verity_set_tap_v8(3, buf, g_bad["HB"], g_bad["M"], g_bad["NB"], BN, D, 13)
            out_w, lse_w = call(ns, t)
            torch.cuda.synchronize()
            tap.verity_set_tap_v8(0, None, 0, 0, 0, 0, 0, 0)
            skipped = int(tap.verity_tap_skipped()) - sk0
            untouched = bool((buf == FILL).all().item())
            out_i, lse_i = call("_vllm_fa2_C", t)
            eq_i = bool(torch.equal(out_w.view(torch.int16), out_i.view(torch.int16)))
            neg = {"D": D, "declared_geometry": [g_bad["HB"], g_bad["M"], g_bad["NB"]], "kernel_geometry": [g_ok["HB"], g_ok["M"], g_ok["NB"]],
                   "why": "R13 mirror: swapped GQA decode declared for a sliding-window (un-swapped) launch", "skipped": skipped, "buffer_untouched": untouched,
                   "out_eq_installed": eq_i, "ok": skipped == 1 and untouched and eq_i}
            summary["ok"] &= neg["ok"]
            summary["negatives"].append(neg)
            print(f"[dump] hd{D} negative windowed-decode geometry: declared HBxM {g_bad['HB']}x{g_bad['M']} vs kernel {g_ok['HB']}x{g_ok['M']}: "
                  f"skipped={skipped} untouched={untouched} -> {'OK' if neg['ok'] else 'FAIL'}", flush=True)
    if a.legacy_so:   # the R9 hd-64-only build must refuse hd 96/128 at the dispatch (stub) -- C-FA2-HDIM fails closed, never a wrong kernel
        # (a child process: both builds register the same torch.ops namespace, so they cannot be loaded into one process)
        import subprocess
        for D in [h for h in hdims if h != 64]:
            code = (f"import sys; sys.path[:0] = {[REPO, os.path.join(REPO, 'vllm-poc')]!r}\n"
                    f"import torch, fa2_tap_xcheck as X\ntorch.ops.load_library({a.legacy_so!r})\n"
                    f"t = X.make_case(X.cases_for({D}, True)[0], seed=3)\n"
                    f"try:\n    X.call({os.path.basename(a.legacy_so)[:-3]!r}, t); torch.cuda.synchronize(); print('NOERROR')\n"
                    f"except Exception as e:\n    print('RAISED', str(e).splitlines()[0][:200])\n")
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE)
            line = ([ln for ln in r.stdout.splitlines() if ln.startswith(("RAISED", "NOERROR"))] or [f"child failed: {r.stderr[-300:]}"])[-1]
            ok = line.startswith("RAISED") and "not compiled" in line
            summary["negatives"].append({"D": D, "legacy_so": a.legacy_so, "raised": line, "ok": ok})
            summary["ok"] &= ok
            print(f"[dump] legacy .so hd{D}: {line} -> {'OK' if ok else 'FAIL'}", flush=True)
    json.dump(summary, open(os.path.join(a.out, "dump_summary.json"), "w"), indent=1)
    print("DUMP", "OK" if summary["ok"] else "FAIL", os.path.join(a.out, "dump_summary.json"))
    sys.exit(0 if summary["ok"] else 1)


# ------------------------------------------------------------------------------------------------------------------ xcheck (oracle)
def f32w(a) -> list[int]:
    return [int(x) for x in np.ascontiguousarray(np.asarray(a, np.float32)).view(np.uint32).reshape(-1)]


def eq(a: int, b: int, zero_ok=False) -> bool:
    return a == b or (zero_ok and {a, b} <= {0, 0x80000000})


def planes(z):
    from verity_capture.experimental.fa2_commit.hidden_stream import StreamLayout
    L = StreamLayout(int(z["HB"]), int(z["M"]), int(z["NB"]), int(z["BN"]), int(z["D"]))
    d = z["stream"]; o = L.offsets; HB, M, N, NB, D = L.HB, L.M, L.N, L.NB, L.D
    return L, dict(S=d[o["S"]:o["P"]].reshape(HB, M, N), P=d[o["P"]:o["RP"]].reshape(HB, M, N), RP=d[o["RP"]:o["ROW"]].reshape(HB, M, N // 2),
                   ROW=d[o["ROW"]:o["O"]].reshape(HB, M, NB, 8), O=d[o["O"]:o["FIN"]].reshape(HB, M, NB, D), FIN=d[o["FIN"]:o["total"]].reshape(HB, M, 4))


def pick_rows(sq: int, nrows: int, rng) -> list[int]:
    if sq <= nrows:
        return list(range(sq))
    cand = {0, sq - 1, 63 if sq > 63 else sq - 1, 64 if sq > 64 else 0, 65 if sq > 65 else 0}
    while len(cand) < nrows:
        cand.add(int(rng.integers(0, sq)))
    return sorted(cand)[:nrows]


def xcheck_file(path: str, O, nrows: int, seed: int = 0, perturb: bool = False) -> dict:
    z = dict(np.load(path))
    L, pl = planes(z)
    D, BN, HB, M, NB = L.D, L.BN, L.HB, L.M, L.NB
    src = int(z["src"]); swapped = bool(z["swapped"]); causal_k = bool(z["causal"]) and not swapped
    H, Hk = int(z["H"]), int(z["Hk"]); ngroups = H // Hk
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
    softcap = float(z["softcap"]) if "softcap" in z else 0.0                 # [R15 capacity] Gemma-2 softcap / sliding window
    wl, wr = (int(z["wl"]), int(z["wr"])) if "wl" in z else (-1, -1)
    if softcap > 0:
        softcap_bits, sl2 = O.softcap_params(D, softcap)
    else:
        softcap_bits, sl2 = 0, O.scale_log2_bits(D)
    # the kernel's visible-column upper bound: causal (wr == 0 via is_causal) or local with window_right == 0 -> row + 1 + sk - sq
    limit_causal = causal_k or (wr == 0 and not swapped)
    assert f32w(z["scale"])[0] == O.P.bits_of(np.float32(D ** -0.5))
    res = {"file": path, "D": D, "BN": BN, "HB": HB, "M": M, "NB": NB, "swapped": swapped, "causal_kernel": causal_k, "softcap": softcap,
           "window": [wl, wr], "rows_compared": 0,
           "rows_partial_domain": 0, "blocks_compared": 0, "leading_masked_blocks_checked": 0, "words_compared": 0, "mismatches": 0,
           "mismatch_examples": [], "per_key_words": {}, "per_key_bad": {}, "oracle_s": 0.0}
    rng = np.random.default_rng(seed)

    def cmp(tag, key, ref, got, zero_ok=False):
        for i, (x, y) in enumerate(zip(ref, got)):
            res["per_key_words"][key] = res["per_key_words"].get(key, 0) + 1
            res["words_compared"] += 1
            if not eq(int(x), int(y), zero_ok):
                res["mismatches"] += 1
                res["per_key_bad"][key] = res["per_key_bad"].get(key, 0) + 1
                if len(res["mismatch_examples"]) < 20:
                    res["mismatch_examples"].append(f"{tag}/{key}[{i}] oracle {int(x):08x} kernel {int(y):08x}")

    for seq in range(len(seqlens_q)):
        sq, sk = seqlens_q[seq], seqlens_k[seq]
        rows_k = M if swapped else sq
        for hh in range(Hk if swapped else H):
            slab = seq * (Hk if swapped else H) + hh
            kvh = hh if swapped else hh // ngroups
            K = [[int(w) for w in k[bt[seq, j // PAGE], j % PAGE, kvh]] for j in range(sk)]
            V = [[int(w) for w in v[bt[seq, j // PAGE], j % PAGE, kvh]] for j in range(sk)]
            for r in pick_rows(rows_k, nrows, rng):
                if swapped:
                    qh = hh * ngroups + r; qrow = [int(w) for w in q[cu_q[seq], qh]]; out_row = out[cu_q[seq], qh]
                else:
                    qh = hh; qrow = [int(w) for w in q[cu_q[seq] + r, qh]]; out_row = out[cu_q[seq] + r, qh]
                tag = f"hd{D}/seq{seq}/slab{slab}/row{r}"
                nbm = n_block_max(rows_k, sk, r // KBLOCK_M, BN, limit_causal)
                # visit order from the ROW step field (int bits): the kernel visits n_block_max-1 .. 0; blocks >= n_block_max are never written
                visited = sorted(range(nbm), key=lambda nb: int(pl["ROW"][slab, r, nb, 2]))
                if src & 8:
                    assert visited == list(range(nbm - 1, -1, -1)), (tag, visited, [int(pl["ROW"][slab, r, nb, 2]) for nb in range(nbm)])
                    assert all(int(pl["ROW"][slab, r, nb, 2]) == FILL for nb in range(nbm, NB)), (tag, "block beyond n_block_max written")
                limit = min(sk, r + 1 + sk - rows_k) if limit_causal else sk
                left = max(0, r + sk - rows_k - wl) if (wl >= 0 and not swapped) else 0      # FA2 Is_local lower bound; 0 for every panel row
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
                head = O.attention_head(qrow, K, V, sk, D, sl2, BN, limit, len(with_keys), softcap_bits, left)
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
    import importlib.util
    spec = importlib.util.spec_from_file_location("fa2_attn_oracle", os.path.join(HERE, "fa2_attn_oracle.py"))
    O = importlib.util.module_from_spec(spec); spec.loader.exec_module(O)
    files = sorted(glob.glob(os.path.join(a.dumps, "hd*.npz")))
    if a.only:
        files = [f for f in files if any(s in os.path.basename(f) for s in a.only.split(","))]
    assert files, a.dumps
    results, bad = [], 0
    for f in files:
        r = xcheck_file(f, O, a.rows)
        results.append(r); bad += r["mismatches"]
        print(f"[xcheck] {os.path.basename(f):28s} rows={r['rows_compared']} (partial-domain {r['rows_partial_domain']}) blocks={r['blocks_compared']} "
              f"leading-masked={r['leading_masked_blocks_checked']} words={r['words_compared']} oracle {r['oracle_s']:.0f}s -> {r['verdict']}", flush=True)
        for ex in r["mismatch_examples"][:8]:
            print("    ", ex)
    neg = xcheck_file(files[0], O, min(a.rows, 2), perturb=True)
    results.append(neg)
    control_ok = neg["mismatches"] >= 5
    print(f"[xcheck] negative control ({os.path.basename(files[0])}): {neg['verdict']} (expected >= 5)")
    per_hd = {}
    for r in results[:-1]:
        d = per_hd.setdefault(str(r["D"]), {"files": 0, "rows": 0, "blocks": 0, "words": 0, "mismatches": 0})
        d["files"] += 1; d["rows"] += r["rows_compared"]; d["blocks"] += r["blocks_compared"]; d["words"] += r["words_compared"]; d["mismatches"] += r["mismatches"]
    summary = {"verdict": ("BIT-EXACT" if bad == 0 else f"{bad} MISMATCHES") + ("" if control_ok else " (NEGATIVE CONTROL FAILED)"), "per_hd": per_hd,
               "negative_control_mismatches": neg["mismatches"], "rows_per_file": a.rows, "results": results}
    json.dump(summary, open(a.out, "w"), indent=1)
    print("XCHECK", summary["verdict"], json.dumps(per_hd), "->", a.out)
    sys.exit(1 if (bad or not control_ok) else 0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("--so", required=True); d.add_argument("--hdims", default="64,96,128"); d.add_argument("--out", required=True)
    d.add_argument("--quick", action="store_true"); d.add_argument("--legacy-so", default=None, help="R9 hd-64-only build: must refuse hd 96/128")
    d.add_argument("--cases", default="std", choices=["std", "gemma", "all"], help="[R15] gemma: softcap 50 / sliding window (4095,0) / un-swapped GQA decode launches (+ the geometry negative)")
    d.set_defaults(fn=cmd_dump)
    x = sub.add_parser("xcheck"); x.add_argument("--dumps", required=True); x.add_argument("--out", required=True); x.add_argument("--rows", type=int, default=6)
    x.add_argument("--only", default=None, help="comma list of substrings of dump file names")
    x.set_defaults(fn=cmd_xcheck)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
