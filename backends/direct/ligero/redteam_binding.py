"""Statement-binding fixtures for this prover (the red team's checklist, ``redteam_vu.md`` F1 / F2 / F19, applied to
the direct Ligero GPU prover).  Three cases, each run through the real prover and verifier:

1. ``zero_witness_as_real_words``  -- a proof of two trivial VUs (``a = b = 0``, ``y16 = 0``) presented as a proof of
   the first two real ``vu-k1536`` VUs (their operands and claimed words).  Must be rejected.
2. ``small_batch_vs_large_verifier`` -- a 2-VU proof (``l = 256``) presented to the verifier configured for
   ``l = 512`` (4 VUs' worth of columns).  Must be rejected -- and it must be rejected on the verifier's OWN sizes
   (``Config`` / ``System``), never on a size read from the proof.
3. ``adaptive_claim`` -- the forgery the pre-fix transcript allowed.  Before the fix the challenges were
   ``H(root)`` and ``H(root, w, h, q)``: the statement (operands, claimed words, layout, sizes) was never absorbed.
   The verifier's chain-test scalar check ``l (q_0 + q_l) = r^T b`` then binds the claimed words only through the
   six BabyBear sums ``sum_j r_end[d, j] y16_j``; a prover who has already produced an honest proof knows
   ``r_end`` (a function of ``root`` alone) and can replace the claimed words by ANY ``y16'`` with
   ``sum_j r_end[d, j] (y16'_j - y16_j) = 0 mod p`` for all ``d``.  With 21 VUs per proof (``l = 2048``) that kernel
   lattice has determinant ``<= p^6 ~ 2^186`` in dimension 21, so LLL returns a nonzero ``delta`` with entries of a
   few hundred: 21 real VUs, proved honestly, are accepted as a proof of 21 WRONG output words.  After the fix
   (``statement_digest`` absorbed into every challenge) the same proof against the forged words is rejected.

    python -m backends.direct.ligero.redteam_binding --device cuda --out results/redteam_binding.json
"""
from __future__ import annotations

import argparse
import json
import sys as _sys
import time
from pathlib import Path

import torch
from verity_numerical.bench import instances

from . import protocol
from .chain import Layout
from .compile import P
from .vu import STEPS, ChainRunner, _vu_data


def _kernel_short_vectors(A: list[list[int]], p: int) -> list[list[int]]:
    """Short nonzero integer vectors delta with A delta = 0 (mod p) for every row of A (LLL on the kernel lattice
    embedded with a large scale on the congruence block), shortest first."""
    from sympy import ZZ
    from sympy.polys.matrices import DomainMatrix

    D, nvar = len(A), len(A[0])
    C = 2 ** 40
    rows = []
    for i in range(nvar):
        rows.append([1 if j == i else 0 for j in range(nvar)] + [C * A[d][i] for d in range(D)])
    for d in range(D):
        rows.append([0] * nvar + [C * p if e == d else 0 for e in range(D)])
    M = DomainMatrix([[ZZ(v) for v in r] for r in rows], (nvar + D, nvar + D), ZZ)
    R = M.lll().to_Matrix()
    out = []
    for r in range(R.rows):
        v = [int(R[r, j]) for j in range(nvar + D)]
        if any(v[nvar:]) or not any(v[:nvar]):
            continue
        out.append((sum(x * x for x in v[:nvar]), v[:nvar]))
    return [v for _, v in sorted(out)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--root", default=str(instances.DEFAULT_ROOT))
    ap.add_argument("--target", type=float, default=-128.0)
    ap.add_argument("--adaptive-vus", type=int, default=21)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    R = ChainRunner(args.device, args.target)
    vu = instances.load("vu-k1536", Path(args.root))
    real2 = [_vu_data(vu, 0), _vu_data(vu, 1)]
    results = {}
    t0 = time.time()

    # 1. zero witness claiming the real words
    zero2 = [([0] * (16 * STEPS), [0] * (16 * STEPS), [0] * STEPS, 0)] * 2
    proof0, _, lay0 = R.prove_vus(zero2, check=False)
    _, pubs_real, lay_real = R.prove_vus(real2, check=False)
    ok, why = R.verify_vus(proof0, pubs_real, lay_real)
    results["zero_witness_as_real_words"] = {"accepted": ok, "why": why, "l": lay0.l}
    print(f"1. zero-witness proof as the real 2-VU statement: {'ACCEPTED -- FORGERY' if ok else 'rejected'} ({why})")

    # 2. small batch against a larger verifier (the verifier's own layout / Config)
    proof_small, (a, b, y16), lay_small = R.prove_vus(real2, check=False)
    lay_big = Layout(512, 2, STEPS)
    pad = lay_big.l - lay_small.l
    a_big, b_big = list(a) + [[0] * 16] * pad, list(b) + [[0] * 16] * pad
    y_big = torch.cat([y16, torch.zeros(pad, dtype=torch.int64)])
    ok, why = protocol.verify(R.p, R.sys, R.tb, R.cfg(lay_big.l), None, a_big, b_big, None, proof_small, R.device,
                              chain=(lay_big, y_big), coins=proof_small.coins)
    results["small_batch_vs_large_verifier"] = {"accepted": ok, "why": why, "l_proof": lay_small.l, "l_verifier": lay_big.l}
    print(f"2. l={lay_small.l} proof against the l={lay_big.l} verifier: {'ACCEPTED -- FORGERY' if ok else 'rejected'} ({why})")

    # 3. adaptive claim: honest proof, then words chosen from the challenge (pre-fix), rejected post-fix
    n_vus = args.adaptive_vus
    vus = [_vu_data(vu, i) for i in range(n_vus)]
    protocol._BIND_STATEMENT = False
    try:
        proof, (a, b, y16), lay = R.prove_vus(vus, check=False)
        ok_h, _why_h = R.verify_vus(proof, (a, b, y16), lay)
        stmt = b""
        _, _, _, rc = protocol._challenge1(stmt, proof.root, R.cfg(lay.l).D, R.sys.m, R.tb.L, R.tb.Q, R.device, lay.l,
                                           proof.coin_commit, proof.coins.r1)
        _, _, end = lay.masks(R.device)
        ends = torch.nonzero(end).flatten().tolist()
        A = [[int(rc[d, 6, j]) for j in ends] for d in range(rc.shape[0])]
        t_l = time.time()
        cands = _kernel_short_vectors(A, P)
        lll_s = time.time() - t_l
        forged = None
        delta, y_f, in_range = None, None, False
        for cand in cands:                    # the shortest kernel vector that keeps every forged word a 16-bit word
            y_try = y16.clone()
            for j, dlt in zip(ends, cand):
                y_try[j] = y16[j] + dlt
            if bool(((y_try >= 0) & (y_try < 2 ** 16)).all()):
                delta, y_f, in_range = cand, y_try, True
                break
        if delta is None and cands:
            delta, y_f = cands[0], y16.clone()
        if delta is not None:
            same_sums = all(sum(A[d][k] * delta[k] for k in range(len(ends))) % P == 0 for d in range(len(A)))
            ok_f, why_f = R.verify_vus(proof, (a, b, y_f), lay) if in_range else (False, "forged words out of range")
            forged = {"delta": delta, "max_abs_delta": max(abs(x) for x in delta), "in_range": in_range,
                      "kernel_ok": same_sums, "accepted_prefix_transcript": ok_f, "why": why_f, "lll_seconds": lll_s}
            print(f"3a. pre-fix transcript (challenges from root only): honest {ok_h}; {n_vus} forged words "
                  f"(|delta| <= {forged['max_abs_delta']}): {'ACCEPTED -- FORGERY' if ok_f else 'rejected'} ({why_f}); LLL {lll_s:.1f}s")
    finally:
        protocol._BIND_STATEMENT = True
    # the same attack against the bound transcript: the honest proof is re-made (challenges now depend on the words)
    proof_b, (a, b, y16), lay = R.prove_vus(vus, check=False)
    ok_hb, why_hb = R.verify_vus(proof_b, (a, b, y16), lay)
    ok_fb, why_fb = (None, "no delta")
    if forged is not None and forged["in_range"]:
        ok_fb, why_fb = R.verify_vus(proof_b, (a, b, y_f), lay)
    print(f"3b. bound transcript: honest {ok_hb} ({why_hb}); the forged words: {'ACCEPTED -- FORGERY' if ok_fb else 'rejected'} ({why_fb})")
    results["adaptive_claim"] = {"n_vus": n_vus, "l": lay.l, "prefix_transcript": {"honest_accepted": ok_h, **(forged or {})},
                                 "bound_transcript": {"honest_accepted": ok_hb, "forged_accepted": ok_fb, "why": why_fb}}
    results["ok"] = (not results["zero_witness_as_real_words"]["accepted"] and not results["small_batch_vs_large_verifier"]["accepted"]
                     and ok_hb and ok_fb is False)
    results["seconds"] = time.time() - t0
    results["note"] = ("pre-fix: challenges H(root), H(root,w,h,q) -- statement unbound; the adaptive_claim forgery is what it "
                       "allowed.  fix: statement_digest(field, hash, system id, l/n/D/t, layout, a, b, y16 | c, y) absorbed "
                       "into every challenge; sizes from the verifier's Config/System; hash name checked.")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    print("OK" if results["ok"] else "FAIL")
    return 0 if results["ok"] else 2


if __name__ == "__main__":
    _sys.exit(main())
