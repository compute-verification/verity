"""Red-team driver (lane red-team-zk): every experiment of ``reports/redteam_zk.md`` against the real prover.

    python -m backends.direct.ligero.redteam.run --device cuda --vus 170 --n-transcripts 48 --out results/redteam_zk_lane.json
    python -m backends.direct.ligero.redteam.run --device cpu --vus 2 --n-transcripts 6          # laptop smoke (< 1 min)

Sections of the report: ``q_leak`` (class 1/5: accumulators from q), ``mask_ranks`` (class 1), ``simulator``
(class 2: honest / adaptive / aborting V*, indistinguishability, q distinguishers), ``two_witness`` (class 1
empirical), ``accounting`` (class 6).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from verity_numerical.bench import instances
from verity_numerical.security import accounting as acc

from .. import protocol
from ..protocol import COIN_HIDING_TERM, CoinOpeningError, Coins, soundness
from ..vu import STEP, STEPS, ChainRunner, _vu_data
from . import mask_rank, q_leak, simulator


def _trivial(n):
    return [([0] * (STEP * STEPS), [0] * (STEP * STEPS), [0] * STEPS, 0)] * n


def _load_real(root: Path, n: int) -> list[tuple]:
    """The recorded ``vu-k1536`` VUs; on a machine without the built arrays (laptop smoke), the ``vu-k1536-neg``
    operands with the honest chain recomputed by the checker (only those with a finite honest chain)."""
    try:
        vu = instances.load("vu-k1536", root)
        return [_vu_data(vu, i % vu.n) for i in range(n)]
    except FileNotFoundError:
        from verity_numerical.checker.params import REAL
        from verity_numerical.checker.step import step_output
        neg = instances.load("vu-k1536-neg", root)
        out = []
        for i in range(neg.n):
            x, w, _ = neg.instance(i)
            try:
                accs, c = [], 0
                for s in range(STEPS):
                    c = step_output(REAL, c, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
                    accs.append(c)
            except Exception:  # noqa: BLE001, S112 -- off-domain operands: no honest chain for this instance
                continue
            # claimed word = the honest epilogue of the last accumulator (RN-even cast to BF16: Epilogue.gen)
            from verity_numerical.checker.gadgets import Epilogue
            e = Epilogue.gen(REAL, c)
            out.append((x, w, accs, int(e["hi"] + e["carry"])))
            if len(out) == n:
                break
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--root", default=str(instances.DEFAULT_ROOT))
    ap.add_argument("--vus", type=int, default=170, help="VUs per proof (170 -> l = 16384, the measured layout)")
    ap.add_argument("--n-proofs", type=int, default=25, help="sub-batch count the configuration is union-bounded over")
    ap.add_argument("--n-transcripts", type=int, default=48)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    dev = args.device
    rng = np.random.default_rng(7)
    report: dict = {"args": vars(args), "device": dev}
    real = _load_real(Path(args.root), args.vus)
    triv = _trivial(args.vus)
    R = ChainRunner(dev, -128.0, n_proofs=args.n_proofs, zk=True)
    lay = R.layout(len(real))
    cfg = R.cfg(lay.l)
    m = R.sys.m
    report["config"] = {"l": cfg.l, "n": cfg.n, "k": cfg.k, "t": cfg.t, "t_pad": cfg.t_pad, "D": cfg.D, "m": m,
                        "committed_rows": int(protocol.n_rows(cfg, m, True)), "vus_per_proof": lay.n_vus}
    print("config", report["config"])

    # ---- q leak: recover every accumulator from the public transcript (honest verifier) ----------------------------
    t0 = time.perf_counter()
    pr, (a, b, y16), _ = R.prove_vus(real, check=False)
    ok, why = R.verify_vus(pr, (a, b, y16), lay)
    assert ok, why
    rec = q_leak.recover_accumulators(pr, R.sys, R.tb, cfg, lay, y16, a, b, dev)
    cmp_ = q_leak.compare_with_recorded(rec, real, STEPS)
    cmp_["seconds"] = time.perf_counter() - t0
    # the same on the trivial witness (whose accumulators are all zero) and the witness-free statistic on both
    ps, (as_, bs, y16s), _ = R.prove_vus(triv, check=False)
    recs = q_leak.recover_accumulators(ps, R.sys, R.tb, cfg, lay, y16s, as_, bs, dev)
    cmp_s = q_leak.compare_with_recorded(recs, triv, STEPS)
    report["q_leak"] = {"real_witness": cmp_, "trivial_witness": cmp_s,
                        "witness_free_statistic_real": q_leak.witness_free_distinguisher(pr, R.sys, R.tb, cfg, lay, y16, a, b, dev),
                        "verdict": "ACCEPTED-LEAK" if cmp_["accumulators_recovered_exactly"] == cmp_["of"] else "inconclusive"}
    print("q leak:", report["q_leak"]["real_witness"], report["q_leak"]["verdict"])

    # ---- mask ranks ----------------------------------------------------------------------------------------------
    report["mask_ranks"] = {"opened_columns_actual_cols": mask_rank.opened_columns_rank(cfg, pr.cols),
                            "messages_explicit_small": mask_rank.message_ranks(16, 8),
                            "real_layout": mask_rank.real_layout_counts(cfg)}
    print("mask ranks:", json.dumps(report["mask_ranks"]["opened_columns_actual_cols"]), json.dumps(report["mask_ranks"]["messages_explicit_small"]))

    # ---- simulator -----------------------------------------------------------------------------------------------
    orc = simulator.ProgrammableOracle()
    orc.install()
    try:
        sim_rep: dict = {}
        # (a) honest verifier: the simulated transcript is accepted by the real verifier (through the programmed oracle)
        V = Coins.sample()
        t0 = time.perf_counter()
        sp = simulator.simulate(V, R.p, R.sys, R.tb, cfg, lay, y16, a, b, dev, orc, rng=rng)
        t_sim = time.perf_counter() - t0
        assert isinstance(sp, protocol.Proof), sp
        ok_s, why_s = R.verify_vus(sp, (a, b, y16), lay, coins=V)
        sim_rep["honest_verifier"] = {"simulated_transcript_accepted": ok_s, "why": why_s, "seconds": t_sim,
                                      "oracle_points_programmed": len(orc.table), "rewinds": sp.timings.get("rewinds")}
        # the same transcript under the UNPROGRAMMED oracle: the challenges no longer match -> rejected (the simulator
        # needs the programmable random oracle; it is not a CRH-only simulator)
        orc.uninstall()
        ok_u, why_u = R.verify_vus(sp, (a, b, y16), lay, coins=V)
        orc.install()
        sim_rep["honest_verifier"]["same_transcript_without_oracle_programming"] = {"accepted": ok_u, "why": why_u}
        print("simulator honest:", sim_rep["honest_verifier"])
        # (b) adaptive V*: the prover aborts; the simulator outputs the same abort
        for slot in (1, 2):
            Vb = simulator.AdaptiveVerifier(slot)
            try:
                R.prove_vus(real, check=False, coins=Vb)
                real_ab = None
            except CoinOpeningError as e:
                real_ab = {"slot": e.slot, "sent": e.sent}
            sb = simulator.simulate(Vb, R.p, R.sys, R.tb, cfg, lay, y16, a, b, dev, orc, rng=rng)
            sim_ab = {"slot": sb.slot, "sent": sb.sent} if isinstance(sb, simulator.AbortTranscript) else None
            sim_rep[f"adaptive_slot{slot}"] = {"real_prover_abort": real_ab, "simulator_abort": sim_ab,
                                               "same": real_ab is not None and sim_ab is not None and real_ab["slot"] == sim_ab["slot"]}
        print("simulator adaptive:", sim_rep["adaptive_slot1"], sim_rep["adaptive_slot2"])
        # (c) selectively aborting V*: abort as a function of root (must not correlate with the witness) ...
        N = args.n_transcripts
        pred_root = lambda V, root, w, h, q, v: root[0] < 64
        ab_real = ab_sim = 0
        for _ in range(N):
            Vc = simulator.AbortingVerifier(pred_root, slot=1)
            try:
                R.prove_vus(real, check=False, coins=Vc)
            except CoinOpeningError:
                ab_real += 1
            Vc2 = simulator.AbortingVerifier(pred_root, slot=1)
            if isinstance(simulator.simulate(Vc2, R.p, R.sys, R.tb, cfg, lay, y16, a, b, dev, orc, rng=rng), simulator.AbortTranscript):
                ab_sim += 1
        sim_rep["aborting_on_root"] = {"n": N, "real_abort_rate": ab_real / N, "sim_abort_rate": ab_sim / N, "expected": 0.25}
        # ... and as a function of q|_H (the leak): abort iff the recovered first accumulator's f-component is odd
        def pred_q(V, root, w, h, q, v):
            """Recover the first VU's first accumulator from q (the verifier knows its own r1); abort iff the
            6 x 3 system is consistent and the recovered f-component is odd."""
            fake = protocol.Proof(root, w, h, [], None, [], "", {}, q, v, V.commitments(), Coins(V.r1, V.s1, V.r2, V.s2))
            rec = q_leak.recover_accumulators(fake, R.sys, R.tb, cfg, lay, y16, a, b, dev)
            return bool(rec["consistent"][1]) and bool(int(rec["components"][1, 2]) & 1)
        ab_real = ab_sim = 0
        for _ in range(max(4, N // 4)):
            Vq = simulator.AbortingVerifier(pred_q, slot=2)
            try:
                R.prove_vus(real, check=False, coins=Vq)
            except CoinOpeningError:
                ab_real += 1
            Vq2 = simulator.AbortingVerifier(pred_q, slot=2)
            if isinstance(simulator.simulate(Vq2, R.p, R.sys, R.tb, cfg, lay, y16, a, b, dev, orc, rng=rng), simulator.AbortTranscript):
                ab_sim += 1
        nq = max(4, N // 4)
        first_f = q_leak.decode_fp32(int(real[0][2][0]))[2]
        sim_rep["aborting_on_q_leak"] = {"n": nq, "real_abort_rate": ab_real / nq, "sim_abort_rate": ab_sim / nq,
                                         "real_first_accumulator_f_odd": bool(first_f & 1),
                                         "note": "the abort event equals a witness bit for the real prover and never fires for the simulator: abort-conditioning inherits the q leak"}
        print("simulator aborting:", sim_rep["aborting_on_root"], sim_rep["aborting_on_q_leak"])
        # (d) indistinguishability: N real vs N simulated accepting transcripts, honest coins, per position
        reals, sims = [], []
        t0 = time.perf_counter()
        for i in range(N):
            Vh = Coins.sample()
            p_r, _, _ = R.prove_vus(real, check=False, coins=Vh)
            reals.append(p_r)
            Vh2 = Coins.sample()
            p_s = simulator.simulate(Vh2, R.p, R.sys, R.tb, cfg, lay, y16, a, b, dev, orc, rng=rng)
            assert isinstance(p_s, protocol.Proof)
            if i < 2:
                ok2, why2 = R.verify_vus(p_s, (a, b, y16), lay, coins=Vh2)
                assert ok2, why2
            sims.append(p_s)
        sim_rep["indistinguishability"] = simulator.indistinguishability(reals, sims, m)
        sim_rep["indistinguishability"]["seconds"] = time.perf_counter() - t0
        # q distinguishers on the same transcripts
        wf_real = [q_leak.witness_free_distinguisher(x, R.sys, R.tb, cfg, lay, y16, a, b, dev)["fraction"] for x in reals[:8]]
        wf_sim = [q_leak.witness_free_distinguisher(x, R.sys, R.tb, cfg, lay, y16, a, b, dev)["fraction"] for x in sims[:8]]
        kw_real = [q_leak.compare_with_recorded(q_leak.recover_accumulators(x, R.sys, R.tb, cfg, lay, y16, a, b, dev), real, STEPS)["accumulators_recovered_exactly"] for x in reals[:8]]
        kw_sim = [q_leak.compare_with_recorded(q_leak.recover_accumulators(x, R.sys, R.tb, cfg, lay, y16, a, b, dev), real, STEPS)["accumulators_recovered_exactly"] for x in sims[:8]]
        sim_rep["q_distinguishers"] = {"witness_free_consistent_fraction": {"real": wf_real, "simulated": wf_sim},
                                       "known_witness_accumulators_matched": {"real": kw_real, "simulated": kw_sim, "of": (STEPS - 1) * lay.n_vus},
                                       "advantage": 1.0 if (min(wf_real) == 1.0 and max(wf_sim) < 0.01) else "see values"}
        print("indistinguishability:", json.dumps({k: v for k, v in sim_rep["indistinguishability"].items()}, default=str))
        print("q distinguishers:", sim_rep["q_distinguishers"])
        report["simulator"] = sim_rep
    finally:
        orc.uninstall()

    # ---- two witnesses, honest coins, per position (the empirical class-1 test at the real layout) ---------------
    N = args.n_transcripts
    A_, B_ = [], []
    for _ in range(N):
        A_.append(R.prove_vus(real, check=False)[0])
        B_.append(R.prove_vus(triv, check=False)[0])
    tw = {}
    for name, pick in (("opened_witness_rows", lambda pr: pr.opened[:m]), ("w", lambda pr: pr.w), ("h", lambda pr: pr.h),
                       ("v", lambda pr: pr.v), ("q", lambda pr: pr.q)):
        tw[name] = simulator.per_position_chi2(np.stack([pick(x) for x in A_]), np.stack([pick(x) for x in B_]))
    # v differs by construction (beta is public and statement-dependent): report it but it is not a leak
    tw["note"] = ("v|_H = beta differs between the two statements (public data); q|_H differs by the witness (the leak); "
                  "opened columns, w, h must be at chance")
    report["two_witness"] = tw
    print("two witness:", json.dumps(tw, default=str))

    # ---- accounting ----------------------------------------------------------------------------------------------
    snd = soundness(cfg, R.tb.L, R.tb.Q, args.n_proofs)
    report["accounting"] = {"soundness": {k: v for k, v in snd.items() if k != "note"},
                            "coin_hiding_term_log2": acc.log2(COIN_HIDING_TERM),
                            "coin_hiding_union_log2": acc.log2(COIN_HIDING_TERM * args.n_proofs),
                            "irs_query_term_check_log2": args.n_proofs and math.log2(args.n_proofs) + cfg.t * math.log2(1 - ((cfg.n - cfg.k) // 2) / cfg.n)}
    print("accounting:", json.dumps(report["accounting"], default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
