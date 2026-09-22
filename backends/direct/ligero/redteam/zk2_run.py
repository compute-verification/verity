"""Re-audit driver (lane red-team-zk-2) of the b-zk-fix labels.

    python -m backends.direct.ligero.redteam.zk2_run --device cuda --vus 170 --n-proofs 25 --n-transcripts 32 --out ...
    python -m backends.direct.ligero.redteam.zk2_run --device cpu --vus 2 --n-proofs 1 --n-transcripts 4        # laptop

Sections: ``u_high_rank`` (attack 1: the opened u_cb rows vs the residual row masks, exact rank at the real layout,
interactive AND fiat-shamir configurations), ``disclosed_Q_high`` (the t values per coordinate, pooled uniformity per
witness), ``opened_mask_rows_two_witness`` (every opened mask row family across two witnesses), ``simulator``
(real hash, hash function identity asserted before/after, honest / adaptive / aborting V* with abort rates at
larger n, rewinds), ``soundness_probes`` (mutated witness in interactive mode: the mask's free sum cannot repair a
broken chain), ``accounting`` (FS q x eps recomputation from accounting.py constants), ``fs_leak_regression``
(the pre-fix q-leak extraction on the FS transcript).
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

from .. import merkle, protocol
from ..compile import P
from ..protocol import CoinOpeningError, Coins, config_for, soundness
from ..vu import STEPS, ChainRunner
from . import q_leak, simulator, zk2
from .run import _load_real, _trivial


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--root", default=str(instances.DEFAULT_ROOT))
    ap.add_argument("--vus", type=int, default=170)
    ap.add_argument("--n-proofs", type=int, default=25)
    ap.add_argument("--n-transcripts", type=int, default=32)
    ap.add_argument("--abort-n", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    dev = args.device
    rng = np.random.default_rng(11)
    report: dict = {"args": vars(args), "device": dev}
    real = _load_real(Path(args.root), args.vus)
    triv = _trivial(args.vus)
    real_hash = merkle.hash_bytes

    for mode in (protocol.MODE_INTERACTIVE, protocol.MODE_FIAT_SHAMIR):
        R = ChainRunner(dev, -128.0, n_proofs=args.n_proofs, zk=True, mode=mode)
        lay = R.layout(len(real))
        cfg = R.cfg(lay.l)
        m = R.sys.m
        sec: dict = {"config": {"l": cfg.l, "n": cfg.n, "k": cfg.k, "t": cfg.t, "t_pad": cfg.t_pad, "D": cfg.D, "m": m,
                                "committed_rows": int(protocol.n_rows(cfg, m, True)), "mode": cfg.mode}}
        print(mode, sec["config"])
        pr, (a, b, y16), _ = R.prove_vus(real, check=False)
        ok, why = R.verify_vus(pr, (a, b, y16), lay)
        assert ok, why
        # ---- attack 1: exact rank of the residual-mask map onto the disclosed Q high part ---------------------------
        t0 = time.perf_counter()
        sec["u_high_rank"] = zk2.u_high_rank(pr, R.sys, R.tb, cfg, lay, y16, a, b, dev)
        sec["u_high_rank"]["seconds"] = time.perf_counter() - t0
        print("u_high_rank:", json.dumps({k: v for k, v in sec["u_high_rank"].items() if k != "per_d"}), sec["u_high_rank"]["per_d"])
        # the disclosed values themselves, real vs trivial witness, pooled uniformity
        ps, (as_, bs, y16s), _ = R.prove_vus(triv, check=False)
        sec["disclosed_Q_high"] = {"real": zk2.pooled_uniform_chi2(zk2.disclosed_Q_high(pr, m, cfg, dev)),
                                   "trivial": zk2.pooled_uniform_chi2(zk2.disclosed_Q_high(ps, m, cfg, dev))}
        # ---- pre-fix extraction on the fixed transcript (both modes) --------------------------------------------------
        rec = q_leak.recover_accumulators(pr, R.sys, R.tb, cfg, lay, y16, a, b, dev)
        sec["q_leak_regression"] = q_leak.compare_with_recorded(rec, real, STEPS)
        sec["q_leak_regression"]["witness_free_fraction"] = q_leak.witness_free_distinguisher(pr, R.sys, R.tb, cfg, lay, y16, a, b, dev)["fraction"]
        print("q leak regression:", sec["q_leak_regression"]["accumulators_recovered_exactly"], "/", sec["q_leak_regression"]["of"])
        # ---- every opened mask-row family across two witnesses ----------------------------------------------------
        N = args.n_transcripts
        A_, B_ = [], []
        for _ in range(N):
            A_.append(R.prove_vus(real, check=False)[0])
            B_.append(R.prove_vus(triv, check=False)[0])
        sec["opened_mask_rows_two_witness"] = zk2.qb_two_witness(A_, B_, cfg, m)
        sec["opened_mask_rows_two_witness"]["q"] = simulator.per_position_chi2(np.stack([x.q for x in A_]), np.stack([x.q for x in B_]))
        sec["opened_mask_rows_two_witness"]["opened_witness_rows"] = simulator.per_position_chi2(
            np.stack([x.opened[:m] for x in A_]), np.stack([x.opened[:m] for x in B_]))
        dq_a = np.stack([zk2.disclosed_Q_high(x, m, cfg, dev) for x in A_])
        dq_b = np.stack([zk2.disclosed_Q_high(x, m, cfg, dev) for x in B_])
        sec["disclosed_Q_high"]["pooled_two_witness"] = {"real": zk2.pooled_uniform_chi2(dq_a), "trivial": zk2.pooled_uniform_chi2(dq_b)}
        print("two witness:", json.dumps({k: (v["positions_below_threshold"], v["positions"]) for k, v in sec["opened_mask_rows_two_witness"].items()}))
        # ---- soundness probes -------------------------------------------------------------------------------------
        def mutate(W):
            W[0, 1] = (int(W[0, 1]) + 1) % P
        pm, (am, bm, ym), _ = R.prove_vus(real, check=False, mutate=mutate)
        okm, whym = R.verify_vus(pm, (am, bm, ym), lay)
        sec["soundness_probes"] = {"mutated_witness_rejected": not okm, "why": whym}
        snd = soundness(cfg, R.tb.L, R.tb.Q, args.n_proofs)
        sec["accounting"] = {k: v for k, v in snd.items() if k != "note"}
        # ---- simulator (interactive only) --------------------------------------------------------------------------
        if cfg.interactive:
            sim: dict = {}
            assert merkle.hash_bytes is real_hash
            V = Coins.sample()
            sp = simulator.simulate(V, R.p, R.sys, R.tb, cfg, lay, y16, a, b, dev, None, rng=rng)
            assert isinstance(sp, protocol.Proof), sp
            assert merkle.hash_bytes is real_hash, "the simulator replaced the hash"
            ok_s, why_s = R.verify_vus(sp, (a, b, y16), lay, coins=V)
            sim["honest"] = {"accepted_real_hash": ok_s, "why": why_s, "hash_identity_preserved": merkle.hash_bytes is real_hash,
                             "rewinds": sp.timings.get("rewinds")}
            ok_o, why_o = R.verify_vus(sp, (a, b, y16), lay, coins=Coins.sample())
            sim["honest"]["other_verifier_coins"] = {"accepted": ok_o, "why": why_o}
            # u_cb of the simulated transcripts: disclosed_Q_high uniform there by construction; compare to real pooled
            sims = [simulator.simulate(Coins.sample(), R.p, R.sys, R.tb, cfg, lay, y16, a, b, dev, None, rng=rng) for _ in range(N)]
            assert all(isinstance(x, protocol.Proof) for x in sims)
            ind = simulator.indistinguishability(A_, sims, m)
            ind["disclosed_Q_high_pooled_sim"] = zk2.pooled_uniform_chi2(np.stack([zk2.disclosed_Q_high(x, m, cfg, dev) for x in sims]))
            sim["indistinguishability_real_vs_sim"] = ind
            # adaptive V*
            for slot in (1, 2):
                Vb = simulator.AdaptiveVerifier(slot)
                try:
                    R.prove_vus(real, check=False, coins=Vb)
                    ra = None
                except CoinOpeningError as e:
                    ra = e.slot
                sb = simulator.simulate(Vb, R.p, R.sys, R.tb, cfg, lay, y16, a, b, dev, None, rng=rng)
                sim[f"adaptive_slot{slot}"] = {"real_abort_slot": ra, "sim_abort_slot": getattr(sb, "slot", None)}
            # selective abort with larger n: on root (slot 1) and on the opened u_cb row parity (slot 2, sees w,h,q,v only:
            # the openings are not yet sent, so use q's first coefficient parity -- a message-dependent predicate)
            nA = args.abort_n
            res = {}
            for name, pred, slot in (("root_byte<64", lambda V, root, w, h, q, v: root[0] < 64, 1),
                                     ("q00_odd", lambda V, root, w, h, q, v: bool(int(q[0, 0]) & 1), 2),
                                     ("q_high_sum_odd", lambda V, root, w, h, q, v: bool(int(q[:, cfg.k:].astype(np.int64).sum()) & 1), 2)):
                ar = asim = 0
                rew = []
                for _ in range(nA):
                    Vc = simulator.AbortingVerifier(pred, slot=slot)
                    try:
                        R.prove_vus(real, check=False, coins=Vc)
                    except CoinOpeningError:
                        ar += 1
                    Vc2 = simulator.AbortingVerifier(pred, slot=slot)
                    out = simulator.simulate(Vc2, R.p, R.sys, R.tb, cfg, lay, y16, a, b, dev, None, rng=rng)
                    if isinstance(out, simulator.AbortTranscript):
                        asim += 1
                    else:
                        rew.append(out.timings.get("rewinds", 0))
                z = (ar - asim) / math.sqrt(max(1e-9, (ar + asim) * (1 - (ar + asim) / (2 * nA)) / 1)) if (ar + asim) else 0.0
                res[name] = {"n": nA, "real_abort_rate": ar / nA, "sim_abort_rate": asim / nA, "z_two_sample": z,
                             "mean_rewinds_sim_nonabort": float(np.mean(rew)) if rew else None, "max_rewinds": int(max(rew)) if rew else None}
                print("abort", name, res[name])
            sim["selective_abort"] = res
            sec["simulator"] = sim
        report[mode] = sec
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1, default=str))

    # ---- accounting: FS re-parameterisation recomputed independently --------------------------------------------------
    l = report[protocol.MODE_INTERACTIVE]["config"]["l"]
    per_proof = acc.per_proof_target(-128.0, args.n_proofs)
    fs_cfg = config_for(l, -128.0, 2, args.n_proofs, zk=True, mode=protocol.MODE_FIAT_SHAMIR)
    int_cfg = config_for(l, -128.0, 2, args.n_proofs, zk=True, mode=protocol.MODE_INTERACTIVE)
    def query_log2(cfg, t):
        e = (cfg.n - cfg.k) // 2
        return math.log2((1 - e / cfg.n) ** t + ((cfg.k + cfg.l) / cfg.n) ** t + (2 * cfg.k / cfg.n) ** t)
    report["accounting_independent"] = {
        "per_proof_target_log2": per_proof, "fs_shift_log2": protocol.FS_QUERIES_LOG2,
        "interactive": {"t": int_cfg.t, "D": int_cfg.D, "t_pad": int_cfg.t_pad, "query_terms_log2": query_log2(int_cfg, int_cfg.t),
                        "field_terms_log2": math.log2(int_cfg.n + 3) - int_cfg.D * math.log2(P)},
        "fiat_shamir": {"t": fs_cfg.t, "D": fs_cfg.D, "t_pad": fs_cfg.t_pad, "query_terms_plus60_log2": query_log2(fs_cfg, fs_cfg.t) + 60,
                        "field_terms_plus60_log2_D6": math.log2(fs_cfg.n + 3) - 6 * math.log2(P) + 60,
                        "field_terms_plus60_log2_D7": math.log2(fs_cfg.n + 3) - 7 * math.log2(P) + 60,
                        "t_minus_1_query_terms_plus60_log2": query_log2(fs_cfg, fs_cfg.t - 1) + 60,
                        "union_log2": math.log2(args.n_proofs) + math.log2(2 ** (query_log2(fs_cfg, fs_cfg.t) + 60) + (fs_cfg.n + 3) / P ** fs_cfg.D * 2 ** 60)},
    }
    print("accounting:", json.dumps(report["accounting_independent"]))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
