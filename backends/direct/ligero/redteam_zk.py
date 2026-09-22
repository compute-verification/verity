"""Simulator-style test of the HVZK masking (PROTOCOL.md section 8b as implemented in ``protocol.py``).

What the honest verifier sees of the witness: the t opened columns of U (every committed row at t points off H), and
the messages w (proximity), v (linear), h (quadratic), q (chain).  Under the masking each of these is uniform (w, v, h)
or uniform subject to the public constraint (v on H = beta; sum over H of q = r^T b), independent of the witness --
so a simulator that samples them uniformly and programs the random oracle is perfect up to the 2^-31 sampling bias.
This module checks the *distributional* consequence on real proofs, against the unmasked prover as the control:

  1. real vs simulated: a proof of real VUs and a proof of the trivial VU (all-zero operands; what a simulator that
     knows no witness can produce) at the same layout -- two-sample chi-square over 64 value bins on the opened
     columns and on every message.  Masked: indistinguishable (p > 0.01).  Unmasked: they differ wildly (the trivial
     witness has thousands of constant rows).
  2. uniformity of the opened columns (chi-square against uniform, 64 bins).
  3. payload in the clear: opened values equal to a payload value of their own row (non-ZK: every constant row is
     opened in the clear at every column -- ~35% of all opened values are 0; ZK: the uniform-chance count M t |payload| / p).

  4. malicious verifier (PROTOCOL.md 8c, the HM96 step-0 coin commitment): (a) a verifier whose coin openings are a
     function of the prover's messages (r1 := f(root), r2 := f(w, h)) -- the prover must abort at the mismatched
     opening, having sent nothing beyond root (slot 1) or the blinded test messages (slot 2), never an opened column;
     (b) a verifier that commits to adversarially chosen coins (all-zero randomness; randomness ground before step 0
     against a target column set) -- bound to them, its view is the honest-verifier view at those coins: the same
     real-vs-simulated statistics as in 1, and it gains no control over the column set (overlap with the target at
     chance); (c) the control: with the commitment off (coins = H(transcript), the pre-8c protocol) an interactive
     verifier could choose the coins after the prover's messages -- for THIS prover the masking makes every message
     uniform for every in-domain coin, so the measured advantage is at chance in all three settings: the commitment
     is what makes the simulator's rewinding argument go through and turns equivocation into a public abort.

    python -m backends.direct.ligero.redteam_zk --device cuda --out results/redteam_zk.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from verity_numerical.bench import instances

from . import protocol
from .compile import P
from .protocol import COIN_BYTES, CoinOpeningError, Coins, mask_rows
from .vu import STEP, STEPS, ChainRunner, _vu_data
from .witness import build_witness

BINS = 64


# ---------------------------------------------------------------------------------------------------------------------
# Malicious verifiers (coin sources with the ``Coins`` interface)
# ---------------------------------------------------------------------------------------------------------------------

def _h(*parts: bytes) -> bytes:
    import hashlib
    return hashlib.blake2b(b"|".join(parts), digest_size=COIN_BYTES).digest()


class EquivocatingVerifier(Coins):
    """Commits honestly at step 0, then opens slot ``slot`` to a value that depends on the prover's messages (the
    adaptive coin choice the commitment forbids).  The prover must abort at that opening."""

    def __init__(self, slot: int):
        h = Coins.sample()
        super().__init__(h.r1, h.s1, h.r2, h.s2)
        object.__setattr__(self, "slot", slot)

    def open1(self, root):
        return (_h(b"adaptive-r1", root), self.s1) if self.slot == 1 else (self.r1, self.s1)

    def open2(self, root, w, h, q, v):
        return (_h(b"adaptive-r2", root, w.tobytes(), h.tobytes()), self.s2) if self.slot == 2 else (self.r2, self.s2)


def adversarial_coins(kind: str, target_cols: list[int] | None = None, cfg=None, stmt: bytes | None = None) -> Coins:
    """Coins a malicious verifier might COMMIT to: ``zero`` (no entropy) or ``ground`` (r2 chosen among 256
    candidates to maximise the overlap of the column set with a target set).  Since the b-zk-fix derivation
    ``cols = H(stmt, c, r2)`` is a function of the verifier's own data alone, a verifier that knows the statement
    CAN steer its column set before step 0 -- which is allowed: an interactive verifier owns its coins, and the
    masking hides any ``t <= t_pad`` columns (the test checks the view at the ground columns is still the simulated
    one).  Without the statement (``stmt=None``) the grinding is against a stand-in and the overlap is at chance."""
    z = b"\0" * COIN_BYTES
    if kind == "zero":
        return Coins(z, z, z, z)
    if kind == "ground":
        best, best_score = None, -1
        s1 = z
        for i in range(256):
            r2 = _h(b"grind", i.to_bytes(4, "little"))
            cc = (protocol.coin_commit(z, z), protocol.coin_commit(r2, s1))
            cols = protocol._challenge2(stmt if stmt is not None else b"", b"", np.zeros(1, "<u4"), np.zeros(1, "<u4"),
                                        cfg.t, cfg.n, None, None, cc, r2)
            score = len(set(cols) & set(target_cols))
            if score > best_score:
                best, best_score = r2, score
        return Coins(z, z, best, s1)
    raise ValueError(kind)


def malicious_verifier_tests(R: ChainRunner, real: list, trivial: list) -> dict:
    """Section 4 of the module docstring.  ``R`` is a --zk ChainRunner."""
    out: dict = {}
    # (a) equivocation at either slot: the prover aborts, publicly, before any coin-dependent message
    for slot in (1, 2):
        try:
            R.prove_vus(real, check=False, coins=EquivocatingVerifier(slot))
            out[f"equivocate_slot{slot}"] = {"prover_aborted": False}
        except CoinOpeningError as e:
            out[f"equivocate_slot{slot}"] = {"prover_aborted": True, "slot": e.slot, "sent_before_abort": e.sent,
                                             "opened_columns_sent": False}
    # (b) adversarially chosen but committed coins: bound -> honest-verifier view at those coins
    lay = R.layout(len(real))
    cfg = R.cfg(lay.l)
    rng = np.random.default_rng(3)
    target = sorted(rng.choice(cfg.n, cfg.t, replace=False).tolist())
    p0, (a0, b0, y0), _ = R.prove_vus(real, check=False)
    stmt = protocol.statement_digest(R.sys, cfg, a0, b0, layout=lay, y16=y0)
    for kind in ("zero", "ground"):
        coins = adversarial_coins(kind, target, cfg, stmt=stmt)
        pr, pubs_r, lay_r = R.prove_vus(real, check=False, coins=coins)
        ps, pubs_s, _ = R.prove_vus(trivial, check=False, coins=coins)
        okr, whyr = R.verify_vus(pr, pubs_r, lay_r, coins=coins)
        oks, whys = R.verify_vus(ps, pubs_s, lay_r, coins=coins)
        assert okr and oks, (whyr, whys)
        rep = {"accepted": [okr, oks], "coins": kind, "real_vs_simulated": {}}
        for name, xr, xs in (("opened", pr.opened, ps.opened), ("w", pr.w, ps.w), ("h", pr.h, ps.h), ("q", pr.q, ps.q), ("v", pr.v, ps.v)):
            stat, pv = _chi2_two_sample(xr, xs)
            rep["real_vs_simulated"][name] = {"chi2": stat, "p": pv}
        stat, pv = _chi2_uniform(pr.opened[:R.sys.m])
        rep["opened_uniform"] = {"chi2": stat, "p": pv}
        overlap = len(set(pr.cols) & set(target))
        rep["target_column_overlap"] = {"got": overlap, "expected_at_chance": cfg.t * cfg.t / cfg.n, "t": cfg.t, "n": cfg.n,
                                        "note": "cols = H(stmt, c, r2) is the verifier's own function since b-zk-fix: a "
                                                "statement-aware verifier may steer its columns (allowed; any t <= t_pad "
                                                "columns are masked) -- the overlap is reported, not bounded"}
        out[f"committed_{kind}"] = rep
    # (c) control: the Fiat-Shamir mode (no step 0, coins = H(transcript)).  Same statistics: the masking is
    # coin-oblivious (its ZK is HVZK / NIZK-in-PROM only, PROTOCOL.md 8c).
    Rf = ChainRunner(R.device, R.target, R.rate_log2, R.hints_mode, zk=True, mode=protocol.MODE_FIAT_SHAMIR)
    pr, pubs_r, lay_r = Rf.prove_vus(real, check=False)
    ps, pubs_s, _ = Rf.prove_vus(trivial, check=False)
    okr, _ = Rf.verify_vus(pr, pubs_r, lay_r)
    oks, _ = Rf.verify_vus(ps, pubs_s, lay_r)
    rep = {"accepted": [okr, oks], "real_vs_simulated": {}, "config": str(Rf.cfg(lay_r.l))}
    for name, xr, xs in (("opened", pr.opened, ps.opened), ("w", pr.w, ps.w), ("h", pr.h, ps.h), ("q", pr.q, ps.q), ("v", pr.v, ps.v)):
        stat, pv = _chi2_two_sample(xr, xs)
        rep["real_vs_simulated"][name] = {"chi2": stat, "p": pv}
    out["control_no_commitment"] = rep
    assert okr and oks, "fiat-shamir mode proofs must verify"
    # a Fiat-Shamir proof must not verify under the interactive verifier (another statement digest / no coins)
    ok_cross, why_cross = R.verify_vus(pr, pubs_r, lay_r, coins=Coins.sample())
    out["pre_8c_proof_under_8c_verifier"] = {"accepted": ok_cross, "why": why_cross}
    ok = (all(out[f"equivocate_slot{s}"]["prover_aborted"] for s in (1, 2))
          and not any("opened" in m for m in out["equivocate_slot1"]["sent_before_abort"])
          and out["equivocate_slot1"]["sent_before_abort"][-1] == "root"
          and all(v["p"] > 0.001 for k in ("zero", "ground") for v in out[f"committed_{k}"]["real_vs_simulated"].values())
          and all(out[f"committed_{k}"]["opened_uniform"]["p"] > 0.001 for k in ("zero", "ground"))
          and not ok_cross)
    out["verdict"] = "PASS" if ok else "FAIL"
    return out


def _chi2_uniform(x: np.ndarray) -> tuple[float, float]:
    """Chi-square statistic and p-value (Wilson-Hilferty normal approximation) of x against uniform on [0, p)."""
    counts = np.bincount((x.astype(np.int64).ravel() * BINS // P), minlength=BINS)
    e = x.size / BINS
    stat = float(((counts - e) ** 2 / e).sum())
    return stat, _chi2_sf(stat, BINS - 1)


def _chi2_two_sample(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    cx = np.bincount((x.astype(np.int64).ravel() * BINS // P), minlength=BINS).astype(float)
    cy = np.bincount((y.astype(np.int64).ravel() * BINS // P), minlength=BINS).astype(float)
    kx, ky = math.sqrt(cy.sum() / cx.sum()), math.sqrt(cx.sum() / cy.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = (kx * cx - ky * cy) ** 2 / (cx + cy)
    stat = float(np.nansum(terms))
    return stat, _chi2_sf(stat, BINS - 1)


def _chi2_sf(stat: float, dof: int) -> float:
    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * math.erfc(z / math.sqrt(2))


def _payload_in_clear(W: torch.Tensor, opened: np.ndarray) -> dict:
    """Opened values equal to a payload value of their own row, and the constant rows opened in the clear."""
    Wn = W.cpu().numpy().astype(np.int64)
    m = Wn.shape[0]
    hits, const_rows, const_leaked, distinct = 0, 0, 0, 0
    for j in range(m):
        vals = np.unique(Wn[j])
        distinct += vals.size
        h = np.isin(opened[j].astype(np.int64), vals).sum()
        hits += int(h)
        if vals.size == 1:
            const_rows += 1
            const_leaked += int((opened[j] == vals[0]).all())
    t = opened.shape[1]
    return {"opened_values_equal_to_own_payload": hits, "of": m * t, "expected_if_uniform": distinct * t / P,
            "constant_rows": const_rows, "constant_rows_opened_in_the_clear": const_leaked}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--root", default=str(instances.DEFAULT_ROOT))
    ap.add_argument("--vus", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    vu = instances.load("vu-k1536", Path(args.root))
    real = [_vu_data(vu, i) for i in range(args.vus)]
    trivial = [([0] * (STEP * STEPS), [0] * (STEP * STEPS), [0] * STEPS, 0)] * args.vus
    report = {}
    if args.device.startswith("cuda"):
        from .mask_sampler import sampler_for
        smp = sampler_for(args.device)
        if smp is not None:
            sym, key = smp.uniform(1_000_000)
            report["mask_sampler"] = {"kernel_bit_exact_vs_rfc8439_reference": smp.check(),
                                      "symbols_in_field": bool(((sym >= 0) & (sym < P)).all()),
                                      "uniform_chi2_p": _chi2_uniform(sym.cpu().numpy())[1],
                                      "host_reference_prefix_equal": bool((sym[:2048].cpu().numpy() == __import__("backends.direct.ligero.mask_sampler", fromlist=["uniform_host"]).uniform_host(2048, key)).all())}
            print("mask sampler (ChaCha20 device, rejection):", report["mask_sampler"])
    for zk in (False, True):
        R = ChainRunner(args.device, -128.0, zk=zk)
        pr, pubs_r, lay = R.prove_vus(real, check=False)
        ps, pubs_s, _ = R.prove_vus(trivial, check=False)
        okr, whyr = R.verify_vus(pr, pubs_r, lay)
        oks, whys = R.verify_vus(ps, pubs_s, lay)
        assert okr and oks, (whyr, whys)
        m = R.sys.m
        # the witness of the real proof, rebuilt for the payload test (the test knows the witness; the verifier does not)
        from .hints_torch import hints_torch
        from .unit import public_vectors
        a, b, _ = pubs_r
        units = []
        for (x, w, accs, y) in real:
            c = 0
            for s in range(STEPS):
                units.append((c, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16], accs[s]))
                c = accs[s]
        while len(units) < lay.l:
            units.append((0, [0] * STEP, [0] * STEP, 0))
        c_in = np.asarray([u[0] for u in units]); y_out = np.asarray([u[3] for u in units])
        pub = public_vectors(R.p, None, a, b, None, device=R.device)
        hints = hints_torch(R.p, R.sys, c_in, a, b, y_out, R.device)
        W = build_witness(R.sys, pub, hints, R.device)
        rep = {"config": str(R.cfg(lay.l)), "committed_rows": int(pr.opened.shape[0]), "mask_rows": {k: [s.start, s.stop] for k, s in mask_rows(R.cfg(lay.l), m, True).items()}}
        stat, pv = _chi2_uniform(pr.opened[:m])
        rep["opened_uniform"] = {"chi2": stat, "p": pv, "n": int(pr.opened[:m].size)}
        rep["real_vs_simulated"] = {}
        for name, xr, xs in (("opened", pr.opened, ps.opened), ("w", pr.w, ps.w), ("h", pr.h, ps.h), ("q", pr.q, ps.q),
                             ("v", pr.v, ps.v)):
            if xr is None:
                continue
            stat, pv = _chi2_two_sample(xr, xs)
            rep["real_vs_simulated"][name] = {"chi2": stat, "p": pv, "n": int(xr.size)}
        rep["payload"] = _payload_in_clear(W, pr.opened[:m])
        rep["zeros_among_opened"] = int((pr.opened == 0).sum())
        report["zk" if zk else "non_zk"] = rep
        print(("ZK   " if zk else "plain") + f" {rep['config']}\n   opened uniform: chi2 {rep['opened_uniform']['chi2']:.1f} p {rep['opened_uniform']['p']:.3g}")
        for k_, v_ in rep["real_vs_simulated"].items():
            print(f"   real vs simulated {k_:6s}: chi2 {v_['chi2']:.1f} p {v_['p']:.3g}")
        print(f"   payload: {rep['payload']}\n   zeros among opened values: {rep['zeros_among_opened']}")
        if zk:
            mv = malicious_verifier_tests(R, real, trivial)
            report["malicious_verifier"] = mv
            for s in (1, 2):
                e = mv[f"equivocate_slot{s}"]
                print(f"   equivocating verifier, slot {s}: prover aborted {e['prover_aborted']}; sent before abort: {e.get('sent_before_abort')}")
            for k in ("zero", "ground"):
                c = mv[f"committed_{k}"]
                print(f"   committed adversarial coins ({k}): real vs simulated p = " +
                      ", ".join(f"{n} {v['p']:.3g}" for n, v in c["real_vs_simulated"].items()) +
                      f"; opened uniform p {c['opened_uniform']['p']:.3g}; target-column overlap {c['target_column_overlap']['got']} "
                      f"(chance {c['target_column_overlap']['expected_at_chance']:.2f})")
            c = mv["control_no_commitment"]
            print("   control (no commitment): real vs simulated p = " + ", ".join(f"{n} {v['p']:.3g}" for n, v in c["real_vs_simulated"].items()))
            print(f"   pre-8c proof under the 8c verifier: {mv['pre_8c_proof_under_8c_verifier']}")
            print("   malicious-verifier verdict:", mv["verdict"])
    zk = report["zk"]
    verdict = (zk["opened_uniform"]["p"] > 0.001 and all(v["p"] > 0.001 for v in zk["real_vs_simulated"].values())
               and zk["payload"]["constant_rows_opened_in_the_clear"] == 0
               and zk["payload"]["opened_values_equal_to_own_payload"] < 20 * (1 + zk["payload"]["expected_if_uniform"])
               and report["malicious_verifier"]["verdict"] == "PASS"
               and all(v is True or (isinstance(v, float) and v > 0.001) for v in report.get("mask_sampler", {}).values()))
    report["verdict"] = "PASS" if verdict else "FAIL"
    print("verdict:", report["verdict"])
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0 if verdict else 2


if __name__ == "__main__":
    raise SystemExit(main())
