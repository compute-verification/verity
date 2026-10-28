"""E2E-120's driver: one served int8 window's committed values taken through today's spot check, printing accept/reject for
the honest run and every attack, and the served design's ε and γ.

    python3 benchmarks/pouw/e2e_120/run.py [--m M --n N --k K --window S] [--seed SEED] [--json]

With no C-Flock, this is the check alone (`window.Round`, the statement's public file by `window.ideal_proof`): it shows the
honest window accepted and the library rejected, and that MITM and skip-filler are not caught by today's duo check. On the
pod, benchmarks/pouw/e2e_120/job.sh stages the same `PouwSegCheckDither_v2{S, duo}` statement through C-Flock --zk and the
Lean verifier of record, so accept/reject there is the verifier's; `run.py --stage OUT` writes one honest statement's
circuit.txt, inst.bin and pub.bin for it.

Stand-in values until vLLM's int8 window exists: `window.ServedValues.stand_in` (seeded int8 W and X). Real values plug in at
`window.ServedValues` (W the registered weights, X the served activations), `window.RoundCoins` (U, the dither d and the receipt key
from the coin; H1's cgemm.cu tags) and `window.Honest.receipts` (the H1 kernel's receipts in this layout; today duo, under
PIN's v3 fx1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import accounting as AC   # noqa: E402
import attacks as AK      # noqa: E402
import window as WD       # noqa: E402


def one_window(shape: WD.Shape, seed: int) -> dict:
    honest = WD.play(AK.honest(shape, seed), shape, seed=seed)
    runs = [{"run": "honest", "accepted": honest.accepted, "why": honest.why, "caught_as_attack": None,
             "draw": vars(honest.draw) if honest.draw else None}]
    for name, (caught, reason) in AK.ATTACKS.items():
        v = AK.run_attack(name, shape, seed=seed)
        runs.append({"run": name, "accepted": v.accepted, "why": v.why, "rejected": not v.accepted,
                     "caught_by_todays_check": caught, "reason_not_caught": reason,
                     "draw": vars(v.draw) if v.draw else None})
    return {"shape": vars(shape), "seed": seed, "runs": runs}


def print_table(result: dict, head: AC.Headline) -> None:
    print(f"E2E-120: one served int8 window ({result['shape']['m']}x{result['shape']['n']}x{result['shape']['k']}, "
          f"S={result['shape']['S']}) through today's check (PouwSegCheckDither_v2 duo), seed {result['seed']}\n")
    print(f"{'run':20s} {'verdict':8s} {'as expected':12s} why")
    for r in result["runs"]:
        if r["run"] == "honest":
            verdict = "ACCEPT" if r["accepted"] else "REJECT"
            exp = "yes" if r["accepted"] else "NO"
            print(f"{r['run']:20s} {verdict:8s} {exp:12s} {r['why'] or 'honest window accepted'}")
        else:
            verdict = "REJECT" if r["rejected"] else "ACCEPT"
            # expectation: a caught attack must be rejected; an uncaught one is accepted (xfail, with a reason)
            as_exp = (r["rejected"] == r["caught_by_todays_check"])
            tag = "yes" if as_exp else "NO"
            if not r["caught_by_todays_check"] and not r["rejected"]:
                tag = "xfail"
            why = r["why"] or (r["reason_not_caught"] or "")
            print(f"{r['run']:20s} {verdict:8s} {tag:12s} {why}")
    print()
    print(head.line())


def stage_one(out: Path, shape: WD.Shape, seed: int) -> dict:
    """Write one honest statement for the pod: its circuit.txt, inst.bin and pub.bin through C-Flock's UnitStager, so the Lean
    verifier checks a real proof of today's `PouwSegCheckDither_v2{S, duo}`."""
    import numpy as np
    from verity.ml.circuits.defs import bind
    from verity_flock import class_statement as CS

    prover = AK.honest(shape, seed)
    rng = __import__("random").Random(seed)
    rd = WD.Round(shape, prover.w_tree.root, prover.session, 1, rng, "duo")
    op = prover.run(rd)
    why = rd.judge(op)
    if why is not None:
        raise SystemExit(f"stage: the honest window did not pass the check: {why}")
    out.mkdir(parents=True, exist_ok=True)
    definition = bind(WD.SG.PouwSegCheckDither, S=shape.S, FOLD="duo")
    stager = CS.UnitStager(definition, "unit/e2e-120-wincheck-dither-duo", WD.in_rows(shape.S, "duo"),
                           [(WD.V2, (0,))], circuit=out / "circuit.txt")
    rows = {r: np.asarray(op.rows[r], np.uint16).reshape(1, -1) for r, *_ in WD.in_rows(shape.S, "duo")}
    salt_key = __import__("hashlib").sha256(b"e2e-120/stage").digest()
    salts = {r: [op.salts[r]] for r in op.salts if r in rows}
    st = stager.stage(rows, out, salt_key, salts=salts)
    return {"pin": stager.pin, "ands": st["ands"], "unit_nnz": st["unit_nnz"], "out": str(out),
            "o0": int(np.asarray(st["outs"]["o0"]).reshape(-1)[0])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stage", type=Path, help="write one honest statement for C-Flock under this directory, nothing else")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    shape = WD.Shape(a.m, a.n, a.k, a.window)
    if a.stage is not None:
        print(json.dumps(stage_one(a.stage, shape, a.seed)))
        return 0
    result = one_window(shape, a.seed)
    head = AC.headline()
    result["accounting"] = {"eps_core": head.eps_core, "eps_skip": head.eps_skip, "eps_machine": head.eps_machine,
                            "gamma": head.gamma, "vector": head.vector, "q": head.q}
    if a.json:
        print(json.dumps(result, indent=1))
    else:
        print_table(result, head)
    ok = result["runs"][0]["accepted"] and all(
        r["rejected"] == r["caught_by_todays_check"] for r in result["runs"][1:])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
