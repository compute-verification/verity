"""M3 proper's dry run (PRF-172): H200's drawn proof block template, ``ScaledMmS8CoordinateRows_v1{K, R, DOT}`` as
`e2e_h200/m3bundle.definition` binds it, as a one-member window of honest values (and a twin whose output is not its
Definition's), proved as ``prove_bundle.py --step`` proves a drawn block (`prove_bundle.prove`: staged, one C-Flock
``--zk`` session behind the Lean firewall, Lean ``verify --zk``, a flipped bit, the tampered arms). It tests that
template's staging and proof path at a small K before the bundle exists, not the bundle's ties to the step root.

    python benchmarks/one_stage/m3_dry_unit.py --K 256 --R 2 --scratch DIR --bin flock-circuit --lean flock-verify \\
        --firewall flock-firewall --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prove_bundle as PB  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.evaluation import evaluate_batch  # noqa: E402
from verity.ml.circuits.ops.boolean import gemm as BG  # noqa: E402
from verity.ml.windows import interface as IF  # noqa: E402


def window(d: Path, K: int, parts: int, tamper: bool) -> None:
    fn = bind(BG.ScaledMmS8CoordinateRows, K=K, R=parts, DOT=BG.GEMM_SM120_S8["DOT"])
    rng = np.random.default_rng(0)
    x, w = (rng.integers(-128, 128, (1, K)).astype(np.int8).view(np.uint8) for _ in range(2))
    sx, sw = ((np.abs(rng.standard_normal(1)) / 127).astype(np.float32).view(np.uint32) for _ in range(2))
    p = K // parts
    ins = [*(x[:, i * p:(i + 1) * p] for i in range(parts)), sx, *(w[:, i * p:(i + 1) * p] for i in range(parts)), sw]
    word = BG.ScaledMmS8CoordinateRows.word({"K": K, "R": parts, "DOT": BG.GEMM_SM120_S8["DOT"]})
    out = np.asarray(evaluate_batch(word, *ins, kernel="reference"), dtype=np.uint16).reshape(1, -1)
    iw = IF.Window(IF.Circuit("m3-dry/scaledmm-rows", ((fn, 1),)), "cd" * 64, D.parse_law("subset:1"), 1)
    iw.record(fn, ins, [out ^ np.uint16(1) if tamper else out], {"dry": "m3"})
    iw.write(iw.close(salt_key=bytes(32)), d, {"loop": "m3-dry"}, registrar=R)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--firewall", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flips", type=int, default=1)
    a = ap.parse_args(argv)
    a.session_timeout, a.lean_jobs = 14400.0, 1
    s, out = Path(a.scratch), Path(a.out)
    window(s / "window", a.K, a.R, False)
    window(s / "tampered", a.K, a.R, True)
    st = PB.Staged(s / "window", out / "honest")
    m = st.mine.members[0]
    bm = {"k": 0, "base": m.base, "local": [0], "draw": {"law": "subset", "population": m.n, "k": 1, "units": [0]}}
    label = f"ScaledMmS8CoordinateRows_v1{{K={a.K}, R={a.R}}}"
    rows, negatives, fk = PB.prove(a, [("b0", label, lambda: st.member(0), bm, out / "honest" / "session-m0")],
                                   (s / "tampered", bm), out)
    for key, r in rows.items():
        print("VERDICT " + json.dumps({"block": label, "accepted": r["accepted"], "k_log": r["k_log"],
                                       "prove_s": r["timing"]["e2e_s"], "verify_s": r["lean"]["verify_s"],
                                       "m0": r["m0"]["accepted"], "lean": r["lean"].get("line")}, sort_keys=True), flush=True)
    return PB._finish({"format": "verity/m3-dry-unit/v1", "K": a.K, "R": a.R}, out, rows, negatives, fk, {})


if __name__ == "__main__":
    raise SystemExit(main())
