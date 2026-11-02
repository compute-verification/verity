"""M3 proper's step registration: the H200 step's commit-slots roots as ``prove_bundle.py --step`` reads them
(`replay_bundle.step_registration`, ``read_steps``), registered before the coin.

    python demos/h200/m3stepreg.py --roots ROOTS.json --weights REGISTRATION.json --program HEX --run RUN --out DIR

``--roots``: m3draw commit's ``roots.json`` (every step's root and slot roots). Each step root is derived again from its slot
roots under the run's context and the layout (`commit.layout`), and the layout's digest must be the file's. ``--weights``:
the weights registration's ``registration.json`` (its ``registered`` list, every weight's root under ``--program``, which must be its
program). The registration names the last step, every step 0 .. s's root (``steps``), every slot root of step s, the
population ``slots x units_per_slot`` and the law ``subset:K``; ``DIR/step-registration.json`` and its digest
(`replay_bundle.step_digest`, the coin's session root) are written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "interfaces")]
from e2e_h200.commit import _slots, layout  # noqa: E402
from e2e_h200.shape import SHAPES  # noqa: E402
from verity_e2e_demo import replay_bundle as RB  # noqa: E402


def build(roots: dict, weights: dict, program: str, run: str, k: int) -> dict:
    S, _ = _slots()
    shape = SHAPES[roots["shape"]]
    lay = layout(shape)
    if S.layout_digest(lay).hex() != roots["layout_digest"] or lay.unit_count != roots["units_a_slot"]:
        raise SystemExit("the roots' layout is not the step's (commit.layout)")
    if weights["program"] != program:
        raise SystemExit(f"the weights are registered under {weights['program'][:16]}…, not {program[:16]}…")
    context, slots = bytes.fromhex(roots["context"]), int(roots["batch"])
    steps = {}
    for r in roots["step_roots"]:
        srs = tuple(bytes.fromhex(x) for x in r["slot_roots"])
        if len(srs) != slots or list(r["slots"]) != list(range(slots)):
            raise SystemExit(f"step {r['step']}: {len(srs)} slot roots, not the lockstep run's {slots}")
        if S.step_tree(context, lay, r["step"], srs).root.hex() != r["root"]:
            raise SystemExit(f"step {r['step']}: its root is not its slot roots' step tree")
        steps[0, int(r["step"])] = RB.StepRecord(b"", {}, srs)
    s = max(t for _, t in steps)
    if sorted(t for _, t in steps) != list(range(s + 1)):
        raise SystemExit("the roots are not every step 0 .. s")
    com = RB.Commitment(context, lay, slots, steps)
    regd = {e["value"]: SimpleNamespace(root=bytes.fromhex(e["root"])) for e in weights["registered"]}
    return RB.step_registration(com, regd, {"law": "subset", "k": k}, run=run, program=program, block=0, step=s,
                                read_steps=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--program", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    reg = build(json.loads(Path(a.roots).read_text()), json.loads(Path(a.weights).read_text()), a.program, a.run, a.k)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "step-registration.json").write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
    print("REGISTERED " + json.dumps({"digest": RB.step_digest(reg), "run": reg["run"], "program": reg["program"],
                                      "step": reg["step"], "step_root": reg["step_root"], "steps": len(reg[RB.STEPS_KEY]),
                                      "population": reg["population"], "law": reg["law"], "weights": len(reg["weights"])}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
