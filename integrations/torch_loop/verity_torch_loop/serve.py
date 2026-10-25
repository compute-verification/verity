"""The torch loop's window, written for the audit from its records alone (`benchmarks/prove_path/records_audit.py`).

    python -m verity_torch_loop.serve --out DIR [--law L] [--requests 3,7] [--steps 2] [--h 128] [--cap 8] [--run-id ID] \\
        [--tamper] [--opening FILE]
    python -m verity_torch_loop.serve --circuit-only [the same shape and --run-id]

The first serves the requests (`loop.TorchLoop`), commits every unit and writes the window (``circuit.json``, the
registration under core's ``one_stage.registration``, the value files) to ``--out``, and prints the circuit record's
SHA-512, the run root, the population and the units whose outputs are not their Definition's (`Window.mismatches`: none
unless ``--tamper``). With ``--opening FILE`` the circuit is its compiler's (`compile`): the loop's schedule is committed as
advice under that compiler, the registration binds both, and the advice's opening (the prover's private file) goes to
``FILE``. The second prints the circuit record's SHA-512 alone, from the loop's shape and no value: the circuit the verifier
registers before the loop runs, when it has no compiler."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity.ml.windows import interface as IF

from . import compile as C, loop as L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--circuit-only", action="store_true")
    ap.add_argument("--law", default="subset:2")
    ap.add_argument("--requests", default="3,7")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--h", type=int, default=L.H)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--run-id", default="", help="the run's name, which its root binds (`TorchLoop`)")
    ap.add_argument("--tamper", action="store_true")
    ap.add_argument("--opening")
    a = ap.parse_args(argv)
    if bool(a.out) == a.circuit_only:
        ap.error("--out DIR, or --circuit-only")
    loop = L.TorchLoop(L.Model(h=a.h), [int(x) for x in a.requests.split(",")], a.steps, a.tamper, a.run_id)
    law = D.parse_law(a.law)
    if a.circuit_only:
        empty = IF.Window(loop.circuit(), loop.run_root, law, a.cap)
        print(json.dumps({"circuit": hashlib.sha512(IF.circuit_bytes(empty.circuit_record())).hexdigest()}))
        return 0
    advice = None
    if a.opening:
        data, c = C.schedule_bytes(loop.requests, loop.steps), C.compiler()
        reg = IF.commit_advice(c, data)
        advice = IF.advice_binding(c, reg.root.hex(), 8 * len(data))
        Path(a.opening).write_text(json.dumps(IF.advice_opening(data, reg)))
    win = loop.window(law, a.cap, advice)
    bad = win.mismatches()
    win.write(win.close(), a.out, {"loop": "torch-loop"}, registrar=R)
    rec = json.loads((Path(a.out) / "registration.json").read_text())
    print(json.dumps({"circuit": rec["window"]["circuit"], "run": loop.run_root, "population": win.window.population,
                      "members": len(win.window.members), "mismatched": {t: len(v) for t, v in bad.items()},
                      "compiler": rec["window"].get("compiler"), "registration": R.digest(rec), "out": str(Path(a.out))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
