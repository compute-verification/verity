"""The torch loop's window, written for the audit from its records alone (`benchmarks/prove_path/records_audit.py`).

    python -m verity_torch_loop.serve --out DIR [--law L] [--requests 3,7] [--steps 2] [--h 64] [--cap 8] [--tamper]
    python -m verity_torch_loop.serve --circuit-only [the same shape]

The first serves the requests (`loop.TorchLoop`), commits every unit and writes the window (``circuit.json``, the
registration under core's ``one_stage.registration``, the value files) to ``--out``, and prints the circuit record's
SHA-512, the run root, the population and the units whose outputs are not their Definition's (`Window.mismatches`: none
unless ``--tamper``). The second prints the circuit record's SHA-512 alone, from the loop's shape and no value: the circuit
the verifier registers before the loop runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity.ml.windows import interface as IF

from . import loop as L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--circuit-only", action="store_true")
    ap.add_argument("--law", default="subset:2")
    ap.add_argument("--requests", default="3,7")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--tamper", action="store_true")
    a = ap.parse_args(argv)
    if bool(a.out) == a.circuit_only:
        ap.error("--out DIR, or --circuit-only")
    loop = L.TorchLoop(L.Model(h=a.h), [int(x) for x in a.requests.split(",")], a.steps, a.tamper)
    law = D.parse_law(a.law)
    if a.circuit_only:
        empty = IF.Window(loop.circuit(), loop.run_root, law, a.cap)
        print(json.dumps({"circuit": hashlib.sha512(IF.circuit_bytes(empty.circuit_record())).hexdigest()}))
        return 0
    win = loop.window(law, a.cap)
    bad = win.mismatches()
    win.write(win.close(), a.out, {"loop": "torch-loop"}, registrar=R)
    rec = json.loads((Path(a.out) / "registration.json").read_text())
    print(json.dumps({"circuit": rec["window"]["circuit"], "run": loop.run_root, "population": win.window.population,
                      "members": len(win.window.members), "mismatched": {t: len(v) for t, v in bad.items()},
                      "registration": R.digest(rec), "out": str(Path(a.out))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
