"""The torch loop's window, written for the audit from its records alone (`benchmarks/prove_path/records_audit.py`).

    python -m verity_torch_loop.serve --out DIR [--law L] [--requests 3,7] [--steps 2] [--h 128] [--cap 8] [--run-id ID] \\
        [--tamper] [--opening FILE] [--pouw h1 [--pouw-out DIR] [--pouw-salt HEX]]
    python -m verity_torch_loop.serve --circuit-only [the same shape and --run-id] [--pouw h1]

The first serves the requests (`loop.TorchLoop`), commits every unit and writes the window (``circuit.json``, the
registration under core's ``one_stage.registration``, the value files) to ``--out``, and prints the circuit record's
SHA-512, the run root, the population and the units whose outputs are not their Definition's (`Window.mismatches`: none
unless ``--tamper``). With ``--opening FILE`` the circuit is its compiler's (`compile`): the loop's schedule is committed as
advice under that compiler, the registration binds both, and the advice's opening (the prover's private file) goes to
``FILE``. The second prints the circuit record's SHA-512 alone, from the loop's shape and no value: the circuit the verifier
registers before the loop runs, when it has no compiler. With ``--pouw h1`` the first also writes the loop's PoUW
window (`pouw`: its linears as the pinned H1 tile) to ``--pouw-out`` (default ``DIR/pouw``), its tiles salted by
``--pouw-salt`` (64 bytes in hex, the auditor's coin; fresh when absent), and the second prints that window's circuit
record's SHA-512 beside the first's."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity.ml.windows import interface as IF

from . import compile as C, loop as L, pouw as PW


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
    ap.add_argument("--pouw", choices=["h1"], help="also the loop's PoUW window, its linears as the pinned H1 tile")
    ap.add_argument("--pouw-out")
    ap.add_argument("--pouw-salt", help="the H1 tiles' salt, 64 bytes in hex (default fresh)")
    a = ap.parse_args(argv)
    if bool(a.out) == a.circuit_only:
        ap.error("--out DIR, or --circuit-only")
    loop = L.TorchLoop(L.Model(h=a.h), [int(x) for x in a.requests.split(",")], a.steps, a.tamper, a.run_id)
    law = D.parse_law(a.law)
    if a.circuit_only:
        empty = IF.Window(loop.circuit(), loop.run_root, law, a.cap)
        out = {"circuit": hashlib.sha512(IF.circuit_bytes(empty.circuit_record())).hexdigest()}
        if a.pouw:
            h1 = PW.H1Loop(loop, bytes(PW.SALT_BYTES))
            h1w = IF.Window(h1.circuit(), loop.run_root, h1.law, PW.CAP)
            out["pouw"] = hashlib.sha512(IF.circuit_bytes(h1w.circuit_record())).hexdigest()
        print(json.dumps(out))
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
    pouw = None
    if a.pouw:
        h1 = PW.H1Loop(loop, bytes.fromhex(a.pouw_salt) if a.pouw_salt else PW.fresh_salt())
        pdir = Path(a.pouw_out or Path(a.out) / "pouw")
        prec = h1.write(h1.window(), pdir, {"loop": "torch-loop/pouw-h1"})
        pouw = {"circuit": json.loads((pdir / "provenance.json").read_text())["circuit"], "tiles": h1.real,
                "work": prec["window"]["public"]["work"], "law": prec["law"], "registration": R.digest(prec), "out": str(pdir)}
    print(json.dumps({"pouw": pouw, "circuit": rec["window"]["circuit"], "run": loop.run_root, "population": win.window.population,
                      "members": len(win.window.members), "mismatched": {t: len(v) for t, v in bad.items()},
                      "compiler": rec["window"].get("compiler"), "registration": R.digest(rec), "out": str(Path(a.out))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
