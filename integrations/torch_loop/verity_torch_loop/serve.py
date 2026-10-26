"""The torch loop's window, written for the audit from its records alone (`benchmarks/prove_path/records_audit.py`).

    python -m verity_torch_loop.serve --out DIR [--law L] [--requests 3,7] [--steps 2] [--h 128] [--cap 8] [--run-id ID] \\
        [--tamper] [--opening FILE] [--pouw h1 [--pouw-out DIR] [--pouw-salt HEX | --pouw-live HOST [--pouw-epoch K]]]
    python -m verity_torch_loop.serve --circuit-only [the same shape and --run-id] [--pouw h1]

The first serves the requests (`loop.TorchLoop`), commits every unit and its activations (`TorchLoop.commit_activations`,
their entries in the registration's window and their salts in `ACTIVATIONS_PRIVATE`) and writes the window
(``circuit.json``, the registration under core's ``one_stage.registration``, the value files) to ``--out``, and prints the circuit record's
SHA-512, the run root, the population and the units whose outputs are not their Definition's (`Window.mismatches`: none
unless ``--tamper``). With ``--opening FILE`` the circuit is its compiler's (`compile`): the loop's schedule is committed as
advice under that compiler, the registration binds both, and the advice's opening (the prover's private file) goes to
``FILE``. The second prints the circuit record's SHA-512 alone, from the loop's shape and no value: the circuit the verifier
registers before the loop runs, when it has no compiler. With ``--pouw h1`` the first also writes the loop's PoUW
window (`pouw`: its linears as the pinned H1 tile) to ``--pouw-out`` (default ``DIR/pouw``), its tiles salted by
``--pouw-salt`` (64 bytes in hex; fresh when absent), and the second prints that window's circuit record's SHA-512
beside the first's. ``--pouw-live HOST`` (with ``--registered --weights``): the salt is a live session's instead, opened
after the tiles' operands are committed, on R = `salt.anchored_epoch_root(W, X, k)` (W the pinned weights' root, X the
operands', k ``--pouw-epoch``), by a development gate on this machine whose auditor host's root is the scratch
directory ``HOST`` (`window_salt.local`, pinned on this tree's commit); its answer and the verifier's ``live`` params go
to ``salt.json`` beside the window. ``--pouw-late``: the session is opened on other operands' X (a prover that takes the
salt first and commits its operands after it), so PoUW's check refuses the window's salt.

    python -m verity_torch_loop.serve --commit-weights OPENING --pin PIN [--h 128]

commits the model's weights once, before any run (`loop.commit_weights`): the prover's opening (their salts) to
``OPENING``, the verifier's pin of their roots to ``PIN``. ``--registered`` (with ``--out`` or ``--circuit-only``): the
circuit reads the weights as registered values (`loop.TorchLoop`'s ``registered``), served from ``--weights OPENING``;
``--tamper-weights``: served from other weights under that opening's salts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from verity.core.protocols.pouw import operands as OP
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity.ml.windows import interface as IF

from . import compile as C, loop as L, pouw as PW

#: the run's committed activations' salts (`loop.activations_opening`), the prover's, beside the members' private bodies
ACTIVATIONS_PRIVATE = "activations-private.json"


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
    ap.add_argument("--pouw-live", metavar="HOST", help="the H1 tiles' salt from a live session of a local gate rooted here")
    ap.add_argument("--pouw-epoch", type=int, default=0, help="with --pouw-live: the salt epoch's number k")
    ap.add_argument("--pouw-late", action="store_true", help="with --pouw-live: open the session on other operands' X")
    ap.add_argument("--registered", action="store_true", help="the circuit reads the weights as registered values")
    ap.add_argument("--weights", help="with --registered --out: the weights' opening (--commit-weights)")
    ap.add_argument("--tamper-weights", action="store_true")
    ap.add_argument("--commit-weights", help="commit the weights: their opening (the prover's) to this file")
    ap.add_argument("--pin", help="with --commit-weights: the verifier's pin of their roots to this file")
    a = ap.parse_args(argv)
    if a.commit_weights:
        if not a.pin:
            ap.error("--commit-weights OPENING --pin PIN")
        model = L.Model(h=a.h)
        values = L.commit_weights(model)
        Path(a.commit_weights).write_text(json.dumps(L.opening(model, values)))
        Path(a.pin).write_text(json.dumps(L.pin(model, values), indent=1, sort_keys=True) + "\n")
        print(json.dumps({v: e["root"] for v, e in L.pin(model, values)["values"].items()}))
        return 0
    if bool(a.out) == a.circuit_only:
        ap.error("--out DIR, or --circuit-only")
    if a.registered and a.out and not a.weights:
        ap.error("--registered --out needs --weights OPENING")
    if a.pouw_live and not (a.pouw and a.registered and a.out and not a.pouw_salt):
        ap.error("--pouw-live needs --pouw h1 --registered --weights OPENING --out, and no --pouw-salt")
    if a.pouw_late and not a.pouw_live:
        ap.error("--pouw-late needs --pouw-live")
    loop = L.TorchLoop(L.Model(h=a.h), [int(x) for x in a.requests.split(",")], a.steps, a.tamper, a.run_id,
                       a.registered, a.tamper_weights)
    law = D.parse_law(a.law)
    if a.circuit_only:
        empty = IF.Window(loop.circuit(), loop.run_root, law, a.cap)
        out = {"circuit": hashlib.sha512(IF.circuit_bytes(empty.circuit_record())).hexdigest()}
        if a.pouw:
            h1 = PW.H1Loop(loop)
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
    weights = L.opened(loop.model, json.loads(Path(a.weights).read_text())) if a.registered else None
    win = loop.window(law, a.cap, advice, weights)
    bad = win.mismatches()
    acts = loop.commit_activations()
    win.write(win.close(), a.out, {"loop": "torch-loop", "activations": L.activations(loop.model, acts)}, registrar=R)
    fd = os.open(Path(a.out) / ACTIVATIONS_PRIVATE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(L.activations_opening(loop.model, acts), f)
    rec = json.loads((Path(a.out) / "registration.json").read_text())
    pouw = None
    if a.pouw:
        h1 = PW.H1Loop(loop, weights=PW.weights_group(loop.model, weights) if weights else None, live=bool(a.pouw_live))
        pdir = Path(a.pouw_out or Path(a.out) / "pouw")
        issued = None
        if a.pouw_live:
            WS = PW.window_salt()
            main_line = [WS.a5_coins().on_main([])]
            anchors = h1.anchors()
            if a.pouw_late:
                width = {v: e["words"] for v, e in h1.operands["values"].items()}
                early = OP.block(h1.program, OP.commit(h1.program, loop.run_root, {v: [[0] * n] for v, n in width.items()}))
                anchors = {**anchors, "window": {**anchors["window"], OP.KEY: early}}
            issued = WS.issue(WS.local(Path(a.pouw_live), main_line), anchors, a.pouw_epoch, main_line, pdir / "salt-session")
            salt = issued.salt
        else:
            salt = bytes.fromhex(a.pouw_salt) if a.pouw_salt else PW.fresh_salt()
        prec = h1.write(h1.window(salt), pdir, {"loop": "torch-loop/pouw-h1"}, issued=issued)
        pouw = {"circuit": json.loads((pdir / "provenance.json").read_text())["circuit"], "tiles": h1.real,
                "work": prec["window"]["public"]["work"], "law": prec["law"], "registration": R.digest(prec), "out": str(pdir),
                "X": h1.X.hex(), "live": issued is not None, "late": a.pouw_late}
    print(json.dumps({"pouw": pouw, "circuit": rec["window"]["circuit"], "run": loop.run_root, "population": win.window.population,
                      "members": len(win.window.members), "mismatched": {t: len(v) for t, v in bad.items()},
                      "compiler": rec["window"].get("compiler"), "registration": R.digest(rec), "out": str(Path(a.out)),
                      "activations": {v: e["root"] for v, e in rec["window"]["activations"]["values"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
