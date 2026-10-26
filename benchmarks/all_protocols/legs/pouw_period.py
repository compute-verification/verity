"""A rehearsal of one torch-loop period through core's credit (`credit.period_credit`), every piece real except the verifier
of record, which is a stand-in, and the gate, which is infra's run on this node under a test key. Run by hand; never
collected.

    cd integrations/vllm && PYTHONPATH=$ROOT:$ROOT/integrations/vllm:$ROOT/integrations/torch_loop:$ROOT/benchmarks/all_protocols/legs \\
      python -m pouw_period --out OUT --tree TGZ --commit SHA --main MAIN_TXT

* **The gate** is infra's (`local_gate.py`: `benchmarks/live_challenger/auditor.py` of the ``--tree`` archive, ``git
  archive SHA`` of `local_gate.ARCHIVE`'s paths), its coins stub bytes (``--coins stub``: no Lean build on this node),
  its key a test key. Its ``period-open`` must take a ``live`` circuit file (`auditor.TEMPLATE`), which credit needs
  and main's gate refuses until #4320 lands, so SHA is a tree with #4320's template. It registers the loop's committed checkpoint (`verity_torch_loop.checkpoint`'s files, role
  `checkpoint.ROLE`) as vy-admin does (``admin model-register``), under the period's
  circuit, the SHA-512 of the loop's declared H1 window Program (`pouw_h1_served.declared` over the loop's two linears at
  T = ``--rows``, S = ``--steps``); opens the period with ``--model TORCH_LOOP_MLP_H128_W8A8``, the draw K the proof
  service sizes (`pouw_h1_challenger.draw_k`), the circuit files and the fixed length; and reserves the window's session.
* **The window** is served through vLLM's H1 option on the loop's own model (`verity_torch_loop.loop.Model`, its two
  W8A8 linears on vLLM's int8 arm, each step's rows the loop's requests), its salt, round and draw coins the gate's
  session's, its ``b/<K>`` salted by the registration's key (``model-key``), the Commit tapped at each linear's
  row_quant, and each call's receipts `h1tile.reference`'s on CPU. Every served y is checked against the loop's own
  `loop.linear`, bit for bit (``same_outputs``).
* **The verifier of record is a stand-in** (``STAND-IN``): `pouw_h1_credit.StandIn` for each drawn call (each drawn
  unit's pub, o0 and o1 rows opened from the prover's tables against the roots of credit's ``--registered`` file, and
  its receipt recomputed from its A and B rows), and a stand-in accepting each round's ``live@<s>`` statement. Their
  circuit files are stand-ins (`STAND_IN_FILES`), which the period's open commits. At the real run the gate is
  vy-challenger-1's, its registry root vy-admin's registration, and ``run`` C-Flock's Lean ``flock-verify`` on each
  drawn statement's proof files, against the circuit files `pouw_h1_circuits.build` lowers.

The record is ``OUT/period.json``: the period's credit (``period_credit``: windows, work, k, δ, ε, or the refusal), the
window's verdicts on the way, its draw's ε at δ = 1/100 and 10⁻⁶ (``window_eps``, `window.eps_at` over the window's
units, padded slots included: a window whose requests fill few of its T rows establishes none), ε(j) of
`CircuitWorkPeriod` at those δ for the credited windows, and two negatives the period must refuse (``negatives``): a
drawn unit's ok row opened as 0 (`pouw_h1_credit.tampered`), and a verifier whose first-parent history of main lacks
the gate's pin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.pouw.circuit import live as LV
from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
from verity_torch_loop import checkpoint as CK
from verity_torch_loop import loop as L
from verity_vllm import protocols as PO
from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_ops
from verity_vllm.protocols import pouw
from verity_vllm.protocols.pouw import h1 as PH
from verity_vllm.protocols.pouw.h1 import challenger as PHC
from verity_vllm.protocols.pouw.h1 import commit as CT
from verity_vllm.protocols.pouw.h1 import served as SV
from verity_vllm.protocols.pouw.h1 import window as PHW

from tests.protocols import pouw_h1_credit as PCR
from tests.protocols import pouw_h1_gpu as GPU
from tests.protocols import pouw_h1_rehearsal as RH

ENTRY = f"pouw:{PH.SCHEME}"
STAND_IN = "stand-in verifier of record (pouw_h1_credit.StandIn; live@<s> accepted unseen)"
LIVE_FILE = b"stand-in circuit file of the live template\n"
QUOTE_DELTAS = ("1/100", "1/1000000")


def stand_in_files(ks) -> dict[str, str]:
    """The circuit files' SHA-512 the period's open commits: `pouw_h1_credit.statements`' stand-in of each K's, and the
    live template's."""
    out = {f"k{k}": hashlib.sha512(f"stand-in circuit file of K = {k}'s template\n".encode()).hexdigest() for k in ks}
    return {**out, CR.LIVE: hashlib.sha512(LIVE_FILE).hexdigest()}


def linears(m: L.Model) -> list[str]:
    return [m.op_path(i) for i in range(len(m.layers))]


def checkpoint(directory: Path) -> None:
    """The role's committed files copied into `directory`, which the registry takes only at 0755 with its files 0644."""
    directory.mkdir(mode=0o755)
    for n in json.loads(CK.MANIFEST.read_text())["checkpoints"][0]["files"]:
        shutil.copyfile(CK.CHECKPOINT / n, directory / n)
        (directory / n).chmod(0o644)


class Arm:
    """vLLM's int8 arm under H1 (`engine.int8_vllm.Int8PlainLinearMethod`): with ``zy_rows``, ``linear_zy`` over that many
    slots, its records handed to ``observe`` as z."""

    int8_arm = E.ARM
    zy_rows: int | None = None

    def apply(self, layer, x, bias=None):
        xq, s_a, y, zy = int8_ops.linear_zy(x, layer.weight, layer.weight_scale, self.zy_rows)
        self.observe(layer, xq, s_a, y, zy)
        return y

    def observe(self, layer, xq, s_a, y, z=None):
        pass


class Linear:
    def __init__(self, layer: L.Layer):
        self.weight, self.weight_scale, self.quant_method = layer.wq, layer.sw.reshape(-1, 1), Arm()


class Model:
    """The loop's model as vLLM's option sees it: its linears under their module names."""

    def __init__(self, m: L.Model):
        self.layers = [Linear(layer) for layer in m.layers]
        self.names = linears(m)

    def modules(self):
        return iter(self.layers)

    def named_modules(self):
        return iter(zip(self.names, self.layers))


def eps_j(j: int, delta: Fraction, M: int = 2, r: float = 2.87, kd: int = 8, p: Fraction = Fraction(7, 8),
          p0: Fraction = Fraction(1, 2), a: float = 1, b: float = 1, u: float = 1) -> float | None:
    """ε(j) of `CircuitWorkPeriod` (Security/Proofs/Pouw/ComposePeriod.lean) at δ for j credited windows of M rounds:
    Λ·ln(1/δ)/(u·j), Λ = max(a/(M·−ln(p + (1 − p)e^{−x})), b/(M·−ln(p0 + (1 − p0)e^{−x}))), x = K_d/(r·M). r is SERVE-H1's,
    assumed for the loop; D₀ = 0. None for no window."""
    if j == 0:
        return None
    x = kd / (r * M)
    lam = max(a / (M * -math.log(float(p) + (1 - float(p)) * math.exp(-x))),
              b / (M * -math.log(float(p0) + (1 - float(p0)) * math.exp(-x))))
    return lam * math.log(1 / float(delta)) / (u * j)


def serve(m: L.Model, requests: list[int], steps: int, rows: int, served: SV.Served, settings: dict, box: dict) -> dict:
    """The loop's requests served through vLLM's H1 option, one engine step a loop step; each served y the loop's own.
    ``box["run"]`` is the window's run from its install on."""
    runner = SimpleNamespace(execute_model=lambda scheduler_output, *a, **kw: None, decode_query_len=1)
    model = Model(m)
    ctx = PO.Context(stage="serve", enabled=(ENTRY,),
                     settings={"pouw": {**settings, "rows": rows, "steps": steps, "retain_steps": steps, "runner": runner,
                                        "served": served}})
    served.tap = CT.CommitTap("embed")
    comp = PO.install(model, ctx)
    box["run"] = pouw.h1_run(ENTRY)
    same, x = [], m.embed[torch.tensor(requests)].clone()
    ids = [int(t) for t in requests]
    try:
        for s in range(steps):
            reqs = [SimpleNamespace(req_id=f"r{i}", prompt_token_ids=[t], sampling_params=SimpleNamespace(max_tokens=steps))
                    for i, t in enumerate(requests)] if s == 0 else []
            runner.execute_model(SimpleNamespace(scheduled_new_reqs=reqs, num_scheduled_tokens={f"r{i}": 1 for i in range(len(requests))},
                                                 total_num_scheduled_tokens=len(requests)))
            served.tap.take("embed/in", torch.tensor(ids), s)
            for w, (name, layer, lin) in enumerate(zip(model.names, model.layers, m.layers)):
                xq, s_a = int8_ops.row_quant(x)
                served.tap.take(f"{CT.site(name)}/0", xq, s)
                served.tap.take(f"{CT.site(name)}/1", s_a, s)
                y = layer.quant_method.apply(layer, x)
                lq, lsx = L.quantize(x)
                same.append(bool(torch.equal(y.view(torch.int16), L.linear(lq, lsx, lin.wq, lin.sw).view(torch.int16))))
                x = torch.relu(y.to(torch.float32)) if w == 0 else y.to(torch.float32)
        box["run"].close()
    finally:
        comp.close()
    return {"run": box["run"], "same_outputs": same}


def rehearse(a: argparse.Namespace) -> dict[str, Any]:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = L.Model()
    requests = [int(t) for t in a.requests.split(",")] if a.requests else [i % L.VOCAB for i in range(a.rows)]
    doc: dict[str, Any] = {"verifier_of_record": STAND_IN, "requests": requests, "steps": a.steps, "rows": a.rows}
    names = linears(m)
    shapes, modules = [(m.h, m.h)] * len(names), dict(enumerate(names))
    program = SV.declared(shapes, modules, a.rows, a.steps)
    desc = SV.descriptor(program)
    circuit = hashlib.sha512(desc).hexdigest()
    layout = SV.declared_layout(shapes, modules, a.rows, a.steps)
    (out / "window-program.json").write_bytes(desc)
    draw_k = PHC.draw_k(layout)
    files = stand_in_files(sorted({k for _n, k, _w in layout["linears"]}))
    length = {"max_tokens": [a.steps] * len(requests), "stop": "fixed"}
    doc.update(circuit=circuit, layout=layout, draw_k=draw_k, circuit_files=files, length=length)

    gdir = Path(tempfile.mkdtemp(prefix="loop-gate-")) / "gate"
    gdir.mkdir()
    gate = RH.Gate([sys.executable, "-I", str(RH.LOCAL_GATE), str(gdir)])
    rc, init = gate.raw("init", "--tree", a.tree, "--commit", a.commit, "--peer", "local", "--coins", "stub")
    if rc:
        doc["refused"] = f"gate init: {init[-2000:].decode(errors='replace')}"
        return doc
    doc["gate"] = json.loads(init)
    pin, lo = doc["gate"]["pubkey"]["public_sha512"], doc["gate"]["ports"][0]
    ck = out / "checkpoint"
    checkpoint(ck)
    doc["model_register"] = gate("admin", "model-register", "--model", CK.ROLE, "--circuit", circuit, "--descriptor",
                                 str(out / "window-program.json"), "--checkpoint", str(ck))
    if "refused" in doc["model_register"]:
        return {**doc, "gate_log": gate.log}
    doc["period_open"] = gate("period-open", "--name", a.period, "--cap", "1", "--circuit", circuit, "--draw-k",
                              str(draw_k), "--rows", str(a.rows), "--model", CK.ROLE,
                              "--circuit-files", json.dumps(files, sort_keys=True, separators=(",", ":")),
                              "--length", json.dumps(length, sort_keys=True, separators=(",", ":")))
    if "refused" in doc["period_open"]:
        return {**doc, "gate_log": gate.log}
    g = GPU.gate_command(gate.argv)
    key = GPU.model_key_of(g, CK.ROLE, circuit)
    reserved = PHW.reserved_of({k: v for k, v in g.reserve("local", a.steps).items() if k in ("session", *PHW.RESERVE_FIELDS)})
    sid = reserved["session"]
    doc["reserved"] = sid

    held: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    words: dict[int, np.ndarray] = {}
    box: dict[str, Any] = {}
    served = SV.Served(registry=key, program=program, program_digest=circuit)

    def receipts(call, codes, s_a, w, s_w):
        u0, n = HW.call_units(served.program)[call.index]
        pre = served.program.fn.bindings["PREFIX"]
        roots = (served.registered[served.a_table(call.index)].root, served.registered[pre + f"b/{call.k}"].root)
        r, c = served.tile
        A = np.zeros((served.m, call.k), np.int8)
        A[:len(codes)] = codes
        held[call.index] = (A, np.asarray(w, np.int8))
        ncb = call.n // c
        got = np.array([HT.reference(A[r * (u // ncb):r * (u // ncb + 1)].view(np.uint8).tolist(),
                                     held[call.index][1][c * (u % ncb):c * (u % ncb + 1)].view(np.uint8).tolist(),
                                     box["run"].salt, u0 + u, roots, pairs=[(0, 0)])[1] for u in range(n)], np.uint32)
        words[call.index] = SV.receipt_words(got)
        return got

    served.receipts = receipts
    settings = {"challenger": f"127.0.0.1:{lo}", "session": reserved, "gate": g}
    pouw.linear_types = lambda: (Linear,)
    got = serve(m, requests, a.steps, a.rows, served, settings, box)
    run = got["run"]
    doc["same_outputs"] = got["same_outputs"]
    doc["served_refused"] = served.refused
    rec = {"commitment": run.commits, "calls": run.manifest()}
    (out / "calls.json").write_text(json.dumps(rec) + "\n")
    doc["unit"] = gate("wait", sid)
    doc["period_close"] = gate("period-close", "--name", a.period)
    doc["evidence"] = RH.evidence(gate, sid, out / "gate")

    gd = out / "gate"
    d, ledger, pins = gd / "fetch" / sid / "out", [gd / "windows.jsonl"], [gd / "pins.jsonl"]
    main = remote.main_lines(a.main)
    pkey = PHC.pinned_key(d / remote.KEY, pin)
    window = {"root": run.commits["session"]["root"]}
    w = PCR.of_run(run, served)
    operands = lambda c: held[c]  # noqa: E731
    stand_in = PCR.StandIn(w, operands)

    def verify(proof, *, circuit_sha512, program, partition, registered):
        if circuit_sha512 == files[CR.LIVE]:
            s = int(registered["ports"]["m"]["value"].rsplit("@", 1)[1])
            found = {port: {"value": e["value"], "root": e["root"]} for port, e in registered["ports"].items()}
            return [PF.Finding(str(x), True, "", hashlib.sha512(program).hexdigest(), found, (s,))
                    for x in proof.sessions], None
        return stand_in(proof, circuit_sha512=circuit_sha512, program=program, partition=partition,
                        registered=registered)

    first = PHC.credit_reason(rec, d, ledger, window, pkey, desc, pins=pins, main=main)
    doc["before_statements"] = {k: first.get(k) for k in ("manifest", "window", "k", "drawn")}
    drawn = first.get("drawn") or []
    sdir = out / "statements"
    st, made = PCR.statements(w, drawn, sdir)
    assert all(files[k] == v for k, v in made.items()), (made, files)
    live = {"descriptor": SV.descriptor(LV.program_v2(a.rows, a.steps, SV.PREFIX)).hex(), "statements": []}
    for s in range(a.steps):
        ld = sdir / f"live{s}"
        (ld / "s0").mkdir(parents=True)
        (ld / "s0" / "session.json").write_text("{}")
        (ld / "circuit.txt").write_bytes(LIVE_FILE)
        (ld / "public.json").write_text("{}")
        one = PHC.statement(s, ld / "circuit.txt", ld / "public.json", [ld / "s0"])
        live["statements"].append({**{x: v for x, v in one.items() if x != "call"}, "round": s})
    win = PHC.credit_reason(rec, d, ledger, window, pkey, desc, st, run=verify, pins=pins, main=main, live=live)
    doc["window_credit"] = {k: win.get(k) for k in ("manifest", "window", "k", "drawn", "calls", "credit", "eps")}
    doc["stand_in_rejected"] = stand_in.rejected
    led = PHC.Ledger.load(ledger, pkey, pins=pins, main=main)
    evidence = PHC.window_evidence(rec, d, window, st, desc, live)
    credit, why = PHC.period_credit(led, a.period, [evidence], run=verify)
    doc["period_credit"] = {"credit": credit, "refused": why}
    neg = {}
    if drawn:
        bad = PCR.tampered(w, "ok-zero", drawn[0], lambda c: words[c], run._key)
        bad_in = PCR.StandIn(bad, operands)
        bst, _ = PCR.statements(bad, drawn, out / "statements-ok-zero")

        def bad_verify(proof, **kw):
            return verify(proof, **kw) if kw["circuit_sha512"] == files[CR.LIVE] else bad_in(proof, **kw)
        neg["ok-zero"] = PHC.period_credit(led, a.period, [PHC.window_evidence(rec, d, window, bst, desc, live)],
                                           run=bad_verify)[1]
    off = PHC.Ledger.load(ledger, pkey, pins=pins, main=main[1:])
    neg["gate-off-main"] = PHC.period_credit(off, a.period, [evidence], run=verify)[1]
    doc["negatives"] = {k: {"refused": v, "ok": v is not None} for k, v in neg.items()}
    population = sum(size * work for size, work in CR.call_work(layout))
    doc["window_eps"] = {}
    for dl in QUOTE_DELTAS:
        try:
            doc["window_eps"][dl] = str(Wd.eps_at(draw_k, Fraction(dl), credit["work"][0], population, 1))
        except (ValueError, TypeError, IndexError):
            doc["window_eps"][dl] = None
    j = len((credit or {}).get("windows") or [])
    doc["eps_j"] = {dl: eps_j(j, Fraction(dl)) for dl in QUOTE_DELTAS}
    doc["eps_j_note"] = ("CircuitWorkPeriod's ε(j) for j credited windows of M = 2 rounds: SERVE-H1's r = 2.87 assumed, "
                         "K_d = 8, p = 7/8, p0 = 1/2, a = b = 1, D₀ = 0, u = 1; ≥ 1 is vacuous")
    doc["gate_log"] = gate.log
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tree", required=True, help="git archive SHA of local_gate.ARCHIVE's paths, gzipped")
    ap.add_argument("--commit", required=True, help="the archive's commit, the gate's pin")
    ap.add_argument("--main", required=True, help="the verifier's first-parent history (git rev-list --first-parent)")
    ap.add_argument("--requests", help="the requests' token ids, comma-separated (default: T of them, filling the rows)")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--rows", type=int, default=32, help="H1's T, a multiple of 32")
    ap.add_argument("--period", default="loop-p0")
    a = ap.parse_args(argv)
    doc = rehearse(a)
    (Path(a.out) / "period.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    pc = doc.get("period_credit") or {}
    print(json.dumps({"period_credit": pc, "window_eps": doc.get("window_eps"), "eps_j": doc.get("eps_j"), "same_outputs": doc.get("same_outputs"),
                      "refused": doc.get("refused"), "stand_in_rejected": doc.get("stand_in_rejected"),
                      "negatives": doc.get("negatives")}, default=str))
    ok = pc.get("credit") and all(v["ok"] for v in (doc.get("negatives") or {}).values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
