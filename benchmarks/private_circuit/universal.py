"""Protocol 2's universal unit end to end: a random hidden netlist, padded to its size class, run by the public
`UniversalUnit_v1{G, N_IN, N_OUT}` under Flock's circuit prover (M0) with `--zk`, and verified by the Lean verifier.

    python benchmarks/private_circuit/universal.py --cls 16,16,8 --n 64 --bin flock-circuit --lean flock-verify --out DIR

* **The program.** A netlist of 3/4 of the class's gates over 3/4 of its inputs, every index in range (`_netlist`), padded to
  the class with no-op gates and unread inputs (`universal.pad`). It is a value: the Call's `op`, `src` and `out` parameters.
* **The statement.** A one-Call Program of the unit, partitioned by `Q_template_instance` (the Call is one unit), each instance
  one run of the netlist on its own random `x`. The unit's class is a `verity/flock-circuit` statement (`class_statement.stage`,
  words packed), one row per source: `op`, `src` and `out` are committed once and opened by every instance (the registered
  program), `x` once per instance; the outputs are hidden rows.
* **The checks.** Before staging, every instance's outputs agree four ways:   the netlist interpreted (`universal.interpret`),
  the hidden program as its own Boolean Definition (`direct`) and the unit, each under `verity.ml.circuits.evaluation` (`evaluate` on the
  first instance, `evaluate_batch` on all), and the flat Definition's transcript that staging lays out (which `class_statement.stage`
  checks against the lowered circuit). The live `serve --zk` / `prove --zk` session must be accepted, then each recorded session
  is replayed by Lean's `flock-verify verify --zk`, which must accept it, and must refuse it without `--zk` and with a proof byte
  flipped.
* **The row.** `rows.jsonl`: the class, the unit's ANDs (traced, closed form and Flock's lowering), ANDs per hidden gate,
  the direct netlist's ANDs, the statement's size (k_log, instances per block), prover time, proof bytes, the live and the Lean
  verifier's times.
* **`--wired`: several units, public topology.** A circuit of four units (`wired_topology`: layer 1 reads layer 0's outputs,
  layer 2 layers 0 and 1, layer 3 layer 2, beside the circuit's input bits), a random hidden netlist per layer, on `n`
  instances: a Program of one `batch` of the unit per layer (`wired_program`), whose wiring is the public layout's. Checked
  against `interpret` composed along the topology and `verity.ml.circuits.evaluation` of the Program; `Q_template_instance` makes every
  member one unit, and the topology the partition derives (`derived_topology`) must be the one built. One class statement of
  `4 n` instances staged with `class_statement.Wiring` (PROTOCOL.md §16.13, Linking: each cut source a link row under its
  writer's salt, META `links` and `definition`), proved live with the verifier's own `--links` file (`class_statement.links_file`)
  and Lean-verified with `--program`, which derives the units and checks every cut wire (`without_program` records the
  refusal without it). Then two cheating provers on the same circuit: one whose layer 1 reads a made-up input in place of
  layer 0's outputs, every unit correct on what it is given, and one whose link rows hold the right values under salts of
  their own. Each must be refused by the live verifier with its links before serving, and by Lean with its program on the
  sessions a live verifier given an empty links list records. Before it, the one-unit statement at `4 n` instances, the
  cost comparison. Linking binds a public topology; it does not hide the topology: the Program is the verifiers', so the
  wiring and the descriptor are public, a leak beyond the size class.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array
from verity.ml.circuits.boolean import trace as TR
from verity.ml.private_circuits import universal as U
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import partition_units as PU

SPEC = "q-template-instance"


def _netlist(rng: random.Random, G: int, N_IN: int, N_OUT: int, n_in: int | None = None) -> U.Netlist:
    g0, n0 = max(1, 3 * G // 4), n_in or max(1, 3 * N_IN // 4)
    gates = tuple((rng.randrange(4), rng.randrange(n0 + g), rng.randrange(n0 + g)) for g in range(g0))
    return U.Netlist(n0, gates, tuple(rng.randrange(n0 + g0) for _ in range(N_OUT)))


def direct(net: U.Netlist, name: str):
    """The hidden netlist as its own Boolean Definition (unregistered): the program run directly."""
    def build(C, x):
        w = list(x)
        for op, a, b in net.gates:
            s = w[a] ^ w[b]
            v = C.AND(w[a], w[b]) if op & 1 else s
            w.append(v.inv() if op >> 1 else v)
        return [w[o] for o in net.outputs]

    return TR.definition(name, 1, build, (("x", Array(net.n_in, U.BIT)),), Array(len(net.outputs), U.BIT),
                         register=False)


def lanes(units, cl, net: U.Netlist, xs: list[list[int]]):
    """`class_statement.class_lanes`'s (inputs, outputs, row identities) for the given netlist and inputs: the program's
    sources are one row for every instance, x one row per instance."""
    act = units.activation(cl.definition)
    uc = cl.representative
    leaf_param = []
    for i, (_n, t) in enumerate(act.fn.signature()[0]):
        leaf_param += [i] * t.leaves
    x_param = len(act.fn.signature()[0]) - 1
    groups = CS.sources(act, uc, CS.owners_of(units, cl.definition.id))
    ins = [[] for _ in uc.inputs]
    want = [[] for _ in uc.outputs]
    rows = [[] for _ in groups]
    for t, x in enumerate(xs):
        words = [b for part in U.arguments(net, x) for b in part]
        run = PU.evaluate(act, words)
        for port, k in zip(ins, uc.inputs):
            port.append(run[k])
        for port, j in zip(want, uc.outputs):
            port.append(run[j])
        for ids, g in zip(rows, groups):
            params = {leaf_param[-1 - uc.inputs[i]] for i in g}
            ids.append(("x", t) if x_param in params else ("program",))
    return ins, want, rows


def _sessions(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_dir() and (p / "session.json").exists()) if d.exists() else []


def lean_verify(lean: str, circuit: Path, pub: Path, partition: Path, sessions: list[Path], zk: bool = True,
                program: Path | None = None) -> tuple[list[dict], float, str]:
    """``flock-verify verify`` of the sessions; ``program``: the program's descriptor, so the verifier derives the units itself."""
    cmd = [lean, "verify", "--circuit", str(circuit), "--public", str(pub), "--partition", str(partition),
           *(["--program", str(program)] if program else []), *(["--zk"] if zk else []), "--session", *map(str, sessions)]
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    dt = time.monotonic() - t0
    v = [json.loads(ln.split(" ", 1)[1]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
    return v, dt, (p.stderr or "")[-600:]


def live_session(a, stage: Path, n: int, d: Path, pin: str, warm: int, runs: int, links: Path | None = None) -> tuple[dict, list[Path]]:
    """``serve --zk`` on 127.0.0.1 (``links``: the verifier's ``--links`` file) and ``prove --zk`` against it, ``warm + runs``
    sessions recorded under ``d/sessions``: the median timed run's LIVE record, and the recorded session directories. A verifier
    that refuses the statement before serving is the record's ``why`` (its REFUSED line), with no session."""
    d.mkdir(parents=True, exist_ok=True)
    sess = d / "sessions"
    log = d / "serve.log"
    srv = subprocess.Popen([a.bin, "serve", "--zk", "--listen", f"127.0.0.1:{a.port}", "--out", str(sess), "--instances",
                            str(stage / f"pub-{n}.bin"), "--circuit", str(stage / "circuit.txt"), "--pin", pin, "--operator", "pc-flock",
                            "--sessions", str(warm + runs), *(["--links", str(links)] if links else [])],
                           stdout=open(log, "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(3000):
            if "SERVING" in log.read_text() or srv.poll() is not None:
                break
            time.sleep(0.2)
        if "SERVING" not in log.read_text():
            srv.wait(timeout=60)
            why = next((json.loads(ln.split("\t", 1)[1])["why"] for ln in log.read_text().splitlines() if ln.startswith("REFUSED\t")),
                       log.read_text()[-400:])
            return {"exit": srv.returncode, "sessions": 0, "accepted": False, "refused_before_serving": True, "why": why}, []
        with open(d / "prove.out", "w") as fo, open(d / "prove.err", "w") as fe:
            timer = ["/usr/bin/time", "-f", "%M", "-o", str(d / "prove.maxrss")] if os.path.exists("/usr/bin/time") else []
            pr = subprocess.run(timer + [a.bin, "prove", "--zk", "--verifier", f"127.0.0.1:{a.port}", "--instances", str(stage / f"inst-{n}.bin"),
                                         "--circuit", str(stage / "circuit.txt"), "--warm", str(warm), "--runs", str(runs)],
                                stdout=fo, stderr=fe, check=False)
        srv.wait(timeout=600)
    finally:
        if srv.poll() is None:
            srv.kill()
            srv.wait()
    live = [json.loads(ln[5:]) for ln in (d / "prove.out").read_text().splitlines() if ln.startswith("LIVE")]
    timed = [x for x in live if not x.get("warm")] or live
    med = sorted(timed, key=lambda x: x.get("prove_total_s") or 0)[len(timed) // 2] if timed else {}
    rec = {"exit": pr.returncode, "sessions": len(live), "accepted": bool(live) and all(x.get("accepted") is True for x in live)
           and pr.returncode == 0 and len(live) == warm + runs, "why": med.get("why"),
           "prove_total_s": med.get("prove_total_s"), "prove_total_s_runs": [x.get("prove_total_s") for x in timed],
           "e2e_s": med.get("e2e_s"), "witness_s": med.get("witness_s"), "proof_bytes": med.get("proof_bytes"),
           "verify_s": (med.get("verdict") or {}).get("verify_s"), "k_log": med.get("k_log"), "prover": med.get("prover"),
           "statement_digest": med.get("statement_digest"),
           "prover_peak_gb": round(int((d / "prove.maxrss").read_text().split()[-1]) / 2**20, 3) if (d / "prove.maxrss").exists() else None}
    if not live:
        rec["stderr_tail"] = (d / "prove.err").read_text()[-800:]
    recorded = _sessions(sess)
    rec["recorded_sessions"] = len(recorded)
    return rec, recorded


def run_class(a, G: int, N_IN: int, N_OUT: int, out: Path) -> dict:
    rng = random.Random(f"pc-flock/{a.seed}/{G}/{N_IN}/{N_OUT}")
    fn = bind(U.UniversalUnit, G=G, N_IN=N_IN, N_OUT=N_OUT)
    small = _netlist(rng, G, N_IN, N_OUT)
    net = U.pad(small, G, N_IN)
    hidden = direct(small, f"HiddenNetlistG{G}N{N_IN}O{N_OUT}S{a.seed}")
    xs_small = [[rng.randrange(2) for _ in range(small.n_in)] for _ in range(a.n)]
    xs = [x + [rng.randrange(2) for _ in range(N_IN - small.n_in)] for x in xs_small]
    want = [U.interpret(small, x) for x in xs_small]
    args = [U.arguments(net, x) for x in xs]
    cols = [np.array([r[i] for r in args], dtype=np.uint8) for i in range(4)]
    # the plain-integer reference on the first instance, the bit-sliced evaluator on all of them
    agree = {"interpret_padded": all(U.interpret(net, x) == w for x, w in zip(xs, want)),
             "direct_evaluate": list(E.evaluate(hidden, xs_small[0])) == want[0] and np.asarray(E.evaluate_batch(
                 hidden, np.array(xs_small, dtype=np.uint8))).reshape(a.n, N_OUT).tolist() == want,
             "unit_evaluate": list(E.evaluate(fn, *args[0])) == want[0] and np.asarray(E.evaluate_batch(
                 fn, *cols)).reshape(a.n, N_OUT).tolist() == want}

    program = PU.one_call(fn)
    query = PO.template_instance_query(fn)
    units = PU.PartitionUnits(program, query=query)
    [(shape, cl)] = units.classes().items()
    ins, outs, idents = lanes(units, cl, net, xs)
    agree["transcript"] = [[outs[j][t] for j in range(N_OUT)] for t in range(a.n)] == want
    if not all(agree.values()):
        raise AssertionError(f"UniversalUnit_v1{{{G},{N_IN},{N_OUT}}}: the outputs disagree before staging: {agree}")

    d = out / f"G{G}-N{N_IN}-O{N_OUT}"
    stage = d / "stage"
    shutil.rmtree(d, ignore_errors=True)
    t0 = time.monotonic()
    salt_key = secrets.token_bytes(32)
    st = CS.stage(units, SPEC, shape, cl, (ins, outs, idents), stage, salt_key, packed=True)
    stage_s = time.monotonic() - t0
    n = st["n"]
    if n != a.n:
        raise AssertionError(f"staging capped the instances at {n}, not {a.n}")
    part = stage / "partition.json"
    part.write_text(json.dumps(PO.build(program, query), sort_keys=True))
    text = (stage / "circuit.txt").read_text()
    pin = FC.digest(text)
    row = {"class": {"G": G, "N_IN": N_IN, "N_OUT": N_OUT, "W": N_IN + G, "K": U.index_bits(G, N_IN)},
           "netlist": {"gates": len(small.gates), "inputs": small.n_in, "outputs": len(small.outputs),
                       "ands": sum(1 for g in small.gates if g[0] & 1), "ands_direct": BS.compile(hidden).counts["and"],
                       "padded_gates": G - len(small.gates)},
           "unit": {"ands_closed_form": U.and_count(G, N_IN, N_OUT), "ands_traced": U.traced(G, N_IN, N_OUT).counts["and"],
                    "ands_lowered": st["ands"], "unit_rows": st["unit_rows"], "unit_nnz": st["unit_nnz"],
                    "ands_per_hidden_gate": round(st["ands"] / G, 2)},
           "statement": {k: st[k] for k in ("n", "k_log", "vus_per_block", "w_in", "w_out", "ports", "shared_rows", "circuit_sha512",
                                            "class_sha512", "circuit_bytes")} | {"shape": shape, "stage_s": round(stage_s, 3)},
           "agree_before_staging": agree, "seed": a.seed}
    if a.stage_only:
        return row

    row["live"], recorded = live_session(a, stage, n, d, pin, a.warm, a.runs)
    if a.lean and recorded:
        circ, pub = stage / "circuit.txt", stage / f"pub-{n}.bin"
        v, dt, err = lean_verify(a.lean, circ, pub, part, recorded)
        row["lean"] = {"accepted": len(v) == len(recorded) and all(x["accepted"] for x in v), "sessions": len(v),
                       "why": sorted({x["why"] for x in v}), "wall_s": round(dt, 3), "wall_s_per_session": round(dt / max(1, len(v)), 3),
                       **({"stderr_tail": err} if not v else {})}
        none = {"accepted": False, "why": "no verdict"}
        neg = {}
        nz = (lean_verify(a.lean, circ, pub, part, recorded[:1], zk=False)[0] or [none])[0]
        neg["without_zk"] = {"refused": not nz["accepted"], "why": nz["why"][:160]}
        bad = d / "tampered" / recorded[0].name
        shutil.copytree(recorded[0], bad)
        proof = max(bad.glob("*.bin"), key=lambda p: p.stat().st_size)
        b = bytearray(proof.read_bytes())
        b[len(b) // 2] ^= 1
        proof.write_bytes(bytes(b))
        fl = (lean_verify(a.lean, circ, pub, part, [bad])[0] or [none])[0]
        neg["proof_byte_flipped"] = {"refused": not fl["accepted"], "why": fl["why"][:160], "file": proof.name}
        # another hidden program of the class on the same x and salts: the same circuit byte for byte, and a public file that
        # differs only where the rows differ (the program's and the outputs'), against which the honest session must fail
        other = U.pad(_netlist(random.Random(f"pc-flock/other/{a.seed}/{G}/{N_IN}/{N_OUT}"), G, N_IN, N_OUT), G, N_IN)
        so = CS.stage(units, SPEC, shape, cl, lanes(units, cl, other, xs), d / "other-program", salt_key, packed=True)
        opub = d / "other-program" / f"pub-{n}.bin"
        op = (lean_verify(a.lean, circ, opub, part, recorded[:1])[0] or [none])[0]
        pa, pb = pub.read_bytes(), opub.read_bytes()
        neg["another_program"] = {"refused": not op["accepted"], "why": op["why"][:160],
                                  "same_circuit": so["circuit_sha512"] == st["circuit_sha512"],
                                  "pub_bytes_differing": sum(x != y for x, y in zip(pa, pb)) + abs(len(pa) - len(pb)), "pub_bytes": len(pa)}
        row["lean"]["negatives"] = neg
    if not a.keep_stage:
        # the circuit is restaged byte for byte from the code; the prover's file holds the rows and salts
        for f in [stage / "circuit.txt", *stage.glob("inst-*.bin")]:
            f.unlink(missing_ok=True)
        for f in [d / "other-program" / "circuit.txt", *d.glob("other-program/inst-*.bin")]:
            f.unlink(missing_ok=True)
        shutil.rmtree(d / "tampered", ignore_errors=True)
    return row


def wired_topology(N_IN: int, N_OUT: int) -> tuple[int, tuple]:
    """The circuit's input bits and four layers of the unit, each layer's x as its parts in order: ``("c", lo, hi)``, the circuit's
    input bits, or ``("u", j, lo, hi)``, layer j's output bits. Layer 1 reads layer 0, layer 2 reads layers 0 and 1, layer 3 reads
    layer 2; layer 3's outputs are the circuit's."""
    if N_IN != 2 * N_OUT:
        raise ValueError(f"the wired circuit needs N_IN = 2 N_OUT, not {N_IN}, {N_OUT}")
    nc = N_IN + 2 * N_OUT
    return nc, ((("c", 0, N_IN),), (("c", N_IN, N_IN + N_OUT), ("u", 0, 0, N_OUT)), (("u", 0, 0, N_OUT), ("u", 1, 0, N_OUT)),
                (("u", 2, 0, N_OUT), ("c", N_IN + N_OUT, nc)))


def _render(parts) -> str:
    return " + ".join(f"c[{p[1]}:{p[2]}]" if p[0] == "c" else f"u{p[1]}[{p[2]}:{p[3]}]" for p in parts)


def wired_program(fn, n: int, nc: int, topo):
    """The wired circuit on ``n`` instances as a Program: per layer one ``batch`` of ``n`` members of the unit, its program (``op``,
    ``src``, ``out``) one parameter shared by the members and its x per member, assembled from the circuit's input and earlier
    layers' outputs (reference rearrangements, no gates). Returns (the composite, the Program)."""
    from verity.ml.circuits.defs import CompositeDefinition
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.refs import array_of, coll_concat

    params, ret = fn.signature()
    n_in = params[-1][1].leaves
    sig = [("c", Array(n, Array(nc, U.BIT)))] + [(f"{nm}{i}", t) for i in range(len(topo)) for nm, t in params[:-1]]

    def body(B, S, c, *ps):
        us = []
        for i, parts in enumerate(topo):
            xs = array_of([coll_concat(Array(n_in, U.BIT), [c[t][p[1]:p[2]] if p[0] == "c" else us[p[1]][t][p[2]:p[3]] for p in parts])
                           for t in range(n)])
            us.append(B.batch(fn, *ps[3 * i:3 * i + 3], xs, axes=(None, None, None, 0), n=n))
        return us[-1]

    name = f"WiredUniversal{len(topo)}x{n}[{fn.id}]"
    root = CompositeDefinition(name, 1, (), lambda S: (tuple(sig), Array(n, ret)), body, register=False)
    return root, Program(root)


def compose(nets, cs, topo, made_up=None):
    """Every layer's x and outputs on every instance, run along the topology by `universal.interpret`. ``made_up``: {layer:
    f(x) -> x}, the input a cheating prover gives that layer in place of its wired one; every layer is still run correctly on
    what it is given, and later layers read its outputs."""
    xs, ys = [[] for _ in topo], [[] for _ in topo]
    for t, c in enumerate(cs):
        for i, parts in enumerate(topo):
            x = [b for p in parts for b in (c[p[1]:p[2]] if p[0] == "c" else ys[p[1]][t][p[2]:p[3]])]
            if made_up and i in made_up:
                x = made_up[i](x)
            xs[i].append(x)
            ys[i].append(U.interpret(nets[i], x))
    return xs, ys


def derived_topology(tp, n: int, n_in: int, nc: int, layers: int) -> dict:
    """What the partition says each layer's x reads (`TemplatePartition.inputs`, the gates the verifier derives from the program),
    as `_render`'s parts, and whether every member reads its own instance's bits; the program parameters, the units' other
    inputs, must be the root's Input gates."""
    owner = {g: (u // n, u % n, j) for u in range(tp.population) for j, g in enumerate(tp.outputs(u))}
    per, own, params_input = [], True, True
    for i in range(layers):
        seen = set()
        for t in range(n):
            ins = tp.inputs(i * n + t)
            params_input &= all(g not in owner for g in ins[:-n_in])
            runs: list[list] = []
            for g in ins[-n_in:]:
                if g < n * nc:
                    key, member, bit = ("c",), g // nc, g % nc
                else:
                    layer, member, bit = owner[g]
                    key = ("u", layer)
                own &= member == t
                if runs and runs[-1][0] == key and runs[-1][2] == bit:
                    runs[-1][2] = bit + 1
                else:
                    runs.append([key, bit, bit + 1])
            seen.add(_render([(*k, lo, hi) for k, lo, hi in runs]))
        per.append(sorted(seen))
    return {"layers": [p[0] if len(p) == 1 else p for p in per], "every_member_reads_its_own_instance": own,
            "program_parameters_are_inputs": params_input}


def run_wired(a, G: int, N_IN: int, N_OUT: int, out: Path) -> dict:
    """The wired circuit (`wired_topology`) on ``a.n`` instances, a random hidden netlist per layer, as one class statement of
    ``4 a.n`` instances (Q_template_instance: every member of every layer one unit); the honest staging and a cheating one whose
    layer 1 reads a made-up input in place of layer 0's outputs, every unit correct on the input it is given."""
    rng = random.Random(f"pc-flock/wired/{a.seed}/{G}/{N_IN}/{N_OUT}")
    fn = bind(U.UniversalUnit, G=G, N_IN=N_IN, N_OUT=N_OUT)
    nc, topo = wired_topology(N_IN, N_OUT)
    L = len(topo)
    smalls = [_netlist(rng, G, N_IN, N_OUT, n_in=N_IN) for _ in topo]
    nets = [U.pad(s, G, N_IN) for s in smalls]
    cs = [[rng.randrange(2) for _ in range(nc)] for _ in range(a.n)]
    root, program = wired_program(fn, a.n, nc, topo)
    xs, ys = compose(nets, cs, topo)
    want = [b for y in ys[-1] for b in y]
    pargs = [p for net in nets for p in U.arguments(net, [0] * N_IN)[:3]]
    flat_c = [b for c in cs for b in c]
    t0 = time.monotonic()
    ev = list(E.evaluate(root, flat_c, *pargs))
    ev_s = time.monotonic() - t0
    evb = np.asarray(E.evaluate_batch(root, np.array([flat_c], dtype=np.uint8), *(np.array([p], dtype=np.uint8) for p in pargs)))
    agree = {"interpret_padded": compose(smalls, cs, topo)[1] == ys, "program_evaluate": ev == want,
             "program_evaluate_batch": evb.reshape(-1).tolist() == want}

    query = PO.template_instance_query(fn)
    units = PU.PartitionUnits(program, query=query)
    [(shape, cl)] = units.classes().items()
    tp = PO.TemplatePartition(program, query)
    derived = derived_topology(tp, a.n, N_IN, nc, L)
    agree["topology_derived"] = derived["layers"] == [_render(p) for p in topo] and derived["every_member_reads_its_own_instance"] \
        and derived["program_parameters_are_inputs"]

    def staged_lanes(xs_, ys_):
        ins = want_ = idents = None
        for i in range(L):
            li, lo, lid = lanes(units, cl, nets[i], xs_[i])
            if [[lo[j][t] for j in range(N_OUT)] for t in range(a.n)] != ys_[i]:
                raise AssertionError(f"layer {i}: the unit's transcript disagrees with interpret")
            lid = [[(*ident, i) for ident in ids] for ids in lid]
            ins, want_, idents = (li, lo, lid) if ins is None else ([p + q for p, q in zip(ins, li)], [p + q for p, q in zip(want_, lo)],
                                                                   [p + q for p, q in zip(idents, lid)])
        return ins, want_, idents

    d = out / f"wired-G{G}-N{N_IN}-O{N_OUT}"
    shutil.rmtree(d, ignore_errors=True)
    stage = d / "stage"
    salt_key = secrets.token_bytes(32)
    wiring = CS.Wiring(program, query, list(range(L * a.n)))
    t0 = time.monotonic()
    st = CS.stage(units, SPEC, shape, cl, staged_lanes(xs, ys), stage, salt_key, packed=True, wiring=wiring)
    stage_s = time.monotonic() - t0
    n = st["n"]
    if n != L * a.n:
        raise AssertionError(f"staging capped the instances at {n}, not {L * a.n}")
    part, desc = stage / "partition.json", stage / "program.desc"
    part.write_text(json.dumps(PO.build(program, query), sort_keys=True))
    from verity.ml.circuits.codec import canonical_json, encode_program
    desc.write_bytes(canonical_json(encode_program(program)))
    text = (stage / "circuit.txt").read_text()
    pin = FC.digest(text)
    # the live verifier's own links, derived from its program and query; and none, under which it records a cheating session for Lean
    links, no_links = stage / "links.json", d / "no-links.json"
    t0 = time.monotonic()
    own = CS.links_file(program, query, text, wiring.indices)
    links_s = time.monotonic() - t0
    links.write_text(json.dumps(own))
    no_links.write_text(json.dumps(own | {"links": []}))

    # the cheat: layer 1's x has layer 0's output bits flipped, so its committed input row is not layer 0's committed output row
    off = sum(p[2] - p[1] if p[0] == "c" else p[3] - p[2] for p in topo[1][:next(k for k, p in enumerate(topo[1]) if p[0] == "u")])
    w0 = next(p[3] - p[2] for p in topo[1] if p[0] == "u")
    flip = {1: lambda x: x[:off] + [1 - b for b in x[off:off + w0]] + x[off + w0:]}
    bxs, bys = compose(nets, cs, topo, flip)
    broken = sum(bxs[1][t][off:off + w0] != bys[0][t][:w0] for t in range(a.n))
    wrong = sum(bys[-1][t] != ys[-1][t] for t in range(a.n))
    bad, salt = d / "made-up-input", d / "wrong-salt"
    sb = CS.stage(units, SPEC, shape, cl, staged_lanes(bxs, bys), bad, salt_key, packed=True, wiring=replace(wiring, strict=False))
    ss = CS.stage(units, SPEC, shape, cl, staged_lanes(xs, ys), salt, salt_key, packed=True, wiring=replace(wiring, writer_salts=False))
    pa, pb = (stage / f"pub-{n}.bin").read_bytes(), (bad / f"pub-{n}.bin").read_bytes()

    row = {"class": {"G": G, "N_IN": N_IN, "N_OUT": N_OUT, "W": N_IN + G, "K": U.index_bits(G, N_IN)}, "wired": True,
           "circuit": {"layers": L, "instances": a.n, "input_bits": nc, "output_bits": N_OUT, "topology": [_render(p) for p in topo],
                       "topology_derived": derived, "units": tp.population, "committed_gates": len(tp.committed()),
                       "netlists": [{"gates": len(s.gates), "inputs": s.n_in, "ands": sum(1 for g in s.gates if g[0] & 1)} for s in smalls],
                       "program_evaluate_s": round(ev_s, 3), "descriptor_bytes": desc.stat().st_size},
           "unit": {"ands_lowered": st["ands"], "unit_rows": st["unit_rows"], "unit_nnz": st["unit_nnz"],
                    "ands_per_hidden_gate": round(st["ands"] / G, 2)},
           "statement": {k: st[k] for k in ("n", "k_log", "vus_per_block", "w_in", "w_out", "ports", "shared_rows", "circuit_sha512",
                                            "class_sha512", "circuit_bytes", "links", "definition")} | {"shape": shape, "stage_s": round(stage_s, 3)},
           "links": {"triples": len(own["links"]), "ports": own["ports"], "derive_s": round(links_s, 3)},
           "agree_before_staging": agree, "seed": a.seed,
           "made_up_input": {"layer": 1, "reads": _render(topo[1]), "flipped_bits": [off, off + w0], "instances_wire_broken": broken,
                             "instances_circuit_output_wrong": wrong, "same_circuit": sb["circuit_sha512"] == st["circuit_sha512"],
                             "pub_bytes": len(pa), "pub_bytes_differing": sum(x != y for x, y in zip(pa, pb)) + abs(len(pa) - len(pb))},
           "wrong_salt": {"rows": "every link row holds its writer's output row, committed under a salt of its own",
                          "same_circuit": ss["circuit_sha512"] == st["circuit_sha512"]}}
    if not all(agree.values()):
        raise AssertionError(f"wired {G},{N_IN},{N_OUT}: the outputs disagree before staging: {agree}")
    if a.stage_only:
        return row

    none = {"accepted": False, "why": "no verdict"}

    def lean_rec(sdir: Path, recorded: list[Path]) -> dict:
        v, dt, err = lean_verify(a.lean, sdir / "circuit.txt", sdir / f"pub-{n}.bin", part, recorded, program=desc)
        return {"accepted": len(v) == len(recorded) and all(x["accepted"] for x in v), "sessions": len(v),
                "why": sorted({x["why"] for x in v}), "wall_s": round(dt, 3), "wall_s_per_session": round(dt / max(1, len(v)), 3),
                "partition": (v[0].get("partition") if v else None), **({"stderr_tail": err} if not v else {})}

    # honest: the live verifier with its own links, then Lean with its own program; without the program Lean cannot check the links
    dd = d / "honest-sessions"
    dd.mkdir(parents=True, exist_ok=True)
    live, recorded = live_session(a, stage, n, dd, pin, a.warm, a.runs, links=links)
    row["honest"] = {"live": live}
    if a.lean and recorded:
        row["honest"]["lean"] = lean_rec(stage, recorded)
        x = (lean_verify(a.lean, stage / "circuit.txt", stage / f"pub-{n}.bin", part, recorded[:1])[0] or [none])[0]
        row["honest"]["lean"]["without_program"] = {"refused": not x["accepted"], "why": x["why"][:200]}
        x = (lean_verify(a.lean, stage / "circuit.txt", bad / f"pub-{n}.bin", part, recorded[:1], program=desc)[0] or [none])[0]
        row["honest"]["lean"]["against_the_other_public_file"] = {"refused": not x["accepted"], "why": x["why"][:160]}
    # each cheat: the live verifier with its own links refuses it before serving; a live verifier given no links records its
    # sessions, which Lean must refuse from its own program
    for tag, sdir in (("made_up_input", bad), ("wrong_salt", salt)):
        rec = row[tag]
        dd = d / f"{tag}-sessions"
        dd.mkdir(parents=True, exist_ok=True)
        rec["live"], _ = live_session(a, sdir, n, dd / "own-links", pin, 0, 1, links=links)
        rec["live_without_links"], recorded = live_session(a, sdir, n, dd / "no-links", pin, 0, a.runs, links=no_links)
        if a.lean and recorded:
            rec["lean"] = lean_rec(sdir, recorded)
        rec["refused_live"] = not rec["live"]["accepted"]
        rec["refused_lean"] = not (rec.get("lean") or {}).get("accepted") if a.lean and recorded else None
        rec["verdict"] = ("refused by both verifiers" if rec["refused_live"] and rec["refused_lean"] else
                          "ACCEPTED: a binding gap" if not rec["refused_live"] and rec["refused_lean"] is False else
                          f"split or unchecked: live refused {rec['refused_live']}, Lean refused {rec['refused_lean']}")
    if not a.keep_stage:
        for s in (stage, bad, salt):
            for f in [s / "circuit.txt", *s.glob("inst-*.bin")]:
                f.unlink(missing_ok=True)
    return row


def write_result(rows: list[dict], a) -> None:
    """``$RESEARCH_RUN_DIR/result.json`` (research/result/v0.1) when run under ``research run``."""
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    if not run_dir:
        return
    ms = []
    for r in rows:
        c = r["class"]
        tag = f"{'wired4_' if r.get('wired') else ''}G{c['G']}_N{c['N_IN']}_O{c['N_OUT']}_n{(r.get('statement') or {}).get('n')}"
        live, lean = ((r.get("honest") or {}).get("live"), (r.get("honest") or {}).get("lean")) if r.get("wired") else (r.get("live"), r.get("lean"))
        mu, ws = r.get("made_up_input") or {}, r.get("wrong_salt") or {}
        for name, v, unit in (("ands", (r.get("unit") or {}).get("ands_lowered"), "count"),
                              ("ands_per_hidden_gate", (r.get("unit") or {}).get("ands_per_hidden_gate"), "count"),
                              ("k_log", (live or {}).get("k_log"), "count"),
                              ("prove_total_s", (live or {}).get("prove_total_s"), "s"),
                              ("proof_bytes", sum((live or {}).get("proof_bytes") or []) or None, "B"),
                              ("live_verify_s", (live or {}).get("verify_s"), "s"),
                              ("lean_verify_s_per_session", (lean or {}).get("wall_s_per_session"), "s"),
                              ("made_up_input_accepted_live", None if "refused_live" not in mu else int(not mu["refused_live"]), "count"),
                              ("made_up_input_accepted_lean", None if mu.get("refused_lean") is None else int(not mu["refused_lean"]), "count"),
                              ("wrong_salt_accepted_live", None if "refused_live" not in ws else int(not ws["refused_live"]), "count"),
                              ("wrong_salt_accepted_lean", None if ws.get("refused_lean") is None else int(not ws["refused_lean"]), "count")):
            if v is not None:
                ms.append({"name": f"{tag}/{name}", "value": v, "unit": unit})
    doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"kind": "private-circuit/universal-e2e/v0", "statement": FC.STATEMENT, "zk": True,
                                    "classes": a.cls, "n": a.n, "warm": a.warm, "runs": a.runs, "seed": a.seed, "wired": a.wired},
           "validation": {"status": "passed" if rows and all(r["pass"] for r in rows) else "failed",
                          "checker": "flock-circuit serve --zk (live) and Lean flock-verify verify --zk, outputs against verity.ml.circuits.evaluation",
                          "detail": f"{sum(r['pass'] for r in rows)} of {len(rows)} rows passed" + "".join(
                              f"; wired {r['class']['G']},{r['class']['N_IN']},{r['class']['N_OUT']} made-up input: "
                              f"{(r.get('made_up_input') or {}).get('verdict')}, wrong salt: {(r.get('wrong_salt') or {}).get('verdict')}"
                              for r in rows if r.get("wired")), "evidence": ["out/rows.jsonl"]},
           "measurements": ms, "measurement_files": ["out/rows.jsonl"], "artifacts": []}
    Path(run_dir, "result.json").write_text(json.dumps(doc, sort_keys=True, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cls", action="append", required=True, help="G,N_IN,N_OUT (repeatable)")
    ap.add_argument("--n", type=int, default=64, help="instances: runs of the one netlist on as many random x")
    ap.add_argument("--bin", default="flock-circuit")
    ap.add_argument("--lean", default=None, help="flock-verify (Lean): replay every recorded session with --zk")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--port", type=int, default=7820)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--keep-stage", action="store_true", help="keep circuit.txt and the prover's inst-N.bin")
    ap.add_argument("--wired", action="store_true", help="each class as the wired circuit of four units (run_wired), after the "
                    "one-unit statement at the same instance count (4 n)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ok, rows = True, []
    jobs = [(c, kind) for c in a.cls for kind in (("one-unit", "wired") if a.wired else ("one-unit",))]
    for c, kind in jobs:
        G, N_IN, N_OUT = (int(v) for v in c.split(","))
        try:
            if kind == "wired":
                row = run_wired(a, G, N_IN, N_OUT, out)
            else:
                row = run_class(argparse.Namespace(**{**vars(a), "n": 4 * a.n}) if a.wired else a, G, N_IN, N_OUT, out)
        except Exception as e:  # noqa: BLE001  (a class that breaks is a row, recorded with why)
            row = {"class": {"G": G, "N_IN": N_IN, "N_OUT": N_OUT}, "wired": kind == "wired", "broke": f"{type(e).__name__}: {e}"[:800]}
        if kind == "wired":
            h = row.get("honest") or {}
            cheats = [row.get(t) or {} for t in ("made_up_input", "wrong_salt")]
            passed = "broke" not in row and all(c.get("same_circuit") for c in cheats) and (a.stage_only or (
                h["live"]["accepted"] and all(c.get("refused_live") for c in cheats) and (not a.lean or (
                    h.get("lean", {}).get("accepted") and h["lean"]["against_the_other_public_file"]["refused"]
                    and h["lean"]["without_program"]["refused"]
                    and all(c.get("refused_lean") for c in cheats)))))
        else:
            passed = "broke" not in row and (a.stage_only or (row["live"]["accepted"] and (not a.lean or (
                row.get("lean", {}).get("accepted") and all(x["refused"] for x in row["lean"]["negatives"].values())
                and row["lean"]["negatives"]["another_program"]["same_circuit"]))))
        row["pass"] = bool(passed)
        ok &= row["pass"]
        rows.append(row)
        with (out / "rows.jsonl").open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
    write_result(rows, a)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
