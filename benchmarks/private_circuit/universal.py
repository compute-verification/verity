"""Protocol 2's universal unit end to end: a random hidden netlist, padded to its size class, run by the public
`UniversalUnit_v1{G, N_IN, N_OUT}` under C-Flock's circuit prover (M0) with `--zk`, and verified by the Lean verifier.

    python benchmarks/private_circuit/universal.py --cls 16,16,8 --n 64 --bin flock-circuit --lean flock-verify --out DIR

* **The program.** A netlist of 3/4 of the class's gates over 3/4 of its inputs, every index in range (`_netlist`), padded to
  the class with no-op gates and unread inputs (`universal.pad`). It is a value: the Call's `op`, `src` and `out` parameters.
* **The statement.** A one-Call Program of the unit, partitioned by `Q_template_instance` (the Call is one unit), each instance
  one run of the netlist on its own random `x`. The unit's class is a `verity/flock-circuit` statement (`class_statement.stage`,
  words packed), one row per source: `op`, `src` and `out` are committed once and opened by every instance (the registered
  program), `x` once per instance; the outputs are hidden rows.
* **The checks.** Before staging, every instance's outputs agree four ways: the netlist interpreted (`universal.interpret`),
  the hidden program as its own Boolean Definition (`direct`) under `verity.evaluation.evaluate`, the unit under
  `verity.evaluation.evaluate`, and the flat Definition's transcript that staging lays out (which `class_statement.stage`
  checks against the lowered circuit). The live `serve --zk` / `prove --zk` session must be accepted, then each recorded session
  is replayed by Lean's `flock-verify verify --zk`, which must accept it, and must refuse it without `--zk` and with a proof byte
  flipped.
* **The row.** `rows.jsonl`: the class, the unit's ANDs (traced, closed form and C-Flock's lowering), ANDs per hidden gate,
  the direct netlist's ANDs, the statement's size (k_log, instances per block), prover time, proof bytes, the live and the Lean
  verifier's times.
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
from pathlib import Path

from verity import evaluation as E
from verity.evaluation import bits as BS
from verity.ir import partition_object as PO
from verity.ir.defs import bind
from verity.ir.types import Array
from verity.ml.boolean import trace as TR
from verity.ml.boolean import universal as U
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import partition_units as PU

SPEC = "q-template-instance"


def _netlist(rng: random.Random, G: int, N_IN: int, N_OUT: int) -> U.Netlist:
    g0, n0 = max(1, 3 * G // 4), max(1, 3 * N_IN // 4)
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


def lean_verify(lean: str, stage: Path, n: int, partition: Path, sessions: list[Path], zk: bool = True) -> tuple[list[dict], float, str]:
    cmd = [lean, "verify", "--circuit", str(stage / "circuit.txt"), "--public", str(stage / f"pub-{n}.bin"),
           "--partition", str(partition), *(["--zk"] if zk else []), "--session", *map(str, sessions)]
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    dt = time.monotonic() - t0
    v = [json.loads(ln.split(" ", 1)[1]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
    return v, dt, (p.stderr or "")[-600:]


def run_class(a, G: int, N_IN: int, N_OUT: int, out: Path) -> dict:
    rng = random.Random(f"pc-flock/{a.seed}/{G}/{N_IN}/{N_OUT}")
    fn = bind(U.UniversalUnit, G=G, N_IN=N_IN, N_OUT=N_OUT)
    small = _netlist(rng, G, N_IN, N_OUT)
    net = U.pad(small, G, N_IN)
    hidden = direct(small, f"HiddenNetlistG{G}N{N_IN}O{N_OUT}S{a.seed}")
    xs_small = [[rng.randrange(2) for _ in range(small.n_in)] for _ in range(a.n)]
    xs = [x + [rng.randrange(2) for _ in range(N_IN - small.n_in)] for x in xs_small]
    want = [U.interpret(small, x) for x in xs_small]
    agree = {"interpret_padded": all(U.interpret(net, x) == w for x, w in zip(xs, want)),
             "direct_evaluate": all(list(E.evaluate(hidden, x)) == w for x, w in zip(xs_small, want)),
             "unit_evaluate": all(list(E.evaluate(fn, *U.arguments(net, x))) == w for x, w in zip(xs, want))}

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
    st = CS.stage(units, SPEC, shape, cl, (ins, outs, idents), stage, secrets.token_bytes(32), packed=True)
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

    sess = d / "sessions"
    log = d / "serve.log"
    srv = subprocess.Popen([a.bin, "serve", "--zk", "--listen", f"127.0.0.1:{a.port}", "--out", str(sess), "--instances",
                            str(stage / f"pub-{n}.bin"), "--circuit", str(stage / "circuit.txt"), "--pin", pin, "--operator", "pc-flock",
                            "--sessions", str(a.warm + a.runs)], stdout=open(log, "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(3000):
            if "SERVING" in log.read_text() or srv.poll() is not None:
                break
            time.sleep(0.2)
        with open(d / "prove.out", "w") as fo, open(d / "prove.err", "w") as fe:
            timer = ["/usr/bin/time", "-f", "%M", "-o", str(d / "prove.maxrss")] if os.path.exists("/usr/bin/time") else []
            pr = subprocess.run(timer + [a.bin, "prove", "--zk", "--verifier", f"127.0.0.1:{a.port}", "--instances", str(stage / f"inst-{n}.bin"),
                                         "--circuit", str(stage / "circuit.txt"), "--warm", str(a.warm), "--runs", str(a.runs)],
                                stdout=fo, stderr=fe, check=False)
        srv.wait(timeout=600)
    finally:
        if srv.poll() is None:
            srv.kill()
            srv.wait()
    live = [json.loads(ln[5:]) for ln in (d / "prove.out").read_text().splitlines() if ln.startswith("LIVE")]
    timed = [x for x in live if not x.get("warm")] or live
    med = sorted(timed, key=lambda x: x.get("prove_total_s") or 0)[len(timed) // 2] if timed else {}
    row["live"] = {"exit": pr.returncode, "sessions": len(live), "accepted": bool(live) and all(x.get("accepted") is True for x in live)
                   and pr.returncode == 0 and len(live) == a.warm + a.runs, "why": med.get("why"),
                   "prove_total_s": med.get("prove_total_s"), "prove_total_s_runs": [x.get("prove_total_s") for x in timed],
                   "e2e_s": med.get("e2e_s"), "witness_s": med.get("witness_s"), "proof_bytes": med.get("proof_bytes"),
                   "verify_s": (med.get("verdict") or {}).get("verify_s"), "k_log": med.get("k_log"), "prover": med.get("prover"),
                   "statement_digest": med.get("statement_digest"),
                   "prover_peak_gb": round(int((d / "prove.maxrss").read_text().split()[-1]) / 2**20, 3) if (d / "prove.maxrss").exists() else None}
    if not live:
        row["live"]["stderr_tail"] = (d / "prove.err").read_text()[-800:]
    recorded = _sessions(sess)
    row["live"]["recorded_sessions"] = len(recorded)
    if a.lean and recorded:
        v, dt, err = lean_verify(a.lean, stage, n, part, recorded)
        row["lean"] = {"accepted": len(v) == len(recorded) and all(x["accepted"] for x in v), "sessions": len(v),
                       "why": sorted({x["why"] for x in v}), "wall_s": round(dt, 3), "wall_s_per_session": round(dt / max(1, len(v)), 3),
                       **({"stderr_tail": err} if not v else {})}
        none = {"accepted": False, "why": "no verdict"}
        nz = (lean_verify(a.lean, stage, n, part, recorded[:1], zk=False)[0] or [none])[0]
        bad = d / "tampered" / recorded[0].name
        shutil.copytree(recorded[0], bad)
        proof = max(bad.glob("*.bin"), key=lambda p: p.stat().st_size)
        b = bytearray(proof.read_bytes())
        b[len(b) // 2] ^= 1
        proof.write_bytes(bytes(b))
        fl = (lean_verify(a.lean, stage, n, part, [bad])[0] or [none])[0]
        row["lean"]["negatives"] = {"without_zk": {"refused": not nz["accepted"], "why": nz["why"][:160]},
                                    "proof_byte_flipped": {"refused": not fl["accepted"], "why": fl["why"][:160], "file": proof.name}}
    if not a.keep_stage:
        # the circuit is restaged byte for byte from the code; the prover's file holds the rows and salts
        for f in [stage / "circuit.txt", *stage.glob("inst-*.bin")]:
            f.unlink(missing_ok=True)
        shutil.rmtree(d / "tampered", ignore_errors=True)
    return row


def write_result(rows: list[dict], a) -> None:
    """``$RESEARCH_RUN_DIR/result.json`` (research/result/v0.1) when run under ``research run``."""
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    if not run_dir:
        return
    ms = []
    for r in rows:
        c = r["class"]
        tag = f"G{c['G']}_N{c['N_IN']}_O{c['N_OUT']}"
        for name, v, unit in (("ands", (r.get("unit") or {}).get("ands_lowered"), "count"),
                              ("ands_per_hidden_gate", (r.get("unit") or {}).get("ands_per_hidden_gate"), "count"),
                              ("prove_total_s", (r.get("live") or {}).get("prove_total_s"), "s"),
                              ("proof_bytes", sum((r.get("live") or {}).get("proof_bytes") or []) or None, "B"),
                              ("live_verify_s", (r.get("live") or {}).get("verify_s"), "s"),
                              ("lean_verify_s_per_session", (r.get("lean") or {}).get("wall_s_per_session"), "s")):
            if v is not None:
                ms.append({"name": f"{tag}/{name}", "value": v, "unit": unit})
    doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"kind": "private-circuit/universal-e2e/v0", "statement": FC.STATEMENT, "zk": True,
                                    "classes": a.cls, "n": a.n, "warm": a.warm, "runs": a.runs, "seed": a.seed},
           "validation": {"status": "passed" if rows and all(r["pass"] for r in rows) else "failed",
                          "checker": "flock-circuit serve --zk (live) and Lean flock-verify verify --zk, outputs against verity.evaluation",
                          "detail": f"{sum(r['pass'] for r in rows)} of {len(rows)} classes passed", "evidence": ["out/rows.jsonl"]},
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
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ok, rows = True, []
    for c in a.cls:
        G, N_IN, N_OUT = (int(v) for v in c.split(","))
        try:
            row = run_class(a, G, N_IN, N_OUT, out)
        except Exception as e:  # noqa: BLE001  (a class that breaks is a row, recorded with why)
            row = {"class": {"G": G, "N_IN": N_IN, "N_OUT": N_OUT}, "broke": f"{type(e).__name__}: {e}"[:800]}
        passed = "broke" not in row and (a.stage_only or (row["live"]["accepted"] and (not a.lean or (
            row.get("lean", {}).get("accepted") and all(x["refused"] for x in row["lean"]["negatives"].values())))))
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
