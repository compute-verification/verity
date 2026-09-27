"""A unit class of any partition (`verity.ir.units`) as a ``verity/flock-circuit`` statement (M0, :mod:`circuit`), proved by
``flock-circuit`` per class.

    python -m verity_flock.class_statement --model tiny --partition q-word --out DIR [--binary PATH] [--n 16] [--selftest]

M0's statement proves a template's identical units with every instance's input rows hashed in the circuit
(``hm96-sha512/row/v1`` under frame-v3-sha512 trees). A class becomes such a template here:

* one unit per instance: the class's circuit (`partition_units.lower_class`), its inputs packed into one row port ``in`` of u16
  words (a word of w bits takes ceil(w / 16) consecutive u16 leaves, low half first; the row padded with unread zero words to a
  whole number of 64-word SHA-512 blocks, as frame-v3 rows are), its outputs into one u16 output port ``out`` (each output
  padded with zero bits to whole u16 words);
* the instances are the class's lanes from `partition_units.lanes` (synthetic inputs lifted through the flat Definition), the
  statement binds the partition's ``verity/partition/v1`` digest and the program's SHA-512 (an owner-function partition, which no
  record can name, binds the SHA-512 of its description under the tag ``verity/flock/experimental-owners``).

This is a synthetic layout: one row per unit instance, not the scheme's layout rule over the partition (which is not specified
yet), and M0's statement publishes the declared outputs, where a partition's unit outputs are committed values.  Every class is one plain
Boolean circuit, with no slot type other than M0's row slots: a unit's pinned-table reads (MUFU) are laid out inline as ordinary
rows by `ir_lower.layout` (the one-hot construction; 154-309 M matrix entries per read, about 20 B of prover memory each), and each
record carries the class's table reads and its unit's matrix entries.  `distinct_calls` builds a served row's workload from its program graph:
a Program with one Call of each distinct Call Definition, whose per-Call partitions have exactly the row's classes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from verity.ir import partition_object as PO
from verity.ir.types import Array, Value
from verity.ir.units import Activation, PartitionUnits, UnitCircuit

from . import circuit as FC
from . import gf2
from . import ir_lower as IL
from . import partition_units as PU

BLOCK_WORDS = 64
#: the selftest cases run on a read-bearing class (each re-parses its 1-3 GB circuit): the honest session, and the two witness tamperings
READ_CASES = ("honest", "unit_internal_bit_flipped", "unit_input_differs_from_its_message_bits")
EXPERIMENTAL = "verity/flock/experimental-owners"


def _words(w: int) -> int:
    return -(-w // 16)


def _pad(n: int, to: int) -> int:
    return max(to, -(-n // to) * to)


@dataclass
class ClassDefinition:
    """What M0's writers read of a Definition: one ``u16`` row parameter and one ``u16`` result array."""

    id: str
    w_in: int
    w_out: int

    def signature(self):
        return (("in", Array(self.w_in, Value(16))),), Array(self.w_out, Value(16))


def class_lowering(act: Activation, uc: UnitCircuit, name: str) -> IL.Lowering:
    """The class's unit on u16 ports: ``in0 .. in{W_in-1}`` (the row's words) and ``out0 .. out{W_out-1}``."""
    w_in = _pad(sum(_words(w) for w in uc.in_widths), BLOCK_WORDS)
    C = gf2.Circuit()
    row = [C.input(f"in{j}", 16) for j in range(w_in)]
    wires, at = {}, 0
    for k, w in zip(uc.inputs, uc.in_widths):
        bits = [b for word in row[at:at + _words(w)] for b in word][:w]
        wires[act.absolute(k)] = bits
        at += _words(w)
    IL.lower_gates(C, [act.gate(j) for j in uc.gates], wires)
    out_bits = []
    for j, w in zip(uc.outputs, uc.out_widths):
        out_bits += list(wires[act.absolute(j)]) + [gf2.ZERO] * (16 * _words(w) - w)
    w_out = max(1, len(out_bits) // 16)
    out_bits += [gf2.ZERO] * (16 * w_out - len(out_bits))
    for j in range(w_out):
        C.output(f"out{j}", out_bits[16 * j:16 * (j + 1)])
    units = IL.Units(target=None, in_widths=[16] * w_in, out_widths=[16] * w_out, in_src=[[("leaf", p) for p in range(w_in)]],
                     out_src=[[("ret", p) for p in range(w_out)]], signature=(uc.shape,), gates=[], arg_gates=[], ret_gates=[])
    lay = IL.layout(C, [f"in{j}" for j in range(w_in)], [f"out{j}" for j in range(w_out)])
    return IL.Lowering(name, ClassDefinition(name, w_in, w_out), units, C, lay)


def pack(uc: UnitCircuit, ins: Sequence[Sequence[int]], want: Sequence[Sequence[int]], w_in: int, w_out: int):
    """The lanes as rows ``(n, w_in)`` and outputs ``(n, w_out)``, u16, in the packing `class_lowering` reads."""
    n = len(ins[0]) if ins else len(want[0])
    rows, outs = np.zeros((n, w_in), np.uint16), np.zeros((n, w_out), np.uint16)
    for dst, ports, widths in ((rows, ins, uc.in_widths), (outs, want, uc.out_widths)):
        at = 0
        for vals, w in zip(ports, widths):
            v = np.asarray(vals, dtype=np.uint64)
            for h in range(_words(w)):
                dst[:, at + h] = ((v >> np.uint64(16 * h)) & np.uint64(0xFFFF)).astype(np.uint16)
            at += _words(w)
    return rows, outs


def _shims(name: str, content: bytes):
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")  # noqa: E731
    sub = SimpleNamespace(id=f"unit-class/{name}", template=SimpleNamespace(id="unit-class"), inputs=[port("in")], outputs=[port("out")],
                          program_digests=lambda statics: {f"unit-class/{name}": hashlib.sha256(content).hexdigest()})
    return sub, SimpleNamespace(subcircuit=sub, name=f"unit-class-{name}", content_digest=hashlib.sha256(content).hexdigest())


_SLOTS: dict = {}


def cache_slot_circuits() -> None:
    """Build the SHA-512 and hm96 slot circuits, their layouts and netlists once: `circuit.compose` rebuilds them for every
    statement, identically, and they are about 95% of a small class's staging time."""
    if _SLOTS:
        return
    from . import sha512_circuit as S

    c3, hm = S.compression3(), S.hm96_row()
    layout, expanded = IL.layout, FC._expanded
    lays: dict = {}
    texts: dict = {}

    def cached_layout(C, *a, **k):
        if C is c3 or C is hm:
            key = (id(C), repr(a), repr(sorted(k.items())))
            if key not in lays:
                lays[key] = layout(C, *a, **k)
            return lays[key]
        return layout(C, *a, **k)

    def cached_expanded(lay, name):
        if any(lay is v for v in lays.values()):
            return texts.setdefault((id(lay), name), expanded(lay, name))
        return expanded(lay, name)

    S.compression3, S.hm96_row = (lambda: c3), (lambda: hm)
    IL.layout, FC._expanded = cached_layout, cached_expanded
    _SLOTS.update(c3=c3, hm=hm)


def distinct_calls(graph_path: str, modules: str | None = None) -> tuple[Any, dict[str, int], list[dict]]:
    """A served row's distinct Call Definitions, from its program graph (``vllm-program-graph/v1``): (a Program with one Call of
    each, the row's Call count of each by descriptor id, a provenance row per Definition). ``modules``: a module-path prefix
    (``model.model.layers.0.``) to keep. Each Definition is bound from the registry and its v0 encoding's SHA-256 is checked
    against the graph's ``definitions_index`` where the graph has it (every group's spec; attention's smallest and largest T).
    A per-Call partition (``Q_word``, an owner function) cuts this Program's Calls exactly as the row's, so its classes are the
    row's and the row's unit counts are the per-Call counts times the Call counts."""
    import re

    from circuit_check import targets as TG
    from verity.ir.codec import descriptor_id
    from verity.ir.defs import CompositeDefinition
    from verity.ir.program import Program
    from verity_vllm.pipeline.program_graph import _encoded, definition_digest

    TG.load_registries()
    g = json.loads(open(graph_path).read())
    index = g.get("definitions_index") or {}
    calls: dict[str, int] = {}
    for x in g["groups"]:
        if modules and not x["module"].startswith(modules):
            continue
        if x.get("varying"):
            (axis, (lo, hi)), = x["varying"].items()
            per = x.get("varying_calls") or {str(v): 1 for v in range(int(lo), int(hi) + 1)}
            base = x["definition_ids"][0]
            for v, n in per.items():
                fid = re.sub(rf"(?<=[{{,]){axis}=\d+", f"{axis}={int(v)}", base)
                calls[fid] = calls.get(fid, 0) + int(n) * (x["calls"] // max(1, sum(int(c) for c in per.values())))
        elif x["spec"]:
            calls[x["spec"]] = calls.get(x["spec"], 0) + x["calls"]
    fns, prov = [], []
    for fid in calls:
        readable = re.sub(r'\{"f64":"([^"]+)"\}', lambda m: repr(float.fromhex(m[1])), fid)
        fn = TG.definition(readable)
        if descriptor_id(fn) != fid:
            raise ValueError(f"{fid}: the registry binds {descriptor_id(fn)}")
        want = index.get(fid)
        got = definition_digest(_encoded(fn)[1][fid]) if want else None
        if want and got != want:
            raise ValueError(f"{fid}: v0 digest {got} is not the row's {want}")
        fns.append(fn)
        prov.append({"definition": fid, "calls": calls[fid], "digest_checked": bool(want)})
    params = [(f"c{i}_{n}", t) for i, fn in enumerate(fns) for n, t in fn.signature()[0]]
    arity = [len(fn.signature()[0]) for fn in fns]

    def body(B, S, *xs):
        from verity.ir.refs import tuple_of
        outs, at = [], 0
        for fn, k in zip(fns, arity):
            outs.append(B.call(fn, *xs[at:at + k]))
            at += k
        return tuple_of(*outs)

    from verity.ir.types import Tuple as TupleT
    ret = TupleT(*[fn.signature()[1] for fn in fns])
    root = CompositeDefinition("DistinctCalls", 1, (), lambda S: (tuple(params), ret), body, register=False)
    return Program(root), calls, prov


def partition_binding(units: PartitionUnits, spec: str) -> tuple[str, str]:
    """(partition digest, program SHA-512) the statement binds."""
    program = PO.program_sha512(units.program)
    if units.query is not None:
        return PO.digest(PO.build(units.program, units.query)), program
    return hashlib.sha512(EXPERIMENTAL.encode() + b"\0" + json.dumps({"program": program, "owners": spec}, sort_keys=True).encode()).hexdigest(), program


def stage(units: PartitionUnits, spec: str, shape: str, cl, lanes_: tuple, out_dir: Path, salt_key: bytes) -> dict:
    """``circuit.txt``, the prover's ``inst-n.bin`` and the verifier's ``pub-n.bin`` for one class."""
    act = units.activation(cl.definition)
    uc = cl.representative
    name = shape[:16]
    ins, want = lanes_
    partition, program = partition_binding(units, spec)
    sub, iset = _shims(name, uc.form.encode())
    low = class_lowering(act, uc, f"unit-class/{name}")
    rows, outs = pack(uc, ins, want, low.definition.w_in, low.definition.w_out)
    got, _cut, ok = low.evaluate(rows.astype(np.uint64))
    text = FC.compose(low, sub, partition=partition, program=program)
    if not int(ok) == (1 << len(rows)) - 1 or not np.array_equal(got.astype(np.uint16), outs):
        raise AssertionError(f"class {name}: the laid-out unit rows disagree with the reference on its lanes")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "circuit.txt").write_text(text)
    n = len(rows)
    FC.write(text, iset, 0, n, out_dir / f"inst-{n}.bin", out_dir / f"pub-{n}.bin", salt_key, rows={"in": rows}, outs={"out": outs})
    meta = json.loads(text.split("\n", 2)[1][5:])
    return {"n": n, "unit_nnz": low.layout.counts["nnz_a"] + low.layout.counts["nnz_b"], "circuit_bytes": len(text), "w_in": low.definition.w_in, "w_out": low.definition.w_out, "unit_rows": low.layout.useful,
            "ands": low.layout.counts["ands"], "k_log": meta["k_log"], "vus_per_block": meta["vus_per_block"],
            "circuit_sha512": FC.digest(text), "class_sha512": FC.class_digest(text)}


def prove(binary: str, d: Path, n: int, port: int, selftest: bool | str = False, timeout: int = 3600) -> dict:
    """A loopback session (``serve`` on 127.0.0.1 and ``prove``), or ``selftest``: the LIVE or SELFTEST record."""
    circ, inst, pub = d / "circuit.txt", d / f"inst-{n}.bin", d / f"pub-{n}.bin"
    if selftest:
        cmd = [binary, "selftest", "--instances", str(inst), "--circuit", str(circ)]
        r = subprocess.run(cmd + (["--only", selftest] if isinstance(selftest, str) else []), capture_output=True, text=True, timeout=timeout)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("SELFTEST", "NEG"))]
        neg = [json.loads(ln.split("\t", 1)[1]) for ln in lines if ln.startswith("NEG")]
        head = next((json.loads(ln.split("\t", 1)[1]) for ln in lines if ln.startswith("SELFTEST")), None)
        failed = [c.get("case") for c in neg if c.get("pass") is not True]
        all_pass = head.get("all_pass") if head is not None else (bool(neg) and not failed)     # --only prints its one case, no summary
        return {"all_pass": all_pass, "cases": len(neg), "failed": failed}
    pin = FC.digest(circ.read_text())
    srv = subprocess.Popen([binary, "serve", "--listen", f"127.0.0.1:{port}", "--out", str(d / "sessions"), "--instances", str(pub),
                            "--circuit", str(circ), "--pin", pin, "--operator", "loopback-stress"],
                           stdout=open(d / "serve.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            if "SERVING" in (d / "serve.log").read_text():
                break
            time.sleep(0.2)
        with open(d / "prove.out", "w") as fo, open(d / "prove.err", "w") as fe:
            timer = ["/usr/bin/time", "-f", "%M", "-o", str(d / "prove.maxrss")] if os.path.exists("/usr/bin/time") else []
            pr = subprocess.Popen(timer + [binary, "prove", "--verifier", f"127.0.0.1:{port}", "--instances", str(inst), "--circuit", str(circ),
                                           "--runs", "1"], stdout=fo, stderr=fe)
            pr.wait()
        prover_gb = int((d / "prove.maxrss").read_text().split()[-1]) / 2**20 if timer else None   # the prover's own peak, KB
        verifier_gb = _vm_hwm(srv.pid)
    finally:
        srv.kill()
        srv.wait()
    out, err = (d / "prove.out").read_text(), (d / "prove.err").read_text()
    live = [json.loads(ln[5:]) for ln in out.splitlines() if ln.startswith("LIVE")]
    if not live:
        return {"accepted": False, "why": (err or out)[-400:]}
    v = live[-1]
    return {k: v.get(k) for k in ("accepted", "why", "prover", "instances", "vus_per_block", "k_log", "nbl", "dense_m", "e2e_s",
                                  "witness_s", "prove_s", "prove_total_s", "proof_bytes", "wire_sent", "wire_recv", "statement_digest")} | \
        {"prover_peak_gb": round(prover_gb, 2) if prover_gb else None, "verifier_peak_gb": verifier_gb}


def _vm_hwm(pid: int) -> float | None:
    """A live process's peak resident memory (VmHWM), GB."""
    try:
        for ln in open(f"/proc/{pid}/status"):
            if ln.startswith("VmHWM:"):
                return round(int(ln.split()[1]) / 2**20, 2)
    except OSError:
        return None
    return None


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", choices=["tiny"])
    src.add_argument("--definition")
    src.add_argument("--program-module", help="MODULE:FUNC (FUNC() returns a Program); MODULE may be a .py path")
    src.add_argument("--program-graph", help="a served row's program graph: its distinct Call Definitions (distinct_calls)")
    ap.add_argument("--modules", default=None, help="with --program-graph: keep groups whose module path starts with this")
    ap.add_argument("--partition", default="q-word")
    ap.add_argument("--out", required=True)
    ap.add_argument("--binary", default=os.environ.get("FLOCK_CIRCUIT", "flock-circuit"))
    ap.add_argument("--n", type=int, default=16, help="instances per class statement (synthetic lanes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", type=int, default=0, help="also run flock-circuit's selftest (31 cases, negatives included) on every "
                    "Nth provable class and on each flat Definition's largest (0: never)")
    ap.add_argument("--max-ands", type=int, default=1 << 24, help="skip classes above this many ANDs (reported)")
    ap.add_argument("--only", default=None, help="comma-separated shape prefixes")
    ap.add_argument("--selftest-only", action="store_true", help="stage and run the selftest cases only (no loopback session), into selftests.jsonl")
    ap.add_argument("--sample", type=int, default=0, help="prove only this many classes, drawn by a seeded hash of their shapes (0: all)")
    ap.add_argument("--port", type=int, default=7610)
    a = ap.parse_args(argv)
    calls: dict[str, int] | None = None
    if a.program_graph:
        program, calls, prov = distinct_calls(a.program_graph, a.modules)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "calls.json").write_text(json.dumps({"calls": calls, "provenance": prov}, indent=1))
    elif a.program_module:
        import importlib.util
        mod, fn = a.program_module.rsplit(":", 1)
        spec = importlib.util.spec_from_file_location("stress_program", mod) if mod.endswith(".py") else None
        if spec:
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        else:
            m = importlib.import_module(mod)
        program = getattr(m, fn)()
    else:
        program = PU._program(a)
    cache_slot_circuits()
    units = PU.partition_units(program, a.partition)
    classes = units.classes()
    key = PU.derive(PU.SOURCE, PU.DOMAIN, {"program": PO.program_sha512(program)[:32], "seed": a.seed})
    salt_key = hashlib.sha256(b"verity/flock/class-statement/stress-salts\0" + str(a.seed).encode()).digest()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / ("selftests.jsonl" if a.selftest_only else "results.jsonl")
    done = {json.loads(ln)["shape"] for ln in results.read_text().splitlines()} if results.exists() else set()
    row_units: dict[str, int] = {}
    if calls is not None:
        from verity.ir.codec import descriptor_id
        for _k, _m, fn, cut in units._calls():
            for sh, c in units._count(cut, fn).items():
                row_units[sh] = row_units.get(sh, 0) + c * calls[descriptor_id(fn)]
    order = sorted(classes.items(), key=lambda kv: len(kv[1].representative.gates))
    if a.sample:
        keep = set(sorted(classes, key=lambda sh: hashlib.sha256(f"{a.seed}/{sh}".encode()).digest())[:a.sample])
        order = [kv for kv in order if kv[0] in keep]
    largest = {}
    for sh, cl in order:
        largest[cl.definition.id if hasattr(cl.definition, "id") else cl.definition] = sh
    for s, cl in order:
        if s in done or (a.only and not any(s.startswith(p) for p in a.only.split(","))):
            continue
        rec: dict[str, Any] = {"shape": s, "definition": cl.representative.definition, "unit": cl.representative.unit,
                               "units": cl.units, "row_units": row_units.get(s), "word_gates": len(cl.representative.gates), "in_bits": sum(cl.representative.in_widths),
                               "out_bits": sum(cl.representative.out_widths)}
        t0 = time.time()
        cc = PU.lower_class(units.activation(cl.definition), cl.representative)
        if cc.circuit is None:
            rec["broke"] = {"at": "lowering", "why": cc.reason, "missing": list(cc.missing)}
        elif cc.counts["ands"] > a.max_ands:
            rec["broke"] = {"at": "size", "why": f"{cc.counts['ands']} ANDs > --max-ands {a.max_ands}"}
            rec["ands"] = cc.counts["ands"]
        else:
            try:
                lanes_ = PU.lanes(units.activation(cl.definition), [cl.representative], key, a.n)[(cl.representative.definition, cl.representative.unit)]
                d = out / s[:16]
                rec["ands"] = cc.counts["ands"]
                rec["table_reads"] = dict(Counter(t for t, _i, _w in cc.circuit.lookups))
                rec["stage"] = stage(units, a.partition, s, cl, lanes_, d, salt_key)
                rec["stage_py_maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
                rec["stage_s"] = round(time.time() - t0, 3)
                if not a.selftest_only:
                    rec["live"] = prove(a.binary, d, rec["stage"]["n"], a.port)
                if a.selftest and (len(done) % a.selftest == 0 or s == largest.get(cl.definition.id)):
                    if rec["table_reads"]:                       # a read-bearing circuit re-parses per case: a few cases, one process each
                        runs = [prove(a.binary, d, rec["stage"]["n"], a.port, selftest=c) for c in READ_CASES]
                        rec["selftest"] = {"all_pass": all(x["all_pass"] for x in runs), "cases": len(runs),
                                           "failed": [c for c, x in zip(READ_CASES, runs) if not x["all_pass"]], "subset": list(READ_CASES)}
                    else:
                        rec["selftest"] = prove(a.binary, d, rec["stage"]["n"], a.port, selftest=True)
                for f in d.glob("*.bin"):
                    f.unlink()
            except Exception as e:  # noqa: BLE001  (a class that breaks is a result, recorded with where it broke)
                rec["broke"] = {"at": "statement or prover", "why": f"{type(e).__name__}: {e}"[:400]}
        rec["seconds"] = round(time.time() - t0, 3)
        with results.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        done.add(s)
        print(json.dumps({k: rec.get(k) for k in ("shape", "definition", "units", "word_gates", "broke")} |
                         ({"accepted": rec["live"].get("accepted"), "prove_s": rec["live"].get("prove_total_s"),
                           "proof_bytes": rec["live"].get("proof_bytes"), "k_log": rec["live"].get("k_log")} if "live" in rec else {})), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
