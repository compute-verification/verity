"""A unit class of any partition (`verity.ir.units`) as a ``verity/flock-circuit`` statement (M0, :mod:`circuit`), proved by
``flock-circuit`` per class.

    python -m verity_flock.class_statement --model tiny --partition q-word --out DIR [--binary PATH] [--n 16] [--selftest]

M0's statement proves a template's identical units with every instance's input rows hashed in the circuit
(``hm96-sha512/row/v1`` under frame-v3-sha512 trees). A class becomes such a template here:

* one unit per instance: the class's circuit (`partition_units.lower_class`), its inputs packed into one u16 row port per source
  (`sources`: a Call parameter it reads, or a class of producing units whose committed outputs it reads; a word of w bits takes
  ceil(w / 16) consecutive u16 leaves, low half first; each row padded with unread zero words to a whole number of 64-word SHA-512
  blocks, as frame-v3 rows are), its outputs into one u16 output port ``out`` (each output padded with zero bits to whole u16 words);
* each source's row is committed once and opened by every instance that reads it (M0's shared rows), rows being keyed by the
  identity of the values they hold, not by value (`class_lanes`);
* the instances are every unit of the class in its flat Definition (`class_lanes`: synthetic inputs lifted through the flat
  Definition, a Call's broadcast parameters drawn once per Call), the statement binds the partition's ``verity/partition/v1``
  digest and the program's SHA-512 (an owner-function partition, which no record can name, binds the SHA-512 of its description
  under the tag ``verity/flock/experimental-owners``).

This is a synthetic layout, not the scheme's layout rule over the partition (which is not specified yet), and M0's statement
publishes the declared outputs, where a partition's unit outputs are committed values.  Every class is one plain
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
import re
import resource
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from verity.ir import partition_object as PO
from verity.ir.types import Array, Value
from verity.ir.cut import DefinitionCut
from verity.ir.units import Activation, PartitionUnits, UnitCircuit

from . import circuit as FC
from . import class_sweep
from . import gf2
from . import ir_lower as IL
from . import partition_units as PU

BLOCK_WORDS = 64
#: the selftest cases run on a read-bearing class (each re-parses its 1-3 GB circuit): the honest session, and the two witness tamperings
READ_CASES = ("honest", "unit_internal_bit_flipped", "unit_input_differs_from_its_message_bits")
#: and with --selftest-gpu, the GPU prover's paths against each other and against the CPU prover (byte-identical proofs)
GPU_CASES = ("gpu_paths_agree", "gpu_proofs_match_cpu")
EXPERIMENTAL = "verity/flock/experimental-owners"


def _words(w: int) -> int:
    return -(-w // 16)


def _pad(n: int, to: int) -> int:
    return max(to, -(-n // to) * to)


@dataclass
class ClassDefinition:
    """What M0's writers read of a Definition: one ``u16`` row parameter per input source and one ``u16`` result array."""

    id: str
    port_words: tuple[int, ...]
    w_out: int

    @property
    def w_in(self) -> int:
        return sum(self.port_words)

    def signature(self):
        return tuple((f"p{i}", Array(w, Value(16))) for i, w in enumerate(self.port_words)), Array(self.w_out, Value(16))


def sources(act: Activation, uc: UnitCircuit, owner_of: dict[int, str]) -> list[list[int]]:
    """The unit's inputs grouped by source, in first-read order: one group per Call parameter it reads leaves of, and one per class
    (shape) of producing units whose committed outputs it reads (``owner_of``: computing gate -> its unit's shape in the flat cut),
    as serving commits a tensor once, not per element. Each group is one row, and every instance reading the same values opens
    the same row."""
    leaf_param = []
    for i, (_n, t) in enumerate(act.fn.signature()[0]):
        leaf_param += [i] * t.leaves
    groups: dict[tuple, list[int]] = {}
    for t, k in enumerate(uc.inputs):
        key = ("p", leaf_param[-1 - k]) if k < 0 else ("u", owner_of.get(k, -1))
        groups.setdefault(key, []).append(t)
    return list(groups.values())


def class_lowering(act: Activation, uc: UnitCircuit, name: str, groups: list[list[int]] | None = None) -> IL.Lowering:
    """The class's unit on u16 ports: one row port per source group (``groups``: input positions; default one row for all), each
    padded to whole 64-word SHA-512 blocks, and ``out0 .. out{W_out-1}``."""
    groups = groups or [list(range(len(uc.inputs)))]
    port_words = tuple(_pad(sum(_words(uc.in_widths[t]) for t in g), BLOCK_WORDS) for g in groups)
    C = gf2.Circuit()
    leaves = []
    for i, w in enumerate(port_words):
        leaves += [C.input(f"in{len(leaves) + j}", 16) for j in range(w)]
    wires, base = {}, 0
    for g, w_port in zip(groups, port_words):
        at = base
        for t in g:
            w = uc.in_widths[t]
            wires[act.absolute(uc.inputs[t])] = [b for word in leaves[at:at + _words(w)] for b in word][:w]
            at += _words(w)
        base += w_port
    IL.lower_gates(C, [act.gate(j) for j in uc.gates], wires)
    out_bits = []
    for j, w in zip(uc.outputs, uc.out_widths):
        out_bits += list(wires[act.absolute(j)]) + [gf2.ZERO] * (16 * _words(w) - w)
    w_out = max(1, len(out_bits) // 16)
    out_bits += [gf2.ZERO] * (16 * w_out - len(out_bits))
    for j in range(w_out):
        C.output(f"out{j}", out_bits[16 * j:16 * (j + 1)])
    w_in = sum(port_words)
    units = IL.Units(target=None, in_widths=[16] * w_in, out_widths=[16] * w_out, in_src=[[("leaf", p) for p in range(w_in)]],
                     out_src=[[("ret", p) for p in range(w_out)]], signature=(uc.shape,), gates=[], arg_gates=[], ret_gates=[])
    lay = IL.layout(C, [f"in{j}" for j in range(w_in)], [f"out{j}" for j in range(w_out)])
    return IL.Lowering(name, ClassDefinition(name, port_words, w_out), units, C, lay)


def pack(uc: UnitCircuit, ins: Sequence[Sequence[int]], want: Sequence[Sequence[int]], w_in: int, w_out: int,
         groups: list[list[int]] | None = None, port_words: Sequence[int] | None = None):
    """The lanes as each port's rows ``{p{i}: (n, words)}``, all ports side by side ``(n, w_in)``, and outputs ``(n, w_out)``, u16, in
    the packing `class_lowering` reads."""
    groups = groups or [list(range(len(uc.inputs)))]
    port_words = port_words or [w_in]
    n = len(ins[0]) if ins else len(want[0])
    rows = np.zeros((n, w_in), np.uint16)
    base = 0
    for g, wp in zip(groups, port_words):
        at = base
        for t in g:
            v = np.asarray(ins[t], dtype=np.uint64)
            for h in range(_words(uc.in_widths[t])):
                rows[:, at + h] = ((v >> np.uint64(16 * h)) & np.uint64(0xFFFF)).astype(np.uint16)
            at += _words(uc.in_widths[t])
        base += wp
    outs = np.zeros((n, w_out), np.uint16)
    at = 0
    for vals, w in zip(want, uc.out_widths):
        v = np.asarray(vals, dtype=np.uint64)
        for h in range(_words(w)):
            outs[:, at + h] = ((v >> np.uint64(16 * h)) & np.uint64(0xFFFF)).astype(np.uint16)
        at += _words(w)
    per_port, base = {}, 0
    for i, wp in enumerate(port_words):
        per_port[f"p{i}"] = rows[:, base:base + wp]
        base += wp
    return rows, outs, per_port


def _shims(name: str, content: bytes, ports: int = 1):
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")  # noqa: E731
    sub = SimpleNamespace(id=f"unit-class/{name}", template=SimpleNamespace(id="unit-class"), inputs=[port(f"p{i}") for i in range(ports)],
                          outputs=[port("out")],
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
    FC._expanded(IL.layout(c3, ["cv0", "msg0", "cv1", "msg1", "cv2", "msg2"], ["out0", "out1", "out2"]), FC.SHA_NET)   # as compose asks
    FC._expanded(IL.layout(hm, ["cv", "pad", "y"], ["bc"]), FC.HM_NET)


#: token-select Calls (the samplers): `--select` leaves them out
SAMPLER = class_sweep.SAMPLER


def distinct_calls(graph_path: str | None, modules: str | None = None, definitions: dict[str, int] | None = None,
                   exclude: re.Pattern | None = None) -> tuple[Any, dict[str, int], list[dict]]:
    """A served row's distinct Call Definitions, from its program graph (``vllm-program-graph/v1``): (a Program with one Call of
    each, the row's Call count of each by descriptor id, a provenance row per Definition). ``modules``: a module-path prefix
    (``model.model.layers.0.``) to keep. Each Definition is bound from the registry and its v0 encoding's SHA-256 is checked
    against the graph's ``definitions_index`` where the graph has it (every group's spec; attention's smallest and largest T).
    A per-Call partition (``Q_word``, an owner function) cuts this Program's Calls exactly as the row's, so its classes are the
    row's and the row's unit counts are the per-Call counts times the Call counts. ``definitions`` ({descriptor id: Call count})
    adds Definitions without a graph (bound from the registry, no digest to check); ``exclude`` drops the ids it matches."""
    from circuit_check import targets as TG
    from verity.ir.codec import descriptor_id
    from verity.ir.defs import CompositeDefinition
    from verity.ir.program import Program
    from verity_vllm.pipeline.program_graph import _encoded, definition_digest

    TG.load_registries()
    g = json.loads(open(graph_path).read()) if graph_path else {"groups": []}
    index = g.get("definitions_index") or {}
    calls: dict[str, int] = dict(definitions or {})
    for fid, n in class_sweep.graph_calls(g, modules).items():
        calls[fid] = calls.get(fid, 0) + n
    if exclude is not None:
        calls = {fid: n for fid, n in calls.items() if not exclude.match(fid)}
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


_BINDINGS: dict = {}


def partition_binding(units: PartitionUnits, spec: str) -> tuple[str, str]:
    """(partition digest, program SHA-512) the statement binds, computed once per partition."""
    if (id(units), spec) in _BINDINGS:
        return _BINDINGS[id(units), spec]
    program = PO.program_sha512(units.program)
    if units.query is not None:
        out = PO.digest(PO.build(units.program, units.query)), program
    else:
        out = hashlib.sha512(EXPERIMENTAL.encode() + b"\0" + json.dumps({"program": program, "owners": spec}, sort_keys=True).encode()).hexdigest(), program
    _BINDINGS[id(units), spec] = out
    return out


_OWNERS: dict = {}


def owners_of(units: PartitionUnits, fn_id: str) -> dict[int, str]:
    """Computing gate -> its unit's shape, in the flat Definition's cut (structure gates, copied into every reader, have none)."""
    if (id(units), fn_id) not in _OWNERS:
        act = units.activation_of(fn_id)
        keep = set(act.graph.keep)
        _OWNERS.clear()
        _OWNERS[id(units), fn_id] = {j: uc.shape for uc in units._flat[fn_id][1] for j in uc.gates if j in keep}
    return _OWNERS[id(units), fn_id]


def broadcast_params(units: PartitionUnits) -> dict[str, tuple[set[int], int]]:
    """Per flat Definition that is a member of a separable Call: the parameters its batch node shares across members (axis None) and
    the member count, as the Call evaluates it (a GEMM coordinate's activation row, read by every coordinate of the Call)."""
    out: dict[str, tuple[set[int], int]] = {}

    def walk(cut):
        if not (isinstance(cut, DefinitionCut) and cut.parts is not None):
            return
        body = cut.fn.body
        for k, (child, m) in enumerate(cut.parts):
            if child is None:
                continue
            node = body.nodes[k]
            if node.form == "batch" and child.fn.id not in out:
                out[child.fn.id] = ({i for i, ax in enumerate(node.axes) if ax is None}, m)
            walk(child)

    for _k, _m, _fn, cut in units._calls():
        walk(cut)
    return out


def class_lanes(units: PartitionUnits, cl, key, n: int, shared: dict[str, tuple[set[int], int]]) -> tuple[list, list, list]:
    """``n`` instances of the class: every unit of that shape in its flat Definition, from as many evaluations as it takes, each
    evaluation's parameters uniform words, except those the Call shares across members (``shared``), which are drawn once per Call
    (group of members). Returns (inputs, outputs, rows): per input position every instance's word, and per source group
    (`sources` of the representative) every instance's row identity: the leaves it reads, and the Call they belong to when all
    are parameters the Call broadcasts, else the evaluation. Instances with one identity open one committed row."""
    act = units.activation(cl.definition)
    mine = [uc for uc in units._flat[cl.definition.id][1] if uc.shape == cl.shape]
    widths = [act.width(-1 - p) for p in range(act.params)]
    leaf_param = []
    for i, (_nm, t) in enumerate(act.fn.signature()[0]):
        leaf_param += [i] * t.leaves
    bcast, members = shared.get(cl.definition.id, (set(), 1))
    groups = sources(act, cl.representative, owners_of(units, cl.definition.id))
    reads = [[(all(x < 0 and leaf_param[-1 - x] in bcast for x in sub), sub) for sub in (tuple(uc.inputs[t] for t in g) for g in groups)]
             for uc in mine]
    ins = [[] for _ in mine[0].inputs]
    want = [[] for _ in mine[0].outputs]
    rows = [[] for _ in groups]
    have, t = 0, 0
    while have < n:
        call = t // max(1, members)
        raw = PU.synthetic_words(key, f"{act.fn.id}#{t}", widths, 1)[0]
        grp = PU.synthetic_words(key, f"{act.fn.id}#call{call}", widths, 1)[0]
        words = [grp[p] if leaf_param[p] in bcast else raw[p] for p in range(len(raw))]
        run = PU.evaluate(act, words)
        for uc, per in zip(mine, reads):
            if have >= n:
                break
            for port, x in zip(ins, uc.inputs):
                port.append(run[x])
            for port, j in zip(want, uc.outputs):
                port.append(run[j])
            for ids, (b, sub) in zip(rows, per):
                ids.append((b, call if b else t, sub))
            have += 1
        t += 1
    return ins, want, rows


def stage(units: PartitionUnits, spec: str, shape: str, cl, lanes_: tuple, out_dir: Path, salt_key: bytes) -> dict:
    """``circuit.txt``, the prover's ``inst-n.bin`` and the verifier's ``pub-n.bin`` for one class: one row port per input source,
    and each source's row committed once, shared by every instance that reads it (``lanes_`` from `class_lanes`; lanes without
    source identities share equal rows)."""
    act = units.activation(cl.definition)
    uc = cl.representative
    name = shape[:16]
    ins, want, idents = lanes_ if len(lanes_) == 3 else (*lanes_, None)
    partition, program = partition_binding(units, spec)
    groups = sources(act, uc, owners_of(units, cl.definition.id))
    low = class_lowering(act, uc, f"unit-class/{name}", groups)
    sub, iset = _shims(name, uc.form.encode(), len(groups))
    rows, outs, per_port = pack(uc, ins, want, low.definition.w_in, low.definition.w_out, groups, low.definition.port_words)
    got, _cut, ok = low.evaluate(rows.astype(np.uint64))
    text = FC.compose(low, sub, partition=partition, program=program)
    if not int(ok) == (1 << len(rows)) - 1 or not np.array_equal(got.astype(np.uint16), outs):
        raise AssertionError(f"class {name}: the laid-out unit rows disagree with the reference on its lanes")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "circuit.txt").write_text(text)
    n = len(rows)
    tables = refs = None
    if idents is not None:
        if len(idents) != len(groups):
            raise AssertionError(f"class {name}: the lanes' source groups are not the statement's")
        tables, refs = {}, {}
        for i, ids in enumerate(idents):
            index: dict[tuple, int] = {}
            at: list[int] = []
            for k, ident in enumerate(ids):
                if ident not in index:
                    index[ident] = len(at)
                    at.append(k)
            ref = np.asarray([index[ident] for ident in ids])
            tables[f"p{i}"], refs[f"p{i}"] = per_port[f"p{i}"][at], ref
            if not np.array_equal(tables[f"p{i}"][ref], per_port[f"p{i}"]):
                raise AssertionError(f"class {name}: instances reading one source disagree on its row")
    head = FC.write(text, iset, 0, n, out_dir / f"inst-{n}.bin", out_dir / f"pub-{n}.bin", salt_key, rows=per_port, outs={"out": outs},
                    share_rows=True, tables=tables, refs=refs)
    meta = json.loads(text.split("\n", 2)[1][5:])
    return {"n": n, "unit_nnz": low.layout.counts["nnz_a"] + low.layout.counts["nnz_b"], "circuit_bytes": len(text), "w_in": low.definition.w_in,
            "w_out": low.definition.w_out, "unit_rows": low.layout.useful, "ports": list(low.definition.port_words),
            "shared_rows": head.get("shared_rows"), "ands": low.layout.counts["ands"], "k_log": meta["k_log"],
            "vus_per_block": meta["vus_per_block"], "circuit_sha512": FC.digest(text), "class_sha512": FC.class_digest(text)}


def prove(binary: str, d: Path, n: int, port: int, selftest: bool | str = False, timeout: int = 3600, gpu: bool = False,
          warm: int = 0, runs: int = 1) -> dict:
    """A loopback session (``serve`` on 127.0.0.1 and ``prove``), or ``selftest``: the LIVE or SELFTEST record."""
    circ, inst, pub = d / "circuit.txt", d / f"inst-{n}.bin", d / f"pub-{n}.bin"
    if selftest:
        cmd = [binary, "selftest", "--instances", str(inst), "--circuit", str(circ)] + (["--gpu"] if gpu else [])
        r = subprocess.run(cmd + (["--only", selftest] if isinstance(selftest, str) else []), capture_output=True, text=True, timeout=timeout)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("SELFTEST", "NEG"))]
        neg = [json.loads(ln.split("\t", 1)[1]) for ln in lines if ln.startswith("NEG")]
        head = next((json.loads(ln.split("\t", 1)[1]) for ln in lines if ln.startswith("SELFTEST")), None)
        failed = [c.get("case") for c in neg if c.get("pass") is not True]
        all_pass = head.get("all_pass") if head is not None else (bool(neg) and not failed)     # --only prints its one case, no summary
        return {"all_pass": all_pass, "cases": len(neg), "failed": failed} | \
            ({"gpu_cases": gc} if (gc := [c for c in neg if c.get("case") in GPU_CASES]) else {})
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
                                           "--runs", str(runs), "--warm", str(warm)] + (["--gpu"] if gpu else []), stdout=fo, stderr=fe)
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
    timed = [x for x in live if not x.get("warm")] or live
    v = sorted(timed, key=lambda x: x.get("prove_total_s") or 0)[len(timed) // 2]      # the median timed run
    return {k: v.get(k) for k in ("accepted", "why", "prover", "instances", "vus_per_block", "k_log", "nbl", "dense_m", "e2e_s",
                                  "witness_s", "prove_s", "prove_total_s", "proof_bytes", "wire_sent", "wire_recv", "statement_digest",
                                  "buckets")} | \
        {"prover_peak_gb": round(prover_gb, 2) if prover_gb else None, "verifier_peak_gb": verifier_gb,
         "verify_s": (v.get("verdict") or {}).get("verify_s"), "timed_runs": len(timed),
         "prove_total_s_runs": [x.get("prove_total_s") for x in timed]}


def _vm_hwm(pid: int) -> float | None:
    """A live process's peak resident memory (VmHWM), GB."""
    try:
        for ln in open(f"/proc/{pid}/status"):
            if ln.startswith("VmHWM:"):
                return round(int(ln.split()[1]) / 2**20, 2)
    except OSError:
        return None
    return None


#: what `_stage_job` reads: set once in `main` before the workers fork
_CTX: dict = {}
#: AND-instances per prove under ``--batch auto``
BATCH_ANDS = 1 << 29


def auto_batch(ands: int, row_units: int) -> int:
    cap = min(1024, row_units, BATCH_ANDS // max(1, ands))
    return max(16, 1 << max(0, cap.bit_length() - 1))


def _stage_job(s: str) -> dict:
    """Lower, lift the lanes of and stage one shape: its record, with ``stage`` when it staged, else ``broke``."""
    c = _CTX
    units, cl, a = c["units"], c["classes"][s], c["a"]
    uc = cl.representative
    rec: dict[str, Any] = {"shape": s, "definition": uc.definition, "unit": uc.unit, "units": cl.units, "row_units": c["row_units"].get(s),
                           "word_gates": len(uc.gates), "in_bits": sum(uc.in_widths), "out_bits": sum(uc.out_widths)}
    t0 = time.time()
    cc = PU.lower_class(units.activation(cl.definition), uc)
    if cc.circuit is None:
        rec["broke"] = {"at": "lowering", "why": cc.reason, "missing": list(cc.missing)}
    elif (a.select == "table-free" and cc.circuit.lookups) or (a.select == "reads" and not cc.circuit.lookups):
        rec["skipped"] = "reads" if cc.circuit.lookups else "table-free"
        rec["ands"] = cc.counts["ands"]
        rec["table_reads"] = dict(Counter(t for t, _i, _w in cc.circuit.lookups))
    elif cc.counts["ands"] > a.max_ands:
        rec["broke"] = {"at": "size", "why": f"{cc.counts['ands']} ANDs > --max-ands {a.max_ands}"}
        rec["ands"] = cc.counts["ands"]
    else:
        try:
            rec["ands"] = cc.counts["ands"]
            rec["table_reads"] = dict(Counter(t for t, _i, _w in cc.circuit.lookups))
            n = auto_batch(cc.counts["ands"], c["row_units"].get(s) or cl.units) if a.batch == "auto" else a.n
            lanes_ = class_lanes(units, cl, c["key"], n, c["shared"])
            rec["stage"] = stage(units, a.partition, s, cl, lanes_, c["out"] / s[:16], c["salt_key"])
            rec["stage_py_maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
        except Exception as e:  # noqa: BLE001  (a class that breaks is a result, recorded with where it broke)
            rec.pop("stage", None)
            rec["broke"] = {"at": "statement", "why": f"{type(e).__name__}: {e}"[:400]}
    rec["stage_s"] = rec["stage_wall_s"] = round(time.time() - t0, 3)
    return rec


def _staged(jobs: list[str], workers: int):
    """`_stage_job` over ``jobs``, records in order: in this process, or in ``workers`` forked processes at most two shapes each
    ahead of the prover. A worker that dies (the OOM killer) breaks the pool: the shapes in flight are staged again in a new one,
    and a shape lost twice is recorded as broken."""
    if workers <= 1:
        yield from map(_stage_job, jobs)
        return
    import multiprocessing as mp
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    queue, lost = list(reversed(jobs)), Counter()
    while queue:
        live: deque = deque()
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            try:
                while queue or live:
                    while queue and len(live) < 2 * workers:
                        s = queue.pop()
                        live.append((s, ex.submit(_stage_job, s)))
                    rec = live[0][1].result()
                    live.popleft()
                    yield rec
            except BrokenProcessPool:
                for s, _f in reversed(live):
                    lost[s] += 1
                    if lost[s] < 2:
                        queue.append(s)
                    else:
                        cl = _CTX["classes"][s]
                        yield {"shape": s, "definition": cl.representative.definition, "units": cl.units, "word_gates": len(cl.representative.gates),
                               "broke": {"at": "statement", "why": "its staging worker died twice (killed, likely out of memory)"}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", choices=["tiny"])
    src.add_argument("--definition")
    src.add_argument("--program-module", help="MODULE:FUNC (FUNC() returns a Program); MODULE may be a .py path")
    src.add_argument("--program-graph", help="a served row's program graph: its distinct Call Definitions (distinct_calls)")
    src.add_argument("--definitions", help="a JSON file {descriptor id: Call count}: those Call Definitions, one Call of each (distinct_calls)")
    ap.add_argument("--modules", default=None, help="with --program-graph: keep groups whose module path starts with this")
    ap.add_argument("--partition", default="q-word")
    ap.add_argument("--out", required=True)
    ap.add_argument("--binary", default=os.environ.get("FLOCK_CIRCUIT", "flock-circuit"))
    ap.add_argument("--selftest-binary", default=None, help="the flock-circuit build with seed injection, for selftests (a proving "
                    "build never has it; default --binary)")
    ap.add_argument("--n", type=int, default=16, help="instances per class statement (synthetic lanes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", type=int, default=0, help="also run flock-circuit's selftest (31 cases, negatives included) on every "
                    "Nth provable class and on each flat Definition's largest (0: never)")
    ap.add_argument("--selftest-gpu", action="store_true", help="run the selftests on the GPU (selftest --gpu; a read-bearing class's "
                    "subset adds GPU_CASES)")
    ap.add_argument("--selftest-cases", default=None, help="comma-separated selftest cases to run instead (one process each)")
    ap.add_argument("--max-ands", type=int, default=1 << 24, help="skip classes above this many ANDs (reported)")
    ap.add_argument("--only", default=None, help="comma-separated shape prefixes")
    ap.add_argument("--selftest-only", action="store_true", help="stage and run the selftest cases only (no loopback session), into selftests.jsonl")
    ap.add_argument("--sample", type=int, default=0, help="prove only this many classes, drawn by a seeded hash of their shapes (0: all)")
    ap.add_argument("--port", type=int, default=7610)
    ap.add_argument("--gpu", action="store_true", help="prove on the GPU (flock-circuit prove --gpu)")
    ap.add_argument("--warm", type=int, default=0)
    ap.add_argument("--runs", type=int, default=1, help="timed runs per shape; the record is the median")
    ap.add_argument("--shapes", default=None, help="a file of shape digests (or prefixes), one per line: prove only these")
    ap.add_argument("--batch", default=None, help="'auto': instances per shape = the least of 1024, the row's units of it and 2^29 / its "
                    "ANDs, as a power of two >= 16 (overrides --n)")
    ap.add_argument("--jobs", type=int, default=1, help="shapes staged in parallel (forked workers); proving stays one at a time")
    ap.add_argument("--select", choices=["table-free", "reads", "units"], default=None,
                    help="leave out the token-select Calls, and prove only the shapes whose unit reads no pinned table (table-free), "
                    "reads one (reads), or either (units); the others are recorded as skipped, unstaged")
    ap.add_argument("--protocol", default="sweep-v0", help="the measurement protocol the result is labelled with (sweep-v0: never a "
                    "table cell)")
    a = ap.parse_args(argv)
    calls: dict[str, int] | None = None
    if a.program_graph or a.definitions:
        defs = json.loads(Path(a.definitions).read_text()) if a.definitions else None
        program, calls, prov = distinct_calls(a.program_graph, a.modules, defs, SAMPLER if a.select else None)
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
    wanted = [ln.strip() for ln in Path(a.shapes).read_text().split() if ln.strip()] if a.shapes else None
    if a.sample:
        keep = set(sorted(classes, key=lambda sh: hashlib.sha256(f"{a.seed}/{sh}".encode()).digest())[:a.sample])
        order = [kv for kv in order if kv[0] in keep]
    largest = {}
    for sh, cl in order:
        largest[cl.definition.id if hasattr(cl.definition, "id") else cl.definition] = sh
    jobs = [s for s, _cl in order if not (s in done or (a.only and not any(s.startswith(p) for p in a.only.split(",")))
                                          or (wanted is not None and not any(s.startswith(p) for p in wanted)))]
    _CTX.update(units=units, classes=classes, row_units=row_units, shared=broadcast_params(units), key=key, salt_key=salt_key,
                out=out, a=a)
    partition_binding(units, a.partition)
    for rec in _staged(jobs, a.jobs):
        s, cl = rec["shape"], classes[rec["shape"]]
        t0 = time.time() - rec.pop("stage_wall_s", 0)
        if "stage" in rec:
            d = out / s[:16]
            try:
                if not a.selftest_only:
                    rec["live"] = prove(a.binary, d, rec["stage"]["n"], a.port, gpu=a.gpu, warm=a.warm, runs=a.runs)
                if a.selftest and (len(done) % a.selftest == 0 or s == largest.get(cl.definition.id)):
                    sb, g = a.selftest_binary or a.binary, a.selftest_gpu
                    if rec["table_reads"] or a.selftest_cases:   # a read-bearing circuit re-parses per case: a few cases, one process each
                        cases = tuple(a.selftest_cases.split(",")) if a.selftest_cases else READ_CASES + (GPU_CASES if g else ())
                        runs = [prove(sb, d, rec["stage"]["n"], a.port, selftest=c, gpu=g) for c in cases]
                        rec["selftest"] = {"all_pass": all(x["all_pass"] for x in runs), "cases": len(runs),
                                           "failed": [c for c, x in zip(cases, runs) if not x["all_pass"]], "subset": list(cases)} | \
                            ({"gpu_cases": gc} if (gc := [c for x in runs for c in x.get("gpu_cases", [])]) else {})
                    else:
                        rec["selftest"] = prove(sb, d, rec["stage"]["n"], a.port, selftest=True, gpu=g)
                    rec["selftest"]["gpu"] = g
            except Exception as e:  # noqa: BLE001  (a class that breaks is a result, recorded with where it broke)
                rec["broke"] = {"at": "prover", "why": f"{type(e).__name__}: {e}"[:400]}
            for f in d.glob("*.bin"):
                f.unlink()
            if (d / "circuit.txt").exists() and (d / "circuit.txt").stat().st_size > 1 << 28:
                (d / "circuit.txt").unlink()                    # the record keeps its digest; restaging rebuilds it byte for byte
        rec["seconds"] = round(time.time() - t0, 3)
        with results.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        done.add(s)
        print(json.dumps({k: rec.get(k) for k in ("shape", "definition", "units", "word_gates", "broke")} |
                         ({"accepted": rec["live"].get("accepted"), "prove_s": rec["live"].get("prove_total_s"),
                           "proof_bytes": rec["live"].get("proof_bytes"), "k_log": rec["live"].get("k_log")} if "live" in rec else {})), flush=True)
    write_result(results, a)
    return 0


def _device(gpu: bool) -> str | None:
    """The device that proved: the GPU's name (nvidia-smi), else the CPU model."""
    try:
        if gpu:
            return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                                  timeout=30).stdout.strip().splitlines()[0]
        return next(ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name"))
    except (OSError, IndexError, StopIteration, subprocess.SubprocessError):
        return None


def write_result(results: Path, a) -> None:
    """``$RESEARCH_RUN_DIR/result.json`` (research/result/v0.1) when run under ``research run``: the settings as the fingerprint, the
    shapes' counts and summed timings as measurements, and every shape's record (keyed by shape digest) in ``results.jsonl``, copied
    beside it as the measurement file."""
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    if not run_dir or not results.exists():
        return
    import shutil

    recs = [json.loads(ln) for ln in results.read_text().splitlines() if ln.strip()]
    shutil.copy(results, Path(run_dir, "class-sweep.jsonl"))
    ok = [r for r in recs if (r.get("live") or {}).get("accepted")]
    rejected = [r for r in recs if r.get("live") and not r["live"].get("accepted")]
    selftests = [r["selftest"] for r in recs if r.get("selftest")]
    doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"kind": "class-sweep/v1", "statement": FC.STATEMENT, "scheme": FC.SCHEME, "partition": a.partition,
                                    "gpu": a.gpu, "warm": a.warm, "runs": a.runs, "batch": a.batch or a.n, "seed": a.seed,
                                    "protocol": a.protocol, "select": a.select, "device": _device(a.gpu), "shapes": sorted(r["shape"] for r in recs)},
           "validation": {"status": "failed" if rejected or any(not x.get("all_pass") for x in selftests) else ("passed" if ok else "not_run"),
                          "checker": "flock-circuit loopback verifier (live coins) and selftest",
                          "detail": f"{len(ok)} of {len(recs)} shapes accepted, {len(rejected)} rejected, "
                                    f"{sum(1 for r in recs if r.get('broke'))} broke, {sum(1 for r in recs if r.get('skipped'))} skipped (--select); "
                                    f"selftests {sum(1 for x in selftests if x.get('all_pass'))}/{len(selftests)}",
                          "evidence": ["class-sweep.jsonl"]},
           "measurements": [{"name": "shapes", "value": len(recs), "unit": "count"},
                            {"name": "shapes_accepted", "value": len(ok), "unit": "count"},
                            {"name": "prove_total_s_sum", "value": round(sum(r["live"]["prove_total_s"] for r in ok), 3), "unit": "s"},
                            {"name": "proof_bytes_sum", "value": sum(sum(r["live"]["proof_bytes"]) for r in ok), "unit": "B"}],
           "measurement_files": ["class-sweep.jsonl"], "artifacts": []}
    tmp = Path(run_dir, "result.json.tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True))
    os.replace(tmp, Path(run_dir, "result.json"))


if __name__ == "__main__":
    raise SystemExit(main())
