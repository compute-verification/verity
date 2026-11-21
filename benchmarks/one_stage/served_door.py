"""A block of the served circuit of the class of record (Llama 3.2 1B, class 4,27,2, four requests, the RTX target, seed
7: circuits' `build_descriptor.py`) through the door, `proofs.verify`, on its one partition (`Q_template_instances`, the
lowering's own query) with blocks ``[0, u), [u, u + 1), [u + 1, population)``, ``u`` a template's first unit.

  python served_door.py build --tree TREE --out DIR                     # DIR/C.json, DIR/query.json
  python served_door.py step block|restrict|lower|door|one|stage --dir DIR --template ID
  python served_door.py step verify --dir DIR --template ID --unit U --lower EXE --lean EXE --session S...
  python served_door.py window --tree TREE --out DIR [--seed N]         # a staged window, for records_audit.py audit

``block``, ``restrict`` and ``lower`` are the door's own steps (`one_circuit.block`, `.restrict`, `.lower`), each in its
own process; ``door`` is `proofs.verify` on the drawn block, whose proof files (a public file naming the block's unit and
an empty session) and registered file (empty) carry the call to the first step that refuses, and are read by no step
before it; ``one`` is `TemplatePartition` under ``Q_template_instance`` of the template alone; ``stage`` stages the block's
proof on a STAGED witness (`stage`) and ``verify`` puts its proved session through the door. Each prints one
``STEP {json}`` line."""
from __future__ import annotations

import argparse
import bisect
import dataclasses
import hashlib
import json
import resource
import sys
import time
from pathlib import Path


def _peak_gib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


def build(a) -> int:
    tree, out = Path(a.tree), Path(a.out)
    sys.path.insert(0, str(tree / "tests/adapters/vllm"))
    from verity.circuits.codec import descriptor_chunks
    from verity_vllm.program.predict import window_circuit as WCirc
    from verity_vllm.program.predict import window_class as WC
    from tests.program.predict import test_steps as S
    t = time.perf_counter()
    cfg = json.loads((tree / "tests/adapters/vllm/tests/program/predict/fixtures/llama32-1b.config.json").read_text())
    reg = S.Registered.of(S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target=dict(S.RTXPRO6000)),
                          default_seed=7)
    circuit = WCirc.lower(reg, WC.WindowClass(4, 27, 2), None, requests=[WC.Request(0, j, 27, 2) for j in range(4)]).circuit
    program, query = circuit.program()
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "C.json", "wb") as f:
        for ch in descriptor_chunks(program):
            f.write(ch)
            n += len(ch)
    (out / "query.json").write_text(json.dumps(query))
    print("STEP " + json.dumps({"step": "build", "wall_s": round(time.perf_counter() - t, 1), "peak_gib": _peak_gib(),
                                "C_bytes": n, "root_nodes": len(program.root.body.nodes)}), flush=True)
    return 0


def _program(c: bytes):
    from verity.circuits import restrict as R
    return R.program(c)


def _partition(d: Path, template: str):
    """The served partition with the template's first unit a block of its own, and that block's index."""
    from verity.circuits import partition_object as PO
    p = d / f"partition-levels-{hashlib.sha256(template.encode()).hexdigest()[:16]}.json"
    if p.exists():
        j = json.loads(p.read_text())
        return j["partition"], j["k"], j["u"], j["population"]
    c = (d / "C.json").read_bytes()
    query = json.loads((d / "query.json").read_text())
    program = _program(c)
    tp = PO.TemplatePartition(program, query)
    u = next(a for _k, t, a, n in tp.ranges if t == template and n)
    blocks = [b for b in ((0, u), (u, u + 1), (u + 1, tp.population)) if b[0] < b[1]]
    part = PO.build(json.loads(c), query, [blocks])
    k = blocks.index((u, u + 1))
    p.write_text(json.dumps({"partition": part, "k": k, "u": u, "population": tp.population}))
    return part, k, u, tp.population


def _ports(c: bytes, query: dict, u: int) -> dict:
    """Unit ``u``'s parameter leaves by what each reads: another unit's output, a registered input, a run input."""
    from verity.circuits import partition_object as PO
    from verity.circuits.program import REGISTERED_ID, is_input
    program = _program(c)
    tp = PO.TemplatePartition(program, query)
    body = program.root.body
    offs = list(body.offsets)
    unit_of = {k: (a, t) for k, t, a, _n in tp.ranges}
    counts = {"wired": 0, "registered_input": 0, "run_input": 0}
    writers: dict[str, int] = {}
    for g in tp.inputs(u):
        k = bisect.bisect_right(offs, g) - 1
        fn = body.nodes[k].fn
        if is_input(fn):
            counts["registered_input" if REGISTERED_ID.fullmatch(fn.id) else "run_input"] += 1
        else:
            a, t = unit_of[k]
            counts["wired"] += 1
            writers[t] = writers.get(t, 0) + 1
    return {**counts, "writers": writers, "leaves": len(tp.inputs(u))}


def _rank(tp, v: int) -> tuple[str, int]:
    """Unit ``v``'s template and its leaf in the template's output field's tree: its rank among the template's units in
    root order (Lean's `HmLinks.treeLeaf`)."""
    t = next(t for _k, t, a, n in tp.ranges if a <= v < a + n)
    before = sum(min(n, max(0, v - a)) for _k, tt, a, n in tp.ranges if tt == t)
    return t, before


def _bits(widths: list[int], values: list[int]) -> list[int]:
    return [(x >> j) & 1 for x, w in zip(values, widths) for j in range(w)]


def _words16(bits: list[int], words: int) -> list[int]:
    bits = bits + [0] * (16 * words - len(bits))
    return [sum(bits[16 * w + j] << j for j in range(16)) for w in range(words)]


def _split_outs(meta: dict, row: list[int]) -> dict:
    """The output row ``row`` as `FC.write`'s ``outs``: each META out port's words, unpadded, in port order."""
    import numpy as np
    out, at = {}, 0
    for nm, w in meta["out_ports"]:
        out[nm] = np.asarray([row[at:at + w]], dtype=np.uint16)
        at += w
    return out


def stage(d: Path, template: str, seed: int = 7) -> dict:
    """A STAGED witness for the block of the template's first unit ``u``: every input port is a row of in(S) read from
    one writer (else refused here by name). Each output field ``j`` of a writer template is one tree ``<template>/o<j>``,
    a leaf for each of its units in root order (`_rank`) holding that field, its rows random but for nothing: the writer
    is not evaluated, so its rows are no served values. A port reads one field of its writer whole, ``j`` from C
    (`class_statement.field_row`). A port whose leaves are one run of a run input node ``w`` reads row ``r`` of
    the tree ``input/n<w>``, the node's leaves in rows of the port's, random at ``r`` and zero elsewhere. A port whose leaves are no one field of its writer whole is refused: no leaf of a
    field's tree holds its row. ``u``'s output is the template evaluated on those rows (`reference.evaluate`), at ``u``'s
    leaf of its own template's tree, the other leaves zero. Writes ``stage-u/``: the prover's and verifier's files
    (`FC.write`, as `credited_door` stages a block) and the verifier's position-free ``--registered`` file."""
    import random
    from types import SimpleNamespace
    import numpy as np
    from verity.circuits import partition_object as PO
    from verity.circuits.evaluation.reference import evaluate
    from verity.circuits.program import REGISTERED_ID, is_input
    from verity.commitments import registered_value as RV
    from verity_flock import circuit as FC
    from verity_flock import class_statement as CS
    part, k, u, pop = _partition(d, template)
    c = (d / "C.json").read_bytes()
    text = (d / f"circuit-{u}.txt").read_text()
    meta = FC.meta_of(text)
    program = _program(c)
    tp = PO.TemplatePartition(program, json.loads((d / "query.json").read_text()))
    fn = program.defs[template]
    params, ret = fn.signature()
    widths = {p: t.leaf_widths() for p, t in params}
    gates = list(tp.inputs(u))
    rng = random.Random(seed)
    owner_cache: dict[int, list[int]] = {}
    offs = list(program.root.body.offsets)
    unit_of = {kk: a for kk, _t, a, _n in tp.ranges}

    def writer(g: int) -> tuple[int, int] | None:
        kk = bisect.bisect_right(offs, g) - 1
        if kk not in unit_of:
            return None
        lo = unit_of[kk]
        n = next(n for k2, _t, a, n in tp.ranges if k2 == kk)
        first = owner_cache.setdefault(lo, list(tp.outputs(lo)))
        guess = lo + (g - min(first)) // max(1, len(first))
        for v in [guess] + list(range(lo, lo + n)):
            if not lo <= v < lo + n:
                continue
            outs = owner_cache.setdefault(v, list(tp.outputs(v)))
            if g in outs:
                return v, outs.index(g)
        return None

    rec: dict = {"unit": u, "block": k, "staged": "writer rows random, not served values", "ports": {}}
    trees: dict[str, dict[int, list[int]]] = {}
    reads: dict[str, tuple[str, int]] = {}
    fresh: dict[str, tuple[int, int, int, list[int]]] = {}
    tree_widths: dict[str, list[int]] = {}
    tree_units: dict[str, str] = {}
    full_out: dict[tuple[str, int], list[int]] = {}
    args, at = [], 0
    for p, t in params:
        n = len(widths[p])
        ws = [writer(g) for g in gates[at:at + n]]
        at += n
        if all(w is None for w in ws):
            g0 = gates[at - n]
            w = bisect.bisect_right(offs, g0) - 1
            fn_w = program.root.body.nodes[w].fn
            if not is_input(fn_w) or REGISTERED_ID.fullmatch(fn_w.id) or gates[at - n:at] != list(range(g0, g0 + n)):
                raise SystemExit(f"STAGE-REFUSED port {p}: its leaves are no run input node's one run of leaves")
            node_leaves = (offs[w + 1] if w + 1 < len(offs) else program.root.gates) - offs[w]
            if node_leaves % n or (g0 - offs[w]) % n:
                raise SystemExit(f"STAGE-REFUSED port {p}: run input node {w}'s {node_leaves} leaves are no rows of {n}")
            vals = [rng.getrandbits(x) for x in widths[p]]
            fresh[p] = (w, (g0 - offs[w]) // n, node_leaves // n, vals)
            rec["ports"][p] = {"run_input_node": w, "row": fresh[p][1], "rows": fresh[p][2], "leaves": n}
            args.append(vals)
            continue
        if any(w is None for w in ws):
            raise SystemExit(f"STAGE-REFUSED port {p}: some leaves read a circuit input and some a unit's output")
        v = ws[0][0]
        if any(w[0] != v for w in ws):
            raise SystemExit(f"STAGE-REFUSED port {p}: its {n} leaves read more than one writer: {ws[:6]}")
        tw, leaf = _rank(tp, v)
        wret = program.defs[tw].signature()[1]
        whole = wret.leaf_widths()
        at_w = [j for _v, j in ws]
        fields = [f[1] if isinstance(f, tuple) else f for f in wret.fields] if hasattr(wret, "fields") else [wret]
        try:
            _v, fj = CS.field_row(ws, lambda _x: wret)
        except ValueError:
            raise SystemExit(f"STAGE-REFUSED port {p}: its leaves {at_w[:6]} are no one output field of unit {v} of {tw} "
                             "whole: a row reads one field's tree") from None
        rec["ports"][p] = {"writer": v, "template": tw, "leaf": leaf, "leaves": n, "writer_leaves": len(whole),
                           "writer_fields": len(fields), "field": fj,
                           "reads_writer_leaves": at_w[:8] + (["..."] if n > 8 else [])}
        full = full_out.setdefault((tw, leaf), [rng.getrandbits(w) for w in whole])
        args.append([full[j] for j in at_w])
        f0 = sum(f.leaves for f in fields[:fj])
        tree = f"{tw}/o{fj}"
        reads[p] = (tree, leaf)
        trees.setdefault(tree, {})[leaf] = full[f0:f0 + fields[fj].leaves]
        tree_widths[tree] = fields[fj].leaf_widths()
        tree_units[tree] = tw
    from verity.circuits.codec import decode_program
    from verity_vllm.definitions.registry.catalog import load
    out = evaluate(decode_program(json.loads(c), load()[0]).defs[template], *args)
    ow = ret.leaf_widths()
    ports = {q[0]: q for q in meta["ports"]}
    rec["meta_ports"] = meta["ports"]
    prog = part["program"]
    regs = {}
    for tree, rows in trees.items():
        units = sum(n for _k, tt, _a, n in tp.ranges if tt == tree_units[tree])
        wn = next(ports[p][1] for p, (t2, _l) in reads.items() if t2 == tree)
        full = [_words16(_bits(tree_widths[tree], rows[i]), wn) if i in rows else [0] * wn for i in range(units)]
        regs[tree] = RV.Registered.commit(prog, tree, full, word_bits=16, schema=RV.SCHEMA_HM96_SHA512_ROW_V2)
    mine_units = sum(n for _k, tt, _a, n in tp.ranges if tt == template)
    _t, my = _rank(tp, u)
    on = ports["o0"][1]
    orow = _words16(_bits(ow, list(out)), on)
    orows = [orow if i == my else [0] * on for i in range(mine_units)]
    oreg = RV.Registered.commit(prog, f"{template}/o0", orows, word_bits=16, schema=RV.SCHEMA_HM96_SHA512_ROW_V2)
    for p, (w, r, rows, vals) in fresh.items():
        wn = ports[p][1]
        regs[p] = RV.Registered.commit(prog, f"input/n{w}", [_words16(_bits(widths[p], vals), wn) if i == r else [0] * wn
                                                              for i in range(rows)],
                                       word_bits=16, schema=RV.SCHEMA_HM96_SHA512_ROW_V2)
        reads[p] = (p, r)
    s = d / f"stage-{u}"
    (s / "tables").mkdir(parents=True, exist_ok=True)
    (s / "circuit.txt").write_text(text)
    arg = dict(zip((p for p, _t in params), args))
    tables = {p: np.asarray([_words16(_bits(widths[p], arg[p]), ports[p][1])], dtype=np.uint16) for p in reads}
    salts = {p: [regs[tw].salts[leaf]] for p, (tw, leaf) in reads.items()} | {"o0": [oreg.salts[my]]}
    registered = {p: regs[tw].statement([leaf])[2] for p, (tw, leaf) in reads.items()} | {"o0": oreg.statement([my])[2]}
    iset = SimpleNamespace(name=f"served-{u}", subcircuit=SimpleNamespace(id=fn.id, inputs=[], outputs=[]))
    FC.write(text, iset, 0, 1, s / "inst-1.bin", s / "pub-1.bin", indices=[u], tables=tables,
             refs={p: [0] for p in reads}, share_rows=True, salts=salts, registered=registered,
             outs=_split_outs(meta, orow))
    (s / "unit.tmpl").write_text(FC.template_of(text))
    (s / "partition.json").write_text(json.dumps(part))
    (s / "program.json").write_bytes(c)
    (s / "draw.json").write_text(json.dumps({"units": [k]}))

    def port(r) -> dict:
        return {"value": r.value, "root": r.root.hex(), "leaves": r.leaves, "words": r.words, "word_bits": 16,
                "domain": r.domain.domain_id.hex(), RV.SCHEMA_KEY: r.schema}
    from verity.protocols.proof.spot.verifier.one_stage import registered as REG
    file = {"format": REG.READS_V2,
            "ports": {p: port(regs[tw]) for p, (tw, _l) in reads.items()} | {"o0": port(oreg)}}
    (s / "registered.json").write_text(json.dumps(file))
    wrong = dict(file, ports=dict(file["ports"], o0=dict(file["ports"]["o0"], root="00" * 64)))
    (s / "registered-wrong-out.json").write_text(json.dumps(wrong))
    rec |= {"trees": {tw: {"leaves": r.leaves, "root": r.root.hex()[:16]} for tw, r in regs.items()},
            "out_tree": {"leaves": oreg.leaves, "leaf": my}, "out_words": on}
    (s / "stage.json").write_text(json.dumps(rec, default=str))
    return rec


def step(a) -> int:
    from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
    d = Path(a.dir)
    c = (d / "C.json").read_bytes()
    t = time.perf_counter()
    rec: dict = {"step": a.name, "template": a.template}
    if a.name == "stage":
        rec |= stage(d, a.template)
    elif a.name == "verify":
        from verity.protocols.proof.spot.verifier.one_stage import proofs as PF
        import os
        s = d / f"stage-{a.unit}"
        os.environ[PF.TMPL_SWITCH] = "1"
        PF.lower_executable = lambda: Path(a.lower)
        PF.executable = lambda: Path(a.lean)
        found, refusal = PF.verify(PF.Proof(s / "circuit.txt", s / "pub-1.bin", tuple(Path(x) for x in a.session)),
                                   program=(s / "program.json").read_bytes(),
                                   partition=json.loads((s / "partition.json").read_text()),
                                   draw=json.loads((s / "draw.json").read_text()),
                                   registered=json.loads((s / a.registered).read_text()))
        rec |= {"registered": a.registered, "refusal": refusal,
                "findings": [dataclasses.asdict(f) for f in found or []]}
    elif a.name == "one":
        from verity.circuits import partition_object as PO
        try:
            PO.TemplatePartition(_program(c), PO.template_instance_query(a.template))
            rec["refused"] = None
        except PO.QueryInapplicable as e:
            rec["refused"] = str(e)[:600]
    else:
        part, k, u, pop = _partition(d, a.template)
        rec |= {"block": k, "unit": u, "population": pop, "partition_s": round(time.perf_counter() - t, 1)}
        t = time.perf_counter()
        if a.name == "block":
            rec["units"] = OC.block(c, part, k)
            rec["ports"] = _ports(c, json.loads((d / "query.json").read_text()), u)
        elif a.name in ("restrict", "lower"):
            c_s = OC.restrict(c, part, [(u, u + 1)])
            rec |= {"restrict_s": round(time.perf_counter() - t, 1), "c_s_bytes": len(c_s)}
            (d / f"c_s-{u}.json").write_bytes(c_s)
            if a.name == "lower":
                from verity.protocols.proof.spot.verifier.one_stage import proofs as PF
                t = time.perf_counter()
                try:
                    exe = Path(a.lower) if a.lower else PF.lower_executable()
                    rec["build_s"] = round(time.perf_counter() - t, 1)
                    t = time.perf_counter()
                    text = OC.lower(c_s, part, lambda: exe, formed=OC.formed_in(c, part, [(u, u + 1)]))
                    (d / f"circuit-{u}.txt").write_bytes(text)
                    from verity.circuits import partition_object as PO
                    (d / f"partition-{u}.json").write_bytes(PO.canonical(part))
                    from verity_flock import circuit as FC
                    meta = FC.meta_of(text.decode())
                    rec |= {"circuit_bytes": len(text), "head": text.split(b"\n", 1)[0].decode()[:300],
                            "ports": [q[:2] for q in meta.get("ports", [])][:40], "refused": None}
                except (OC.NotDerived, OC.Refused) as e:
                    rec["refused"] = f"{type(e).__name__}: {e}"
        elif a.name == "door":
            from verity.protocols.proof.spot.verifier.one_stage import proofs as PF
            f = d / f"door-{u}"
            (f / "session").mkdir(parents=True, exist_ok=True)
            (f / "session" / "session.json").write_text("{}")
            (f / "pub.bin").write_text(json.dumps({"units": {"indices": [u]}}) + "\n")
            (f / "circuit.txt").write_text("")
            found, refusal = PF.verify(PF.Proof(f / "circuit.txt", f / "pub.bin", (f / "session",)), program=c,
                                       partition=part, draw={"units": [k]}, registered={})
            rec |= {"findings": found, "refusal": refusal}
    rec |= {"wall_s": round(time.perf_counter() - t, 1), "peak_gib": _peak_gib()}
    print("STEP " + json.dumps(rec, default=str), flush=True)
    return 0


def window(a) -> int:
    """The adapter's staged window (`test_window_rows`' TINY class and requests, random weights, every node computed, its
    scan state the execution phase's own, `window_rows.state_value` under the run root) recorded, closed and written as
    serving writes a window (`Window.supply`, `Rows.record`, `close`, `write` under the docket), for ``records_audit.py
    audit``."""
    tree, out = Path(a.tree), Path(a.out)
    sys.path.insert(0, str(tree / "tests/adapters/vllm"))
    from tests.program.predict import test_window_rows as TW
    from verity.circuits.codec import program_digest
    from verity.commitments import docket_digest as DD
    from verity.commitments.periods import interface as IF
    from verity.protocols.proof.spot.prover import docket as DK
    from verity_vllm.program.predict import scan_state as SS
    from verity_vllm.program.predict import window_rows as WR
    t = time.perf_counter()
    reg = TW._registered()
    low = TW.WC.lower(reg, TW.CLASS, None, TW.GRAIN, requests=TW.REQUESTS)
    fields = TW._weights(low.fn, a.seed)
    bits = [[w >> k & 1 for w in row for k in range(16)] for row in TW.STATE]
    values = {**TW.WW.weight_values(reg, fields, None, TW.GRAIN, keep_rows=True),
              SS.SCAN_STATE: WR.state_value(program_digest(low.circuit.program()[0]), TW.CLASS, bits, window=TW.RUN)}
    rows = WR.Rows(low, TW.unserved_slots(low, TW.STATE, lambda label, lo, hi: fields[label][lo:hi]), values)
    win = IF.Window(low.circuit, TW.RUN, TW.DESIGN, 64, registered=values, forms=WR.FORMS)
    win.supply()
    rows.record(win)
    win.write(win.close(), out, {"served_door": "window"}, registrar=DK)
    pins: dict[str, dict] = {}
    for p, v in sorted(values.items()):
        if v.window is None:
            pins.setdefault(v.program, {})[p] = v.entry()
    (out / "pins.json").write_text(json.dumps([{"program": g, "values": e} for g, e in sorted(pins.items())]))
    print("STEP " + json.dumps({"step": "window", "wall_s": round(time.perf_counter() - t, 1), "peak_gib": _peak_gib(),
                                "run": TW.RUN, "units": dict(rows.n),
                                "circuit_sha512": hashlib.sha512((out / "circuit.json").read_bytes()).hexdigest(),
                                "registration_sha512": hashlib.sha512((out / "registration.json").read_bytes()).hexdigest(),
                                "docket_digest": DD.digest(json.loads((out / "registration.json").read_text()))},
                               default=str), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tree", required=True)
    b.add_argument("--out", required=True)
    w = sub.add_parser("window")
    w.add_argument("--tree", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--seed", type=int, default=3)
    s = sub.add_parser("step")
    s.add_argument("name", choices=("block", "restrict", "lower", "door", "one", "stage", "verify"))
    s.add_argument("--dir", required=True)
    s.add_argument("--template", required=True)
    s.add_argument("--unit", type=int, help="verify: the staged block's unit")
    s.add_argument("--lower", help="lower, verify: flock-lower-check (else built)")
    s.add_argument("--lean", help="verify: verity-verify-proof")
    s.add_argument("--registered", default="registered.json", help="verify: the stage's registered file")
    s.add_argument("--session", nargs="*", default=[], help="verify: the session directories")
    a = ap.parse_args()
    return build(a) if a.cmd == "build" else window(a) if a.cmd == "window" else step(a)


if __name__ == "__main__":
    sys.exit(main())
