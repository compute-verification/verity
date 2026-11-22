"""Linking (`HmOut`) on a small wired program: two layers of the universal unit, layer 1 reading layer 0's outputs, one
`Q_template_instance` class staged with `class_statement.Wiring` and the tree of the template's output field (leaf u unit
u's output row): layer 0's statement (``writers``) opens its output rows at their leaves, and layer 1's statements read
them, each link row the writer's leaf by path (a tree port). The honest staging's link rows have their writers' output rows'
`b || c`; a prover that gives layer 1 a made-up input, and one that commits the right value under a salt of its own, stage the
same circuit with link rows that differ, as do a reader whose read is moved to another unit's leaf (``other_leaf``) and one
whose link row is a row of other words under its writer's salt (``other_words``). When ``FLOCK_CIRCUIT`` names a build
of ``flock-circuit``, the live verifier refuses the cheats with its own links file. The Lean verifier admits no class
statement with links; the same cheats, on the templated statement C wires (`_tmpl`), are refused by
`verity-verify-proof statement` and by the sampled proofs' setup, each by its own name, the honest one accepted. Linking
binds a public topology; it does not hide the topology."""
from __future__ import annotations

import functools
import json
import os
import random
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from verity.circuits import partition_object as PO
from verity.circuits.codec import canonical_json, descriptor_id, encode_program, program_digest
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import array_of, coll_concat
from verity.circuits.types import Array
from verity.circuits.private_circuits import universal as U
from verity.protocols.proof.spot.verifier.one_stage import partition as P
from verity.protocols.proof.spot.verifier.one_stage import registered as RG
from verity.commitments import hm96
from verity.commitments import registered_value as RV

import verity_difftest as D
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import partition_units as PU

G, N_IN, N_OUT, N = 4, 4, 2, 2
NC = N_IN + N_OUT
#: layer 0 reads c[0:4]; layer 1 reads c[4:6], then layer 0's outputs
WIRED = ((("c", 0, N_IN),), (("c", N_IN, NC), ("u", 0, 0, N_OUT)))
FLAT = ((("c", 0, N_IN),),)
#: the templated statements' unit: every parameter whole u16 words, src past one SHA-512 block, as the templated layout
#: needs; layer 1's x is layer 0's whole output, as C's chain wires a layer
TG, TN = 120, 16
TWIRED = ((("c", 0, TN),), (("u", 0, 0, TN),))
TFLAT = ((("c", 0, TN),),)
MODE = "iv/hybrid/4"
V2 = RG.SCHEMA_V2
SPEC = "q-template-instance"
LEAN = Path(__file__).resolve().parents[6] / "verity" / "protocols" / "proof" / "flock" / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "verity-verify-proof"
SPOT = LEAN.parents[1] / "spot" / "verifier" / "lean"

#: the stagings every test reads take about 20 s
pytestmark = pytest.mark.slow


def _program(topo, g=G, n_in=N_IN, n_out=N_OUT, nc=NC):
    fn = bind(U.UniversalUnit, G=g, N_IN=n_in, N_OUT=n_out)
    params, ret = fn.signature()
    sig = [("c", Array(N, Array(nc, U.BIT)))] + [(f"{nm}{i}", t) for i in range(len(topo)) for nm, t in params[:-1]]

    def body(B, S, c, *ps):
        us = []
        for i, parts in enumerate(topo):
            xs = array_of([coll_concat(Array(n_in, U.BIT), [c[t][p[1]:p[2]] if p[0] == "c" else us[p[1]][t][p[2]:p[3]] for p in parts])
                           for t in range(N)])
            us.append(B.batch(fn, *ps[3 * i:3 * i + 3], xs, axes=(None, None, None, 0), n=N))
        return us[-1]

    root = CompositeDefinition(f"WiredLinksTest{len(topo)}", 1, (), lambda S: (tuple(sig), Array(N, ret)), body, register=False)
    return fn, Program(root)


def _lanes(units, cl, tp, inputs: list[int], made_up=None):
    """Every unit's lanes run in population order on the root's input gates ``inputs``, each reading earlier units' outputs
    as the program wires them (``made_up``: {unit: f(words) -> words}, the parameter leaves a cheating prover gives it); the row
    identities as the bench's (`universal.lanes`): x one row per instance and layer, the circuit description one row per layer."""
    act, uc = units.activation(cl.definition), cl.representative
    leaf_param = []
    for i, (_n, t) in enumerate(act.fn.signature()[0]):
        leaf_param += [i] * t.leaves
    x_param = leaf_param[-1]
    groups = CS.sources(act, uc, CS.owners_of(units, cl.definition.id))
    value = dict(enumerate(inputs))
    ins, want, rows = [[] for _ in uc.inputs], [[] for _ in uc.outputs], [[] for _ in groups]
    for u in range(tp.population):
        words = [value[g] for g in tp.inputs(u)]
        if made_up and u in made_up:
            words = made_up[u](words)
        run = PU.evaluate(act, words)
        for port, k in zip(ins, uc.inputs):
            port.append(run[k])
        for port, j in zip(want, uc.outputs):
            port.append(run[j])
        for g, j in zip(tp.outputs(u), uc.outputs):
            value[g] = run[j]
        layer, t = divmod(u, N)
        for ids, g in zip(rows, groups):
            ids.append(("x", t, layer) if x_param in {leaf_param[-1 - uc.inputs[i]] for i in g} else ("program", layer))
    return ins, want, rows


def _one_draw(d: Path, n: int) -> tuple[dict, dict]:
    """One fresh salt per instance per port (``own``'s rows), and the same draw at each shared table row's first reader (the
    honest and ``wrong_salt`` rows), the layout read from ``d``'s public file: a wired staging of the same circuit and source
    identities. ``wrong_salt``'s rows then differ from the honest ones only where it cheats."""
    names = [p[0] for p in _meta(d)["ports"]]
    head, body = (d / f"pub-{n}.bin").read_bytes().split(b"\n", 1)
    counts = [json.loads(head)["shared_rows"][x] for x in names]
    o = 128 * sum(counts)
    refs = [int.from_bytes(body[o + 4 * k:o + 4 * k + 4], "little") for k in range(n * len(names))]
    own = {x: [os.urandom(192) for _ in range(n)] for x in names}
    firsts = {x: {} for x in names}
    for k in range(n):
        for j, x in enumerate(names):
            firsts[x].setdefault(refs[k * len(names) + j], k)
    shared = {x: [own[x][firsts[x][r]] for r in range(c)] for x, c in zip(names, counts)}
    return own, shared


def _cut(lanes: tuple, lo: int, hi: int) -> tuple:
    return tuple([p[lo:hi] for p in part] for part in lanes)


def _stage(topo, d: Path, wired: bool):
    """The program, its partition and the stagings: honest (with `Wiring`), and with ``wired`` the writers (layer 0, its
    output rows the tree's), the two cheats among layer 1's statements (x reads layer 0's outputs flipped; the honest rows
    under link salts of their own), and the honest lanes with every instance's own rows (``own``, ``share_rows=False``), each
    with the verifier's `--registered` (``<name>.reads.json``); else the honest lanes staged without `Wiring` (``plain``)."""
    rng = random.Random(f"test-wired-links/{len(topo)}")
    fn, program = _program(topo)
    nets = [U.pad(U.PrivateCircuit(N_IN, tuple((rng.randrange(4), rng.randrange(N_IN + g), rng.randrange(N_IN + g))
                                               for g in range(G)),
                                   tuple(rng.randrange(N_IN + G) for _ in range(N_OUT))), G, N_IN) for _ in topo]
    inputs = [rng.randrange(2) for _ in range(N * NC)] + [b for net in nets for p in U.arguments(net, [0] * N_IN)[:3] for b in p]
    query = PO.template_instance_query(fn)
    units = PU.PartitionUnits(program, query=query)
    [(shape, cl)] = units.classes().items()
    tp = PO.TemplatePartition(program, query)
    assert min(g for u in range(tp.population) for g in tp.outputs(u)) >= len(inputs)
    (d / "partition.json").write_text(json.dumps(PO.build(program, query), sort_keys=True))
    (d / "program.desc").write_bytes(canonical_json(encode_program(program)))
    wiring = CS.Wiring(program, query, list(range(tp.population)))
    if not wired:
        st = {"honest": CS.stage(units, SPEC, shape, cl, _lanes(units, cl, tp, inputs), d / "honest", packed=True, wiring=wiring)}
        st["plain"] = CS.stage(units, SPEC, shape, cl, _lanes(units, cl, tp, inputs), d / "plain", packed=True)
    else:
        honest = _lanes(units, cl, tp, inputs)
        template = descriptor_id(fn)
        tree = RV.Registered.commit(program_digest(program), f"{template}/out",
                                   [[p[u] for p in honest[1]] for u in CS.tree_leaves(tp, template)])
        readers = replace(wiring, indices=list(range(N, tp.population)), tree=tree)
        st = {"writers": CS.stage(units, SPEC, shape, cl, _cut(honest, 0, N), d / "writers", packed=True,
                                  wiring=replace(wiring, indices=list(range(N)), tree=tree))}
        flip = {N + t: (lambda w: w[:-N_OUT] + [1 - b for b in w[-N_OUT:]]) for t in range(N)}
        st["made_up"] = CS.stage(units, SPEC, shape, cl, _cut(_lanes(units, cl, tp, inputs, flip), N, tp.population),
                                 d / "made_up", packed=True, wiring=replace(readers, strict=False))
        own, shared = _one_draw(d / "made_up", N)
        st["honest"] = CS.stage(units, SPEC, shape, cl, _cut(honest, N, tp.population), d / "honest", packed=True,
                                wiring=readers, salts=shared)
        st["wrong_salt"] = CS.stage(units, SPEC, shape, cl, _cut(honest, N, tp.population), d / "wrong_salt", packed=True,
                                    wiring=replace(readers, writer_salts=False), salts=shared)
        st["own"] = CS.stage(units, SPEC, shape, cl, _cut(honest, N, tp.population), d / "own", packed=True, wiring=readers,
                             share_rows=False, salts=own)
        text = (d / "honest" / "circuit.txt").read_text()
        (d / "links.json").write_text(json.dumps(CS.links_file(program, query, text, list(range(tp.population)))))
    return fn, program, query, tp, st


def _edit(src: Path, dst: Path, pub: str, port: str, refs: dict) -> None:
    """``src``'s statement with port ``port``'s refs moved (instance i reads table row ``refs[i]``); its rows, and every row's
    stated position and path, unchanged."""
    shutil.copytree(src, dst)
    names = [p[0] for p in _meta(src)["ports"]]
    head, body = (src / pub).read_bytes().split(b"\n", 1)
    counts = [json.loads(head)["shared_rows"][x] for x in names]
    j, o = names.index(port), 128 * sum(counts)
    body = bytearray(body)
    for i, r in refs.items():
        at = o + 4 * (i * len(names) + j)
        body[at:at + 4] = r.to_bytes(4, "little")
    (dst / pub).write_bytes(head + b"\n" + bytes(body))


def _tmpl(topo, d: Path) -> None:
    """``topo``'s layers of the `TG`-gate unit as templated statements (`UnitStager` bound to C and its partition, `MODE`),
    each port a registered read the verifier derives from C: op, src and out each layer's run input node's one row, x
    layer 0's rows of c and layer 1's its writers' output rows, read by path in the output field's tree (`<template>/o0`,
    leaf u unit u's output), which o0 opens at the unit's own leaf. Writes ``writers`` (layer 0) and its verifier's
    ``writers.reads.json`` under ``d``; with two layers, layer 1's ``honest`` and ``readers.reads.json``, and the cheats:
    x made up (``made_up``), the honest x under salts of the prover's own (``wrong_salt``), other words under the writer's
    salt (``other_words``), a read moved to the other writer's leaf (``other_leaf``), x committed under a root of its own
    (``own_root``, with ``own_root.reads.json`` naming that root), and the block circuit's x row named y (``no_row``).
    A cheat's file is written with the honest outputs, so its only fault is x."""
    rng = random.Random(f"test-wired-links/tmpl/{len(topo)}")
    fn, program = _program(topo, TG, TN, TN, TN)
    query = PO.template_instance_query(fn)
    part = PO.build(program, query)
    (d / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (d / "program.desc").write_bytes(canonical_json(encode_program(program)))
    pd, template = program_digest(program), descriptor_id(fn)
    params = fn.signature()[0]
    nodes = {nd.name: k for k, nd in enumerate(program.root.body.nodes) if getattr(nd, "name", None)}
    bound = {"partition": P.digest(part), "program": pd}

    @functools.cache
    def stager(x: str):
        return CS.UnitStager(fn, f"unit/{template}", [(x if nm == "x" else nm, V2, t.leaves, [nm]) for nm, t in params],
                             [(V2, (0,))], bind=bound, layout=MODE)

    def word(bits):
        return tuple(sum(int(b) << j for j, b in enumerate(bits[k:k + 16])) for k in range(0, len(bits), 16))

    def commit(value: str, rows):
        return RV.Registered.commit(pd, value, rows, schema=V2)

    nets = [U.pad(U.PrivateCircuit(TN, tuple((rng.randrange(4), rng.randrange(TN + g), rng.randrange(TN + g)) for g in range(TG)),
                                   tuple(rng.randrange(TN + TG) for _ in range(TN))), TG, TN) for _ in topo]
    c = [[rng.randrange(2) for _ in range(TN)] for _ in range(N)]
    xs = [c]
    for net in nets:
        xs.append([U.interpret(net, x) for x in xs[-1]])
    tree = commit(f"{template}/o0", [word(o) for layer in xs[1:] for o in layer])
    assert CS.tree_leaves(PO.TemplatePartition(program, query), template) == list(range(N * len(topo)))
    regs = [{p: commit(f"input/n{nodes[f'{p}{i}']}", [word(U.arguments(net, [0] * TN)[j])]) for j, p in enumerate(("op", "src", "out"))}
            for i, net in enumerate(nets)]
    regs[0]["x"] = commit(f"input/n{nodes['c']}", [word(x) for x in c])

    def write(name: str, layer: int, x=None, x_rows=None, x_salts=None, port: str = "x") -> None:
        units = list(range(layer * N, layer * N + N))
        ports = {**{p: (r, [0], [0] * N) for p, r in regs[layer].items() if p != "x"},
                 "x": (x or regs[layer].get("x") or tree, list(range(N)), list(range(N))), "o0": (tree, units, list(range(N)))}
        tables, refs, salts, header, own = {}, {}, {}, {}, {}
        for p, (r, at, ref) in ports.items():
            rows, salts[p], header[p] = r.statement(at)
            tables[p], refs[p], own[p] = np.asarray(rows, np.uint16), ref, r.entry() | RV.schema_entry(V2)
        if x_rows is not None:
            tables["x"] = np.asarray(x_rows, np.uint16)
        if x_salts is not None:
            salts["x"] = x_salts
        outs = {"o0": tables.pop("o0")}
        del refs["o0"]
        rows = {p: tables[p][refs[p]] for p in tables}
        st = stager(port)
        ren = lambda m: {(port if k == "x" else k): v for k, v in m.items()}  # noqa: E731
        if x_rows is None:
            st.reference_check(ren(rows), outs)
        (d / name).mkdir()
        (d / name / "circuit.txt").write_text(st.text)
        FC.write(st.text, st.iset, 0, N, d / name / "inst.bin", d / name / "pub.bin", rows=ren(rows), outs=outs, salts=ren(salts),
                 digests=(st.pin, st.class_sha512), indices=units, share_rows=True, tables=ren(tables), refs=ren(refs),
                 registered=ren(header), meta=st.meta)
        if name in ("writers", "honest", "own_root"):
            (d / f"{'readers' if name == 'honest' else name}.reads.json").write_text(
                json.dumps({"format": RG.READS_V2, "ports": ren(own)}))

    write("writers", 0)
    if len(topo) == 1:
        return
    honest, salts, _e = tree.statement(range(N))
    flip = lambda r: tuple(v ^ 0xFFFF for v in r)  # noqa: E731
    write("honest", 1)
    write("made_up", 1, x_rows=[flip(r) for r in honest])
    write("wrong_salt", 1, x_salts=[hm96.SHA512.fresh_salt() for _ in range(N)])
    write("other_words", 1, x_rows=[flip(honest[0]), *honest[1:]])
    _edit(d / "honest", d / "other_leaf", "pub.bin", "x", refs={0: 1, 1: 0})
    write("own_root", 1, x=commit(f"{template}/x", honest))
    write("no_row", 1, port="y")


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    CS.cache_slot_circuits()
    d = tmp_path_factory.mktemp("wired")
    flat = tmp_path_factory.mktemp("flat")
    return d, _stage(WIRED, d, True), flat, _stage(FLAT, flat, False)


@pytest.fixture(scope="module")
def templated(tmp_path_factory):
    d, flat = tmp_path_factory.mktemp("tmpl"), tmp_path_factory.mktemp("tmpl-flat")
    _tmpl(TWIRED, d)
    _tmpl(TFLAT, flat)
    return d, flat


def _meta(d: Path) -> dict:
    return json.loads((d / "circuit.txt").read_text().split("\n", 2)[1][5:])


def _bc(d: Path, n: int):
    """The public file's `b || c` of instance ``i``'s row ``name``: a shared-row file's table row, by the instance's ref, else
    the instance's own (instance-major, ports in order)."""
    meta = _meta(d)
    head, body = (d / f"pub-{n}.bin").read_bytes().split(b"\n", 1)
    head = json.loads(head)
    names = [p[0] for p in meta["ports"]]
    if "shared_rows" not in head:
        return lambda i, name: body[128 * (i * len(names) + names.index(name)):][:128]
    counts = [head["shared_rows"][x] for x in names]
    starts = [sum(counts[:k]) for k in range(len(counts))]
    o = 128 * sum(counts)
    refs = [int.from_bytes(body[o + 4 * k:o + 4 * k + 4], "little") for k in range(n * len(names))]
    return lambda i, name: body[128 * (starts[names.index(name)] + refs[i * len(names) + names.index(name)]):][:128]


def test_a_wired_unit_reads_its_writer_through_one_link_row_and_the_triples_are_the_cut_wires(staged):
    d, (fn, _p, _q, tp, st), _flat, _ = staged
    meta = _meta(d / "honest")
    assert meta["definition"] == descriptor_id(fn) and st["honest"]["definition"] == descriptor_id(fn)
    [link] = meta["links"]
    x0 = len(tp.inputs(0)) - N_IN
    assert link["leaves"] == [x0 + N_IN - N_OUT, x0 + N_IN]           # layer 1's x after its two circuit-input bits
    assert {s["circuit_sha512"] for k, s in st.items() if k != "writers"} == {st["honest"]["circuit_sha512"]}
    assert "links" not in _meta(d / "writers")
    links = json.loads((d / "links.json").read_text())
    assert links["format"] == CS.LINKS_FORMAT and links["ports"] == [link["port"]]
    assert links["links"] == [[N + t, link["port"], t] for t in range(N)]


def test_honest_link_rows_are_their_writers_output_rows_and_each_cheat_breaks_them(staged):
    """Reader instance t (unit N + t) reads writer t, instance t of the writers' statement."""
    d, _st, _flat, _ = staged
    port = _meta(d / "honest")["links"][0]["port"]
    bc = {k: _bc(d / k, N) for k in ("writers", "honest", "made_up", "wrong_salt", "own")}
    assert all(bc["honest"](t, port) == bc["writers"](t, "out") for t in range(N))
    # every instance's own rows: each reader's link row is its own, and still its writer's output row under the writer's salt
    assert all(bc["own"](t, port) == bc["writers"](t, "out") for t in range(N))
    assert all(bc["made_up"](t, port) != bc["writers"](t, "out") for t in range(N))
    assert all(bc["wrong_salt"](t, port) != bc["writers"](t, "out") for t in range(N))


def test_link_triples_refuse_a_cut_wire_no_link_carries_and_a_partial_read(staged):
    _d, (_fn, _p, _q, tp, _st), _flat, _ = staged
    indices = list(range(tp.population))
    reads = CS.cut_reads(tp, indices)
    x0 = len(tp.inputs(0)) - N_IN
    n_out = lambda v: len(tp.outputs(v))  # noqa: E731
    with pytest.raises(ValueError, match="no link port carries it"):
        CS.link_triples({}, reads, indices, n_out)
    with pytest.raises(ValueError, match="not one unit's whole output row"):
        CS.link_triples({"links": [{"port": "p0", "leaves": [x0 + N_IN - N_OUT, x0 + N_IN - 1]},
                                   {"port": "p1", "leaves": [x0 + N_IN - 1, x0 + N_IN]}]}, reads, indices, n_out)
    with pytest.raises(ValueError, match="not one of the statement's units"):
        CS.link_triples({"links": [{"port": "p0", "leaves": [x0 + N_IN - N_OUT, x0 + N_IN]}]}, reads[N:], indices[N:], n_out)


def _lean(*args: str) -> tuple[dict | None, str]:
    """The statement's setup: its JSON when accepted; refused only with status 1 (any other status is an error)."""
    p = subprocess.run([str(BIN), "statement", *args], capture_output=True, text=True)
    assert p.returncode in (0, 1), (p.returncode, p.stdout + p.stderr)
    return (json.loads(p.stdout.splitlines()[0]) if p.returncode == 0 else None), p.stdout + p.stderr


def _named(out: str) -> bool:
    """No panic and no traceback: a refusal by its own name."""
    return not any(s in out for s in ("PANIC", "index out of bounds", "Traceback"))


NOT_OPEN = "does not open"
NO_ROOTS = "the verifier holds no registered roots"
NO_PROGRAM = "a port's reads are derived from C, the verifier's program (--program)"
NO_ROW = "io: C wires UniversalUnit_v1{G=120,N_IN=16,N_OUT=16}'s parameter x from another block, and the block circuit has no row of it"
CROSSING = "row x of in(S) is committed as UniversalUnit_v1{G=120,N_IN=16,N_OUT=16}/x, and C wires it from"


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_row_reads_one_output_field_of_its_writer_whole_at_the_positions_c_gives():
    from verity.circuits.types import Tuple, Value

    pair, wide = Tuple(Value(32), Value(32)), Tuple(Array(2, Value(16)), Value(16))
    assert [CS.field_span(wide, o) for o in range(4)] == [(0, 2), None, (1, 1), None]
    assert CS.field_span(Value(16), 0) == (0, 1)
    ret = {0: pair, 1: wide}.__getitem__
    assert CS.field_row([(0, 0)], ret) == (0, 0) and CS.field_row([(0, 1)], ret) == (0, 1)
    assert CS.field_row([(1, 0), (1, 1)], ret) == (1, 0) and CS.field_row([(1, 2)], ret) == (1, 1)
    assert CS.field_row([None, None], ret) is None
    for bad in ([(0, 0), (0, 1)], [(0, 1), (0, 0)], [(1, 0)], [(1, 1)], [(1, 1), (1, 2)], [(1, 1), (1, 0)],
                [(0, 0), None], [None, (0, 1)], [(1, 0), (0, 1)]):
        with pytest.raises(ValueError, match="not one output field"):
            CS.field_row(bad, ret)


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_lean_accepts_honest_links_and_refuses_each_cheat_and_links_without_the_program(templated):
    """The templated statements (`_tmpl`) set up by `verity-verify-proof statement`: a reader's x is no port of its own
    but a row of in(S) C wires (`OneCircuit.io`), opened by path in its writer's output field's tree, so a block circuit
    with no row of x is refused by io's name (the META links a class statement declared, and its check that a tree port
    reads every link, are gone from the templated form). The class statement's per-instance rows (``own``) have no
    templated form: every port is a registered read, x one row per instance already."""
    d, flat = templated
    D.build(LEAN, "verity-verify-proof")

    def at(where: Path, name: str, reads: str | None, program: bool = True):
        return _lean("--statement", FC.TMPL_STATEMENT, "--circuit", str(where / name / "circuit.txt"),
                     "--public", str(where / name / "pub.bin"), "--partition", str(where / "partition.json"),
                     *(["--program", str(where / "program.desc")] if program else []),
                     *(["--registered", str(where / f"{reads}.reads.json")] if reads else []))

    for where, name, reads, note in ((d, "writers", "writers", "no wire between the statement's units"),
                                     (d, "honest", "readers", f"Linking checked: {N} wire(s)"),
                                     (flat, "writers", "writers", "no wire between the statement's units")):
        ok, out = at(where, name, reads)
        print(where.name, name, ok and ok.get("partition_check"))
        assert ok is not None, out
        assert ok["partition_check"]["note"].startswith("units derived") and note in ok["partition_check"]["note"], out
    for cheat, reads, why in (("made_up", "readers", f"port x: table row 0 {NOT_OPEN}"),
                              ("wrong_salt", "readers", f"port x: table row 0 {NOT_OPEN}"),
                              ("other_words", "readers", f"port x: table row 0 {NOT_OPEN}"),
                              ("other_leaf", "readers", "port x: unit 2 reads position 1, and the verifier's program reads 0"),
                              ("own_root", "own_root", CROSSING),
                              ("no_row", "readers", NO_ROW)):
        bad, out = at(d, cheat, reads)
        print(cheat, out.strip().splitlines()[-1])
        assert bad is None and why in out and _named(out), (cheat, out)
    for name, reads, program, why in (("honest", None, True, NO_ROOTS), ("made_up", None, True, NO_ROOTS),
                                      ("honest", "readers", False, NO_PROGRAM)):
        bad, out = at(d, name, reads, program)
        print(name, reads, program, out.strip().splitlines()[-1])
        assert bad is None and why in out and _named(out), (name, out)


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_sampled_proofs_own_setup_refuses_each_cheat(templated):
    """The same statements set up by the sampled proofs' own check (`OneStageZk.sessionAt`, through `spot-difftest`'s
    `session_setup`), each with its verifier's registered reads."""
    d, _flat = templated
    D.build(SPOT, "spot-difftest")
    program = (d / "program.desc").read_bytes().hex()
    partition = json.loads((d / "partition.json").read_text())
    cases = (("writers", "writers", None), ("honest", "readers", None), ("made_up", "readers", f"port x: table row 0 {NOT_OPEN}"),
             ("wrong_salt", "readers", f"port x: table row 0 {NOT_OPEN}"), ("other_words", "readers", f"port x: table row 0 {NOT_OPEN}"),
             ("other_leaf", "readers", "port x: unit 2 reads position 1"), ("own_root", "own_root", CROSSING),
             ("no_row", "readers", NO_ROW))
    lines = [json.dumps({"fn": "session_setup", "input": {
        "circuit": (d / name / "circuit.txt").read_bytes().hex(), "public": (d / name / "pub.bin").read_bytes().hex(),
        "tables": [], "program": program, "partition": partition,
        "registered": json.loads((d / f"{reads}.reads.json").read_text()), "draw": None}}) for name, reads, _ in cases]
    p = subprocess.run([str(SPOT / ".lake" / "build" / "bin" / "spot-difftest")], input="\n".join(lines) + "\n",
                       capture_output=True, text=True)
    assert p.returncode == 0 and _named(p.stderr), p.stderr[-2000:]
    got = [json.loads(x) for x in p.stdout.splitlines()]
    assert len(got) == len(cases), p.stdout + p.stderr
    for (name, _reads, why), g in zip(cases, got):
        verdict = g.get("output", {"refused": g}).get("refused", g)
        print("session_setup", name, verdict)
        assert (verdict is None) if why is None else (isinstance(verdict, str) and why in verdict), (name, g)


@pytest.mark.skipif(not (os.environ.get("FLOCK_CIRCUIT") and Path(os.environ["FLOCK_CIRCUIT"]).exists()),
                    reason="FLOCK_CIRCUIT names no flock-circuit build")
def test_the_live_verifier_checks_links_against_its_own_file(staged):
    d, _st, flat, (_f2, _p2, _q2, tp2, _st2) = staged
    n = N

    def live(sdir: Path, k: int, *extra: str) -> str:
        p = subprocess.run([os.environ["FLOCK_CIRCUIT"], "statement", "--circuit", str(sdir / "circuit.txt"),
                            "--instances", str(sdir / f"pub-{k}.bin"), *extra], capture_output=True, text=True)
        return p.stdout

    links = ["--links", str(d / "links.json")]
    assert "STATEMENT\t" in live(d / "honest", n, *links) and "STATEMENT\t" in live(d / "own", n, *links)
    for cheat in ("made_up", "wrong_salt"):
        assert "the reader's row is not the writer's committed output row" in live(d / cheat, n, *links)
    assert "pass the verifier's own links file" in live(d / "honest", n)
    assert "STATEMENT\t" in live(flat / "plain", tp2.population)
    assert "the circuit declares no links" in live(flat / "plain", tp2.population, *links)
