"""Linking (`HmOut`) on a small wired program: two layers of the universal unit, layer 1 reading layer 0's outputs, one
`Q_template_instance` class staged with `class_statement.Wiring` and the tree of the template's output field (leaf u unit
u's output row): layer 0's statement (``writers``) opens its output rows at their leaves, and layer 1's statements read
them, each link row the writer's leaf by path (a tree port). The honest staging's link rows have their writers' output rows'
`b || c`; a prover that gives layer 1 a made-up input, and one that commits the right value under a salt of its own, stage the
same circuit with link rows that differ, as do a reader whose read is moved to another unit's leaf (``other_leaf``) and one
whose link row is a row of other words under its writer's salt (``other_words``); the honest reader's link may start at
a word of its writer's row where no return field starts (``wrong_offset``), its tree port a width other than its writer
row's or none, and it may be staged as an inner statement with plain leaves (``plain_leaves``). The Lean verifier
(`verity-verify-proof statement --program --registered`, when a Lean toolchain is present) accepts the writers and the
honest readers and refuses every cheat, a linked statement without its program, and one without registered reads; a
one-layer statement, with no cut wire, is accepted with its units derived when staged with `Wiring`, and as before when
staged without. When ``FLOCK_CIRCUIT`` names a build of ``flock-circuit``, the live verifier gives the same verdicts
with its own links file. Linking binds a public topology; it does not hide the topology."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from verity.circuits import partition_object as PO
from verity.circuits.codec import canonical_json, descriptor_id, encode_program, program_digest
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import array_of, coll_concat
from verity.circuits.types import Array
from verity.circuits.private_circuits import universal as U
from verity.protocols.proofs.spot.verifier.one_stage import registered as RG
from verity.commitments import registered_value as RV

from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import partition_units as PU

G, N_IN, N_OUT, N = 4, 4, 2, 2
NC = N_IN + N_OUT
#: layer 0 reads c[0:4]; layer 1 reads c[4:6], then layer 0's outputs
WIRED = ((("c", 0, N_IN),), (("c", N_IN, NC), ("u", 0, 0, N_OUT)))
FLAT = ((("c", 0, N_IN),),)
SPEC = "q-template-instance"
LEAN = Path(__file__).resolve().parents[6] / "verity" / "protocols" / "proofs" / "flock" / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "verity-verify-proof"
SPOT = LEAN.parents[1] / "spot" / "verifier" / "lean"

#: the stagings every test reads take about 20 s
pytestmark = pytest.mark.slow


def _program(topo):
    fn = bind(U.UniversalUnit, G=G, N_IN=N_IN, N_OUT=N_OUT)
    params, ret = fn.signature()
    sig = [("c", Array(N, Array(NC, U.BIT)))] + [(f"{nm}{i}", t) for i in range(len(topo)) for nm, t in params[:-1]]

    def body(B, S, c, *ps):
        us = []
        for i, parts in enumerate(topo):
            xs = array_of([coll_concat(Array(N_IN, U.BIT), [c[t][p[1]:p[2]] if p[0] == "c" else us[p[1]][t][p[2]:p[3]] for p in parts])
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
        st["mixed"] = CS.stage(units, SPEC, shape, cl, honest, d / "mixed", packed=True, wiring=wiring)
        port = _meta(d / "honest")["links"][0]["port"]
        _edit(d / "honest", d / "other_leaf", N, port, refs={0: 1, 1: 0})
        _edit_meta(d / "honest", d / "wrong_offset", lambda m: m["links"][0].update({"at": 1}))
        plain, CS.LEAF_SCHEME = CS.LEAF_SCHEME, FC.LEAF_SCHEME_PLAIN
        try:
            CS.stage(units, SPEC, shape, cl, _cut(honest, N, tp.population), d / "plain_leaves", packed=True, wiring=readers,
                     salts=shared)
        finally:
            CS.LEAF_SCHEME = plain
        st["other_words"] = CS.stage(units, SPEC, shape, cl, _cut(_lanes(units, cl, tp, inputs, {N: flip[N]}), N, tp.population),
                                     d / "other_words", packed=True, wiring=replace(readers, strict=False), salts=shared)
        for name, k in (("writers", "writers"), ("readers", "honest"), ("own", "own")):
            (d / f"{name}.reads.json").write_text(json.dumps({"format": RG.READS, "ports": st[k]["registered_reads"]}))
        wrong = json.loads(json.dumps(st["honest"]["registered_reads"]))
        wrong[port]["tree"]["field"] = "aux"
        (d / "wrong_field.reads.json").write_text(json.dumps({"format": RG.READS, "ports": wrong}))
        wide = json.loads(json.dumps(st["honest"]["registered_reads"]))
        wide[port]["words"] += 64
        (d / "wrong_width.reads.json").write_text(json.dumps({"format": RG.READS, "ports": wide}))
        del wide[port]["words"]
        (d / "no_words.reads.json").write_text(json.dumps({"format": RG.READS, "ports": wide}))
        text = (d / "honest" / "circuit.txt").read_text()
        (d / "links.json").write_text(json.dumps(CS.links_file(program, query, text, list(range(tp.population)))))
    return fn, program, query, tp, st


def _edit(src: Path, dst: Path, n: int, port: str, refs: dict) -> None:
    """``src``'s statement with port ``port``'s refs moved (instance i reads table row ``refs[i]``); its rows, and every row's
    stated position and path, unchanged."""
    shutil.copytree(src, dst)
    names = [p[0] for p in _meta(src)["ports"]]
    head, body = (src / f"pub-{n}.bin").read_bytes().split(b"\n", 1)
    counts = [json.loads(head)["shared_rows"][x] for x in names]
    j, o = names.index(port), 128 * sum(counts)
    body = bytearray(body)
    for i, r in refs.items():
        at = o + 4 * (i * len(names) + j)
        body[at:at + 4] = r.to_bytes(4, "little")
    (dst / f"pub-{n}.bin").write_bytes(head + b"\n" + bytes(body))


def _edit_meta(src: Path, dst: Path, edit) -> None:
    """``src``'s statement with its circuit's META changed by ``edit`` (in place on the parsed object) and the circuit's
    digest restated wherever the statement names it, every other byte unchanged."""
    shutil.copytree(src, dst)
    old = (src / "circuit.txt").read_bytes()
    head, line, rest = old.decode().split("\n", 2)
    meta = json.loads(line[5:])
    edit(meta)
    compact = '": ' not in line
    new = "\n".join([head, "META " + json.dumps(meta, separators=(",", ":") if compact else None), rest]).encode()
    (dst / "circuit.txt").write_bytes(new)
    was, now = hashlib.sha256(old).hexdigest().encode(), hashlib.sha256(new).hexdigest().encode()
    for f in dst.iterdir():
        if f.is_file() and f.name != "circuit.txt" and was in (b := f.read_bytes()):
            f.write_bytes(b.replace(was, now))


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    CS.cache_slot_circuits()
    d = tmp_path_factory.mktemp("wired")
    flat = tmp_path_factory.mktemp("flat")
    return d, _stage(WIRED, d, True), flat, _stage(FLAT, flat, False)


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
    p = subprocess.run([str(BIN), "statement", *args], capture_output=True, text=True)
    return (json.loads(p.stdout.splitlines()[0]) if p.returncode == 0 else None), p.stdout + p.stderr


NO_TREE = "and no tree port (tree) reads it by path"
NOT_OPEN = "does not open"
NO_ROOTS = "the verifier holds no registered roots"
NO_READ = "the verifier's program reads no row of"


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_lean_accepts_honest_links_and_refuses_each_cheat_and_links_without_the_program(staged):
    d, _st, flat, (_f2, _p2, _q2, tp2, _st2) = staged
    subprocess.run(["lake", "build", "verity-verify-proof"], cwd=LEAN, check=True, capture_output=True)
    bound = ["--partition", str(d / "partition.json"), "--program", str(d / "program.desc")]

    def at(name: str, *extra: str, reads: str | None = "readers.reads.json", n: int = N):
        return _lean("--circuit", str(d / name / "circuit.txt"), "--public", str(d / name / f"pub-{n}.bin"), *extra,
                     *(["--registered", str(d / reads)] if reads else []))

    ok, out = at("writers", *bound, reads="writers.reads.json")
    print("writers", ok and ok["partition_check"]["note"])
    assert ok is not None and ok["partition_check"]["units"] == "derived", out
    for name, reads in (("honest", "readers.reads.json"), ("own", "own.reads.json")):
        ok, out = at(name, *bound, reads=reads)
        print(name, ok and ok["partition_check"]["note"])
        assert ok is not None, out
        note = ok["partition_check"]["note"]
        assert ok["partition_check"]["units"] == "derived" and f"Linking checked: {N} cut wire(s)" in note and "unit-class/" in note
    # each cheat's link rows open no leaf of the registered tree at the writer's position (checkPort), and the moved read's
    # position is not the one the verifier's program gives (treeReads)
    for cheat, why in (("made_up", NOT_OPEN), ("wrong_salt", NOT_OPEN), ("other_words", NOT_OPEN),
                       ("other_leaf", "and the verifier's program reads")):
        bad, out = at(cheat, *bound)
        print(cheat, out.strip().splitlines()[-1])
        assert bad is None and why in out, (cheat, out)
    # the honest reader with its tree port naming another output field of the writer's template: a link row reads its
    # writer's whole output row, so no other field's tree
    bad, out = at("honest", *bound, reads="wrong_field.reads.json")
    print("wrong_field", out.strip().splitlines()[-1])
    assert bad is None and "reads its writer's whole output row" in out, out
    # a link reads one whole return field of its writer, from the field's first word (the unit's one field, at 0), and
    # nothing else (HmLinks.checkUnit)
    bad, out = at("wrong_offset", *bound)
    print("wrong_offset", out.strip().splitlines()[-1])
    assert bad is None and f"read unit 0's outputs [1, {1 + N_OUT}), which are not one whole return field" in out, out
    # a link row is its tree port's registered row, whose width the port names (Registered.checkPort's Port.fits)
    port = _meta(d / "honest")["links"][0]["port"]
    bad, out = at("honest", *bound, reads="wrong_width.reads.json")
    print("wrong_width", out.strip().splitlines()[-1])
    assert bad is None and f"registered reads: port {port}'s rows are" in out, out
    bad, out = at("honest", *bound, reads="no_words.reads.json")
    print("no_words", out.strip().splitlines()[-1])
    assert bad is None and "a tree port names its rows' words" in out, out
    # an inner statement's description reads no row by path: a linked one is refused by checkLinked's name
    bad, out = at("plain_leaves", *bound)
    print("plain_leaves", out.strip().splitlines()[-1])
    assert bad is None and NO_TREE in out, out
    # without registered reads every linked statement is refused, the honest one included: one whose header names
    # registered rows as reading no root, and one that names none (layers 0 and 1 in one statement) by checkLinked's name
    for name, why, n in (("honest", NO_ROOTS, N), ("made_up", NO_ROOTS, N), ("wrong_salt", NO_ROOTS, N),
                         ("mixed", NO_TREE, 2 * N)):
        bad, out = at(name, *bound, reads=None, n=n)
        print(name, "without --registered:", out.strip().splitlines()[-1])
        assert bad is None and why in out, (name, out)
    bad, out = at("honest", "--partition", str(d / "partition.json"), reads=None)
    print("no-program", out.strip().splitlines()[-1])
    assert bad is None and "Linking needs the verifier's own program" in out, out
    # a one-layer class: staged with Wiring, no cut wire and no links, its units derived; staged without, as before (its
    # template the size class's own, so its units are taken as stated)
    k = tp2.population
    ok, out = _lean("--circuit", str(flat / "honest" / "circuit.txt"), "--public", str(flat / "honest" / f"pub-{k}.bin"),
                    "--partition", str(flat / "partition.json"), "--program", str(flat / "program.desc"))
    assert ok is not None and "no cut wire" in ok["partition_check"]["note"], out
    assert "links" not in _meta(flat / "honest") and not {"links", "definition"} & set(_meta(flat / "plain"))
    ok, out = _lean("--circuit", str(flat / "plain" / "circuit.txt"), "--public", str(flat / "plain" / f"pub-{k}.bin"),
                    "--partition", str(flat / "partition.json"))
    assert ok is not None and ok["partition_check"]["units"] == "as stated", out


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_sampled_proofs_own_setup_refuses_each_cheat(staged):
    """The same statements set up by the sampled proofs' own check (`OneStageZk.sessionAt`, through `spot-difftest`'s
    `session_setup`), each with the readers' registered reads."""
    d, *_ = staged
    subprocess.run(["lake", "build", "spot-difftest"], cwd=SPOT, check=True, capture_output=True)
    program = (d / "program.desc").read_bytes().hex()
    partition = json.loads((d / "partition.json").read_text())
    lines = []
    cases = (("honest", "readers"), ("own", "own"), ("made_up", "readers"), ("wrong_salt", "readers"),
             ("other_leaf", "readers"), ("other_words", "readers"), ("honest", "wrong_field"))
    for name, reads in cases:
        lines.append(json.dumps({"fn": "session_setup", "input": {
            "circuit": (d / name / "circuit.txt").read_bytes().hex(), "public": (d / name / f"pub-{N}.bin").read_bytes().hex(),
            "tables": [], "program": program, "partition": partition,
            "registered": json.loads((d / f"{reads}.reads.json").read_text()), "draw": None}}))
    p = subprocess.run([str(SPOT / ".lake" / "build" / "bin" / "spot-difftest")], input="\n".join(lines) + "\n",
                       capture_output=True, text=True)
    got = [json.loads(x) for x in p.stdout.splitlines()]
    assert len(got) == len(cases), p.stdout + p.stderr
    verdicts = {}
    for (name, reads), g in zip(cases, got):
        key = "wrong_field" if reads == "wrong_field" else name
        verdicts[key] = g.get("output", {}).get("refused", g)
        print("session_setup", key, verdicts[key])
    # its tree ports come with no reads, so every reader is refused by its instances' reads (checkPort), the honest one
    # included; the cheats' rows open no leaf of the registered tree
    for name, why in (("honest", NO_READ), ("own", NO_READ), ("made_up", NOT_OPEN), ("wrong_salt", NOT_OPEN),
                      ("other_leaf", NO_READ), ("other_words", NOT_OPEN), ("wrong_field", NO_READ)):
        assert verdicts[name] is not None and why in verdicts[name], (name, verdicts)


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
