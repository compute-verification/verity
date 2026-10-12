"""Linking (PROTOCOL.md §16.13) on a small wired class statement: two layers of the universal unit, layer 1 reading layer 0's
outputs, one `Q_template_instance` class staged with `class_statement.Wiring`. The honest staging's link rows have their
writers' output rows' `b || c`; a prover that gives layer 1 a made-up input, and one that commits the right value under a salt
of its own, stage the same circuit with link rows that differ. The Lean verifier (`flock-verify statement --program`, when a Lean
toolchain is present) accepts the honest statement and refuses both cheats and a linked statement without its program; a
one-layer statement, with no cut wire, is accepted with its units derived when staged with `Wiring`, and as before when
staged without. When ``FLOCK_CIRCUIT`` names a build of ``flock-circuit``, the
live verifier gives the same verdicts with its own links file. Linking binds a public topology; it does not hide the
topology."""
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

from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.codec import canonical_json, descriptor_id, encode_program
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import array_of, coll_concat
from verity.ml.circuits.types import Array
from verity.ml.private_circuits import universal as U

from verity_flock import class_statement as CS
from verity_flock import partition_units as PU

G, N_IN, N_OUT, N = 4, 4, 2, 2
NC = N_IN + N_OUT
#: layer 0 reads c[0:4]; layer 1 reads c[4:6], then layer 0's outputs
WIRED = ((("c", 0, N_IN),), (("c", N_IN, NC), ("u", 0, 0, N_OUT)))
FLAT = ((("c", 0, N_IN),),)
SPEC = "q-template-instance"
LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"

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
    identities as the bench's (`universal.lanes`): x one row per instance and layer, the netlist one row per layer."""
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


def _stage(topo, d: Path, wired: bool):
    """The program, its partition and the stagings: honest (with `Wiring`), and with ``wired`` the two cheats (layer 1's x reads
    layer 0's outputs flipped; the honest rows under link salts of their own), else the honest lanes staged without `Wiring`
    (``plain``)."""
    rng = random.Random(f"test-wired-links/{len(topo)}")
    fn, program = _program(topo)
    nets = [U.pad(U.Netlist(N_IN, tuple((rng.randrange(4), rng.randrange(N_IN + g), rng.randrange(N_IN + g)) for g in range(G)),
                            tuple(rng.randrange(N_IN + G) for _ in range(N_OUT))), G, N_IN) for _ in topo]
    inputs = [rng.randrange(2) for _ in range(N * NC)] + [b for net in nets for p in U.arguments(net, [0] * N_IN)[:3] for b in p]
    query = PO.template_instance_query(fn)
    units = PU.PartitionUnits(program, query=query)
    [(shape, cl)] = units.classes().items()
    tp = PO.TemplatePartition(program, query)
    assert min(g for u in range(tp.population) for g in tp.outputs(u)) >= len(inputs)
    (d / "partition.json").write_text(json.dumps(PO.build(program, query), sort_keys=True))
    (d / "program.desc").write_bytes(canonical_json(encode_program(program)))
    key = hashlib.sha256(b"test-wired-links").digest()
    wiring = CS.Wiring(program, query, list(range(tp.population)))
    st = {"honest": CS.stage(units, SPEC, shape, cl, _lanes(units, cl, tp, inputs), d / "honest", key, packed=True, wiring=wiring)}
    if not wired:
        st["plain"] = CS.stage(units, SPEC, shape, cl, _lanes(units, cl, tp, inputs), d / "plain", key, packed=True)
    else:
        flip = {N + t: (lambda w: w[:-N_OUT] + [1 - b for b in w[-N_OUT:]]) for t in range(N)}
        st["made_up"] = CS.stage(units, SPEC, shape, cl, _lanes(units, cl, tp, inputs, flip), d / "made_up", key, packed=True,
                                 wiring=replace(wiring, strict=False))
        st["wrong_salt"] = CS.stage(units, SPEC, shape, cl, _lanes(units, cl, tp, inputs), d / "wrong_salt", key, packed=True,
                                    wiring=replace(wiring, writer_salts=False))
        text = (d / "honest" / "circuit.txt").read_text()
        (d / "links.json").write_text(json.dumps(CS.links_file(program, query, text, wiring.indices)))
    return fn, program, query, tp, st


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    CS.cache_slot_circuits()
    d = tmp_path_factory.mktemp("wired")
    flat = tmp_path_factory.mktemp("flat")
    return d, _stage(WIRED, d, True), flat, _stage(FLAT, flat, False)


def _meta(d: Path) -> dict:
    return json.loads((d / "circuit.txt").read_text().split("\n", 2)[1][5:])


def _bc(d: Path, n: int):
    """The public file's `b || c` of instance ``i``'s row ``name`` (a shared-row file: its table row, by the instance's ref)."""
    meta = _meta(d)
    head, body = (d / f"pub-{n}.bin").read_bytes().split(b"\n", 1)
    head = json.loads(head)
    names = [p[0] for p in meta["ports"]]
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
    assert {s["circuit_sha512"] for s in st.values()} == {st["honest"]["circuit_sha512"]}
    links = json.loads((d / "links.json").read_text())
    assert links["format"] == CS.LINKS_FORMAT and links["ports"] == [link["port"]]
    assert links["links"] == [[N + t, link["port"], t] for t in range(N)]


def test_honest_link_rows_are_their_writers_output_rows_and_each_cheat_breaks_them(staged):
    d, (_fn, _p, _q, tp, _st), _flat, _ = staged
    n = tp.population
    port = _meta(d / "honest")["links"][0]["port"]
    triples = json.loads((d / "links.json").read_text())["links"]
    bc = {k: _bc(d / k, n) for k in ("honest", "made_up", "wrong_salt")}
    assert all(bc["honest"](r, port) == bc["honest"](w, "out") for r, _p, w in triples)
    assert all(bc["made_up"](r, port) != bc["made_up"](w, "out") for r, _p, w in triples)
    assert all(bc["wrong_salt"](r, port) != bc["wrong_salt"](w, "out") for r, _p, w in triples)
    # the wrong-salt rows hold the right values: its writers' output rows are the honest ones
    assert all(bc["wrong_salt"](w, "out") == bc["honest"](w, "out") for _r, _p, w in triples)


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


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_lean_accepts_honest_links_and_refuses_each_cheat_and_links_without_the_program(staged):
    d, (_fn, _p, _q, tp, _st), flat, (_f2, _p2, _q2, tp2, _st2) = staged
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    n = tp.population
    bound = ["--partition", str(d / "partition.json"), "--program", str(d / "program.desc")]
    ok, out = _lean("--circuit", str(d / "honest" / "circuit.txt"), "--public", str(d / "honest" / f"pub-{n}.bin"), *bound)
    assert ok is not None, out
    note = ok["partition_check"]["note"]
    assert ok["partition_check"]["units"] == "derived" and f"Linking checked: {N} cut wire(s)" in note and "unit-class/" in note
    for cheat in ("made_up", "wrong_salt"):
        bad, out = _lean("--circuit", str(d / cheat / "circuit.txt"), "--public", str(d / cheat / f"pub-{n}.bin"), *bound)
        assert bad is None and "links: cut wire" in out and "the reader's row" in out, out
    bad, out = _lean("--circuit", str(d / "honest" / "circuit.txt"), "--public", str(d / "honest" / f"pub-{n}.bin"),
                     "--partition", str(d / "partition.json"))
    assert bad is None and "Linking needs the verifier's own program" in out, out
    # a one-layer class: staged with Wiring, no cut wire and no links, its units derived; staged without, as before (its
    # template the unit class's own, so its units are taken as stated)
    k = tp2.population
    ok, out = _lean("--circuit", str(flat / "honest" / "circuit.txt"), "--public", str(flat / "honest" / f"pub-{k}.bin"),
                    "--partition", str(flat / "partition.json"), "--program", str(flat / "program.desc"))
    assert ok is not None and "no cut wire" in ok["partition_check"]["note"], out
    assert "links" not in _meta(flat / "honest") and not {"links", "definition"} & set(_meta(flat / "plain"))
    ok, out = _lean("--circuit", str(flat / "plain" / "circuit.txt"), "--public", str(flat / "plain" / f"pub-{k}.bin"),
                    "--partition", str(flat / "partition.json"))
    assert ok is not None and ok["partition_check"]["units"] == "as stated", out


@pytest.mark.skipif(not (os.environ.get("FLOCK_CIRCUIT") and Path(os.environ["FLOCK_CIRCUIT"]).exists()),
                    reason="FLOCK_CIRCUIT names no flock-circuit build")
def test_the_live_verifier_checks_links_against_its_own_file(staged):
    d, (_fn, _p, _q, tp, _st), flat, (_f2, _p2, _q2, tp2, _st2) = staged
    n = tp.population

    def live(sdir: Path, k: int, *extra: str) -> str:
        p = subprocess.run([os.environ["FLOCK_CIRCUIT"], "statement", "--circuit", str(sdir / "circuit.txt"),
                            "--instances", str(sdir / f"pub-{k}.bin"), *extra], capture_output=True, text=True)
        return p.stdout

    links = ["--links", str(d / "links.json")]
    assert "STATEMENT\t" in live(d / "honest", n, *links)
    for cheat in ("made_up", "wrong_salt"):
        assert "the reader's row is not the writer's committed output row" in live(d / cheat, n, *links)
    assert "pass the verifier's own links file" in live(d / "honest", n)
    assert "STATEMENT\t" in live(flat / "plain", tp2.population)
    assert "the circuit declares no links" in live(flat / "plain", tp2.population, *links)
