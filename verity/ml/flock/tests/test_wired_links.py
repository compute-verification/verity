"""A small wired class statement: two layers of the universal unit, layer 1 reading layer 0's outputs, one
`Q_template_instance` class staged with `class_statement.Wiring`. Layer 1 reads layer 0's outputs through a link port, a row
committed under a salt of its own, and the statement declares no links: that the link row is its writer's output row is the
recursive statement's to check, so C-Flock verifies each statement alone, over its own rows and salts. The Lean verifier
(`flock-verify statement --program`, when a Lean toolchain is present) accepts the honest statement and the one whose link
row is made up, and refuses a statement that declares links (META `links`) and a links file (`--links`). When
``FLOCK_CIRCUIT`` names a build of ``flock-circuit``, the live verifier gives the same verdicts."""
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
    """The program, its partition and the stagings: honest (with `Wiring`), and with ``wired`` the cheat (layer 1's x reads
    layer 0's outputs flipped), else the honest lanes staged without `Wiring` (``plain``)."""
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
        meta = _meta(d / "honest")
        x0 = len(tp.inputs(0)) - N_IN
        lines = (d / "honest" / "circuit.txt").read_text().split("\n", 2)
        lines[1] = "META " + json.dumps({**meta, "links": [{"port": "p0", "leaves": [x0 + N_IN - N_OUT, x0 + N_IN]}]},
                                        separators=(",", ":"), sort_keys=True)
        (d / "linked").mkdir()
        (d / "linked" / "circuit.txt").write_text("\n".join(lines))
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


def test_a_wired_unit_reads_its_writer_through_a_link_port_and_the_statement_declares_no_links(staged):
    d, (fn, _p, _q, tp, st), _flat, _ = staged
    meta = _meta(d / "honest")
    assert meta["definition"] == descriptor_id(fn) and st["honest"]["definition"] == descriptor_id(fn)
    assert "links" not in meta
    x0 = len(tp.inputs(0)) - N_IN
    assert st["honest"]["link_ports"] == [[x0 + N_IN - N_OUT, x0 + N_IN]]   # layer 1's x after its two circuit-input bits
    assert {s["circuit_sha512"] for s in st.values()} == {st["honest"]["circuit_sha512"]}
    assert "links" in _meta(d / "linked")


def test_no_reader_row_is_committed_under_its_writers_salt(staged):
    """A link row's `b || c` is its own: no input row of any instance equals any instance's output row, so the public file
    shows no wire."""
    d, (_fn, _p, _q, tp, _st), _flat, _ = staged
    n = tp.population
    names = [p[0] for p in _meta(d / "honest")["ports"]]
    for k in ("honest", "made_up"):
        bc = _bc(d / k, n)
        outs = {bc(w, "out") for w in range(n)}
        assert not any(bc(r, p) in outs for r in range(n) for p in names if p != "out")


def _lean(*args: str) -> tuple[dict | None, str]:
    p = subprocess.run([str(BIN), "statement", *args], capture_output=True, text=True)
    return (json.loads(p.stdout.splitlines()[0]) if p.returncode == 0 else None), p.stdout + p.stderr


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_lean_verifies_each_statement_alone_and_refuses_links(staged):
    d, (_fn, _p, _q, tp, _st), flat, (_f2, _p2, _q2, tp2, _st2) = staged
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    n = tp.population
    bound = ["--partition", str(d / "partition.json"), "--program", str(d / "program.desc")]
    for k in ("honest", "made_up"):
        ok, out = _lean("--circuit", str(d / k / "circuit.txt"), "--public", str(d / k / f"pub-{n}.bin"), *bound)
        assert ok is not None and ok["partition_check"]["units"] == "derived", out
        assert "Linking" not in ok["partition_check"]["note"] and "unit-class/" in ok["partition_check"]["note"], out
    bad, out = _lean("--circuit", str(d / "linked" / "circuit.txt"), "--public", str(d / "honest" / f"pub-{n}.bin"), *bound)
    assert bad is None and "links between proof blocks are recursion's" in out, out
    bad, out = _lean("--circuit", str(d / "honest" / "circuit.txt"), "--public", str(d / "honest" / f"pub-{n}.bin"), *bound,
                     "--links", str(d / "partition.json"))
    assert bad is None and "links between proof blocks are recursion's" in out, out
    k = tp2.population
    ok, out = _lean("--circuit", str(flat / "honest" / "circuit.txt"), "--public", str(flat / "honest" / f"pub-{k}.bin"),
                    "--partition", str(flat / "partition.json"), "--program", str(flat / "program.desc"))
    assert ok is not None and ok["partition_check"]["units"] == "derived", out
    assert not {"links", "definition"} & set(_meta(flat / "plain"))
    ok, out = _lean("--circuit", str(flat / "plain" / "circuit.txt"), "--public", str(flat / "plain" / f"pub-{k}.bin"),
                    "--partition", str(flat / "partition.json"))
    assert ok is not None and ok["partition_check"]["units"] == "as stated", out


@pytest.mark.skipif(not (os.environ.get("FLOCK_CIRCUIT") and Path(os.environ["FLOCK_CIRCUIT"]).exists()),
                    reason="FLOCK_CIRCUIT names no flock-circuit build")
def test_the_live_verifier_verifies_each_statement_alone_and_refuses_links(staged):
    d, (_fn, _p, _q, tp, _st), flat, (_f2, _p2, _q2, tp2, _st2) = staged
    n = tp.population

    def live(sdir: Path, k: int, *extra: str, circuit: Path | None = None) -> str:
        p = subprocess.run([os.environ["FLOCK_CIRCUIT"], "statement", "--circuit", str(circuit or sdir / "circuit.txt"),
                            "--instances", str(sdir / f"pub-{k}.bin"), *extra], capture_output=True, text=True)
        return p.stdout + p.stderr

    assert "STATEMENT\t" in live(d / "honest", n) and "STATEMENT\t" in live(d / "made_up", n)
    assert "links between proof blocks are recursion's" in live(d / "honest", n, circuit=d / "linked" / "circuit.txt")
    assert "links between proof blocks are recursion's" in live(d / "honest", n, "--links", str(d / "partition.json"))
    assert "STATEMENT\t" in live(flat / "plain", tp2.population)
