"""Class-fixed regions: two private circuits of one class, `UniversalUnit_v1{G, N_IN, N_OUT}`, staged as the bench stages the
universal unit (`benchmarks/private_circuit/universal.py`: a one-Call Program partitioned by `Q_template_instance`,
`class_statement.stage` packed, each description source one row every instance opens, x one row per instance) on the same x
under one staging salt key, give one statement but for its regions' bytes. One description is the bench's (a random netlist
padded to the class), the other a uniformly random bit string of the class (any index, reads past the wires included); each
staging builds its own Program, units and partition.

* **The statement.** The circuit is the same byte for byte: its digest and class digest, and META's ports, ranges, leaf
  scheme, partition and program digest. So is every stage record, the partition object and the Program's descriptor. V*'s
  schedule (`rec_live.schedule_for`) and its outer statements under the cap (`rec_shape.outer`), each a function of the
  statement's public shape, are the same.
* **The regions' bytes.** The public files have one length and one header but its ports' frame-v3 roots (the same where the
  port's rows are), and the same refs; the prover's files the same salts. A region is a row port's `b || c` of every instance (`HmRow.regions`, whose
  names, free bits and positions read the circuit alone): a table row's `b || c` is the same exactly where its rows are, so
  x's are and the description's differ, and the test cannot pass vacuously.
* **Lean.** The Lean verifier (`flock-verify statement`, when a Lean toolchain is present) derives one statement digest (it
  hashes n, the blocks, m, k_log, the pin, the region count and Δ), m, region count and Δ from both, and differs only in the
  public digest, Σ and the Hello that carries Σ: what `ExecClass.lean`'s `shapeOf` drops.

(128, 64, 32) is the small class's check where a description row spans two leaves (its ``src``, 128 words), which (4, 4, 2)'s
single-leaf rows never do. The deployed class (8090, 64, 32) is opt-in (``VERITY_FLOCK_SAME_SHAPE_FULL=1``, a pod's): its
circuit is 31.8 GB and one staging took hours on node 1, at k_log 27 (fast100's bound) with a 14,208-word description row."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.codec import canonical_json, encode_program
from verity.ml.circuits.defs import bind
from verity.ml.private_circuits import universal as U

from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import partition_units as PU
from verity_flock import rec_live as RL
from verity_flock import rec_shape as RS
from verity_flock import rec_vstage as S

SPEC = "q-template-instance"
N = 4
LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
BC = 128
FULL = "VERITY_FLOCK_SAME_SHAPE_FULL"

#: two stagings of a class a test reads, after the slot circuits' cache (about 12 s)
pytestmark = pytest.mark.slow


def _description(rng: random.Random, G: int, N_IN: int, N_OUT: int, any_index: bool) -> U.Netlist:
    """The bench's description (3/4 of the gates over 3/4 of the inputs, in range, padded to the class), or any bit string of
    the class: every index any K-bit value."""
    if any_index:
        k = U.index_bits(G, N_IN)
        return U.Netlist(N_IN, tuple((rng.randrange(4), rng.randrange(1 << k), rng.randrange(1 << k)) for _ in range(G)),
                         tuple(rng.randrange(1 << k) for _ in range(N_OUT)))
    g0, n0 = max(1, 3 * G // 4), max(1, 3 * N_IN // 4)
    gates = tuple((rng.randrange(4), rng.randrange(n0 + g), rng.randrange(n0 + g)) for g in range(g0))
    return U.pad(U.Netlist(n0, gates, tuple(rng.randrange(n0 + g0) for _ in range(N_OUT))), G, N_IN)


def _lanes(units, cl, net: U.Netlist, xs: list[list[int]]):
    """The bench's `universal.lanes`: (inputs, outputs, row identities), the description's sources one row for every instance,
    x one row per instance."""
    act, uc = units.activation(cl.definition), cl.representative
    leaf_param = []
    for i, (_n, t) in enumerate(act.fn.signature()[0]):
        leaf_param += [i] * t.leaves
    x_param = len(act.fn.signature()[0]) - 1
    groups = CS.sources(act, uc, CS.owners_of(units, cl.definition.id))
    ins, want, rows = [[] for _ in uc.inputs], [[] for _ in uc.outputs], [[] for _ in groups]
    for t, x in enumerate(xs):
        run = PU.evaluate(act, [b for part in U.arguments(net, x) for b in part])
        for port, k in zip(ins, uc.inputs):
            port.append(run[k])
        for port, j in zip(want, uc.outputs):
            port.append(run[j])
        for ids, g in zip(rows, groups):
            ids.append(("x", t) if x_param in {leaf_param[-1 - uc.inputs[i]] for i in g} else ("program",))
    return ins, want, rows


def _stage(cls: tuple[int, int, int], xs: list[list[int]], net: U.Netlist, d: Path, key: bytes) -> dict:
    """One staging from scratch: the Program, its units and partition, the lanes and `class_statement.stage`."""
    fn = bind(U.UniversalUnit, G=cls[0], N_IN=cls[1], N_OUT=cls[2])
    program = PU.one_call(fn)
    query = PO.template_instance_query(fn)
    units = PU.PartitionUnits(program, query=query)
    [(shape, cl)] = units.classes().items()
    lanes = _lanes(units, cl, net, xs)
    st = CS.stage(units, SPEC, shape, cl, lanes, d, key, packed=True)
    (d / "partition.json").write_text(json.dumps(PO.build(program, query), sort_keys=True))
    (d / "program.desc").write_bytes(canonical_json(encode_program(program)))
    return {"stage": st, "x_ports": {f"p{i}" for i, ids in enumerate(lanes[2]) if ids[0][0] == "x"},
            "description_ports": {f"p{i}" for i, ids in enumerate(lanes[2]) if ids == [("program",)] * len(xs)}}


@pytest.fixture(scope="module", params=[(4, 4, 2), (128, 64, 32), pytest.param(
    (8090, 64, 32), marks=pytest.mark.skipif(not os.environ.get(FULL), reason=f"stages the deployed class twice, a 31.8 GB circuit "
                                             f"each, hours on a pod (set {FULL}=1)"))], ids=lambda c: "x".join(map(str, c)))
def staged(request, tmp_path_factory):
    CS.cache_slot_circuits()
    cls = request.param
    rng = random.Random(f"test-universal-same-shape/{cls}")
    xs = [[rng.randrange(2) for _ in range(cls[1])] for _ in range(N)]
    nets = [_description(rng, *cls, any_index=False), _description(rng, *cls, any_index=True)]
    key = hashlib.sha256(b"test-universal-same-shape").digest()
    d = tmp_path_factory.mktemp("same-shape")
    return {k: {**_stage(cls, xs, net, d / k, key), "dir": d / k, "net": net} for k, net in zip("ab", nets)}


def _files(d: Path) -> dict:
    """The staging's files cut where `circuit.write` lays them out (shared rows): the prover's rows and salts and the public
    `b || c` of every table row, by port; the refs; both headers."""
    text = (d / "circuit.txt").read_text()
    ports = FC._meta_ports(json.loads(text.split("\n", 2)[1][5:]))
    inst, pub = (d / f"inst-{N}.bin").read_bytes(), (d / f"pub-{N}.bin").read_bytes()
    head, body = inst.split(b"\n", 1)
    phead, public = pub.split(b"\n", 1)
    head = json.loads(head)
    counts = {p.name: head["shared_rows"][p.name] for p in ports}
    rows, salts, bc, at = {}, {}, {}, 0
    for p in ports:
        rows[p.name] = [body[at + 2 * p.words * r:at + 2 * p.words * (r + 1)] for r in range(counts[p.name])]
        at += 2 * p.words * counts[p.name]
    for p in ports:
        salts[p.name] = [body[at + FC.SALT_BYTES * r:at + FC.SALT_BYTES * (r + 1)] for r in range(counts[p.name])]
        at += FC.SALT_BYTES * counts[p.name]
    assert body[at:] == public
    at = 0
    for p in ports:
        bc[p.name] = [public[at + BC * r:at + BC * (r + 1)] for r in range(counts[p.name])]
        at += BC * counts[p.name]
    assert len(public) == at + 4 * N * len(ports)
    return {"text": text, "head": head, "pub_head": json.loads(phead), "pub_len": len(pub), "rows": rows, "salts": salts,
            "bc": bc, "refs": public[at:]}


def test_two_descriptions_of_a_class_stage_one_circuit_and_one_shape(staged):
    a, b = staged["a"], staged["b"]
    assert a["net"] != b["net"]
    assert (a["dir"] / "circuit.txt").read_bytes() == (b["dir"] / "circuit.txt").read_bytes()
    assert a["stage"] == b["stage"] and a["stage"]["n"] == N
    for f in ("partition.json", "program.desc"):
        assert (a["dir"] / f).read_bytes() == (b["dir"] / f).read_bytes()
    assert (a["x_ports"], a["description_ports"]) == (b["x_ports"], b["description_ports"])
    assert a["x_ports"] and a["description_ports"]
    shapes = []
    for s in (a, b):
        circuit = s["dir"] / "circuit.txt"
        meta, n = S.meta(circuit), S.instances(s["dir"] / f"pub-{N}.bin")
        sh, regions = S.shape(meta, n), len(meta["ports"])
        cap = RS.Cap.of(circuit, n, 2 + RL.LINK_POINTS * regions)
        shapes.append({"shape": (sh.m, sh.k_log), "regions": regions, "cap": cap, "outer": RS.outer(cap),
                       "schedule": RL.schedule_for({"dense_m": sh.m, "k_log": sh.k_log, "nbl": sh.m - sh.k_log, "regions": regions})})
    assert shapes[0] == shapes[1] and shapes[0]["outer"]


def test_the_statements_differ_only_in_the_regions_bytes(staged):
    a, b = staged["a"], staged["b"]
    fa, fb = _files(a["dir"]), _files(b["dir"])
    assert fa["pub_len"] == fb["pub_len"] and fa["refs"] == fb["refs"] and fa["salts"] == fb["salts"]
    for h in ("head", "pub_head"):
        va, vb = fa[h].pop("frame_v3"), fb[h].pop("frame_v3")
        assert fa[h] == fb[h]
        assert {k for k in va if va[k] != vb[k]} == {"roots"}
        assert all((va["roots"][p] == vb["roots"][p]) == (fa["rows"][p] == fb["rows"][p]) for p in fa["rows"])
    differ = set()
    for p in fa["rows"]:
        assert [len(r) for r in fa["bc"][p]] == [len(r) for r in fb["bc"][p]]
        for ra, rb, ca, cb in zip(fa["rows"][p], fb["rows"][p], fa["bc"][p], fb["bc"][p], strict=True):
            assert (ra == rb) == (ca == cb)
            if ra != rb:
                differ.add(p)
    assert not differ & a["x_ports"]
    assert a["description_ports"] & differ


def _lean(d: Path) -> dict:
    p = subprocess.run([str(BIN), "statement", "--circuit", str(d / "circuit.txt"), "--public", str(d / f"pub-{N}.bin"),
                        "--partition", str(d / "partition.json")], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(p.stdout.splitlines()[0])


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_lean_derives_one_statement_but_for_its_public_digest(staged):
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    a, b = (_lean(staged[k]["dir"]) for k in "ab")
    assert {k for k in a if a[k] != b[k]} == {"public_sha512", "sigma", "hello"}
    assert a["hello"].replace(a["sigma"], b["sigma"]) == b["hello"]
    meta = S.meta(staged["a"]["dir"] / "circuit.txt")
    assert (a["m"], a["regions"]) == (S.shape(meta, N).m, len(meta["ports"]))
    assert a["circuit_sha512"] == staged["a"]["stage"]["circuit_sha512"] and a["partition_check"]["units"] == "as stated"
