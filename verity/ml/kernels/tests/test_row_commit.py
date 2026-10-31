"""``row_commit.cuh`` against ``verity.ml.commitments``: its frame-v3-sha512 leaf, pad and node are `CommitmentDomain`'s at
every length of uint below 2^32, through the compile-time and the run-time forms (the leaf's also a body a length,
`frame_leaf_nb`, and the node's also through `Compact`), and its builders' leaf at a rank and a position apart is too; its
trees give ``frame_v3/vectors_sha512.json``'s leaves, pads and roots; its hm96-sha512 leaf, through every table and loop of
the mask, gives ``hm96/vectors_sha512.json``'s commit strings and leaves under both keys, the gates' reference on fresh rows
and salts and the vectors' hiding leaves; and the gates' Definitions compute what it does.
``row_commit_cases.cu`` runs the cases, built by the host's C++ compiler and, where nvcc and a GPU are present, on the
device; the device build is cached (`cuda_build`), so a GPU job builds it (``cuda_lib``) before it leases a GPU."""
from __future__ import annotations

import ctypes
import hashlib
import json
import random
from functools import cache
from pathlib import Path
from typing import Callable

import cuda_build
import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments import hm96
from verity.ml.commitments.frame_v3.vectors_sha512 import domain
from verity.ml.commitments.gates import frame_v3 as F
from verity.ml.commitments.gates import hm96 as H
from verity.ml.commitments.gates.sha512 import midstate, words_be
from verity.ml.commitments.merkle import merkle_depth
from verity.ml.kernels import ROW_COMMIT, SHA512

SOURCE = Path(__file__).resolve().with_name("row_commit_cases.cu")
COMMITMENTS = Path(F.__file__).resolve().parents[1]
FV = json.loads((COMMITMENTS / "frame_v3" / "vectors_sha512.json").read_text())
HV = json.loads((COMMITMENTS / "hm96" / "vectors_sha512.json").read_text())
INST = hm96.SHA512
KEYS = {"default": bytes.fromhex(HV["default_key"]), "other": bytes.fromhex(HV["other_key"])}
ROW, SEG = "hm96-sha512/row/v2", "hm96-sha512/row-seg/v1"
#: the hm96 schemas (18 and 22 bytes, the run-time forms') and two plain row schemas (13 and 17)
SCHEMAS = (ROW, SEG, "sha512/row/v2", "sha512/row-seg/v1")
#: both sides of every step of uint's length below 2^32
EDGES = (0, 1, 255, 256, 65535, 65536, (1 << 24) - 1, 1 << 24, (1 << 32) - 1)
LEAF, PAD, NODE, HM96, EMPTY, LEAF_AT = range(6)

u8 = ctypes.c_uint8


class Case(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int32), ("form", ctypes.c_int32), ("sl", ctypes.c_int32), ("level", ctypes.c_int32),
                ("pos", ctypes.c_uint64), ("position", ctypes.c_uint64), ("did", u8 * 64), ("schema", u8 * 32),
                ("value", u8 * 64), ("L", u8 * 64), ("R", u8 * 64), ("x", u8 * 64), ("y", u8 * 192)]


class Out(ctypes.Structure):
    _fields_ = [("bc", u8 * 128), ("hs", u8 * 64), ("out", u8 * 64), ("status", ctypes.c_int32), ("pad_", ctypes.c_int32 * 3)]


def case(kind: int, form: int = 0, *, did: bytes = b"", schema: str = ROW, pos: int = 0, position: int = 0, level: int = 0,
         value: bytes = b"", L: bytes = b"", R: bytes = b"", x: bytes = b"", y: bytes = b"") -> Case:
    s = schema.encode()
    k = Case(kind=kind, form=form, sl=len(s), level=level, pos=pos, position=position)
    for name, data in (("did", did), ("schema", s), ("value", value), ("L", L), ("R", R), ("x", x), ("y", y)):
        ctypes.memmove(getattr(k, name), data, len(data))
    return k


Run = Callable[..., list]


def _load(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    assert lib.rc_case_bytes() == ctypes.sizeof(Case) and lib.rc_out_bytes() == ctypes.sizeof(Out)
    return lib


@cache
def cuda_lib() -> ctypes.CDLL:
    return _load(cuda_build.device_lib(SOURCE, (ROW_COMMIT, SHA512)))


def _runner(lib: ctypes.CDLL, cuda: bool = False) -> Run:
    """run(cases, key) -> each case's Out, every case's status 0"""
    tables: dict[bytes, tuple] = {}

    def run(cases: list[Case], key: bytes = KEYS["default"]) -> list[Out]:
        if not cases:
            return []
        if key not in tables:
            t = (ctypes.create_string_buffer(8192), ctypes.create_string_buffer(256 * 272), ctypes.create_string_buffer(256 * 256))
            lib.rc_tables(key, *t)
            tables[key] = t
        n = len(cases)
        outs = (Out * n)()
        if cuda:
            assert lib.rc_cases_cuda((Case * n)(*cases), n, key, *tables[key], outs) == 0
        else:
            lib.rc_cases((Case * n)(*cases), n, key, *tables[key], outs)
        assert [o.status for o in outs] == [0] * n
        return list(outs)

    return run


@pytest.fixture(scope="module")
def host(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    return _load(cuda_build.host_lib(SOURCE, (), tmp_path_factory.mktemp("host") / "row_commit.so"))


def _hex(s: str) -> bytes:
    return bytes.fromhex(s)


# --- the checks, each over a runner (the host's or the device's) ------------------------------------------------------------

def _check_leaves(run: Run) -> None:
    rng = random.Random("row-commit/leaves")
    cases, want = [], []
    for schema in SCHEMAS:
        did = rng.randbytes(64)
        for pos in EDGES:
            v = rng.randbytes(64)
            for form in (0, 1, 2) if schema in (ROW, SEG) else (0, 2):
                cases.append(case(LEAF, form, did=did, schema=schema, pos=pos, value=v))
                want.append(F.leaf_ref(did, schema, v, pos))
    for rank, position in ((0, 7), (3, 65536), (1 << 20, 5), ((1 << 32) - 1, (1 << 40) + 3)):
        did, v = rng.randbytes(64), rng.randbytes(64)
        cases.append(case(LEAF_AT, did=did, pos=rank, position=position, value=v))
        want.append(F.leaf_at_ref(did, ROW, v, rank, position))
    assert [bytes(o.out) for o in run(cases)] == want


def _check_pads_and_nodes(run: Run) -> None:
    rng = random.Random("row-commit/nodes")
    d = domain(1 << 20, "row-commit/nodes")
    did = d.domain_id
    cases = [case(PAD, form, did=did, pos=rank) for rank in EDGES for form in (0, 1)]
    want = [d.pad(rank) for rank in EDGES for _ in (0, 1)]
    for level in (0, 1, 7, 31, 32):
        for index in EDGES:
            left, right = rng.randbytes(64), rng.randbytes(64)
            cases += [case(NODE, form, did=did, level=level, pos=index, L=left, R=right) for form in (0, 1, 2)]
            want += [d.node(level, index, left, right)] * 3
    cases.append(case(EMPTY, did=did))
    want.append(d.empty_root())
    assert [bytes(o.out) for o in run(cases)] == want


def _tree(run: Run, did: bytes, schema: str, values: list[bytes], form: int) -> tuple[list[bytes], list[bytes], bytes]:
    """The leaves, pads and root of the frame-v3-sha512 tree over values (merkle.tree_levels' shape)."""
    out = lambda os: [bytes(o.out) for o in os]  # noqa: E731
    leaves = out(run([case(LEAF, did=did, schema=schema, pos=i, value=v) for i, v in enumerate(values)]))
    width = 1 << merkle_depth(len(leaves))
    pads = out(run([case(PAD, form, did=did, schema=schema, pos=r) for r in range(len(leaves), width)]))
    level, depth = leaves + pads, 0
    while len(level) > 1:
        level = out(run([case(NODE, form, did=did, level=depth, pos=i // 2, L=level[i], R=level[i + 1])
                         for i in range(0, len(level), 2)]))
        depth += 1
    return leaves, pads, level[0]


def _check_trees(run: Run) -> None:
    trees = FV["row_trees"] + FV["bit_row_trees"] + FV["binding64_trees"]
    assert len(trees) == 8
    for t in trees:
        assert t["domain"]["range"][0] == 0
        did = _hex(t["domain"]["domain_id"])
        for form in (0, 1, 2):
            leaves, pads, root = _tree(run, did, t["schema"], [_hex(v) for v in t["digests"]], form)
            assert (leaves, pads, root) == ([_hex(v) for v in t["leaves"]], [_hex(v) for v in t["pads"]], _hex(t["root"]))
    e = FV["empty_root"]
    assert bytes(run([case(EMPTY, did=_hex(e["domain"]["domain_id"]))])[0].out) == _hex(e["root"])


def _hm96_cases() -> dict[str, list[tuple]]:
    """Per key: (x, y, did, schema, pos, b || c, h, frame leaf) from the vectors and from the gates' reference."""
    rng = random.Random("row-commit/hm96")
    out: dict[str, list[tuple]] = {"default": [], "other": []}

    def add(name: str, x: bytes, y: bytes, bc: bytes, h: bytes, did: bytes, schema: str, pos: int, leaf: bytes | None = None):
        out[name].append((x, y, did, schema, pos, bc, h, leaf if leaf is not None else F.leaf_ref(did, schema, h, pos)))

    for i, v in enumerate(HV["leaves"]):
        add("default", _hex(v["inner_digest"]), _hex(v["salt"]), _hex(v["b"]) + _hex(v["c"]), _hex(v["leaf"]),
            rng.randbytes(64), (ROW, SEG)[i % 2], EDGES[i % len(EDGES)])
    for i, m in enumerate(HV["masks"]):
        key, bc = KEYS[m["key"]], _hex(m["mask"]) + _hex(m["salt_digest"])
        add(m["key"], bytes(64), _hex(m["salt"]), bc, INST.tree_leaf(key, bc), rng.randbytes(64), (SEG, ROW)[i % 2],
            EDGES[-1 - i % len(EDGES)])
    for i in range(6):
        x, y = rng.randbytes(64), rng.randbytes(192)
        b, c, h = H.reference(x, y)
        add("default", x, y, b + c, h, rng.randbytes(64), (ROW, SEG)[i % 2], rng.randrange(1 << (8 * (i % 4 + 1))))
    for v in FV["bit_row_hiding_leaves"]:
        assert v["key_sha512"] == hashlib.sha512(KEYS["default"]).hexdigest()
        add("default", _hex(v["digest"]), _hex(v["salt"]), _hex(v["bc"]), _hex(v["hm96_leaf"]), _hex(v["domain"]["domain_id"]),
            v["schema"], v["rank"], _hex(v["leaf"]))
    return out


def _check_hm96(run: Run, forms: int) -> None:
    for name, cases in _hm96_cases().items():
        for form in range(forms):
            got = run([case(HM96, form, did=did, schema=schema, pos=pos, x=x, y=y) for x, y, did, schema, pos, *_ in cases],
                      KEYS[name])
            assert [(bytes(o.bc), bytes(o.hs), bytes(o.out)) for o in got] == [c[5:] for c in cases], (name, form)


# --- the tests ----------------------------------------------------------------------------------------------------------------

def test_the_package_names_the_header_and_it_includes_the_one_sha512() -> None:
    assert ROW_COMMIT.resolve() == SOURCE.parents[1] / "row_commit.cuh"
    assert '#include "sha512.cuh"' in ROW_COMMIT.read_text()
    assert '#include "../row_commit.cuh"' in SOURCE.read_text()


def test_leaves_at_every_uint_length_are_the_domains(host: ctypes.CDLL) -> None:
    _check_leaves(_runner(host))


def test_pads_nodes_and_the_empty_root_are_the_domains(host: ctypes.CDLL) -> None:
    _check_pads_and_nodes(_runner(host))


def test_trees_give_the_vectors_roots(host: ctypes.CDLL) -> None:
    _check_trees(_runner(host))


def test_the_hm96_plan_is_the_prefixes_midstates(host: ctypes.CDLL) -> None:
    for key in KEYS.values():
        out = (ctypes.c_uint64 * 16)()
        host.rc_hm96_plan(key, out)
        assert list(out) == midstate(INST.salt_prefix) + midstate(INST.leaf_prefix(key))


def test_hm96_leaves_through_every_mask_are_the_vectors_and_the_reference(host: ctypes.CDLL) -> None:
    assert host.rc_hm96_forms() == 5
    _check_hm96(_runner(host), host.rc_hm96_forms())


def _digest(bits) -> bytes:
    return b"".join(sum(int(b) << j for j, b in enumerate(w)).to_bytes(8, "big") for w in np.asarray(bits).reshape(8, 64))


def _bits(d: bytes) -> list[int]:
    return [(w >> j) & 1 for w in words_be(d) for j in range(64)]


@pytest.mark.slow
def test_the_definitions_compute_what_the_header_does(host: ctypes.CDLL) -> None:
    """Minutes of tracing: the hm96 leaf, a one-byte frame leaf of each hm96 schema and a one-byte node held to the gates'
    Definitions themselves, where the other tests reach them through the references the commitments suite holds them to."""
    run, rng = _runner(host), random.Random("row-commit/definitions")
    hc = _hm96_cases()["default"][:2]
    got = run([case(HM96, did=did, schema=schema, pos=pos, x=x, y=y) for x, y, did, schema, pos, *_ in hc])
    args = [H.arguments(x, y) for x, y, *_ in hc]
    ev = evaluate_bits(bind(H.Hm96Sha512Leaf), *(np.array([a[k] for a in args], dtype=np.uint8) for k in range(2)))
    for o, e in zip(got, np.asarray(ev).reshape(len(hc), 3, 512)):
        assert (bytes(o.bc), bytes(o.hs)) == (_digest(e[0]) + _digest(e[1]), _digest(e[2]))
    did = rng.randbytes(64)
    for schema in (ROW, SEG):
        xs, value = (0, 1, 200, 255), rng.randbytes(64)
        got = run([case(LEAF, did=did, schema=schema, pos=x, value=value) for x in xs])
        ev = evaluate_bits(bind(F.FrameV3Leaf, B=1, SCHEMA=schema), np.array([_bits(did)] * len(xs)),
                           np.array([_bits(value)] * len(xs)), np.array([[(x >> j) & 1 for j in range(8)] for x in xs], dtype=np.uint8))
        assert [bytes(o.out) for o in got] == [_digest(e) for e in ev], schema
    nodes = [(0, 0), (3, 255), (31, 17), (32, 128)]
    left, right = rng.randbytes(64), rng.randbytes(64)
    got = run([case(NODE, did=did, level=k, pos=i, L=left, R=right) for k, i in nodes])
    ev = evaluate_bits(bind(F.FrameV3Node, B=1), np.array([_bits(did)] * len(nodes)),
                       np.array([[(k >> j) & 1 for j in range(8)] for k, _ in nodes], dtype=np.uint8),
                       np.array([[(i >> j) & 1 for j in range(8)] for _, i in nodes], dtype=np.uint8),
                       np.zeros((len(nodes), 1), dtype=np.uint8), np.array([_bits(left)] * len(nodes)),
                       np.array([_bits(right)] * len(nodes)))
    assert [bytes(o.out) for o in got] == [_digest(e) for e in ev]


@pytest.mark.slow
@pytest.mark.skipif(cuda_build.gpu_arch() is None, reason="no nvcc or no CUDA device")
def test_the_device_computes_every_case_as_the_host(host: ctypes.CDLL) -> None:
    """Slow for its nvcc build, where the cache has none: the device's loads, funnel shifts and vector stores, which the host
    build doesn't have."""
    run = _runner(cuda_lib(), cuda=True)
    _check_leaves(run)
    _check_pads_and_nodes(run)
    _check_trees(run)
    _check_hm96(run, host.rc_hm96_forms())
