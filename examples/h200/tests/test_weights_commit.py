"""The weights committer (`e2e_h200.weights_commit`) at MINI_ROWS, MINI with the frozen shape's one registered row a ``wrow``
part: its streamed roots are core's `Registered.commit`'s under the witness's salts, its entries the verifier's own
(``registered_entries`` of the served Program), and every ``wrow``/``sw`` read of every ScaledMm proof block, located from the
Program's own references, opens against them; a wrong path, position or salt is refused. Nothing calls
``registered.parameter_gates`` but the Program's ground truth here. Building MINI_ROWS's Program (~45 s) is the cost: only
its references say which rows a proof block reads, which the committer must agree with without building it."""
from __future__ import annotations

import json
import os
import stat

import numpy as np
import pytest
from e2e_h200 import weights_commit as WC
from e2e_h200.program import program
from e2e_h200.shape import MINI, MINI_ROWS
from e2e_h200 import served_weights as SW

from verity.core.protocols.sampled_proofs.one_stage import registered as REG
from verity.ml.circuits.codec import program_digest

SHAPE = MINI_ROWS
SEED = SW.TEST_SEED
#: small enough that every linear's rows are several chunks, so a path crosses the chunk roots
CHUNK = 1 << 10


def _no_gates(*_a, **_k):
    raise AssertionError("the committer went through registered.parameter_gates")


@pytest.fixture(scope="module")
def prog():
    return program(SHAPE)


@pytest.fixture(scope="module")
def digest(prog):
    return program_digest(prog)


@pytest.fixture(scope="module")
def committed(tmp_path_factory, digest):
    state = tmp_path_factory.mktemp("state")
    mp = pytest.MonkeyPatch()
    mp.setattr(REG, "parameter_gates", _no_gates)
    try:
        doc = WC.commit(SHAPE, SEED, digest, state, workers=2, chunk_bytes=CHUNK)
        return state, doc, WC.open_state(state, SEED)
    finally:
        mp.undo()


def _witness_salts(state, t: SW.Tensor) -> list[bytes]:
    base = WC._bases(SHAPE)[[x.name for x in SW.tensors(SHAPE)].index(t.name)]
    data = (state / "salts.bin").read_bytes()[base * WC.SALT:(base + t.leaves) * WC.SALT]
    return [data[k * WC.SALT:(k + 1) * WC.SALT] for k in range(t.leaves)]


def test_the_reader_is_the_programs_without_its_gate_lists(prog, digest, monkeypatch):
    want = SW.reader(SHAPE, prog)
    monkeypatch.setattr(REG, "parameter_gates", _no_gates)
    got = WC.reader(SHAPE, digest)
    assert got == want
    assert dict(got.widths)["l0.wq.scale"] == 16


def test_the_streamed_roots_are_registered_commits_under_the_witness_salts(committed, digest):
    state, doc, _s = committed
    assert doc["chunks"] > len(SW.tensors(SHAPE))
    roots = {}
    for t, e in zip(SW.tensors(SHAPE), doc["registered"]):
        b = dict(doc["widths"])[t.name]
        rows = np.frombuffer(SW.tensor_bytes(SHAPE, SEED, t.name), "<u2" if b == 16 else "<u4")
        want = REG.Registered.commit(digest, t.name, rows.reshape(t.leaves, -1).tolist(), _witness_salts(state, t),
                                     word_bits=b, schema=t.schema)
        assert e == want.entry(), t.name
        roots[t.name] = want.root
    assert doc["values_digest"] == SW.registration_digest(SHAPE, WC.reader(SHAPE, digest), roots)


def test_its_entries_are_the_verifiers_own(committed, prog):
    _state, doc, _s = committed
    words = {e["value"]: e["words"] for e in doc["registered"]}
    schemas = {t.name: t.schema for t in SW.tensors(SHAPE)}
    want = REG.registered_entries(prog, words, schemas)
    for e in doc["registered"]:
        w = want[e["value"]]
        assert (e["leaves"], e["words"], e.get(REG.WIDTH_KEY, REG.WORD_BITS), e["domain"]) == \
            (w["leaves"], w["words"], w["word_bits"], w["domain"]), e["value"]


def test_the_state_and_witness_are_private_and_the_witness_recommits_the_same_roots(committed, digest, tmp_path):
    state, doc, _s = committed
    assert stat.S_IMODE(os.stat(state).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(state / "salts.bin").st_mode) == 0o600
    assert os.path.getsize(state / "salts.bin") == doc["rows"] * WC.SALT
    again = WC.commit(SHAPE, SEED, digest, tmp_path / "again", witness=str(state / "salts.bin"), salts="witness",
                      chunk_bytes=4 * CHUNK)
    assert again["registered"] == doc["registered"] and again["salts"] == "witness"
    with pytest.raises(FileExistsError):
        WC.commit(SHAPE, SEED, digest, state, chunk_bytes=CHUNK)


def _program_reads(prog):
    """Every ScaledMm proof block's registered reads, from the Program's references: ``{(op, n, port): (param, bits)}``,
    `bits` the parameter leaves (one-bit leaves) unit n's port reads."""
    names = [n for n, _ in prog.fn.signature()[0]]
    out = {}
    for node in prog.fn.body.nodes:
        if not node.fn.id.startswith("ScaledMmS8CoordinateRows_v2"):
            continue
        ports = [n for n, _ in node.fn.signature()[0]]
        for port, arg, ax in zip(ports, node.args, node.axes):
            if not (port.startswith("wrow") or port == "sw"):
                continue
            assert ax == 0, (node.name, port)
            per = len(arg.refs) // node.n
            for n in range(node.n):
                refs = [arg.refs[j] for j in range(n * per, (n + 1) * per)]
                assert {(s, i) for s, i, _ in refs} == {("p", refs[0][1])}, (node.name, port)
                out[node.name, n, port] = names[refs[0][1]], [k for _, _, k in refs]
    return out


def test_every_drawn_scaled_mm_blocks_wrow_and_sw_reads_open_against_the_registration(committed, prog, monkeypatch):
    _state, doc, s = committed
    truth = _program_reads(prog)
    ops = sorted({op for op, _n, _p in truth})
    assert ops == sorted(t.name for t in SW.tensors(SHAPE) if t.kind == "s8")
    for (op, n, port), (param, bits) in truth.items():
        value, pos = WC.reads(SHAPE, op, n, port)
        row_bits = 16 * SW.tensor(SHAPE, value).words
        assert value == param and bits == list(range(pos * row_bits, (pos + 1) * row_bits)), (op, n, port)
    for op in ops:
        assert {p for o, _n, p in truth if o == op} == set(WC.scaled_mm_ports(SHAPE, op)), op
    monkeypatch.setattr(REG, "parameter_gates", _no_gates)
    for op in ops:
        for n in (0, SW.tensor(SHAPE, op).rows // 2 + 1, SW.tensor(SHAPE, op).rows - 1):
            got = WC.open_block(s, op, n)
            assert set(got) == set(WC.scaled_mm_ports(SHAPE, op))
            assert all(v["ok"] for v in got.values()), (op, n, {p: v["ok"] for p, v in got.items()})
    assert s.value("l0.wup").program == doc["program"]


def test_a_read_is_refused_with_a_wrong_path_position_or_salt(committed):
    _state, _doc, s = committed
    v = s.value("l0.wdown")
    e = v.entry()
    root, pos = bytes.fromhex(e["root"]), 37
    bc, path = v.bcs[pos], v.path(pos)
    assert REG.verify_read(v.domain, root, pos, bc, path, v.schema)
    flipped = bytearray(path)
    flipped[-1] ^= 1
    assert not REG.verify_read(v.domain, root, pos, bc, bytes(flipped), v.schema)
    assert not REG.verify_read(v.domain, root, pos + 1, bc, path, v.schema)
    other = REG._INST.commit_string(REG._INST.default_key, REG.row_digest(v.row(pos), v.word_bits, v.schema), os.urandom(WC.SALT))
    assert not REG.verify_read(v.domain, root, pos, other, path, v.schema)
    assert not REG.verify_read(s.value("l0.wq").domain, root, pos, bc, path, v.schema)


def test_the_cli_commits_and_opens_a_proof_block(tmp_path, digest, capsys):
    state = tmp_path / "state"
    assert WC.main(["commit", "--shape", SHAPE.name, "--program", digest, "--state", str(state), "--seed", "test",
                    "--workers", "1", "--out", str(tmp_path / "entries.json")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["program"] == digest and summary["rows"] == sum(t.leaves for t in SW.tensors(SHAPE))
    assert WC.main(["open", "--state", str(state), "--seed", "test", "--op", "l0.wup", "--coordinate", "5"]) == 0
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["verdict"] == "OPEN" and verdict["reads"]["sw"] == {"value": "l0.wup.scale", "position": 5, "ok": True}
    with pytest.raises(ValueError, match="another seed"):
        WC.open_state(state, SW.DEMO_SEED)


def test_a_shape_whose_wrow_part_is_no_registered_row_is_refused():
    with pytest.raises(ValueError, match="no one registered row"):
        WC.reads(MINI, "l0.wq", 0, "wrow0")
    assert WC.scaled_mm_ports(SW.H200, "l0.wup") == [f"wrow{r}" for r in range(2)] + ["sw"]
