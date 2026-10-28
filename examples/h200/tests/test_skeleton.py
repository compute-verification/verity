"""The skeleton at the tiny shape and at MINI (every level of circuits' FAMILIES map) on the CPU: every unit of a step
replays bit for bit, the driver's rows equal the replay lane's independent `cpu_step`, a tampered table doesn't replay, the commit backends agree, the store's layout stages
every tensor, and `run.py` writes its record and bundle."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import torch

import cpu_step as CS
import run as RUN
from e2e_h200 import store as STO
from e2e_h200.commit import Committer, fixture_salt, fixture_secrets
from e2e_h200.driver import Driver
from e2e_h200.legs import pous_spot_reads, wipe_reload
from e2e_h200.shape import MINI, SMALL, TINY
from e2e_h200.store import GeneratorStore, P2Store
from e2e_h200.units import Replayer
from e2e_h200.weights import Weights

BATCH, STEPS = 2, 4
PROMPTS = np.array([list(b"hi"), list(b"ok")])
TAMPER = "l0.wup"


#: each shape and the `cpu_step` forms of its map: RMSNorm's chunk and the chunks a group sums
FORMS = {TINY.name: CS.Forms(rms_chunk=TINY.width, rms_group=1, amax_chunk=16, part_fanin=2, pv_k=32),
         MINI.name: CS.Forms(rms_chunk=32, rms_group=2, amax_chunk=16, part_fanin=2, pv_k=32)}


def drive(store, backend="reference", tamper=None, device="cpu", shape=TINY):
    d = Driver(shape, store, BATCH, Committer(backend, *fixture_secrets("test")), device, clock=lambda: 0.0, tamper=tamper)
    return d, d.run(PROMPTS, STEPS)


@pytest.fixture(scope="module")
def weights():
    return Weights(TINY)


@pytest.fixture(scope="module")
def honest(weights):
    return drive(GeneratorStore(weights))


@pytest.fixture(scope="module", params=[TINY, MINI], ids=lambda s: s.name)
def shaped(request):
    """(shape, weights, driver, results) at the tiny shape and at MINI."""
    w = Weights(request.param)
    return (request.param, w) + drive(GeneratorStore(w), shape=request.param)


def replayer(weights, d, results, shape=TINY):
    return Replayer(shape, weights, d.steps, [r.tokens_in for r in results])


def test_tok_is_one_coordinate_a_row_and_the_driver_reads_token_rows(weights, honest):
    tok = weights.array("tok")
    assert tok.shape == (TINY.width, TINY.vocab)
    v = int(PROMPTS[0, 0])
    assert np.array_equal(honest[0].tok[v].view(torch.int16).numpy().view(np.uint16), tok[:, v])


def test_every_unit_of_every_step_replays(shaped):
    shape, weights, d, results = shaped
    rp = replayer(weights, d, results, shape)
    units = [u for s in range(STEPS) for u in rp.every(s)]
    assert len({u.template.split("{")[0] for u in units}) == 11
    assert [u.doc() for u in units if not u.ok][:3] == []


def test_a_draw_covers_every_family(shaped):
    shape, weights, d, results = shaped
    rp = replayer(weights, d, results, shape)
    units = rp.draw(STEPS - 1, 2, b"test")
    assert {u.family for u in units} == {f.name for f in rp.families(STEPS - 1)} and all(u.ok for u in units)


def test_driver_rows_equal_cpu_step(shaped):
    """The replay lane's `cpu_step` (16-bit unit forms, written apart from the driver) at the shape's map's RMSNorm chunks
    and groups: the same bytes out, and every residual row and logit row equal."""
    from e2e_h200 import served_weights as SW
    shape, _, d, results = shaped
    st = CS.Step(shape, lambda n: SW.generate(shape, SW.TEST_SEED, n), batch=BATCH, forms=FORMS[shape.name])
    out = CS.decode(st, [b"hi", b"ok"], STEPS - 1)
    assert out == [bytes(PROMPTS[b].tolist()) + bytes(int(r.tokens_out[b]) for r in results[1:]) for b in range(BATCH)]
    rows = 0
    for u in st.rec.t.values():
        for (_, lab), o in zip(u.labels, u.outputs):
            if lab["op"] in ("head.epilogue", "mlp_residual"):
                tab = "head" if lab["layer"] is None else f"l{lab['layer']}.mlp_residual"
                assert np.array_equal(d.steps[lab["pos"]].tables[tab][0].view(np.uint16), np.asarray(o).reshape(BATCH, -1))
                rows += 1
    assert rows == 2 * STEPS


def test_tampered_table_fails_replay(weights, honest):
    d, results = drive(GeneratorStore(weights), tamper=TAMPER)
    assert results[0].root != honest[1][0].root
    units = replayer(weights, d, results).draw(0, 4, b"test", only=lambda f: f.name == TAMPER)
    assert units and not any(u.ok for u in units)


def test_host_committer_agrees_with_reference(honest):
    if not any(shutil.which(c) for c in ("g++", "c++", "clang++")):
        pytest.skip("no C++ compiler on the path for row_table_host.cpp")
    d, results = honest
    for st, r in zip(d.steps, results):
        assert Committer("host", *fixture_secrets("test")).commit(st).root == r.root


def identity_store(weights, monkeypatch):
    """A P2Store whose rows hold the payload in the clear and whose decode is the identity: the layout without P2's cost."""
    wb = weights.nbytes
    rows = np.zeros((STO.nblocks(wb), STO.ROW), np.uint8)
    for j in range(rows.shape[0]):
        lo = j * STO.PAYLOAD
        rows[j, 2:] = np.frombuffer(weights.stream(lo, min(lo + STO.PAYLOAD, wb)).ljust(STO.PAYLOAD, b"\0"), np.uint8)
    monkeypatch.setattr(STO, "decode_block", lambda prefix, j, row: bytes(row[2:]))
    return P2Store(weights, bytes(24), rows)


def test_store_layout_stages_and_loads_every_tensor(weights, monkeypatch):
    s = identity_store(weights, monkeypatch)
    for name, t in weights.t.items():
        if t.kind == "s8":
            dst = torch.empty(weights.dims[name], dtype=torch.int8)
            assert torch.equal(s.stage(name, dst), torch.from_numpy(weights.codes(name))), name
        assert np.array_equal(s.load(name).reshape(-1).view(np.uint8), weights.array(name).reshape(-1).view(np.uint8)), name
    assert [STO.blocks_of(STO.PAYLOAD - 1, 2).start, STO.blocks_of(STO.PAYLOAD - 1, 2).stop] == [0, 2]
    assert list(STO.blocks_of(STO.PAYLOAD, STO.PAYLOAD)) == [1]


def test_cpu_s32_sum_is_exact_across_chunks(monkeypatch):
    from e2e_h200 import ops as O
    monkeypatch.setattr(O, "F64_CHUNK", 3 * 4096)
    g = torch.Generator().manual_seed(7)
    x = torch.randint(-128, 128, (3, 4096), dtype=torch.int8, generator=g)
    w = torch.randint(-128, 128, (11, 4096), dtype=torch.int8, generator=g)
    x[0], w[0] = -128, -128
    want = (x.to(torch.int64) @ w.to(torch.int64).t()).to(torch.int32)
    assert torch.equal(O.s32_mm(x, w), want) and int(want[0, 0]) == 4096 * 2 ** 14


def test_parallel_generator_stages_the_serial_codes():
    w = Weights(SMALL)
    s = GeneratorStore(w, workers=2)
    try:
        for name in ("l0.wup", "l1.wdown", "head"):
            dst = torch.empty(w.dims[name], dtype=torch.int8)
            assert torch.equal(s.stage(name, dst), torch.from_numpy(w.codes(name))), name
            assert np.array_equal(s.stage(name), w.codes(name)), name
    finally:
        s.close()


def test_wipe_reload_recomputes_the_root(weights, honest, monkeypatch):
    s = identity_store(weights, monkeypatch)
    monkeypatch.setattr("e2e_h200.legs.decode_block", STO.decode_block)
    _, results = drive(s)
    assert [r.root for r in results] == [r.root for r in honest[1]]
    leg = wipe_reload(s, lambda: drive(s)[1][-1].root, results[-1].root)
    assert leg["store_digest_equal"] and leg["step_root_equal"]
    assert pous_spot_reads(s, 8, b"test")["mismatched_blocks"] == []


def test_decode_refuses_c_at_least_p():
    with pytest.raises(STO.Refused):
        STO.decode_block(STO.S.expand_prefix(STO.PARAMS, bytes(24)), 0, b"\xff" * STO.ROW)


@pytest.mark.slow
def test_p2_block_round_trips():
    prefix = STO.S.expand_prefix(STO.PARAMS, bytes(range(24)))
    payload = bytes((7 * i) & 0xFF for i in range(STO.PAYLOAD))
    row = STO.encode_block(prefix, 70_000, payload)
    assert len(row) == STO.ROW and STO.decode_block(prefix, 70_000, row) == payload


def test_run_writes_record_and_bundle(tmp_path):
    rc = RUN.main(["--out", str(tmp_path), "--shape", "tiny", "--batch", "2", "--steps", "2", "--draw", "1"])
    assert rc == 0
    rec = json.loads((tmp_path / "record.json").read_text())
    assert rec["replay"]["mismatches"] == 0 and len(rec["step_roots"]) == 2
    assert rec["commit"]["backend"] == "slots" and rec["replay"]["opening"]["mismatches"] == 0
    assert rec["replay"]["opening"]["units_opened"] == rec["replay"]["drawn"] > 0
    assert rec["weights"]["registration_digest_test_key"] and rec["registration"]["blocks"][0]["name"] == "model"
    assert set(rec["legs"]) == {"integrity", "pous", "bandwidth", "certifier", "wipe", "pouw"}
    assert all(u["opened"]["verified"] for u in json.loads((tmp_path / "bundle.json").read_text())["units"])


def test_run_replays_kept_tables_under_a_by_table_commit(tmp_path):
    rc = RUN.main(["--out", str(tmp_path), "--shape", "tiny", "--batch", "2", "--steps", "2", "--commit", "reference",
                   "--draw", "1"])
    rec = json.loads((tmp_path / "record.json").read_text())
    assert rc == 0 and rec["replay"]["opening"] is None and rec["replay"]["drawn"] > 0


@pytest.mark.parametrize("commit", ["slots", "reference"])
def test_run_exits_3_on_a_tampered_table(tmp_path, commit):
    rc = RUN.main(["--out", str(tmp_path), "--shape", "tiny", "--batch", "2", "--steps", "1", "--commit", commit,
                   "--draw", "1", "--tamper", TAMPER])
    assert rc == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_step_commits_the_cpu_roots(weights, honest):
    _, results = drive(GeneratorStore(weights), device="cuda")
    assert [r.root for r in results] == [r.root for r in honest[1]]
    roots = []
    for dev in ("cpu", "cuda"):
        c = Committer("slots", *fixture_secrets("test"), TINY, dev, fixture_salt("test"))
        roots.append([r.root for r in Driver(TINY, GeneratorStore(weights), BATCH, c, dev, keep=False).run(PROMPTS, STEPS)])
    assert roots[0] == roots[1]
