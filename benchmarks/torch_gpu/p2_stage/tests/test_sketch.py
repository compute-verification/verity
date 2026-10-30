"""The bandwidth audit's receipts beside the staging decode (`sketch.py`) on the CPU's host twins against serve.py's
fixture: each block's π, σ however the decode's chunks fall, y and the chained roots; a pass refuses a unit it didn't
read and rows that don't hold whole units; `full.py --receipts` over the toy's encoded store gives the fixture's root,
and `bw_ref.py` passes the kept pass and refuses one changed."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "pous"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout as L  # noqa: E402
import sketch as K  # noqa: E402
import stage as S  # noqa: E402
import test_p2_stage as T  # noqa: E402
from p2_v1 import serve  # noqa: E402


@pytest.fixture(scope="module")
def fx() -> dict:
    data = serve.fixture_bytes()
    assert hashlib.sha512(data).hexdigest() == serve.FIXTURE_SHA512
    return json.loads(data)


def toy_rows() -> torch.Tensor:
    return torch.frombuffer(bytearray(serve.toy()[2]), dtype=torch.uint8).view(-1, L.ROW)


def salts_of(p: dict) -> bytes:
    return b"".join(bytes.fromhex(s) for s in p["leaf_salts"])


def test_bw_check_holds_the_host_twins_to_the_fixture(tmp_path):
    import bw_check

    assert bw_check.main(["--out", str(tmp_path), "--device", "cpu", "--random", "300", "--sample", "40"]) == 0
    res = json.loads((tmp_path / "bw_check.json").read_text())
    passes = res["fixture"]["passes"]
    assert res["fixture"]["pinned"] and [p["pass"] for p in passes] == [0, 1]
    assert all(p["seed"] and p["pi"] == {"match": 32, "of": 32, "first_bad": []} and p["root"] for p in passes)
    assert all(s["match"] == 8 for p in passes for s in p["sigma"].values()) and set(passes[0]["sigma"]) == {"1", "3", "5", "33"}
    assert all(p["y"]["match"] == 8 for p in passes)
    assert res["random"]["ok"] and res["random"]["sigma_sampled"] == 42


def test_a_pass_covers_the_units_the_tensors_lie_in():
    def t(a: int, b: int) -> SimpleNamespace:
        return SimpleNamespace(first_block=a, block_range=range(a, b))

    assert K.plan_units([t(6, 11), t(0, 6)]) == 3 and K.plan_units([t(0, 8)]) == 2
    with pytest.raises(ValueError, match=r"units 2 \.\. 2 hold no tensor's block"):
        K.plan_units([t(0, 8), t(12, 13)])


def test_a_pass_refuses_what_it_did_not_read_and_moves_with_one_byte(fx):
    rows, units, p = toy_rows(), fx["store"]["units"], fx["passes"][0]
    seed = bytes.fromhex(p["seed"])
    tree = K.Tree(units, "cpu")
    salts = tree.salts(salts_of(p))
    P = K.Pass(seed, units, "cpu")
    P.read(rows, 0, 0, 28)
    with pytest.raises(ValueError, match="1 of 8 units weren't read"):
        P.finish(tree, salts)
    with pytest.raises(ValueError, match="whole"):
        P.read(rows[2:], 2, 2, 9)
    P.read(rows, 0, 29, 30)
    P.read(rows, 0, 32, 33)
    assert P.finish(tree, salts).hex() == p["root"] and P.start_ns <= P.end_ns
    bad = rows.clone()
    bad[13, 100] ^= 1
    Q = K.Pass(seed, units, "cpu")
    Q.read(bad, 0, 0, 33)
    assert (Q.sigma != P.sigma).any(dim=1).nonzero().flatten().tolist() == [3]
    assert Q.finish(tree, salts) != P.root
    other = K.Pass(bytes.fromhex(fx["passes"][1]["seed"]), units, "cpu")
    other.read(rows, 0, 0, 33)
    assert other.finish(tree, salts) != P.root


def test_the_decode_sketches_each_chunk_before_decoding_it(fx, monkeypatch):
    """A store with a pass attached sketches the units of every chunk `decode_rows` reads, in chunks that split
    units, before the decode; the toy's layers read units 0 .. 7."""
    monkeypatch.setattr(S.gpu, "Kernels", T.RefKernels)
    doc = serve.toy()[1]
    lay = L.store_layout(doc, L.registered_prefix(bytes.fromhex(doc["salt"])))
    p = fx["passes"][1]
    P = K.Pass(bytes.fromhex(p["seed"]), fx["store"]["units"], "cpu")
    st = S.Store(lay, toy_rows(), 0, "unused", chunk=3)
    st.sketch = P
    for layer, _, size, _ in doc["staging"]:
        S.decode_layer(st, layer, torch.empty(size, dtype=torch.uint8))
    assert bool(P.seen.all())
    assert [bytes(r.numpy().astype("<i4").tobytes()).hex() for r in P.sigma] == p["sigmas"]


def test_full_receipts_on_the_toy_store_give_the_fixture_root_and_bw_ref_checks_them(tmp_path, monkeypatch, fx):
    """full.py --receipts over layer 1 alone, of the toy's store with a segment of padding past the head: layer 0's
    units are read from their segments and sketched alone, the padding is in no receipt, so the pass covers the
    tensors' 8 units of the store's 11 and its root is the fixture's; bw_ref.py on the kept pass passes, and refuses a
    pass over the store's units, a y changed or a σ changed."""
    import bw_ref
    import full

    doc = json.loads((T.TOY / "toy.json").read_text())
    manifest, stream = T.toy_stream(doc)
    bps = doc["layout"]["params"]["blocks_per_segment"]
    padded = {**doc["layout"], "params": {**doc["layout"]["params"], "segments": 4}, "blocks": 4 * bps}
    (tmp_path / "layout.json").write_text(json.dumps(padded))
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    (tmp_path / "w.bin").write_bytes(stream)
    setup = {"commitment": doc["commitment"], "manifest": padded["params"],
             "layout_sha512": hashlib.sha512((tmp_path / "layout.json").read_bytes()).hexdigest()}
    root = tmp_path / "store"
    c_star = np.frombuffer((T.TOY / "c-star.bin").read_bytes(), dtype=np.uint8).reshape(-1, L.ROW)
    pad = np.random.default_rng(7).integers(0, 256, (4 * bps - len(c_star), L.ROW), dtype=np.uint8)
    T.encoded_store(root, np.concatenate([c_star, pad]), bps, setup)
    monkeypatch.setattr(S.gpu, "Kernels", T.RefKernels)
    p = fx["passes"][0]
    (tmp_path / "salts.bin").write_bytes(salts_of(p))
    store = ["--layout", str(tmp_path / "layout.json"), "--weights", str(tmp_path / "m.json"), "--encoded", str(root),
             "--source", doc["source"], "--commitment", doc["commitment"]]
    keep, out = tmp_path / "keep", tmp_path / "out"
    assert full.main(store + ["--w", str(tmp_path / "w.bin"), "--lib", "unused", "--device", "cpu", "--reps", "1",
                              "--readers", "2", "--hashers", "2", "--out", str(out), "--layers", "1", "--receipts",
                              "--challenge", fx["challenge"], "--previous", fx["registration"]["digest"],
                              "--salts", str(tmp_path / "salts.bin"), "--keep", str(keep)]) == 0
    recs = [json.loads(x) for x in (out / "full.json").read_text().splitlines()]
    staged = [r for r in recs if r["step"] == "stage"]
    assert {r["layer"] for r in staged} == {1} and all("decode_sketch_s" in r and "sketch_s" in r for r in staged)
    rc = next(r for r in recs if r["step"] == "summary")["receipts"]
    assert (rc["seed"], rc["root"], rc["units"], rc["leaves"]) == (p["seed"], p["root"], 8, 1)
    assert (rc["store_units"], rc["padding_units_left_out"], rc["blocks_past_head_in_plan"]) == (11, 3, 0)
    lay = L.store_layout(padded, T.PREFIX)
    held = range(lay.layer(1)[0].first_block // K.PER, -(-lay.layer(1)[-1].block_range.stop // K.PER))
    assert rc["gap_units"] == 8 - len(held) == 5
    assert rc["receipts_step_s"] == pytest.approx(rc["decode_sketch_step_s"] + rc["gap_sketch_s"] + rc["finish_s"], abs=1e-5)
    assert rc["start_ns"] <= rc["end_ns"] and rc["overhead_s"] == pytest.approx(rc["receipts_step_s"] - rc["decode_step_s"],
                                                                                abs=1e-5)
    ref = store + ["--keep", str(keep), "--sample", "3", "--workers", "2"]
    assert bw_ref.main(ref + ["--out", str(tmp_path / "r1")]) == 0
    res = json.loads((tmp_path / "r1" / "bw_ref.json").read_text())
    assert res["root_cpu"]["match"] and res["y"]["mismatches"] == 0 and res["sigma"] == {**res["sigma"], "units": 8,
                                                                                          "mismatches": 0}
    rec = json.loads((keep / "pass.json").read_text())
    (keep / "pass.json").write_text(json.dumps({**rec, "units": 11}))
    with pytest.raises(SystemExit, match="covers 11 units, not the 8 the store's tensors lie in"):
        bw_ref.main(ref + ["--out", str(tmp_path / "r0")])
    y = bytearray((keep / "y.bin").read_bytes())
    y[64 * 5] ^= 1
    (keep / "y.bin").write_bytes(bytes(y))
    (keep / "pass.json").write_text(json.dumps({**rec, "y_sha512": hashlib.sha512(bytes(y)).hexdigest()}))
    assert bw_ref.main(ref + ["--out", str(tmp_path / "r2")]) == 1
    assert json.loads((tmp_path / "r2" / "bw_ref.json").read_text())["y"]["first"] == [5]
    y[64 * 5] ^= 1
    (keep / "y.bin").write_bytes(bytes(y))
    sigma = bytearray((keep / "sigma.bin").read_bytes())
    sigma[64 * 2] ^= 1
    (keep / "sigma.bin").write_bytes(bytes(sigma))
    (keep / "pass.json").write_text(json.dumps({**rec, "sigma_sha512": hashlib.sha512(bytes(sigma)).hexdigest()}))
    assert bw_ref.main(ref + ["--out", str(tmp_path / "r3")]) == 1
    res = json.loads((tmp_path / "r3" / "bw_ref.json").read_text())
    assert res["sigma"]["first"] == [2] and res["y"]["first"] == [2] and not res["root_cpu"]["match"]
