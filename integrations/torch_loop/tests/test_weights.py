"""The seeded weights and their registration root (`verity_torch_loop.weights`, CMP-95 interface 1)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from verity_torch_loop import weights as W

FIXTURE = Path(__file__).parent / "fixtures" / "weights-v1.json"
SEED, KEY = W.TEST_SEED, W.TEST_SALT_KEY


def test_fixture_is_the_generators():
    """The committed fixture is what the generator and the registration give now: a change to either moves it."""
    assert json.loads(W.dumps(W.fixture())) == json.loads(FIXTURE.read_text())


def test_row_stream_is_sha512_counter():
    key = hashlib.sha512(b"verity/seeded-weights/v1\0" + SEED + b"\0l0.wq").digest()
    want = b"".join(hashlib.sha512(key + (5).to_bytes(8, "little") + i.to_bytes(8, "little")).digest() for i in range(3))
    assert W.row_bytes(SEED, "l0.wq", 5, 150) == want[:150]
    codes = W.generate(W.H200, SEED, "l0.wq", [5])[0]
    assert codes.dtype == np.int8 and codes[:150].tobytes() == want[:150]


def test_element_ranges():
    for t in W.tensors(W.PHASE1):
        x = W.generate(W.PHASE1, SEED, t.name)
        if t.kind == "scale":
            assert x.dtype == np.float32 and (x >= 2.0 ** t.exponent).all() and (x < 2.0 ** (t.exponent + 1)).all()
        elif t.kind == "gain":
            f = (x.astype(np.uint32) << 16).view(np.float32)
            assert ((f >= 1) & (f < 2)).all()
        elif t.kind == "embed":
            f = np.abs((x.astype(np.uint32) << 16).view(np.float32))
            assert ((f >= 0.5) & (f < 1)).all()
    assert W.tensor(W.H200, "l0.wq.scale").exponent == W.scale_exponent(16384) - 3 == -17
    assert W.tensor(W.H200, "l0.wdown.scale").exponent == -15


def test_h200_sizes():
    ts = W.tensors(W.H200)
    layer_s8 = sum(t.nbytes for t in ts if t.kind == "s8" and t.name.startswith("l"))
    assert layer_s8 == 139_695_489_024 and f"{layer_s8 / 2**30:.2f}" == "130.10"
    assert all(t.leaves & (t.leaves - 1) == 0 for t in ts), "a tree with pads"
    assert [t.offset for t in ts] == [sum(u.nbytes for u in ts[:i]) for i in range(len(ts))]
    assert W.tensor(W.H200, "l0.wup").words == 128 and W.tensor(W.H200, "l0.wup").leaves == 65536 * 64


def test_stream_and_write(tmp_path):
    whole = W.stream(W.TINY, SEED, 0, W.stream_bytes(W.TINY))
    assert whole == b"".join(W.tensor_bytes(W.TINY, SEED, t.name) for t in W.tensors(W.TINY))
    assert W.stream(W.TINY, SEED, 1000, 9000) == whole[1000:9000]
    out = tmp_path / "w.bin"
    W.write(W.TINY, SEED, out, workers=2, chunk_bytes=4096)
    assert out.read_bytes() == whole
    m = W.manifest(W.TINY, SEED, digests=True, workers=2)
    assert m == W.manifest(W.TINY, SEED, digests=True)
    assert [t["sha256"] for t in m["tensors"]] == [hashlib.sha256(W.tensor_bytes(W.TINY, SEED, t.name)).hexdigest()
                                                   for t in W.tensors(W.TINY)]


def test_root_is_core_commit():
    """Each tensor's root and entry are core's ``Registered.commit`` of its rows under the same salts, in one chunk or many,
    in one process or two."""
    roots = W.root(W.TINY, SEED, KEY)
    for t in W.tensors(W.TINY):
        c = W.commit(W.TINY, SEED, KEY, t.name)
        assert c.root == roots[t.name] and c.entry() == W.entry(W.TINY, SEED, t.name, roots[t.name])
    assert W.root(W.TINY, SEED, KEY, chunk_bytes=64, workers=2) == roots
    assert W.root(W.TINY, SEED, KEY, ["l0.wq", "head.scale"]) == {k: roots[k] for k in ("l0.wq", "head.scale")}


def test_registration_binds_seed_and_key():
    roots = W.root(W.TINY, SEED, KEY)
    d = W.registration_digest(W.TINY, SEED, roots)
    assert W.registration_digest(W.TINY, SEED, W.root(W.TINY, SEED, bytes(32))) != d
    other = b"another seed"
    assert W.registration_digest(W.TINY, other, W.root(W.TINY, other, KEY)) != d


def test_refusals():
    with pytest.raises(ValueError, match="salt key is 32 bytes"):
        W.root(W.TINY, SEED, b"short")
    with pytest.raises(ValueError, match="has no tensor"):
        W.generate(W.TINY, SEED, "l9.wq")
    with pytest.raises(ValueError, match="has rows"):
        W.generate(W.TINY, SEED, "l0.wq", [W.tensor(W.TINY, "l0.wq").rows])
    with pytest.raises(ValueError, match="multiple of row_codes"):
        W.tensors(W.Shape("bad", 48, 1, 1, 16, 1, 64, 256, 8, 32))
