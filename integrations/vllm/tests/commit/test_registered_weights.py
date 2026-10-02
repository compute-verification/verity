"""The hiding weights root (`commit.registered_weights`, ``registered-hm96-sha512/v1``): every registered tensor committed as proofs'
registered values are (`verity_sampled_proofs.one_stage.registered`, restated here from core's primitives), the root binding, a guess
of the weights not confirmed without the registration key, the key off the record and in the replay bundle, and the CPU replay
opening the checkpoint's tensors against the root with it."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity.commitments import hm96
from verity.commitments.merkle import tree_levels
from verity.commitments.rowleaf import ROLE_X, row_leaf, sha512_row_digest
from verity_vllm.commit import hiding, registered_weights as RW, scheme  # noqa: N812

PROGRAM = "ab" * 32
GEO = bytes(range(32))
KEY = bytes(range(100, 132))


def _proofs_root(program: str, value: str, rows: list[list[int]], salts: list[bytes]) -> tuple[bytes, dict]:
    """`Registered.commit(program, value, rows, salts)`'s root and entry, one row at a time through core's reference."""
    inst = hm96.SHA512
    bcs = [inst.commit_string(inst.default_key, sha512_row_digest(r, 16, ROLE_X), s) for r, s in zip(rows, salts)]
    dom = RW.domain(program, value, len(rows[0]), len(rows))
    root = tree_levels(dom, [row_leaf(dom, i, inst.tree_leaf(inst.default_key, bc), RW.SCHEMA) for i, bc in enumerate(bcs)])[-1][0]
    return root, {"value": value, "root": root.hex(), "leaves": len(rows), "words": len(rows[0]), "domain": dom.domain_id.hex()}


def _words(raw: bytes, width: int) -> list[list[int]]:
    rows = [raw[o:o + width] for o in range(0, len(raw), width)]
    return [list(np.frombuffer(r + b"\0" * (len(r) % 2), dtype="<u2")) for r in rows]


@pytest.mark.parametrize("shape,dtype", [((3, 8), np.uint16), ((5,), np.float32), ((), np.float32), ((2, 3), np.uint8)])
def test_a_tensors_root_is_proofs_registered_value_over_its_rows(shape, dtype):
    a = np.random.default_rng(3).integers(0, 200, shape).astype(dtype)
    raw = a.tobytes()
    rows, width = RW.rows_of(shape, len(raw))
    assert rows == (int(np.prod(shape[:-1])) if len(shape) > 1 else 1) and rows * width == len(raw)
    salts = RW.salts(KEY, 4, rows)
    root, entry = RW.value_root(PROGRAM, "model.w", 4, raw, shape, KEY)
    want = _proofs_root(PROGRAM, "model.w", _words(raw, width), [salts[192 * r:192 * (r + 1)] for r in range(rows)])
    assert (root, entry) == want
    assert len(set(salts[192 * r:192 * (r + 1)] for r in range(rows))) == rows and RW.salts(KEY, 5, rows) != salts


def test_the_root_binds_every_byte_the_key_the_value_and_the_program():
    raw = bytes(range(64))
    root = RW.value_root(PROGRAM, "w", 0, raw, (4, 16), KEY)[0]
    assert RW.value_root(PROGRAM, "w", 0, raw, (4, 16), KEY)[0] == root
    flipped = bytearray(raw)
    flipped[37] ^= 1
    for other in (RW.value_root(PROGRAM, "w", 0, bytes(flipped), (4, 16), KEY), RW.value_root(PROGRAM, "w", 0, raw, (4, 16), bytes(32)),
                  RW.value_root(PROGRAM, "v", 0, raw, (4, 16), KEY), RW.value_root("cd" * 32, "w", 0, raw, (4, 16), KEY),
                  RW.value_root(PROGRAM, "w", 1, raw, (4, 16), KEY), RW.value_root(PROGRAM, "w", 0, raw, (8, 8), KEY)):
        assert other[0] != root


def test_a_guess_of_the_weights_is_not_confirmed_without_the_key():
    """The committer's value is one of two candidates; with the root alone (and any key but the committer's) both look alike."""
    a, b = bytes(32), bytes([1]) + bytes(31)
    root = RW.value_root(PROGRAM, "w", 0, a, (2, 16), KEY)[0]
    for guess_key in (bytes(32), b"\xff" * 32):
        assert root not in {RW.value_root(PROGRAM, "w", 0, g, (2, 16), guess_key)[0] for g in (a, b)}
    assert RW.value_root(PROGRAM, "w", 0, a, (2, 16), KEY)[0] == root != RW.value_root(PROGRAM, "w", 0, b, (2, 16), KEY)[0]


def _model():
    torch = pytest.importorskip("torch")
    m = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.LayerNorm(4))
    m[0].weight.data = m[0].weight.data.to(torch.bfloat16)
    m.register_buffer("cos_sin_cache", torch.arange(6, dtype=torch.float32))
    m.register_buffer("empty", torch.zeros(0))
    return m


def _committer(model, leaf_scheme=hiding.DEFAULT_LEAF_SCHEME, key=None):
    from verity_vllm.commit.committer.native_host import NativeHostCommitter

    com = NativeHostCommitter(None, run_id="rw", chunk=16, hash_threads=1, geo_digest=GEO, leaf_scheme=leaf_scheme)
    com.model = model
    if key is not None:
        com._registration_key = key
    return com


def test_the_hiding_committer_registers_every_tensor_once_under_the_registered_root(monkeypatch):
    monkeypatch.delenv("VERITY_WEIGHTS_HASH", raising=False)
    model = _model()
    com = _committer(model)
    w = com.register_weights()
    t = com.weights_tree
    names = [n for n, *_ in t["names"]]
    assert names == ["0.weight", "0.bias", "1.weight", "1.bias", "cos_sin_cache"]
    assert w["scheme"] == t["scheme"] == RW.NAME and w["root"] == t["root"] == RW.tree_root(t) and w["tensors"] == 5
    assert [e["value"] for e in t["registered"]] == names and [e["root"] for e in t["registered"]] == t["tensor_roots"]
    params = dict(list(model.named_parameters()) + list(model.named_buffers()))
    for i, (name, dt, shape, nb) in enumerate(t["names"]):
        raw = params[name].detach().contiguous().reshape(-1).view(__import__("torch").uint8).numpy()
        assert RW.tensor_root(t, i, raw, com._registration_key).hex() == t["tensor_roots"][i] and nb == raw.size
    assert com.register_weights()["root"] == w["root"], "a re-registration of the same weights is the same root"
    assert _committer(model).register_weights()["root"] != w["root"], "each committer draws its own key"
    assert _committer(model, key=com._registration_key).register_weights()["root"] == w["root"]
    key_hex = com._registration_key.hex()
    assert key_hex not in json.dumps(w) and key_hex not in json.dumps(t), "the key is never on the record"
    with __import__("torch").no_grad():
        model[1].bias[0] += 1
    assert com.register_weights()["root"] != w["root"]


def test_vllm_v1_keeps_its_sha256_weights_root(monkeypatch):
    monkeypatch.delenv("VERITY_WEIGHTS_HASH", raising=False)
    model = _model()
    com = _committer(model, scheme.NAME)
    w, t = com.register_weights(), com.weights_tree
    assert "scheme" not in t and t["chunk"] == 16 and not hasattr(com, "_registration_key")
    params = dict(list(model.named_parameters()) + list(model.named_buffers()))
    roots = [RW.tensor_root(t, i, params[n].detach().contiguous().reshape(-1).view(__import__("torch").uint8).numpy(), None)
             for i, (n, *_r) in enumerate(t["names"])]
    assert [r.hex() for r in roots] == t["tensor_roots"]
    assert w["root"] == RW.tree_root(t) == scheme.weights_root(GEO, scheme.names_digest(t["names"]), roots).hex()


def test_the_registered_root_is_hashed_on_the_host_only(monkeypatch):
    pytest.importorskip("torch")
    from verity_vllm.commit.committer import leafhash

    monkeypatch.setattr(leafhash, "ext", lambda: None)
    monkeypatch.setenv("VERITY_WEIGHTS_HASH", "device")
    with pytest.raises(ValueError, match="hashed on the host only"):
        _committer(_model()).register_weights()


def test_a_registered_tree_opens_only_with_its_key():
    raw = np.arange(32, dtype=np.uint8)
    root, e = RW.value_root(PROGRAM, "w", 0, raw, (2, 16), KEY)
    tree = {"scheme": RW.NAME, "program": PROGRAM, "names": [["w", "uint8", [2, 16], 32]], "tensor_roots": [root.hex()],
            "geo_digest": GEO.hex(), "root": RW.weights_root(GEO, [["w", "uint8", [2, 16], 32]], [root]).hex(), "registered": [e]}
    assert RW.tensor_root(tree, 0, raw, KEY) == root and RW.tree_root(tree) == tree["root"]
    assert RW.tensor_root(tree, 0, raw, bytes(32)) != root
    with pytest.raises(ValueError, match="registration key"):
        RW.tensor_root(tree, 0, raw, None)


def test_the_replay_bundle_carries_the_registration_key_beside_the_tree(tmp_path, monkeypatch):
    monkeypatch.delenv("VERITY_WEIGHTS_HASH", raising=False)
    from tests.check.opened import commit_steps
    from tests.commit.test_store_dump import STEPS
    from verity_vllm.pipeline import replay_bundle as RB

    com, rc = commit_steps(STEPS)
    com.model = _model()
    rc.weight_registration = com.register_weights()
    p = RB.write_store(tmp_path, 1, com, rc)
    assert (p / "weights" / RW.KEY_FILE).read_bytes() == com._registration_key
    tree = json.loads((p / "weights" / "tree.json").read_text())
    assert tree["scheme"] == RW.NAME and tree["root"] == rc.weight_registration["root"] and set(tree["extra"]) == {"cos_sin_cache"}
