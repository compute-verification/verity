"""The committers' bytes as constants.

Every other test of the committers holds a root to the reference under the run's own fresh step keys.  Here the tensors and
the step keys are fixed, so the bytes each part of a committer makes are the ones written below: the offline core's step
roots, run root, context and layout digests (step hashing), a leaf opening and a range opening (retention and openings), a
dumped store's files, the weights' registration, and the windowed collector's roots over its CPU stand-ins.  A change that
moves one of them changes the commitment, wherever the code that makes it lives.

The modules under test: ``verity_vllm.commit.committer.native_host`` (the offline core), ``verity_vllm.commit.step_rows``,
``verity_vllm.commit.store_dump`` and ``verity_vllm.commit.committer.native_collect`` (the windowed collector).
"""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from tests.check import opened as O
from tests.commit import test_native_collect_plan_cache as windowed
from verity_vllm.commit import step_rows as R
from verity_vllm.commit import store_dump as SD

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")


def expand(label: str, n: int) -> bytes:
    """``n`` bytes that are a function of ``label``: SHA-256 over the label and a counter."""
    out = b""
    for i in range(-(-n // 32)):
        out += hashlib.sha256(f"{label}/{i}".encode()).digest()
    return out[:n]


def sha256(*parts: bytes) -> str:
    return hashlib.sha256(b"".join(parts)).hexdigest()


K = R.ROW_BYTES // 256                   # sizes in 256-byte units: one step inside a row, one over several, one over many
STEPS = {s: [(f"m{s}.{i}", expand(f"pins/{s}/{i}", n)) for i, n in enumerate(sizes)]
         for s, sizes in enumerate([(100 * K,), (700 * K, 3, 513 * K), (2048 * K, 1000 * K)])}
KEYS = {s: expand(f"pins/key/{s}", R.KEY_BYTES) for s in range(4)}
WEIGHTS_KEY = expand("pins/weights", R.KEY_BYTES)

RUN_ROOT = ("c03d1bd482073f1de7e865b05600d0ebe85095f4ed723272188f3aab28b1e7bd"
            "ac69158e23cb5f79bc8dd7228c099824845c97cc97c2252dc90902f814f46990")
STEP_SHAPES = [(1, 1600), (5, 19411), (12, 48768)]                               # leaves and bytes of each step
STEP_ROOTS = "b1cbb4b250e11a886fe7300c63159bd99f8f9dff43a8b6f61a05ceaf1a43177e"
TREE_ROOTS = "c17353a4190200e4bdd21a97cf11bbee71dfa46ba20c61a40e73bb20b0b26b0b"
LAYOUT_DIGESTS = "026fffa146c18de73954deace2ef5847136110acb8498996f4d42c51b2fbb41b"
CTX_DIGESTS = "e543c38bd089accb8c1c3889465eccce9babbc354469d8d72d8e5df48a822627"
LEAF_OPENING = "61abe8e837236bb3745985d4226630f04ef9f0e2828a9c47e0076d0379d7e676"
RANGE_OPENING = "79b28349afc4de57ac132def4e330a1fbb94c89b4dfed62904032f4cb3cac66c"
STORE_FILES = ["keys.bin", "levels/s0.bin", "levels/s1.bin", "levels/s2.bin", "raw/s0.bin", "raw/s1.bin", "raw/s2.bin"]
STORE_DIGESTS = "2e1ca123abaaf0706709a777d80435155d564fbffe1a75927160911d50e0b642"
STORE_RECORD = "9c90eaf8a06fbade99fea9c1768391278a275522bb6464c095a2719f8db2d2c6"
WEIGHTS_ROOT = ("ef253a3ca19e730a63fe5d3df9a592b4da72ddc9c80177ac5e8703b861752e5d"
                "77b605b5fd112c4f0e827704a44ec71fa5e86d09d2b17b1a3e9fe7771c487199")
WEIGHTS_TREE = "60fa462b8c6c20b147cfe7298597f54c83cdd93e965ab3570d1e73524d7def7d"
WINDOWED_RUN_ROOT = ("767d3a7b3bbb68fd8617f268a6284e51a083f664ed2ea754a489fb149dc10737"
                     "0e1d748bb9c4adb4fcd49fd671a0869ca3e1c2cba902b6b6dce5456c934db0ae")
WINDOWED_STEP_ROOTS = "b2109b339f1b2a52b2c474a0f27ca7d46a1e6e99d76db87268c26b7c92339ac7"


def committed():
    return O.commit_steps(STEPS, run_id="pins", row_keys={s: KEYS[s] for s in STEPS})


def test_the_offline_cores_roots_and_digests():
    com, rc = committed()
    assert rc.run_root.hex() == RUN_ROOT
    assert [(s.n_leaves, s.n_bytes) for s in rc.steps] == STEP_SHAPES
    assert sha256(*(s.root for s in rc.steps)) == STEP_ROOTS
    assert sha256(*(s.merkle_root for s in rc.steps)) == TREE_ROOTS
    assert sha256(*(com.layout_digest(com._layouts[s]) for s in sorted(com._layouts))) == LAYOUT_DIGESTS
    assert sha256(*(com._ctx_digest(s) for s in sorted(com._layouts))) == CTX_DIGESTS


def test_a_leaf_opening_and_a_range_opening():
    com, rc = committed()
    (leaf,) = com.open([(1, 2)])
    assert com.verify(rc, leaf) and sha256(leaf.value, leaf.salt, *leaf.path) == LEAF_OPENING
    rng = com.open_range(2, 1, 4)
    siblings = [side or b"-" for pair in rng.path for side in pair]
    assert com.verify_range(rc, rng) and sha256(rng.value, rng.salts, *siblings) == RANGE_OPENING


def test_a_dumped_stores_files(tmp_path):
    com, rc = committed()
    digests: dict[str, str] = {}
    SD.dump(com, rc, tmp_path, digests=digests)
    assert sorted(digests) == STORE_FILES
    assert sha256(json.dumps(sorted(digests.items())).encode()) == STORE_DIGESTS
    assert sha256((tmp_path / "store.json").read_bytes()) == STORE_RECORD


class _Model(torch.nn.Module):
    """Four registered tensors whose values are a function of their index: no generator, so no torch version moves them."""

    def __init__(self) -> None:
        super().__init__()

        def values(n: int, mul: int, dtype: torch.dtype) -> torch.Tensor:
            return ((torch.arange(n, dtype=torch.int64) * mul) % 251 - 125).to(dtype)

        self.proj = torch.nn.Linear(40, 120, bias=True)
        self.big = torch.nn.Parameter(values(R.ROW_BYTES + 77, 7, torch.int8), requires_grad=False)
        self.register_buffer("scale", torch.tensor(0.5))
        with torch.no_grad():
            self.proj.weight.copy_(values(120 * 40, 3, torch.bfloat16).float().reshape(120, 40))
            self.proj.bias.copy_(values(120, 5, torch.float32))


def test_the_weights_registration(monkeypatch):
    monkeypatch.setenv("VERITY_WEIGHTS_HASH", "host")
    com = O.committer(run_id="pins")
    com.model = _Model()
    com.weights_tree = {"scheme": R.NAME, "key": WEIGHTS_KEY.hex()}           # a re-registration takes the registration's key
    w = com.register_weights()
    assert (w["root"], w["tensors"], w["bytes"]) == (WEIGHTS_ROOT, 4, 23857)
    assert sha256(repr(sorted((k, v) for k, v in com.weights_tree.items() if k != "key")).encode()) == WEIGHTS_TREE


@pytest.mark.parametrize("retain", ["none", "host"])
@pytest.mark.parametrize("native_end", [True, False], ids=["native-end", "python-end"])
def test_the_windowed_collectors_roots(monkeypatch, retain, native_end):
    """The same roots keeping no bytes and keeping them on the host, with the step end in one native call or in several, on the
    run that learns the plans and on the run that follows them."""
    com = windowed.committer(monkeypatch, retain=retain)
    com.native_end = native_end
    sched = windowed.schedule(100, len(KEYS))
    for _run in ("learns", "follows"):
        com.replay_row_keys = dict(KEYS)
        rc, keys = windowed.run(com, sched)
        assert keys == KEYS
        assert rc.run_root.hex() == WINDOWED_RUN_ROOT
        assert [(s.n_leaves, s.n_bytes) for s in rc.steps] == [(5, 17168)] * len(KEYS)
        assert sha256(*(s.root for s in rc.steps)) == WINDOWED_STEP_ROOTS
