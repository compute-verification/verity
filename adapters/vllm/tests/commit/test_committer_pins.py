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

RUN_ROOT = ("dc383743d414c311105097a9253b06dbfe27d60bc52f28b5a5d5d059f40f6de0"
            "7ae986e2c4f6972b959d15eb2992b58cf32b60c64ffc4046b934e04ff0254d49")
STEP_SHAPES = [(1, 1600), (5, 19411), (12, 48768)]                               # leaves and bytes of each step
STEP_ROOTS = "30233ecaad7966e2da2d7bd9e2b4b50ad17abb4deae03b735efc8d994db9aefc"
TREE_ROOTS = "4c3fa181a990ad208f6315f0ac3b124547170e4e7713d32b3d3816309bff9244"
LAYOUT_DIGESTS = "7077f32bf5bf2ec654667dd047447e8cc152ccc0f42c99f26041736fb72777d7"
CTX_DIGESTS = "1cfe770299944ab9b973c89af7ec614fbf65f8b2268ce7212f7400807a4c070b"
LEAF_OPENING = "67201ce867d92ae2a1399e48c88c8c0fd2a3103b2dce844222a990c7419cb6fb"
RANGE_OPENING = "034b4a411a79e5ecd8122b7357ed817956028e0bba4aa3ac83062eb1fceffe0d"
STORE_FILES = ["keys.bin", "levels/s0.bin", "levels/s1.bin", "levels/s2.bin", "raw/s0.bin", "raw/s1.bin", "raw/s2.bin"]
STORE_DIGESTS = "ecc25dffc73f477689b24308479a98ab7b209676a9e7a15ded8529925a5b5743"
STORE_RECORD = "c6b4a2d53ba98a5ba3caaf43d43cbd90ea6044a82a03cbe8ba44e94f1d33cd12"
WEIGHTS_ROOT = ("153c809fd95969c856b38f4f7dba39bb1828b97d52c4f66b8b32a7e893fe1043"
                "bae87b9212e2966a336749eee27bf906c4781278c04ac1919be4bb0d96f69c5d")
WEIGHTS_TREE = "616b4a03a6de0a74e3f7fbc04cf6c2ee545b3f956e54297c71b7d08d5c165a77"
WINDOWED_RUN_ROOT = ("a4be800397a080bce4d10062267ac8fcc7766856b2c55f6ab055eea1e0fa4766"
                     "ff36f87a5554ff3d9dc6dec4cd365ff8d6d2b150b3ae7722fcebf49faa22ed2d")
WINDOWED_STEP_ROOTS = "bc7eb94f6c9096f6f2d897100600de276025b9bcf03460f852507e2b89639df4"


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
