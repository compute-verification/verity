"""`pouw_h1_tables_device`: H1's call tables hashed by the Commit's row-table kernels, held bit for bit to
`pouw_h1_requests.hiding_rows`, which the prover rebuilds a drawn call's tables with (`pouw_h1_prover.open_call`). Over A-table
rows (K s8 codes and an f32 scale) and z-table records (``ZY_BYTES``), one row to a few hundred, K from 2 to two segments and
off an 8-byte boundary, at H1's table nonces: the root, and every opening of `hiding_rows`' tree verifies under it. The CPU
twin (`host_roots`, the kernels' functions under g++) runs everywhere with a C++ compiler; `H1Run` on the twin commits the
tables and manifest it commits with `hiding_rows`, and its prover directory reopens every call, also with each call's roots
read only when its round closes (the GPU's side-stream path). The device tests need CUDA and nvcc and skip elsewhere."""
from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw.audit import HM96, rows_domain  # noqa: E402
from verity.core.protocols.pouw.protocol import tagged  # noqa: E402
from verity.ml.commitments.limits import VerificationLimits  # noqa: E402
from verity.ml.commitments.merkle import Commitment, verify_opening  # noqa: E402
from verity_vllm.engine import int8_ops  # noqa: E402
from verity_vllm.protocols.pouw import h1 as PH  # noqa: E402
from verity_vllm.protocols.pouw.h1 import prover as PV  # noqa: E402
from verity_vllm.protocols.pouw.h1 import records as ZR  # noqa: E402
from verity_vllm.protocols.pouw.h1 import requests as RQ  # noqa: E402
from verity_vllm.protocols.pouw.h1 import tables_device as TD  # noqa: E402

KEY = bytes(range(32))
#: (rows, K, owner, table): an A table's T slots or a z table's records (K = ZY_BYTES - 4)
CASES = [(1, 2, 0, "a"), (3, 6, 5, "z"), (64, 4096, 7, "a"), (5, ZR.ZY_BYTES - 4, 9, "z"), (33, 1022, 11, "a"),
         (17, 1020, 1, "z"), (300, 62, 3, "a"), (2, ZR.ZY_BYTES - 4, (1 << 20) + 1, "z")]

needs_cxx = pytest.mark.skipif(not any(shutil.which(c) for c in ("g++", "c++", "clang++")), reason="no C++ compiler")


def _rows(n, k, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(-128, 128, (n, k), dtype=np.int8), rng.integers(0, 1 << 32, n, dtype=np.uint32)


def _reference(n, k, owner, name):
    """The table's rows, `hiding_rows`' tree of them under H1's binding and nonce for call `owner`, and the twin's input."""
    codes, words = _rows(n, k, n * 7 + k)
    binding = tagged(RQ.TABLE_TAGS[name], hashlib.sha512(b"salt").digest(), owner)
    nonce = RQ.TABLES_NONCE + 3 * owner + RQ.TABLE_NONCES[name]
    leaves = [codes[i].tobytes() + words[i].astype("<u4").tobytes() for i in range(n)]
    tree = RQ.hiding_rows(binding, owner, leaves, KEY, nonce)
    return tree, (rows_domain(binding, owner, n, HM96).domain_id, nonce, codes, words)


@needs_cxx
@pytest.mark.parametrize("n,k,owner,name", CASES)
def test_the_cpu_twin_is_hiding_rows(n, k, owner, name):
    """The kernels' functions on the CPU give `hiding_rows`' root, and each of its tree's openings (value, salt, path)
    verifies under that root: the rows the prover reopens are the ones the twin committed."""
    tree, table = _reference(n, k, owner, name)
    root = TD.host_roots([table], KEY)[0]
    assert root == tree.commitment.root
    for p in sorted({0, n // 2, n - 1}):
        o = tree.open(p)
        assert verify_opening(tree.domain, Commitment(root, n), o, RQ.ROW_SCHEMA, VerificationLimits())
        assert o.value[:-RQ.HM96_SALT] == table[2][p].tobytes() + table[3][p].astype("<u4").tobytes()


def test_the_layout_refuses_what_a_table_cant_be():
    for n, k, key, did, nonce in ((0, 2, KEY, bytes(64), 0), (1, 3, KEY, bytes(64), 0), (1, 0, KEY, bytes(64), 0),
                                  (1, 2, KEY, bytes(63), 0), (1, 2, KEY[:31], bytes(64), 0), (1, 2, KEY, bytes(64), 1 << 64),
                                  ((1 << 32) // 3 + 1, 2, KEY, bytes(64), 0)):
        with pytest.raises(ValueError):
            TD.layout(did, key, nonce, n, k)
    tab, args = TD.layout(bytes(64), KEY, (5 << 32) + 3, 3, 1022)
    assert list(tab[:6]) == [0, 0, 1022 | 3 << 32, 2 << 32, 0, 4] and list(args.nonce) == [0, 3, 5] and args.T == 1


def test_the_path_is_the_modes(monkeypatch):
    for mode in ("gpu", "host", "python"):
        monkeypatch.setattr(TD, "MODE", mode)
        assert TD.path(torch.zeros(1)) == mode
    monkeypatch.setattr(TD, "MODE", "auto")
    assert TD.path(torch.zeros(1)) == "python" and TD.path() == "python"
    monkeypatch.setattr(TD, "MODE", "cuda")
    with pytest.raises(ValueError, match="auto, gpu, host, python"):
        TD.path()


def test_record_rows_splits_a_record_as_the_kernels_take_it():
    records = np.random.default_rng(0).integers(0, 256, (3, ZR.ZY_BYTES), dtype=np.uint8)
    for got in (TD.record_rows(records), [x.numpy() for x in TD.record_rows(torch.from_numpy(records))]):
        codes, words = got
        assert codes.shape == (3, ZR.ZY_BYTES - 4) and words.shape == (3,)
        assert [codes[i].tobytes() + words[i].astype("<u4").tobytes() for i in range(3)] == [r.tobytes() for r in records]


K, ROWS, N = 64, 32, 64
SALT = hashlib.sha512(b"salt").digest()


def _window(tmp_path, mode, monkeypatch, device="cpu", tag=""):
    """A two-step window of one linear: step 0 serves 3 rows with their z/y records, step 1 is empty (`close`). Its record
    and prover directory."""
    monkeypatch.setattr(TD, "MODE", mode)
    out, prover = tmp_path / f"w-{mode}{tag}", tmp_path / f"p-{mode}{tag}"
    g = torch.Generator().manual_seed(3)
    weight = torch.randint(-127, 128, (N, K), generator=g, dtype=torch.int8).to(device)
    scale = (torch.rand(N, 1, generator=g) * 0.01 + 1e-4).float().to(device)
    x = (torch.randn(3, K, generator=g) * 3).bfloat16().to(device)
    run = PH.H1Run("t", SALT, {0: "q"}, [(N, K)], ROWS, 2, transcript_dir=str(out), key=KEY, prover=str(prover))
    xq, s_a = int8_ops.row_quant(x)
    _y, zy = int8_ops.scaled_mm_zy(xq, s_a, weight, scale, ROWS)
    run.record(0, xq, s_a, weight, scale, zy)
    run.finish()
    return json.loads((out / PH.CALLS_FILE).read_text()), prover


@needs_cxx
def test_a_window_on_the_twin_commits_what_hiding_rows_commits_and_its_calls_reopen(tmp_path, monkeypatch):
    """`H1Run` hashing its call tables by the kernels' functions commits `hiding_rows`' A and z roots, so the same manifest,
    calls_root and draw root; `open_call` rebuilds each call (served and empty) from the prover's directory under them."""
    hashed, twin = [], TD.host_roots
    monkeypatch.setattr(TD, "host_roots", lambda tables, key: hashed.append(len(tables)) or twin(tables, key))
    got, prover = _window(tmp_path, "host", monkeypatch)
    want, _ = _window(tmp_path, "python", monkeypatch)
    assert hashed == [2, 2]
    assert got["calls"] == want["calls"] and got["commitment"]["calls_root"] == want["commitment"]["calls_root"]
    assert got["commitment"]["draw"]["root"] == want["commitment"]["draw"]["root"]
    assert [c["served"] for c in got["calls"]] == [3, 0] and all(c["tables"]["a"] and c["tables"]["z"] for c in got["calls"])
    for i, c in enumerate(got["calls"]):
        opened = PV.open_call(prover, got, i, SALT)
        assert opened.a.commitment.root.hex() == c["tables"]["a"] and opened.z.commitment.root.hex() == c["tables"]["z"]


@needs_cxx
def test_a_window_whose_tables_hash_beside_the_forward_records_each_root_before_its_round(tmp_path, monkeypatch):
    """With the tables launched and read later (`device_hash`'s `Pending`, here the twin's roots held until asked and never
    ready before), `H1Run` waits for a call's roots when its round closes, so it commits what hashing them in the call
    commits; every launch is read."""
    launched = []

    class Later:
        def __init__(self, tables, key):
            roots = TD.host_roots(tables, key)
            self.roots, self.ready = lambda: launched.remove(self) or roots, lambda: False
            launched.append(self)

    monkeypatch.setattr(TD, "device_hash", Later)
    got, prover = _window(tmp_path, "gpu", monkeypatch)
    want, _ = _window(tmp_path, "python", monkeypatch)
    assert launched == []
    assert got["calls"] == want["calls"] and got["commitment"]["calls_root"] == want["commitment"]["calls_root"]
    assert got["commitment"]["draw"]["root"] == want["commitment"]["draw"]["root"]
    for i in range(len(got["calls"])):
        PV.open_call(prover, got, i, SALT)


def _gpu():
    if not torch.cuda.is_available() or TD.nvcc() is None:
        pytest.skip("needs CUDA and nvcc")


@pytest.mark.parametrize("n,k,owner,name", CASES)
def test_the_device_is_hiding_rows(n, k, owner, name):
    _gpu()
    tree, (did, nonce, codes, words) = _reference(n, k, owner, name)
    on = (did, nonce, torch.from_numpy(codes).cuda(), torch.from_numpy(words.view(np.int32)).cuda())
    assert TD.device_roots([on, on], KEY) == [tree.commitment.root] * 2
    pending = [TD.device_hash([on], KEY) for _ in range(TD.STREAMS + 1)]
    assert [p.roots() for p in pending] == [[tree.commitment.root]] * (TD.STREAMS + 1) and all(p.ready() for p in pending)


def test_auto_takes_the_gpu_for_cuda_tensors(monkeypatch):
    _gpu()
    monkeypatch.setattr(TD, "MODE", "auto")
    assert TD.available() and TD.path(torch.zeros(1, device="cuda")) == "gpu" and TD.path(torch.zeros(1)) == "python"


def test_a_window_on_the_gpu_commits_what_hiding_rows_commits(tmp_path, monkeypatch):
    _gpu()
    got, prover = _window(tmp_path, "auto", monkeypatch, device="cuda")
    want, _ = _window(tmp_path, "python", monkeypatch, device="cuda", tag="-ref")
    assert got["calls"] == want["calls"] and got["commitment"]["draw"]["root"] == want["commitment"]["draw"]["root"]
    for i in range(len(got["calls"])):
        PV.open_call(prover, got, i, SALT)
