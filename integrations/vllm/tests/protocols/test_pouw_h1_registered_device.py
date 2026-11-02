"""`pouw_h1_registered_device`: H1's registered values hashed by `pouw_cuda/registered_rows.cu`, held byte for byte to
`pouw_h1_registered.commit`'s NumPy path and to core's `registered.Registered.commit`. Over random rows of 16-bit words (one row
to a few thousand, one word to a z/y record's 3072, a row whose bytes sit 2 and 4 bytes off an 8-byte boundary), under each
domain binding, nonces past 2^32 and 2^63, and given salts: the ``b || c`` strings, every tree level, the root, domain and
leaf count. The CPU twin (`host_commit`, the kernels' functions under g++) runs everywhere with a C++ compiler; the device
tests need CUDA and nvcc and skip elsewhere: there, rows go in several chunks and a tree wide enough for a launch a level,
`commit` takes the GPU under ``VERITY_H1_REGISTER=gpu`` (and by default), and a small served H1 window registers the same
entries, digest and manifest rows on either path, and the same again, with the same receipts, served from the device
with its z/y records hashed where the arm wrote them."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw.audit import hm96_salt
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity_vllm.protocols.pouw.h1 import registered as NR
from verity_vllm.protocols.pouw.h1 import registered_device as D

PROGRAM, WINDOW = "ab" * 64, "cd" * 64
KEY = bytes(range(32))
CASES = [(1, 1, {}, 0), (2, 2, {"window": WINDOW}, 7), (3, 66, {}, (1 << 42) + 3), (5, 1026, {"window": WINDOW}, 1 << 50),
         (8, 3, {"produced": "ef" * 64}, (1 << 63) + 5), (4, 5, {"session": "s-1"}, 11), (33, 3072, {"window": WINDOW}, 2),
         (64, 10, {}, 3), (17, 2050, {"window": WINDOW}, (1 << 42) + 9), (300, 7, {}, 5)]

needs_cxx = pytest.mark.skipif(not any(shutil.which(c) for c in ("g++", "c++", "clang++")), reason="no C++ compiler")


def _words(n: int, w: int) -> np.ndarray:
    return np.random.default_rng(n * 1000 + w).integers(0, 1 << 16, (n, w), dtype=np.uint16)


def _numpy(words, nonce, given=None, **binding):
    dom = G.domain(PROGRAM, "h1/v", words.shape[1], words.shape[0], G.WORD_BITS, **binding)
    return NR._numpy(words, KEY, nonce, given, dom.domain_id), dom.domain_id


def _core(words, nonce, given=None, **binding):
    n = len(words)
    salts = ([hm96_salt(KEY, nonce, j) for j in range(n)] if given is None else [given[192 * j:192 * (j + 1)] for j in range(n)])
    return G.Registered.commit(PROGRAM, "h1/v", words.tolist(), salts, schema=G.SCHEMA_V2, **binding)


def _gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or D.nvcc() is None:
        pytest.skip("needs CUDA and nvcc")
    return torch


@needs_cxx
@pytest.mark.parametrize("n,w,binding,nonce", CASES)
def test_the_cpu_twin_is_the_numpy_path(n, w, binding, nonce):
    """The kernels' functions on the CPU give the NumPy path's strings and levels, so its root, under the same domain."""
    words = _words(n, w)
    want, did = _numpy(words, nonce, **binding)
    assert D.host_commit(words, KEY, nonce, None, did) == want


@needs_cxx
def test_the_cpu_twin_is_cores_commit_with_given_salts_and_unaligned_rows():
    """Given salts (a public table's) and rows of 2 W bytes 2 and 4 bytes off alignment: core's ``b || c`` and levels."""
    for n, w in ((3, 5), (6, 2050), (2, 1)):
        words = _words(n, w)
        given = b"".join(bytes([j + 1]) * 192 for j in range(n))
        ref = _core(words, 0, given, window=WINDOW)
        assert D.host_commit(words, KEY, 0, given, ref.domain.domain_id) == (ref.bcs, ref.levels)


#: (context, first position, salts): a window's open root as `Served` has it, short and empty contexts, positions past 2^32
#: and up to the last u64
PUBLIC = [("ab" * 64, 0, 5), ("cd" * 64, 86011, 40), ("w", (1 << 40) + 3, 3), ("", 0, 1), ("ef" * 64, (1 << 64) - 2, 2)]


def _head(context: str) -> bytes:
    ctx = context.encode()
    return CC.PUBLIC_ROW_SALT_DOMAIN + len(ctx).to_bytes(8, "big") + ctx


def _public(context: str, first: int, n: int) -> bytes:
    return b"".join(CC.public_row_salt(context, u) for u in range(first, first + n))


@needs_cxx
@pytest.mark.parametrize("context,first,n", PUBLIC)
def test_public_salts_are_cores_on_the_cpu_twin_and_the_numpy_path(context, first, n, monkeypatch):
    """`public_salts` of `core_check.public_row_salt`'s head: core's salts, by the kernels' function on the CPU and by
    `hashlib`."""
    want = _public(context, first, n)
    assert D.host_public_salts(_head(context), first, n) == want
    monkeypatch.setattr(NR, "MODE", "numpy")
    assert NR.public_salts(_head(context), first, n) == want


def test_public_salts_refuse_what_they_cant_derive():
    for head, first, n in ((bytes(321), 0, 1), (b"h", 0, 0), (b"h", -1, 1), (b"h", (1 << 64) - 1, 2)):
        with pytest.raises(ValueError):
            D.pub(head, first, n)
    with pytest.raises(ValueError, match="u64"):
        NR.public_salts(b"h", (1 << 64) - 1, 2)


def test_the_layout_refuses_what_a_value_cant_be():
    for n, rb, key, did in ((0, 2, KEY, bytes(64)), (D.MAX_ROWS + 1, 2, KEY, bytes(64)), (1, 0, KEY, bytes(64)),
                            (1, 2, KEY, bytes(63)), (1, 2, KEY[:31], bytes(64))):
        with pytest.raises(ValueError):
            D.table(n, rb, key, 0, False, did)
    assert D.table(1, 2, b"", 0, True, bytes(64)).W == 1
    with pytest.raises(ValueError, match="u64"):
        D.table(1, 2, KEY, 1 << 64, False, bytes(64))


def test_the_path_is_the_modes(monkeypatch):
    monkeypatch.setattr(NR, "MODE", "numpy")
    assert NR.path() == "numpy"
    monkeypatch.setattr(NR, "MODE", "gpu")
    assert NR.path() == "gpu"
    monkeypatch.setattr(NR, "MODE", "cuda")
    with pytest.raises(ValueError, match="auto, gpu or numpy"):
        NR.path()


@pytest.mark.parametrize("n,w,binding,nonce", CASES)
def test_the_device_is_the_numpy_path_and_core(n, w, binding, nonce, monkeypatch):
    """On the GPU, ``commit`` registers what the NumPy path and core register: entry, root, domain, leaves, strings, levels,
    paths, and a read of every row opens under core's `verify_read`."""
    _gpu()
    words = _words(n, w)
    monkeypatch.setattr(NR, "MODE", "gpu")
    got = NR.commit(PROGRAM, "h1/v", words, KEY, nonce, True, **binding)
    monkeypatch.setattr(NR, "MODE", "numpy")
    want = NR.commit(PROGRAM, "h1/v", words, KEY, nonce, True, **binding)
    ref = _core(words, nonce, **binding)
    assert got.entry() == want.entry() == ref.entry() and got.domain == ref.domain and got.leaves == ref.leaves == n
    assert got.bcs == want.bcs == ref.bcs and got.levels == want.levels == ref.levels
    assert [got.path(p) for p in range(n)] == [ref.path(p) for p in range(n)]
    assert all(G.verify_read(got.domain, ref.root, p, got.bcs[p], got.path(p), G.SCHEMA_V2) for p in range(n))


def test_the_device_takes_given_salts_chunks_and_wide_trees(monkeypatch):
    """Given salts, rows copied in many chunks, and a tree whose lower levels are a launch each (over 1024 nodes): the
    NumPy path's strings and levels."""
    _gpu()
    monkeypatch.setattr(D, "CHUNK_BYTES", 4096)
    for n, w, given in ((3, 5, True), (6, 2050, True), (5000, 2, False), (2049, 66, False)):
        words = _words(n, w)
        ys = b"".join(j.to_bytes(4, "little") * 48 for j in range(1, n + 1)) if given else None
        want, did = _numpy(words, (1 << 42) + n, ys, window=WINDOW)
        assert D.device_commit(words, KEY, (1 << 42) + n, ys, did) == want


@pytest.mark.parametrize("context,first,n", [*PUBLIC, ("ab" * 64, 1 << 33, 70000)])
def test_public_salts_on_the_device_are_cores(context, first, n, monkeypatch):
    _gpu()
    monkeypatch.setattr(NR, "MODE", "gpu")
    assert NR.public_salts(_head(context), first, n) == D.device_public_salts(_head(context), first, n) == _public(
        context, first, n)


def test_auto_takes_the_gpu_where_it_runs(monkeypatch):
    _gpu()
    monkeypatch.setattr(NR, "MODE", "auto")
    assert D.available() and NR.path() == "gpu"


def test_a_served_window_registers_the_same_on_either_path(monkeypatch):
    """A small H1 window (`test_pouw_h1_served._window`) and the period's weights registered on the GPU and with NumPy: every
    entry, the served digest, each call's manifest row and each public table's salts are the same."""
    torch = _gpu()
    from tests.protocols import test_pouw_h1_served as T

    def window(where):
        monkeypatch.setattr(NR, "MODE", where)
        torch.manual_seed(0)
        weights = T._period(T._weights())
        run, _ws, _got = T._window(T.W1, weights=weights, served=1)
        pubs = {v: r.salts for v, r in run.served.registered.items() if "pub@" in v}
        return (weights.entries(), run.served.entries(), run.served.digest(), [run.tables[c.index] for c in run.calls],
                pubs)

    got, want = window("gpu"), window("numpy")
    assert got == want and got[4] and len(got[1]) == 2 * len(T.SHAPES) * T.STEPS + (T.STEPS + 1) * len(T.KS)


def test_a_window_served_on_the_device_registers_what_its_host_copy_does(monkeypatch):
    """The window above with receipts, served with the arm's tensors on the GPU, so each call's z/y records stay where the
    arm wrote them and are hashed there: every entry, the served digest, each call's manifest row, each z table's root and
    each receipt are those of the same window served on the host, registered on the GPU and with NumPy."""
    torch = _gpu()
    from tests.protocols import test_pouw_h1_served as T

    def window(where, device):
        monkeypatch.setattr(NR, "MODE", where)
        torch.manual_seed(0)
        weights, made = T._period(T._weights()), T._Prepared()
        run, _ws, _got = T._window(T.W1, weights=weights, served=2, receipts=made, device=device)
        assert not run.served.refused, run.served.refused
        zy = {v: r.root for v, r in run.served.registered.items() if "zy@" in v}
        return (run.served.entries(), run.served.digest(), [run.tables[c.index] for c in run.calls], zy,
                {c: r.tobytes() for c, r in made.items()})

    got, host, numpy = window("gpu", "cuda"), window("gpu", None), window("numpy", None)
    assert got == host == numpy and got[3] and len(got[4]) == len(T.SHAPES) * T.STEPS
