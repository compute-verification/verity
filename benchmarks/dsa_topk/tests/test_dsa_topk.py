"""The canonical top-k row (`dsa_topk.reference`): rows of at most k keys are vLLM's, a longer row takes the registered
relation's top-k set (`topk_set`: the fast kernels' key, ties to the smaller index) in ascending order; the torch
reference agrees with it; the wrappers run the real op first and the pass only on calls that can hold a row longer than k, in
the right row mode; install wraps once; the plugin is off unless asked; and, where Triton is importable, its kernels
(in Triton's interpreter) rewrite any fast selection, right, wrong or garbage, into the reference."""
import json
import math
import os
import random
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Triton builds its own library's kernels (tl.min, tl.sum) for the interpreter only when this is set at its first import
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dsa_topk


def f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


WORDS = [0xFFFFFFFF, 0xFFC00000, 0xFF800001, 0xFF800000, 0xFF7FFFFF, 0xBF800000, 0x80000001, 0x80000000, 0x00000000,
         0x00000001, 0x00800000, 0x3F800000, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF]


def signed(word: int) -> int:
    return struct.unpack("<i", struct.pack("<I", word))[0]


def test_ordered_is_the_fast_kernels_key():
    # -NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN, the registered relation's key less 2**31
    keys = [dsa_topk.ordered(signed(w)) for w in WORDS]
    assert keys == sorted(set(keys))
    assert all(-(1 << 31) <= k < (1 << 31) for k in keys)
    assert [k + (1 << 31) for k in keys] == [registered().topk_key(w) for w in WORDS]


def registered():
    from verity.definitions.registry import dsa_topk as T

    return T


def test_short_rows_are_vllms():
    assert dsa_topk.reference([], 4) == [-1, -1, -1, -1]
    assert dsa_topk.reference([5.0, -1.0, 2.0], 4) == [0, 1, 2, -1]
    assert dsa_topk.reference([5.0, -1.0, 2.0, 0.0], 4) == [0, 1, 2, 3]


def test_long_row_takes_largest_keys_ties_to_smaller_index_ascending():
    assert dsa_topk.reference([3.0, 9.0, 1.0, 7.0, 5.0], 3) == [1, 3, 4]
    assert dsa_topk.reference([1.0, 1.0, 1.0, 1.0, 0.0], 2) == [0, 1]
    assert dsa_topk.reference([1.0, 1.0, 1.0, 1.0, 2.0], 3) == [0, 1, 4]
    assert dsa_topk.reference([-0.0, 0.0, -0.0, 0.0], 2) == [1, 3]
    assert dsa_topk.reference([math.inf, -math.nan, 1.0, math.nan], 2) == [0, 3]
    assert dsa_topk.reference([-math.nan, -math.inf, -1.0, -math.nan], 2) == [1, 2]


@pytest.mark.parametrize("dist", ["randn", "int", "edge"])
def test_reference_is_the_registered_relations_set(dist):
    T = registered()
    k = 6
    data, lo, n = rows_case(2, 60, 24, k, dist)
    for row, a, m in zip(data, lo, n):
        vals = [struct.unpack("<f", struct.pack("<f", v))[0] for v in row[a:a + m]]
        if m <= k:
            continue
        got = dsa_topk.reference(vals, k)
        words = [struct.unpack("<I", struct.pack("<f", v))[0] for v in vals]
        assert set(got) == T.topk_set(words, k) and T.order_ok(words, got, k) == 1


def rows_case(seed: int, rows: int, cols: int, k: int, dist: str):
    rng = random.Random(seed)
    data, lo, n = [], [], []
    for _ in range(rows):
        if dist == "int":
            row = [float(rng.randrange(4)) for _ in range(cols)]
        elif dist == "edge":
            row = [rng.choice([math.nan, -math.nan, 0.0, -0.0, math.inf, -math.inf, rng.gauss(0, 1)]) for _ in range(cols)]
        else:
            row = [rng.gauss(0, 1) for _ in range(cols)]
        a = rng.randrange(cols - 2 * k)
        m = rng.choice([-3, 0, 1, k - 1, k, k + 1, cols - a, cols - a, rng.randrange(k + 1, cols - a + 1),
                        rng.randrange(cols - a + 1)])
        data.append(row)
        lo.append(a)
        n.append(min(m, cols - a))
    return data, lo, n


@pytest.mark.parametrize("dist", ["randn", "int", "edge"])
def test_torch_reference_agrees(dist):
    torch = pytest.importorskip("torch")
    k = 6
    data, lo, n = rows_case(1, 40, 24, k, dist)
    got = dsa_topk.reference_rows(torch.tensor(data, dtype=torch.float32), lo, n, k).tolist()
    want = [dsa_topk.reference([struct.unpack("<f", struct.pack("<f", v))[0] for v in row[a:a + max(m, 0)]], k)
            for row, a, m in zip(data, lo, n)]
    assert got == want


class Recorder(dsa_topk.Canon):
    def __init__(self):
        super().__init__()
        self.runs = []

    def run(self, op, logits, out, k, mode, a, b=None, **kw):
        self.runs.append((op, mode, kw))


def fake_ns(log):
    def op(name):
        def real(*args):
            log.append((name, args))
        return real
    return SimpleNamespace(**{name: op(name) for name in dsa_topk.OPS})


def test_wrappers_dispatch(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(dsa_topk, "_capturing", lambda: False)
    log, canon = [], Recorder()
    ns = fake_ns(log)
    assert dsa_topk.install(ns, canon) is canon
    k = 4
    out = torch.empty(3, k, dtype=torch.int32)
    wide, narrow = torch.zeros(3, 16), torch.zeros(3, 4)
    ks = ke = torch.zeros(3, dtype=torch.int32)

    ns.top_k_per_row_prefill(narrow, ks, ke, out, 3, 4, 1, k)
    ns.top_k_per_row_prefill(wide, ks, ke, out, 0, 16, 1, k)
    assert canon.runs == []
    ns.top_k_per_row_prefill(wide, ks, ke, out, 3, 16, 1, k)
    assert canon.runs[-1] == ("top_k_per_row_prefill", 0, dict(rows=3, ncols=16, ls0=16, ls1=1))
    assert log[-1][0] == "top_k_per_row_prefill" and log[-1][1][0] is wide

    ns.top_k_per_row_decode(wide, 1, torch.zeros(3, 1, dtype=torch.int32), out, 3, 16, 1, k)
    assert canon.runs[-1] == ("top_k_per_row_decode", 1, dict(rows=3, ncols=16, ls0=16, ls1=1, next_n=1))
    ns.top_k_per_row_decode(wide, 3, torch.zeros(1, dtype=torch.int32), out, 3, 16, 1, k)
    assert canon.runs[-1] == ("top_k_per_row_decode", 2, dict(rows=3, ncols=16, ls0=16, ls1=1, next_n=3))

    n = len(canon.runs)
    ns.persistent_topk(wide, ks, out, None, k, 4)  # every row clamped to max_seq_len <= k
    ns.persistent_topk(narrow, ks, out, None, k, 16)
    ns.cooperative_topk(wide, ks, out, None, k, 4)  # the batch's longest row is at most k
    assert len(canon.runs) == n
    ns.persistent_topk(wide, ks, out, None, k, 12)
    assert canon.runs[-1] == ("persistent_topk", 1, dict(rows=3, ncols=12, ls0=16, ls1=1))
    ns.cooperative_topk(wide, ks, out, None, k, 5)
    assert canon.runs[-1] == ("cooperative_topk", 1, dict(rows=3, ncols=16, ls0=16, ls1=1))
    monkeypatch.setattr(dsa_topk, "_capturing", lambda: True)
    ns.cooperative_topk(wide, ks, out, None, k, 4)  # a captured graph replays with other lengths
    assert canon.runs[-1][0] == "cooperative_topk" and len(canon.runs) == n + 3
    assert canon.calls == {"top_k_per_row_prefill": 3, "top_k_per_row_decode": 2, "persistent_topk": 3,
                           "cooperative_topk": 3}


def test_install_wraps_once_and_skips_absent_ops():
    log = []
    ns = fake_ns(log)
    del ns.cooperative_topk
    first = dsa_topk.install(ns)
    wrapped = ns.persistent_topk
    assert dsa_topk.install(ns) is first and ns.persistent_topk is wrapped
    assert not hasattr(ns, "cooperative_topk")
    assert ns.persistent_topk.real.__name__ == "real"


#: the pin's schemas (csrc/libtorch_stable/torch_bindings.cpp), for toy CPU ops of the same names
SCHEMAS = {
    "top_k_per_row_prefill": "(Tensor logits, Tensor row_starts, Tensor row_ends, Tensor(a!) indices, int num_rows, "
                             "int stride0, int stride1, int top_k) -> ()",
    "top_k_per_row_decode": "(Tensor logits, int next_n, Tensor seq_lens, Tensor(a!) indices, int num_rows, "
                            "int stride0, int stride1, int top_k) -> ()",
    "persistent_topk": "(Tensor logits, Tensor lengths, Tensor(a!) output, Tensor workspace, int k, "
                       "int max_seq_len) -> ()",
    "cooperative_topk": "(Tensor logits, Tensor lengths, Tensor(a!) output, Tensor workspace, int k, "
                        "int max_seq_len) -> ()",
}
TOY_OUT = {"top_k_per_row_prefill": 3, "top_k_per_row_decode": 3, "persistent_topk": 2, "cooperative_topk": 2}


def toy_ops(torch, ns: str, names):
    """``torch.ops.<ns>`` with toy CPU ops of `SCHEMAS`'s names, each filling its output with 7."""
    lib = torch.library.Library(ns, "DEF")
    for name in names:
        def impl(*args, i=TOY_OUT[name]):
            args[i].fill_(7)
        lib.define(name + SCHEMAS[name])
        lib.impl(name, impl, "CPU")
    return lib, getattr(torch.ops, ns)


def toy_args(torch, name):
    k, logits = 4, torch.zeros(2, 16)
    out, lens, ws = torch.zeros(2, k, dtype=torch.int32), torch.zeros(2, dtype=torch.int32), torch.zeros(1)
    return out, {"top_k_per_row_prefill": (logits, lens, lens, out, 2, 16, 1, k),
                 "top_k_per_row_decode": (logits, 1, lens.reshape(2, 1), out, 2, 16, 1, k),
                 "persistent_topk": (logits, lens, out, ws, k, 16),
                 "cooperative_topk": (logits, lens, out, ws, k, 16)}[name]


def test_wrapped_ops_keep_their_overloads_under_a_dispatch_mode(monkeypatch):
    """A TorchDispatchMode (a capture's recorder) makes torch look the op up as ``torch.ops.<ns>.<name>.default``,
    which a bare wrapper lacks: the control fails that way, and `install`'s wrappers pass."""
    torch = pytest.importorskip("torch")
    from torch.utils._python_dispatch import TorchDispatchMode

    class Seen(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.funcs = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.funcs.append(func)
            return func(*args, **(kwargs or {}))

    monkeypatch.setattr(dsa_topk, "_capturing", lambda: False)
    _bare_lib, bare = toy_ops(torch, "dsa_topk_bare", ["persistent_topk"])
    real = bare.persistent_topk
    bare.persistent_topk = lambda *a: real(*a)
    _, args = toy_args(torch, "persistent_topk")
    with Seen(), pytest.raises(AttributeError, match="default"):
        bare.persistent_topk(*args)

    _lib, ns = toy_ops(torch, "dsa_topk_toy", dsa_topk.OPS)
    canon = Recorder()
    dsa_topk.install(ns, canon)
    for name in dsa_topk.OPS:
        out, args = toy_args(torch, name)
        with Seen() as seen:
            getattr(ns, name)(*args)
        fn = getattr(ns, name)
        assert fn.dsa_topk_canon is canon and seen.funcs == [fn.real.default] and fn.default is fn.real.default
        assert bool((out == 7).all())
    assert [r[0] for r in canon.runs] == list(dsa_topk.OPS)


def test_register_is_off_unless_asked(monkeypatch):
    calls = []
    monkeypatch.setattr(dsa_topk, "install", lambda *a, **k: calls.append(a))
    for v in (None, "", "0"):
        if v is None:
            monkeypatch.delenv(dsa_topk.ENV, raising=False)
        else:
            monkeypatch.setenv(dsa_topk.ENV, v)
        dsa_topk.register()
    assert calls == []
    monkeypatch.setenv(dsa_topk.ENV, "sorted")
    with pytest.raises(ValueError, match="canonical"):
        dsa_topk.register()
    monkeypatch.setenv(dsa_topk.ENV, "canonical")
    monkeypatch.delenv(dsa_topk.LOG_ENV, raising=False)
    dsa_topk.register()
    assert calls == [()]


def test_log_rewrites_its_record_while_serving(monkeypatch, tmp_path):
    monkeypatch.setattr(dsa_topk, "_capturing", lambda: False)
    monkeypatch.setattr(dsa_topk, "FLUSH_EVERY", 3)
    canon = dsa_topk.Canon()
    dsa_topk.log_to(canon, str(tmp_path))
    path = tmp_path / f"dsa_topk.{os.getpid()}.json"
    canon.note("persistent_topk")
    canon.note("persistent_topk")
    assert not path.exists()
    canon.note("cooperative_topk")
    rec = json.loads(path.read_text())
    assert rec["calls"]["persistent_topk"] == 2 and rec["calls"]["cooperative_topk"] == 1
    monkeypatch.setattr(dsa_topk, "_capturing", lambda: True)
    for _ in range(3):
        canon.note("persistent_topk")
    assert json.loads(path.read_text())["calls"]["persistent_topk"] == 2


def test_site_declares_the_plugin(tmp_path):
    d = dsa_topk.site(tmp_path)
    eps = (d / "verity_dsa_topk-0.0.dist-info" / "entry_points.txt").read_text()
    assert "[vllm.general_plugins]" in eps and "verity_dsa_topk = dsa_topk:register" in eps
    for name in ("dsa_topk.py", "dsa_topk_kernels.py"):
        assert (d / name).read_text() == Path(dsa_topk.__file__).with_name(name).read_text()


def fast_selection(torch, x, lo, n, k, how, rng):
    """What a fast kernel might write: the right set in a shuffled order with ties broken at random (``atomic``), or a
    wrong selection (``dup``: the right set with one index other than the k-th key's written twice; ``wrong``: random
    distinct indices; ``garbage``: anything, out of range included)."""
    out = torch.full((x.shape[0], k), -1, dtype=torch.int32)
    for r in range(x.shape[0]):
        m = max(int(n[r]), 0)
        if m <= k:
            out[r, :m] = torch.arange(m, dtype=torch.int32)
            continue
        if how in ("atomic", "dup"):
            key = [dsa_topk.ordered(int(b)) for b in x[r, lo[r]:lo[r] + m].contiguous().view(torch.int32)]
            order = sorted(range(m), key=lambda i: (key[i], rng.random()), reverse=True)[:k]
            if how == "dup":
                order[1] = order[0]
            rng.shuffle(order)
            out[r] = torch.tensor(order, dtype=torch.int32)
        elif how == "wrong":
            out[r] = torch.tensor(rng.sample(range(m), k), dtype=torch.int32)
        else:
            out[r] = torch.tensor([rng.randrange(-5, m + 5) for _ in range(k)], dtype=torch.int32)
    return out


def interpreted(monkeypatch):
    pytest.importorskip("triton")
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(dsa_topk, "_KERNELS", {})
    return torch


@pytest.mark.parametrize("seg_min_keys", [1 << 20, 24])
@pytest.mark.parametrize("dist", ["randn", "int", "edge"])
def test_kernels_canonicalize_in_interpreter(monkeypatch, dist, seg_min_keys):
    torch = interpreted(monkeypatch)
    rng = random.Random(3)
    k, rows, cols = 12, 10, 90
    data, lo, n = rows_case(5, rows, cols, k, dist)
    x = torch.tensor(data, dtype=torch.float32)
    lo_t, n_t = torch.tensor(lo, dtype=torch.int32), torch.tensor(n, dtype=torch.int32)
    ref = dsa_topk.reference_rows(x, lo, n, k)
    assert sum(m > k for m in n) >= rows // 3
    for how in ("atomic", "dup", "wrong", "garbage"):
        canon = dsa_topk.Canon(seg_min_keys=seg_min_keys, block=32)
        canon._sms[x.device] = 64
        assert (canon.segments(x.device, rows, cols) > 1) == (seg_min_keys < cols)
        out = fast_selection(torch, x, lo, n, k, how, rng)
        canon.run("top_k_per_row_prefill", x, out, k, 0, lo_t, lo_t + n_t, rows=rows, ncols=cols, ls0=cols, ls1=1)
        assert torch.equal(out, ref), how
        stats = dict(zip(dsa_topk.STATS, canon.stats(x.device).tolist()))
        assert stats["rows_long"] == sum(m > k for m in n)
        if how == "atomic" and dist == "randn":
            assert stats["rows_bisected"] == 0


@pytest.mark.parametrize("sms", [1, 64])
@pytest.mark.parametrize("mode", [1, 2])
def test_kernels_decode_rows_in_interpreter(monkeypatch, mode, sms):
    torch = interpreted(monkeypatch)
    rng = random.Random(4)
    k, rows, cols, next_n = 8, 6, 64, 2
    x = torch.tensor([[float(rng.randrange(6)) for _ in range(cols)] for _ in range(rows)])
    if mode == 1:
        seq = torch.tensor([3, 70, 9, 40, 64, -2], dtype=torch.int32)
        n = seq.clamp(0, cols)
    else:
        seq = torch.tensor([30, 9, 64], dtype=torch.int32)
        r = torch.arange(rows)
        n = (seq[r // next_n] - next_n + r % next_n + 1).clamp(0, cols)
    lo = [0] * rows
    ref = dsa_topk.reference_rows(x, lo, n.tolist(), k)
    out = fast_selection(torch, x, lo, n.tolist(), k, "atomic", rng)
    canon = dsa_topk.Canon(seg_min_keys=16, block=32)
    canon._sms[x.device] = sms
    assert (canon.segments(x.device, rows, cols) > 1) == (sms > 1)
    canon.run("top_k_per_row_decode", x, out, k, mode, seq, rows=rows, ncols=cols, ls0=cols, ls1=1, next_n=next_n)
    assert torch.equal(out, ref)
