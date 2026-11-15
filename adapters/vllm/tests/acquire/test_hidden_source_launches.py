"""The M1 hidden-stream sources of `verity_vllm.acquire.sources.hidden_source`, FlashAttention-2's and FlashAttention-3's, with a stand-in
tap build on `torch.ops`: the taps are CUDA builds, vLLM's attention ops are not installed where the CPU suites run, and the stream buffer
is allocated on the CPU here.  What is held is the wrapper's side of a launch.  Outside a committed step the installed kernel runs.  Inside
one the tap is armed with the launch's geometry, the tapped kernel runs and the stream buffer is committed under the launch's name.  A
launch the build has no kernel for, or one the kernel skips, is a collector error and one more expected X-03 row.  `uninstall` puts back
the op and disarms the tap.  The stream's words are `verity_vllm.properties.fa_tap_exactness`'s, on a GPU."""
from __future__ import annotations

import re
import types
from typing import Any

import pytest
import torch

from tests.acquire.test_vocab_range_source import Recorder
from verity_vllm.acquire.sources import hidden_source as H
from verity_vllm.engine import hooks

HD64, HD128 = 1 << 2, 1 << 4                                    # a build's head-dim mask: bit head_dim // 32
SRC, SRC_MS = 59, 123                                           # a build's source mask without and with the MS class
FA2_SO, FA3_SO = "/taps/verity_fa2_matReq.so", "/taps/verity_fa3_matReq.so"


class Committer(Recorder):
    """The recording committer, with the spans a wrapper times itself into; it keeps the tensor it is handed, not a copy."""

    #: the X-03 rows a wrapper counts on its committer
    x03_expected: int

    def __init__(self) -> None:
        super().__init__()
        self.timed: list[tuple[str, int, int]] = []
        self.spans = types.SimpleNamespace(add=self._span)

    def _span(self, kind: str, _seconds: float, *, step: int, nbytes: int) -> None:
        self.timed.append((kind, step, nbytes))

    def _acquire(self, name: str, t: Any, job: Any) -> None:
        self.acquired.append((name, t))


class Tap:
    """What a source reads of a tap build's namespace.  It records how it was armed and what it launched."""

    def __init__(self, *, mode: int = 3, src: int = SRC, hdims: int = HD64 | HD128) -> None:
        self.mode, self.src, self.hdims = mode, src, hdims
        self.armed: list[tuple] = []
        self.launches: list[tuple] = []
        self.skipped = 0
        self.skips = False                                      # the kernel refuses what was declared: it counts the launch, writes nothing

    def verity_tap_mode(self) -> int:
        return self.mode

    def verity_tap_src(self) -> int:
        return self.src

    def verity_tap_skipped(self) -> int:
        return self.skipped

    def launch(self, *args: Any) -> str:
        self.launches.append(args)
        self.skipped += self.skips
        return "tapped"


class Hd64Tap(Tap):
    """The FlashAttention-2 build of the hd-64 layout alone (tap layout 7): armed by two calls."""

    def verity_set_tap(self, *a: Any) -> None:
        self.armed.append(("set_tap", *a))

    def verity_set_tap_tag(self, tag: int) -> None:
        self.armed.append(("tag", tag))

    def varlen_fwd(self, *args: Any) -> str:
        return self.launch(*args)


class PortableTap(Hd64Tap):
    """The FlashAttention-2 build that declares its tile (tap layout 8): armed by one call, with a mask of the head dims it compiles."""

    def verity_tap_hdims(self) -> int:
        return self.hdims

    def verity_set_tap_v8(self, *a: Any) -> None:
        self.armed.append(a)


class Fa3Tap(Tap):
    """The FlashAttention-3 build (tap layout 9)."""

    def verity_tap_hdims(self) -> int:
        return self.hdims

    def verity_tap_layout_version(self) -> int:
        return 9

    def verity_set_tap_v9(self, *a: Any) -> None:
        self.armed.append(a)

    def verity_set_tap_off(self) -> None:
        self.armed.append(("off",))

    def fwd(self, *args: Any) -> str:
        return self.launch(*args)


@pytest.fixture
def ops(monkeypatch):
    """`torch.ops` with nothing to load and vLLM's two installed attention kernels, each recording its launches; `put(name, tap)` sets a
    tap build's namespace.  `torch.zeros` allocates on the CPU and records each device it was asked for."""
    loaded: list[str] = []
    devices: list[Any] = []
    installed: list[tuple] = []
    zeros = torch.zeros

    def on_cpu(*a: Any, device: Any = None, **kw: Any) -> Any:
        if device is not None:
            devices.append(device)
        return zeros(*a, **kw)

    def kernel(*args: Any) -> str:
        installed.append(args)
        return "installed"

    def put(name: str, tap: Any) -> Any:
        monkeypatch.setattr(torch.ops, name, tap, raising=False)
        return tap

    monkeypatch.setattr(torch.ops, "load_library", loaded.append)
    monkeypatch.setattr(torch, "zeros", on_cpu)
    fa2, fa3 = put("_vllm_fa2_C", types.SimpleNamespace(varlen_fwd=kernel)), put("_vllm_fa3_C", types.SimpleNamespace(fwd=kernel))
    return types.SimpleNamespace(put=put, loaded=loaded, devices=devices, installed=installed, kernel=kernel, fa2=fa2, fa3=fa3)


def _fa2(ops: Any, tap: Tap | None = None) -> tuple[Committer, Any, Any]:
    com, tap = Committer(), ops.put("verity_fa2_matReq", tap or PortableTap())
    return com, tap, H.HiddenM1Source(com, FA2_SO, ns="verity_fa2_matReq")


def _fa3(ops: Any, tap: Tap | None = None) -> tuple[Committer, Any, Any]:
    com, tap = Committer(), ops.put("verity_fa3_matReq", tap or Fa3Tap())
    return com, tap, H.HiddenM1SourceFA3(com, FA3_SO)


def _queries(b: int, heads: int, kv_heads: int, head_dim: int, max_q: int) -> tuple[Any, Any, Any]:
    """(q, the paged k, cu_seqlens_q) of `b` sequences of `max_q` query rows each."""
    return (torch.zeros(b * max_q, heads, head_dim), torch.zeros(4, 16, kv_heads, head_dim),
            torch.arange(b + 1, dtype=torch.int32) * max_q)


def _fa2_args(*, b: int = 2, heads: int = 4, kv_heads: int = 2, head_dim: int = 64, max_q: int = 5, max_k: int = 130,
              window: tuple[int, int] = (-1, -1), num_splits: int = 1) -> tuple:
    """A launch of `_vllm_fa2_C.varlen_fwd`, positional as vLLM makes it."""
    q, k, cu_q = _queries(b, heads, kv_heads, head_dim, max_q)
    # q, k, v, out, cu_seqlens_q, cu_seqlens_k, seqused_k, leftpad_k, block_table, alibi_slopes, max_seqlen_q, max_seqlen_k, dropout_p,
    # softmax_scale, zero_tensors, causal, window_left, window_right, softcap, return_softmax, num_splits, gen
    return (q, k, k, None, cu_q, None, None, None, None, None, max_q, max_k, 0.0, 1.0, False, True, window[0], window[1], 0.0, False,
            num_splits, None)


def _fa3_args(*, b: int = 2, heads: int = 4, kv_heads: int = 2, head_dim: int = 64, max_q: int = 5, max_k: int = 130,
              paged: bool = True, num_splits: int | None = 1) -> tuple:
    """A launch of `_vllm_fa3_C.fwd`: 31 positions, of which the wrapper reads the eight the source names."""
    F = H.HiddenM1SourceFA3
    q, k, cu_q = _queries(b, heads, kv_heads, head_dim, max_q)
    args: list[Any] = [None] * 31
    args[F._Q], args[F._K], args[F._CU_Q], args[F._MAXQ], args[F._MAXK] = q, k, cu_q, max_q, max_k
    args[F._PAGE_TABLE], args[F._NUM_SPLITS] = (torch.zeros(1) if paged else None), num_splits
    return tuple(args)


def _words(HB: int, M: int, NB: int, BN: int, D: int, ms: bool = False) -> int:
    """The stream buffer's words: the layout's, rounded up to whole chunks."""
    return -(-H.mat_total_words(HB, M, NB, BN, D, ms) // H.CHUNK_WORDS) * H.CHUNK_WORDS


# --- FlashAttention-2 ------------------------------------------------------------------------------------------------------------------------

def test_the_fa2_source_reads_the_build_it_loads(ops):
    _com, tap, src = _fa2(ops, PortableTap(src=SRC_MS, hdims=HD64))
    assert ops.loaded == [FA2_SO] and src.ns is tap and src.installed_ns is ops.fa2 and src.orig_op is ops.kernel
    assert (src.mode, src.src, src.v8, src.ms, src.active) == (3, SRC_MS, True, True, False)
    assert src.stats["tap_layout"] == 8 and src.stats["hdims"] == [64] and src.stats["row3"] == 0 and src.stats["ms"] is True
    hd64 = _fa2(ops, Hd64Tap())[2]
    assert not hd64.v8 and hd64.stats["tap_layout"] == 7 and hd64.stats["hdims"] == [64] and hd64.ms is False
    with pytest.raises(RuntimeError, match=r"verity_fa2_matReq: compiled tap mode 2 != 3 \(M1\)"):
        _fa2(ops, PortableTap(mode=2))


def test_outside_a_committed_step_the_fa2_wrapper_launches_the_installed_kernel(ops):
    com, tap, src = _fa2(ops)
    args = _fa2_args()
    assert src._wrapper()(*args) == "installed" and ops.installed == [args]
    assert tap.armed == [] and tap.launches == [] and com.acquired == [] and ops.devices == [] and src.launch == 0


def test_a_fa2_launch_of_a_step_arms_the_tap_with_its_geometry_and_commits_the_stream_buffer(ops):
    com, tap, src = _fa2(ops)
    fwd = src._wrapper()
    com._cur = types.SimpleNamespace(step=4)
    prefill, decode, windowed = _fa2_args(), _fa2_args(max_q=1), _fa2_args(max_q=1, window=(3, 0))
    assert [fwd(*a) for a in (prefill, decode, windowed)] == ["tapped"] * 3 and tap.launches == [prefill, decode, windowed]
    assert ops.installed == [] and com.errors == [] and ops.devices == ["cuda"] * 3
    # b = 2 sequences of 4 heads over 2 KV heads, 130 keys in blocks of 128: slabs of (sequence, head) with one row per query; a decode
    # launch without a window folds each KV group's heads into the rows; with a window it does not
    geometries = [(8, 5, 2), (4, 2, 2), (8, 1, 2)]
    assert [name for name, _t in com.acquired] == [f"fa2.m1/L{i}/g{HB}x{M}x{NB}" for i, (HB, M, NB) in enumerate(geometries)]
    bufs = [t for _name, t in com.acquired]
    assert [(t.dtype, t.numel()) for t in bufs] == [(torch.int32, _words(*g, 128, 64)) for g in geometries]
    assert len(tap.armed) == 3 and all(a[1] is t for a, t in zip(tap.armed, bufs, strict=True))
    assert [(a[0], *a[2:]) for a in tap.armed] == [(3, *g, 128, 64, tag) for tag, g in enumerate(geometries)]
    nbytes = [4 * t.numel() for t in bufs]
    assert (src.launch, src.launches_in_step, src.stats["launches"], src.stats["bytes"]) == (3, 3, 3, sum(nbytes))
    assert src.stats["geometries"] == {str(g): n for g, n in zip(geometries, nbytes, strict=True)}
    assert com.timed == [("acquisition", 4, n) for n in nbytes]
    src.on_step_begin()                                         # the next step: its streams are named from 0, the tap's tag runs on
    fwd(*prefill)
    assert com.acquired[-1][0] == "fa2.m1/L0/g8x5x2" and com.acquired[-1][1] is bufs[0] and tap.armed[-1][-1] == 3
    assert ops.devices == ["cuda"] * 3                          # one buffer per geometry
    [d] = src.declaration_classes()
    assert (d["class"], d["how"], d["tensors"], d["bytes"], d["tap_layout"], d["hdims"]) == (
        "fa2_hidden_m1_stream", "bound-as-leaf", 4, sum(nbytes) + nbytes[0], 8, [64, 128])
    assert "row3" not in d and "ms" not in d


def test_a_fa2_tile_other_than_the_hd64_one_is_named_in_its_stream_and_a_build_with_ms_commits_a_longer_one(ops):
    com, tap, src = _fa2(ops, PortableTap(src=SRC_MS))
    com._cur = types.SimpleNamespace(step=0)
    fwd = src._wrapper()
    fwd(*_fa2_args(head_dim=128))
    fwd(*_fa2_args())
    assert [name for name, _t in com.acquired] == ["fa2.m1/L0/g8x5x3x64x128", "fa2.m1/L1/g8x5x2"]      # 130 keys in blocks of 64, of 128
    assert [t.numel() for _name, t in com.acquired] == [_words(8, 5, 3, 64, 128, ms=True), _words(8, 5, 2, 128, 64, ms=True)]
    assert com.acquired[1][1].numel() > _words(8, 5, 2, 128, 64) and tap.armed[0][2:] == (8, 5, 3, 64, 128, 0)
    assert list(src.stats["geometries"]) == ["(8, 5, 3, 64, 128)", "(8, 5, 2)"]
    assert src.declaration_classes()[0]["ms"] == "max * scale per key block"


@pytest.mark.parametrize(("launch", "why"), [
    ({"head_dim": 128}, r"fa2 launch 0: fa2 tap verity_fa2_matReq: head_dim 128 not compiled \(hdims \[64\]\)"),
    ({"num_splits": 2}, r"fa2 launch 0: fa2 tap: num_splits=2 \(only the num_splits==1 batch-invariant kernel is tapped\)"),
])
def test_a_fa2_launch_the_build_has_no_kernel_for_is_an_error_and_an_expected_row_under_the_installed_kernel(ops, launch, why):
    com, tap, src = _fa2(ops, PortableTap(hdims=HD64))
    com._cur = types.SimpleNamespace(step=0)
    args = _fa2_args(**launch)
    assert src._wrapper()(*args) == "installed" and ops.installed == [args]
    assert com.x03_expected == 1 and len(com.errors) == 1 and tap.armed == [] and com.acquired == [] and src.launch == 0
    assert re.match(why, com.errors[0])


def test_a_fa2_launch_the_kernel_skips_is_an_error_and_an_expected_row_and_commits_nothing(ops):
    com, tap, src = _fa2(ops)
    com._cur = types.SimpleNamespace(step=0)
    tap.skips = True
    fwd = src._wrapper()
    assert fwd(*_fa2_args()) == "tapped" and com.acquired == [] and com.timed == [] and com.x03_expected == 1
    assert com.errors == ["fa2 launch 0: skipped by the kernel (declared tile BN=128 D=64, HB=8 M=5 NB=2, num_splits=1); stream not bound"]
    assert (src.launch, src.launches_in_step, src.stats["launches"], src.stats["bytes"]) == (1, 1, 0, 0)
    tap.skips = False
    fwd(*_fa2_args())                                           # the launch after it: the step's second stream, the tap's second tag
    assert [name for name, _t in com.acquired] == ["fa2.m1/L1/g8x5x2"] and tap.armed[-1][-1] == 1 and com.x03_expected == 1


def test_the_hd64_only_build_is_armed_by_its_two_calls(ops):
    com, tap, src = _fa2(ops, Hd64Tap())
    com._cur = types.SimpleNamespace(step=0)
    assert src._wrapper()(*_fa2_args()) == "tapped"
    [(name, buf)] = com.acquired
    assert name == "fa2.m1/L0/g8x5x2" and tap.armed[0][3] is buf
    assert [(a[:3], a[4:]) if a[0] == "set_tap" else a for a in tap.armed] == [(("set_tap", 3, None), (8, 5, 0, 2)), ("tag", 0)]


def test_the_fa2_source_installed_replaces_the_op_and_uninstalled_restores_it_and_disarms_the_tap(ops):
    com, tap, src = _fa2(ops)
    live = hooks.live()
    src.install()
    assert src.active and ops.fa2.varlen_fwd is not ops.kernel and hooks.live() != live
    args = _fa2_args()
    assert torch.ops._vllm_fa2_C.varlen_fwd(*args) == "installed"                     # no step: through the wrapper to the kernel
    com._cur = types.SimpleNamespace(step=0)
    assert torch.ops._vllm_fa2_C.varlen_fwd(*args) == "tapped" and len(com.acquired) == 1
    tap.skipped = 7
    src.uninstall()
    assert not src.active and ops.fa2.varlen_fwd is ops.kernel and hooks.live() == live
    assert src.stats["skipped"] == 7 and tap.armed[-1] == ("set_tap", 0, None, None, 0, 0, 0, 0) and com.errors == []
    src.uninstall()                                             # a second time: nothing to put back
    assert ops.fa2.varlen_fwd is ops.kernel and hooks.live() == live


def test_a_skipped_count_that_cannot_be_read_at_uninstall_is_a_collector_error(ops):
    class Unread(PortableTap):
        def verity_tap_skipped(self) -> int:
            raise RuntimeError("the build is gone")

    com, tap, src = _fa2(ops, Unread())
    src.install()
    src.uninstall()
    assert com.errors == ["fa2 tap skipped count: RuntimeError('the build is gone')"] and src.stats["skipped"] is None
    assert ops.fa2.varlen_fwd is ops.kernel and tap.armed[-1][:2] == ("set_tap", 0) and not src.active


# --- FlashAttention-3 ------------------------------------------------------------------------------------------------------------------------

def test_the_fa3_source_reads_the_build_it_loads_and_refuses_one_that_is_not_fa3s(ops):
    _com, tap, src = _fa3(ops)
    assert ops.loaded == [FA3_SO] and src.ns is tap and src.installed_ns is ops.fa3 and src.orig_op is ops.kernel
    assert (src.mode, src.src, src.v8, src.ms, src.active) == (3, SRC, True, False, False)
    assert src.stats["tap_layout"] == 9 and src.stats["hdims"] == [64, 128] and src.stats["backend"] == "fa3" and src.stats["tiles"] == {}
    with pytest.raises(RuntimeError, match=r"verity_fa3_matReq: not an FA3 \(v9\) tap build"):
        _fa3(ops, PortableTap())
    with pytest.raises(RuntimeError, match=r"verity_fa3_matReq: compiled tap mode 1 != 3 \(M1\)"):
        _fa3(ops, Fa3Tap(mode=1))


def test_outside_a_committed_step_the_fa3_wrapper_launches_the_installed_kernel(ops):
    com, tap, src = _fa3(ops)
    args = _fa3_args()
    assert src._wrapper()(*args) == "installed" and ops.installed == [args]
    assert tap.armed == [] and tap.launches == [] and com.acquired == [] and ops.devices == []


def test_a_fa3_launch_of_a_step_arms_the_tap_with_its_packed_geometry_and_commits_the_stream_buffer(ops):
    com, tap, src = _fa3(ops)
    fwd = src._wrapper()
    com._cur = types.SimpleNamespace(step=2)
    one_wg, two_wg, hd128, no_splits = _fa3_args(), _fa3_args(max_q=40), _fa3_args(head_dim=128), _fa3_args(num_splits=None)
    assert [fwd(*a) for a in (one_wg, two_wg, hd128, no_splits)] == ["tapped"] * 4
    assert tap.launches == [one_wg, two_wg, hd128, no_splits] and ops.installed == [] and com.errors == []
    # slabs of (sequence, KV head), rows of (query row, head of the group); the tile is the kernel's for the packed rows and head dim
    geometries = [(4, 10, 1, 192, 64), (4, 80, 2, 128, 64), (4, 10, 2, 128, 128), (4, 10, 1, 192, 64)]
    assert [name for name, _t in com.acquired] == [f"fa3.m1/L{i}/g{HB}x{M}x{NB}x{BN}x{D}" for i, (HB, M, NB, BN, D) in enumerate(geometries)]
    bufs = [t for _name, t in com.acquired]
    assert [t.numel() for t in bufs] == [_words(*g) for g in geometries] and bufs[3] is bufs[0] and ops.devices == ["cuda"] * 3
    assert all(a[1] is t for a, t in zip(tap.armed, bufs, strict=True))
    assert [(a[0], *a[2:]) for a in tap.armed] == [(3, *g, tag) for tag, g in enumerate(geometries)]
    nbytes = [4 * t.numel() for t in bufs]
    assert (src.launch, src.launches_in_step, src.stats["launches"], src.stats["bytes"]) == (4, 4, 4, sum(nbytes))
    assert src.stats["tiles"] == {"64x192": 2, "192x128": 1, "64x128": 1}
    assert src.stats["geometries"] == {str(g): n for g, n in zip(geometries[:3], nbytes[:3], strict=True)}
    assert com.timed == [("acquisition", 2, n) for n in nbytes]
    [d] = src.declaration_classes()
    assert (d["class"], d["tensors"], d["bytes"], d["tap_layout"], d["hdims"], d["tiles"]) == (
        "fa3_hidden_m1_stream", 4, sum(nbytes), 9, [64, 128], src.stats["tiles"])
    assert d["tiles"] is not src.stats["tiles"] and "row3" not in d and "ms" not in d


@pytest.mark.parametrize(("launch", "why"), [
    ({"paged": False}, r"fa3 launch 0: not the varlen paged-KV launch the tap covers"),
    ({"head_dim": 128}, r"fa3 launch 0: fa3 tap verity_fa3_matReq: head_dim 128 not compiled \(hdims \[64\]\)"),
    ({"num_splits": 2}, r"fa3 launch 0: fa3 tap: num_splits=2 \(only the num_splits==1 batch-invariant kernel is tapped\)"),
])
def test_a_fa3_launch_the_tap_does_not_cover_is_an_error_and_an_expected_row_under_the_installed_kernel(ops, launch, why):
    com, tap, src = _fa3(ops, Fa3Tap(hdims=HD64))
    com._cur = types.SimpleNamespace(step=0)
    args = _fa3_args(**launch)
    assert src._wrapper()(*args) == "installed" and ops.installed == [args]
    assert com.x03_expected == 1 and len(com.errors) == 1 and tap.armed == [] and com.acquired == [] and src.launch == 0
    assert re.match(why, com.errors[0])


def test_a_fa3_launch_the_kernel_skips_is_an_error_and_an_expected_row_and_commits_nothing(ops):
    com, tap, src = _fa3(ops)
    com._cur = types.SimpleNamespace(step=0)
    tap.skips = True
    fwd = src._wrapper()
    assert fwd(*_fa3_args()) == "tapped" and com.acquired == [] and com.timed == [] and com.x03_expected == 1
    skipped = "fa3 launch 0: skipped by the kernel (declared tile kBlockM=64 BN=192 D=64, HB=4 M=10 NB=1, num_splits=1); stream not bound"
    assert com.errors == [skipped]
    assert (src.launch, src.launches_in_step, src.stats["launches"], src.stats["tiles"]) == (1, 1, 0, {})
    tap.skips = False
    fwd(*_fa3_args())
    assert [name for name, _t in com.acquired] == ["fa3.m1/L1/g4x10x1x192x64"] and tap.armed[-1][-1] == 1 and com.x03_expected == 1


def test_the_fa3_source_installed_replaces_the_op_and_uninstalled_restores_it_and_turns_the_tap_off(ops):
    com, tap, src = _fa3(ops)
    live = hooks.live()
    src.install()
    assert src.active and ops.fa3.fwd is not ops.kernel and ops.fa2.varlen_fwd is ops.kernel
    com._cur = types.SimpleNamespace(step=0)
    assert torch.ops._vllm_fa3_C.fwd(*_fa3_args()) == "tapped" and len(com.acquired) == 1
    tap.skipped = 2
    src.uninstall()
    assert not src.active and ops.fa3.fwd is ops.kernel and hooks.live() == live
    assert src.stats["skipped"] == 2 and tap.armed[-1] == ("off",) and com.errors == []


# --- the source for a build --------------------------------------------------------------------------------------------------------------------

def test_the_source_made_for_a_build_is_its_attention_backends_and_another_backends_build_is_refused(ops):
    ops.put("verity_fa2_matReq", PortableTap())
    ops.put("verity_fa3_matReq", Fa3Tap())
    com = Committer()
    fa2, fa3 = H.make_hidden_source(com, FA2_SO), H.make_hidden_source(com, FA3_SO, flash_attn_version=3)
    assert type(fa2) is H.HiddenM1Source and type(fa3) is H.HiddenM1SourceFA3
    assert (fa2.ns_name, fa3.ns_name) == ("verity_fa2_matReq", "verity_fa3_matReq")           # the build file's stem
    assert type(H.make_hidden_source(com, FA2_SO, flash_attn_version=2, manifest={"query": {}})) is H.HiddenM1Source
    with pytest.raises(RuntimeError, match=r"is an FA2 build but the engine executes FA3: use verity_fa3_matReq.so"):
        H.make_hidden_source(com, FA2_SO, flash_attn_version=3)
    with pytest.raises(RuntimeError, match=r"is an FA3 build but the engine executes FA2: use verity_fa2_matReq.so"):
        H.make_hidden_source(com, FA3_SO, flash_attn_version=2)
    with pytest.raises(RuntimeError, match=r"ROW word 3 mode 0 .*guarded-max policy and needs mode 1"):
        H.make_hidden_source(com, FA2_SO, manifest={"query": {"guarded_max": {"words": 5}}})

    class Native(H.HiddenM1Source):
        """A committer's own source for the build."""

    assert type(H.make_hidden_source(com, FA2_SO, fa2_cls=Native, fa3_cls=Native)) is Native
    ops.put("named", Fa3Tap())
    assert type(H.make_hidden_source(com, FA3_SO, ns="named", fa2_cls=Native)) is H.HiddenM1SourceFA3
