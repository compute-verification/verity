"""`tree_allreduce`: the serving node's tree parsed as its probe prints it, the gathered words summed in that tree's f32 order and
rounded once to bf16 (against numpy's IEEE f32 adds, and refused by an order that differs), FlashInfer's route under ``mnnvl``
and other dtypes left to vLLM, FlashInfer's calls under another backend summed as ``mnnvl`` sums them, the custom all-reduce
given the serving node's cap on a node without symmetric memory, FlashInfer's fused all-reduce + RMSNorm under another backend
computed as ``mnnvl``'s kernels compute it (against a thread-by-thread model of them, with the launch FlashInfer picks), and the
plugin's site registering it."""
from __future__ import annotations

import functools
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TA = importlib.import_module("tree_allreduce")

NODE_A = "(((((((0,2),3),1),5),6),4),7)"
NCCL_SYMM = {"on": False}
FI_WS = "vllm.distributed.device_communicators.flashinfer_all_reduce"
MiB = 1 << 20


@pytest.fixture(autouse=True)
def _pynccl_allocator(monkeypatch):
    """vLLM's `pynccl_allocator.is_symmetric_memory_enabled`, answering ``NCCL_SYMM["on"]``."""
    NCCL_SYMM["on"] = False
    mod = types.ModuleType("vllm.distributed.device_communicators.pynccl_allocator")
    mod.is_symmetric_memory_enabled = lambda: NCCL_SYMM["on"]
    monkeypatch.setitem(sys.modules, "vllm.distributed.device_communicators.pynccl_allocator", mod)


@pytest.fixture(autouse=True)
def _fi_workspace(monkeypatch):
    """vLLM's FlashInfer all-reduce module with the serving node's ``mnnvl`` workspace; a test sets ``_fi_ar_workspace``."""
    mod = types.ModuleType(FI_WS)
    mod._fi_ar_workspace = SimpleNamespace(backend="mnnvl")
    monkeypatch.setitem(sys.modules, FI_WS, mod)
    return mod


def _bf16(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << 16).view(np.float32)


def _rne(x: np.ndarray) -> np.ndarray:
    u = x.view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _reference(words: np.ndarray, order: list[int]) -> np.ndarray:
    """f32 adds left to right in ``order`` (a left-deep tree), then RNE to bf16."""
    acc = _bf16(words[order[0]])
    for r in order[1:]:
        acc = (acc + _bf16(words[r])).astype(np.float32)
    return _rne(acc)


def _torch_words(words: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(words.astype(np.uint16).view(np.int16).copy()).view(torch.bfloat16)


def _words(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def test_the_probes_tree_parses_to_its_leaf_order_and_a_malformed_one_is_refused():
    t = TA.parse_tree(NODE_A)
    assert TA.leaves(t) == [0, 2, 3, 1, 5, 6, 4, 7]
    for bad in ("(0,1,2)", "((0,1),'a')", "[0,1]"):
        with pytest.raises(ValueError):
            TA.parse_tree(bad)


def test_gathered_words_sum_in_the_trees_f32_order_and_round_once():
    rng = np.random.default_rng(7)
    n = 4096
    exp = rng.integers(0x01, 0xFE, size=(8, n), dtype=np.uint16)          # normals of every exponent
    words = (rng.integers(0, 2, size=(8, n), dtype=np.uint16) << 15) | (exp << 7) | rng.integers(0, 128, size=(8, n), dtype=np.uint16)
    words[:, :64] = rng.integers(0, 0x80, size=(8, 64), dtype=np.uint16)  # subnormals, which the tree keeps (FTZ off)
    got = _words(TA.reduce_gathered(_torch_words(words).reshape(-1), 8, TA.parse_tree(NODE_A)))
    assert np.array_equal(got, _reference(words, [0, 2, 3, 1, 5, 6, 4, 7]))


def test_a_tree_with_right_subtrees_sums_each_in_a_row_of_its_own_into_the_given_output():
    def ref(words: np.ndarray, t) -> np.ndarray:
        return _bf16(words[t]) if isinstance(t, int) else (ref(words, t[0]) + ref(words, t[1])).astype(np.float32)
    rng = np.random.default_rng(11)
    words = rng.integers(0x3000, 0x4800, size=(8, 64), dtype=np.uint16)
    t = TA.parse_tree("((0,(1,(2,3))),((4,5),(6,7)))")
    assert TA.buffers(t) == 3 and TA.buffers(TA.parse_tree(NODE_A)) == 1
    out, f32 = torch.empty(64, dtype=torch.bfloat16), torch.full((3, 100), float("nan"))
    assert TA.reduce_gathered(_torch_words(words).reshape(-1), 8, t, out, f32) is out
    assert np.array_equal(_words(out), _rne(ref(words, t)))


def test_another_order_gives_other_bits_so_the_check_can_fail():
    """1e8 + 1 - 1e8 in f32: rank order keeps none of the 1s that the tree's order adds after the large ones cancel."""
    vals = np.array([3e8, 1.0, -3e8, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    words = _rne(vals)[:, None].repeat(4, axis=1)
    tree = _words(TA.reduce_gathered(_torch_words(words).reshape(-1), 8, TA.parse_tree(NODE_A)))
    ranks = _words(TA.reduce_gathered(_torch_words(words).reshape(-1), 8, TA.parse_tree("(((((((0,1),2),3),4),5),6),7)")))
    assert np.array_equal(tree, _reference(words, [0, 2, 3, 1, 5, 6, 4, 7]))
    assert not np.array_equal(tree, ranks)


class _Comm:
    """`CudaCommunicator` over 8 ranks whose inputs are ``inputs``: FlashInfer's route takes calls up to ``fi_max`` words, the custom
    all-reduce those of exactly ``ca_eq`` words, torch's symmetric memory is on as on the serving node, and pynccl's all-gather
    lays the ranks' words out rank-major (``gathers`` counts its calls)."""

    def __init__(self, inputs: list[torch.Tensor], fi_max: int, world: int = 8, ca_eq: int = -1) -> None:
        self.world_size, self.inputs, self.gathers = world, inputs, 0
        self.fi_ar_comm = SimpleNamespace(disabled=False, should_use_fi_ar=lambda x: x.numel() <= fi_max)
        self.ca_comm = SimpleNamespace(disabled=False, max_size=8 * MiB, should_custom_ar=lambda x: x.numel() == ca_eq)
        self.symm_mem_comm = SimpleNamespace(disabled=False)
        self.pynccl_comm = SimpleNamespace(disabled=False, all_gather=self._gather)
        self.qr_comm = self.fi_pcie_ipc_ar_comm = self.aiter_ar_comm = None
        self.at = 0

    def _gather(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        n = inp.numel()
        out.copy_(torch.cat([x.reshape(-1)[self.at:self.at + n] for x in self.inputs]))
        self.at, self.gathers = self.at + n, self.gathers + 1


def test_the_wrapper_leaves_flashinfers_and_the_custom_routes_and_other_dtypes_to_vllm_and_sums_the_rest_in_the_tree():
    rng = np.random.default_rng(3)
    words = rng.integers(0x3000, 0x4800, size=(8, 2, 6), dtype=np.uint16)
    inputs = [_torch_words(w) for w in words]
    seen = []
    ar = TA.wrap(lambda self, x: seen.append(x.shape) or x, TA.parse_tree(NODE_A))
    assert ar(_Comm(inputs, fi_max=12), inputs[5]) is inputs[5]
    assert ar(_Comm(inputs, fi_max=4, ca_eq=12), inputs[5]) is inputs[5] and seen == [(2, 6), (2, 6)]
    out = ar(_Comm(inputs, fi_max=4), inputs[5])
    assert out.shape == (2, 6) and out.dtype == torch.bfloat16
    assert np.array_equal(_words(out).reshape(-1), _reference(words.reshape(8, -1), [0, 2, 3, 1, 5, 6, 4, 7]))
    f32 = torch.ones(2, 6)
    assert ar(_Comm(inputs, fi_max=4), f32) is f32
    with pytest.raises(RuntimeError, match="not over the group's 4 ranks"):
        ar(_Comm(inputs[:4], fi_max=4, world=4), inputs[0])


def test_a_call_is_gathered_in_chunks_with_the_same_bits_into_one_workspace_allocated_once(monkeypatch):
    rng = np.random.default_rng(5)
    words = rng.integers(0x3000, 0x4800, size=(8, 3, 7), dtype=np.uint16)
    inputs = [_torch_words(w) for w in words]
    monkeypatch.setattr(TA, "CHUNK", 5)
    comm = _Comm(inputs, fi_max=4)
    ar = TA.wrap(None, TA.parse_tree(NODE_A))
    out = ar(comm, inputs[2])
    assert comm.gathers == 5
    assert np.array_equal(_words(out).reshape(-1), _reference(words.reshape(8, -1), [0, 2, 3, 1, 5, 6, 4, 7]))
    ws = comm._tree_ws
    assert ws[0].shape == (8 * 5,) and ws[1].shape == (1, 5)
    comm.at = 0
    assert np.array_equal(_words(ar(comm, inputs[2])), _words(out)) and comm._tree_ws is ws


def test_a_route_the_tree_was_not_measured_with_is_refused():
    inputs = [torch.ones(4, dtype=torch.bfloat16)] * 8
    ar = TA.wrap(None, TA.parse_tree(NODE_A))
    comm = _Comm(inputs, fi_max=0)
    comm.qr_comm = SimpleNamespace(disabled=False)
    with pytest.raises(RuntimeError, match="qr_comm on"):
        ar(comm, inputs[0])
    NCCL_SYMM["on"] = True
    with pytest.raises(RuntimeError, match="NCCL's symmetric memory on"):
        ar(_Comm(inputs, fi_max=0), inputs[0])


def _from_zero(words: np.ndarray) -> np.ndarray:
    """``mnnvl``'s sum: f32 adds onto +0 in rank order, then RNE to bf16."""
    acc = np.zeros(words.shape[1], dtype=np.float32)
    for r in range(words.shape[0]):
        acc = (acc + _bf16(words[r])).astype(np.float32)
    return _rne(acc)


def test_reduce_from_zero_adds_the_ranks_in_order_onto_plus_zero_and_differs_from_a_rank_order_tree_only_in_a_zeros_sign():
    rng = np.random.default_rng(13)
    exp = rng.integers(0x01, 0xFE, size=(8, 512), dtype=np.uint16)
    words = (rng.integers(0, 2, size=(8, 512), dtype=np.uint16) << 15) | (exp << 7) | rng.integers(0, 128, size=(8, 512), dtype=np.uint16)
    words[:, :32] = rng.integers(0, 0x80, size=(8, 32), dtype=np.uint16)
    words[:, 40] = 0x8000
    out, f32 = torch.empty(512, dtype=torch.bfloat16), torch.full((1, 600), float("nan"))
    assert TA.reduce_from_zero(_torch_words(words).reshape(-1), 8, out, f32) is out
    got = _words(out)
    assert np.array_equal(got, _from_zero(words)) and got[40] == 0x0000
    ranks = _words(TA.reduce_gathered(_torch_words(words).reshape(-1), 8, TA.parse_tree("(((((((0,1),2),3),4),5),6),7)")))
    assert ranks[40] == 0x8000
    assert np.array_equal(np.delete(got, 40), np.delete(ranks, 40))


def test_flashinfers_calls_are_summed_as_mnnvl_sums_them_when_its_workspace_has_another_backend(_fi_workspace, capsys):
    rng = np.random.default_rng(17)
    words = rng.integers(0x3000, 0x4800, size=(8, 2, 6), dtype=np.uint16)
    inputs = [_torch_words(w) for w in words]
    ar = TA.wrap(lambda self, x: x, TA.parse_tree(NODE_A))
    assert ar(_Comm(inputs, fi_max=12), inputs[5]) is inputs[5]
    _fi_workspace._fi_ar_workspace = SimpleNamespace(backend="trtllm")
    comm = _Comm(inputs, fi_max=12)
    out = ar(comm, inputs[5])
    assert out.shape == (2, 6) and comm.gathers == 1
    assert np.array_equal(_words(out).reshape(-1), _from_zero(words.reshape(8, -1)))
    comm.at = 0
    ar(comm, inputs[5])
    assert capsys.readouterr().err.count("workspace is trtllm on this node") == 1
    _fi_workspace._fi_ar_workspace = None
    with pytest.raises(RuntimeError, match="no all-reduce workspace"):
        ar(_Comm(inputs, fi_max=12), inputs[5])


def test_without_symmetric_memory_the_custom_all_reduce_gets_the_serving_nodes_cap(monkeypatch, capsys):
    sizes = types.ModuleType("vllm.distributed.device_communicators.all_reduce_utils")
    sizes.CUSTOM_ALL_REDUCE_MAX_SIZES = {"9.0": {2: 64 * MiB, 4: 32 * MiB, 6: MiB // 2, 8: MiB // 4}}
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(get_device_capability=lambda: SimpleNamespace(as_version_str=lambda: "9.0"))
    monkeypatch.setitem(sys.modules, sizes.__name__, sizes)
    monkeypatch.setitem(sys.modules, platforms.__name__, platforms)
    inputs = [torch.ones(2, 6, dtype=torch.bfloat16)] * 8

    def comm(symm=None, **ca):
        c = _Comm(inputs, fi_max=12)
        c.symm_mem_comm = symm
        c.ca_comm.__dict__.update(ca)
        return c
    for c, cap in ((comm(), MiB // 4), (comm(SimpleNamespace(disabled=True)), MiB // 4),
                   (comm(SimpleNamespace(disabled=False)), None), (comm(disabled=True), None), (comm(max_size=MiB // 4), None)):
        assert TA.served_custom_cap(c) == cap and c.ca_comm.max_size == (cap or c.ca_comm.max_size)
    c = comm()
    c.world_size = 3
    assert TA.served_custom_cap(c) is None and c.ca_comm.max_size == 8 * MiB
    c = comm()
    c.ca_comm = None
    assert TA.served_custom_cap(c) is None
    c = comm()
    ar = TA.wrap(lambda self, x: x, TA.parse_tree(NODE_A))
    ar(c, inputs[0])
    ar(c, inputs[0])
    assert c.ca_comm.max_size == MiB // 4 and capsys.readouterr().err.count("capped at 262144 bytes") == 1


DIM = 6144


def _oneshot(t: int) -> bool:
    """``mnnvl``'s AUTO strategy at TP8: one-shot while ``t * dim * world * 2`` bytes fit `MNNVL_ONE_SHOT_THRESHOLD`."""
    return t * DIM * 8 * 2 <= 64 * 1024 * 8 * 2


def test_the_launch_is_flashinfers_for_every_row_count_vllms_fusion_takes():
    for t in range(1, 43):
        want = (96, 8) if t <= 16 else (192, 4) if t <= 33 else (384, 2)
        assert TA.mnnvl_grid(t, DIM, 132) == want, t
    assert TA.mnnvl_grid(66, DIM, 132) == (384, 2) and TA.mnnvl_grid(67, DIM, 132) == (768, 1)
    for dim in (6000, 4104):
        with pytest.raises(NotImplementedError):
            TA.mnnvl_grid(4, dim, 132)


def _shfl_xor_sum(lanes: np.ndarray) -> np.float32:
    """32 lanes each doing ``val += __shfl_xor_sync(val, mask)`` for mask 16..1; every lane ends equal."""
    v = lanes.astype(np.float32)
    for mask in (16, 8, 4, 2, 1):
        v = (v + v[np.arange(32) ^ mask]).astype(np.float32)
    assert (v == v[0]).all()
    return v[0]


def _scale_np(s, dim: int, eps: float):
    return np.float32((np.float64(s) / dim + eps) ** -0.5)


def _scale_torch(s, dim: int, eps: float):
    return torch.from_numpy(np.array([_scale_np(x, dim, eps) for x in s.numpy()], dtype=np.float32))


def _kernel(ar: np.ndarray, res: np.ndarray, gamma: np.ndarray, oneshot: bool, bf16_squares: bool | None = None,
            sequential: bool = False) -> tuple:
    """``mnnvl``'s fused kernels thread by thread: thread ``p`` of a row's cluster holds 8 elements, sums their squares from +0,
    lanes butterfly within a warp, warp 0 butterflies the block's warp sums (zeros past them), and the cluster's block sums add
    in order. Returns the normed rows, the residual and each row's f32 sum of squares. ``bf16_squares`` (one-shot's rounding
    by default) and ``sequential`` (one running sum) put other arithmetic in, to show the check can fail."""
    t = ar.shape[0]
    block, cluster = TA.mnnvl_grid(t, DIM, 132)
    h = _rne((_bf16(ar) + _bf16(res)).astype(np.float32))
    hv, g = _bf16(h), (_bf16(gamma) + np.float32(0)).astype(np.float32)
    exact = (hv * hv).astype(np.float32)
    sq = _bf16(_rne(exact)) if (oneshot if bf16_squares is None else bf16_squares) else exact
    out, fulls = np.empty_like(h), np.empty(t, dtype=np.float32)
    for row in range(t):
        full = np.float32(0)
        if sequential:
            for x in sq[row]:
                full = np.float32(full + x)
        else:
            threads = sq[row].reshape(cluster * block, 8)
            ts = np.zeros(cluster * block, dtype=np.float32)
            for j in range(8):
                ts = (ts + threads[:, j]).astype(np.float32)
            for b in range(cluster):
                warps = [_shfl_xor_sum(w) for w in ts[b * block:(b + 1) * block].reshape(block // 32, 32)]
                smem = np.zeros(32, dtype=np.float32)
                smem[:len(warps)] = warps
                full = np.float32(full + _shfl_xor_sum(smem))
        fulls[row] = full
        r = _scale_np(full, DIM, 1e-5)
        if oneshot:
            v = ((hv[row] * r).astype(np.float32) * g).astype(np.float32)
        else:
            v = (r * (hv[row] * g).astype(np.float32)).astype(np.float32)
        out[row] = _rne(v)
    return out, h, fulls


def _row_words(rng, t: int, scale: float = 0.3) -> np.ndarray:
    return _rne((rng.standard_normal((t, DIM)) * scale).astype(np.float32))


def test_the_fused_norm_is_the_kernels_arithmetic_at_each_launch_and_strategy():
    rng = np.random.default_rng(19)
    gamma = _rne((rng.random(DIM) + 0.5).astype(np.float32))
    for t in (3, 10, 11, 21, 40):
        ar, res = _row_words(rng, t, 2.0), _row_words(rng, t)
        sums = []

        def scale(s, dim, eps):
            sums.append(s.numpy().copy())
            return _scale_torch(s, dim, eps)
        normed, h = TA.mnnvl_add_rmsnorm(_torch_words(ar), _torch_words(res), _torch_words(gamma), 1e-5, 0.0, _oneshot(t), 132,
                                         scale)
        want, want_h, fulls = _kernel(ar, res, gamma, _oneshot(t))
        assert np.array_equal(_words(h), want_h) and np.array_equal(_words(normed), want), t
        assert np.array_equal(sums[0].view(np.uint32), fulls.view(np.uint32)), t
        for other in (_kernel(ar, res, gamma, _oneshot(t), sequential=True)[2],
                      _kernel(ar, res, gamma, _oneshot(t), bf16_squares=not _oneshot(t))[2]):
            assert not np.array_equal(other, fulls), t


def _fused_stubs(monkeypatch, comm):
    fc = types.ModuleType("flashinfer.comm")
    fc.AllReduceFusionPattern = SimpleNamespace(kAllReduce=0, kARResidualRMSNorm=1, kARResidualRMSNormFP8Quant=2)

    class Strategy:
        ONESHOT, TWOSHOT = 0, 1

        @staticmethod
        def select_strategy(tp, tokens, dim, dtype):
            return Strategy.ONESHOT if tokens * dim * tp * 2 <= 64 * 1024 * 8 * 2 else Strategy.TWOSHOT
    mar = types.ModuleType("flashinfer.comm.trtllm_mnnvl_ar")
    mar.MNNVLAllreduceFusionStrategy = Strategy
    ps = types.ModuleType("vllm.distributed.parallel_state")
    ps.get_tp_group = lambda: SimpleNamespace(device_communicator=comm)
    for m in (fc, mar, ps):
        monkeypatch.setitem(sys.modules, m.__name__, m)
    monkeypatch.setitem(sys.modules, "flashinfer", types.SimpleNamespace(comm=fc, __path__=[]))
    monkeypatch.setattr(TA, "mufu_scale", lambda: _scale_torch)
    monkeypatch.setattr(TA, "device_sms", lambda device: 132)


def test_fused_calls_on_another_workspace_are_computed_as_mnnvl_computes_them_and_other_patterns_refused(monkeypatch, capsys):
    rng = np.random.default_rng(23)
    t = 21
    ranks = np.stack([_row_words(rng, t) for _ in range(8)])
    inputs = [_torch_words(w) for w in ranks]
    comm = _Comm(inputs, fi_max=0)
    _fused_stubs(monkeypatch, comm)
    seen = []

    def served(*a, **kw):
        seen.append(kw["workspace"].backend)
        return "vllm"

    @functools.wraps(served)
    def api(*a, **kw):  # FlashInfer's own decorator sets __wrapped__ too
        return served(*a, **kw)
    fc = sys.modules["flashinfer.comm"]
    fc.allreduce_fusion = api
    for _ in range(2):
        TA.install_fused(TA.parse_tree(NODE_A), torch.device("cpu"))
    fused = fc.allreduce_fusion
    assert fused.__wrapped__ is api
    res_w, gamma_w = _row_words(rng, t), _rne((rng.random(DIM) + 0.5).astype(np.float32))

    def call(backend, pattern=1, **extra):
        x, res = inputs[3].clone(), _torch_words(res_w)
        kw = dict(input=x, workspace=SimpleNamespace(backend=backend), pattern=pattern, launch_with_pdl=True, output=None,
                  residual_out=res, norm_out=x, quant_out=None, scale_out=None, residual_in=res,
                  rms_gamma=_torch_words(gamma_w), rms_eps=1e-5, scale_factor=None, layout_code=None, use_oneshot=False,
                  fp32_acc=True, weight_bias=0.0, **extra)
        return fused(**kw), x, res
    assert call("mnnvl")[0] == "vllm" and seen == ["mnnvl"]
    got, x, res = call("trtllm")
    assert got is x and comm.gathers == 1
    want, want_h, _ = _kernel(_from_zero(ranks.reshape(8, -1)).reshape(t, DIM), res_w, gamma_w, oneshot=False)
    assert np.array_equal(_words(res), want_h) and np.array_equal(_words(x), want)
    comm.at = 0
    call("trtllm")
    assert capsys.readouterr().err.count("runs on a trtllm workspace") == 1
    with pytest.raises(RuntimeError, match="doesn't cover"):
        call("trtllm", pattern=2)
    assert fused(1, workspace=SimpleNamespace(backend="trtllm")) == "vllm"


def test_site_registers_the_plugin_with_its_tree(tmp_path, monkeypatch):
    d = TA.site(tmp_path / "site", NODE_A, "a probe")
    assert (d / "verity_tree_allreduce-0.0.dist-info" / "entry_points.txt").read_text() == (
        "[vllm.general_plugins]\nverity_tree_allreduce = tree_allreduce:register\n")
    assert json.loads((d / TA.CONFIG).read_text()) == {"tree": NODE_A, "source": "a probe"}
    cc = types.ModuleType("vllm.distributed.device_communicators.cuda_communicator")
    orig = lambda self, x: x  # noqa: E731
    cc.CudaCommunicator = type("CudaCommunicator", (), {"all_reduce": orig})
    monkeypatch.setitem(sys.modules, "vllm.distributed.device_communicators.cuda_communicator", cc)
    monkeypatch.setitem(sys.modules, "vllm.distributed.device_communicators",
                        types.SimpleNamespace(cuda_communicator=cc, __path__=[]))
    monkeypatch.syspath_prepend(str(d))
    monkeypatch.delitem(sys.modules, "tree_allreduce", raising=False)
    plugin = importlib.import_module("tree_allreduce")
    assert plugin.__file__ == str(d / "tree_allreduce.py")
    plugin.register()
    assert cc.CudaCommunicator.all_reduce.__wrapped__ is orig
    with pytest.raises(ValueError):
        TA.site(tmp_path / "bad", "(0,1,2)", "")
