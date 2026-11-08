"""`tree_allreduce`: the serving node's tree parsed as its probe prints it, the gathered words summed in that tree's f32 order and
rounded once to bf16 (against numpy's IEEE f32 adds, and refused by an order that differs), FlashInfer's route under ``mnnvl``
and other dtypes left to vLLM, FlashInfer's calls under another backend summed as ``mnnvl`` sums them, the custom all-reduce
given the serving node's cap on a node without symmetric memory, and the plugin's site registering it."""
from __future__ import annotations

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
