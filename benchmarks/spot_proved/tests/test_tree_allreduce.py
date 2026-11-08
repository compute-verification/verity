"""`tree_allreduce`: the serving node's tree parsed as its probe prints it, the gathered words summed in that tree's f32 order and
rounded once to bf16 (against numpy's IEEE f32 adds, and refused by an order that differs), FlashInfer's route and other dtypes
left to vLLM, and the plugin's site registering it."""
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


@pytest.fixture(autouse=True)
def _pynccl_allocator(monkeypatch):
    """vLLM's `pynccl_allocator.is_symmetric_memory_enabled`, answering ``NCCL_SYMM["on"]``."""
    NCCL_SYMM["on"] = False
    mod = types.ModuleType("vllm.distributed.device_communicators.pynccl_allocator")
    mod.is_symmetric_memory_enabled = lambda: NCCL_SYMM["on"]
    monkeypatch.setitem(sys.modules, "vllm.distributed.device_communicators.pynccl_allocator", mod)


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
    all-reduce those of exactly ``ca_eq`` words, and pynccl's all-gather lays the ranks' words out rank-major (``gathers`` counts
    its calls)."""

    def __init__(self, inputs: list[torch.Tensor], fi_max: int, world: int = 8, ca_eq: int = -1) -> None:
        self.world_size, self.inputs, self.gathers = world, inputs, 0
        self.fi_ar_comm = SimpleNamespace(disabled=False, should_use_fi_ar=lambda x: x.numel() <= fi_max)
        self.ca_comm = SimpleNamespace(disabled=False, should_custom_ar=lambda x: x.numel() == ca_eq)
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


def test_a_call_is_gathered_in_chunks_with_the_same_bits(monkeypatch):
    rng = np.random.default_rng(5)
    words = rng.integers(0x3000, 0x4800, size=(8, 3, 7), dtype=np.uint16)
    inputs = [_torch_words(w) for w in words]
    monkeypatch.setattr(TA, "CHUNK", 5)
    comm = _Comm(inputs, fi_max=4)
    out = TA.wrap(None, TA.parse_tree(NODE_A))(comm, inputs[2])
    assert comm.gathers == 5
    assert np.array_equal(_words(out).reshape(-1), _reference(words.reshape(8, -1), [0, 2, 3, 1, 5, 6, 4, 7]))


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
