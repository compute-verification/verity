"""`acquire.sources.int8_source`: the exact-int8 quantizers' results committed under the manifest's names -- a linear's (x_q, s_a) as
its Int8RowQuant_v1 call boundaries (claimed from the call-boundary source), an attention module's KV quantization as `<attention>/0..3`
-- bit for bit what the served ops computed; nothing outside a committed step or the manifest's sites."""
from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources import int8_source as I8
from verity_vllm.acquire.sources import taps
from verity_vllm.engine import int8_attention as A
from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_ops

LIN, ATTN = "model.layers.0.mlp.down_proj", "model.layers.0.self_attn.attn"
S, G, D, BS = 5, 2, 32, 4


def _manifest(steps=(0, 1)):
    ids = []
    for e in steps:
        ids += [{"family": "call_boundaries", "op_path": f"{LIN}/row_quant_{e}", "output_member": m, "step": e, "engine_step": e,
                 "spec": "Int8RowQuant_v1", "request_id": "r0"} for m in ("0", "1")]
        ids += [{"family": "instance_outputs", "op_path": ATTN, "output_member": m, "step": e, "engine_step": e, "spec": "Int8KvQuant_v1",
                 "request_id": "r0"} for m in I8.KV_MEMBERS]
        ids.append({"family": "instance_outputs", "op_path": LIN, "output_member": "0", "step": e, "engine_step": e, "spec": "ScaledMmS8_v1"})
    return {"ranks": 1, "identities": ids}


class _Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.weight = nn.Parameter(torch.randint(-127, 128, (8, 64), generator=g, dtype=torch.int8), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.rand(8, 1, generator=g) * 1e-2 + 1e-4, requires_grad=False)

    def forward(self, x):
        return int8_ops.linear(x, self.weight, self.weight_scale)[2]


class _Attn(nn.Module):
    def forward(self, kc, vc, blocks, ks):
        return int8_ops.kv_quant(kc, vc, blocks, S, ks)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.down_proj = _Linear()
        self.model.layers[0].self_attn = nn.Module()
        self.model.layers[0].self_attn.attn = _Attn()


class _Committer:
    name = "fake"

    def __init__(self) -> None:
        self._cur = None
        self.got: dict[str, torch.Tensor] = {}
        self.errors: list[str] = []
        self.sources: list = []

    def _acquire(self, name, t, job):
        assert name not in self.got, name
        self.got[name] = t.clone()

    def attach_source(self, src):
        self.sources.append(src)


def _kv_args():
    g = torch.Generator().manual_seed(3)
    k, v = ((torch.randn(S, G, D, generator=g) * 2).bfloat16() for _ in range(2))
    blocks = torch.tensor([1, 0], dtype=torch.int32)
    kc, vc = torch.zeros(2, BS, G, D, dtype=torch.bfloat16), torch.zeros(2, BS, G, D, dtype=torch.bfloat16)
    for t in range(S):
        kc[blocks[t // BS], t % BS], vc[blocks[t // BS], t % BS] = k[t], v[t]
    return (kc, vc, blocks, torch.tensor([0.03, 0.05])), (k, v)


def _run(src, com, model, step):
    com._cur = types.SimpleNamespace(step=step)
    src.on_step_begin()
    x = (torch.randn(3, 64, generator=torch.Generator().manual_seed(step)) * 3).bfloat16()
    model.model.layers[0].mlp.down_proj(x)
    args, kv = _kv_args()
    model.model.layers[0].self_attn.attn(*args)
    return x, kv, args[3]


def test_the_quantizers_results_are_committed_under_the_manifests_names_as_the_ops_computed_them():
    com, model, man = _Committer(), _Model(), _manifest()
    classes, src = I8.attach(com, I8.applies(man), man, ("instance_outputs",), model=model)
    assert classes == ("instance_outputs", "call_boundaries") and com.sources == [src]
    src.install()
    try:
        x, (k, v), ks = _run(src, com, model, 1)
    finally:
        src.uninstall()
    xq, s_a = E.quantize_rows(x)
    assert torch.equal(com.got[f"{LIN}/row_quant_1/0"], xq) and torch.equal(com.got[f"{LIN}/row_quant_1/1"], s_a)
    for m, t in zip(I8.KV_MEMBERS, A.kv_quant(k, v, ks)):
        assert torch.equal(com.got[f"{ATTN}/{m}"], t.reshape(-1)), m
    assert sorted(com.got) == sorted([f"{LIN}/row_quant_1/0", f"{LIN}/row_quant_1/1", *(f"{ATTN}/{m}" for m in I8.KV_MEMBERS)])
    assert not com.errors and src.stats["quant_tensors"] == 2 and src.stats["kv_tensors"] == 4
    assert int8_ops.row_quant is not None and not isinstance(int8_ops.row_quant, types.FunctionType)      # the op is back
    decl = {c["class"]: c for c in src.declaration_classes()}
    assert decl["call_boundaries"]["tensors"] == 2 and decl[I8.KV_CLASS]["tensors"] == 4


class _Read(nn.Module):
    """A batch's int8 attention as the served layer runs it: through the paged caches (`paged_attention`), or over the windows
    a credited read put in a step's K and V rows (`window_attention`, `engine.int8_window`)."""

    def __init__(self, credited: bool) -> None:
        super().__init__()
        self.credited = credited

    def forward(self, q, caches, rows, block_table, qsl, lens, ks):
        out = torch.empty_like(q)
        if self.credited:
            int8_ops.window_attention(q, *rows, qsl, lens, ks, D ** -0.5, out)
        else:
            int8_ops.paged_attention(q, *caches, qsl, lens, block_table, ks, D ** -0.5, out)
        return out


def test_a_credited_read_commits_the_attentions_kv_quantization_as_the_paged_read_does():
    """Under a credited read the layer quantizes with `kv_quant_window`, not `kv_quant`: its results are the same
    `<attention>/0..3`, bit for bit, so a Commit whose warm-up read the pages and whose run reads through the store's
    session lays out the same steps (a source that saw only `kv_quant` left every step of such a run uncommitted)."""
    g = torch.Generator().manual_seed(5)
    lens = [S, 3]                                                        # a prefill of S tokens and a decode row over 3
    ks, block_table = torch.tensor([0.03, 0.05]), torch.tensor([[1, 0], [2, 3]], dtype=torch.int32)
    kc, vc = torch.zeros(4, BS, G, D, dtype=torch.bfloat16), torch.zeros(4, BS, G, D, dtype=torch.bfloat16)
    windows = []
    for r, s in enumerate(lens):
        k, v = ((torch.randn(s, G, D, generator=g) * 2).bfloat16() for _ in range(2))
        for t in range(s):
            kc[block_table[r, t // BS], t % BS], vc[block_table[r, t // BS], t % BS] = k[t], v[t]
        windows.append((k, v))
    rows = (torch.cat([k for k, _ in windows]), torch.cat([v for _, v in windows]), [0, lens[0], lens[0] + lens[1]])
    q = (torch.randn(S + 1, 2 * G, D, generator=g) * 2).bfloat16()
    qsl, seq = torch.tensor([0, S, S + 1], dtype=torch.int32), torch.tensor(lens, dtype=torch.int32)
    got, outs = [], []
    for credited in (False, True):
        com, model, man = _Committer(), _Model(), _manifest(steps=(1,))
        model.model.layers[0].self_attn.attn = _Read(credited)
        _classes, src = I8.attach(com, True, man, (), explicit_classes=True, model=model)
        src.install()
        try:
            com._cur = types.SimpleNamespace(step=1)
            src.on_step_begin()
            outs.append(model.model.layers[0].self_attn.attn(q, (kc, vc), rows, block_table, qsl, seq, ks))
        finally:
            src.uninstall()
        assert not com.errors and src.stats["kv_tensors"] == 4
        got.append(com.got)
    assert sorted(got[1]) == sorted(f"{ATTN}/{m}" for m in I8.KV_MEMBERS)
    want = [torch.cat([t.reshape(-1) for t in ts]) for ts in zip(*(A.kv_quant(k, v, ks) for k, v in windows))]
    for m, t in zip(I8.KV_MEMBERS, want):
        assert torch.equal(got[0][f"{ATTN}/{m}"], t) and torch.equal(got[1][f"{ATTN}/{m}"], t), m
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
    assert not isinstance(int8_ops.kv_quant_window, types.FunctionType)                 # the op is back


def test_nothing_is_committed_outside_a_committed_step_or_a_site():
    com, model, man = _Committer(), _Model(), _manifest(steps=(0,))
    _classes, src = I8.attach(com, True, man, (), explicit_classes=True, model=model)
    src.install()
    try:
        model.model.layers[0].mlp.down_proj((torch.randn(2, 64)).bfloat16())                  # no job
        _run(src, com, model, 3)                                                                # a step without a row_quant site
    finally:
        src.uninstall()
    assert sorted(com.got) == sorted(f"{ATTN}/{m}" for m in I8.KV_MEMBERS)


def test_the_call_boundary_source_leaves_int8_quantizer_boundaries_to_this_source():
    man = _manifest()
    assert CB.manifest_sites(man) == []
    assert CB.unattached(man, []) and not CB.unattached(man, taps.attached(man))
    assert I8.SHORT_NAME in taps.attached(man) and I8.SHORT_NAME not in taps.attached({"identities": []})


def test_x03_holds_the_quantizer_names_to_the_steps_sites():
    src = I8.Int8Source(_Committer(), _Model(), I8.quant_sites(_manifest()), [])
    names = {f"{LIN}/row_quant_0/0": 1, f"{LIN}/row_quant_0/1": 1, f"{LIN}": 1}
    rest, bad = src.split_step_names(names, 0, False)
    assert rest == {LIN: 1} and bad == {}
    _rest, bad = src.split_step_names(names, 1, False)
    assert bad == {f"{LIN}/row_quant_0/0": (1, 0), f"{LIN}/row_quant_0/1": (1, 0), f"{LIN}/row_quant_1/0": (0, 1), f"{LIN}/row_quant_1/1": (0, 1)}


def test_a_manifest_without_int8_sites_refuses_the_source():
    with pytest.raises(SystemExit):
        I8.attach(_Committer(), True, {"identities": []}, (), model=_Model())
