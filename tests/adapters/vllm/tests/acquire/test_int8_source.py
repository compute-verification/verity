"""`acquire.sources.int8_source`: the exact-int8 quantizers' results committed under the capture plan's names -- a linear's (x_q, s_a)
as its Int8RowQuant call boundary's two members (claimed from the call-boundary source), at every engine step; an attention
module's KV quantization as `<attention>/kv_quant.0..3`, declared as the plan's Int8KvQuant Definition's ports -- bit for bit what
the served ops computed; nothing outside a committed step or the plan's sites; and the binder binds every value of the attention
under the plan's own spelling."""
from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from verity_vllm.acquire.sources import int8_source as I8
from verity_vllm.engine import int8_attention as A
from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_ops
from verity_vllm.query.manifest.bind import Binder

LIN, ATTN = "model.layers.0.mlp.down_proj", "model.layers.0.self_attn.attn"
S, G, D, BS = 5, 2, 32, 4


QUANT_OP = f"{LIN}/row_quant"


def _class(op, member, capture, width, dtype, definition):
    return {"op_path": op, "member": member, "capture": capture, "width": width, "dtype": dtype, "definitions": [definition]}


def _plan(quant=True, version=1):
    """The plan classes (`capture_plan.of_build`) of a row with one int8 linear and one int8 attention."""
    kv = [_class(ATTN, f"kv_quant.{p}", "module_output", w, d, "Int8KvQuant_v1")
          for p, w, d in zip(I8.KV_PORTS, (D, D, D, D), ("i8", "i8", "f32", "f32"))]
    return [*([_class(QUANT_OP, "0", "call_boundary", 64, "i8", f"Int8RowQuant_v{version}"),
               _class(QUANT_OP, "1", "call_boundary", 1, "f32", f"Int8RowQuant_v{version}")] if quant else []),
            _class(LIN, "0", "module_output", 8, "bf16", "ScaledMmS8_v1"), *kv,
            _class(ATTN, "out", "module_output", D, "bf16", "Int8LutAttentionRow_v1")]


def _identities(steps=(0,)):
    """What the seal's binder places of the attention at each step (`serve_seal.served_binder_of`): each member of its plan rows."""
    return {"identities": [{"op_path": ATTN, "output_member": m, "invocation": 0, "identity": None, "request_id": "*", "step": None,
                            "engine_step": e, "rank": 0}
                           for e in steps for m in [*(f"kv_quant.{p}" for p in I8.KV_PORTS), "out"]], "requests_declared": []}


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
        return int8_ops.kv_quant_batch(kc, vc, blocks[None], [S], ks)


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
        self._cur: types.SimpleNamespace | None = None
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


@pytest.mark.parametrize("version", [1, 5])
def test_the_quantizers_results_are_committed_under_the_plans_names_as_the_ops_computed_them(version):
    com, model, plan = _Committer(), _Model(), _plan(version=version)
    classes, src = I8.attach(com, I8.applies(plan), plan, ("module_output",), model=model)
    assert classes == ("module_output", "call_boundary") and com.sources == [src]
    src.install()
    try:
        x, (k, v), ks = _run(src, com, model, 1)
    finally:
        src.uninstall()
    xq, s_a = E.quantize_rows(x)
    assert torch.equal(com.got[f"{QUANT_OP}/0"], xq) and torch.equal(com.got[f"{QUANT_OP}/1"], s_a)
    for m, t in zip(I8.KV_LABELS, A.kv_quant(k, v, ks)):
        assert torch.equal(com.got[f"{ATTN}/{m}"], t.reshape(-1)), m
    assert sorted(com.got) == sorted([f"{QUANT_OP}/0", f"{QUANT_OP}/1", *(f"{ATTN}/{m}" for m in I8.KV_LABELS)])
    assert not com.errors and src.stats["quant_tensors"] == 2 and src.stats["kv_tensors"] == 4
    assert int8_ops.row_quant is not None and not isinstance(int8_ops.row_quant, types.FunctionType)      # the op is back
    decl = {c["class"]: c for c in src.declaration_classes()}
    assert decl["call_boundary"]["tensors"] == 2 and decl[I8.KV_CLASS]["tensors"] == 4


class _Read(nn.Module):
    """A batch's int8 attention as the served layer runs it: through the paged caches (`paged_attention`), or over the windows
    a credited read put in a step's K and V rows (`window_attention`)."""

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
    """Under a credited read the layer quantizes with `kv_quant_window_batch`, not `kv_quant_batch`: its results are the same
    `<attention>/kv_quant.0..3`, bit for bit, so a Commit whose warm-up read the pages and whose run reads through the store's
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
        com, model, man = _Committer(), _Model(), _plan()
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
    assert sorted(got[1]) == sorted(f"{ATTN}/{m}" for m in I8.KV_LABELS)
    want = [torch.cat([t.reshape(-1) for t in ts]) for ts in zip(*(A.kv_quant(k, v, ks) for k, v in windows))]
    for m, t in zip(I8.KV_LABELS, want):
        assert torch.equal(got[0][f"{ATTN}/{m}"], t) and torch.equal(got[1][f"{ATTN}/{m}"], t), m
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
    assert not isinstance(int8_ops.kv_quant_window_batch, types.FunctionType)           # the op is back


def test_nothing_is_committed_outside_a_committed_step_or_a_site():
    com, model, plan = _Committer(), _Model(), _plan(quant=False)
    _classes, src = I8.attach(com, True, plan, (), explicit_classes=True, model=model)
    src.install()
    try:
        model.model.layers[0].mlp.down_proj((torch.randn(2, 64)).bfloat16())                  # no job
        _run(src, com, model, 3)                                                                # a linear the plan names no site of
    finally:
        src.uninstall()
    assert sorted(com.got) == sorted(f"{ATTN}/{m}" for m in I8.KV_LABELS)


def test_a_site_is_committed_at_every_engine_step_under_the_plans_op():
    """The plan names a call boundary by its op, its fx node's count over a request dropped: a served step batches requests at
    any step of their own, so the names are the same at each."""
    com, model, plan = _Committer(), _Model(), _plan()
    _classes, src = I8.attach(com, True, plan, (), explicit_classes=True, model=model)
    src.install()
    try:
        for step in (0, 7):
            com.got.clear()
            _run(src, com, model, step)
            assert {f"{QUANT_OP}/0", f"{QUANT_OP}/1"} <= set(com.got)
    finally:
        src.uninstall()
    assert not com.errors and src.stats["quant_tensors"] == 4


def test_the_call_boundary_source_leaves_int8_quantizer_boundaries_to_this_source():
    plan = _plan()
    assert [I8.claim(c) for c in plan[:3]] == [I8.SHORT_NAME, I8.SHORT_NAME, None]
    assert I8.quant_sites(plan) == {(0, LIN): QUANT_OP} and (41, LIN) in I8.quant_sites(plan)
    assert I8.applies(plan) and not I8.applies([plan[2]])


def test_x03_holds_the_quantizer_names_to_the_sites_at_every_step():
    src = I8.Int8Source(_Committer(), _Model(), I8.quant_sites(_plan()), [])
    names = {f"{QUANT_OP}/0": 1, f"{QUANT_OP}/1": 1, f"{LIN}": 1}
    for step in (0, 1):
        rest, bad = src.split_step_names(names, step, False)
        assert rest == {LIN: 1} and bad == {}
    _rest, bad = src.split_step_names({LIN: 1, f"{QUANT_OP}/0": 2}, 1, False)
    assert bad == {f"{QUANT_OP}/0": (2, 1), f"{QUANT_OP}/1": (0, 1)}


@pytest.mark.parametrize("edit,why", [
    (lambda p: p[0].update(member="q"), r"row_quant/q: an Int8RowQuant call boundary's member is its output 0"),
    (lambda p: p.append({**p[0], "op_path": f"{LIN}/row_quant_b"}), r"has two Int8RowQuant ops"),
    (lambda p: p[0]["definitions"].append("Gemm_v1"), r"row_quant/0 is Int8RowQuant's and another Definition's")])
def test_a_plan_the_source_cannot_tap_is_refused_by_name(edit, why):
    plan = _plan()
    edit(plan)
    with pytest.raises(ValueError, match=why):
        I8.quant_sites(plan)


def test_a_credited_reads_window_quantization_is_committed_as_the_pages_is():
    """Under a credited cache read the attention quantizes its windows with `verity_int8::kv_quant_window_batch`: the same
    `<attention>/kv_quant.0..3`, bit for bit, two requests at one step concatenated in batch order, each its own slice of the batch's
    padded results (the Commit left 4,096 of them unbound)."""
    com, model, man = _Committer(), _Model(), _plan()
    (_kc, _vc, _b, ks), (k, v) = _kv_args()
    rows_k, rows_v = torch.cat([k, k[:2]]), torch.cat([v, v[:2]])
    model.model.layers[0].self_attn.attn.forward = lambda: int8_ops.kv_quant_window_batch(rows_k, rows_v, [0, S], [S, 2], ks)
    _classes, src = I8.attach(com, True, man, (), explicit_classes=True, model=model)
    src.install()
    try:
        com._cur = types.SimpleNamespace(step=0)
        src.on_step_begin()
        model.model.layers[0].self_attn.attn()
    finally:
        src.uninstall()
    for j, m in enumerate(I8.KV_LABELS):
        want = torch.cat([A.kv_quant(k[:n], v[:n], ks)[j].reshape(-1) for n in (S, 2)])
        assert torch.equal(com.got[f"{ATTN}/{m}"], want), m
    assert not com.errors and src.stats["kv_tensors"] == 4


def test_a_plan_without_int8_sites_refuses_the_source():
    with pytest.raises(SystemExit):
        I8.attach(_Committer(), True, [], (), model=_Model())
    with pytest.raises(SystemExit):
        I8.attach(_Committer(), True, _plan(quant=False)[:1], (), model=_Model())


def test_the_seals_binder_places_the_kv_quantization_by_the_plans_member_names():
    """The plan spells a hinted multi-port Call's Values `<hint>.<port>` and the attention's return `out`; the source commits
    `<attention>/kv_quant.0..3`, so the binder of the plan's rows (`serve_seal.served_binder_of`, which carry no Definition and no
    port) places each by its label and the return at its one remaining member.  Committed as bare digits, `<attention>/0..3` were
    slots of the attention's return beside the return itself, and none of the five bound."""
    plan, man = _plan(), _identities()
    src = I8.Int8Source(_Committer(), _Model(), {}, I8.kv_modules(plan))
    assert src.kv == [ATTN] and not hasattr(src, "ports")
    names = [*(f"{ATTN}/{m}" for m in I8.KV_LABELS), ATTN]
    bound = Binder(man["identities"], rank=0, ports={}).bind(0, names)
    assert [(op, m, how) for op, m, _k, how in bound] == [*((ATTN, f"kv_quant.{p}", "label") for p in I8.KV_PORTS), (ATTN, "out", "slot")]
    digits = Binder(man["identities"], rank=0, ports={}).bind(0, [*(f"{ATTN}/{p}" for p in I8.KV_PORTS), ATTN])
    assert {how for *_x, how in digits} == {"unbound"}
