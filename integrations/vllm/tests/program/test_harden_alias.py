"""R11 cb-harden (1) — mutation / alias effects of the Bet A circuit builder (`torch.export(strict=False)` training IR ->
`verity_vllm.program.frontend.derive`).

What is under test. `derive` translates the pre-dispatch export graph WITH in-place ops kept in place (no functionalisation):
a rule-owned mutating op (`rotary_embedding`, `fused_add_rms_norm`, `rms_norm`, `silu_and_mul`) rebinds the mutated fx node
to the pure result (`Ctx.rebind`) and marks every other alias of the same storage whose leaf runs overlap as `stale`; reading a
stale alias refuses (`mutation`). The property that must NEVER fail: a consumer reads the value **as of the time it was read**
in eager PyTorch — never silently the pre-mutation value (a naive "graph of ops" would hand a pre-taken view its
pre-mutation refs) and never silently a value that eager would not produce. Fail-closed refusal is acceptable; a Program that
evaluates to something eager does not produce is a SEMANTIC FAILURE.

Oracle. Eager PyTorch on CPU: this file registers CPU kernels for the `vllm_iface` stand-in ops (which otherwise have fake
impls only) that follow the registered kinds' stated semantics (`RoPEHead_v1`: `bf16(fma(x, c, -(y*s)))`; `RMSNormTriton`;
`RMSNormFusedCuda`), so the eager run mutates real storage exactly like vLLM's kernels do. The derived Program is
evaluated with `materialize` (registered gate semantics) and compared to eager: dataflow verdicts use a 1-bf16-ulp
tolerance and are DISCRIMINATING — every case also has the "wrong read" alternative (pre- vs post-mutation value), which
differs from the right one by O(1), and the test asserts the derived value is NOT the wrong one.

Cases (verdict semantics in the docstrings; the R11 report table is built from `RESULTS`):
    A  disjoint aliases: rotary writes q,k through split views; v (disjoint) stays readable       -> Program == eager
    B  read-before-mutation snapshot (`q.clone()` before rotary), q returned after              -> Program == eager (snap = pre, q = post)
    C  pre-taken view of the mutated operand read after the mutation                              -> refuse(stale) | Program == eager(post)
    D  pre-taken OVERLAPPING slice of the base read after                                          -> refuse(stale) | Program == eager(post)
    E  base tensor read after a view of it was mutated (`NegStaleAlias`)                           -> refuse(stale) | Program == eager(post)
    F  sibling view taken from the (stale) base AFTER the mutation                                 -> refuse | Program == eager(post)
    G  two rule-owned mutations of the same operand (rotary twice)                                 -> Program == eager (double rotation)
    H  fused_add_rms_norm rebinding BOTH mutated operands                                          -> Program == eager
    I  view of `residual` taken before fused_add_rms_norm, read after                              -> refuse(stale) | Program == eager(post)
    J  alloc written through a same-shape view, BASE returned (vLLM Attention's `output` pattern) -> refuse(uninitialised) | Program == eager
    K  view of an alloc taken BEFORE the write, read after                                         -> refuse(uninitialised) | Program == eager
    X  export-level facts: in-place op kept (no auto_functionalized), USER_OUTPUT-only signature; buffer / user-input mutations
       through a CUSTOM mutable op are NOT surfaced in `graph_signature.output_specs` (measured; ATen `add_` on a buffer is) —
       derive's own storage tracking refuses the buffer write and declares the input write (`mutation_effects`)
    Y  plain ATen in-place on intermediates (add_/mul_/slice write/index_put_) and `as_strided`      -> refused (fail-closed)

Run (pod venv, CPU): `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. nice -n 10 python -m pytest -q -p no:cacheprovider
tests/program/test_harden_alias.py`; `HARDEN_OUT=<dir>` writes `alias_results.json`.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from verity_vllm.program.frontend.derive import Unsupported, derive, derive_report
from verity_vllm.program.frontend import derive_examples as ex
from verity_vllm.program.frontend.rules import vllm_iface as vi
from verity_vllm.program.kernels.serve3_reference import materialize

PROF = dict(mode="profile", profile="b1-eager")
C = dict(T=3, NH=2, KVH=1, D=8, H=16, MAX_POS=16, EPS=1e-5, POS0=2)
QKV = (C["NH"] + 2 * C["KVH"]) * C["D"]
NHD, KVHD = C["NH"] * C["D"], C["KVH"] * C["D"]
BF = torch.bfloat16

RESULTS: list[dict] = []


# ---------------------------------------------------------------------------------------------------------
# eager oracle: CPU kernels for the stand-in ops, following the registered kinds' stated semantics
# ---------------------------------------------------------------------------------------------------------


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float32)


def _fma_sub(x, c, y, s):
    """`fma(x, c, -(y*s))` in f32 semantics: `y*s` rounded to f32, then ONE rounding of `x*c - (y*s)` (exact in f64)."""
    ys = (y.double() * s.double()).float().double()
    return (x.double() * c.double() - ys).float()


def _fma_add(x, c, y, s):
    ys = (y.double() * s.double()).float().double()
    return (x.double() * c.double() + ys).float()


def _cpu_rotary(positions, query, key, head_size, cos_sin_cache, is_neox):
    assert is_neox
    R = head_size // 2
    cs = _f32(cos_sin_cache[positions])  # [T, D]
    cos, sin = cs[:, None, :R], cs[:, None, R:]

    def rot(t):
        T = t.shape[0]
        x = _f32(t).reshape(T, -1, head_size)
        x1, x2 = x[..., :R], x[..., R:]
        lo = _fma_sub(x1, cos, x2, sin)  # x[i]*cos - x[i+R]*sin
        hi = _fma_add(x2, cos, x1, sin)  # x[i+R]*cos + x[i]*sin
        t.copy_(torch.cat([lo, hi], -1).to(t.dtype).reshape(t.shape))

    rot(query)
    if key is not None:
        rot(key)


def _rms(x32: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    var = x32.pow(2).mean(-1, keepdim=True)
    return ((x32 * torch.rsqrt(var + eps)).to(BF).float() * _f32(w)).to(BF)


def _cpu_rms_norm(result, x, weight, eps):
    result.copy_(_rms(_f32(x), weight, eps))


def _cpu_fused_add_rms_norm(x, residual, weight, eps):
    z = (_f32(x) + _f32(residual)).to(BF)
    residual.copy_(z)
    x.copy_(_rms(_f32(z), weight, eps))


def _cpu_silu_and_mul(result, x):
    d = x.shape[-1] // 2
    g, u = _f32(x[..., :d]), _f32(x[..., d:])
    result.copy_((torch.nn.functional.silu(g).to(BF).float() * u).to(BF))


def _register_cpu_kernels() -> None:
    for name, fn in (("rotary_embedding", _cpu_rotary), ("rms_norm", _cpu_rms_norm), ("fused_add_rms_norm", _cpu_fused_add_rms_norm),
                     ("silu_and_mul", _cpu_silu_and_mul)):
        try:
            torch.library.register_kernel(f"{vi._LIB_NAME}::{name}", "CPU", fn)
        except RuntimeError as e:  # already registered in this process
            if "already" not in str(e).lower():
                raise


_register_cpu_kernels()


# ---------------------------------------------------------------------------------------------------------
# evaluation of a derived Program against the eager module
# ---------------------------------------------------------------------------------------------------------


def _bf16_bits(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t.detach().contiguous().view(torch.int16).numpy().astype(np.uint16).reshape(-1)]


def _from_bits(words: np.ndarray, shape) -> torch.Tensor:
    return torch.from_numpy(words.astype(np.uint16).view(np.int16)).view(BF).reshape(shape).float()


def _evaluate(rep, module: nn.Module, inputs: list[torch.Tensor], out_shapes: list[tuple[int, ...]]) -> list[torch.Tensor]:
    """Program inputs = the example tensors (row-major leaves) then the weights record in declared order."""
    prog = rep.program
    tensors = dict(module.named_parameters())
    tensors.update(dict(module.named_buffers()))
    words = [_bf16_bits(x) for x in inputs]
    wbits: list[int] = []
    for w in rep.provenance["weights"]:
        wbits += _bf16_bits(tensors[w["name"]])
    if wbits:
        words.append(wbits)
    t = materialize(prog, words)
    flat = np.array([np.uint32(t[g]) for g in prog.circuit.root_outputs()], dtype=np.uint32)
    outs, k = [], 0
    for shp in out_shapes:
        n = int(np.prod(shp))
        outs.append(_from_bits(flat[k:k + n], shp))
        k += n
    assert k == flat.size, (k, flat.size)
    return outs


def _ulp_close(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, float]:
    """Within one bf16 ulp (relative 2^-7) plus a tiny absolute slack; returns (ok, max abs diff)."""
    d = (a - b).abs().max().item()
    ok = torch.allclose(a, b, rtol=1 / 128, atol=1 / 256)
    return ok, d


def _far(a: torch.Tensor, b: torch.Tensor) -> bool:
    """The two candidate readings (pre- vs post-mutation) are distinguishable: max |a-b| > 0.25."""
    return (a - b).abs().max().item() > 0.25


def _record(case: str, expected: str, observed: str, verdict: str, cause: str = "", **extra) -> None:
    RESULTS.append({"case": case, "expected": expected, "observed": observed, "verdict": verdict, "cause": cause, **extra})


def _dump():
    out = os.environ.get("HARDEN_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(RESULTS, open(os.path.join(out, "alias_results.json"), "w"), indent=1)


@pytest.fixture(scope="module", autouse=True)
def _dump_results():
    yield
    _dump()


# ---------------------------------------------------------------------------------------------------------
# source modules (ordinary PyTorch, vLLM-style in-place kernel interface)
# ---------------------------------------------------------------------------------------------------------


class _Rot(nn.Module):
    """Base: `qkv = x + y` (a fresh intermediate; bit-exact `ResidualAdd`), split into q/k/v views, rotary in place on q, k."""

    def __init__(self) -> None:
        super().__init__()
        self.rotary_emb = ex.RotaryEmbedding(C["D"], C["MAX_POS"], 10000.0)

    def positions(self, x):
        return torch.arange(C["POS0"], C["POS0"] + x.shape[0])

    def split(self, qkv):
        return qkv.split([NHD, KVHD, KVHD], dim=-1)


class A_Disjoint(_Rot):
    def forward(self, x, y):
        qkv = x + y
        q, k, v = self.split(qkv)
        self.rotary_emb(self.positions(x), q, k)
        return q, k, v


class B_Snapshot(_Rot):
    def forward(self, x, y):
        qkv = x + y
        q, k, v = self.split(qkv)
        snap = q.clone()  # read BEFORE the mutation: must keep the pre-rotation value
        self.rotary_emb(self.positions(x), q, k)
        return snap, q


class C_StaleView(_Rot):
    def forward(self, x, y):
        qkv = x + y
        q, k, v = self.split(qkv)
        qa = q.view(x.shape[0], C["NH"], C["D"])  # alias taken before
        self.rotary_emb(self.positions(x), q, k)
        return qa.reshape(x.shape[0], NHD), k  # eager: post-rotation q


class D_OverlapSlice(_Rot):
    def forward(self, x, y):
        qkv = x + y
        qk = qkv[:, :NHD + KVHD]  # overlaps q and k, not v
        q, k, v = self.split(qkv)
        self.rotary_emb(self.positions(x), q, k)
        return qk, v


class F_SiblingAfter(_Rot):
    def forward(self, x, y):
        qkv = x + y
        q, k, v = self.split(qkv)
        self.rotary_emb(self.positions(x), q, k)
        q2 = qkv[:, :NHD]  # a NEW view of the base, taken after the mutation: eager reads the rotated q
        return q2, v


class G_Double(_Rot):
    def forward(self, x, y):
        qkv = x + y
        q, k, v = self.split(qkv)
        self.rotary_emb(self.positions(x), q, k)
        self.rotary_emb(self.positions(x), q, k)
        return q, k


class _Norm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C["H"], dtype=BF))


class H_FusedBoth(_Norm):
    def forward(self, x, y, r):
        h = x + y
        res = r + y
        torch.ops.vllm_iface.fused_add_rms_norm(h, res, self.weight, C["EPS"])
        return h, res


class I_FusedResView(_Norm):
    def forward(self, x, y, r):
        h = x + y
        res = r + y
        rv = res.view(-1)  # alias of residual taken before
        torch.ops.vllm_iface.fused_add_rms_norm(h, res, self.weight, C["EPS"])
        return h, rv.view(x.shape[0], C["H"])  # eager: post-mutation residual


class J_AllocViewBase(_Norm):
    def forward(self, x, y):
        h = x + y
        out = torch.empty_like(h)
        ov = out.view(x.shape[0], C["H"])
        torch.ops.vllm_iface.rms_norm(ov, h, self.weight, C["EPS"])
        return out  # eager: the written values (through the view)


class J0_AllocViewReturned(_Norm):
    def forward(self, x, y):
        h = x + y
        out = torch.empty_like(h)
        ov = out.view(x.shape[0], C["H"])
        torch.ops.vllm_iface.rms_norm(ov, h, self.weight, C["EPS"])
        return ov  # the conventional pattern (the written view is what is read)


class K_AllocViewBefore(_Norm):
    def forward(self, x, y):
        h = x + y
        out = torch.empty_like(h)
        ov = out.view(-1)  # alias of the allocation taken BEFORE the write
        torch.ops.vllm_iface.rms_norm(out, h, self.weight, C["EPS"])
        return ov.view(x.shape[0], C["H"])


class X_UserInputMutation(_Norm):
    def forward(self, x, r):
        torch.ops.vllm_iface.fused_add_rms_norm(x, r, self.weight, C["EPS"])  # writes two prescribed inputs
        return x, r


# ---------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------


def _inputs(n: int, width: int, seed: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(C["T"], width, generator=g) * 0.8).to(BF) for _ in range(n)]


def _eager(m: nn.Module, inputs):
    with torch.no_grad():
        outs = m(*[t.clone() for t in inputs])
    return [o.float() for o in (outs if isinstance(outs, (tuple, list)) else (outs,))]


def _alloc_view_rules():
    """`DEFAULT_RULES` + `AllocView` (the vLLM-bindings rule that lets `output = torch.empty(...); output.view(...)` be written by a
    kernel — vLLM's `Attention.forward` pattern). It lives in `rules/vllm_bindings.py`, i.e. only the vLLM ruleset has it."""
    from verity_vllm.program.frontend.rules import DEFAULT_RULES
    from verity_vllm.program.frontend.rules.vllm_bindings import AllocViewRule
    return DEFAULT_RULES.extended("default+AllocView", [AllocViewRule()])


def _derive(m: nn.Module, inputs, rules=None):
    return derive_report(m, tuple(torch.empty_like(t) for t in inputs), **PROF, name=type(m).__name__, rules=rules)


def _reason(rep) -> str:
    u = rep.unsupported
    return f"Unsupported({u.reason}) [{u.node}]: {u.hint}"


def _check_program_matches(case, rep, m, inputs, want: list[torch.Tensor], wrong: list[torch.Tensor] | None):
    """Program produced: it must evaluate to eager (1 ulp) and, where a wrong reading exists, be far from it."""
    got = _evaluate(rep, m, inputs, [tuple(w.shape) for w in want])
    diffs = []
    for i, (g, w) in enumerate(zip(got, want)):
        ok, d = _ulp_close(g, w)
        diffs.append(d)
        if wrong is not None and _far(want[i], wrong[i]):
            wrong_read = _ulp_close(g, wrong[i])[0]
            assert not wrong_read, f"{case}: output {i} equals the WRONG reading (pre-/post-mutation confusion) — SEMANTIC FAILURE"
        assert ok, f"{case}: output {i} differs from eager by {d} (max abs) — SEMANTIC FAILURE"
    return diffs


# ---------------------------------------------------------------------------------------------------------
# A, B, G, H: Programs that must be produced and match eager
# ---------------------------------------------------------------------------------------------------------


def test_A_disjoint_aliases_program_matches_eager():
    m = A_Disjoint()
    inputs = _inputs(2, QKV, 1)
    rep = _derive(m, inputs)
    assert rep.ok, _reason(rep)
    assert "RotaryEmbedding[vllm]_v1" in rep.provenance["rules_applied"] and "ResidualAdd[profile]_v1" in rep.provenance["rules_applied"]
    want = _eager(m, inputs)
    # the wrong reading of v would be... there is none (v is never written); the wrong reading of q,k is the un-rotated qkv
    qkv0 = (inputs[0] + inputs[1]).float()
    wrong = [qkv0[:, :NHD], qkv0[:, NHD:NHD + KVHD], qkv0[:, NHD + KVHD:]]
    diffs = _check_program_matches("A", rep, m, inputs, want, wrong)
    assert rep.provenance["mutation_effects"] == [], rep.provenance["mutation_effects"]  # no prescribed input was written
    _record("A disjoint aliases (q,k rotated in place; v read after)", "Program == eager; v not stale", f"Program; max|Δ|={max(diffs):.3g}", "PASS")


def test_B_snapshot_before_mutation_keeps_the_pre_mutation_value():
    m = B_Snapshot()
    inputs = _inputs(2, QKV, 2)
    rep = _derive(m, inputs)
    assert rep.ok, _reason(rep)
    want = _eager(m, inputs)  # [snap (pre), q (post)]
    assert _far(want[0], want[1]), "test gap: pre and post rotation coincide"
    wrong = [want[1], want[0]]  # snap read post / q read pre
    diffs = _check_program_matches("B", rep, m, inputs, want, wrong)
    _record("B snapshot (clone) before rotary; q after", "snap = pre-mutation, q = post-mutation", f"Program; max|Δ|={max(diffs):.3g}", "PASS")


def test_G_two_mutations_of_the_same_operand_compose():
    m = G_Double()
    inputs = _inputs(2, QKV, 3)
    rep = _derive(m, inputs)
    assert rep.ok, _reason(rep)
    assert sum(1 for a in rep.provenance["applications"] if a["rule"] == "RotaryEmbedding[vllm]_v1") == 2
    want = _eager(m, inputs)
    once = _eager(A_Disjoint(), inputs)[:2]  # the wrong reading: only one rotation applied
    diffs = _check_program_matches("G", rep, m, inputs, want, once)
    _record("G rotary applied twice to q,k", "Program == eager (double rotation), not the single rotation", f"Program; max|Δ|={max(diffs):.3g}", "PASS")


def test_H_fused_add_rms_norm_rebinds_both_mutated_operands():
    m = H_FusedBoth()
    inputs = _inputs(3, C["H"], 4)
    rep = _derive(m, inputs)
    assert rep.ok, _reason(rep)
    want = _eager(m, inputs)  # [normed(h+res), h+res]
    h0, r0 = (inputs[0] + inputs[1]).float(), (inputs[2] + inputs[1]).float()
    wrong = [h0, r0]  # pre-mutation values of both operands
    diffs = _check_program_matches("H", rep, m, inputs, want, wrong)
    _record("H fused_add_rms_norm (x!, residual!) both returned", "both rebound to post-mutation values", f"Program; max|Δ|={max(diffs):.3g}", "PASS")


def test_J0_alloc_written_through_view_and_the_view_returned():
    m = J0_AllocViewReturned()
    inputs = _inputs(2, C["H"], 5)
    plain = _derive(m, inputs)  # without the vLLM `AllocView` rule a view of an allocation has no rule at all (fail-closed)
    assert not plain.ok and plain.unsupported.reason == "decomposition", plain.unsupported
    rep = _derive(m, inputs, rules=_alloc_view_rules())
    assert rep.ok, _reason(rep)
    want = _eager(m, inputs)
    diffs = _check_program_matches("J0", rep, m, inputs, want, None)
    _record("J0 alloc written through a same-shape view; the view returned (vLLM Attention pattern)",
            "DEFAULT_RULES: refused (no AllocView); +AllocView: Program == eager", f"refused(decomposition) / Program; max|Δ|={max(diffs):.3g}", "PASS")


# ---------------------------------------------------------------------------------------------------------
# C, D, E, F, I, J, K: refuse(stale/uninitialised) OR a Program equal to eager's post-mutation reading — never the stale value
# ---------------------------------------------------------------------------------------------------------


def _refuse_or_match(case: str, m: nn.Module, inputs, wrong_fn, expect_reason: str, label: str, rules=None):
    rep = _derive(m, inputs, rules=rules)
    want = _eager(m, inputs)
    wrong = wrong_fn(inputs, want)
    if not rep.ok:
        u = rep.unsupported
        assert u.reason == expect_reason, f"{case}: refused for {u.reason!r}, expected {expect_reason!r}: {u.hint}"
        _record(label, f"refuse({expect_reason}) or Program == eager post-mutation", f"refused: {_reason(rep)[:220]}", "PASS (fail-closed)",
                "derive refuses the alias read; eager semantics not silently substituted")
        return
    diffs = _check_program_matches(case, rep, m, inputs, want, wrong)
    _record(label, f"refuse({expect_reason}) or Program == eager post-mutation", f"Program; max|Δ|={max(diffs):.3g}", "PASS")


def test_C_pre_taken_view_read_after_mutation():
    inputs = _inputs(2, QKV, 6)

    def wrong(inputs, want):
        qkv0 = (inputs[0] + inputs[1]).float()
        return [qkv0[:, :NHD], qkv0[:, NHD:NHD + KVHD]]

    _refuse_or_match("C", C_StaleView(), inputs, wrong, "mutation", "C view of q taken before rotary, read after")


def test_D_overlapping_slice_read_after_mutation():
    inputs = _inputs(2, QKV, 7)

    def wrong(inputs, want):
        qkv0 = (inputs[0] + inputs[1]).float()
        return [qkv0[:, :NHD + KVHD], qkv0[:, NHD + KVHD:]]

    _refuse_or_match("D", D_OverlapSlice(), inputs, wrong, "mutation", "D overlapping slice qkv[:, :q+k] taken before, read after")


def test_E_base_read_after_view_mutation_negstalealias():
    rep = derive_report(ex.NegStaleAlias(dict(H=C["H"], NH=C["NH"], KVH=C["KVH"], D=C["D"], MAX_POS=C["MAX_POS"], ROPE_THETA=10000.0)),
                        (torch.empty(C["T"], C["H"], dtype=BF),), **PROF, name="NegStaleAlias")
    assert not rep.ok and rep.unsupported.reason == "mutation", rep.unsupported
    assert "stale" in rep.unsupported.hint, rep.unsupported.hint
    _record("E base qkv returned after rotary wrote q,k views (NegStaleAlias)", "refuse(mutation, stale alias)", f"refused: {_reason(rep)[:200]}",
            "PASS (fail-closed)")


def test_F_sibling_view_taken_after_mutation():
    inputs = _inputs(2, QKV, 8)

    def wrong(inputs, want):
        qkv0 = (inputs[0] + inputs[1]).float()
        return [qkv0[:, :NHD], qkv0[:, NHD + KVHD:]]

    _refuse_or_match("F", F_SiblingAfter(), inputs, wrong, "mutation", "F new view qkv[:, :q] taken AFTER rotary (base stale)")


def test_I_residual_view_before_fused_add_rms_norm_read_after():
    inputs = _inputs(3, C["H"], 9)

    def wrong(inputs, want):
        return [want[0], (inputs[2] + inputs[1]).float()]

    _refuse_or_match("I", I_FusedResView(), inputs, wrong, "mutation", "I view of residual taken before fused_add_rms_norm, read after")


def test_J_alloc_written_through_view_base_returned():
    inputs = _inputs(2, C["H"], 10)
    _refuse_or_match("J", J_AllocViewBase(), inputs, lambda i, w: None, "mutation", "J alloc written through same-shape view; BASE returned",
                     rules=_alloc_view_rules())


def test_K_alloc_view_before_write_read_after():
    inputs = _inputs(2, C["H"], 11)
    _refuse_or_match("K", K_AllocViewBefore(), inputs, lambda i, w: None, "mutation", "K view of alloc taken before rms_norm wrote the alloc; view read after",
                     rules=_alloc_view_rules())


# ---------------------------------------------------------------------------------------------------------
# X: what torch.export records, and that derive consumes it
# ---------------------------------------------------------------------------------------------------------


def _export(m: nn.Module, inputs):
    with torch.no_grad():
        return torch.export.export(m, tuple(torch.empty_like(t) for t in inputs), strict=False)


def test_X1_export_keeps_rule_owned_in_place_ops_in_place_no_functionalisation():
    ep = _export(A_Disjoint(), _inputs(2, QKV, 12))
    targets = [n.target for n in ep.graph.nodes if n.op == "call_function"]
    names = [str(t) for t in targets]
    inplace = [t for t in targets if isinstance(t, torch._ops.OpOverload) and t._schema.is_mutable]
    assert any("rotary_embedding" in s for s in names), names
    assert inplace and all("rotary_embedding" in str(t) for t in inplace), names
    assert not any(isinstance(t, torch._ops.HigherOrderOperator) for t in targets), names  # no auto_functionalized wrapper
    kinds = [s.kind.name for s in ep.graph_signature.output_specs]
    assert kinds == ["USER_OUTPUT"] * 3, kinds
    assert not ep.range_constraints, ep.range_constraints  # fully static specialisation
    _record("X1 export facts (A)", "in-place op kept (no auto_functionalized); USER_OUTPUT-only; no range constraints",
            f"targets={sorted(set(names))}; output_specs={kinds}; range_constraints={dict(ep.range_constraints)}", "PASS")


class X4_AtenBufferInPlace(nn.Module):
    """Control for X2: the SAME buffer mutation through an ATen in-place op instead of a custom mutable op."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scratch", torch.zeros(C["T"], C["H"], dtype=BF))

    def forward(self, x):
        self.scratch.add_(x)
        return self.scratch


def test_X2_buffer_written_by_a_custom_mutable_op_is_not_in_the_export_signature_but_derive_refuses():
    """MEASURED export limitation: the training IR records a BUFFER_MUTATION output spec for ATen in-place ops on buffers (X4) but
    NOT for a custom op with a `Tensor(a!)` operand (vLLM's whole kernel interface) — `_translate`'s output-spec check cannot see
    it; derive's own `rebind` -> `is_weight_storage` refusal is what fails closed."""
    m = ex.NegMutatesWeight(T=C["T"], H=C["H"])
    ep = _export(m, [torch.empty(C["T"], C["H"], dtype=BF)])
    kinds = [s.kind.name for s in ep.graph_signature.output_specs]
    ctrl = _export(X4_AtenBufferInPlace(), [torch.empty(C["T"], C["H"], dtype=BF)])
    ctrl_kinds = [s.kind.name for s in ctrl.graph_signature.output_specs]
    rep = derive_report(m, (torch.empty(C["T"], C["H"], dtype=BF),), **PROF, name="NegMutatesWeight")
    assert not rep.ok and rep.unsupported.reason == "mutation", rep.unsupported
    ctrl_rep = derive_report(X4_AtenBufferInPlace(), (torch.empty(C["T"], C["H"], dtype=BF),), **PROF, name="X4")
    assert not ctrl_rep.ok and ctrl_rep.unsupported.reason == "mutation", ctrl_rep.unsupported
    b2m = dict(ep.graph_signature.buffers_to_mutate), dict(ctrl.graph_signature.buffers_to_mutate)
    verdict = "PASS (fail-closed by derive; export signature silent)" if "BUFFER_MUTATION" not in kinds + ctrl_kinds else "PASS (fail-closed; export records it)"
    _record("X2 buffer written by custom mutable op (NegMutatesWeight) vs ATen add_ on a buffer (X4)",
            "derive refuses both (mutation); export signature records the mutation",
            f"custom-op output_specs={kinds}; aten add_ output_specs={ctrl_kinds}; buffers_to_mutate={b2m}; derive: {_reason(rep)[:120]} / {_reason(ctrl_rep)[:100]}",
            verdict, "export limitation (torch 2.13 training IR, strict=False): NO buffer mutation — custom-op (a!) or ATen add_ — is surfaced in "
                     "graph_signature (buffers_to_mutate = {}); `_translate`'s output-spec check cannot fire; derive's own storage tracking is the guard")
    # load-bearing record of the measured limitation (flip these if torch starts recording mutations in the training IR)
    assert kinds == ["USER_OUTPUT"] and ctrl_kinds == ["USER_OUTPUT"], (kinds, ctrl_kinds)
    assert b2m == ({}, {}), b2m


def test_X3_user_inputs_written_by_a_custom_mutable_op_program_matches_eager_outputs_and_declares_the_effect():
    """`fused_add_rms_norm(x, r)` on two prescribed inputs, both returned. Export does not record USER_INPUT_MUTATION for the custom
    op (measured; see X2). derive rebinds both to the pure results and DECLARES the effect in `mutation_effects` ("the Input gates keep
    the pre-mutation value") — the Program's OUTPUTS must equal eager's returned values; the side effect on the caller's tensors is
    outside a pure Program and is what the declaration is for."""
    m = X_UserInputMutation()
    inputs = _inputs(2, C["H"], 13)
    ep = _export(m, inputs)
    kinds = [s.kind.name for s in ep.graph_signature.output_specs]
    rep = _derive(m, inputs)
    want = _eager(m, inputs)  # [normed(x+r), x+r]
    if not rep.ok:
        assert rep.unsupported.reason == "mutation", rep.unsupported
        _record("X3 prescribed inputs written by fused_add_rms_norm, both returned", "refuse(mutation) or outputs == eager + declared effect",
                f"output_specs={kinds}; refused: {_reason(rep)[:140]}", "PASS (fail-closed)")
        return
    wrong = [inputs[0].float(), inputs[1].float()]  # the pre-mutation Input values
    diffs = _check_program_matches("X3", rep, m, inputs, want, wrong)
    eff = rep.provenance["mutation_effects"]
    assert {e["input"] for e in eff} == set(ep.graph_signature.user_inputs), eff
    assert rep.provenance["returned_inputs"] == [], rep.provenance["returned_inputs"]  # the returns are the pure results, not the Inputs
    _record("X3 prescribed inputs written by fused_add_rms_norm, both returned", "outputs == eager post-mutation; effect declared per input",
            f"output_specs={kinds}; Program; max|Δ|={max(diffs):.3g}; mutation_effects={[e['input'] for e in eff]}", "PASS (declared effect)",
            "export limitation (no USER_INPUT_MUTATION for custom ops); derive declares the effect; caller-side aliasing of x/r after the call is outside the Program")


# ---------------------------------------------------------------------------------------------------------
# Y: plain ATen in-place / overlapping views without a rule -> refused
# ---------------------------------------------------------------------------------------------------------


class Y_AddInPlace(nn.Module):
    def forward(self, x, y):
        h = x + y
        h.add_(y)
        return h


class Y_MulInPlace(nn.Module):
    def forward(self, x, y):
        h = x + y
        h.mul_(y)
        return h


class Y_SliceWrite(nn.Module):
    def forward(self, x, y):
        h = x + y
        h[:, :4] = y[:, :4]
        return h


class Y_IndexPut(nn.Module):
    def forward(self, x, y):
        h = x + y
        idx = torch.tensor([0, 2])
        h.index_put_((idx,), y[:2])
        return h


class Y_AsStrided(nn.Module):
    def forward(self, x, y):
        h = x + y
        a = h.as_strided((C["T"], C["H"]), (0, 1))  # every row aliases row 0 (self-overlapping)
        return a + y


class Y_WeightDataWrite(_Norm):
    """R10 open item A6: a write through `.data` of a parameter inside forward (rebinding weight storage) must be refused."""

    def forward(self, x, y):
        self.weight.data.mul_(2.0)
        return x + y  # the weight write is the only effect


class Y_ParamInPlace(_Norm):
    def forward(self, x, y):
        self.weight.add_(1.0)
        return x + y


@pytest.mark.parametrize("cls", [Y_AddInPlace, Y_MulInPlace, Y_SliceWrite, Y_IndexPut, Y_AsStrided, Y_WeightDataWrite, Y_ParamInPlace])
def test_Y_unowned_in_place_and_overlapping_views_are_refused(cls):
    _test_Y(cls)


def _test_Y(cls):
    m = cls()
    inputs = _inputs(2, C["H"], 14)
    rep = _derive(m, inputs)
    assert not rep.ok, f"{cls.__name__}: derived a Program for an un-owned mutation / overlapping view"
    u = rep.unsupported
    if cls is Y_AsStrided:
        assert u.reason in ("unsupported_view", "decomposition", "numerics_unregistered"), u
    elif cls in (Y_WeightDataWrite, Y_ParamInPlace):
        assert u.reason in ("mutation", "hidden_state", "export"), u
    else:
        assert u.reason == "mutation", u
    _record(f"Y {cls.__name__}", "refused", f"refused: {_reason(rep)[:160]}", "PASS (fail-closed)")


# ---------------------------------------------------------------------------------------------------------
# T: Bet A's PRODUCTION carrier — `verity_cba::triton_launch(str kernel, Tensor(a!)[] tensors, str meta_json)` marks EVERY
# tensor argument as mutated (over-declared). The vLLM binding rules (`rules/vllm_bindings.py`) decide from the pinned kernel
# (name + source sha256 + constexprs + geometry) which operand the kernel actually writes. R10 open item A1: input_ptr /
# weight_ptr must stay readable after a launch; a kernel that is not the pinned one (another body, another name) must refuse.
# ---------------------------------------------------------------------------------------------------------

import json as _json  # noqa: E402

from verity_vllm.program.frontend.rules import triton_iface as _ti  # noqa: E402  (defines verity_cba::triton_launch, pure torch)

_INPLACE_KERNEL = "vllm.model_executor.determinism.batch_invariant._rms_norm_inplace_kernel"  # hypothetical: writes input_ptr


def _rms_pins():
    from verity_vllm.program.frontend.rules.vllm_bindings import RMS_PINS
    return RMS_PINS


def _rms_meta(R: int, N: int, *, kernel: str | None = None, src_sha256: str | None = None) -> str:
    pins = _rms_pins()
    meta = {"kernel": kernel or pins["kernel"], "src_sha256": src_sha256 or pins["src_sha256"], "triton": "test", "grid": [R],
            "scalars": {"input_row_stride": N, "output_row_stride": N, "n_cols": N, "eps": C["EPS"], **pins["constexprs"]},
            "tensor_args": ["input_ptr", "weight_ptr", "output_ptr"], "constexprs": list(pins["constexprs"]), "geometry": {"num_warps": pins["num_warps"]},
            "tensor_meta": [None, None, None]}
    return _json.dumps(meta)


def _cpu_triton_launch(kernel, tensors, meta_json):
    """Eager semantics of the carrier for the test: the pinned RMS kernel writes output_ptr; the hypothetical in-place kernel writes input_ptr."""
    meta = _json.loads(meta_json)
    x, w, out = tensors
    eps = float(meta["scalars"]["eps"])
    if kernel == _rms_pins()["kernel"]:
        out.copy_(_rms(_f32(x), w, eps))
    elif kernel == _INPLACE_KERNEL:
        x.copy_(_rms(_f32(x), w, eps))
    else:
        raise RuntimeError(f"no CPU semantics for {kernel}")


try:
    torch.library.register_kernel(f"{_ti._LIB_NAME}::triton_launch", "CPU", _cpu_triton_launch)
except RuntimeError as _e:
    if "already" not in str(_e).lower():
        raise


def _vllm_rules():
    from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset
    return vllm_ruleset()


class T1_TritonRmsInputReadable(_Norm):
    def forward(self, x, y):
        h = x + y
        out = torch.empty_like(h)
        torch.ops.verity_cba.triton_launch(_rms_pins()["kernel"], [h, self.weight, out], _rms_meta(C["T"], C["H"]))
        return out, h  # input_ptr read AFTER the launch: over-declared (a!) must not stale it; its value is the pre-launch one (eager agrees)


class T2_TritonUnpinnedKernelWritesInput(_Norm):
    def forward(self, x, y):
        h = x + y
        torch.ops.verity_cba.triton_launch(_INPLACE_KERNEL, [h, self.weight, h], _rms_meta(C["T"], C["H"], kernel=_INPLACE_KERNEL))
        return h


class T3_TritonRmsOtherBody(_Norm):
    def forward(self, x, y):
        h = x + y
        out = torch.empty_like(h)
        torch.ops.verity_cba.triton_launch(_rms_pins()["kernel"], [h, self.weight, out], _rms_meta(C["T"], C["H"], src_sha256="00" * 32))
        return out


class T4_TritonRmsOutputAliasBefore(_Norm):
    def forward(self, x, y):
        h = x + y
        out = torch.empty_like(h)
        ov = out.view(-1)
        torch.ops.verity_cba.triton_launch(_rms_pins()["kernel"], [h, self.weight, out], _rms_meta(C["T"], C["H"]))
        return ov.view(x.shape[0], C["H"])


class T5_TritonRmsInPlaceSameTensor(_Norm):
    def forward(self, x, y):
        h = x + y
        torch.ops.verity_cba.triton_launch(_rms_pins()["kernel"], [h, self.weight, h], _rms_meta(C["T"], C["H"]))
        return h


def test_T1_pinned_triton_rms_launch_leaves_input_ptr_readable_and_matches_eager():
    m = T1_TritonRmsInputReadable()
    inputs = _inputs(2, C["H"], 21)
    rep = _derive(m, inputs, rules=_vllm_rules())
    assert rep.ok, _reason(rep)
    assert "TritonRmsNorm[vllm-bi]_v1" in rep.provenance["rules_applied"], rep.provenance["rules_applied"]
    want = _eager(m, inputs)  # [rms(h), h]
    wrong = [want[1], want[0]]  # output read as the input / input read as the output
    diffs = _check_program_matches("T1", rep, m, inputs, want, wrong)
    assert rep.provenance["mutation_effects"] == []
    _record("T1 triton_launch(_rms_norm_kernel, [h, w, out]) — input_ptr returned after the launch", "Program == eager; h not stale (over-declared (a!) ignored for input_ptr/weight_ptr)",
            f"Program; max|Δ|={max(diffs):.3g}", "PASS")


def test_T2_unpinned_kernel_that_would_write_input_ptr_is_refused():
    m = T2_TritonUnpinnedKernelWritesInput()
    inputs = _inputs(2, C["H"], 22)
    rep = _derive(m, inputs, rules=_vllm_rules())
    assert not rep.ok, "a launch of an unpinned kernel derived to a Program"
    assert rep.unsupported.reason in ("mutation", "custom_op_without_binding"), rep.unsupported
    _record("T2 triton_launch of an unpinned kernel (hypothetical in-place RMS writing input_ptr)", "refused (no rule binds the kernel)",
            f"refused: {_reason(rep)[:200]}", "PASS (fail-closed)")


def test_T3_pinned_kernel_name_with_another_source_sha_is_refused():
    m = T3_TritonRmsOtherBody()
    inputs = _inputs(2, C["H"], 23)
    rep = _derive(m, inputs, rules=_vllm_rules())
    assert not rep.ok, "same kernel name, another body: derived to a Program"
    assert "sha256" in rep.unsupported.hint, rep.unsupported.hint
    _record("T3 pinned kernel name, different kernel-source sha256", "refused naming the pin", f"refused: {_reason(rep)[:200]}", "PASS (fail-closed)")


def test_T4_output_alias_taken_before_the_launch_read_after():
    inputs = _inputs(2, C["H"], 24)
    _refuse_or_match("T4", T4_TritonRmsOutputAliasBefore(), inputs, lambda i, w: None, "mutation",
                     "T4 view of the output allocation taken before the launch, read after", rules=_vllm_rules())


def test_T5_launch_with_output_ptr_is_input_ptr():
    inputs = _inputs(2, C["H"], 25)

    def wrong(inputs, want):
        return [(inputs[0] + inputs[1]).float()]  # the un-normalised h

    _refuse_or_match("T5", T5_TritonRmsInPlaceSameTensor(), inputs, wrong, "mutation", "T5 pinned RMS launch with output_ptr IS input_ptr (in place)",
                     rules=_vllm_rules())
