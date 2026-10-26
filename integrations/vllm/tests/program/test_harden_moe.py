"""R11 cb-harden (2) — MoE routing independence of the Bet A circuit builder.

The property: the Program Bet A produces for a mixture-of-experts block must be INDEPENDENT of the routing decisions taken on
the example input used at export time; the per-token expert selection must appear as a VALUE of the Program, never as its
structure (which experts / which rows are wired). Two exports with inputs that route differently must give one digest; the
Program must evaluate correctly on a third input with yet another routing.

What Bet A can express today (measured here, not assumed):
    M1  Mixtral-style sparse MoE (`derive_examples.NegMoE`: softmax -> topk -> expert weights GATHERED by the routing index)
        is REFUSED, and the refusal is structural at the routing (the same reason and node for two example inputs). Since no
        Program exists, nothing can be baked in; the routing-independence property is vacuously preserved (fail-closed).
    M2  the padded sparse form (top-k mask as a value, ALL experts run on ALL tokens, mask * expert output) is refused at
        `aten.topk` — the missing piece is a registered kind for top-k selection (routing as a value), not a structural one.
    M3  the dense-mixture form (softmax routing weights as values, all experts on all tokens: "capacity = every token, every
        expert present") DERIVES in reference mode; digest independent of the example input's values (two inputs whose argmax
        routes differ -> one digest); evaluates on a third input against eager; the routing weights are Program values
        (SoftmaxRef calls + the gate Linear), and no rule application recorded an input value.
    M4  the top-1 (Switch) form with `argmax` + `eq` one-hot masks: refuse-or-match (records the missing kind if refused).

Applicability condition recorded for the report: Bet A's only derivable MoE form is the all-experts / all-tokens padded
mixture; sparse top-k routing (Mixtral / OLMoE-style gather) is not applicable to Bet A today. (The eager pipeline's MoE
constructor — `src/veritor/constructors/moe.py` — commits the `top_k` route ids per token as transcript data, i.e. routing IS
a value there; Bet A has no counterpart kind yet.)

Run (pod venv, CPU): `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. nice -n 10 python -m pytest -q -p no:cacheprovider
tests/program/test_harden_moe.py`; `HARDEN_OUT=<dir>` writes `moe_results.json`.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend import derive_examples as ex
from verity_vllm.program.kernels.serve3_reference import materialize

REF = dict(mode="reference")
T, H, I, E, K = 4, 8, 16, 4, 2

RESULTS: list[dict] = []


def _record(case, expected, observed, verdict, cause="", **extra):
    RESULTS.append({"case": case, "expected": expected, "observed": observed, "verdict": verdict, "cause": cause, **extra})


@pytest.fixture(scope="module", autouse=True)
def _dump_results():
    yield
    out = os.environ.get("HARDEN_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(RESULTS, open(os.path.join(out, "moe_results.json"), "w"), indent=1)


def _reason(rep) -> str:
    u = rep.unsupported
    return f"Unsupported({u.reason}) [{u.node}] @ {u.module_path}: {u.hint}"


def _f32_bits(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t.detach().contiguous().view(torch.int32).numpy().astype(np.uint32).reshape(-1)]


def _evaluate(rep, m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    tensors = dict(m.named_parameters())
    tensors.update(dict(m.named_buffers()))
    wbits: list[int] = []
    for w in rep.provenance["weights"]:
        wbits += _f32_bits(tensors[w["name"]])
    t = materialize(rep.program, [_f32_bits(x), wbits])
    flat = np.array([np.uint32(t[g]) for g in rep.program.circuit.root_outputs()], dtype=np.uint32)
    return torch.from_numpy(flat.astype(np.int32)).view(torch.float32).reshape(T, H)


def _routes(m, x) -> list[list[int]]:
    """Which experts the router would choose per token (top-k of the softmax): what a routing-baked Program would depend on."""
    with torch.no_grad():
        return torch.topk(F.softmax(m.gate(x), dim=-1), K, dim=-1).indices.tolist()


def _inputs_with_different_routing(m) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three inputs whose top-k routes differ pairwise for at least one token (searched, then asserted)."""
    xs = []
    for seed in range(50):
        x = torch.randn(T, H, generator=torch.Generator().manual_seed(100 + seed)) * 1.5
        if all(_routes(m, x) != _routes(m, y) for y in xs):
            xs.append(x)
        if len(xs) == 3:
            return xs[0], xs[1], xs[2]
    raise AssertionError("test gap: could not find three inputs with pairwise different routings")


# ---------------------------------------------------------------------------------------------------------
# source modules
# ---------------------------------------------------------------------------------------------------------


class Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.ModuleList([nn.Linear(H, I, bias=False) for _ in range(E)])
        self.w2 = nn.ModuleList([nn.Linear(I, H, bias=False) for _ in range(E)])

    def expert(self, e: int, x):
        return self.w2[e](F.relu(self.w1[e](x)))


class PaddedTopKMoE(Experts):
    """Padded sparse form: every expert runs on every token; the top-k mask (a value) selects. Routing enters as a value."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(H, E, bias=False)

    def forward(self, x):
        probs = F.softmax(self.gate(x), dim=-1)
        w, idx = torch.topk(probs, K, dim=-1)
        mask = torch.zeros_like(probs).scatter(1, idx, w)  # [T, E]: routing weights at the chosen experts, 0 elsewhere
        out = mask[:, 0:1].expand(T, H) * self.expert(0, x)
        for e in range(1, E):
            out = out + mask[:, e:e + 1].expand(T, H) * self.expert(e, x)
        return out


class DenseMixtureMoE(Experts):
    """All experts present, capacity = every token: `sum_e softmax(gate(x))[:, e] * expert_e(x)`; routing weights are values."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(H, E, bias=False)

    def forward(self, x):
        probs = F.softmax(self.gate(x), dim=-1)
        out = probs[:, 0:1].expand(T, H) * self.expert(0, x)
        for e in range(1, E):
            out = out + probs[:, e:e + 1].expand(T, H) * self.expert(e, x)
        return out


class Top1SwitchMoE(Experts):
    """Top-1 routing as a value: one-hot(argmax) masks; all experts run on all tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(H, E, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        sel = torch.argmax(logits, dim=-1)  # [T] int64
        out = (sel == 0).to(torch.float32)[:, None].expand(T, H) * self.expert(0, x)
        for e in range(1, E):
            out = out + (sel == e).to(torch.float32)[:, None].expand(T, H) * self.expert(e, x)
        return out


# ---------------------------------------------------------------------------------------------------------
# M1: the repo's sparse MoE is refused structurally and deterministically
# ---------------------------------------------------------------------------------------------------------


def test_M1_sparse_gather_moe_is_refused_at_the_routing_for_every_example_input():
    torch.manual_seed(0)
    m = ex.NegMoE(H=H, I=I, E=E, k=K)
    reps = [derive_report(m, (torch.empty(T, H),), **REF, name="NegMoE") for _ in range(2)]
    for rep in reps:
        assert not rep.ok, "a sparse-gather MoE derived to a Program: its structure would encode the example input's routing"
    u0, u1 = reps[0].unsupported, reps[1].unsupported
    assert (u0.reason, u0.node) == (u1.reason, u1.node), (str(u0), str(u1))  # deterministic refusal
    assert u0.reason in ("dynamic_control_flow", "side_input", "decomposition", "numerics_unregistered"), str(u0)
    # the refusal is AT the routing: the refusing node is the top-k / the routing-index gather / the routed weight read
    assert any(s in str(u0.node) for s in ("topk", "index", "getitem", "select", "gather", "softmax")), str(u0)
    _record("M1 NegMoE (softmax->topk->gather experts by index)", "refused structurally at the routing, deterministic",
            f"refused: {_reason(reps[0])[:260]}", "PASS (fail-closed)", "no Program exists -> routing cannot be baked in")


# ---------------------------------------------------------------------------------------------------------
# M2: padded sparse form refused at topk (missing kind), never a Program with the mask folded away
# ---------------------------------------------------------------------------------------------------------


def test_M2_padded_topk_mask_form_refused_at_topk_or_matches_eager():
    torch.manual_seed(1)
    m = PaddedTopKMoE()
    xa, xb, xc = _inputs_with_different_routing(m)
    reps = [derive_report(m, (torch.empty_like(x),), **REF, name="PaddedTopKMoE") for x in (xa, xb)]
    if not reps[0].ok:
        u = reps[0].unsupported
        assert (u.reason, u.node) == (reps[1].unsupported.reason, reps[1].unsupported.node)
        assert "topk" in str(u.node) or "scatter" in str(u.node), str(u)
        _record("M2 padded top-k mask MoE (all experts, mask value)", "Program with one digest for A/B, == eager on C; or refused at topk",
                f"refused: {_reason(reps[0])[:220]}", "PASS (fail-closed)", "missing kind: top-k selection as a value (not a derive bug)")
        return
    assert reps[0].digest == reps[1].digest, "digest depends on the example input (routing baked in)"
    got = _evaluate(reps[0], m, xc)
    with torch.no_grad():
        want = m(xc)
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-5), (got - want).abs().max()
    _record("M2 padded top-k mask MoE", "one digest for A/B; == eager on C", f"Program {reps[0].digest[:12]}; max|Δ|={(got - want).abs().max():.3g}", "PASS")


# ---------------------------------------------------------------------------------------------------------
# M3: the dense mixture — routing weights are values, digest independent of the example input, eager match on a third input
# ---------------------------------------------------------------------------------------------------------


def test_M3_dense_mixture_digest_independent_of_example_routing_and_matches_eager():
    torch.manual_seed(2)
    m = DenseMixtureMoE()
    xa, xb, xc = _inputs_with_different_routing(m)
    ra, rb = _routes(m, xa), _routes(m, xb)
    assert ra != rb
    # export with CONCRETE example inputs (not meta): derive must not read their values
    rep_a = derive_report(m, (xa,), **REF, name="DenseMixtureMoE")
    rep_b = derive_report(m, (xb,), **REF, name="DenseMixtureMoE")
    assert rep_a.ok, _reason(rep_a)
    assert rep_b.ok, _reason(rep_b)
    assert rep_a.digest == rep_b.digest, f"digest differs with the example input's routing: {rep_a.digest} vs {rep_b.digest}"
    # routing is a value: the gate Linear and the softmax are Program calls; every expert is present (E w1 + E w2 Linears)
    rules = [a["rule"] for a in rep_a.provenance["applications"]]
    assert rules.count("Softmax[reference]_v1") == 1, rules
    assert rules.count("Linear[reference]_v2") == 1 + 2 * E, rules.count("Linear[reference]_v2")
    roles = {w["name"] for w in rep_a.provenance["weights"]}
    assert roles == {"gate.weight"} | {f"w1.{e}.weight" for e in range(E)} | {f"w2.{e}.weight" for e in range(E)}, roles
    # the provenance of the two derivations is identical up to the example-input description (values never enter it)
    pa, pb = dict(rep_a.provenance), dict(rep_b.provenance)
    assert pa["applications"] == pb["applications"] and pa["weights"] == pb["weights"]
    # the third input (a third routing) evaluates against eager
    got = _evaluate(rep_a, m, xc)
    with torch.no_grad():
        want = m(xc)
    d = (got - want).abs().max().item()
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-5), d
    _record("M3 dense mixture (softmax weights x all experts)", "one digest for inputs A/B with different top-k routes; == eager on C",
            f"digest {rep_a.digest[:16]} for both; routes A={ra} B={rb}; max|Δ| on C={d:.3g}", "PASS",
            "routing weights are SoftmaxRef values; applicability: all E experts present, capacity = all T tokens")


# ---------------------------------------------------------------------------------------------------------
# M4: top-1 one-hot form: refuse (records the missing kind) or match
# ---------------------------------------------------------------------------------------------------------


def test_M4_top1_onehot_form_refuse_or_match():
    torch.manual_seed(3)
    m = Top1SwitchMoE()
    xa, xb, xc = _inputs_with_different_routing(m)
    reps = [derive_report(m, (x,), **REF, name="Top1SwitchMoE") for x in (xa, xb)]
    if not reps[0].ok:
        u = reps[0].unsupported
        assert (u.reason, u.node) == (reps[1].unsupported.reason, reps[1].unsupported.node)
        _record("M4 top-1 one-hot (argmax -> eq -> f32 mask)", "one digest for A/B, == eager on C; or refused at the mask construction",
                f"refused: {_reason(reps[0])[:220]}", "PASS (fail-closed)", "missing kind for the mask (argmax/eq/cast chain), not a derive bug")
        return
    assert reps[0].digest == reps[1].digest
    got = _evaluate(reps[0], m, xc)
    with torch.no_grad():
        want = m(xc)
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-5), (got - want).abs().max()
    _record("M4 top-1 one-hot", "one digest for A/B; == eager on C", f"Program {reps[0].digest[:12]}", "PASS")
