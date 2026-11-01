"""`TargetProfile.weight_only_calls` (the norm-chain rules of `vllm_bindings`): a Program of several forwards issues GemmaRMSNorm's
`weight.float() + 1.0` (`AddScalarBf16_v1{N,C=1}`) once per norm weight by default (`"once"`), every later forward reading that value, and
at every forward under a declared `"per-forward"`, as vLLM executes it.  Unset and `"once"` derive the same Program; `"once"` differs from
`"per-forward"` only by the repeated adds, its outputs are bit-equal to the per-forward Program's on edge words (NaN, +-inf, +-0, subnormal
weights and activations, w = -1), and `query.cross_call` finds no recompute in it (per-forward: one Call-level class, the repeated adds).

Run (pod venv): `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/program/test_weight_only_once.py`
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from verity.ml.circuits.codec import descriptor_id, program_digest
from verity.ml.circuits.refs import runs
from verity_vllm.program.frontend.rules.vllm_bindings import clear_observations, observe_export_facts, observe_target_profile
from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET, TargetProfile
from tests.program.test_norm_chain import BF, GemmaNorm, _bi_facts, _derive, _emitted, _evaluate

H = 16
FORWARDS = 3
ADD = f"AddScalarBf16_v1{{N={H},C=1.0}}"


class Forwards(nn.Module):
    """Two norms, each run in every one of three forwards: the request wrapper's shape, where each forward re-runs `weight.float() + 1.0`."""

    def __init__(self) -> None:
        super().__init__()
        self.a, self.b = GemmaNorm(H), GemmaNorm(H)

    def forward(self, x0, x1, x2):
        return tuple(self.b(self.a(x)) for x in (x0, x1, x2))


@pytest.fixture(autouse=True)
def _observed():
    clear_observations()
    observe_export_facts(_bi_facts())
    yield
    clear_observations()


def _derived(weight_only_calls: str | None, weights: tuple[torch.Tensor, torch.Tensor] | None = None):
    observe_target_profile(DEFAULT_TARGET.with_overrides(weight_only_calls=weight_only_calls))
    m = Forwards()
    if weights is not None:
        with torch.no_grad():
            m.a.weight.copy_(weights[0])
            m.b.weight.copy_(weights[1])
    xs = [torch.zeros(2, H, dtype=BF) for _ in range(FORWARDS)]
    rep = _derive(m, *xs)
    assert rep.unsupported is None, f"refused: {rep.unsupported}"
    return m, rep


def _specs(rep) -> list[str]:
    return [n.fn.id for n in rep.program.fn.body.nodes]


def _rows(rep) -> dict:
    return {"params": [[n, str(t)] for n, t in rep.program.fn.body.params],
            "rows": [{"i": k, "spec": descriptor_id(n.fn), "form": n.form, "name": n.name, "args": [[list(r) for r in runs(a.refs)] for a in n.args]}
                     for k, n in enumerate(rep.program.fn.body.nodes)]}


@pytest.mark.slow
def test_unset_and_once_derive_the_same_program_and_per_forward_adds_per_forward():
    _m, unset = _derived(None)
    _m, once = _derived("once")
    assert program_digest(unset.program) == program_digest(once.program) and _emitted(unset) == _emitted(once)
    _m, per = _derived("per-forward")
    assert _specs(per).count(ADD) == 2 * FORWARDS and _emitted(per).count(f"{ADD} x1") == 2 * FORWARDS


def test_once_issues_the_add_once_per_norm_weight():
    _m, per = _derived("per-forward")
    _m, once = _derived("once")
    assert _specs(once).count(ADD) == 2 and _emitted(once).count(f"{ADD} x1") == 2
    reads = [e for e in _emitted(once) if e.startswith(f"{ADD} x0 (reads ")]
    assert len(reads) == 2 * (FORWARDS - 1)
    assert sorted(s for s in _specs(per) if s != ADD) == sorted(s for s in _specs(once) if s != ADD)
    assert program_digest(once.program) != program_digest(per.program)


EDGE_BF16 = [0x7FC0, 0xFFC1, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x0001, 0x807F, 0x0080, 0x7F7F, 0xFF7F, 0xBF80, 0x3F80, 0x4000, 0xC2F7, 0x3C00]


def _bf16(words: list[int]) -> torch.Tensor:
    return torch.tensor(np.array(words, dtype=np.uint16).view(np.int16)).view(torch.bfloat16)


@pytest.mark.slow
def test_once_is_bit_equal_on_edge_words():
    rng = np.random.default_rng(57)
    for k in range(4):
        wa = _bf16(EDGE_BF16 if k == 0 else rng.integers(0, 1 << 16, H).tolist())
        wb = _bf16(EDGE_BF16[::-1] if k == 0 else [EDGE_BF16[i] if rng.random() < 0.5 else int(rng.integers(0, 1 << 16)) for i in range(H)])
        xs = [_bf16([EDGE_BF16[(i + 3 * j + k) % len(EDGE_BF16)] if (i + j) % 3 else int(rng.integers(0, 1 << 16)) for i in range(2 * H)]).reshape(2, H)
              for j in range(FORWARDS)]
        m, per = _derived("per-forward", (wa, wb))
        _m, once = _derived("once", (wa, wb))
        got_per, got_once = _evaluate(per, m, xs), _evaluate(once, m, xs)
        assert got_per.shape == got_once.shape == (FORWARDS * 2 * H,)
        assert np.array_equal(got_per, got_once), (k, np.flatnonzero(got_per != got_once)[:8])


def test_cross_call_finds_the_repeated_adds_only_per_forward():
    X = pytest.importorskip("verity.ml.definitions.cross_call")
    _m, per = _derived("per-forward")
    _m, once = _derived("once")
    r = X.check_program(_rows(per), strict=False)
    (a,) = r["recomputes"]
    add = next(n.fn for n in per.program.fn.body.nodes if n.fn.id == ADD)
    assert (a["definition"], a["level"], a["calls"]) == (descriptor_id(add), "call", 2 * (FORWARDS - 1))
    assert X.check_program(_rows(once))["ok"]


def test_the_knob_is_a_declared_static_that_keeps_undeclared_digests():
    e = TargetProfile()
    assert "weight_only_calls" not in e.to_json() and e.effective_weight_only_calls() == "once"
    o = e.with_overrides(weight_only_calls="per-forward")
    assert o.to_json()["weight_only_calls"] == "per-forward" and o.digest() != e.digest() and TargetProfile.from_json(o.to_json()) == o
    assert o.execution_label == e.execution_label and o.engine_args() == e.engine_args()          # not another execution
    for bad in ("twice", "ONCE", True, 1):
        with pytest.raises(ValueError, match="weight_only_calls"):
            TargetProfile(weight_only_calls=bad)
