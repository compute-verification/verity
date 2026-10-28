"""A tensor-parallel rank's stochastic Program takes the TP1 Program's inputs, then the peers.  `vllm_meta.with_peers` wraps a
`make_wrapper` module for the rank's export: the stochastic wrappers' `seed` and `splits` come after the prompt, in the TP1 order, and
the peers (`peers_<k>`, one per collective) after them, so the replay and the manifest find the sampler's inputs by the names TP1 gives
them (`check.replay.index.SEED_PARAM`, `query.manifest.format.SPLITS_PARAM`).  The plain `WithPeers` took the prompt alone, and the seed
landed in the peers: derive_step failed with `forward() missing 1 required positional argument: 'seed'`.  The
greedy rank Program is the `WithPeers` one, its digest unchanged.

The model is a stand-in for a TP rank's vLLM model (`forward(input_ids, positions)`, `compute_logits`) with one all-reduce per forward
on the rank's `CollectiveBus`; the probe is `vllm_meta.probe_collectives` without vLLM's forward context.  CPU, meta tensors; no vLLM.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from verity.ml.circuits.codec import encode_program, program_digest
from verity_vllm.program.frontend import derive_examples as ex
from verity_vllm.program.frontend import vllm_meta as vm
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.export_ops import CollectiveBus, ensure_ops
from verity_vllm.program.frontend.rules import DEFAULT_RULES
from verity_vllm.program.frontend.rules.base import RuleSet
from verity_vllm.program.frontend.rules.vllm_bindings.collectives import AllReduce2Rule
from verity_vllm.program.frontend.rules.vllm_sampling import VLLM_SAMPLING_RULES
from verity.ml.definitions.registry import b1

PROF = {"mode": "profile", "profile": "b1-eager-v3"}
LP, T = 3, 2
STOCH = {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": "derived"}
SINGLE = dict(STOCH, splits=32, geometry={"sampler_batch": 1, "num_sms": 84})
SELECT = "GumbelTopPTokenSelectSharedGreedy_v1"


def _ruleset() -> RuleSet:
    """The default rules, the sampling rule and the world-2 all-reduce (the full vLLM ruleset's BI norm rule needs export facts the
    stand-in model does not have)."""
    return RuleSet("default+stoch+tp", list(DEFAULT_RULES.rules) + list(VLLM_SAMPLING_RULES) + [AllReduce2Rule()], list(DEFAULT_RULES.families))


def _config() -> dict:
    C = dict(b1.B0_CONFIG)
    C.update(NL=1, V=64, MAX_POS=64)
    return C


class _ToyRankLM(nn.Module):
    """vLLM's model interface with a row-parallel all-reduce of the final hidden state on `bus` (None: the TP1 model)."""

    def __init__(self, C: dict, bus: CollectiveBus | None = None) -> None:
        super().__init__()
        self.lm = ex.CausalLM(C)
        self.bus = bus

    def forward(self, input_ids, positions):
        h = self.lm.model(input_ids, positions, False, None)
        return h if self.bus is None else self.bus.all_reduce(h)

    def compute_logits(self, hidden):
        return self.lm.lm_head(hidden)


def _wrapper(kind: str, sampling: dict, bus: CollectiveBus | None):
    with torch.device("meta"):
        return vm.make_wrapper(kind, _ToyRankLM(_config(), bus), pos0=LP, M=1, LP=LP, T=T, sampling=sampling)


def _probe(mod, inputs, bus: CollectiveBus) -> None:
    bus.reset("probe")
    with torch.no_grad(), torch.device("meta"):
        torch.export.export(mod, tuple(inputs), strict=False)


def _derive(mod, inputs):
    rep = derive_report(mod, tuple(inputs), **PROF, config=_config(), rules=_ruleset(), name="vLLMRank[Toy]")
    assert rep.ok, rep.unsupported
    return rep


def _rank(kind: str, sampling: dict, rank: int = 0):
    ensure_ops()
    bus = CollectiveBus(rank=rank, world=2)
    mod, inputs = _wrapper(kind, sampling, bus)
    _probe(mod, inputs, bus)
    wrapped, all_inputs = vm.with_peers(mod, inputs, bus)
    return wrapped, all_inputs, bus, len(inputs)


def _params(rep) -> list[tuple[str, str]]:
    params, _ret = rep.flat_definition.signature()
    return [(n, str(t).replace(" ", "")) for n, t in params]


def _base(n) -> str:
    return getattr(n.fn, "base_id", None) or n.fn.id.split("[")[0].split("{")[0]


@pytest.mark.slow
@pytest.mark.parametrize("kind,sampling,form", [("request", STOCH, vm.SplitsWithPeers), ("step", STOCH, vm.SplitsWithPeers),
                                                ("request", SINGLE, vm.SeededWithPeers), ("step", SINGLE, vm.SeededWithPeers)],
                         ids=["request-derived-S", "step-derived-S", "request-single-S", "step-single-S"])
def test_a_stochastic_rank_program_takes_the_tp1_inputs_then_the_peers(kind, sampling, form):
    wrapped, inputs, bus, n_own = _rank(kind, sampling)
    assert type(wrapped) is form and len(bus.record) == (1 if kind == "step" else T + 1)
    rank = _derive(wrapped, inputs)
    tp1 = _derive(*_wrapper(kind, sampling, None))
    own, tp1_params = _params(rank), _params(tp1)
    n_peer = len(bus.record)
    # the rank Program's parameters: the TP1 Program's own inputs (the step wrapper's first one named `prompt`, as every rank Program's),
    # one bf16 peer per collective, then the TP1 Program's weights (their fields under the wrapper's `inner.`, as every rank Program's)
    assert [n for n, _ in own[:n_own]] == ["prompt", "seed", "splits"][:n_own]
    assert own[1:n_own] == tp1_params[1:n_own] and own[0][1] == tp1_params[0][1]
    assert [n for n, _ in own[n_own:n_own + n_peer]] == [f"peers_{k}" for k in range(n_peer)]
    assert all(t == f"Array<{r['shape'][0]},Array<{r['shape'][1]},Value<16>>>" for (_n, t), r in zip(own[n_own:n_own + n_peer], bus.record))
    assert [(n, t.replace("{inner.", "{").replace(",inner.", ",")) for n, t in own[n_own + n_peer:]] == tp1_params[n_own:]
    # the selects are TP1's (operands: row, top_p, seed, pos, temperature, S), each reading the seed at parameter 1 and (derived S) its
    # event's entry of `splits` at parameter 2
    sel = [n for n in rank.flat_definition.body.nodes if _base(n).startswith("GumbelTopPTokenSelect")]
    sel1 = [n for n in tp1.flat_definition.body.nodes if _base(n).startswith("GumbelTopPTokenSelect")]
    assert [n.fn.id for n in sel] == [n.fn.id for n in sel1] and len(sel) == (1 if kind == "step" else T + 1)
    for s_i, n in enumerate(sel):
        assert n.args[2].refs[0][:2] == ("p", 1), (s_i, n.args[2].refs)
        if n_own == 3:
            assert n.args[5].refs[0] == ("p", 2, s_i), (s_i, n.args[5].refs)
    assert sum(1 for n in rank.flat_definition.body.nodes if _base(n) == "AllReduce2_v1") == sum(r["shape"][0] for r in bus.record)
    assert rank.program.fn.bindings["SRC"]["class"] == f"verity-vllm/with-peers/v1[{type(wrapped.inner).DERIVE_SOURCE}]"


def test_the_seed_and_splits_are_inputs_on_every_rank():
    """Two seeds are one Program on a rank as on TP1 (the seed and S are bound at run time); the two ranks' Programs differ only where
    their collectives place their own partials."""
    a, ia, _, _ = _rank("request", STOCH, rank=0)
    b, ib, _, _ = _rank("request", STOCH, rank=1)
    ra, rb = _derive(a, ia), _derive(b, ib)
    assert [n for n, _ in _params(ra)] == [n for n, _ in _params(rb)]
    assert ra.digest == _derive(*_rank("request", STOCH, rank=0)[:2]).digest
    assert ra.digest != rb.digest


def test_the_greedy_rank_program_is_the_with_peers_one():
    ensure_ops()
    bus = CollectiveBus(rank=0, world=2)
    mod, inputs = _wrapper("request", None, bus)
    _probe(mod, inputs, bus)
    wrapped, all_inputs = vm.with_peers(mod, inputs, bus)
    assert type(wrapped) is vm.WithPeers
    rep = _derive(wrapped, all_inputs)
    old = _derive(vm.WithPeers(mod, bus), tuple(inputs) + tuple(bus.example_peers()))
    assert rep.digest == old.digest == program_digest(encode_program(rep.program))
    assert rep.digest == "42d4f7c1c5d41a7f6e0f66c8b2aa498e5ccd7570eb58526ebc000f61e8e67467b2ff2fd6d62014d6beffee7eeebca2430d5c35ba68987af03a42dfddd83c7f5a"


@pytest.mark.parametrize("n", [0, 4])
def test_with_peers_refuses_a_wrapper_it_does_not_know(n):
    bus = CollectiveBus(rank=0, world=2)
    with pytest.raises(ValueError, match="1 to 3 inputs of its own"):
        vm.with_peers(nn.Identity(), tuple(torch.empty(1, device="meta") for _ in range(n)), bus)
