"""The frontend rule for the STOCHASTIC token select: `vllm_iface.gumbel_topp_token_select(logits, seed, temperature, top_p,
pos, splits)` -> one `GumbelTopPTokenSelect_v1{V}(logits_row, top_p, seed, pos, temp, splits)` per logits row (verity.ml.definitions.registry.sampling).

What the op stands for (vLLM d9105ea80 `Sampler.sample` for a request whose `SamplingParams` are temperature T > 0, top_p < 1, top_k 0,
an explicit `seed`, no penalties / logit bias / min_p / logprobs):

    logits32 = f32(logits_bf16)                                 # `torch.empty_like(logits, dtype=float32).copy_(logits)`
    _temperature_kernel:      logits32 /= T                      # skipped when T in {0, 1}
    _apply_topp_split (S):    logits32[v] = keep_v ? logits32[v] : -inf   # `_topp_sb_{stats,step,mask}_kernel`, S splits
    _gumbel_sample_kernel:    y = logits32 + g(tl.randint(seed, pos))     # Gumbel-max; `pos` = positions[logits_indices]
    token = first index of max_v y[v]                            # block argmax + torch.argmax + gather

Operand roles (the coordinator's rulings): `seed` is a RUNTIME input of the Program (a forward
input of the wrapper, `request[r].seed` in the workload Program); `temperature`, `top_p` and `pos` are STRUCTURAL (the declared policy
and the step's row position, constants of the Program: `Const32[f32 bits]`, `Const64[pos]`); `S` (splits) is a DERIVED input of the
workload -- S_t = SplitsFor_v1(|live(t)|, num_SMs) (registry/topp_split.py) -- that the request wrapper reads from its request-owned
`splits` Input (int32 [T+1], one entry per sampling event) and the op takes as a 1-element int32 value collection: the 6th operand of
the composite, never a static of the Definition (the earlier `{V,S}` static form is gone; a literal S inside forward is hidden state and is
refused like a literal seed).  [single-request workloads] A single-request workload's wrapper passes
`splits=None` and `S` = SplitsFor_v1(1, num_SMs): the one live request makes S that constant at every event, and the rule emits it as
`Const32[S]`, the 4th constant of the row in operand order (after temperature), exactly as the fold emits the observed S.  Only the logits
row is a produced Program value; the sampled token is the family's bound output (`sampled_token_ids`).
"""
from __future__ import annotations

import struct

import torch

from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.torch_frontend import _tensor_type, shape_of
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import dtype_is, rows_of, shape
from verity_vllm.program.frontend.rules.vllm_iface import targets_for

BF16 = torch.bfloat16


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class GumbelTopPTokenSelectRule(OpRule):
    """`vllm_iface.gumbel_topp_token_select(logits bf16 [.., V], seed int64 [1], temperature: float, top_p: float, pos: int, splits int32 [1])`
    -> per row `GumbelTopPTokenSelect_v1{V}(row, Const32[top_p], seed[0], Const64[pos], Const32[temperature], splits[0])` (its
    `registry.sampling_vocab.SHARED_GREEDY` entry under the target's `sampler_construction = shared-greedy`), an I32 token id per row (the op's
    int32 result).  Refused: a temperature of 0 (that is the greedy `TokenSelect_v1`, not a stochastic select — do not
    relabel), a seed that is not a 64-bit 1-element value collection (the seed must be a Program input, never a literal), a non-static
    pos, a `splits` that is not a 32-bit 1-element value collection (S is the workload's derived value, read from the request's `splits`
    Input per event: `SplitsFor_v1(|live(t)|, num_SMs)`; a literal S is not a policy and not a geometry the Program may assume) unless
    `splits` is None and `S` is a launched split count: the registered constant (`topp_geometry.registered_splits`), where each row is
    `GumbelTopPTokenSelect_v2{V,S}(row, ..., Const32[S])` instead (the same operands; its top-p keep word as word gates at S)."""
    modes = frozenset({"profile"})
    info = RuleInfo("GumbelTopPTokenSelect[vllm]", 3, "op-binding",
                    applies_to="vllm_iface.gumbel_topp_token_select(Tensor logits, Tensor seed, float temperature, float top_p, int pos, Tensor? splits, int? S=None) -> Tensor",
                    preconditions=("logits is a bf16 value collection [.., V]", "seed is a 1-element 64-bit value collection (a Program input)",
                                   "temperature is a static float > 0 (0 is greedy: TokenSelect_v1, not this rule)", "0 < top_p <= 1 static",
                                   "pos >= 0 static", "splits is a 1-element 32-bit value collection (the request's derived-S Input entry for the event), "
                                   "or splits is None and S a static split count in {1, 2, 4, 8, 16, 32} (the registered split count)"),
                    evidence_domain="structural; numerics: GumbelTopPTokenSelect_v1{V} (exact-model-tested: difftest vs vLLM d9105ea80 "
                                    "_temperature_kernel / _topp_sb_* / _gumbel_sample_kernel on L40S, 72/72 rows bit-exact; a GPU fixture "
                                    "at every split count S = 1..32, tests/program/test_topp_split_geometry.py)")

    def targets(self):
        return targets_for("gumbel_topp_token_select")

    @staticmethod
    def _args(node):
        a = list(node.args) + [None] * 7
        kw = node.kwargs or {}
        logits = kw.get("logits", a[0])
        seed = kw.get("seed", a[1])
        temperature = kw.get("temperature", a[2])
        top_p = kw.get("top_p", a[3])
        pos = kw.get("pos", a[4])
        splits = kw.get("splits", a[5])
        S = kw.get("S", a[6])
        return logits, seed, temperature, top_p, pos, splits, S

    def applies(self, node, ctx):
        logits, seed, temperature, top_p, pos, splits, S = self._args(node)
        if not isinstance(logits, torch.fx.Node) or not isinstance(ctx.val(logits), Coll):
            return "logits is not a value collection"
        if not dtype_is(ctx, logits, BF16):
            return f"numerics_unregistered: logits dtype {ctx.dtype_of(logits)}; GumbelTopPTokenSelect_v1 is defined over bf16 logits"
        if not shape(ctx, logits):
            return "logits is a scalar"
        if not isinstance(seed, torch.fx.Node) or not isinstance(ctx.val(seed), Coll):
            return "seed is not a value collection (the seed must be a Program input, not a literal)"
        sshape, sw = shape_of(ctx.val(seed).type)
        if sw != 64 or int(torch.Size(sshape).numel()) != 1:
            return f"seed is not a 1-element 64-bit value ({sshape}, {sw}-bit)"
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            return f"temperature {temperature!r} is not a static number"
        if float(temperature) <= 0.0:
            return f"temperature {temperature!r}: 0 is the greedy TokenSelect_v1 (not a stochastic select); negative is not a policy"
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not (0.0 < float(top_p) <= 1.0):
            return f"top_p {top_p!r} is not a static number in (0, 1]"
        if isinstance(pos, bool) or not isinstance(pos, int) or pos < 0:
            return f"pos {pos!r} is not a static non-negative int"
        if splits is None:
            from verity.ml.definitions.registry.sampling import SPLIT_COUNTS
            if isinstance(S, bool) or not isinstance(S, int) or S not in SPLIT_COUNTS:
                return f"splits is None but S {S!r} is not a launched split count {SPLIT_COUNTS} (the registered constant)"
            return True
        if S is not None:
            return "both a splits entry and a constant S: one form per event"
        if not isinstance(splits, torch.fx.Node) or not isinstance(ctx.val(splits), Coll):
            return ("splits is not a value collection: S is the workload's derived split count (SplitsFor_v1(|live|, num_SMs)), read from the "
                    "request's `splits` Input per sampling event -- never a literal of the wrapper")
        pshape, pw = shape_of(ctx.val(splits).type)
        if pw != 32 or int(torch.Size(pshape).numel()) != 1:
            return f"splits is not a 1-element 32-bit value ({pshape}, {pw}-bit)"
        return True

    def emit(self, node, ctx):
        logits, seed, temperature, top_p, pos, splits, S = self._args(node)
        x = ctx.coll(logits)
        rows, lead = rows_of(x)
        V = rows[0].type.n
        from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile  # vllm_bindings imports the rule modules
        # a constant split count (the registered S): the select specialised on it, its keep word as word gates, the same
        # six operands (the Const32[S] still read, in operand order)
        fn = ctx.kind("token_select_stochastic", node, V=V, CONSTRUCTION=active_target_profile().effective_sampler_construction()) \
            if splits is not None else ctx.kind("token_select_stochastic_split", node, V=V, S=int(S))
        hint = ctx.hint_name(node)
        seed_c = ctx.coll(seed).reshape(Array(1, Value(64)))[0]
        splits_c = None if splits is None else ctx.coll(splits).reshape(Array(1, Value(32)))[0]   # the event's S: the `splits` Input entry
        tbits, pbits = f32_bits(temperature), f32_bits(top_p)
        # One Const triple PER ROW, emitted in OPERAND order (top_p, pos, temperature) right before the row's select: the same
        # instance sequence the fold emits for the observed sampler chain (`patterns.TokenSelectGumbelTopP` -> `fold._const`, one
        # Const per Compute operand), so `cb_a.compare` sees canonical-equal steps.  Per row because the fold's rows are per request
        # (their positions differ); the request wrapper samples one row per step anyway.
        outs = []
        for i, r in enumerate(rows):
            topp_c = ctx.call(ctx.kind("const", node, W=32, BITS=pbits), node=node, rule=self, name=f"{hint}/top_p")
            pos_c = ctx.call(ctx.kind("const", node, W=64, BITS=int(pos) & 0xFFFFFFFFFFFFFFFF), node=node, rule=self, name=f"{hint}/pos")
            temp_c = ctx.call(ctx.kind("const", node, W=32, BITS=tbits), node=node, rule=self, name=f"{hint}/temperature")
            s_c = splits_c if splits_c is not None else ctx.call(ctx.kind("const", node, W=32, BITS=int(S)), node=node, rule=self, name=f"{hint}/splits")
            outs.append(ctx.call(fn, r, topp_c, seed_c, pos_c, temp_c, s_c, node=node, rule=self, name=f"{hint}[{i}]"))
        from verity.ml.circuits.refs import array_of
        out = array_of(outs).reshape(_tensor_type(lead, 32)) if lead else outs[0]
        ctx.define(node, out)
        S_rec = ("Program input (derived: SplitsFor_v1(|live(t)|, num_SMs), the request's `splits` entry for this event)" if splits_c is not None
                 else f"Const32[{int(S)}] (the registered constant SplitsFor_v1(1, num_SMs))")
        ctx.record(self, node, {"V": V, "S": S_rec,
                                "rows": len(rows), "temperature": float(temperature), "temperature_bits": f"0x{tbits:08x}",
                                "top_p": float(top_p), "top_p_bits": f"0x{pbits:08x}", "pos": int(pos), "seed": "Program input (runtime)",
                                "torch_dtype": str(ctx.dtype_of(node)), "carried_as": "I32"},
                   emitted=f"{fn.id} x{len(rows)} (+ per row: Const32[top_p], Const64[pos], Const32[temperature]"
                           + (", in operand order; splits = the Input entry)" if splits_c is not None else ", Const32[S], in operand order)"))


VLLM_SAMPLING_RULES = (GumbelTopPTokenSelectRule(),)

__all__ = ["GumbelTopPTokenSelectRule", "VLLM_SAMPLING_RULES", "f32_bits"]
