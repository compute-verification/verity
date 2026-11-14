"""The frontend rule for the STOCHASTIC token select: `vllm_iface.gumbel_topp_token_select(logits, seed, temperature, top_p,
pos, S)` -> one `GumbelTopPTokenSelect_v2{V,S}(logits_row, top_p, seed, pos, temp, Const32[S])` per logits row
(verity.definitions.registry.sampling).

What the op stands for (vLLM d9105ea80 `Sampler.sample` for a request whose `SamplingParams` are temperature T > 0, top_p < 1, top_k 0,
an explicit `seed`, no penalties / logit bias / min_p / logprobs):

    logits32 = f32(logits_bf16)                                 # `torch.empty_like(logits, dtype=float32).copy_(logits)`
    _temperature_kernel:      logits32 /= T                      # skipped when T in {0, 1}
    _apply_topp_split (S):    logits32[v] = keep_v ? logits32[v] : -inf   # `_topp_sb_{stats,step,mask}_kernel`, S splits
    _gumbel_sample_kernel:    y = logits32 + g(tl.randint(seed, pos))     # Gumbel-max; `pos` = positions[logits_indices]
    token = first index of max_v y[v]                            # block argmax + torch.argmax + gather

Operand roles: `seed` is a RUNTIME input of the Program (a forward input of the wrapper, `request[r].seed` in the workload Program);
`temperature`, `top_p` and `pos` are STRUCTURAL (the declared policy and the step's row position, constants of the Program:
`Const32[f32 bits]`, `Const64[pos]`); `S` is the registered split count (`topp_geometry.registered_splits`), the one every served
launch runs at (`engine.topp_pin`), emitted as `Const32[S]`, the 4th constant of the row in operand order (after temperature),
exactly as the fold emits the observed S.  No S is a value of the live batch.  Only the logits row is a produced Program value;
the sampled token is the family's bound output (`sampled_token_ids`).
"""
from __future__ import annotations

import struct

import torch
from verity.circuits.refs import Coll, array_of
from verity.circuits.types import Array, Value
from verity.definitions.registry.sampling import SPLIT_COUNTS

from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import dtype_is, rows_of, shape
from verity_vllm.program.frontend.rules.vllm_iface import targets_for
from verity_vllm.program.frontend.strided_views import _tensor_type
from verity_vllm.program.frontend.value_types import shape_of

BF16 = torch.bfloat16


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class GumbelTopPTokenSelectRule(OpRule):
    """`vllm_iface.gumbel_topp_token_select(logits bf16 [.., V], seed int64 [1], temperature: float, top_p: float, pos: int, S: int)`
    -> per row `GumbelTopPTokenSelect_v2{V,S}(row, Const32[top_p], seed[0], Const64[pos], Const32[temperature], Const32[S])`, an I32
    token id per row (the op's int32 result).  Refused: a temperature of 0 (that is the greedy `TokenSelect_v1`, not a stochastic
    select — do not relabel), a seed that is not a 64-bit 1-element value collection (the seed must be a Program input, never a
    literal), a non-static pos, and an S that is not a launched split count (the registered constant)."""
    modes = frozenset({"profile"})
    info = RuleInfo("GumbelTopPTokenSelect[vllm]", 4, "op-binding",
                    applies_to="vllm_iface.gumbel_topp_token_select(Tensor logits, Tensor seed, float temperature, float top_p, int pos, int S) -> Tensor",
                    preconditions=("logits is a bf16 value collection [.., V]", "seed is a 1-element 64-bit value collection (a Program input)",
                                   "temperature is a static float > 0 (0 is greedy: TokenSelect_v1, not this rule)", "0 < top_p <= 1 static",
                                   "pos >= 0 static", "S a static split count in {1, 2, 4, 8, 16, 32} (the registered split count)"),
                    evidence_domain="structural; numerics: GumbelTopPTokenSelect_v1{V} (exact-model-tested: difftest vs vLLM d9105ea80 "
                                    "_temperature_kernel / _topp_sb_* / _gumbel_sample_kernel on L40S, 72/72 rows bit-exact; a GPU fixture "
                                    "at every split count S = 1..32, tests/program/test_topp_split_geometry.py)")

    def targets(self):
        return targets_for("gumbel_topp_token_select")

    @staticmethod
    def _args(node):
        a = list(node.args) + [None] * 6
        kw = node.kwargs or {}
        return tuple(kw.get(k, a[i]) for i, k in enumerate(("logits", "seed", "temperature", "top_p", "pos", "S")))

    def applies(self, node, ctx):
        logits, seed, temperature, top_p, pos, S = self._args(node)
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
        if isinstance(S, bool) or not isinstance(S, int) or S not in SPLIT_COUNTS:
            return f"S {S!r} is not a launched split count {SPLIT_COUNTS} (the registered constant)"
        return True

    def emit(self, node, ctx):
        logits, seed, temperature, top_p, pos, S = self._args(node)
        x = ctx.coll(logits)
        rows, lead = rows_of(x)
        V = rows[0].type.n
        # the select specialised on the registered S, its keep word as word gates, the same six operands (the Const32[S] still read,
        # in operand order)
        fn = ctx.kind("token_select_stochastic_split", node, V=V, S=int(S))
        hint = ctx.hint_name(node)
        seed_c = ctx.coll(seed).reshape(Array(1, Value(64)))[0]
        tbits, pbits = f32_bits(temperature), f32_bits(top_p)
        # One Const quadruple PER ROW, emitted in OPERAND order (top_p, pos, temperature, S) right before the row's select: the same
        # instance sequence the fold emits for the observed sampler chain (`patterns.TokenSelectGumbelTopP` -> `fold._const`, one
        # Const per Compute operand), so `cb_a.compare` sees canonical-equal steps.  Per row because the fold's rows are per request
        # (their positions differ); the request wrapper samples one row per step anyway.
        outs = []
        for i, r in enumerate(rows):
            topp_c = ctx.call(ctx.kind("const", node, W=32, BITS=pbits), node=node, rule=self, name=f"{hint}/top_p")
            pos_c = ctx.call(ctx.kind("const", node, W=64, BITS=int(pos) & 0xFFFFFFFFFFFFFFFF), node=node, rule=self, name=f"{hint}/pos")
            temp_c = ctx.call(ctx.kind("const", node, W=32, BITS=tbits), node=node, rule=self, name=f"{hint}/temperature")
            s_c = ctx.call(ctx.kind("const", node, W=32, BITS=int(S)), node=node, rule=self, name=f"{hint}/splits")
            outs.append(ctx.call(fn, r, topp_c, seed_c, pos_c, temp_c, s_c, node=node, rule=self, name=f"{hint}[{i}]"))
        out = array_of(outs).reshape(_tensor_type(lead, 32)) if lead else outs[0]
        ctx.define(node, out)
        ctx.record(self, node, {"V": V, "S": f"Const32[{int(S)}] (the registered split count)",
                                "rows": len(rows), "temperature": float(temperature), "temperature_bits": f"0x{tbits:08x}",
                                "top_p": float(top_p), "top_p_bits": f"0x{pbits:08x}", "pos": int(pos), "seed": "Program input (runtime)",
                                "torch_dtype": str(ctx.dtype_of(node)), "carried_as": "I32"},
                   emitted=f"{fn.id} x{len(rows)} (+ per row: Const32[top_p], Const64[pos], Const32[temperature], Const32[S], in operand order)")


VLLM_SAMPLING_RULES = (GumbelTopPTokenSelectRule(),)

__all__ = ["GumbelTopPTokenSelectRule", "VLLM_SAMPLING_RULES", "f32_bits"]
