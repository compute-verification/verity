"""What every decoder of `build.decoder` writes with: the refusal (`Unexpressible`), the vocabulary's kinds (`kind`), the served
class, the circuit types of a shape, the tensor-parallel rank's view of the other ranks' partials and of the schedule (`_Peers`), the
batch a request's calls run in (`launch_batch`, `_Launch`) or a batch-invariant row's launch context (`launch_rows`, `launch_seqlen_q`) and the
token select over a row's whole-vocabulary logits (`select_token`)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from verity.ml.circuits.refs import Coll, Refs, concat
from verity.ml.circuits.types import Array, Type, Value


class Unexpressible(Exception):
    """φ asks for a structure the vocabulary cannot express.  The message names the fact precisely."""


def kind(role: str, **statics):
    """The vocabulary's kind for `role` at `statics` (the object the rules' `ctx.kind` returns), or `Unexpressible` naming why not."""
    from verity_vllm.program.frontend import targets as TG
    from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3 as VOC
    try:
        fn = VOC.kind(role, **statics)
    except TG.UnregisteredTarget as e:
        raise Unexpressible(f"numerics_unregistered: {e}") from None
    if fn is None:
        raise Unexpressible(f"vocabulary {VOC.name!r} has no kind for role {role!r} with {statics}")
    return fn


def served_class(vllm_class: str) -> str:
    """The class vLLM instantiates for a family (`FAMILY_FACTS[...]["vllm_class"]`): the one its registry maps the architecture to
    ("X (= Y)" is Y), else the named class."""
    alias = re.search(r"\(= (\w+)\)", vllm_class)
    return alias.group(1) if alias else re.match(r"\w+", vllm_class).group(0)


def _arr(shape: tuple[int, ...], w: int) -> Type:
    t: Type = Value(w)
    for n in reversed(shape):
        t = Array(int(n), t)
    return t


def _arr_shape(t: Type) -> tuple[int, ...]:
    shape = []
    while isinstance(t, Array):
        shape.append(t.n)
        t = t.elem
    return tuple(shape)


def _ptypes(spec) -> tuple:
    params, _ = spec.signature()
    return tuple(t for _, t in params)


@dataclass(frozen=True)
class _Stochastic:
    """The row's stochastic policy as the select reads it: the seed (and, for several requests, the `splits`) Inputs, the f32 bits of
    temperature and top_p, and S: "derived" (an Input entry per event) or the single request's constant split count."""
    seed: Coll
    splits: Coll | None
    S: Any
    temperature_bits: int
    top_p_bits: int


def select_token(call, k, logits: Refs, V: int, target, event: int, pos: int, policy: _Stochastic | None) -> Coll:
    """The token of sampling event `event` from a row's V logits at position `pos` (`call(fn, *refs)` emits, `k(role, **statics)` is
    the kind): greedy `TokenSelect`, or `GumbelTopPTokenSelect[vllm]` over the derived `splits` entry or at the single request's
    constant S, its per-row constants in operand order (top_p, pos, temperature[, S])."""
    if policy is None:
        return call(k("token_select", V=V), logits)
    fn = (k("token_select_stochastic", V=V, CONSTRUCTION=target.effective_sampler_construction()) if policy.S == "derived"
          else k("token_select_stochastic_split", V=V, S=int(policy.S)))
    top_p = call(k("const", W=32, BITS=policy.top_p_bits))
    pos_c = call(k("const", W=64, BITS=int(pos) & 0xFFFFFFFFFFFFFFFF))
    temp = call(k("const", W=32, BITS=policy.temperature_bits))
    splits = policy.splits.refs.slice(event, event + 1) if policy.S == "derived" else call(k("const", W=32, BITS=int(policy.S))).refs
    return call(fn, logits, top_p.refs, policy.seed.refs, pos_c.refs, temp.refs, splits)


class _Peers:
    """A tensor-parallel rank's view of the other ranks' partials (`vllm_meta.WithPeers`): world - 1 peer Inputs per collective, in
    call order, the other ranks in rank order; `n` counts the Inputs the collectives have read.  `schedule` is the rank's schedule
    Input (`CollectiveBus.schedule_key`, the MoE align's key per pair), read in call order; `k` counts the words read."""

    def __init__(self, inputs: list[Coll], world: int, rank: int, schedule: Coll | None = None) -> None:
        self.inputs, self.world, self.rank, self.n = list(inputs), world, rank, 0
        self.schedule, self.k = schedule, 0

    def take(self, own: list) -> list[list]:
        """One collective's rows per rank, in rank order: this rank's `own` rows (refs) and each of the next world - 1 peer Inputs'."""
        peers = self.inputs[self.n:self.n + self.world - 1]
        self.n += self.world - 1
        N = own[0].count
        others = [[p.refs.slice(i * N, (i + 1) * N) for i in range(len(own))] for p in peers]
        return others[:self.rank] + [list(own)] + others[self.rank:]

    def rows(self, own: list) -> list:
        """Each row's operand of a per-row collective: that row of every rank, rank order."""
        ranks = self.take(own)
        return [concat([r[i] for r in ranks]) for i in range(len(own))]

    def whole(self, own: list):
        """The operand of one collective over all rows: every rank's rows, rank-major (`collectives.AllReduceRule`'s parts)."""
        return concat([concat(r) for r in self.take(own)])

    def key(self, n: int):
        """The next n words of the schedule."""
        out = self.schedule.refs.slice(self.k, self.k + n)
        self.k += n
        return out


class _Launch:
    """The batch a request's calls run in (`pipeline.launch_context.request_launch_batch`, parsed as `targets.parse_launch_context`
    parses the Build's `--launch-context`): per request step, every sequence the engine step schedules as (rows, start).  `step` is
    the request step being written; without a batch a call is its own launch."""

    def __init__(self, batch: list | None = None) -> None:
        from verity_vllm.program.frontend import targets as TG
        self.ctx, self.step = TG.parse_launch_context({"batch": batch}) if batch else None, 0

    def tokens(self, own: int, sampled: bool = False) -> int:
        """The rows of the launch `own` rows run in (`targets.launch_tokens`); after the logits selection (`sampled`), its sequences."""
        from verity_vllm.program.frontend import targets as TG
        try:
            return TG.launch_tokens(own, self.step, self.ctx, sampled=sampled)
        except ValueError as e:
            raise Unexpressible(f"request step {self.step}: {e}") from None

    def keys(self, own: list[int]) -> list[int]:
        """Every query row's key count in the launch (`targets.launch_keys`), `own`'s without a batch."""
        from collections import Counter
        from verity_vllm.program.frontend import targets as TG
        keys = TG.launch_keys(self.step, self.ctx)
        if keys is None:
            return list(own)
        if Counter(own) - Counter(keys):
            raise Unexpressible(f"request step {self.step}: the launch batch's key counts {keys} do not hold this request's {own}")
        return keys


def launch_batch(phi, LP: int, T: int) -> list | None:
    """The batch each call of φ's (LP, T) request runs in where its kernel choices read the launch (`launch_context.batch_dispatch`:
    a target without batch invariance co-scheduling several requests), as the Build is given it (`launch_context.shape_contexts`);
    None for a request that is its own launch.  Requests of the shape co-scheduled differently have no one Program."""
    from verity_vllm.pipeline import launch_context as LC
    w = {"requests": phi.requests, "engine": {"target": {"batch_invariant": phi.target_profile().batch_invariant}}}
    if not LC.batch_dispatch(w):
        return None
    batch = LC.shape_contexts(w).get((int(LP), int(T)))
    if batch is None or batch == "conflict":
        raise Unexpressible(f"vocabulary cannot express: the launch batch of request shape LP {LP}, T {T}: "
                            f"{'its requests are co-scheduled differently' if batch else 'no request of the workload has it'}")
    return batch


def launch_rows(phi, LP: int, T: int, C: int = 0) -> dict | None:
    """The launch context a batch-invariant row's (LP, T[, C]) request runs under, as the Build is given it (`launch_context.spec_for`:
    per request step, the launch's max_seqlen_q, every co-scheduled sequence's rows counted), parsed as `targets.parse_launch_context`
    parses `--launch-context`; None without the workload's requests, or for a shape none of them has (the Build's own rows)."""
    if not phi.requests:
        return None
    from verity_vllm.pipeline import launch_context as LC
    from verity_vllm.program.frontend import targets as TG
    from verity_vllm.program.prefix_plan import shape
    v = LC.shape_contexts({"requests": phi.requests, "engine": {"target": {"batch_invariant": True}}}).get(shape(LP, T, C))
    return None if v is None else TG.parse_launch_context(v)


def launch_seqlen_q(ctx: dict, own_rows: int, step: int, tp, accepted: bool) -> int:
    """The max_seqlen_q a request's attention at request `step` reads under its launch context (`targets.launch_seqlen_q`, as the
    Build's attention binding reads it): FA3's tile has no one value for a shape whose requests run in different launches ('conflict');
    FA2's reads the request's own rows."""
    from verity_vllm.program.frontend import targets as TG
    try:
        return TG.launch_seqlen_q(own_rows, step, ctx)[0]
    except TG.LaunchContextConflict as e:
        if tp.effective_flash_attn_version() == 3 and not accepted:
            raise Unexpressible(f"vocabulary cannot express: request step {step}: FA3's tile (kBlockN) is not one value -- {e}") from None
        return own_rows
    except TG.LaunchContextInvalid as e:
        raise Unexpressible(f"vocabulary cannot express: request step {step}: {e}") from None
