"""What every decoder of `build.decoder` writes with: the refusal (`Unexpressible`), the vocabulary's kinds (`kind`), the served
class, the IR types of a shape, and the tensor-parallel rank's view of the other ranks' partials and of the schedule (`_Peers`)."""

from __future__ import annotations

import re

from verity.ml.circuits.refs import Coll, concat
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
