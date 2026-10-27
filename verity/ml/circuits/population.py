"""The population programs a window registers under a template query: `population_program` (instances of one template,
`TemplateInstances_v1{template, N}`) and `mixed_population_program` (of several, `TemplatePopulations_v1{templates, counts}`),
whose binding records name the templates by descriptor id. v0 of each, which names only the counts, is kept for the records
made with it (`POPULATION_VERSIONS`).

Sampled proofs' one-stage protocol applies its template queries to them (`one_stage.partition`, which names them too), and an
integration builds its window's program with them: both sides build the same program from the same templates and counts.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Sequence

from .codec import descriptor_id
from .defs import CompositeDefinition, SpecializedDefinition, bind
from .ports import In, Leaves, Out, Ports, port_types
from .program import Program
from .refs import coll_concat
from .types import Array, Tuple

#: The population programs' versions, the builders' default first. v1 names its templates (each one's descriptor id) in its
#: binding record beside the counts, so the Definition's identity determines its body. v0 names the counts alone, so two
#: populations of equal counts over different templates share its id; it stays buildable for the records made with it.
POPULATION_VERSIONS = (1, 0)


def _population_version(version: int) -> int:
    if version not in POPULATION_VERSIONS:
        raise ValueError(f"no population program v{version}: the versions are {POPULATION_VERSIONS}")
    return version


def mixed_population_program(members: Sequence[tuple[SpecializedDefinition, int]], *, version: int = 1) -> Program:
    """`TemplatePopulations_v1{templates, counts}`: one batch per member `(template, n)`, in order; member j's parameters are
    `j.<name>`, arrays over its instances, and the result is the tuple of the batches' results. `templates` are the members'
    descriptor ids, in order. A member whose population declares ports (`_instance_ports`) keeps them, under `j.<name>` and at
    its batch's offset in the result. `version=0` builds `TemplatePopulations_v0{counts}` (`POPULATION_VERSIONS`)."""
    if not members or any(n < 1 for _, n in members):
        raise ValueError("every member has at least one instance")
    _population_version(version)
    sigs = [t.signature() for t, _ in members]

    def signature(S):
        params = tuple((f"{j}.{name}", Array(n, t)) for j, ((ps, _), n) in enumerate(zip(sigs, S.counts)) for name, t in ps)
        return params, Tuple(*(Array(n, ret) for (_, ret), n in zip(sigs, S.counts)))

    def body(B, S, *args):
        outs, at = [], 0
        for (t, _), (ps, _) in zip(members, sigs):
            outs.append(B.batch(t, *args[at:at + len(ps)], axes=(0,) * len(ps)))
            at += len(ps)
        return coll_concat(signature(S)[1], outs)

    each = [_instance_ports(t) for t, _ in members]

    def ports(S: Any) -> Ports:
        params, sinks, at = {}, [], 0
        for j, ((_ps, ret), f, n) in enumerate(zip(sigs, each, S.counts)):
            if f is not None:
                p = f(SimpleNamespace(N=n))
                params.update((f"{j}.{name}", leaves) for name, leaves in p.params.items())
                sinks += [(at + lo, at + hi, o) for lo, hi, o in (p.ret.runs if p.ret is not None else ())]
            at += n * len(ret.leaf_widths())
        return Ports(params=params, ret=Leaves(tuple(sinks)) if sinks else None)

    named = {} if version == 0 else {"templates": tuple(descriptor_id(t) for t, _ in members)}
    defn = CompositeDefinition("TemplatePopulations", version, (*named, "counts"), signature, body,
                               doc="independent instances of " + ", ".join(f"{n} x {t.id}" for t, n in members), register=False,
                               ports=ports if any(f is not None for f in each) else None)
    return Program(bind(defn, **named, counts=tuple(n for _, n in members)))


def _instance_ports(template: SpecializedDefinition) -> Callable[[Any], Ports] | None:
    """The ports of `template`'s population, for a unit (a template that takes a coin or a position and returns a sink, a
    salted tile such as PoUW's `PouwH1Tile`) or a template whose declared inputs are positions alone (a key count, as in
    `ops.gemm.PvRowPad`): each instance's coin and position leaves keep their kind and each instance's sinks stay sinks, so
    the population is a unit too. None for any other template, whose population declares no ports (its id and digest
    unchanged); a seed's population (a coin and no sink) is so refused, since only PoUW's `seeds.py` mints seeds
    (`tests/test_seed_ports.py`)."""
    pt = port_types(template)
    if pt is None:
        return None
    params, ret = template.signature()
    kinds = [(name, len(t.leaf_widths()), [(lo, hi, p.kind) for lo, hi, p, _w in rs if p.kind in ("coin", "position")])
             for (name, t), rs in zip(params, pt.params)]
    kinds = [(name, w, rs) for name, w, rs in kinds if rs]
    sinks, w_ret = [(lo, hi) for lo, hi, o, _w in pt.ret if o.sink], len(ret.leaf_widths())
    if not kinds or not sinks and any(k == "coin" for _name, _w, rs in kinds for _lo, _hi, k in rs):
        return None

    def ports(S: Any) -> Ports:
        return Ports(params={name: Leaves(tuple((i * w + lo, i * w + hi, In(kind=k)) for i in range(S.N) for lo, hi, k in rs))
                             for name, w, rs in kinds},
                     ret=Leaves(tuple((i * w_ret + lo, i * w_ret + hi, Out(sink=True)) for i in range(S.N) for lo, hi in sinks))
                     if sinks else None)
    return ports


def population_program(template: SpecializedDefinition, n: int, *, version: int = 1) -> Program:
    """`TemplateInstances_v1{template, N=n}`: `n` independent instances of `template` as one batch, every parameter an array
    over instances; `template` is its descriptor id. Instance i of the batch is instance i of the input set it is built for.
    A unit's population declares its coins, positions and sinks per instance (`_instance_ports`). `version=0` builds
    `TemplateInstances_v0{N=n}` (`POPULATION_VERSIONS`)."""
    if n < 1:
        raise ValueError("a population has at least one instance")
    _population_version(version)
    params, ret = template.signature()

    def signature(S):
        return tuple((name, Array(S.N, t)) for name, t in params), Array(S.N, ret)

    def body(B, S, *args):
        return B.batch(template, *args, axes=(0,) * len(args))

    named = {} if version == 0 else {"template": descriptor_id(template)}
    defn = CompositeDefinition("TemplateInstances", version, (*named, "N"), signature, body,
                               doc=f"{n} independent instances of {template.id}", register=False,
                               ports=_instance_ports(template))
    return Program(bind(defn, **named, N=n))


__all__ = ["POPULATION_VERSIONS", "mixed_population_program", "population_program"]
