"""The population programs a window registers under a template query: `population_program` (instances of one template,
`TemplateInstances_v1{template, N}`) and `mixed_population_program` (of several, `TemplatePopulations_v1{templates, counts}`),
whose binding records name the templates by descriptor id. v0 of each, which names only the counts, is kept for the records
made with it (`POPULATION_VERSIONS`).

Sampled proofs' one-stage protocol applies its template queries to them (`one_stage.partition`, which names them too), and a
serving stack builds its window's program with them: both sides build the same program from the same templates and counts.
"""
from __future__ import annotations

from typing import Sequence

from .codec import descriptor_id
from .defs import CompositeDefinition, SpecializedDefinition, bind
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
    descriptor ids, in order. `version=0` builds `TemplatePopulations_v0{counts}` (`POPULATION_VERSIONS`)."""
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

    named = {} if version == 0 else {"templates": tuple(descriptor_id(t) for t, _ in members)}
    defn = CompositeDefinition("TemplatePopulations", version, (*named, "counts"), signature, body,
                               doc="independent instances of " + ", ".join(f"{n} x {t.id}" for t, n in members), register=False)
    return Program(bind(defn, **named, counts=tuple(n for _, n in members)))


def population_program(template: SpecializedDefinition, n: int, *, version: int = 1) -> Program:
    """`TemplateInstances_v1{template, N=n}`: `n` independent instances of `template` as one batch, every parameter an array
    over instances; `template` is its descriptor id. Instance i of the batch is instance i of the input set it is built for.
    `version=0` builds `TemplateInstances_v0{N=n}` (`POPULATION_VERSIONS`)."""
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
                               doc=f"{n} independent instances of {template.id}", register=False)
    return Program(bind(defn, **named, N=n))


__all__ = ["POPULATION_VERSIONS", "mixed_population_program", "population_program"]
