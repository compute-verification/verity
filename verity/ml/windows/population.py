"""A window's population: the verifier's program and query over the templates a window's units instantiate, and the template
partition that query makes of it (``verity.ml.circuits.partition_object``).  Core's one-stage partition names the same builders
and refuses more (a unit wider than the widest class, a template of no gate) when it registers a window.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.population import mixed_population_program, population_program as template_population


def population_program(templates: Sequence[tuple[Any, int]], several: bool = False) -> tuple[Any, dict[str, Any]]:
    """The verifier's population program and query over ``(template Definition, instances)`` in order: ``TemplateInstances`` and
    ``Q_template_instance`` for one template, ``TemplatePopulations`` and ``Q_template_instances`` for several (or, ``several``,
    for one)."""
    if len(templates) == 1 and not several:
        (t, n), = templates
        return template_population(t, n), PO.template_instance_query(t)
    return mixed_population_program(list(templates)), PO.template_instances_query([t for t, _ in templates])


def template_partition(program: Any, query: Mapping[str, Any]) -> PO.TemplatePartition:
    """``query`` applied to ``program`` (``partition_object.evaluate``): a window's query is a template query, so its units are
    template instances."""
    part = PO.evaluate(program, query)
    if not isinstance(part, PO.TemplatePartition):
        raise ValueError(f"a window's query is a template query ({PO.Q_TEMPLATE_INSTANCE[0]} or "
                         f"{PO.Q_TEMPLATE_INSTANCES[0]}), not {dict(query)!r}")
    return part


__all__ = ["population_program", "template_partition"]
