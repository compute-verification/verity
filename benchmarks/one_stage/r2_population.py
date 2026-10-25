"""R2's other partition, which every arm's registration-partition-mismatch negative registers: the window's population with
its first template one instance short, or one over where it has a single instance, since a population has at least one
(`partition.population_program`). A one-unit window is the case one short would leave empty.

    import r2_population as R2
    other = R2.program([(template, n)], mixed=False)            # TemplateInstances, as A0 and A2 build theirs
    other = R2.program(list(zip(defs, ns)), mixed=True)         # TemplatePopulations, as A3, A6, A7 and A11 do
"""

from __future__ import annotations

from typing import Any, Sequence

from verity.core.protocols.sampled_proofs.one_stage import partition as P


def resized(population: Sequence[tuple[Any, int]]) -> list[tuple[Any, int]]:
    """`(template, n)` per template, in order, the first one instance short, or one over where it has a single instance."""
    return [(d, (n - 1 if n > 1 else n + 1) if i == 0 else n) for i, (d, n) in enumerate(population)]


def program(population: Sequence[tuple[Any, int]], *, mixed: bool, version: int = P.POPULATION_VERSIONS[0]):
    """The population program over `resized(population)` at `version`, built as the arm builds its own: `TemplatePopulations`
    when `mixed`, else `TemplateInstances` over the one template."""
    other = resized(population)
    if mixed:
        return P.mixed_population_program(other, version=version)
    if len(other) != 1:
        raise ValueError("a TemplateInstances population has one template")
    return P.population_program(*other[0], version=version)
