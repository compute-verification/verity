"""A class-statement workload's population: core's ``population_program`` of its VU Definition at ``n_max``, every instance
one unit of ``Q_template_instance`` v0 (``partition_units``' ``q-template-instance``). The class sweep loads this file by
path (``70-class-sweep.sh PROGRAM=<this file>:program``, with ``PT_VU`` and ``PT_N`` set), so it imports nothing of its
package."""
from __future__ import annotations

import os
from typing import Any


def population(vu: str, n: int) -> Any:
    from circuit_check import targets as TG
    from verity.core.protocols.sampled_proofs.one_stage import partition as P

    TG.load_registries()
    return P.population_program(TG.definition(vu), n)


def program() -> Any:
    return population(os.environ["PT_VU"], int(os.environ["PT_N"]))
