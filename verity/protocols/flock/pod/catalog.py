"""Definition ids bound through circuit-check's catalog, every registry loaded first: the resolver Flock's id-naming CLIs take
(``class_statement --resolver verity/protocols/flock/pod/catalog.py:definition``, 70-class-sweep.sh's and 86-zk-cpu-steps.sh's
``RESOLVER``)."""
from __future__ import annotations


def definition(spec: str):
    """The registered Definition ``spec`` (a Definition id) names."""
    from circuit_check import targets as TG

    TG.load_registries()
    return TG.definition(spec)
