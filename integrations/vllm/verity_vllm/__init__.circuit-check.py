"""circuit-check's sidecar of `verity_vllm` itself, in `circuit_check.targets`' forms (`circuit_check.sidecars`): what
circuit-check needs of the integration without importing it.  It registers the package: circuit-check names no integration
and finds the sidecars under a package that has this file.  And it gives the reader of a served workload's request
Program."""
from __future__ import annotations

from typing import Any, Callable, Iterator


def request_program(path: str) -> tuple[Any, Any, Callable[[], Iterator[Any]]]:
    """A Build's request Program (its ``instances.json.gz``, or the Build's directory) as circuit-check's row check reads it:
    the Program's view, its population as `word.call_groups` reads one (the required values, the literal Calls and each
    Call's module), and its instance rows, from the first at each call."""
    from pathlib import Path
    from types import SimpleNamespace

    from verity_vllm.correspondence.reader_for_query import Correspondence
    from verity_vllm.query.module_body import literal_calls
    from verity_vllm.query.program_view import from_instances, iter_instance_rows
    from verity_vllm.query.required import population

    p = Path(path)
    inst = str(p / "instances.json.gz" if p.is_dir() else p)
    P = from_instances(inst)
    corr = Correspondence(P)
    Q = SimpleNamespace(required=population(P, corr).required, literals=literal_calls(P), program_path=corr.module_of)
    return P, Q, lambda: iter_instance_rows(inst)
