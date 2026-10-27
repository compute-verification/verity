"""The toy's NCI declaration: what onsite's registration check (`NciCheck.Protocol`, ``verity/nci/v1``) reads of a window
beside the run's registration record (`compose.registration_case`), built from the window's records alone, as the
verifier holds them (``period.json``'s ``nci``):

* ``program``: the verifier's Program for the window (`interface.of_records`: the circuit record its registration binds,
  each template resolved in Verity's catalog) as canonical descriptor text, so NCI's Program is the one integrity audits:
  its digest is the registration's ``program``, which `declaration` checks;
* ``parameters``: that Program's function's parameters, ``[[name, type], ...]`` as the descriptor writes types;
* ``owner``: one isolation unit, 0, holding every Call of the root, and ``"input"`` for each input node;
* ``theta`` and ``provenance``: none, since the toy registers no baseline at its deployment's start, so every input a
  unit reads is a novel read;
* ``alpha`` and ``gamma``: `ALPHA`, and the Program's gates as Γ.

NCI checks the record's ``program`` against its Program only per θ entry (``theta-unbound``), so with θ empty the
binding of NCI's Program to the audited one is this builder's, not the check's.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verity.core.protocols.sampled_proofs.one_stage import draw as D
from verity.ml.circuits import codec
from verity.ml.circuits.program import is_input
from verity.ml.windows import interface as IF

#: The toy's demonstration policy, not one an auditor stated: novel reads of at most 235 bits a gate. Each unit of the
#: unwired toy reads ``x`` and ``wrow`` (128 bytes each) and ``sx`` and ``sw`` (32 bits each), 2112 bits over its 9
#: gates (234.7 a gate); the wired toy's one isolation unit needs 74 to 83.
ALPHA = [235, 1]


def of_program(program: Any, alpha: list[int] = ALPHA, gamma: int | None = None) -> dict[str, Any]:
    """The declaration (module docstring) of ``program``, Γ its gates unless ``gamma``."""
    desc = codec.encode_program(program)
    return {"alpha": list(alpha), "gamma": program.gates if gamma is None else gamma,
            "owner": ["input" if is_input(n.fn) else 0 for n in program.root.body.nodes],
            "parameters": [[n, t.to_json()] for n, t in program.fn.signature()[0]],
            "program": codec.canonical_json(desc).decode(), "provenance": [], "theta": []}


def declaration(window_dir: str | Path, law: str | Mapping[str, Any], alpha: list[int] = ALPHA,
                gamma: int | None = None) -> dict[str, Any]:
    """The declaration of the window in ``window_dir`` under ``law`` (``subset:K`` and the like, or parsed): the
    verifier's Program from its records, refused unless its digest is the registration's ``program``."""
    window, doc = IF.of_records(window_dir, D.parse_law(law) if isinstance(law, str) else law, IF.resolve)
    out = of_program(window.program, alpha, gamma)
    if (got := codec.program_digest(codec.encode_program(window.program))) != doc["program"]:
        raise ValueError(f"{window_dir}: the Program rebuilt from its records is {got}, not its registration's "
                         f"{doc['program']}")
    return out


__all__ = ["ALPHA", "declaration", "of_program"]
