"""The verifier's own bindings of one of V*'s statements (`verify --rec-bindings F`, `Flock.RecVStar.checkMeta`): the
`verity/partition/v1` digest, the program's SHA-512 and ``program_digests``, the three fields of META that `vstar`'s
``BINDINGS`` file holds (`VStar.Bindings`), computed by the auditor from the verifier's own partition object and program,
never read from the prover's circuit file.

    python -m verity_flock.rec_bindings PARTITION PROGRAM > bindings.json

``program_digests`` is core's codec digest of the one-Definition program of the template the partition's query names, keyed
by its descriptor id (`circuit.program_digests`), the value the Lean verifier compares META's with but does not encode
itself. The template's Definition is the program's own, decoded (`codec.decode_program`) and bound again from its registered
family (`RecOpen`, `InnerRepCheck`) at the decoded statics; the bound template is checked to rebuild the verifier's program
(`population_program` of it at the program's ``N``) before its digest is taken.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from verity.ml.circuits.codec import canonical_json, decode_program, descriptor_id, program_digest
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.program import Program

from . import rec_open as _rec_open  # noqa: F401  (registers RecOpen)
from . import rec_residuals as _rec_residuals  # noqa: F401  (registers InnerRepCheck)
from . import rec_sparse as _rec_sparse  # noqa: F401  (registers RecSparse)

POPULATION = re.compile(r"TemplateInstances_v0\{N=([1-9][0-9]*)\}")


def bindings(partition: dict, program: dict) -> dict:
    """``{"partition", "program", "program_digests"}`` of the statement whose program is ``program`` (a descriptor) and whose
    partition object is ``partition``: `ValueError` unless the object names the program, its query names one template of it,
    and the program is `population_program` of that template."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P

    prog = program_digest(program)
    if partition.get("program") != prog:
        raise ValueError(f"the partition object's program {partition.get('program')}, the program's {prog}")
    tid = ((partition.get("query") or {}).get("params") or {}).get("template")
    entry = program.get("definitions", {}).get(tid)
    if entry is None:
        raise ValueError(f"the partition's template {tid!r} is not a Definition of the program")
    family = REGISTRY.defs.get(entry.get("id"))
    if family is None:
        raise ValueError(f"the template {entry.get('id')!r} is no registered family")
    m = POPULATION.fullmatch(program.get("program_fn", ""))
    if m is None:
        raise ValueError(f"the program's function {program.get('program_fn')!r} is not TemplateInstances_v0{{N}}")
    template = bind(family, **decode_program(program, REGISTRY).defs[tid].bindings)
    if descriptor_id(template) != tid:
        raise ValueError(f"the template bound again is {descriptor_id(template)!r}, not {tid!r}")
    if program_digest(P.population_program(template, int(m[1]))) != prog:
        raise ValueError(f"the program is not population_program of {entry['id']} at N={m[1]}")
    return {"partition": P.digest(partition), "program": prog, "program_digests": {tid: program_digest(Program(template))}}


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m verity_flock.rec_bindings PARTITION PROGRAM", file=sys.stderr)
        return 2
    partition, program = (json.loads(Path(a).read_text()) for a in args)
    try:
        b = bindings(partition, program)
    except ValueError as e:
        print(f"rec-bindings: {e}", file=sys.stderr)
        return 1
    sys.stdout.write(canonical_json(b).decode() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
