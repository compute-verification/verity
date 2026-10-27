"""The verifier's own bindings of one of V*'s statements (`verify --rec-bindings F`, `Flock.RecVStar.checkMeta`): the
`verity/partition/v1` digest, the program's SHA-512 and ``program_digests``, the three fields of META that `vstar`'s
``BINDINGS`` file holds (`VStar.Bindings`), computed by the auditor from the verifier's own partition object and program,
never read from the prover's circuit file.

    python -m verity_flock.rec_bindings PARTITION PROGRAM [--vstar UNIT [--holo]] > bindings.json

``program_digests`` is core's codec digest of the one-Definition program of the template the partition's query names, keyed
by its descriptor id (`circuit.program_digests`), the value the Lean verifier compares META's with but does not encode
itself. The template's Definition is the program's own, decoded (`codec.decode_program`) and bound again from its registered
family (`RecOpen`, `InnerRepCheck`) at the decoded statics; the bound template is checked to rebuild the verifier's program
(`population_program` of it at the program's ``N`` and version, v1 or an old statement's v0) before its digest is taken.
With ``--vstar UNIT`` (``vstar`` in `bindings`) the template must be V*'s Definition of the statement whose unit is
``UNIT`` (`vstar_definition`, ``--holo`` over a private inner session), as `Flock.RecVStar.checkTemplate` holds the bindings'
key to it, and the digest is that Definition's: a part relabelled under another template is refused.
"""
from __future__ import annotations

import json
import re
import sys
from functools import cache
from pathlib import Path

from verity.ml.circuits.codec import canonical_json, decode_program, descriptor_id, program_digest
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.program import Program

from . import rec_open as _rec_open  # noqa: F401  (registers RecOpen)
from . import rec_residuals as _rec_residuals  # noqa: F401  (registers InnerRepCheck)
from . import rec_sparse as _rec_sparse  # noqa: F401  (registers RecSparse)

#: `population_program`'s program function at each version: v1 names its template, v0 (an old statement's) its count alone
POPULATION = {1: re.compile(r'TemplateInstances_v1\{template=".*",N=([1-9][0-9]*)\}'),
              0: re.compile(r"TemplateInstances_v0\{N=([1-9][0-9]*)\}")}


@cache
def _part_S(m: int, k_log: int, claims: int, holo: bool, opening: bool, j: int) -> dict:
    from . import rec_algebra as A

    sh = A.Shape(m, 6, claims, salt_len=0, opening_only=True) if opening else A.Shape(m, k_log, claims, holo=holo)
    ps = _rec_residuals.parts(_rec_residuals.spec(sh))
    if j >= len(ps):
        raise ValueError(f"part {j} of {len(ps)} at {sh}")
    return ps[j].S


def vstar_definition(unit: str, holo: bool = False):
    """V*'s Definition of the statement whose unit is ``unit`` (`Flock.RecVStar.templateId`'s): a level's of the table or of
    R ``RecOpen{LANES, H}``; a part of the algebra (at the inner session's holo flag ``holo``) or of R's opening
    ``InnerRepCheck{S}`` at its part's S (`rec_residuals.parts`, as `rec_vstage.stage` and `stage_opening` cut it);
    RecSparse at its shape; a cap's RecCap. `ValueError` for a unit that is none of V*'s."""
    from . import rec_cap

    n = r"([0-9]+)"
    if m := re.fullmatch(rf"rec-open(?:-stacked)?/l{n}-h{n}", unit):
        return _rec_open.definition(int(m[1]), int(m[2]))
    if m := re.fullmatch(rf"rec-residuals/m{n}-k{n}-c{n}-p{n}", unit):
        return _rec_residuals.definition(_part_S(int(m[1]), int(m[2]), int(m[3]), holo, False, int(m[4])))
    if m := re.fullmatch(rf"rec-opening/m{n}-c{n}-p{n}", unit):
        return _rec_residuals.definition(_part_S(int(m[1]), 6, int(m[2]), False, True, int(m[3])))
    if m := re.fullmatch(rf"rec-sparse/n{n}-e{n}-s{n}-t{n}", unit):
        return _rec_sparse.definition(*map(int, m.groups()))
    if re.fullmatch(rf"rec-cap/n{n}(?:-l{n})?", unit):
        return rec_cap.definition()
    raise ValueError(f"{unit}: not a unit of V*'s statements")


def bindings(partition: dict, program: dict, vstar=None) -> dict:
    """``{"partition", "program", "program_digests"}`` of the statement whose program is ``program`` (a descriptor) and whose
    partition object is ``partition``: `ValueError` unless the object names the program, its query names one template of it,
    and the program is `population_program` of that template; with ``vstar`` (V*'s Definition of the statement,
    `vstar_definition`), unless that template is ``vstar``, whose digest is then taken."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P

    prog = program_digest(program)
    if partition.get("program") != prog:
        raise ValueError(f"the partition object's program {partition.get('program')}, the program's {prog}")
    tid = ((partition.get("query") or {}).get("params") or {}).get("template")
    if vstar is not None and tid != (want := descriptor_id(vstar)):
        raise ValueError(f"the partition's template {_brief(tid)} is not V*'s template of the statement {_brief(want)}")
    entry = program.get("definitions", {}).get(tid)
    if entry is None:
        raise ValueError(f"the partition's template {tid!r} is not a Definition of the program")
    family = REGISTRY.defs.get(entry.get("id"))
    if family is None:
        raise ValueError(f"the template {entry.get('id')!r} is no registered family")
    fn = program.get("program_fn", "")
    version, m = next(((v, m) for v, pat in POPULATION.items() if (m := pat.fullmatch(fn))), (None, None))
    if m is None:
        raise ValueError(f"the program's function {fn!r} is not TemplateInstances_v1{{template,N}} or TemplateInstances_v0{{N}}")
    template = vstar if vstar is not None else bind(family, **decode_program(program, REGISTRY).defs[tid].bindings)
    if descriptor_id(template) != tid:
        raise ValueError(f"the template bound again is {descriptor_id(template)!r}, not {tid!r}")
    if program_digest(P.population_program(template, int(m[1]), version=version)) != prog:
        raise ValueError(f"the program is not population_program v{version} of {entry['id']} at N={m[1]}")
    return {"partition": P.digest(partition), "program": prog, "program_digests": {tid: program_digest(Program(template))}}


def _brief(t) -> str:
    """A descriptor id as an error shows it (`Flock.RecVStar.briefId`): a part's S runs to tens of kB."""
    return repr(t) if not isinstance(t, str) or len(t) <= 160 else f"{t[:120]!r}… ({len(t)} characters)"


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    holo = "--holo" in args
    args = [a for a in args if a != "--holo"]
    unit = None
    if len(args) == 4 and args[2] == "--vstar":
        unit, args = args[3], args[:2]
    if len(args) != 2 or (holo and unit is None):
        print("usage: python -m verity_flock.rec_bindings PARTITION PROGRAM [--vstar UNIT [--holo]]", file=sys.stderr)
        return 2
    partition, program = (json.loads(Path(a).read_text()) for a in args)
    try:
        b = bindings(partition, program, vstar_definition(unit, holo) if unit is not None else None)
    except ValueError as e:
        print(f"rec-bindings: {e}", file=sys.stderr)
        return 1
    sys.stdout.write(canonical_json(b).decode() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
