"""A word derivation's report with its Program on bits (`derived`), which the Build takes when `TargetProfile.ir == "boolean"`:
each Call of the Boolean version of its Definition (`program.boolean.boolean_function`), σ-checked node by node."""
from __future__ import annotations

from dataclasses import replace

from verity.ml.circuits.codec import descriptor_id, encode_program, program_digest
from verity.ml.circuits.liveness import dead_gates
from verity.ml.circuits.program import Program
from verity.ml.definitions.boolean import NoBooleanVersion, boolean_function, sigma_check
from verity_vllm.program.frontend.derive import DeriveReport, Unsupported
from verity_vllm.program.frontend.program_identity import check_identity


def derived(rep, *, check_registry: bool = True):
    """A word derivation's report (`frontend.derive.DeriveReport`) with its Program on bits (`boolean_function`), the word root and
    digest in its provenance (``ir``); a refusal (``numerics_unregistered``) names every Call with no Boolean version.  A lifted
    Program (σ-checked node by node) names the word Program it computes as its ``word``, as a Boolean Definition names its word:
    the Build writes its instance sequence through that link (`pipeline.build_instances`)."""
    if not rep.ok:
        return rep
    prov = dict(rep.provenance, ir={"ir": "boolean", "word_root": rep.root, "word_digest": rep.digest})
    try:
        fn = boolean_function(rep.program.fn)
    except NoBooleanVersion as e:
        u = Unsupported("numerics_unregistered", hint=f"ir=boolean: {e}", detail={"missing": e.missing})
        return DeriveReport(None, u, prov | {"unsupported": u.to_json()})
    if fn.name.startswith("Derived["):
        sc = sigma_check(rep.program.fn, fn)
        if not sc["ok"]:
            u = Unsupported("identity", hint=f"ir=boolean: not σ of the word function: {'; '.join(sc['mismatches'])}", detail=sc)
            return DeriveReport(None, u, prov | {"unsupported": u.to_json()})
        prov["ir"]["sigma_check"] = {k: sc[k] for k in ("nodes", "definitions", "ok")}
    program = Program(fn, annotations=dict(rep.program.annotations))
    if "sigma_check" in prov["ir"]:
        program.word = rep.program  # type: ignore[attr-defined]
    check_identity(program, check_registry=check_registry)
    desc = encode_program(program)
    digest, root, dead = program_digest(desc), descriptor_id(program.fn), dead_gates(program, limit=32)
    # each Call and node index is the word Program's, and word leaf j is element j: the correspondence carries over unchanged
    prov["result"] = dict(prov["result"], root=root, digest=digest, gates=program.gates, dead_gates=dead.total)
    return replace(rep, program=program, provenance=prov, digest=digest, root=root, gates=program.gates, dead=dead,
                   descriptor=desc if rep.descriptor is not None else None)


__all__ = ["derived"]
