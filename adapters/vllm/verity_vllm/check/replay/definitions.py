"""The Definitions the sampled replay evaluates are the ones the Build of record bound, and its row kernels compute them.

A row kernel answers for its family's Definition in the replay's own tree, and the Commit bound the Program the Build of record
compiled from the Build's tree.  A Definition whose body moved between the two trees would have the replay grade the committed
words against a circuit the Commit never bound, and the Program digest the loaders check is the Build's recorded string, which no
tree change moves.  `refusal` reads the Build's ``descriptor.json.gz``, holds it to the digest of record, and holds each
specialization the replay evaluates, with every Definition it reaches, to the descriptor's entry for it
(`verity.circuits.codec.definition_digest`: the entry with the v1 interning and aliases resolved, the same for one Definition
in any Program).

That holds the Definitions, not the code that grades by them.  A primitive is held to its entry, its signature and ports only:
its evaluator is code, named by its id.  And the replay grades a row by its family's row kernel, not by the Definition.  So
before any row is graded, `kernel_refusal` runs core `self_check` of each graded family's row kernel against the Definition's
reference, which evaluates through the primitives' evaluators (`kernel_registry.check`, once a family per process).  That is a
sample, the limit of what this module holds: 6 seeded instances of each of the kernel's check specialisations, small statics the
reference evaluates, not the served ones.  A kernel that is its Definition there but not on a committed row's operands, or not at
the served statics, grades that row by the kernel; what holds it there is the kernel's own tests, not this check.  A delegate's
family (`delegate.DELEGATES`) has no row kernel and is not self-checked here: its protocol answers for it.

A Boolean Build's row names two Definitions (`build_instances._word_rows`): ``spec``, the word Definition whose family's row
kernel grades it, and ``bits``, the Boolean Definition the Program Calls, the one of the two the Build's descriptor holds.
`refusal` holds ``bits`` to the descriptor's entry, and to the row's ``spec`` by the word Definition this tree's ``bits``
names, the link the Build wrote the row through.  `kernel_refusal` then holds the row kernel to the gates, by the self-check's
own sample: at each of the kernel's check specialisations, the Boolean Definition of the row's ``bits`` family that names it as
its word (`verity.definitions.boolean.boolean_version`, refused where this tree has none) gives the kernel's rows on the
self-check's inputs, bit-sliced (`_gates_differ`).  At those small statics the Boolean Definition's function-valued statics are
this tree's newest Boolean versions, refused where the Build binds that function at another version and otherwise left to this
tree, and its body may reach Definitions the served one does not.  Nothing holds the kernel to the gates at a served binding,
and the replay's record says so (`premises`).
"""
from __future__ import annotations

import gzip
import json
import os
import re
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

__all__ = ["of_record", "premises"]

SHOWN = 5
#: the instances of each check specialisation a row kernel is sampled on, against its Definition and against a Boolean row's gates
SAMPLES = 6
#: what a replay that evaluates a Boolean Definition's row holds of its kernel and what it does not, on its record (`premises`)
BOOLEAN_PREMISE = (f"the replay holds a Boolean row's kernel to the gates of the Boolean Definition its Build bound by sample only: "
                   f"{SAMPLES} inputs at each of the kernel check's small statics, where a function-valued static the Build binds "
                   f"nowhere is this tree's newest Boolean version and the body may reach Definitions the served one does not; "
                   f"nothing holds the kernel to the gates at a served binding")
#: family -> why its row kernel is not its Definition in this tree ([] when it is), checked once a process
_KERNEL_CHECKS: dict[str, list[str]] = {}
#: (word family, Boolean family) -> why the word family's row kernel is not that Boolean family's gates ([] when it is) and the
#: families of the function-valued statics those gates were bound with (`_gates_differ`), checked once a process
_GATE_CHECKS: dict[tuple[str, str], tuple[list[str], set[str]]] = {}
#: the family of each function-valued static in a descriptor id, at any depth (`codec.descriptor_id` nests a function's id as a
#: JSON string, its quotes escaped)
_FN = re.compile(r'"fn\\*":\\*"([^"\\{]+)')
#: what building a Definition's body raises at statics its lowering does not take
_UNBUILT = (AssertionError, AttributeError, LookupError, NotImplementedError, TypeError, ValueError)


def evaluated_specs(rows: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    """The specializations of the rows the replay evaluates, each ``(spec, bound)``: ``spec`` is the row's word Definition, of a
    family with a row kernel (`rows.ROWS`) or a protocol's delegate, and ``bound`` the Definition its Build's descriptor holds
    the row's Call under, a Boolean Build's row's ``bits`` and otherwise ``spec``."""
    from verity.definitions.numerics.rows import ROWS

    from verity_vllm.check.replay.delegate import DELEGATES
    from verity_vllm.query.manifest.format import spec_family

    return sorted({(str(r["spec"]), str(r.get("bits") or r["spec"])) for r in rows
                   if spec_family(str(r["spec"])) in ROWS or spec_family(str(r["spec"])) in DELEGATES})


def premises(rows: Iterable[dict[str, Any]]) -> list[str]:
    """What a replay of ``rows`` holds only in part, for its record: `BOOLEAN_PREMISE` when it evaluates a Boolean Definition's
    row."""
    return [BOOLEAN_PREMISE] if any(bound != spec for spec, bound in evaluated_specs(rows)) else []


def of_record(dirs: dict[str, str], digs: dict[str, str], rows: dict[str, list[dict[str, Any]]], *, what: str) -> None:
    """Raise, by name, unless each request's Build dir (``dirs``) holds its Program of record (``digs``), the Definitions this
    tree evaluates its rows (``rows``) by are the ones that Build bound, and the row kernels it evaluates them with pass their
    self-check (`kernel_refusal`).  One check per Build dir."""
    by_dir: dict[str, tuple[str, set[tuple[str, str]]]] = {}
    for k, d in dirs.items():
        by_dir.setdefault(d, (digs[k], set()))[1].update(evaluated_specs(rows[k]))
    for d, (digest, specs) in sorted(by_dir.items()):
        why = refusal(d, digest, sorted(specs))
        if why:
            raise RuntimeError(f"{what}: REFUSED (not the Definitions of record): {why}")
    why = kernel_refusal(set().union(*(s for _, s in by_dir.values())))
    if why:
        raise RuntimeError(f"{what}: REFUSED (a row kernel is not its Definition): {why}")


def kernel_refusal(specs: Iterable[tuple[str, str]]) -> str | None:
    """Why a row kernel this tree grades ``specs`` (`evaluated_specs`) by is not its family's Definition, or None: each ``spec``'s
    family's row kernel against the reference by core `self_check` (`kernel_registry.check`; module docstring), once a family
    per process.  A kernel that passes and grades a Boolean Definition's row (``bound`` is not ``spec``) is then held to the
    gates of that row's ``bits`` family on the same sample (`_gates_differ`, once a pair of families per process), and the
    function-valued statics those gates were bound with to the Build's: one whose function a ``bound`` of ``specs`` binds at
    another version is refused."""
    from verity.definitions.numerics import kernel_registry as KR
    from verity.definitions.numerics.rows import KERNEL, ROWS

    from verity_vllm.commit import challenge as CH
    from verity_vllm.query.manifest.format import spec_family

    specs = sorted(set(specs))
    bits = {(spec_family(spec), spec_family(bound)) for spec, bound in specs if bound != spec}
    built = {f for spec, bound in specs if bound != spec for f in _FN.findall(bound)}
    bad: list[str] = []
    for family in sorted({spec_family(spec) for spec, _bound in specs} & set(ROWS)):
        if family not in _KERNEL_CHECKS:
            _KERNEL_CHECKS[family] = KR.check(family, KERNEL, CH.generator, SAMPLES)
        bad += [f"{family}: {why}" for why in _KERNEL_CHECKS[family]]
        if _KERNEL_CHECKS[family]:
            continue
        for pair in sorted(p for p in bits if p[0] == family):
            if pair not in _GATE_CHECKS:
                _GATE_CHECKS[pair] = _gates_differ(*pair)
            whys, bound_with = _GATE_CHECKS[pair]
            bad += [f"{family}: {why}" for why in whys]
            for f in sorted(bound_with):
                theirs = sorted(g for g in built if _name(g) == _name(f))
                if theirs and f not in theirs:
                    bad.append(f"{family}: its gates are checked with {f}, and the Build binds {', '.join(theirs)}")
    if not bad:
        return None
    more = f" (and {len(bad) - SHOWN} more)" if len(bad) > SHOWN else ""
    return "; ".join(bad[:SHOWN]) + more


def _name(family: str) -> str:
    """A family without its version: `Gemm` of `Gemm_v3`."""
    return family.rsplit("_v", 1)[0]


def _gates_differ(family: str, bits: str) -> tuple[list[str], set[str]]:
    """Why the row kernel of word ``family`` does not compute the gates of the Boolean family ``bits``, named, and the families
    of the function-valued statics the gates were bound with.  At each check specialisation of the kernel's self-check, on that
    self-check's inputs (`kernel_registry.check` and core `self_check` derive both so): the rows the kernel computes against the
    Boolean Definition of ``bits`` that names the specialisation as its word, evaluated bit-sliced through its word view
    (`evaluate_words`).  A specialisation no Definition of ``bits`` names, whose gates this tree does not build, or on which the
    kernel computes no row, is named too: nothing was compared there."""
    import zlib

    import numpy as np
    from verity.circuits.codec import descriptor_id
    from verity.circuits.evaluation import evaluate_words, sample_uniform
    from verity.circuits.evaluation.batch import type_dtype, type_shape
    from verity.circuits.evaluation.reference import resolve, signature_of, statics_of
    from verity.definitions.boolean import NoBooleanVersion, boolean_version
    from verity.definitions.numerics import kernel_registry as KR
    from verity.definitions.numerics.rows import KERNEL

    from verity_vllm.commit import challenge as CH

    (reg,) = (r for r in KR.registered() if (r.key, r.kernel) == (family, KERNEL))
    out: list[str] = []
    bound_with: set[str] = set()
    for t in map(resolve, reg.check_targets(CH.generator(zlib.crc32(f"{family}/{KERNEL}".encode())))):
        try:
            b = boolean_version(t, family=bits)
        except NoBooleanVersion as e:
            out.append(str(e))
            continue
        fid = descriptor_id(b)
        bound_with.update(_FN.findall(fid))
        params, _ret = signature_of(t)
        statics = statics_of(t)
        inputs = (reg.core.sample or sample_uniform)(CH.generator(0), SAMPLES, params, **statics)
        args = [np.asarray(a).reshape(SAMPLES, *type_shape(pt)).astype(type_dtype(pt), copy=False) for a, pt in zip(inputs, params)]
        got = reg.core.fn(*args, **statics)
        vals, declined = (None, np.ones(SAMPLES, dtype=bool)) if got is NotImplemented else got
        kept = np.arange(SAMPLES) if declined is None else np.flatnonzero(~np.asarray(declined, dtype=bool))
        if not kept.size:
            out.append(f"{t.id}: the row kernel computes none of its self-check's {SAMPLES} inputs, so no row is held to {fid}'s gates")
            continue
        rows = np.asarray(vals).reshape(SAMPLES, -1)[kept].astype(np.uint64)
        try:
            gates = np.asarray(evaluate_words(b, *(a[kept] for a in args))).reshape(kept.size, -1).astype(np.uint64)
        except _UNBUILT as e:
            out.append(f"{t.id}: this tree does not build {fid}'s gates, so no row is held to them ({type(e).__name__}: {e})")
            continue
        differ = kept[(rows != gates).any(axis=1)]
        if differ.size:
            out.append(f"{t.id}: the row kernel's rows are not {fid}'s gates on {differ.size} of the {kept.size} inputs it computes "
                       f"(the first: input {int(differ[0])} of its self-check's)")
    return out, bound_with


def refusal(program_dir: str, digest: str, specs: Iterable[tuple[str, str]]) -> str | None:
    """Why the Definitions this tree evaluates ``specs`` (`evaluated_specs`) by are not the ones the Build at ``program_dir``
    bound into Program ``digest``, or None.  With no ``specs`` nothing is evaluated, so nothing is read.  Each ``bound`` is held to
    the descriptor's entry, and a Boolean one to its row's ``spec`` as well (`_word_differs`).  Of a Boolean Build a None says no
    more than that: what holds its rows' kernels to its gates is `kernel_refusal`."""
    from verity.circuits.codec import program_digest
    from verity.definitions.registry.catalog import load

    specs = list(specs)
    if not specs:
        return None
    p = os.path.join(program_dir, "descriptor.json.gz")
    if not os.path.exists(p):
        return f"{program_dir} has no descriptor.json.gz: the Definitions its Build bound cannot be read"
    with gzip.open(p, "rt") as f:
        desc = json.load(f)
    got = program_digest(desc)
    if got != digest:
        return f"{p} is Program {got[:16]}, not the Program of record {str(digest)[:16]}"
    registry, _errors = load()
    ours: dict[str, Any] = {}
    seen: dict[str, str | None] = {}
    skew: list[str] = []
    for spec, bound in specs:
        try:
            fn = _ours(desc, bound, registry, ours)
            skew += [f"{bound}: {why}" for why in _differs(desc, fn, seen) + (_word_differs(fn, spec) if bound != spec else [])]
        except Exception as e:  # noqa: BLE001  -- a specialization this tree cannot rebuild is not shown to be the Build's
            skew.append(f"{bound}: this tree does not rebuild it ({type(e).__name__}: {e})")
    if not skew:
        return None
    more = f" (and {len(skew) - SHOWN} more)" if len(skew) > SHOWN else ""
    return f"{p}: this tree's Definitions are not the ones the Build of record bound: " + "; ".join(skew[:SHOWN]) + more


def _word_differs(fn: Any, spec: str) -> list[str]:
    """Why this tree's Boolean Definition ``fn`` is not on ``spec``, the word Definition its row is graded as, named: the word
    Definition it names (`verity.circuits.boolean.trace.word_of`), as the Build linked the row (`build_instances._word_rows`)."""
    from verity.circuits.boolean.trace import word_of
    from verity.circuits.codec import descriptor_id
    from verity.circuits.defs import CompositeDefinition, bind

    word = word_of(fn)
    link = descriptor_id(bind(word) if isinstance(word, CompositeDefinition) else word)
    return [] if link == spec else [f"this tree's names the word Definition {link}, and its row is graded as {spec}"]


def _ours(desc: dict[str, Any], fid: str, registry: Any, ours: dict[str, Any]) -> Any:
    """This tree's specialization for the descriptor's ``fid``: the registry's Definition bound at the Build's binding record."""
    from verity.circuits.codec import descriptor_id, lookup_primitive
    from verity.circuits.defs import CompositeDefinition, bind

    if fid in ours:
        return ours[fid]
    e = desc["definitions"].get(fid)
    if e is None:
        raise KeyError(f"the Build's descriptor holds no {fid}")
    if e["form"] == "primitive":
        fn = lookup_primitive(registry, fid)
    else:
        d = registry.defs.get(e["id"])
        if not isinstance(d, CompositeDefinition):
            raise KeyError(f"{e['id']} is not a composite Definition of this tree's registry")
        fn = bind(d, **{k: _static(desc, v, registry, ours) for k, v in e["bindings"].items()})
        if descriptor_id(fn) != fid:
            raise ValueError(f"this tree binds the Build's record as {descriptor_id(fn)}")
    ours[fid] = fn
    return fn


def _static(desc: dict[str, Any], v: Any, registry: Any, ours: dict[str, Any]) -> Any:
    """A binding record's value (`codec._static_json`) as this tree's: a function reference as this tree's specialization."""
    if isinstance(v, dict) and "fn" in v:
        return _ours(desc, v["fn"], registry, ours)
    if isinstance(v, dict) and "f64" in v:
        return float.fromhex(v["f64"])
    if isinstance(v, dict) and "dict" in v:
        return {k: _static(desc, x, registry, ours) for k, x in v["dict"].items()}
    if isinstance(v, list):
        return tuple(_static(desc, x, registry, ours) for x in v)
    return v


def _differs(desc: dict[str, Any], fn: Any, seen: dict[str, str | None]) -> list[str]:
    """Each Definition ``fn`` reaches in this tree whose entry is not the Build's descriptor's, named.  ``seen`` keeps each
    Definition's answer, so one that many specializations reach is encoded once."""
    from verity.circuits.codec import definition_digest, definition_entry, definitions

    out: list[str] = []
    # definitions reads a program's root and fn: here the specialization alone
    for g, d in definitions(SimpleNamespace(root=fn, fn=fn)).items():
        if g not in seen:
            if g not in desc["definitions"]:
                seen[g] = f"this tree's reaches {g}, which the Build's descriptor does not hold"
            elif (ours := definition_digest(definition_entry(d))) != (theirs := definition_digest(desc["definitions"][g], desc)):
                seen[g] = f"{g} is not the Definition the Build bound (entry {ours[:16]}, the Build's {theirs[:16]})"
            else:
                seen[g] = None
        why = seen[g]
        if why:
            out.append(why)
    return out
