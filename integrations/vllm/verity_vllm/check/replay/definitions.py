"""The Definitions the sampled replay evaluates are the ones the Build of record bound, and its row kernels compute them.

A row kernel answers for its family's Definition in the replay's own tree, and the Commit bound the Program the Build of record
compiled from the Build's tree.  A Definition whose body moved between the two trees would have the replay grade the committed
words against a circuit the Commit never bound, and the Program digest the loaders check is the Build's recorded string, which no
tree change moves.  `refusal` reads the Build's ``descriptor.json.gz``, holds it to the digest of record, and holds each
specialization the replay evaluates, with every Definition it reaches, to the descriptor's entry for it
(`verity.ml.circuits.annotations.definition_digest`: the entry with the v1 interning resolved, the same for one Definition in any
Program).

That holds the Definitions, not the code that grades by them.  A primitive is held to its entry, its signature and ports only:
its evaluator is code, named by its id.  And the replay grades a row by its family's row kernel, not by the Definition.  So
before any row is graded, `kernel_refusal` runs core `self_check` of each graded family's row kernel against the Definition's
reference, which evaluates through the primitives' evaluators (`kernel_registry.check`, once a family per process).  That is a
sample, the limit of what this module holds: 6 seeded instances of each of the kernel's check specialisations, small statics the
reference evaluates, not the served ones.  A kernel that is its Definition there but not on a committed row's operands, or not at
the served statics, grades that row by the kernel; what holds it there is the kernel's own tests, not this check.  A delegate's
family (`delegate.DELEGATES`) has no row kernel and is not self-checked here: its protocol answers for it.
"""
from __future__ import annotations

import gzip
import json
import os
from types import SimpleNamespace
from typing import Any, Iterable

SHOWN = 5
#: family -> why its row kernel is not its Definition in this tree ([] when it is), checked once a process
_KERNEL_CHECKS: dict[str, list[str]] = {}


def evaluated_specs(rows: Iterable[dict[str, Any]]) -> list[str]:
    """The specializations of the rows the replay evaluates: a family with a row kernel (`rows.ROWS`) or a protocol's delegate."""
    from verity_vllm.check.replay.delegate import DELEGATES
    from verity.ml.definitions.numerics.rows import ROWS
    from verity_vllm.query.manifest.format import spec_family

    return sorted({str(r["spec"]) for r in rows if spec_family(str(r["spec"])) in ROWS or spec_family(str(r["spec"])) in DELEGATES})


def of_record(dirs: dict[str, str], digs: dict[str, str], rows: dict[str, list[dict[str, Any]]], *, what: str) -> None:
    """Raise, by name, unless each request's Build dir (``dirs``) holds its Program of record (``digs``), the Definitions this
    tree evaluates its rows (``rows``) by are the ones that Build bound, and the row kernels it evaluates them with pass their
    self-check (`kernel_refusal`).  One check per Build dir."""
    by_dir: dict[str, tuple[str, set[str]]] = {}
    for k, d in dirs.items():
        by_dir.setdefault(d, (digs[k], set()))[1].update(evaluated_specs(rows[k]))
    for d, (digest, specs) in sorted(by_dir.items()):
        why = refusal(d, digest, sorted(specs))
        if why:
            raise RuntimeError(f"{what}: REFUSED (not the Definitions of record): {why}")
    why = kernel_refusal(set().union(*(s for _, s in by_dir.values())))
    if why:
        raise RuntimeError(f"{what}: REFUSED (a row kernel is not its Definition): {why}")


def kernel_refusal(specs: Iterable[str]) -> str | None:
    """Why a row kernel this tree grades ``specs`` by is not its family's Definition, or None: each family's row kernel against
    the reference by core `self_check` (`kernel_registry.check`; module docstring), once a family per process."""
    from verity_vllm.commit import challenge as CH
    from verity.ml.definitions.numerics import kernel_registry as KR
    from verity.ml.definitions.numerics.rows import KERNEL, ROWS
    from verity_vllm.query.manifest.format import spec_family

    bad: list[str] = []
    for family in sorted({spec_family(str(s)) for s in specs} & set(ROWS)):
        if family not in _KERNEL_CHECKS:
            _KERNEL_CHECKS[family] = KR.check(family, KERNEL, CH.generator)
        bad += [f"{family}: {why}" for why in _KERNEL_CHECKS[family]]
    if not bad:
        return None
    more = f" (and {len(bad) - SHOWN} more)" if len(bad) > SHOWN else ""
    return "; ".join(bad[:SHOWN]) + more


def refusal(program_dir: str, digest: str, specs: Iterable[str]) -> str | None:
    """Why the Definitions this tree evaluates ``specs`` by are not the ones the Build at ``program_dir`` bound into Program
    ``digest``, or None.  With no ``specs`` nothing is evaluated, so nothing is read."""
    from verity.ml.circuits.codec import program_digest
    from verity.ml.definitions.registry.catalog import load

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
    for fid in specs:
        try:
            skew += [f"{fid}: {why}" for why in _differs(desc, _ours(desc, fid, registry, ours), seen)]
        except Exception as e:  # noqa: BLE001  -- a specialization this tree cannot rebuild is not shown to be the Build's
            skew.append(f"{fid}: this tree does not rebuild it ({type(e).__name__}: {e})")
    if not skew:
        return None
    more = f" (and {len(skew) - SHOWN} more)" if len(skew) > SHOWN else ""
    return f"{p}: this tree's Definitions are not the ones the Build of record bound: " + "; ".join(skew[:SHOWN]) + more


def _ours(desc: dict[str, Any], fid: str, registry: Any, ours: dict[str, Any]) -> Any:
    """This tree's specialization for the descriptor's ``fid``: the registry's Definition bound at the Build's binding record."""
    from verity.ml.circuits.codec import descriptor_id, lookup_primitive
    from verity.ml.circuits.defs import CompositeDefinition, bind

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
    """Each Definition ``fn`` reaches in this tree whose entry is not the Build's descriptor's, named."""
    from verity.ml.circuits.annotations import definition_digest
    from verity.ml.circuits.codec import encode_program

    # encode_program reads a program's root, fn and annotations: here the specialization alone, in the Build's schema
    mine = encode_program(SimpleNamespace(root=fn, fn=fn, annotations={}), desc["schema"])
    out = []
    for g in mine["definitions"]:
        if g not in seen:
            if g not in desc["definitions"]:
                seen[g] = f"this tree's reaches {g}, which the Build's descriptor does not hold"
            elif definition_digest(mine, g) != definition_digest(desc, g):
                seen[g] = f"{g} is not the Definition the Build bound (entry {definition_digest(mine, g)[:16]}, the Build's " \
                          f"{definition_digest(desc, g)[:16]})"
            else:
                seen[g] = None
        if seen[g]:
            out.append(seen[g])
    return out
