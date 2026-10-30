"""Every ``*_evidence.json`` under ``verity/ml/circuits`` is keyed by what it describes and holds only well-formed ids.

Each file maps a registered key (an instruction id, a model or pipeline name, a target name, a Definition id) to roles,
and each role to an id, a list of ids or nested roles.  The prose the key carries (an instruction's ``evidence``, a model's
``validated``, a Definition's ``conformance``) names each role in backticks, so the text and the file line up by role.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition
from verity.ml.circuits.ops import kv_cache, moe, prims
from verity.ml.circuits.silicon import instructions, integer, models, total_fp8
from verity.ml.circuits.targets import target

CIRCUITS = Path(instructions.__file__).resolve().parents[1]

RUN = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")
ART = re.compile(r"art:[0-9a-f]{8,64}")
NOTE = re.compile(r"note:\d{8}T\d{4}Z-[a-z0-9-]+")
DOSSIER = re.compile(r"art:[0-9a-f]{64}")
DEVICE = re.compile(r"GPU-[0-9a-f]{8}(-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?")
DATASET = re.compile(r"docs/data/[a-z0-9-]+|[a-z0-9]+(-[a-z0-9]+)*-v\d+")
#: Backticked words in the prose that are not roles: the record-only probe family.
NOT_ROLES = {"specials"}
TICK = re.compile(r"`([^`]+)`")
ROLE_PATH = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*")


def _definitions(module) -> dict[str, str]:
    return {d.id: d.conformance for d in vars(module).values() if isinstance(d, (PrimitiveDefinition, CompositeDefinition))}


def _own(module, names) -> dict[str, str]:
    return {m.name: m.validated for m in (getattr(module, n) for n in names) if hasattr(m, "validated")}


_QUOTED = {m.validated for m in models.MODELS.values() if isinstance(m, integer.IntegerMma)}

#: module -> (its EVIDENCE_IDS, key -> the prose that cites the entry's roles, roles the key's fields carry instead)
FILES = {
    "silicon/instructions_evidence.json": (
        instructions.EVIDENCE_IDS,
        {i.id: [e for e in i.evidence if e not in _QUOTED] for i in instructions.INSTRUCTIONS.values()}, {"dossier"}),
    "silicon/models_evidence.json": (
        models.EVIDENCE_IDS, {name: [m.validated] for name, m in models.MODELS.items()}, set()),
    "silicon/integer_evidence.json": (
        integer.EVIDENCE_IDS, {name: [v] for name, v in _own(integer, integer.__all__).items()}, set()),
    "silicon/total_fp8_evidence.json": (
        total_fp8.EVIDENCE_IDS, {name: [v] for name, v in _own(total_fp8, total_fp8.__all__).items()}, set()),
    "targets/target_evidence.json": (target.EVIDENCE_IDS, {name: [] for name in target.TARGETS}, {"dossier"}),
    "ops/prims_evidence.json": (prims.EVIDENCE_IDS, {k: [v] for k, v in _definitions(prims).items()}, set()),
    "ops/moe_evidence.json": (moe.EVIDENCE_IDS, {k: [v] for k, v in _definitions(moe).items()}, set()),
    "ops/kv_cache_evidence.json": (kv_cache.EVIDENCE_IDS, {k: [v] for k, v in _definitions(kv_cache).items()}, set()),
}


def _leaves(role: str, value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _leaves(f"{role}.{k}", v)
    elif isinstance(value, list):
        assert value, f"{role}: an empty list"
        for v in value:
            assert isinstance(v, str), f"{role}: a list holds ids, not {v!r}"
            yield role, v
    else:
        yield role, value


def _paths(entry: dict, prefix: str = ""):
    for k, v in entry.items():
        yield prefix + k
        if isinstance(v, dict):
            yield from _paths(v, f"{prefix}{k}.")


def _well_formed(role: str, value) -> bool:
    last = role.rsplit(".", 1)[-1]
    if not isinstance(value, str):
        return False
    if last == "dossier":
        return bool(DOSSIER.fullmatch(value))
    if last == "device":
        return bool(DEVICE.fullmatch(value))
    if last == "dataset":
        return bool(DATASET.fullmatch(value))
    return any(p.fullmatch(value) for p in (RUN, ART, NOTE))


def test_every_evidence_file_is_checked_here() -> None:
    on_disk = {p.relative_to(CIRCUITS).as_posix() for p in CIRCUITS.rglob("*_evidence.json")}
    assert on_disk == set(FILES), f"add {sorted(on_disk - set(FILES))} to FILES (or drop {sorted(set(FILES) - on_disk)})"


@pytest.mark.parametrize("rel", sorted(FILES))
def test_keys_are_registered_and_ids_well_formed(rel: str) -> None:
    ids, prose, _ = FILES[rel]
    unknown = sorted(set(ids) - set(prose))
    assert not unknown, f"{rel}: {unknown} name no registered key; key each entry by what it describes"
    for key, entry in ids.items():
        assert isinstance(entry, dict) and entry, f"{rel}[{key}]: an entry maps roles to ids"
        for role, v in entry.items():
            for path, value in _leaves(role, v):
                assert _well_formed(path, value), f"{rel}[{key}].{path}: {value!r} is not a well-formed id for that role"


@pytest.mark.parametrize("rel", sorted(FILES))
def test_the_prose_names_every_role_and_only_roles(rel: str) -> None:
    ids, prose, carried = FILES[rel]
    for key, entry in ids.items():
        named = {n for text in prose[key] for n in TICK.findall(text)}
        paths = set(_paths(entry))
        for path in paths - carried:
            parts = path.split(".")
            covered = {".".join(parts[:i]) for i in range(1, len(parts) + 1)} & named
            assert covered or any(n.startswith(path + ".") for n in named), \
                f"{rel}[{key}]: role {path!r} is named nowhere in its prose; name it in backticks where the text cites it"
        for n in named - NOT_ROLES:
            if ROLE_PATH.fullmatch(n):
                assert n in paths, f"{rel}[{key}]: the prose names `{n}`, which is no role of the entry"
