"""Every ``*_evidence.json`` under ``verity/ml/circuits`` is keyed by what it describes and holds only ``art:`` ids.

Each file maps a registered key (an instruction id, a model name, a target name, a Definition id) to roles, and each role
to an ``art:`` id.  The prose the key carries (an instruction's ``evidence``, a model's ``validated``, a Definition's
``conformance``) names each role in backticks, so the text and the file line up by role.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition
from verity.ml.circuits.ops import prims
from verity.ml.circuits.silicon import instructions, integer, models
from verity.ml.circuits.targets import target

CIRCUITS = Path(instructions.__file__).resolve().parents[1]

ART = re.compile(r"art:[0-9a-f]{8,64}")
DOSSIER = re.compile(r"art:[0-9a-f]{64}")
#: Backticked words in the prose that are not roles: the record-only probe family.
NOT_ROLES = {"specials"}
TICK = re.compile(r"`([^`]+)`")
ROLE = re.compile(r"[a-z][a-z0-9_]*")


def _definitions(module) -> dict[str, str]:
    return {d.id: d.conformance for d in vars(module).values() if isinstance(d, (PrimitiveDefinition, CompositeDefinition))}


_QUOTED = {m.validated for m in models.MODELS.values() if isinstance(m, integer.IntegerMma)}

#: module -> (its EVIDENCE_IDS, key -> the prose that cites the entry's roles, roles the key's fields carry instead)
FILES = {
    "silicon/instructions_evidence.json": (
        instructions.EVIDENCE_IDS,
        {i.id: [e for e in i.evidence if e not in _QUOTED] for i in instructions.INSTRUCTIONS.values()}, {"dossier"}),
    "silicon/integer_evidence.json": (
        integer.EVIDENCE_IDS, {m.name: [m.validated] for m in models.MODELS.values() if isinstance(m, integer.IntegerMma)},
        set()),
    "targets/target_evidence.json": (target.EVIDENCE_IDS, {name: [] for name in target.TARGETS}, {"dossier"}),
    "ops/prims_evidence.json": (prims.EVIDENCE_IDS, {k: [v] for k, v in _definitions(prims).items()}, set()),
}


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
        for role, value in entry.items():
            pattern = DOSSIER if role == "dossier" else ART
            assert isinstance(value, str) and pattern.fullmatch(value), \
                f"{rel}[{key}].{role}: {value!r} is not an art: id (an evidence file holds art: ids only)"


@pytest.mark.parametrize("rel", sorted(FILES))
def test_the_prose_names_every_role_and_only_roles(rel: str) -> None:
    ids, prose, carried = FILES[rel]
    for key, texts in prose.items():
        entry = ids.get(key, {})
        named = {n for text in texts for n in TICK.findall(text)}
        for role in set(entry) - carried - named:
            pytest.fail(f"{rel}[{key}]: role {role!r} is named nowhere in its prose; name it in backticks where the text cites it")
        for n in named - NOT_ROLES:
            if ROLE.fullmatch(n):
                assert n in entry, f"{rel}[{key}]: the prose names `{n}`, which is no role of the entry"
