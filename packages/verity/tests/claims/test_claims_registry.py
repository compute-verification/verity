"""The claims vocabulary: composed assumptions, stable ids, kinds."""

from __future__ import annotations

import pytest

from verity.claims import ENTRIES, INSTANCES, PROPERTIES, assumption, get


def test_a_primitive_assumption_is_property_slash_instance_with_its_label_and_standing() -> None:
    e = get("cr/sha-512")
    assert (e.id, e.kind, e.name, e.label, e.standing) == ("cr/sha-512", "assumption", "Collision resistance of SHA-512", "CR · SHA-512", "standard")
    assert assumption("ro", "poseidon2-koalabear").standing == "algebraic"
    assert all(get(f"{p}/{i}").id == f"{p}/{i}" for p in PROPERTIES for i in INSTANCES)


def test_entries_are_keyed_by_id_and_unknown_ids_raise() -> None:
    assert all(k == e.id for k, e in ENTRIES.items()) and all(k == p.id for k, p in PROPERTIES.items())
    assert all(k == i.id for k, i in INSTANCES.items())
    assert {e.kind for e in ENTRIES.values()} == {"guarantee", "assumption", "model", "setup"}
    for bad in ("cr/md5", "nope/sha-256", "nothing"):
        with pytest.raises(KeyError):
            get(bad)


def test_composed_ids_never_shadow_entries() -> None:
    assert not set(ENTRIES) & {f"{p}/{i}" for p in PROPERTIES for i in INSTANCES}


def test_bare_ids_are_kebab_case_without_a_slash() -> None:
    import re

    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", k) for k in ENTRIES)
    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", k) for k in (*PROPERTIES, *INSTANCES))
