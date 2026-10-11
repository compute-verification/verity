"""The claims vocabulary: composed assumptions, stable ids, kinds."""

from __future__ import annotations

import pytest

from verity.core.claims import ENTRIES, INSTANCES, PROPERTIES, assumption, get


def test_a_primitive_assumption_is_property_slash_instance_with_its_label_and_standing() -> None:
    e = get("cr/sha-512")
    assert (e.id, e.kind, e.name, e.label, e.standing) == ("cr/sha-512", "assumption", "Collision resistance of SHA-512", "CR · SHA-512", "standard")
    assert assumption("cr", "poseidon2-koalabear").standing == "algebraic" and get("xof/shake-256").label == "XOF · SHAKE-256"
    assert all(get(f"{p}/{i}").id == f"{p}/{i}" for p in PROPERTIES for i in INSTANCES)
    e = get("ecr/sha-512")
    assert (e.name, e.label, e.standing) == ("Expected-time collision resistance of SHA-512", "ECR · SHA-512", "standard")
    e = get("prf/sha-512")
    assert (e.name, e.label, e.standing) == ("Pseudorandom function of SHA-512", "PRF · SHA-512", "standard")
    e = get("cr/turboshake128")
    assert (e.name, e.label, e.standing) == ("Collision resistance of TurboSHAKE128", "CR · TurboSHAKE128", "standard")
    assert get("uniform/python-secrets").name == "Independent uniform output of Python's secrets"
    assert get("uniform/flock-coin-server").name == "Independent uniform output of Flock's coin server"
    e = get("euf-cma/lms-sha512-n64")
    assert (e.label, e.standing) == ("EUF-CMA · LMS over SHA-512", "standard")
    assert "lms-sha512-n32" not in INSTANCES
    e = get("uniform/io-getrandombytes")
    assert (e.name, e.label, e.standing) == ("Independent uniform output of IO.getRandomBytes", "Uniform · IO.getRandomBytes",
                                             "standard")


def test_entries_are_keyed_by_id_and_unknown_ids_raise() -> None:
    assert all(k == e.id for k, e in ENTRIES.items()) and all(k == p.id for k, p in PROPERTIES.items())
    assert all(k == i.id for k, i in INSTANCES.items())
    assert {e.kind for e in ENTRIES.values()} == {"guarantee", "assumption", "model"}
    for bad in ("cr/md5", "nope/sha-256", "nothing"):
        with pytest.raises(KeyError):
            get(bad)


def test_composed_ids_never_shadow_entries() -> None:
    assert not set(ENTRIES) & {f"{p}/{i}" for p in PROPERTIES for i in INSTANCES}


def test_bare_ids_are_kebab_case_without_a_slash() -> None:
    import re

    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", k) for k in ENTRIES)
    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", k) for k in (*PROPERTIES, *INSTANCES))


def test_bandwidth_device_claims_are_measured_and_cited_separately() -> None:
    device, inst = get("bandwidth-device"), get("bandwidth-instantiation")
    assert (device.kind, device.standing) == ("model", None)
    assert all((get(i).kind, get(i).standing) == ("assumption", "measured")
               for i in ("bandwidth-s-chip", "bandwidth-side-cap", "bandwidth-regen-time"))
    assert (inst.kind, inst.standing) == ("assumption", "conjectured") and "uniformly random blocks" in inst.name


def test_the_sites_nine_assumptions_resolve_and_the_retired_ones_do_not() -> None:
    site = ("cr/sha-256", "cr/sha-512", "cr/blake3", "cr/blake2b", "cr/poseidon2-babybear", "cr/poseidon2-koalabear", "xof/shake-256",
            "random-oracle", "honest-verifier")
    assert all(get(i).id == i and get(i).kind == "assumption" for i in site)
    assert not [k for k in ENTRIES if any(r in k for r in ("beacon", "auditor", "challenger"))]
