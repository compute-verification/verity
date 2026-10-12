"""The pluggable leaf scheme (``leaf/``): the registry, and the D0 contract that moving Poseidon2 behind the interface
changed NOTHING on the wire -- ``fp8-ada+hash`` == ``fp8-ada+poseidon2`` == ``fp8-ada`` byte for byte (system bytes,
``sys_id``, the Rust pins, the checked-in v5 fixtures) -- on the CPU."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from . import hashauth, hashchain, protocol  # noqa: E402
from .leaf import base as leaf_base  # noqa: E402
from .leaf import registry  # noqa: E402
from .leaf.poseidon2 import Poseidon2Leaf  # noqa: E402
from .relations import relation  # noqa: E402
from .serialize import read_statement, read_system, statement_bytes, system_bytes, verify_files  # noqa: E402

RUST = Path(__file__).resolve().parents[2] / "ligero-verify"
FIXTURES = RUST / "fixtures"


def _rust_pins() -> dict[str, tuple[str, str]]:
    """``relation.rs``: ``name -> (hashed_sys_id, hashed_table_digest)`` of every relation with a hashed pin."""
    src = (RUST / "src" / "relation.rs").read_text()
    out = {}
    for m in re.finditer(r'name: "([\w-]+)",.*?hashed_sys_id: "([0-9a-f]*)",\s*hashed_table_digest: "([0-9a-f]*)"', src, re.S):
        if m.group(2):
            out[m.group(1)] = (m.group(2), m.group(3))
    assert {"fp8-ada", "bf16-hopper", "fp8-hopper", "bf16-ampere"} <= set(out)
    return out


def _rust_leaf_consts() -> dict[str, str]:
    """``leaf.rs``: the Poseidon2 scheme's ``schema`` / ``params_sha256`` / ``digest_elems`` as the verifier pins them."""
    src = (RUST / "src" / "leaf.rs").read_text()
    blk = re.search(r"pub static POSEIDON2: LeafScheme = LeafScheme \{(.*?)\};", src, re.S).group(1)
    return {k: v.strip().strip('"') for k, v in re.findall(r"(\w+): ([^,\n]+),", blk)}


# ---------------------------------------------------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------------------------------------------------

def test_registry_names_aliases_and_relation_strings() -> None:
    assert registry.DEFAULT == "poseidon2" and "poseidon2" in registry.names()
    assert registry.canonical("hash") == registry.canonical("poseidon2") == registry.canonical(None) == "poseidon2"
    assert isinstance(registry.get(), Poseidon2Leaf) and isinstance(registry.get("hash"), Poseidon2Leaf)
    assert registry.get() is not registry.get()                  # a fresh instance per call (gadgets hold per-compose state)
    with pytest.raises(KeyError):
        registry.get("nope")
    assert registry.split_relation_name("fp8-ada") == ("fp8-ada", "poseidon2")
    assert registry.split_relation_name("fp8-ada+hash") == ("fp8-ada", "poseidon2")
    assert registry.split_relation_name("fp8-ada+poseidon2") == ("fp8-ada", "poseidon2")
    assert registry.relation_name("fp8-ada", "poseidon2") == registry.relation_name("fp8-ada", "hash") == "fp8-ada"
    assert registry.relation_name("fp8-ada", "blake3") == "fp8-ada+blake3"
    for bad in ("+poseidon2", "fp8-ada+a+b"):
        with pytest.raises(ValueError):
            registry.split_relation_name(bad)
    with pytest.raises(KeyError):
        registry.split_relation_name("fp8-ada+nope")
    assert registry.by_schema(Poseidon2Leaf.schema).name == "poseidon2"
    leaf_base.check_scheme(registry.get())


def test_poseidon2_scheme_matches_the_rust_verifier_constants() -> None:
    s = Poseidon2Leaf()
    rc = _rust_leaf_consts()
    assert (rc["name"], rc["schema"], rc["params_sha256"], int(rc["digest_elems"])) == (s.name, s.schema, s.params_sha256, s.digest_elems)
    assert s.carry_elems == 8 and s.hash_name == "poseidon2-babybear-w24"
    rows = np.random.default_rng(3).integers(0, 256, size=(5, 32 * 48), dtype=np.int64)
    d = s.native(rows, word_bits=8, role=0)
    assert d.shape == (5, 8) and d.dtype == np.int64 and (0 <= d).all() and (d < protocol.P).all()
    b = s.leaf_bytes(d[0])
    assert len(b) == 32 and b == b"".join(int(x).to_bytes(4, "big") for x in d[0])


# ---------------------------------------------------------------------------------------------------------------------
# D0: byte identity
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["fp8-ada", "bf16-hopper", "fp8-hopper", "bf16-ampere"])
def test_hash_poseidon2_and_default_compose_to_one_system_with_the_rust_pin(name: str) -> None:
    rel = relation(name)
    default = hashchain.compose(rel)
    raw = system_bytes(default.sys)
    for leaf in ("hash", "poseidon2", Poseidon2Leaf()):
        hr = hashchain.compose(rel, leaf)
        assert hr.leaf.name == "poseidon2" and system_bytes(hr.sys) == raw, f"{name}+{leaf}: system bytes differ"
        assert hr.census == default.census
    assert protocol.system_id(default.sys).hex() == _rust_pins()[name][0], f"{name}: the Rust hashed_sys_id pin moved"
    assert default.census["leaf"] == "poseidon2" and default.census["digest_elems"] == 8 and default.census["carry_elems"] == 8
    assert default.census["linked_rows"] == 19 and default.census["rows_digest_pins"] == 17


@pytest.mark.parametrize("fixture,name", [("fp8-ada-hash", "fp8-ada"), ("bf16-ampere-hash", "bf16-ampere")])
def test_checked_in_v5_fixtures_read_as_poseidon2_and_round_trip(fixture: str, name: str) -> None:
    """The fixtures were written before the leaf interface existed: they parse as the default scheme, serialize back to
    the same bytes, and their system.bin is what ``compose`` builds today."""
    d = FIXTURES / fixture
    raw = (d / "sub_00.stmt").read_bytes()
    st = read_statement(raw)
    assert st.v5 and st.relation == name and st.base_relation == name and st.leaf == "poseidon2"
    assert isinstance(st.leaf_scheme(), Poseidon2Leaf)
    assert statement_bytes(st) == raw
    assert st.digests().shape == (st.n_vus, 16)
    sysb = (d / "system.bin").read_bytes()
    hr = hashchain.compose(relation(name))
    assert system_bytes(hr.sys) == sysb
    man = json.loads((d / "manifest.json").read_text())
    assert (man["relation"]["name"], man["authentication"], man["system"]["rows"]) == (name, "included-hash", hr.sys.m)
    assert read_system(sysb)["m"] == hr.sys.m and sysb[-32:] == protocol.system_id(hr.sys)


def test_fixture_fp8_ada_verifies_in_python_through_the_generic_runner() -> None:
    """``verify_files`` resolves the runner from the relation string ("fp8-ada" -> HashedRelationRunner, poseidon2) and
    accepts the fixture proof with its own coins; the multiproof check runs through ``LeafScheme.leaf_bytes``."""
    from .serialize import read_coins
    d = FIXTURES / "fp8-ada-hash"
    coins = read_coins((d / "sub_00.coins").read_bytes())
    ok, why, _ = verify_files(d / "sub_00.stmt", d / "sub_00.proof", "cpu", own_coins=coins)
    assert ok, why
    # the digest table is what the block authenticates: a lane flipped fails the multiproof (through leaf_bytes)
    st = read_statement((d / "sub_00.stmt").read_bytes())
    dig = st.digests().copy()
    dig[3, 9] ^= 1
    ok, why = hashauth.verify_hash_auth(st.hash_auth, st.hash_sibs, digests=dig,
                                        y_words=st.y16[np.arange(st.n_vus) * st.steps + st.steps - 1], y_bits=22, leaf="poseidon2")
    assert not ok and why.startswith("auth: b:")
    ok, why = hashauth.verify_hash_auth(st.hash_auth, st.hash_sibs, digests=st.digests(),
                                        y_words=st.y16[np.arange(st.n_vus) * st.steps + st.steps - 1], y_bits=22, leaf="poseidon2")
    assert ok, why


def test_digest_bytes_are_the_pre_registry_bytes() -> None:
    """``hashauth.digest_bytes`` (what the statement digest absorbs) spells the scheme's schema and parameters digest
    exactly where the ``+hash`` statements had Poseidon2's -- the Rust ``hash_digest_bytes`` mirrors it."""
    st = read_statement((FIXTURES / "fp8-ada-hash" / "sub_00.stmt").read_bytes())
    b = hashauth.digest_bytes(st.hash_auth)
    assert b == hashauth.digest_bytes(st.hash_auth, leaf="poseidon2") == hashauth.digest_bytes(st.hash_auth, leaf=Poseidon2Leaf())
    from verity.ml.commitments import poseidon2_babybear as h2
    from verity.ml.commitments.rowleaf import SCHEMA_ROW
    assert b.startswith(b"|auth=ligero-b/auth/v2h|schema=" + SCHEMA_ROW.encode() + b"|p2=" + bytes.fromhex(h2.CONSTANTS_SHA256) + b"|idx=")
