"""The registration's hiding commitment (`Spot.RegistrationCommitment.piece`) held to its Python mirror
(`registration_commitment`): the whole digest, the leaf, `opens`'s refusal and the bound coin agree on generated
registrations; each part of the registration moved after the commitment (circuit, partition, sampling design, compiler
digest, advice cap), a compiler confused with none and a wrong salt are refused by name in both; and coins bound to the
commitment are a function of it over the challenger's raw coins."""

from __future__ import annotations

import copy
import hashlib
import itertools
import random
import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.circuits.codec import canonical_json
from verity.protocols.proofs.spot.verifier import registration_commitment as RC
from verity.protocols.proofs.spot.verifier.challenger import challenger as C, coins as T

PACKAGE = (Path(__file__).resolve().parents[7] / "verity" / "protocols" / "proofs" / "spot" / "verifier" / "lean")
PIECES = "spot-difftest"
needs_lake = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

COMPILER = ("cd" * 64, 4096)
RECORD = {"format": "verity/registration/v1", "protocol": "verity/sampled-proofs/one-stage/v1",
          "program": "11" * 64, "query": "22" * 64, "partition": "33" * 64, "population": 288,
          "law": {"law": "subset", "k": 4}, "scheme": "frame-v3-sha512/hm96-sha512",
          "roots": [{"port": "x", "root": "44" * 64, "leaves": 4, "domain": "55" * 64}], "leaf_layer": None,
          "anchors": [], "previous": None,
          "window": {"kind": "served", "circuit": "66" * 64, "run": "ab" * 64,
                     "compiler": {"name": "b", "digest": COMPILER[0]}, "advice": {"root": "77" * 64, "bits": 100}}}
SALT = bytes(range(192))
COIN = bytes([0x5a]) * 64


def _moved(path: tuple[str, ...], value) -> dict:
    rec = copy.deepcopy(RECORD)
    at = rec
    for k in path[:-1]:
        at = at[k]
    at[path[-1]] = value
    return rec


#: each part of the registration moved after the commitment: (name, the record, the log's compiler the verifier reads)
MOVED = [
    ("program", _moved(("program",), "12" * 64), COMPILER),
    ("circuit", _moved(("window", "circuit"), "67" * 64), COMPILER),
    ("partition", _moved(("partition",), "34" * 64), COMPILER),
    ("law", _moved(("law",), {"law": "subset", "k": 5}), COMPILER),
    ("compiler-digest-window", _moved(("window", "compiler", "digest"), "ce" * 64), COMPILER),
    ("compiler-digest-log", RECORD, ("ce" * 64, COMPILER[1])),
    ("cap", RECORD, (COMPILER[0], COMPILER[1] + 1)),
    ("compiler-none", RECORD, None),
]


def test_an_honest_opening_holds():
    d = RC.whole(RECORD, COMPILER)
    RC.opens(RC.commitment(d, SALT), d, SALT, RECORD, COMPILER)
    c, d2, salt = RC.commit(RECORD, COMPILER)
    assert d2 == d and len(salt) == 192
    RC.opens(c, d, salt, RECORD, COMPILER)


@pytest.mark.parametrize("name, rec, compiler", MOVED, ids=[m[0] for m in MOVED])
def test_a_part_moved_after_the_commitment_is_refused(name, rec, compiler):
    d = RC.whole(RECORD, COMPILER)
    c = RC.commitment(d, SALT)
    assert RC.whole(rec, compiler) != d
    with pytest.raises(RC.Refused, match=f"^{RC.DIGEST}$"):
        RC.opens(c, d, SALT, rec, compiler)
    with pytest.raises(RC.Refused, match=f"^{RC.OPENING}$"):
        RC.opens(c, RC.whole(rec, compiler), SALT, rec, compiler)


def test_a_compiler_committed_as_none_and_read_as_present_is_refused():
    d = RC.whole(RECORD, None)
    with pytest.raises(RC.Refused, match=f"^{RC.DIGEST}$"):
        RC.opens(RC.commitment(d, SALT), d, SALT, RECORD, COMPILER)


@pytest.mark.parametrize("salt", [bytes([1]) + SALT[1:], SALT[:-1], SALT + b"\0"])
def test_a_wrong_salt_is_refused(salt):
    d = RC.whole(RECORD, COMPILER)
    with pytest.raises(RC.Refused, match=f"^{RC.OPENING}$"):
        RC.opens(RC.commitment(d, SALT), d, salt, RECORD, COMPILER)


def test_the_whole_digest_is_tagged_sha512_over_the_docket_digest_and_the_compiler():
    docket = hashlib.sha512(b"verity/registration/v1\0" + canonical_json(RECORD)).hexdigest()
    body = '{"compiler":{"cap":4096,"digest":"' + COMPILER[0] + '"},"record":"' + docket + '"}'
    assert RC.whole(RECORD, COMPILER) == hashlib.sha512(RC.TAG.encode() + b"\0" + body.encode()).digest()


def _coins() -> T.ChallengerCoins:
    ch = C.LiveChallenger("verity/test/coins/v1", 0, [("draw", 8)], random.Random("host").randbytes)
    return T.ChallengerCoins(C.LiveRecord(ch, "draw", itertools.count().__next__))


def test_bound_coins_follow_the_commitment_and_are_a_function_of_it():
    c = RC.commitment(RC.whole(RECORD, COMPILER), SALT)
    raw, bound = _coins(), RC.Bound(_coins())
    with pytest.raises(ValueError):
        bound(1)
    assert bound.commit(c).after == c
    raw.commit(c)
    plain = raw(3 * 64)
    got = bound(100) + bound(92)
    assert got == b"".join(RC.bound_coin(plain[i:i + 64], c) for i in range(0, 192, 64))
    assert RC.bound_coin(plain[:64], bytes([0xee]) * 64) != got[:64]
    with pytest.raises(ValueError):
        RC.Bound(_coins()).commit(c[:63])


def _compiler(j: dict) -> tuple[str, int] | None:
    c = j.get("compiler")
    return None if c is None else (c["digest"], c["cap"])


def reference(j: dict) -> dict:
    rec, compiler = j["record"], _compiler(j)
    salt, c = bytes.fromhex(j["salt"]), bytes.fromhex(j["commitment"])
    d = RC.whole(rec, compiler)
    try:
        RC.opens(c, bytes.fromhex(j["digest"]), salt, rec, compiler)
        why = ""
    except RC.Refused as e:
        why = str(e)
    return {"whole": d.hex(), "commitment": RC.commitment(d, salt).hex() if len(salt) == 192 else "",
            "opens": why, "coin": RC.bound_coin(bytes.fromhex(j["coin"]), c).hex()}


def case(rec: dict, compiler: tuple[str, int] | None, committed: tuple | None = None, digest: bytes | None = None,
         salt: bytes = SALT) -> dict:
    """The piece's input: the registration the verifier reads (``rec``, ``compiler``), the commitment to ``committed``
    (default the same registration) under `SALT`, and the opening's ``digest`` (default the committed one) and ``salt``."""
    d0 = RC.whole(*(committed or (rec, compiler)))
    j = {"record": rec, "digest": (digest or d0).hex(), "salt": salt.hex(),
         "commitment": RC.commitment(d0, SALT).hex(), "coin": COIN.hex()}
    if compiler is not None:
        j["compiler"] = {"digest": compiler[0], "cap": compiler[1]}
    return j


HEX = st.sampled_from(["11", "12", "33", "cd", "ce"]).map(lambda h: h * 64)
INPUTS = st.builds(
    lambda program, partition, k, cdig, cap, none, other: case(
        {**RECORD, "program": program, "partition": partition, "law": {"law": "subset", "k": k}},
        None if none else (cdig, cap), committed=(RECORD, COMPILER) if other else None),
    HEX, HEX, st.integers(1, 6), HEX, st.integers(0, 1 << 20), st.booleans(), st.booleans())
PORT = D.Port(PIECES, "registration_commitment", reference, INPUTS, package=PACKAGE, of_record="lean",
              need_rejects=False)

pieces = D.server_fixture(PORT)

CASES = [("honest", case(RECORD, COMPILER), "")]
CASES += [(f"{n}-digest", case(r, c, committed=(RECORD, COMPILER)), RC.DIGEST) for n, r, c in MOVED]
CASES += [(f"{n}-opening", case(r, c, committed=(RECORD, COMPILER), digest=RC.whole(r, c)), RC.OPENING)
          for n, r, c in MOVED]
CASES += [("none-read-as-present", case(RECORD, COMPILER, committed=(RECORD, None)), RC.DIGEST),
          ("wrong-salt", case(RECORD, COMPILER, salt=bytes([1]) + SALT[1:]), RC.OPENING),
          ("short-salt", case(RECORD, COMPILER, salt=SALT[:-1]), RC.OPENING)]


@needs_lake
@pytest.mark.parametrize("name, j, want", CASES, ids=[c[0] for c in CASES])
def test_lean_refuses_each_moved_part_and_a_wrong_salt_by_name(name, j, want, pieces):
    assert reference(j)["opens"] == want
    assert pieces.call("registration_commitment", j) == ("output", D.canonical(reference(j)))


@needs_lake
def test_the_registration_commitment_piece_agrees(pieces):
    stats = D.run(PORT, examples=300, server=pieces)
    assert stats.outputs > 100, stats
