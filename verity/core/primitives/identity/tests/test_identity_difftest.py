"""`core_identity` against the Python reference, through the core difftest harness (`verity/core/difftest`).

`identity-difftest` (`Identity.Difftest.table`) serves streaming SHA-512 absorbed in pieces (against `hashlib.sha512`),
`taggedSha512` (against `identity.tagged_sha512`) and `identityDigestSha512` (against `identity.identity_digest_sha512`).
Tags are drawn from all of Unicode, the empty tag and non-strings among them, which Python refuses and so must Lean;
payloads cross the 128-byte block boundaries and the padding's 112-byte edge; manifests are any JSON with integers (Python
writes no float into a request, so a float manifest is out of the harness's reach: `Flock.canon` refuses a non-integer
number, as Python refuses a float).
"""

from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.ml.commitments.identity import identity_digest_sha512, tagged_sha512
from verity_difftest import protocol as P

LEAN = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

REJECTS = (D.Reject, TypeError, ValueError)
"""What the reference raises on an input it refuses: the harness's own, and `identity`'s (`TypeError` for an empty or
non-string tag)."""


def _str(i, k: str) -> str:
    v = i.get(k)
    if not isinstance(v, str):
        raise D.Reject(f"{k} is missing or not a string")
    return v


def sha512(i) -> dict:
    payload = P.unhex(_str(i, "payload"))
    pieces = i.get("pieces")
    if not isinstance(pieces, list) or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in pieces):
        raise D.Reject("pieces is missing or not a list of naturals")
    if sum(pieces) != len(payload):
        raise D.Reject("the pieces are not the payload")
    return {"sha512": hashlib.sha512(payload).hexdigest()}


def tagged(i) -> dict:
    if "tag" not in i:
        raise D.Reject("tag missing")
    return {"digest": tagged_sha512(i["tag"], P.unhex(_str(i, "payload"))).hex()}


def identity(i) -> dict:
    if "tag" not in i or "manifest" not in i:
        raise D.Reject("tag or manifest missing")
    return {"digest": identity_digest_sha512(i["tag"], i["manifest"]).hex()}


LENGTHS = [0, 1, 17, 111, 112, 113, 127, 128, 129, 239, 240, 255, 256, 257]
"""Around each block boundary and the padding's edge (a pending block of 112 bytes or more takes a second block)."""
TAGS = st.one_of(st.sampled_from(["", "t", "verity-pous/w-commitment/v2", "é", "\U0001f600", "a\x00b", "verity/x" * 40]),
                 st.text(max_size=40), st.sampled_from([0, None, ["t"]]))


@st.composite
def payloads(draw) -> str:
    rng = random.Random(draw(st.integers(0, 1 << 32)))
    data = rng.randbytes(draw(st.sampled_from(LENGTHS + [rng.randrange(2000)]))).hex()
    return draw(st.sampled_from([data] * 8 + [data.upper() or "Z", data + "0", data + " "]))


@st.composite
def sha512_inputs(draw) -> dict:
    payload = draw(payloads())
    n = len(payload) // 2
    rng = random.Random(draw(st.integers(0, 1 << 32)))
    cuts = sorted(rng.randrange(n + 1) for _ in range(draw(st.integers(0, 4))))
    pieces = [b - a for a, b in zip([0] + cuts, cuts + [n])]
    if draw(st.integers(0, 9)) == 0:
        pieces.append(1)
    return {"payload": payload, "pieces": pieces}


JSON = st.recursive(
    st.none() | st.booleans() | st.integers(-(1 << 130), 1 << 130) | st.text(max_size=12),
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=6), inner, max_size=4),
    max_leaves=12)

FNS = {
    "sha512": D.Port("identity-difftest", "sha512", sha512, sha512_inputs(), package=LEAN),
    "tagged": D.Port("identity-difftest", "tagged", tagged, st.fixed_dictionaries({"tag": TAGS, "payload": payloads()}),
                     package=LEAN, rejects=REJECTS),
    "identity": D.Port("identity-difftest", "identity", identity, st.fixed_dictionaries({"tag": TAGS, "manifest": JSON}),
                       package=LEAN, rejects=REJECTS),
}
server = D.server_fixture(FNS["sha512"])


@pytest.mark.parametrize("name", sorted(FNS))
def test_the_function_agrees(name, server):
    stats = D.run(FNS[name], examples=500, server=server)
    assert stats.outputs > 150 and stats.rejects > 10, stats


@pytest.mark.parametrize("name", ["tagged", "identity"])
def test_the_empty_tag_is_refused(name, server):
    """Python's `TypeError("hash tag must be a nonempty string")`, which neither Lean copy raised before this package."""
    value = {"tag": "", "payload": "00"} if name == "tagged" else {"tag": "", "manifest": {"a": 1}}
    with pytest.raises(TypeError, match="nonempty"):
        FNS[name].reference(value)
    assert D.replay(FNS[name], value, server)[0] == "reject"
