"""`verity.ml.crypto.lms`'s HSS: a device's identity, the lower key it certifies for each party, the roles' digests a lower
key signs, and the two-level signature, verified and refused, on small trees (H = 4)."""
from __future__ import annotations

import hashlib
import stat

import pytest

from verity.ml.crypto import lms as R

SMALL = R.Params(64, 1, 4)
DEPLOYMENT = hashlib.sha512(b"a deployment configuration").digest()
OTHER = hashlib.sha512(b"another deployment configuration").digest()
ATTACHMENT = bytes(range(32))
I = bytes(range(16))
LOWER = bytes(range(80))
ROOT = bytes(range(64, 128))
#: the role digests of I, DEPLOYMENT, "memory-challenger", ATTACHMENT, "server-3", period 7: the party key over LOWER, the
#: period's root over ROOT with 5 records, and block 2's liveness, each its first 16 bytes
PINNED = {"party-key": "aac07cba2b023595971ea1f884604caf", "period-root": "890d26452f5bbaeafce3da7ebb6b3ba2",
          "liveness": "ea5dac1a35dcab621d4d99aa83e7bd5d"}


@pytest.fixture
def device(tmp_path) -> R.Identity:
    return R.Identity.init(tmp_path / "device", SMALL)


def _messages() -> dict[str, bytes]:
    return {"party-key": R.party_key_message(I, DEPLOYMENT, "memory-challenger", LOWER),
            "period-root": R.period_root_message(I, DEPLOYMENT, "memory-challenger", ATTACHMENT, "server-3", 7, 5, ROOT),
            "liveness": R.liveness_message(I, DEPLOYMENT, "memory-challenger", ATTACHMENT, "server-3", 7, 2)}


def test_pinned_role_digests() -> None:
    """Each role's digest is pinned, so a change to a message's bytes is a visible change (the Lean holds them too)."""
    assert {k: m[:16].hex() for k, m in _messages().items()} == PINNED


def test_role_digests_bind_every_field() -> None:
    """Changing any one field of a message changes its digest, and the three roles never share one."""
    base = _messages()
    assert len(set(base.values())) == 3 and all(len(m) == 64 for m in base.values())
    variants = [R.party_key_message(I, OTHER, "memory-challenger", LOWER),
                R.party_key_message(I, DEPLOYMENT, "network-certifier", LOWER),
                R.party_key_message(I, DEPLOYMENT, "memory-challenger", LOWER[::-1]),
                R.period_root_message(I, DEPLOYMENT, "memory-challenger", ATTACHMENT[::-1], "server-3", 7, 5, ROOT),
                R.period_root_message(I, DEPLOYMENT, "memory-challenger", ATTACHMENT, "server-4", 7, 5, ROOT),
                R.period_root_message(I, DEPLOYMENT, "memory-challenger", ATTACHMENT, "server-3", 8, 5, ROOT),
                R.period_root_message(I, DEPLOYMENT, "memory-challenger", ATTACHMENT, "server-3", 7, 6, ROOT),
                R.period_root_message(I, DEPLOYMENT, "memory-challenger", ATTACHMENT, "server-3", 7, 5, ROOT[::-1]),
                R.liveness_message(I, DEPLOYMENT, "memory-challenger", ATTACHMENT, "server-3", 7, 3),
                R.liveness_message(I[::-1], DEPLOYMENT, "memory-challenger", ATTACHMENT, "server-3", 7, 2)]
    assert not set(variants) & set(base.values()) and len(set(variants)) == len(variants)


@pytest.mark.parametrize("bad", [dict(deployment=b"\0" * 63), dict(period=-1), dict(period=2 ** 64), dict(period=True)])
def test_malformed_fields_are_refused(bad) -> None:
    args = dict(I=I, deployment=DEPLOYMENT, party="p", attachment=ATTACHMENT, link="l", period=0, count=1, root=ROOT) | bad
    with pytest.raises(ValueError):
        R.period_root_message(**args)


def test_a_party_signs_roots_and_liveness_that_verify_under_the_identity(device) -> None:
    """The party's lower key, certified by the identity for the party in the deployment, signs a period's root and a
    block's liveness, and both verify under the identity alone."""
    party = device.issue(DEPLOYMENT, "memory-challenger")
    root = party.period_root(ATTACHMENT, "server-3", 7, 5, ROOT)
    I_low = party.lower[:16]
    m = R.period_root_message(I_low, DEPLOYMENT, "memory-challenger", ATTACHMENT, "server-3", 7, 5, ROOT)
    assert R.hss_verify(SMALL, device.public, DEPLOYMENT, "memory-challenger", root, m)
    live = party.liveness(ATTACHMENT, "server-3", 7, 2)
    assert R.verify_liveness(SMALL, device.public, DEPLOYMENT, "memory-challenger", ATTACHMENT, "server-3", 7, 2, live)
    assert live.signature.q == root.signature.q + 1
    assert R.HssSignature.decode(SMALL, live.encode()) == live


def test_a_signature_is_refused_off_its_party_deployment_or_fields(device, tmp_path) -> None:
    party = device.issue(DEPLOYMENT, "memory-challenger")
    live = party.liveness(ATTACHMENT, "server-3", 7, 2)
    ok = dict(params=SMALL, identity=device.public, deployment=DEPLOYMENT, party="memory-challenger", attachment=ATTACHMENT,
              link="server-3", period=7, block=2, sig=live)
    assert R.verify_liveness(**ok)
    other = R.Identity.init(tmp_path / "other", SMALL)
    for bad in [dict(identity=other.public), dict(deployment=OTHER), dict(party="network-certifier"),
                dict(attachment=ATTACHMENT[::-1]), dict(link="server-4"), dict(period=8), dict(block=3)]:
        assert not R.verify_liveness(**(ok | bad)), bad
    stranger = other.issue(DEPLOYMENT, "memory-challenger")
    forged = R.HssSignature(stranger.lower, stranger.certificate, stranger.liveness(ATTACHMENT, "server-3", 7, 2).signature)
    assert not R.verify_liveness(**(ok | dict(sig=forged)))
    swapped = R.HssSignature(live.lower, live.certificate, party.period_root(ATTACHMENT, "server-3", 7, 2, ROOT).signature)
    assert not R.verify_liveness(**(ok | dict(sig=swapped)))
    short = R.HssSignature(live.lower[:-1], live.certificate, live.signature)
    assert not R.verify_liveness(**(ok | dict(sig=short)))


def test_the_identity_signs_one_leaf_per_lower_key_and_rotates(device) -> None:
    """Each issue certifies a new lower tree at the identity's next leaf; the party's newest is the one `party` opens."""
    a = device.issue(DEPLOYMENT, "memory-challenger")
    b = device.issue(DEPLOYMENT, "memory-challenger")
    c = device.issue(OTHER, "memory-challenger")
    assert [a.certificate.q, b.certificate.q, c.certificate.q] == [0, 1, 2]
    assert len({a.lower, b.lower, c.lower}) == 3
    assert device.party(DEPLOYMENT, "memory-challenger").lower == b.lower
    with pytest.raises(R.StateError):
        device.party(DEPLOYMENT, "network-certifier")


def test_seeds_are_private_and_never_made_twice(device) -> None:
    party = device.issue(DEPLOYMENT, "memory-challenger")
    for p in [device.directory / R.IDENTITY_SEED, party.directory / R.LOWER_SEED]:
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert stat.S_IMODE(device.directory.stat().st_mode) == 0o700
    with pytest.raises(FileExistsError):
        R.Identity.init(device.directory, SMALL)


def test_a_lower_key_whose_certificate_is_not_the_identitys_is_refused(device, tmp_path) -> None:
    party = device.issue(DEPLOYMENT, "memory-challenger")
    other = R.Identity.init(tmp_path / "other", SMALL)
    with pytest.raises(R.StateError, match="certificate"):
        R.PartyKey(party.directory, other.public, DEPLOYMENT, "memory-challenger", SMALL)
    with pytest.raises(R.StateError, match="certificate"):
        R.PartyKey(party.directory, device.public, DEPLOYMENT, "network-certifier", SMALL)


def test_a_used_up_lower_key_refuses_to_sign(device) -> None:
    party = device.issue(DEPLOYMENT, "memory-challenger")
    for b in range(2 ** SMALL.H):
        party.liveness(ATTACHMENT, "server-3", 0, b)
    with pytest.raises(R.StateError, match="used"):
        party.liveness(ATTACHMENT, "server-3", 0, 2 ** SMALL.H)


@pytest.mark.parametrize("name", ["", ".hidden", "a/b", "é"])
def test_a_party_name_outside_the_alphabet_is_refused(device, name) -> None:
    with pytest.raises(ValueError):
        device.issue(DEPLOYMENT, name)
