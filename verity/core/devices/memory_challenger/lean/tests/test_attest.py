"""A device's HSS role digests and checks in Lean (`Lms.Hss`, `MemoryChallenger.Attest`, served by
`memory-challenger-difftest`) against `verity.ml.crypto.lms` and `verity.core.attest`: the three role digests, two-level
verification, the period tree's root, the period check a period's proof runs and the liveness check, each on drawn inputs
around a real device's signatures (a small identity, H = 2) that include what both sides must refuse."""

from __future__ import annotations

import hashlib
import shutil

import mc_cases as M
import pytest
import verity_difftest as D
from hypothesis import strategies as st

from verity.core import attest
from verity.ml.crypto import lms

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

EXE = "memory-challenger-difftest"
REFUSALS = (D.Reject, ValueError)
P2 = lms.Params(64, 1, 2)
DEPLOYMENT = hashlib.sha512(b"memory-challenger-lean/deployment").digest()
ATTACHMENT = bytes(range(32))
PARTY, LINK, PERIOD = "memory-challenger", "server-3", 7
RECORDS = [hashlib.sha512(b"memory-challenger-lean/record %d" % i).digest()[: 8 + 9 * i] for i in range(5)]


def port(fn, reference, inputs, **kw) -> D.Port:
    return D.Port(EXE, fn, reference, inputs, package=M.PACKAGE, rejects=REFUSALS, **kw)


server = D.server_fixture(port("attest_root", lambda v: v, st.none()))


@pytest.fixture(scope="module")
def device(tmp_path_factory) -> dict:
    """A device's identity, its party's lower key, the party's liveness record of block 2 and its signed root over
    `RECORDS`, and a stranger's identity and party signing the same."""
    me = lms.Identity.init(tmp_path_factory.mktemp("device"), P2)
    stranger = lms.Identity.init(tmp_path_factory.mktemp("stranger"), P2)
    party, theirs = me.issue(DEPLOYMENT, PARTY), stranger.issue(DEPLOYMENT, PARTY)
    root = attest.root(RECORDS)
    return {"identity": me.public, "stranger": stranger.public, "lower": party.lower,
            "live": party.liveness(ATTACHMENT, LINK, PERIOD, 2).encode(),
            "signed": party.period_root(ATTACHMENT, LINK, PERIOD, len(RECORDS), root).encode(),
            "theirs": theirs.period_root(ATTACHMENT, LINK, PERIOD, len(RECORDS), root).encode(), "root": root}


def hexed(s: st.SearchStrategy) -> st.SearchStrategy:
    return s.map(bytes.hex)


def mostly(good: st.SearchStrategy, bad: st.SearchStrategy, n: int = 4) -> st.SearchStrategy:
    return st.integers(0, n).flatmap(lambda i: good if i else bad)


TEXT = st.text(st.characters(codec="utf-8"), max_size=8)
PARTIES = mostly(st.just(PARTY), st.one_of(st.just("network-certifier"), TEXT))
LINKS = mostly(st.just(LINK), st.one_of(st.just("server-4"), TEXT))
U64 = mostly(st.integers(0, 9), st.sampled_from([2 ** 64 - 1, 2 ** 64, 2 ** 70]))
DEPLOYMENTS = mostly(st.just(DEPLOYMENT), st.sampled_from([DEPLOYMENT[::-1], DEPLOYMENT[:-1], b""])).map(bytes.hex)
IDENT = mostly(st.binary(min_size=16, max_size=16), st.binary(max_size=20))


_flip = M._flip


def _params(v) -> lms.Params:
    return lms.Params(64, M.nat(v, "W"), M.nat(v, "H"))


def test_role_digests(server):
    """Each role's digest over drawn fields, malformed ones (an identifier not 16 bytes, a deployment hash not 64 bytes,
    an index past a u64) refused by both."""
    base = {"I": hexed(IDENT), "deployment": DEPLOYMENTS, "party": PARTIES, "attachment": hexed(st.binary(max_size=40)),
            "link": LINKS, "period": U64}
    cases = [
        ("party_key_message", lambda v: lms.party_key_message(M.hex_(v, "I"), M.hex_(v, "deployment"), M._str(v, "party"),
                                                              M.hex_(v, "lower")).hex(),
         st.fixed_dictionaries({k: base[k] for k in ("I", "deployment", "party")} | {"lower": hexed(st.binary(max_size=90))})),
        ("period_root_message", lambda v: lms.period_root_message(
            M.hex_(v, "I"), M.hex_(v, "deployment"), M._str(v, "party"), M.hex_(v, "attachment"), M._str(v, "link"),
            M.nat(v, "period"), M.nat(v, "count"), M.hex_(v, "root")).hex(),
         st.fixed_dictionaries(base | {"count": U64, "root": hexed(st.binary(max_size=70))})),
        ("liveness_message", lambda v: lms.liveness_message(
            M.hex_(v, "I"), M.hex_(v, "deployment"), M._str(v, "party"), M.hex_(v, "attachment"), M._str(v, "link"),
            M.nat(v, "period"), M.nat(v, "block")).hex(),
         st.fixed_dictionaries(base | {"block": U64}))]
    for fn, reference, inputs in cases:
        stats = D.run(port(fn, reference, inputs), examples=200, server=server)
        assert stats.outputs > 40 and stats.rejects > 10, (fn, stats)


def test_attest_root(server):
    inputs = st.fixed_dictionaries({"records": st.lists(hexed(st.binary(max_size=80)), max_size=9)})
    stats = D.run(port("attest_root", lambda v: attest.root(M.hexes(v, "records")).hex(), inputs, need_rejects=False),
                  examples=200, server=server)
    assert stats.outputs == 200, stats


def hss_verify(v) -> bool:
    p = _params(v)
    return lms.hss_verify(p, M.hex_(v, "identity"), M.hex_(v, "deployment"), M._str(v, "party"),
                          lms.HssSignature.decode(p, M.hex_(v, "signature")), M.hex_(v, "message"))


def test_hss_verify(server, device):
    """The liveness record and the signed root under their messages, the identity, the deployment and the party, each
    now and then another one's, and the signature flipped, cut, or another identity's."""
    live, signed = device["live"], device["signed"]
    I = device["lower"][:16]
    messages = {live: lms.liveness_message(I, DEPLOYMENT, PARTY, ATTACHMENT, LINK, PERIOD, 2),
                signed: lms.period_root_message(I, DEPLOYMENT, PARTY, ATTACHMENT, LINK, PERIOD, len(RECORDS),
                                                device["root"])}

    @st.composite
    def inputs(draw) -> dict:
        sig = draw(st.sampled_from([live, signed]))
        m, x = messages[sig], draw(st.integers(0, 2 ** 20))
        sig = draw(mostly(st.just(sig), st.sampled_from([_flip(sig, x), sig[:-1], device["theirs"]])))
        return {"W": 1, "H": draw(mostly(st.just(2), st.sampled_from([1, 3, 31]))),
                "identity": draw(mostly(st.just(device["identity"]), st.sampled_from([device["stranger"],
                                                                                      device["identity"][:-1]]))).hex(),
                "deployment": draw(DEPLOYMENTS), "party": draw(PARTIES), "signature": sig.hex(),
                "message": draw(mostly(st.just(m), st.sampled_from([_flip(m, x), m[:-1]]))).hex()}

    stats = D.run(port("hss_verify", hss_verify, inputs()), examples=150, server=server)
    assert stats.outputs > 60 and stats.rejects > 5, stats


def attest_check(v) -> bool:
    p = _params(v)
    sig = lms.HssSignature.decode(p, M.hex_(v, "signature"))
    reads = [(M.nat(r, "position"), M.hex_(r, "record"), M.hexes(r, "path")) for r in M.list_(v, "reads")]
    root, count = M.hex_(v, "root"), M.nat(v, "count")
    m = lms.period_root_message(sig.lower[:16], M.hex_(v, "deployment"), M._str(v, "party"), M.hex_(v, "attachment"),
                                M._str(v, "link"), M.nat(v, "period"), count, root)
    if not lms.hss_verify(p, M.hex_(v, "identity"), M.hex_(v, "deployment"), M._str(v, "party"), sig, m):
        return False
    return all(attest.verify_opening(root, count, pos, rec, path) for pos, rec, path in reads)


def test_attest_check(server, device):
    """The device's signed root and openings of its records, each field now and then another (a record, a position, a
    path, the root, the count, the period, the link, the attachment, the party, the deployment, the identity) or the
    signature flipped, cut or a stranger's."""
    root = device["root"]

    @st.composite
    def inputs(draw) -> dict:
        x = draw(st.integers(0, 2 ** 20))
        positions = draw(st.lists(st.integers(0, len(RECORDS) - 1), max_size=3))
        reads = []
        for pos in positions:
            rec, path = RECORDS[pos], list(attest.path(RECORDS, pos))
            pos, rec, path = draw(mostly(st.just((pos, rec, path)), st.sampled_from([
                (pos, _flip(rec, x), path), ((pos + 1) % len(RECORDS), rec, path), (pos, rec, path[:-1]),
                (pos, rec, path[::-1]), (pos + len(RECORDS), rec, path)]), 6))
            reads.append({"position": pos, "record": rec.hex(), "path": [s.hex() for s in path]})
        sig = draw(mostly(st.just(device["signed"]), st.sampled_from([_flip(device["signed"], x), device["signed"][:-1],
                                                                       device["theirs"], device["live"]]), 6))
        return {"W": 1, "H": 2, "identity": draw(mostly(st.just(device["identity"]), st.sampled_from(
                    [device["stranger"], device["identity"][:-1]]), 8)).hex(),
                "deployment": draw(mostly(st.just(DEPLOYMENT.hex()), DEPLOYMENTS, 8)),
                "party": draw(mostly(st.just(PARTY), PARTIES, 8)),
                "attachment": draw(mostly(st.just(ATTACHMENT), st.just(ATTACHMENT[::-1]), 8)).hex(),
                "link": draw(mostly(st.just(LINK), LINKS, 8)), "period": draw(mostly(st.just(PERIOD), U64, 8)),
                "count": draw(mostly(st.just(len(RECORDS)), st.sampled_from([len(RECORDS) + 1, 4, 2 ** 64]), 8)),
                "root": draw(mostly(st.just(root), st.sampled_from([_flip(root, x), root[:-1]]), 8)).hex(),
                "signature": sig.hex(), "reads": reads}

    stats = D.run(port("attest_check", attest_check, inputs()), examples=200, server=server)
    assert stats.outputs > 80 and stats.rejects > 5, stats


def attest_liveness(v) -> bool:
    p = _params(v)
    return lms.verify_liveness(p, M.hex_(v, "identity"), M.hex_(v, "deployment"), M._str(v, "party"),
                               M.hex_(v, "attachment"), M._str(v, "link"), M.nat(v, "period"), M.nat(v, "block"),
                               lms.HssSignature.decode(p, M.hex_(v, "signature")))


def test_attest_liveness(server, device):
    """The party's liveness record of block 2, each field now and then another, and the signature flipped, cut, or the
    party's signed root."""

    @st.composite
    def inputs(draw) -> dict:
        x = draw(st.integers(0, 2 ** 20))
        sig = draw(mostly(st.just(device["live"]), st.sampled_from([_flip(device["live"], x), device["live"][:-1],
                                                                     device["signed"]])))
        return {"W": 1, "H": 2, "identity": draw(mostly(st.just(device["identity"]), st.just(device["stranger"]))).hex(),
                "deployment": draw(DEPLOYMENTS), "party": draw(PARTIES),
                "attachment": draw(mostly(st.just(ATTACHMENT), st.just(ATTACHMENT[:-1]))).hex(), "link": draw(LINKS),
                "period": draw(mostly(st.just(PERIOD), U64)), "block": draw(mostly(st.just(2), U64)),
                "signature": sig.hex()}

    stats = D.run(port("attest_liveness", attest_liveness, inputs()), examples=150, server=server)
    assert stats.outputs > 60 and stats.rejects > 3, stats
