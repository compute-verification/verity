"""The memory challenger's record formats and crypto in Lean (`memory-challenger-difftest`) against `memory_challenger.py`,
`streamed_wipe.py` and `verity.commitments.crypto.lms`: the coins and their tree, the round records' digest, the header and the
signed message, the label digest, LMS signing at a leaf and verification, the setup record's bytes, message and check,
and the streamed wipe's signed message, start and end messages and an egress count's unit, each on sampled inputs that
include what both sides must refuse (`test_verdict.py` and `test_wipe_verdict.py` hold the verdicts)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil

import mc_cases as M
import pytest
import verity_difftest as D
import wipe_cases as W
from hypothesis import strategies as st

from verity.commitments.crypto import lms
from verity.protocols.accounting.pous.device import memory_challenger as mc
from verity.protocols.accounting.pous.verifier import streamed_wipe as sw

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

EXE = "memory-challenger-difftest"
#: What the Python raises where the Lean refuses: a decoder's `Reject`, a refusal of its own (`ValueError`, which
#: `InvalidArtifact` is), an empty `source` (`TypeError`) and a width overflowed (`OverflowError`).
REFUSALS = (D.Reject, ValueError, TypeError, OverflowError)


def port(fn, reference, inputs, **kw) -> D.Port:
    return D.Port(EXE, fn, reference, inputs, package=M.PACKAGE, rejects=REFUSALS, **kw)


server = D.server_fixture(port("coin_digest", lambda v: v, st.none()))


def hexed(s: st.SearchStrategy) -> st.SearchStrategy:
    return s.map(bytes.hex)


#: A hex string the Lean and `P.unhex` refuse, or a value that isn't a string.
BAD_HEX = st.sampled_from(["0", "AB", "zz", "0 1", 1, None, ["00"]])


def malformed(valid: st.SearchStrategy) -> st.SearchStrategy:
    """`valid`'s dictionaries with one field dropped, refused by both sides, or made unreadable."""
    def edit(d: dict, x: int, bad) -> dict:
        k = sorted(d)[x % len(d)]
        return {key: v for key, v in d.items() if key != k} if x % 3 == 0 else d | {k: bad}
    return st.builds(edit, valid, st.integers(0, 99), st.one_of(BAD_HEX, st.integers(-3, -1), st.just("x")))


def mostly(good: st.SearchStrategy, bad: st.SearchStrategy, n: int = 9) -> st.SearchStrategy:
    """`good` `n` times in `n + 1`, else `bad`."""
    return st.integers(0, n).flatmap(lambda i: good if i else bad)


def with_malformed(valid: st.SearchStrategy) -> st.SearchStrategy:
    return mostly(valid, st.one_of(malformed(valid), st.none()), 4)


def WIDE(bits: int) -> st.SearchStrategy:
    """A natural of `bits` bits, now and then one past them."""
    return mostly(st.integers(0, 2 ** bits - 1), st.integers(2 ** bits, 2 ** (bits + 2)))


def coin_digest(v) -> str:
    return mc.coin_digest(M.hex_(v, "session"), M.nat(v, "j"), M.nat(v, "index")).hex()


def test_coin_digest(server):
    inputs = with_malformed(st.fixed_dictionaries({"session": hexed(st.binary(max_size=40)), "j": WIDE(32), "index": WIDE(64)}))
    stats = D.run(port("coin_digest", coin_digest, inputs), examples=300, server=server)
    assert stats.outputs > 100, stats


def coin_tree(v) -> dict:
    t = mc.CoinTree.draw(M.hex_(v, "source"), M.hex_(v, "session"), M.nat(v, "blocks"), M.nat(v, "k"), key=M.hex_(v, "key"),
                         salts=M.hexes(v, "salts"))
    return {"indices": list(t.indices), "root": t.root.hex(), "openings": [t.opening(j).to_json() for j in range(t.k)]}


#: hm96 keys: valid ones, and one with its top bit set or short.
KEYS = mostly(st.sampled_from(M.COIN_KEYS), st.sampled_from([b"\x80" + M.COIN_KEYS[0][1:], M.COIN_KEYS[0][:-1]]))
SALT = mostly(st.binary(min_size=192, max_size=192), st.binary(min_size=191, max_size=191), 20)


@st.composite
def coin_tree_inputs(sample) -> dict:
    """Each refusal now and then: no rounds, a salt too many or too few or short, a bad key, a session not 32 bytes, no
    labels or more than a u64 index holds, no source."""
    k = sample(mostly(st.integers(1, 5), st.just(0)))
    n = sample(mostly(st.just(k), st.sampled_from([k + 1, max(0, k - 1)])))
    session = sample(mostly(st.just(32), st.sampled_from([31, 33])).flatmap(lambda m: st.binary(min_size=m, max_size=m)))
    return {"source": sample(hexed(mostly(st.binary(min_size=1, max_size=6), st.just(b"")))), "session": session.hex(),
            "blocks": sample(mostly(st.integers(1, 60), st.one_of(st.just(0), WIDE(64)))), "k": k, "key": sample(hexed(KEYS)),
            "salts": [s.hex() for s in sample(st.lists(SALT, min_size=n, max_size=n))]}


def test_coin_tree(server):
    stats = D.run(port("coin_tree", coin_tree, with_malformed(coin_tree_inputs())), examples=200, server=server)
    assert stats.outputs > 50, stats


def verify_coin(v) -> bool:
    return mc.verify_coin(M.commitment_of(M._get(v, "commitment")), M.opening_of(M._get(v, "opening")))


@st.composite
def verify_coin_inputs(sample) -> dict:
    s = sample(M.sessions())
    c, o = s.commitment, sample(st.sampled_from(s.openings))
    x = sample(st.integers(0, 2 ** 16))
    o = sample(st.sampled_from([
        o, dataclasses.replace(o, index=o.index + 1), dataclasses.replace(o, index=o.index + c.blocks),
        dataclasses.replace(o, j=o.j + 1), dataclasses.replace(o, j=2 ** 32 + o.j), dataclasses.replace(o, salt=M._flip(o.salt, x)),
        dataclasses.replace(o, salt=o.salt[:-1]), dataclasses.replace(o, path=o.path[:-1]), dataclasses.replace(o, path=o.path + (bytes(64),)),
        dataclasses.replace(o, path=tuple(p[:-1] for p in o.path))]))
    c = sample(st.sampled_from([
        c, dataclasses.replace(c, session=hashlib.sha512(b"other").digest()[:32]), dataclasses.replace(c, k=c.k + 1),
        dataclasses.replace(c, k=2 ** 33), dataclasses.replace(c, blocks=o.index), dataclasses.replace(c, coin_key=c.coin_key[:-1]),
        dataclasses.replace(c, coin_root=M._flip(c.coin_root, x)), dataclasses.replace(c, coin_root=c.coin_root[:-1])]))
    return {"commitment": c.to_json(), "opening": o.to_json()}


def test_verify_coin(server):
    stats = D.run(port("verify_coin", verify_coin, with_malformed(verify_coin_inputs())), examples=200, server=server)
    assert stats.outputs > 100, stats


ROUND = st.fixed_dictionaries({"j": WIDE(32), "index": WIDE(64), "send_ns": WIDE(64), "receipt_ns": WIDE(64),
                               "digest": hexed(st.binary(max_size=70))})
#: Mostly in range: `WIDE` overflows half the time, which would leave few records whose digest both sides compute.
NARROW_ROUND = st.fixed_dictionaries({"j": st.integers(0, 2 ** 32 - 1), "index": st.integers(0, 2 ** 64 - 1),
                                      "send_ns": st.integers(0, 2 ** 64 - 1), "receipt_ns": st.integers(0, 2 ** 64 - 1),
                                      "digest": hexed(st.binary(min_size=64, max_size=64))})
ROUNDS = st.lists(mostly(NARROW_ROUND, st.one_of(ROUND, malformed(NARROW_ROUND)), 15), max_size=6)


def records_digest(v) -> str:
    return mc.records_digest([M.round_of(r) for r in M.list_(v, "rounds")]).hex()


def test_records_digest(server):
    inputs = with_malformed(st.fixed_dictionaries({"rounds": ROUNDS}))
    stats = D.run(port("records_digest", records_digest, inputs), examples=300, server=server)
    assert stats.outputs > 100, stats


COMMITMENT = st.fixed_dictionaries({
    "session": hexed(st.binary(max_size=40)), "coin_root": hexed(st.binary(max_size=70)), "coin_key": hexed(st.binary(max_size=260)),
    "blocks": st.integers(0, 2 ** 70), "k": st.integers(0, 2 ** 70), "limit_ns": st.integers(0, 2 ** 70),
    "block_bytes": st.integers(0, 2 ** 70), "binding": hexed(st.binary(max_size=8))})
#: Any text: the header's JSON writes every character past ASCII as `\uXXXX`, those past the BMP as a surrogate pair.
ENDED = st.one_of(st.sampled_from([mc.COMPLETE, mc.LATE]), st.text(max_size=12))


def header(v) -> str:
    c = M.commitment_of(M._get(v, "commitment"))
    return json.dumps({**c.header(), "ended": M._str(v, "ended"), "rounds": M.nat(v, "rounds")}, sort_keys=True,
                      separators=(",", ":"))


def test_header(server):
    inputs = with_malformed(st.fixed_dictionaries({"commitment": st.one_of(COMMITMENT, malformed(COMMITMENT)), "ended": ENDED,
                                                   "rounds": st.integers(0, 2 ** 70)}))
    stats = D.run(port("header", header, inputs), examples=300, server=server)
    assert stats.outputs > 100, stats


def message(v) -> str:
    return mc.message(M.commitment_of(M._get(v, "commitment")), M._str(v, "ended"),
                      [M.round_of(r) for r in M.list_(v, "rounds")]).hex()


def test_message(server):
    inputs = with_malformed(st.fixed_dictionaries({"commitment": COMMITMENT, "ended": ENDED, "rounds": ROUNDS}))
    stats = D.run(port("message", message, inputs), examples=300, server=server)
    assert stats.outputs > 100, stats


def block_digest(v) -> str:
    return mc.block_digest(M.hex_(v, "block")).hex()


def test_block_digest(server):
    inputs = with_malformed(st.fixed_dictionaries({"block": hexed(st.binary(max_size=300))}))
    stats = D.run(port("block_digest", block_digest, inputs), examples=300, server=server)
    assert stats.outputs > 100, stats


def mc_message(v) -> str:
    return lms.memory_challenger_message(M.hex_(v, "I"), M.hex_(v, "session"), M.hex_(v, "record")).hex()


def test_mc_message(server):
    identifiers = st.one_of(st.binary(min_size=16, max_size=16), st.binary(max_size=20))
    inputs = with_malformed(st.fixed_dictionaries({"I": hexed(identifiers), "session": hexed(st.binary(max_size=40)),
                                                   "record": hexed(st.binary(max_size=70))}))
    stats = D.run(port("mc_message", mc_message, inputs), examples=300, server=server)
    assert stats.outputs > 100, stats


def sign_at(v) -> dict:
    height = M.nat(v, "height")
    if height not in M.ts.HEIGHTS:
        raise D.Reject(f"a test signer's height is in {M.ts.HEIGHTS}, not {height}")
    key = M.test_key(M.hex_(v, "secret"), height)
    sig = lms.sign_at(key, M.nat(v, "q"), lms.memory_challenger_message(key.I, M.hex_(v, "session"), M.hex_(v, "record")))
    return {"public_key": lms.public_key(key).hex(), "signature": sig.encode().hex()}


@pytest.mark.slow
def test_sign_at(server):
    """At heights 1 to 3 (a key's tree is 2^height leaves on both sides), 0 and 11 refused, and leaves past the last."""
    inputs = with_malformed(st.fixed_dictionaries({
        "secret": hexed(st.sampled_from([b"", *M.SECRETS])), "height": mostly(st.integers(1, 3), st.sampled_from([0, 11]), 4),
        "q": mostly(st.integers(0, 1), st.integers(0, 9), 4), "session": hexed(st.binary(max_size=40)), "record": hexed(st.binary(max_size=70))}))
    stats = D.run(port("sign_at", sign_at, inputs), examples=200, server=server)
    assert stats.outputs > 50, stats
    assert M.ts.TestSigner(M.SECRETS[0], height=2).public_key == lms.public_key(M.test_key(M.SECRETS[0], 2))


def lms_verify(v) -> bool:
    scheme = M._str(v, "scheme")
    if scheme not in M.SCHEMES_ALL:
        raise D.Reject(f"no scheme {scheme}")
    return M.SCHEMES_ALL[scheme](M.hex_(v, "public_key"), M.hex_(v, "session"), M.hex_(v, "record"), M.hex_(v, "signature"))


@st.composite
def lms_verify_inputs(sample) -> dict:
    signer = sample(st.one_of(st.builds(M.test_signer, st.sampled_from(M.SECRETS), st.just(2), st.integers(0, 3)),
                            st.just(M.test_signer(M.SECRETS[0], 1, 1)), st.builds(M.deployed_signer)))
    session, record = sample(st.binary(max_size=40)), sample(st.binary(max_size=64))
    sig, pub, x = signer.sign(session, record), signer.public_key, sample(st.integers(0, 2 ** 16))
    sig = sample(mostly(st.just(sig), st.sampled_from([M._flip(sig, x), sig[:-1], b"", (2 ** 32 - 1 - x).to_bytes(4, "big") + sig[4:],
                                (x % 4 + 4).to_bytes(4, "big") + sig[4:], bytes(len(sig))]), 2))
    pub = sample(mostly(st.just(pub), st.sampled_from([pub[:-1], M._flip(pub, x), M.test_signer(M.STRANGER, 2, 0).public_key]), 3))
    session, record = sample(mostly(st.just((session, record)), st.sampled_from([(session + b"x", record), (session, M._flip(record, x))]), 3))
    scheme = sample(mostly(st.just(signer.scheme), st.sampled_from([mc.LMS, M.ts.scheme(1), M.ts.scheme(3), "none"]), 3))
    return {"scheme": scheme, "public_key": pub.hex(), "session": session.hex(), "record": record.hex(), "signature": sig.hex()}


def test_lms_verify(server):
    stats = D.run(port("lms_verify", lms_verify, with_malformed(lms_verify_inputs())), examples=200, server=server)
    assert stats.outputs > 80, stats


#: A setup record's fields, its blocks now and then past a u64, which no encoder takes.
SETUP_RECORD = st.fixed_dictionaries({
    "blocks": WIDE(64), "commitment": hexed(st.binary(max_size=70)), "salt": hexed(st.binary(max_size=30)),
    "root": hexed(st.binary(max_size=70))})
#: A setup record's fields, well formed: the message's inputs, whose record's malformed fields `SETUP_RECORD` samples.
WELL_FORMED_SETUP_RECORD = st.fixed_dictionaries({
    "blocks": st.integers(0, 2 ** 64 - 1), "commitment": hexed(st.binary(max_size=70)),
    "salt": hexed(st.binary(max_size=30)), "root": hexed(st.binary(max_size=70))})


def setup_record_of(v) -> mc.SetupRecord:
    return mc.SetupRecord(M.nat(v, "blocks"), M.hex_(v, "commitment"), M.hex_(v, "salt"), M.hex_(v, "root"))


def setup_record(v) -> str:
    return setup_record_of(v).value().hex()


def test_setup_record(server):
    stats = D.run(port("setup_record", setup_record, with_malformed(SETUP_RECORD)), examples=300, server=server)
    assert stats.outputs > 100, stats
    honest = {"blocks": 983_040, "commitment": "ab" * 64, "salt": "cd" * 24, "root": "ef" * 64}
    value = bytes.fromhex(D.replay(port("setup_record", setup_record, st.none()), honest, server)[1].strip('"'))
    assert value.startswith(mc.SETUP_RECORD_TAG + (983_040).to_bytes(8, "big")) and len(value) == 230


def setup_record_message(v) -> str:
    return mc.setup_record_message(M.hex_(v, "I"), M.hex_(v, "session"), setup_record_of(M._get(v, "record"))).hex()


def test_setup_record_message(server):
    I = mostly(st.binary(min_size=16, max_size=16), st.binary(max_size=20))
    inputs = with_malformed(st.fixed_dictionaries({"I": hexed(I), "session": hexed(st.binary(max_size=40)),
                                                   "record": WELL_FORMED_SETUP_RECORD}))
    stats = D.run(port("setup_record_message", setup_record_message, inputs), examples=300, server=server)
    assert stats.outputs > 100, stats


def setup_record_ok(v) -> bool:
    signed = M._exactly(mc.SignedSetupRecord.from_json, M._get(v, "setup"))
    return mc.setup_record_ok(signed, scheme=M._str(v, "scheme"), public_key=M.hex_(v, "public_key"),
                              session=M.hex_(v, "session"), schemes=M.SCHEMES_ALL)


def _record(s: mc.SignedSetupRecord, **edits) -> mc.SignedSetupRecord:
    return dataclasses.replace(s, record=dataclasses.replace(s.record, **edits))


@st.composite
def setup_record_ok_inputs(draw) -> dict:
    """A setup record a test signer or a deployed-parameter key signed, then now and then its signature, a field, its
    scheme or key changed, its JSON not a record's (a key extra or missing, hex in capitals), or checked against another
    scheme, key or session."""
    signer = draw(st.one_of(st.builds(M.test_signer, st.sampled_from(M.SECRETS), st.just(2), st.integers(0, 3)),
                            st.builds(M.deployed_signer)))
    session = draw(st.binary(min_size=32, max_size=32))
    signed = mc.sign_setup_record(signer, session, blocks=draw(WIDE(20)), commitment=draw(st.binary(min_size=64, max_size=64)),
                                  salt=draw(st.binary(min_size=24, max_size=24)), root=draw(st.sampled_from(M.SETUP_ROOTS)))
    x, r = draw(st.integers(0, 2 ** 16)), signed.record
    signed = draw(mostly(st.just(signed), st.sampled_from([
        dataclasses.replace(signed, signature=M._flip(signed.signature, x)),
        _record(signed, blocks=r.blocks + 1), _record(signed, blocks=2 ** 64 + r.blocks),
        _record(signed, commitment=M._flip(r.commitment, x)), _record(signed, salt=M._flip(r.salt, x)),
        _record(signed, root=M.SETUP_ROOTS[0] if r.root != M.SETUP_ROOTS[0] else M.SETUP_ROOTS[1]),
        dataclasses.replace(signed, scheme="none"),
        dataclasses.replace(signed, public_key=M.test_signer(M.STRANGER, 2, 0).public_key)]), 2))
    j = signed.to_json()
    j = draw(mostly(st.just(j), st.sampled_from([
        j | {"passed": 1}, j | {"extra": 1}, {k: v for k, v in j.items() if k != "salt"},
        j | {"root": j["root"].upper()}]), 4))
    scheme = draw(mostly(st.just(signer.scheme), st.sampled_from([mc.LMS, M.ts.scheme(1), M.ts.scheme(2), "none"]), 4))
    public_key = draw(mostly(st.just(signer.public_key), st.just(M.test_signer(M.STRANGER, 2, 0).public_key), 4))
    session = draw(mostly(st.just(session), st.just(session[:-1] + bytes([session[-1] ^ 1])), 4))
    return {"public_key": public_key.hex(), "scheme": scheme, "session": session.hex(), "setup": j}


def test_setup_record_ok(server):
    stats = D.run(port("setup_record_ok", setup_record_ok, with_malformed(setup_record_ok_inputs())), examples=200,
                  server=server)
    assert stats.outputs > 80, stats


def test_a_setup_record_says_no_outcome(server):
    """The record holds what the device was given and nothing about setup's check: a signed record verifies, and its
    JSON with a `passed` key beside its fields is no record, refused by both decoders."""
    signer, session = M.test_signer(M.SECRETS[0], 2, 1), bytes(range(32))
    ok = port("setup_record_ok", setup_record_ok, st.none())
    signed = mc.sign_setup_record(signer, session, blocks=983_040, commitment=bytes(64), salt=bytes(24),
                                  root=M.SETUP_ROOTS[0])
    case = {"public_key": signer.public_key.hex(), "scheme": signer.scheme, "session": session.hex(),
            "setup": signed.to_json()}
    assert D.replay(ok, case, server) == ("output", "true")
    for passed in (0, 1):
        assert D.replay(ok, case | {"setup": signed.to_json() | {"passed": passed}}, server)[0] == "reject"


WIPE_PARAMS = st.builds(
    lambda labels, k, b, limit, names, cuts: sw.WipeParams(labels, k, 1 + b % k, limit, W.tiers_of(labels, cuts, names)),
    st.integers(1, 2 ** 40), st.integers(1, 2 ** 32 - 1), st.integers(0, 2 ** 32), st.integers(1, 2 ** 64),
    st.permutations(W.TIER_NAMES), st.lists(st.integers(1, 2 ** 40), max_size=2))
WIPE_COMMITMENT = st.builds(lambda server, session, p: sw.WipeCommitment(server, session, p).to_json(),
                            st.binary(min_size=1, max_size=12), st.binary(max_size=40), WIPE_PARAMS)
#: An integer of either sign, mostly within a record field's width.
SIGNED = mostly(st.integers(0, 2 ** 32 - 1), st.integers(-2 ** 70, 2 ** 70), 4)
WIPE_BATCH = st.fixed_dictionaries({"b": SIGNED, "first": SIGNED, "count": SIGNED, "send_ns": SIGNED, "receipt_ns": SIGNED,
                                    "wrong": SIGNED, "digest": hexed(st.binary(max_size=70))})
WIPE_BATCHES = st.lists(mostly(WIPE_BATCH, malformed(WIPE_BATCH), 15), max_size=5)
#: A start mark as the header writes it: none or a digest, now and then bytes of another length.
WIPE_MARK = hexed(mostly(st.one_of(st.just(b""), st.binary(min_size=sw.MARK_BYTES, max_size=sw.MARK_BYTES)),
                         st.binary(max_size=70), 5))
#: A signed session's fields; `MemoryChallenger.StreamedWipe.Signed.ofJson` refuses a start mark of another length.
WIPE_SIGNED = st.fixed_dictionaries({
    "commitment": mostly(WIPE_COMMITMENT, malformed(WIPE_COMMITMENT), 9), "ended": ENDED, "start_mark": WIPE_MARK,
    "fill_start_ns": SIGNED, "fill_end_ns": SIGNED, "batches": WIPE_BATCHES,
    "scheme": st.sampled_from([mc.LMS, M.ts.scheme(1), "none", ""]), "public_key": hexed(st.binary(max_size=70)),
    "signature": hexed(st.binary(max_size=140))})
WIPE_MESSAGE_KEYS = ("commitment", "ended", "start_mark", "fill_start_ns", "fill_end_ns", "batches")


def wipe_message(v) -> str:
    return sw.message(W.commitment_of(M._get(v, "commitment")), M._str(v, "ended"), M.hex_(v, "start_mark"),
                      W.int_(v, "fill_start_ns"), W.int_(v, "fill_end_ns"),
                      [W.batch_of(r) for r in M.list_(v, "batches")]).hex()


def wipe_start_message(v) -> str:
    return sw.start_message(W.commitment_of(M._get(v, "commitment"))).hex()


def wipe_end_message(v) -> str:
    return sw.end_message(W.signed_of(M._get(v, "signed"))).hex()


def test_wipe_message(server):
    """The signed message over sampled commitments (tier names past ASCII among them), batch records of either sign and
    width, start marks of any length, and fill times of either sign; commitments `WipeParams` refuses and
    records past a field's width refused by both."""
    commitment = mostly(WIPE_COMMITMENT, malformed(WIPE_COMMITMENT), 4)
    inputs = with_malformed(st.fixed_dictionaries({"commitment": commitment, "ended": ENDED, "start_mark": WIPE_MARK,
                                                   "fill_start_ns": SIGNED, "fill_end_ns": SIGNED,
                                                   "batches": WIPE_BATCHES}))
    stats = D.run(port("wipe_message", wipe_message, inputs), examples=400, server=server)
    assert stats.outputs > 100 and stats.rejects > 30, stats


def test_wipe_start_message(server):
    """The start message over sampled commitments (tier names past ASCII among them); commitments `WipeParams` refuses
    refused by both."""
    commitment = mostly(WIPE_COMMITMENT, malformed(WIPE_COMMITMENT), 3)
    inputs = with_malformed(st.fixed_dictionaries({"commitment": commitment}))
    stats = D.run(port("wipe_start_message", wipe_start_message, inputs), examples=300, server=server)
    assert stats.outputs > 100 and stats.rejects > 30, stats


def test_wipe_end_message(server):
    """The end message over sampled signed sessions, their signatures of any length; a start mark neither empty nor 64
    bytes, a commitment `WipeParams` refuses and a record past a field's width refused by both."""
    inputs = with_malformed(st.fixed_dictionaries({"signed": mostly(WIPE_SIGNED, malformed(WIPE_SIGNED), 5)}))
    stats = D.run(port("wipe_end_message", wipe_end_message, inputs), examples=300, server=server)
    assert stats.outputs > 80 and stats.rejects > 30, stats


def test_wipe_messages_of_sessions(server):
    """On sessions `WipeMemoryChallenger` ran, with a start mark and with none, the Lean's signed message, start message and end
    message are the Python's own, and the signed header carries the certificate's SHA-512."""
    params = sw.WipeParams(8, 3, 1, 1_000, (("gpu", 3 * sw.LABEL_BYTES), ("host-dram", 5 * sw.LABEL_BYTES)))
    seen = set()
    for certificate in (None, b"a start certificate", b"a start certificatf"):
        s = W.run(params, M.test_signer(M.SECRETS[0], 1, 1), certificate=certificate)
        j = s.signed.to_json()
        assert j["start_mark"] == ("" if certificate is None else hashlib.sha512(certificate).hexdigest())
        for fn, reference, value, python in (
                ("wipe_message", wipe_message, {k: j[k] for k in WIPE_MESSAGE_KEYS}, s.signed.message()),
                ("wipe_start_message", wipe_start_message, {"commitment": j["commitment"]}, sw.start_message(s.commitment)),
                ("wipe_end_message", wipe_end_message, {"signed": j}, sw.end_message(s.signed))):
            assert D.replay(port(fn, reference, st.none()), value, server) == ("output", D.canonical(python.hex())), fn
        seen.add((s.signed.message(), sw.start_message(s.commitment), sw.end_message(s.signed)))
    assert len({m for m, _, _ in seen}) == len({e for _, _, e in seen}) == 3 and len({a for _, a, _ in seen}) == 1


def test_wipe_egress_units(server):
    """Whole MiB rounded up, negative counts as Python's floor division rounds them, and the 15-bit saturation."""
    edges = [0, 1, -1, W.U - 1, W.U, W.U + 1, -W.U, -W.U - 1, sw.EGRESS_MAX_UNITS * W.U, sw.EGRESS_MAX_UNITS * W.U + 1]
    bits = st.one_of(st.sampled_from(edges), st.integers(-2 ** 70, 2 ** 70), st.integers(-4 * W.U, 4 * W.U))
    inputs = with_malformed(st.fixed_dictionaries({"egress_bits": bits}))
    stats = D.run(port("wipe_egress_units", lambda v: sw.egress_units(W.int_(v, "egress_bits")), inputs), examples=300,
                  server=server)
    assert stats.outputs > 150 and stats.rejects > 30, stats
