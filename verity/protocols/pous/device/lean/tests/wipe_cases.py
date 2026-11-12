"""The Python side of the streamed wipe's difftests: `streamed_wipe.py` (the reference, unchanged) behind the Lean port's
JSON formats (`MemoryChallenger.WipeJudge`), and the sessions the tests sample.

The decoders mirror the Lean's: byte strings lowercase hex, the parameters naturals that `WipeParams` accepts, a batch
record's fields and the fill's times integers of either sign, a start mark empty or 64 bytes, and the record
exactly the canonical JSON of what it decodes to. Past decoding every reference is the Python's own function on the
decoded objects.

Sessions are run by `streamed_wipe.Challenger` itself against `streamed_wipe.Prover`, on an injected clock, with a mark
source that returns the start certificate a test gives or with none, and signed by a test signer's key at a leaf the test
picks (`mc_cases.AtLeaf`) or by a deployed-parameter key (`mc_cases.deployed_signer`). No real key is read."""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Any

from hypothesis import strategies as st

import mc_cases as M
import verity_difftest as D
from verity.protocols.pous.device import memory_challenger as mc
from verity.protocols.pous.verifier import streamed_wipe as sw

#: Every scheme the port verifies, by id (`MemoryChallenger.KNOWN_SCHEMES`).
KNOWN = {**mc.SCHEMES, **M.ts.SCHEMES}
SECRET = b"streamed-wipe-lean/auditor secret"
SERVERS = [b"server-7", b"s", "é".encode()]
#: Tier names, ASCII and not, which the header writes as `json.dumps` does.
TIER_NAMES = ["gpu", "host-dram", "é ü", 'a"b\\', "\t\x00", "𝄞"]
U = sw.EGRESS_UNIT_BITS


# --- decoding, as `MemoryChallenger.StreamedWipe` decodes ----------------------------------------------------------------

def int_(j: Any, k: str) -> int:
    """`J.asInt`: an integer of either sign, not a boolean."""
    v = M._get(j, k)
    if not isinstance(v, int) or isinstance(v, bool):
        raise D.Reject(f"{k}: not an integer")
    return v


def commitment_of(j: Any) -> sw.WipeCommitment:
    tiers = []
    for t in M.list_(j, "tiers"):
        if not (isinstance(t, list) and len(t) == 2 and isinstance(t[0], str)):
            raise D.Reject("tiers: a tier is [name, bytes]")
        if not isinstance(t[1], int) or isinstance(t[1], bool) or t[1] < 0:
            raise D.Reject("tiers: not a natural number")
        tiers.append((t[0], t[1]))
    labels, k, batch, limit = M.nat(j, "labels"), M.nat(j, "k"), M.nat(j, "batch"), M.nat(j, "limit_ns")
    try:
        params = sw.WipeParams(labels, k, batch, limit, tuple(tiers))
    except ValueError as e:
        raise D.Reject(str(e)) from None
    return sw.WipeCommitment(M.hex_(j, "server"), M.hex_(j, "session"), params)


def batch_of(j: Any) -> sw.BatchRecord:
    return sw.BatchRecord(int_(j, "b"), int_(j, "first"), int_(j, "count"), int_(j, "send_ns"), int_(j, "receipt_ns"),
                          int_(j, "wrong"), M.hex_(j, "digest"))


def mark_of(j: Any) -> bytes:
    """`Signed.ofJson`'s start mark: lowercase hex of no bytes or of `MARK_BYTES`."""
    mark = M.hex_(j, "start_mark")
    if len(mark) not in (0, sw.MARK_BYTES):
        raise D.Reject("start_mark: a start mark is 128 lowercase hex digits, or empty")
    return mark


def signed_of(j: Any) -> sw.SignedWipe:
    return sw.SignedWipe(commitment_of(M._get(j, "commitment")), M._str(j, "ended"), mark_of(j),
                         int_(j, "fill_start_ns"), int_(j, "fill_end_ns"),
                         tuple(batch_of(r) for r in M.list_(j, "batches")),
                         M._str(j, "scheme"), M.hex_(j, "public_key"), M.hex_(j, "signature"))


# --- sessions ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Session:
    """A signed wipe, the commitment published when it opened, the trusted keys, and the signer (to re-sign an edit)."""

    signed: sw.SignedWipe
    commitment: sw.WipeCommitment
    trusted: tuple[bytes, ...]
    signer: M.AtLeaf

    def resign(self, **edits) -> Session:
        s = dataclasses.replace(self.signed, **edits)
        try:
            sig = self.signer.sign(s.commitment.session, s.message())
        except (ValueError, OverflowError):  # an edit past the record's widths has no message; the old signature stays
            sig = s.signature
        return dataclasses.replace(self, signed=dataclasses.replace(s, signature=sig))


def tiers_of(labels: int, cuts: list[int], names: list[str], outside: int = 0) -> tuple[tuple[str, int], ...]:
    """The fill's `labels · 64` bytes cut at `cuts` (labels) into tiers named `names`, each at least one label, and
    `outside` bytes the fill doesn't reach (`R`) in the last."""
    edges = [0, *sorted({c for c in cuts if 0 < c < labels}), labels]
    tiers = [(names[i], sw.LABEL_BYTES * (b - a)) for i, (a, b) in enumerate(zip(edges, edges[1:]))]
    tiers[-1] = (tiers[-1][0], tiers[-1][1] + outside)
    return tuple(tiers)


def run(params: sw.WipeParams, signer: M.AtLeaf, *, server: bytes = SERVERS[0], session_id: bytes = bytes(32),
        start: int = 1_000, fill_ns: int = 7, spends=lambda b: 1, tampered=frozenset(),
        coins: bytes = b"coins", certificate: bytes | None = None) -> Session:
    """One session of `Challenger` against the honest `Prover`: batch `b`'s answer takes `spends(b)` ns, and the batches in
    `tampered` have their first answer's last bit flipped. With `certificate`, the challenger's mark source returns it,
    and the session signs its SHA-512 as its start mark; with none the mark is empty."""
    clock = M.Clock(start)
    c = sw.Challenger(SECRET, server, session_id, params, signer, log=M.ts.session_log(), clock=clock, coin_secret=coins,
                      mark_source=None if certificate is None else lambda m: certificate)
    prover = sw.Prover(params)
    for first, data in c.fill(max(1, params.labels // 3)):
        prover.receive(first, data)
        clock.now += fill_ns
    b = 0

    def answer(idx):
        nonlocal b
        clock.now += spends(b)
        out = prover.answer(idx)
        if b in tampered:
            out = out[:63] + bytes([out[63] ^ 1]) + out[64:]
        b += 1
        return out

    c.run(answer)
    return Session(c.close(), c.commitment, (signer.public_key,), signer)


#: The evidence's egress, in MiB: none given, a count in range, the range's ends, and now and then one outside it.
EGRESS = st.one_of(st.none(), st.integers(0, 4), st.integers(0, sw.EGRESS_MAX_UNITS),
                   st.sampled_from([None, 0, 1, sw.EGRESS_MAX_UNITS - 1, sw.EGRESS_MAX_UNITS,
                                    sw.EGRESS_MAX_UNITS + 1, -1, 10 ** 15]))
#: A sampled session's start certificate, by the number its coins are sampled under: none (no mark source, an empty
#: mark), or the bytes its mark source returns.
CERTIFICATES = [None, b"a start certificate", b"\0", bytes(range(256)) * 3]


@st.composite
def sessions(draw, signer: M.AtLeaf | None = None) -> Session:
    """An honest run of `Challenger` at a sampled batch size, every answer on time and right (some at the deadline itself),
    or one batch late or one answer wrong, with a start mark or none; signed by `signer` or a sampled test signer at a
    sampled leaf."""
    if signer is None:
        height = draw(st.integers(1, 2))
        signer = M.test_signer(draw(st.sampled_from(M.SECRETS)), height, draw(st.integers(0, 2 ** height - 1)))
    labels, k = draw(st.integers(1, 48)), draw(st.integers(1, 6))
    batch = draw(st.one_of(st.just(1), st.integers(1, k)))
    limit = draw(st.integers(1, 10 ** 6))
    names = draw(st.permutations(TIER_NAMES))
    cuts = draw(st.lists(st.integers(1, max(1, labels - 1)), max_size=2))
    outside = draw(st.sampled_from([0, 0, 1, sw.LABEL_BYTES, 4096]))
    params = sw.WipeParams(labels, k, batch, limit, tiers_of(labels, cuts, names, outside))
    batches = params.batches
    spend = draw(st.lists(st.one_of(st.integers(0, limit), st.just(limit)), min_size=batches, max_size=batches))
    fault, at = draw(st.sampled_from(["none", "none", "late", "wrong"])), draw(st.integers(0, batches - 1))
    if fault == "late":
        spend[at] = limit + 1 + draw(st.integers(0, 3))
    return run(params, signer, server=draw(st.sampled_from(SERVERS)),
               session_id=hashlib.sha512(b"wipe session %d" % draw(st.integers(0, 3))).digest()[:32],
               start=draw(st.integers(0, 2 ** 40)), fill_ns=draw(st.integers(0, 10 ** 6)), spends=lambda b: spend[b],
               tampered=frozenset({at} if fault == "wrong" else ()),
               coins=b"coins %d" % (n := draw(st.integers(0, 3))), certificate=CERTIFICATES[n])


def _batch(s: Session, x: int, **edits) -> Session:
    """Batch `x mod n`'s fields edited by `edits[field](record)`, re-signed."""
    batches = list(s.signed.batches)
    r = x % len(batches)
    batches[r] = dataclasses.replace(batches[r], **{k: v(batches[r]) for k, v in edits.items()})
    return s.resign(batches=tuple(batches))


def _unsigned(s: Session, **edits) -> Session:
    return dataclasses.replace(s, signed=dataclasses.replace(s.signed, **edits))


def _signature(s: Session, sig: bytes) -> Session:
    return _unsigned(s, signature=sig)


def _stranger(s: Session, x: int) -> Session:
    other = M.test_signer(M.STRANGER, 1, x % 2)
    return _unsigned(s, public_key=other.public_key, signature=other.sign(s.signed.commitment.session, s.signed.message()))


def _replayed(s: Session, x: int) -> Session:
    """The signer's signature at the same leaf over another session's record: a replayed leaf."""
    b = s.signed.batches
    other = dataclasses.replace(s.signed, batches=b[:-1] if x % 2 else b + b[-1:])
    return _signature(s, s.signer.sign(s.signed.commitment.session, other.message()))


def _params(s: Session, **edits) -> sw.WipeCommitment:
    return dataclasses.replace(s.commitment, params=dataclasses.replace(s.commitment.params, **edits))


def _more_labels(s: Session, x: int) -> sw.WipeCommitment:
    p = s.commitment.params
    extra = 1 + x % 3
    tiers = p.tiers[:-1] + ((p.tiers[-1][0], p.tiers[-1][1] + sw.LABEL_BYTES * extra),)
    return _params(s, labels=p.labels + extra, tiers=tiers)


def _first_before_fill(s: Session, x: int) -> Session:
    return s.resign(fill_end_ns=s.signed.batches[0].send_ns + 1 + x % 5)


def _scheme(s: Session, x: int) -> Session:
    return _unsigned(s, scheme=M.PRINTABLE_SCHEMES[x % len(M.PRINTABLE_SCHEMES)])


def _other_mark(s: Session, x: int) -> bytes:
    """A start mark that is not the session's: another certificate's, the session's with one bit flipped, or none."""
    mark = s.signed.start_mark
    others = [hashlib.sha512(b"another certificate %d" % (x % 5)).digest(), M._flip(mark, x) if mark else bytes(64), b""]
    return next(m for m in others[x % 3:] + others if m != mark)


#: One adversarial edit of a session, `(session, x) -> session`, `x` a sampled natural: a signature or key that isn't the
#: session's, a commitment that isn't the published one, a wrong label, a late or early answer, a wrong batch number,
#: first index or count, a batch dropped or added, a session that didn't complete, the fill not before the first batch,
#: a field past its width, and the start mark edited under the old signature or signed anew (a mark
#: the judge reads as any other).
MUTATIONS = [
    lambda s, x: _signature(s, M._flip(s.signed.signature, x)),
    lambda s, x: _signature(s, s.signed.signature[:-1 - x % 3]),
    lambda s, x: _signature(s, b""),
    lambda s, x: dataclasses.replace(s, trusted=()),
    lambda s, x: dataclasses.replace(s, trusted=(M._flip(s.signed.public_key, x),)),
    _stranger,
    lambda s, x: dataclasses.replace(_stranger(s, x), trusted=s.trusted + (M.test_signer(M.STRANGER, 1, x % 2).public_key,)),
    _scheme,
    _replayed,
    lambda s, x: _signature(s, s.signer.sign(b"another session", s.signed.message())),
    lambda s, x: dataclasses.replace(s, commitment=_params(s, limit_ns=s.commitment.params.limit_ns + 1)),
    lambda s, x: dataclasses.replace(s, commitment=dataclasses.replace(s.commitment, server=s.commitment.server + b"x")),
    lambda s, x: s.resign(commitment=_more_labels(s, x)),
    lambda s, x: s.resign(commitment=dataclasses.replace(s.commitment, session=M._flip(s.commitment.session, x))),
    lambda s, x: dataclasses.replace(s.resign(commitment=_params(s, k=s.commitment.params.k + 1)),
                                     commitment=_params(s, k=s.commitment.params.k + 1)),
    lambda s, x: _batch(s, x, wrong=lambda r: r.wrong + 1 + x % 2),
    lambda s, x: _batch(s, x, wrong=lambda r: -1),
    lambda s, x: _batch(s, x, digest=lambda r: M._flip(r.digest, x)),
    lambda s, x: _batch(s, x, receipt_ns=lambda r: r.send_ns + s.commitment.params.limit_ns + 1 + x),
    lambda s, x: _batch(s, x, receipt_ns=lambda r: r.send_ns - 1 - x % 3),
    lambda s, x: _batch(s, x, receipt_ns=lambda r: r.send_ns + s.commitment.params.limit_ns),
    lambda s, x: _batch(s, x, b=lambda r: r.b + 1),
    lambda s, x: _batch(s, x, b=lambda r: 2 ** 32 + r.b),
    lambda s, x: _batch(s, x, first=lambda r: r.first + 1),
    lambda s, x: _batch(s, x, count=lambda r: r.count + 1),
    lambda s, x: _batch(s, x, count=lambda r: r.count - 1),
    lambda s, x: _batch(s, x, send_ns=lambda r: -1 - x),
    lambda s, x: _batch(s, x, send_ns=lambda r: 2 ** 64 + x),
    lambda s, x: _unsigned(s, batches=_batch(s, x, receipt_ns=lambda r: r.receipt_ns + 1).signed.batches),
    _first_before_fill,
    lambda s, x: s.resign(fill_start_ns=-1 - x, fill_end_ns=s.signed.fill_end_ns - 2 ** 70),
    lambda s, x: s.resign(batches=s.signed.batches + s.signed.batches[-1:]),
    lambda s, x: s.resign(batches=s.signed.batches[:-1]),
    lambda s, x: s.resign(batches=()),
    lambda s, x: s.resign(ended=[sw.LATE, sw.WRONG, sw.COMPLETE, "", "ünïcode\n\x01"][x % 5]),
    lambda s, x: _unsigned(s, start_mark=_other_mark(s, x)),
    lambda s, x: s.resign(start_mark=_other_mark(s, x)),
]


def _at(path: tuple[str, ...], f):
    """`f` applied at `path` of nested objects."""
    def go(j, x):
        return f(j, x) if not path else j | {path[0]: _at(path[1:], f)(j[path[0]], x)}
    return go


def _drop(j: dict, x: int) -> dict:
    keys = sorted(j)
    return {k: v for k, v in j.items() if k != keys[x % len(keys)]}


def _tiers(f):
    return _at(("commitment",), lambda c, x: c | {"tiers": f(c["tiers"], x)})


def _v1(j: dict, x: int) -> dict:
    """The record as the v1 format wrote it: no start mark, a signed egress count, and the v1 protocol in both
    commitments."""
    v1 = {"protocol": "verity-pous/streamed-wipe/v1"}
    signed = {k: v for k, v in j["signed"].items() if k != "start_mark"} | {"egress_bits": x}
    return j | {"commitment": j["commitment"] | v1, "signed": signed | {"commitment": signed["commitment"] | v1}}


def _slewed_clock(j: dict, x: int) -> dict:
    """The record with `CLOCK_MONOTONIC ns`, the memory challenger's clock, named in both commitments where a wipe's
    header names the raw clock."""
    slewed = {"clock": "CLOCK_MONOTONIC ns"}
    return j | {"commitment": j["commitment"] | slewed,
                "signed": j["signed"] | {"commitment": j["signed"]["commitment"] | slewed}}


def _marks(mark: str, x: int) -> str:
    """A `start_mark` that is not one: upper case, or hex of another length."""
    other = mark or hashlib.sha512(b"a certificate").hexdigest()
    return [other.upper(), other[:-2], other + "00", other[:64], "00", other[:-1], other + other][x % 7]


#: One edit of a valid record's JSON: a field extra or missing, a header constant not the protocol's, the v1 format,
#: the slewed monotonic clock named for the raw one, parameters `WipeParams` refuses, a tier not a pair, hex not
#: lowercase, a start mark of another length, a signed session that names an egress, a boolean or string where an
#: integer goes, a scheme id the port doesn't verify.
JSON_MUTATIONS = [
    lambda j, x: j | {"extra": x},
    _drop,
    _at(("commitment",), _drop),
    _at(("commitment",), lambda c, x: c | {"protocol": "verity-pous/streamed-wipe/v0"}),
    _at(("commitment",), lambda c, x: c | {"protocol": "verity-pous/streamed-wipe/v1"}),
    _at(("signed", "commitment"), lambda c, x: c | {"protocol": "verity-pous/streamed-wipe/v1"}),
    _v1,
    _at(("signed",), lambda s, x: s | {"start_mark": _marks(s["start_mark"], x)}),
    _at(("signed",), lambda s, x: s | {"start_mark": [None, 0, True, [s["start_mark"]]][x % 4]}),
    _at(("commitment",), lambda c, x: c | {"records_tree": "verity-pous/round-records/frame-v3-sha512/v1"}),
    _at(("commitment",), lambda c, x: c | {"label_bytes": 32}),
    _at(("commitment",), lambda c, x: c | {"clock": "CLOCK_REALTIME ns"}),
    _at(("commitment",), lambda c, x: c | {"clock": "CLOCK_MONOTONIC ns"}),
    _at(("signed", "commitment"), lambda c, x: c | {"clock": "CLOCK_MONOTONIC ns"}),
    _slewed_clock,
    _at(("commitment",), lambda c, x: c | {"session": c["session"].upper() or "A"}),
    _at(("commitment",), lambda c, x: c | {"labels": [-1, 0, c["labels"] + 1, str(c["labels"]), True][x % 5]}),
    _at(("commitment",), lambda c, x: c | {"k": [0, c["batch"] - 1, 2 ** 32, -c["k"]][x % 4]}),
    _at(("commitment",), lambda c, x: c | {"batch": [0, c["k"] + 1, -1][x % 3]}),
    _at(("commitment",), lambda c, x: c | {"limit_ns": [0, -1, None][x % 3]}),
    _tiers(lambda t, x: []),
    _tiers(lambda t, x: t + [t[0]]),
    _tiers(lambda t, x: [[t[0][0], 0]] + t),
    _tiers(lambda t, x: [[t[0][0], t[0][1] + 64]] + t[1:]),
    _tiers(lambda t, x: [[t[0][0], t[0][1] - 64]] + t[1:]),
    _tiers(lambda t, x: [t[0][:1]] + t[1:]),
    _tiers(lambda t, x: [t[0] + [1]] + t[1:]),
    _tiers(lambda t, x: [[1, t[0][1]]] + t[1:]),
    _tiers(lambda t, x: [{"gpu": t[0][1]}] + t[1:]),
    _at(("signed",), lambda s, x: s | {"note": x}),
    _at(("signed",), _drop),
    _at(("signed", "commitment"), lambda c, x: c | {"clock": "CLOCK_REALTIME ns"}),
    _at(("signed", "commitment"), lambda c, x: c | {"tiers": []}),
    _at(("signed", "batches"), lambda b, x: [b[0] | {"extra": 1}] + b[1:] if b else [{}]),
    _at(("signed", "batches"), lambda b, x: [{k: v for k, v in b[0].items() if k != sorted(b[0])[x % 7]}] if b else [1]),
    _at(("signed", "batches"), lambda b, x: [b[0] | {"receipt_ns": [str(b[0]["receipt_ns"]), True, None][x % 3]}]
        + b[1:] if b else [None]),
    _at(("signed", "batches"), lambda b, x: {"0": b}),
    _at(("signed",), lambda s, x: s | {"egress_bits": [0, x, True, None][x % 4]}),
    _at(("signed",), lambda s, x: s | {"fill_end_ns": False}),
    _at(("signed",), lambda s, x: s | {"ended": 1}),
    _at(("signed",), lambda s, x: s | {"signature": s["signature"].upper() if s["signature"] else "0"}),
    _at(("signed",), lambda s, x: s | {"public_key": s["public_key"] + "0"}),
    lambda j, x: j | {"trusted": [k.upper() for k in j["trusted"]] or ["A"]},
    lambda j, x: j | {"trusted": "00"},
    lambda j, x: j | {"schemes": j["schemes"] + [["none", mc.LMS + " ", M.ts.scheme(11), "test/"][x % 4]]},
    lambda j, x: j | {"schemes": j["schemes"] + [x]},
]
