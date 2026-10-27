"""The network certifier's output value (`out_commit`): committed at the registration's shape (`OutShape`) as the
reading program's pinned registered value under the certifier's salts, its head signed for the egress link-window, and
the verifier's read of the record refusing a wrong key, session, link, window, width, row count or domain, and any
flipped byte, by name; at the wired toy's 128 words a row as at one."""

from __future__ import annotations

import itertools
import json

import pytest

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value
from verity.ml.crypto import lms
from verity.core.protocols.certifier import out_commit as O
from verity.core.protocols.certifier.certificate import Issuer, LmsKey, LmsSigner, SignerRefused
from verity.core.protocols.certifier.tests import signers as S
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity.core.protocols.sampled_proofs.one_stage import registered as G

ROWS, WORDS, WINDOW = 4, 8, 3


def _program(rows: int = ROWS, words: int = WORDS) -> Program:
    def sig(S_):
        return (("out", Array(rows, Array(words, Value(16)))),), Array(rows, Array(words, Value(16)))

    return Program(bind(CompositeDefinition("OutCommitTest", 0, (), sig, lambda B, S_, out: out, register=False)))


PROG = _program()
VALUES = [[(i * 31 + j * 7) % 65536 for j in range(WORDS)] for i in range(ROWS)]


def _domain(prog: Program = PROG, words: int = WORDS) -> str:
    """The verifier's domain for `out`, pinned: `registration.check` holds a pinned value across windows."""
    entries = G.registered_entries(prog, {"out": words}, {"out": G.SCHEMA_V2}, window="ab" * 64, fixed=["out"])
    return entries["out"]["domain"]


def _shape(prog: Program = PROG, words: int = WORDS, rows: int = ROWS, **kw) -> O.OutShape:
    return O.OutShape(P.program_digest(prog), "out", words, rows, **kw)


SHAPE = _shape()


@pytest.fixture(scope="module")
def signed(tmp_path_factory) -> tuple[O.CertifiedOut, LmsKey]:
    s = S.signer(tmp_path_factory.mktemp("out-signer"))
    return O.commit_out(VALUES, shape=SHAPE, key=s, session=S.SESSION, window=WINDOW), s.public_key()


@pytest.fixture
def fresh(tmp_path):
    """A factory of signers with fresh leaf state, each under its own directory."""
    made = itertools.count()

    def make() -> LmsSigner:
        d = tmp_path / f"signer-{next(made)}"
        d.mkdir()
        return S.signer(d)

    return make


def _record(out: O.CertifiedOut) -> dict:
    return json.loads(json.dumps(out.public))


def _flip(h: str, i: int = 0) -> str:
    return h[:i] + format(int(h[i], 16) ^ 1, "x") + h[i + 1:]


def _refused(name: str, doc, key, session=S.SESSION, shape=SHAPE, window=WINDOW, **kw) -> None:
    with pytest.raises(O.OutRefused) as e:
        O.check_out(doc, key, session, shape, window=window, **kw)
    assert e.value.name == name and str(e.value).startswith(f"{name}: "), str(e.value)


def test_round_trip(signed):
    out, key = signed
    doc = _record(out)
    assert O.check_out(doc, key, S.SESSION, SHAPE, window=WINDOW) == out.root
    assert SHAPE.domain.domain_id.hex() == _domain() == out.domain
    assert O.OutShape.of(json.loads(json.dumps(SHAPE.doc()))) == SHAPE
    assert set(doc) == {"format", "link", "direction", "window", "session", "value", "program", "words", "rows",
                        "word_bits", "row_schema", "domain", "root", "certificate"}
    assert (doc["link"], doc["direction"], doc["window"]) == (O.LINK, "egress", WINDOW)
    assert (doc["value"], doc["words"], doc["rows"], doc["word_bits"], doc["row_schema"]) == \
        ("out", WORDS, ROWS, 16, G.SCHEMA_V2)
    text = json.dumps(doc)
    assert not any(s.hex() in text for s in out.salts), "the record carries no salt"
    assert all(len(s) == O.SALT_BYTES == 192 for s in out.salts) and len(set(out.salts)) == ROWS
    assert O.read_witness(out.witness) == list(out.salts)


def test_root_is_the_registered_commit(signed, fresh):
    """The root is the one `Registered.commit` gives the reading program's `out` under the same salts, in the domain the
    verifier derives for it, so the Lean verifier's registered reads open the certifier's root as the prover's."""
    out, _ = signed
    reg = G.Registered.commit(P.program_digest(PROG), "out", VALUES, out.salts, 16, G.SCHEMA_V2)
    assert out.root == reg.root.hex() and out.domain == reg.domain.domain_id.hex() == _domain()
    assert reg.entry() == out.committed.entry()
    again = O.commit_out(VALUES, shape=SHAPE, key=Issuer(fresh(), S.SESSION), session=S.SESSION, window=WINDOW,
                         salts=out.salts)
    assert again.root == out.root, "the same salts commit the same root"
    rows, salts, entry = out.statement([2, 0])
    assert rows == [tuple(VALUES[2]), tuple(VALUES[0])] and salts == [out.salts[2], out.salts[0]]
    for p, path in zip(entry["positions"], entry["paths"]):
        assert G.verify_read(out.committed.domain, out.head.root, p, out.committed.bcs[p], bytes.fromhex(path),
                             G.SCHEMA_V2)


def test_fresh_salts_each_commit(fresh):
    a = O.commit_out(VALUES, shape=SHAPE, key=fresh(), session=S.SESSION, window=WINDOW)
    b = O.commit_out(VALUES, shape=SHAPE, key=fresh(), session=S.SESSION, window=WINDOW)
    assert a.domain == b.domain and a.root != b.root and not set(a.salts) & set(b.salts)


def test_wrong_key(signed):
    out, _ = signed
    other = LmsKey(S.SMALL, lms.public_key(S.key(b"another certifier")))
    _refused("out-key", _record(out), other)


def test_wrong_session(signed, fresh):
    out, key = signed
    _refused("out-session", _record(out), key, session=S.OTHER_SESSION)
    elsewhere = O.commit_out(VALUES, shape=SHAPE, key=fresh(), session=S.OTHER_SESSION, window=WINDOW)
    doc = _record(elsewhere)
    _refused("out-session", doc, key)
    doc["session"] = S.SESSION.hex()
    _refused("out-signature", doc, key)


def test_wrong_link_or_window(signed):
    out, key = signed
    _refused("out-window", _record(out), key, window=WINDOW + 1)
    _refused("out-link", _record(out), key, link="another link")


def test_window_replay(fresh):
    """`out`'s domain is the same every window, so window 0's record must not pin window 1's root."""
    issuer = Issuer(fresh(), S.SESSION)
    w0 = O.commit_out(VALUES, shape=SHAPE, key=issuer, session=S.SESSION, window=0)
    w1 = O.commit_out(VALUES, shape=SHAPE, key=issuer, session=S.SESSION, window=1)
    key = issuer.signer.public_key()
    assert w0.domain == w1.domain and w0.root != w1.root
    assert O.check_out(_record(w1), key, S.SESSION, SHAPE, window=1) == w1.root
    _refused("out-window", _record(w0), key, window=1)
    doc = _record(w0)
    doc["window"] = 1
    _refused("out-signature", doc, key, window=1)


def test_wrong_domain_width_or_rows(signed):
    """The verifier's own shape decides: another reading program or value is `out-domain`, another width, word size or
    row schema `out-width` (1 against 8, 8 against 128, 32-bit words, row/v1), another row count `out-rows`."""
    out, key = signed
    _refused("out-domain", _record(out), key, shape=_shape(_program(ROWS, WORDS + 1), WORDS, ROWS))
    _refused("out-domain", _record(out), key, shape=O.OutShape(SHAPE.program, "outs", WORDS, ROWS))
    for shape in (_shape(words=1), _shape(words=128), _shape(word_bits=32), _shape(schema=G.SCHEMA)):
        _refused("out-width", _record(out), key, shape=shape)
    _refused("out-rows", _record(out), key, shape=_shape(rows=ROWS - 1))
    with pytest.raises(ValueError, match="an OutShape"):
        O.check_out(_record(out), key, S.SESSION, _domain(), window=WINDOW)


def test_round_trip_at_the_wired_width(fresh):
    """The wired toy's `out`: R·S rows of the down_proj's 128 words, its root `Registered.commit`'s at that width, and a
    record of it refused under a one-word shape, as a one-word record under the wired one."""
    rows = [[(r * 131 + j) % 65536 for j in range(128)] for r in range(4)]
    wide, narrow = _shape(_program(4, 128), 128, 4), _shape(_program(4 * 128, 1), 1, 4 * 128)
    issuer = Issuer(fresh(), S.SESSION)
    got = O.commit_out(rows, shape=wide, key=issuer, session=S.SESSION, window=0)
    key = issuer.signer.public_key()
    assert O.check_out(_record(got), key, S.SESSION, wide, window=0) == got.root
    assert got.root == G.Registered.commit(wide.program, "out", rows, got.salts, 16, G.SCHEMA_V2).root.hex()
    _refused("out-width", _record(got), key, shape=O.OutShape(wide.program, "out", 1, 4 * 128), window=0)
    one = O.commit_out([[w] for r in rows for w in r], shape=narrow, key=issuer, session=S.SESSION, window=1)
    _refused("out-width", _record(one), key, shape=O.OutShape(narrow.program, "out", 128, 4), window=1)
    doc = _record(got)
    doc["words"] = 1
    _refused("out-width", doc, key, shape=wide, window=0)


@pytest.mark.parametrize("field, name", [
    ("root", "out-digest"), ("domain", "out-domain"), ("program", "out-domain"), ("session", "out-session"),
    ("certificate.key", "out-key"), ("certificate.digest", "out-digest"), ("certificate.signature", "out-signature"),
])
@pytest.mark.parametrize("at", [0, 37])
def test_flipped_byte(signed, field, name, at):
    out, key = signed
    doc = _record(out)
    holder, k = (doc["certificate"], field.split(".")[1]) if "." in field else (doc, field)
    holder[k] = _flip(holder[k], at % len(holder[k]))
    _refused(name, doc, key)


@pytest.mark.parametrize("field, value, name", [
    ("rows", ROWS + 1, "out-rows"), ("words", WORDS + 1, "out-width"), ("word_bits", 32, "out-width"),
    ("value", "outs", "out-domain"), ("row_schema", G.SCHEMA, "out-width"), ("direction", "ingress", "out-link"),
])
def test_edited_field(signed, field, value, name):
    out, key = signed
    doc = _record(out)
    doc[field] = value
    _refused(name, doc, key)


@pytest.mark.parametrize("edit", ["certificate", "extra", "missing", "string window", "bool window", "format",
                                  "schema", "short root", "certificate keys"])
def test_not_a_record(signed, edit):
    out, key = signed
    doc = _record(out)
    if edit == "certificate":
        doc = out.certificate
    elif edit == "extra":
        doc["salts"] = out.witness.hex()
    elif edit == "missing":
        del doc["root"]
    elif edit == "string window":
        doc["window"] = str(WINDOW)
    elif edit == "bool window":
        doc["window"] = True
    elif edit == "format":
        doc["format"] = "verity-network-certifier/transfer-registration/v1"
    elif edit == "schema":
        doc["row_schema"] = "u16"
    elif edit == "short root":
        doc["root"] = doc["root"][:-2]
    else:
        doc["certificate"]["salt"] = "00"
    _refused("out-record", doc, key)


def test_signer_refusal(fresh):
    """A signer signs window after window of its one session, and refuses another session, so that window has no out
    certificate."""
    s = fresh()
    for w in (0, 1):
        out = O.commit_out(VALUES, shape=SHAPE, key=s, session=S.SESSION, window=w)
        assert O.check_out(_record(out), s.public_key(), S.SESSION, SHAPE, window=w) == out.root
    with pytest.raises(SignerRefused, match="refused the out certificate of out window 2"):
        O.commit_out(VALUES, shape=SHAPE, key=s, session=S.OTHER_SESSION, window=2)


def test_commit_refuses(fresh):
    with pytest.raises(ValueError, match="4 rows of 8 leaves"):
        O.commit_out([*VALUES[:-1], VALUES[-1][:-1]], shape=SHAPE, key=fresh(), session=S.SESSION, window=0)
    with pytest.raises(ValueError, match="4 rows of 8 leaves"):
        O.commit_out(VALUES[:-1], shape=SHAPE, key=fresh(), session=S.SESSION, window=0)
    with pytest.raises(ValueError, match="another certifier session"):
        O.commit_out(VALUES, shape=SHAPE, key=Issuer(fresh(), S.OTHER_SESSION), session=S.SESSION, window=0)
    with pytest.raises(ValueError, match="program digest"):
        O.OutShape("ab", "out", WORDS, ROWS)
    for bad in ({**SHAPE.doc(), "words": 0}, {**SHAPE.doc(), "row_schema": "u16"}, {**SHAPE.doc(), "extra": 1}):
        with pytest.raises(ValueError, match="out shape"):
            O.OutShape.of(bad)
