"""The network certifier's output value (`out_commit`): committed as the reading program's pinned registered value
under the certifier's salts, its head signed for the egress link-window, and the verifier's read of the record refusing
a wrong key, session, link, window or domain, and any flipped byte, by name."""

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


@pytest.fixture(scope="module")
def signed(tmp_path_factory) -> tuple[O.CertifiedOut, LmsKey]:
    s = S.signer(tmp_path_factory.mktemp("out-signer"))
    return O.commit_out(VALUES, program=PROG, words=WORDS, key=s, session=S.SESSION, window=WINDOW), s.public_key()


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


def _refused(name: str, doc, key, session=S.SESSION, domain=None, window=WINDOW, **kw) -> None:
    with pytest.raises(O.OutRefused) as e:
        O.check_out(doc, key, session, _domain() if domain is None else domain, window=window, **kw)
    assert e.value.name == name and str(e.value).startswith(f"{name}: "), str(e.value)


def test_round_trip(signed):
    out, key = signed
    doc = _record(out)
    assert O.check_out(doc, key, S.SESSION, _domain(), window=WINDOW) == out.root
    assert O.check_out(doc, key, S.SESSION, out.committed.domain, window=WINDOW) == out.root, "a CommitmentDomain too"
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
    again = O.commit_out(VALUES, program=P.program_digest(PROG), words=WORDS, key=Issuer(fresh(), S.SESSION),
                         session=S.SESSION, window=WINDOW, salts=out.salts)
    assert again.root == out.root, "a program digest commits as its program"
    rows, salts, entry = out.statement([2, 0])
    assert rows == [tuple(VALUES[2]), tuple(VALUES[0])] and salts == [out.salts[2], out.salts[0]]
    for p, path in zip(entry["positions"], entry["paths"]):
        assert G.verify_read(out.committed.domain, out.head.root, p, out.committed.bcs[p], bytes.fromhex(path),
                             G.SCHEMA_V2)


def test_fresh_salts_each_commit(fresh):
    a = O.commit_out(VALUES, program=PROG, words=WORDS, key=fresh(), session=S.SESSION, window=WINDOW)
    b = O.commit_out(VALUES, program=PROG, words=WORDS, key=fresh(), session=S.SESSION, window=WINDOW)
    assert a.domain == b.domain and a.root != b.root and not set(a.salts) & set(b.salts)


def test_wrong_key(signed):
    out, _ = signed
    other = LmsKey(S.SMALL, lms.public_key(S.key(b"another certifier")))
    _refused("out-key", _record(out), other)


def test_wrong_session(signed, fresh):
    out, key = signed
    _refused("out-session", _record(out), key, session=S.OTHER_SESSION)
    elsewhere = O.commit_out(VALUES, program=PROG, words=WORDS, key=fresh(), session=S.OTHER_SESSION, window=WINDOW)
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
    w0 = O.commit_out(VALUES, program=PROG, words=WORDS, key=issuer, session=S.SESSION, window=0)
    w1 = O.commit_out(VALUES, program=PROG, words=WORDS, key=issuer, session=S.SESSION, window=1)
    key = issuer.signer.public_key()
    assert w0.domain == w1.domain and w0.root != w1.root
    assert O.check_out(_record(w1), key, S.SESSION, _domain(), window=1) == w1.root
    _refused("out-window", _record(w0), key, window=1)
    doc = _record(w0)
    doc["window"] = 1
    _refused("out-signature", doc, key, window=1)


def test_wrong_domain(signed):
    out, key = signed
    _refused("out-domain", _record(out), key, domain=_domain(_program(ROWS * 2, WORDS)))
    _refused("out-domain", _record(out), key, domain=_domain(_program(ROWS * 2, WORDS // 2), WORDS // 2))


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
    ("rows", ROWS + 1, "out-domain"), ("words", WORDS + 1, "out-domain"), ("word_bits", 32, "out-domain"),
    ("value", "outs", "out-domain"), ("row_schema", G.SCHEMA, "out-digest"), ("direction", "ingress", "out-link"),
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
        out = O.commit_out(VALUES, program=PROG, words=WORDS, key=s, session=S.SESSION, window=w)
        assert O.check_out(_record(out), s.public_key(), S.SESSION, _domain(), window=w) == out.root
    with pytest.raises(SignerRefused, match="refused the out certificate of out window 2"):
        O.commit_out(VALUES, program=PROG, words=WORDS, key=s, session=S.OTHER_SESSION, window=2)


def test_commit_refuses(fresh):
    with pytest.raises(ValueError, match="each of 8 leaves"):
        O.commit_out([*VALUES[:-1], VALUES[-1][:-1]], program=PROG, words=WORDS, key=fresh(), session=S.SESSION,
                     window=0)
    with pytest.raises(ValueError, match="another certifier session"):
        O.commit_out(VALUES, program=PROG, words=WORDS, key=Issuer(fresh(), S.OTHER_SESSION), session=S.SESSION,
                     window=0)
    with pytest.raises(ValueError, match="program digest"):
        O.commit_out(VALUES, program="ab", words=WORDS, key=fresh(), session=S.SESSION, window=0)
