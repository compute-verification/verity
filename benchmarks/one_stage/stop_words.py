"""A stop class's certified words table as the stop audit's drivers are handed it: each released request's
``max_tokens`` and count of prompt ids (`boundary_stop.COLUMNS`), one row a release in the class's release order, two
u32 little-endian, committed as the network certifier commits a table of an execution phase (`connection_rows.Table`)
and listed in a head the certifier signs once.

No schedule of the certifier's lists a words table (`connection_rows.tables_of`), so no device makes one and
`connection_rows.check_registration` reads none. The table here is a stand-in certifier's (`sign`): the two words of
each span of a class as that certifier parsed them from the request it released, under a test-size key of the
certifier's scheme whose seed is public. It certifies nothing about a serve. Three files, one a party:

* ``words.json``, public: the head and its one certificate (`document`).
* ``certifier.json``, the verifier's own copy of the certifier's key, session and schedule (`verifier_copy`). `read`
  is the verifier's: the head's certificate under that key in that session, each table's domain as the schedule gives
  it, and then its own ``--registered`` port of the class's table (`connection_rows.port`), which is what
  `boundary_stop.stop_registration` takes. A head that lists no such table has no port: the class has no certified
  words, and registers no stop record.
* ``words-opening.json``, the prover's: the certifier's rows and salts (`opening`), on which it stages its stop
  blocks' ``words`` port (`staged`). It is witness, and no verifier's half reads it."""
from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verity.commitments.crypto import lms
from verity.commitments.merkle import CommitmentDomain
from verity.protocols.certifier.verifier import connection_rows as CR
from verity.protocols.certifier.verifier.certificate import Certificate, Direction, Issuer, LmsKey, LmsSigner
from verity.protocols.certifier.verifier.warden import WardenParams
from verity_vllm.pipeline import boundary_rows as BR, boundary_stop as BS

FORMAT = "verity/one-stage/stop-words/v0"
OPENING = "verity/one-stage/stop-words-opening/v0"
#: a row: a release's ``max_tokens`` and its count of prompt ids
ROW = struct.Struct("<II")
assert ROW.size * 8 == 32 * len(BS.COLUMNS)
#: the stand-in certifier's link, its session and its key's parameter set: N = 64, W = 1, a tree of 2^6 leaves
LINK, SESSION, KEY = "stop-words", hashlib.sha512(b"stop-words/session").digest()[:32], lms.Params(64, 1, 6)
PUBLIC, CERTIFIER, PRIVATE = "words.json", "certifier.json", "words-opening.json"


def name(cls: int = 0) -> str:
    """The words table of the certifier's class ``cls``."""
    return f"words.{cls}"


def parsed(spans: Sequence[BR.Span], max_tokens: Mapping[int, int] | None = None) -> list[tuple[int, int]]:
    """Each span's request's two words as a certifier that released the request the engine served parses them, in the
    spans' order. ``max_tokens``: by span, the ``max_tokens`` of the request the certifier released where the engine
    served another."""
    return [((max_tokens or {}).get(i, s.sampling["max_tokens"]), len(s.prompt)) for i, s in enumerate(spans)]


def table(window: int, rows: Sequence[tuple[int, int]], salts: Sequence[bytes] | None = None, *, cls: int = 0,
          link: str = LINK, session: bytes = SESSION, params: WardenParams | None = None) -> CR.Table:
    """The certifier's opening of a class's words table of ``rows``. ``salts``: one a row, fresh by default."""
    mine = [CR._INST.fresh_salt() for _ in rows] if salts is None else [bytes(s) for s in salts]
    if len(mine) != len(rows) or len(set(mine)) != len(mine) or any(len(s) != CR.SALT_BYTES for s in mine):
        raise ValueError(f"one {CR.SALT_BYTES}-byte salt a row, each used once")
    return CR.Table(link, window, name(cls), WardenParams() if params is None else params, bytes(session),
                    tuple(ROW.pack(*r) for r in rows), tuple(mine))


def certifier(state: Path, seed: bytes = b"stop-words/key", session: bytes = SESSION) -> Issuer:
    """The stand-in certifier's signing step: a key of `KEY` from the public ``seed``, its leaf state at ``state``."""
    return Issuer(LmsSigner.create(lms.keygen(KEY, hashlib.sha512(seed).digest()), state), bytes(session))


def verifier_copy(issuer: Issuer, params: WardenParams | None = None) -> dict[str, Any]:
    """The verifier's own copy of the certifier (``certifier.json``): its key, its session and its schedule's digest."""
    key = issuer.signer.public_key()
    return {"key": {"N": key.params.N, "W": key.params.W, "H": key.params.H, "public": key.public.hex()},
            "session": issuer.session.hex(), "parameters": (WardenParams() if params is None else params).digest.hex()}


def document(issuer: Issuer, *tables: CR.Table) -> dict[str, Any]:
    """An execution phase's head listing ``tables`` and the certifier's one certificate of it (``words.json``)."""
    first = tables[0]
    head = CR.Head(first.link, first.window, first.params.digest, tuple(t.entry for t in tables))
    c = issuer.sign(head.link, Direction.EGRESS, head.window, head.digest())
    if c is None:
        raise ValueError("the certifier's signer refused the head")
    return {"format": FORMAT, "link": head.link, "window": head.window, "parameters": head.parameters.hex(),
            "tables": [{"name": e.name, "rows": e.rows, "row_bytes": e.row_bytes, "domain": e.domain.hex(),
                        "root": e.root.hex()} for e in head.tables],
            "certificate": {"key": c.key_id.hex(), "digest": c.digest.hex(), "signature": c.signature.hex()}}


def opening(t: CR.Table) -> dict[str, Any]:
    """The certifier's opening of ``t`` as it hands it to the prover (``words-opening.json``): witness."""
    return {"format": OPENING, "link": t.link, "window": t.window, "name": t.name, "session": t.session.hex(),
            "rows": [r.hex() for r in t.rows], "salts": [s.hex() for s in t.salts]}


def opened(doc: Mapping[str, Any], params: WardenParams | None = None) -> CR.Table:
    """`opening`'s table again."""
    if doc.get("format") != OPENING:
        raise ValueError(f"not a {OPENING} file")
    return CR.Table(doc["link"], doc["window"], doc["name"], WardenParams() if params is None else params,
                    bytes.fromhex(doc["session"]), tuple(bytes.fromhex(r) for r in doc["rows"]),
                    tuple(bytes.fromhex(s) for s in doc["salts"]))


def read(doc: Mapping[str, Any], mine: Mapping[str, Any], cls: int = 0,
         params: WardenParams | None = None) -> tuple[dict[str, Any], CommitmentDomain] | None:
    """The verifier's read of ``words.json`` under its own copy of the certifier (``mine``, `verifier_copy`'s) and its
    schedule: its ``--registered`` port of class ``cls``'s words table and that table's domain as it derived it, or
    None where the head lists none. ValueError, naming what fails, for a document that is no head of that certifier's:
    another format or schedule, a certificate that doesn't sign the head or doesn't verify under the key in the
    session, a table under another domain than the schedule gives it at its count, or rows that are not two u32."""
    params = WardenParams() if params is None else params
    if doc.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} document")
    if mine.get("parameters") != params.digest.hex() or doc.get("parameters") != params.digest.hex():
        raise ValueError("the head is under another schedule than the verifier's")
    try:
        k, session = mine["key"], bytes.fromhex(mine["session"])
        key = LmsKey(lms.Params(k["N"], k["W"], k["H"]), bytes.fromhex(k["public"]))
        link, window = doc["link"], doc["window"]
        entries, domains = [], {}
        for t in doc["tables"]:
            dom = CR.domain(params, session, link, window, t["name"], t["rows"], t["row_bytes"])
            if dom.domain_id.hex() != t["domain"]:
                raise ValueError(f"the head's table {t['name']} is under another domain than the schedule gives it")
            entries.append(CR.Entry(t["name"], t["rows"], t["row_bytes"], dom.domain_id, bytes.fromhex(t["root"])))
            domains[t["name"]] = dom
        head = CR.Head(link, window, params.digest, tuple(entries))
        c = doc["certificate"]
        certificate = Certificate(bytes.fromhex(c["key"]), link, Direction.EGRESS, window, bytes.fromhex(c["digest"]),
                                  bytes.fromhex(c["signature"]))
    except (KeyError, TypeError) as e:
        raise ValueError(f"the head doesn't read: {type(e).__name__}: {e}") from None
    if certificate.digest != head.digest():
        raise ValueError("the certificate doesn't sign the head")
    if not certificate.verify(key, session):
        raise ValueError("the certificate doesn't verify under the network certifier's key in the session")
    if name(cls) not in [e.name for e in head.tables]:
        return None
    if head.table(name(cls)).row_bytes != ROW.size:
        raise ValueError(f"a row of {name(cls)} is {ROW.size} bytes: {', '.join(BS.COLUMNS)}, a u32 each")
    return CR.port(head, name(cls)), domains[name(cls)]


def staged(t: CR.Table, rows: Sequence[tuple[int, int]] | None = None) -> dict[str, Any]:
    """What a prover stages its stop blocks' ``words`` port on, from the certifier's opening ``t``: the rows as 16-bit
    leaf words, each row's salt and its frame-v3 path in the certifier's tree, and the table's value and root.
    ``rows``: a negative's prover's, other words than the certifier's under the certifier's salts and paths."""
    data = b"".join(t.rows if rows is None else [ROW.pack(*r) for r in rows])
    return {"rows": np.frombuffer(data, dtype="<u2").reshape(len(t.rows), -1).copy(), "salts": list(t.salts),
            "paths": [t.path(i).hex() for i in range(len(t.rows))], "value": t.value, "root": t.entry.root.hex()}


def sign(issuer: Issuer, spans: Sequence[BR.Span], window: int, out: Path, *, link: str = LINK,
         max_tokens: Mapping[int, int] | None = None) -> dict[str, Path]:
    """The stand-in certifier's three files of a class of ``spans`` under ``out`` (module docstring), its head signed
    by ``issuer`` (`certifier`'s: one signer signs every head of a key). ``max_tokens``: `parsed`'s."""
    out.mkdir(parents=True, exist_ok=True)
    t = table(window, parsed(spans, max_tokens), link=link, session=issuer.session)
    files = {"words": out / PUBLIC, "certifier": out / CERTIFIER, "opening": out / PRIVATE}
    files["words"].write_text(json.dumps(document(issuer, t), indent=1) + "\n")
    files["certifier"].write_text(json.dumps(verifier_copy(issuer), indent=1) + "\n")
    files["opening"].write_text(json.dumps(opening(t)) + "\n")
    return files


def read_files(words: str | Path | None, mine: str | Path | None,
               cls: int = 0) -> tuple[dict[str, Any], CommitmentDomain] | None:
    """`read` on the two files a driver is handed (``--words``, ``--certifier``); None where it is handed no head."""
    if words is None:
        return None
    if mine is None:
        raise ValueError("--words is read under the verifier's own copy of the network certifier: --certifier")
    return read(json.loads(Path(words).read_text()), json.loads(Path(mine).read_text()), cls)


def port_of(words: str | Path | None, mine: str | Path | None, cls: int = 0) -> dict[str, Any] | None:
    """`read_files`' port alone, which is what a stop record's registration takes."""
    got = read_files(words, mine, cls)
    return None if got is None else got[0]


WORDS_HELP = ("the network certifier's signed head listing the class's words table (stop_words.py: words.json); "
              "without it a stop class has no certified words and registers no stop record")
CERTIFIER_HELP = "the verifier's own copy of the network certifier's key, session and schedule (certifier.json)"
OPENING_HELP = ("the network certifier's opening of the class's words table, as it hands it to the prover "
                "(words-opening.json): the prover's half reads it")
