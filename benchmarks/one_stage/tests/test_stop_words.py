"""`stop_words.py`: the stand-in network certifier's files of a stop class's words table, and the verifier's read of
them. The class is the stop audit's rehearsal's (`stop_audit_small.committed`)."""

from __future__ import annotations

import importlib
import json
import struct
import sys
from pathlib import Path

import pytest

from verity.commitments.windows import rows as CS
from verity_vllm.pipeline import boundary_rows as BR, boundary_stop as BS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
T = importlib.import_module("test_two_stage_driver")
SW = importlib.import_module("stop_words")

N = T.N


def _signed(out: Path, **more: object) -> tuple[dict[str, Path], dict, dict]:
    """The certifier's files of the stop class's table under ``out``; its head and the verifier's copy, as read."""
    files = T._certified(T._stop_class().spans, out, **more)
    return files, json.loads(files["words"].read_text()), json.loads(files["certifier"].read_text())


def test_the_verifier_reads_its_port_of_a_class_from_the_certifiers_signed_head(tmp_path):
    """A row is the request's ``max_tokens`` then its count of prompt ids, two u32, in the class's order. The
    verifier's read of the signed head, under its own copy of the certifier's key, session and schedule, is its
    ``--registered`` port of the class's table: N rows of four 16-bit leaves under the root the certifier signed,
    proof block r at row r, which is what a stop record's registration takes. The public head holds no row and no
    salt; the opening, the prover's, holds both. A head that lists no table for the class has no port, and a stop
    class without one is refused ``stop-uncertified``."""
    spans = T._stop_class().spans
    files, doc, mine = _signed(tmp_path)
    assert SW.parsed(spans) == [(s.sampling["max_tokens"], len(s.prompt)) for s in spans] == [(4, 3)] * N
    port, dom = SW.read(doc, mine)
    t = SW.opened(json.loads(files["opening"].read_text()))
    assert list(t.rows) == [struct.pack("<II", 4, 3)] * N and len(set(t.salts)) == N
    assert (port["value"], port["root"], port["leaves"], port["words"]) == (t.value, t.entry.root.hex(), N, 4)
    assert port["reads"] == list(range(N)) and port["domain"] == dom.domain_id.hex() == t.domain.domain_id.hex()
    assert BS.certified(port, N) == {t.value: {"root": port["root"], "leaves": N, "words": 4, "domain": port["domain"]}}
    assert SW.port_of(files["words"], files["certifier"]) == port and SW.read_files(None, None) is None
    public = files["words"].read_text() + files["certifier"].read_text()
    assert not any(x.hex() in public for x in (*t.rows, *t.salts))
    assert SW.read(doc, mine, cls=1) is None
    with pytest.raises(BS.StopRefused, match=f"^{BS.UNCERTIFIED}: "):
        BS.certified(SW.read(doc, mine, cls=1), N)
    with pytest.raises(ValueError, match="under the verifier's own copy"):
        SW.read_files(files["words"], None)


def test_the_certifiers_rows_open_its_root_as_the_stager_commits_them(tmp_path):
    """What the prover stages on (`staged`): the certifier's rows as 16-bit leaf words, which with the certifier's
    salts commit under the table's domain to the root the certifier signed, as the stager commits a port
    (`rows.commit_ports`); each row's path opens that root at its position; and a negative's prover's rows keep
    the certifier's salts and paths, and commit to another root."""
    files, doc, mine = _signed(tmp_path)
    port, dom = SW.read(doc, mine)
    t = SW.opened(json.loads(files["opening"].read_text()))
    st = SW.staged(t)
    assert st["rows"].shape == (N, 4) and st["rows"].tolist() == [[4, 0, 3, 0]] * N
    assert (st["value"], st["root"]) == (port["value"], port["root"])

    def root(rows) -> str:
        roots, _ = CS.commit_ports({"v": dom}, {"v": rows}, {}, {"v": b"".join(st["salts"])}, schemas={"v": BR.SCHEMA})
        return roots["v"].hex()
    assert root(st["rows"]) == port["root"]
    assert all(BS.G.verify_read(dom, bytes.fromhex(port["root"]), i, t.bcs[i], bytes.fromhex(st["paths"][i]), BR.SCHEMA)
               for i in range(N))
    engine = SW.staged(t, [(5, 3)] * N)
    assert engine["rows"].tolist() == [[5, 0, 3, 0]] * N and root(engine["rows"]) != port["root"]
    assert (engine["salts"], engine["paths"], engine["root"]) == (st["salts"], st["paths"], st["root"])
    assert BS.read_refusal(port, dom, {"value": t.value, "positions": list(range(N)), "paths": st["paths"]}, t.bcs) is None


def test_a_head_that_is_not_the_certifiers_is_refused_by_name(tmp_path):
    """`read` refuses, each by name: a head whose table's root, count or name is changed after signing (the
    certificate doesn't sign it); a certificate under another key or in another session; a head or a verifier under
    another schedule; a table under another domain than the schedule gives it; rows that are not two u32; another
    format; and a document with a field missing. A changed certifier's table, one ``max_tokens`` another, is a head
    the same key signs, under the same value and domain and another root."""
    files, doc, mine = _signed(tmp_path / "a")
    port, _ = SW.read(doc, mine)
    (entry,) = doc["tables"]

    def tables(**change: object) -> dict:
        return {**doc, "tables": [{**entry, **change}]}
    for wrong, why in ((tables(root="00" * 64), "doesn't sign the head"),
                       (tables(rows=N + 1), "another domain than the schedule gives it"),
                       (tables(name="words.1"), "another domain than the schedule gives it"),
                       (tables(domain="00" * 64), "another domain than the schedule gives it"),
                       ({**doc, "window": 1}, "another domain than the schedule gives it"),
                       ({**doc, "parameters": "00" * 64}, "another schedule"),
                       ({**doc, "format": "verity/one-stage/stop-words/v9"}, "not a "),
                       ({k: v for k, v in doc.items() if k != "certificate"}, "doesn't read: KeyError")):
        with pytest.raises(ValueError, match=why):
            SW.read(wrong, mine)
    for theirs, why in (({**mine, "session": "00" * 32}, "another domain than the schedule gives it"),
                        ({**mine, "parameters": "00" * 64}, "another schedule"),
                        ({**mine, "key": {**mine["key"], "public": "00" * len(bytes.fromhex(mine["key"]["public"]))}},
                         "doesn't verify under the network certifier's key")):
        with pytest.raises(ValueError, match=why):
            SW.read(doc, theirs)
    _, other, theirs = _signed(tmp_path / "b")
    assert theirs == mine
    with pytest.raises(ValueError, match="doesn't sign the head"):
        SW.read({**doc, "certificate": other["certificate"]}, mine)

    (tmp_path / "c").mkdir()
    issuer = SW.certifier(tmp_path / "c" / "signer-state.json")
    three = SW.CR.Table(SW.LINK, 0, SW.name(), SW.WardenParams(), issuer.session,
                        tuple(struct.pack("<III", 4, 3, 0) for _ in range(N)),
                        tuple(bytes([k + 1]) * SW.CR.SALT_BYTES for k in range(N)))
    with pytest.raises(ValueError, match="is 8 bytes: max_tokens, prompt_ids, a u32 each"):
        SW.read(SW.document(issuer, three), SW.verifier_copy(issuer))

    span = next(i for i, x in enumerate(T._stop_class().spans) if x.finish == "length")
    _, changed, theirs = _signed(tmp_path / "d", max_tokens={span: 3})
    got, _ = SW.read(changed, theirs)
    assert theirs == mine and {k for k in port if port[k] != got[k]} == {"root"}
