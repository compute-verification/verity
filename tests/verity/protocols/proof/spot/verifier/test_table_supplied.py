"""A table's supplied rows (`Window.supply`: its root, its domain id and the opening verdict's 1) are the rows R7 recomputes
from the program's digest: a docket whose `supplied` names the table's root and domain id by `bytes` accepts the window's
own rows in both verifiers, and each row's public salts are the program's (`registered_value.supplied_salts`), the digest
`Window.give` and the verifier's `Supplied.root` salt with, never the partition's. It runs Lean's registration check beside
the Python twin (minutes, with the executable built): no cheaper test holds a window's own committed rows to Lean's R7."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from verity.circuits.codec import descriptor_id, program_digest
from verity.circuits.evaluation.batch import evaluate_batch
from verity.commitments import registered_value as RV
from verity.commitments import row_table as RT
from verity.commitments.gates import embedding as E
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.periods import interface as I
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2, bit_row
from verity.protocols.proof.spot.prover import docket as DK
from verity.protocols.proof.spot.verifier.one_stage import docket as R, registered as G, verifier as V

EMB = I.resolve("EmbeddingRead_v2{V=4,H=16}")
EID = descriptor_id(EMB)
RUN = "ab" * 64
DESIGN = {"law": "stratified", "k": 2}
SCHEME = "frame-v3-sha512/hm96-sha512"


class Table:
    """`EMB`'s four-row table under `hm96-sha512/row-seg/v1`, as core registers it."""

    def __init__(self) -> None:
        rng = np.random.default_rng(7)
        self.rows = rng.integers(0, 1 << 16, (4, 16), dtype=np.uint16)
        self.salts = [bytes(rng.integers(0, 256, E.SALT_BYTES, dtype=np.uint8)) for _ in range(4)]
        self.dom = CommitmentDomain(hashlib.sha512(b"embed").digest(), -2, RangeIndexedDomain(0, 4), hash="sha512")
        self.tree = RT.commit(self.dom, [bit_row((r.tolist(), 16)) for r in self.rows], self.salts)
        self.name = self.value = "embed"
        self.root, self.words, self.leaves, self.word_bits = self.tree.root, 16, 4, 16

    def entry(self) -> dict:
        return {"value": "embed", "root": self.tree.root.hex(), "leaves": 4, "words": 16, "domain": self.dom.domain_id.hex()}

    def path(self, tok: int) -> list[bytes]:
        p = self.tree.path(tok)
        return [p[k:k + 64] for k in range(0, len(p), 64)]


def _window(out) -> tuple[I.Window, Table, dict]:
    """Three reads of one table through `Window.supply`, `close` and `write`, as `one_linear.py` writes a window."""
    table = Table()
    reg = I.Registered({"embed": 64}, (I.Table("embed", "embed", 0, 16, 4),), (),
                       (I.TableRead(EID, "root", "embed", "root", 0, 3), I.TableRead(EID, "domain", "embed", "domain", 0, 3)),
                       (I.TableOpening(EID, "embed", "tok", ("row", "salt", "path"), "o1", 0, 3),))
    win = I.Window(I.Circuit("embedding", ((EMB, 3),), registered=reg), RUN, DESIGN, 2, registered={"embed": table})
    win.supply()
    toks = [3, 3, 1]
    got = [E.reference(4, t, table.tree.root, table.dom.domain_id, table.rows[t].tolist(), table.salts[t], table.path(t))
           for t in toks]

    def digest(d: bytes) -> np.ndarray:
        return np.frombuffer(d, ">u8").astype(np.uint64)

    ins = [np.array(toks, dtype=np.uint32), np.stack([digest(table.tree.root)] * 3), np.stack([digest(table.dom.domain_id)] * 3),
           table.rows[toks], np.stack([np.frombuffer(table.salts[t], ">u8").astype(np.uint64) for t in toks]),
           np.stack([np.stack([digest(s) for s in table.path(t)]) for t in toks])]
    win.record(EMB, ins, [np.array([h for h, _ in got], dtype=np.uint16), np.array([ok for _, ok in got], dtype=np.uint16)])
    out.mkdir()
    win.write(win.close(), out, {"table": "salt"}, registrar=DK)
    return win, table, json.loads((out / "registration.json").read_text())


def test_a_tables_supplied_rows_are_the_rows_r7_recomputes_from_the_programs_digest(tmp_path):
    """The window's root and domain rows, registered in a docket whose `supplied` names the table's root and domain id
    (`bytes`, the sources today's R7 reads), are the rows both verifiers compute: their salts are the program's, and R7 holds
    the record's roots to them. Salted with the partition's digest, R7 refuses them."""
    win, table, window = _window(tmp_path / "window")
    prog, query = win.window.program, win.window.query
    pd = program_digest(prog)
    assert pd == window["program"] != win.window.partition_digest
    values = {f"embed/{of}": win.registered[f"embed/{of}"] for of in ("root", "domain")}
    rec = DK.record(prog, query, DESIGN, SCHEME, window["roots"], {"kind": "served", "row": 7, "run": "r-table"},
                    leaf_layer=window["leaf_layer"], registered=[v.entry() for v in values.values()])
    check = {"program": prog, "query": query, "sampling_design": DESIGN, "scheme": SCHEME,
             "ports": {r["port"]: {"leaves": r["leaves"], "domain": r["domain"]} for r in window["roots"]},
             "leaf_layer": window["leaf_layer"], "registered": {v: x.words for v, x in values.items()},
             "schemas": {v: V2 for v in values},
             "supplied": {"embed/root": [[{"bytes": table.tree.root.hex()}]],
                          "embed/domain": [[{"bytes": table.dom.domain_id.hex()}]]}}
    got = V.check(rec, **check)
    assert got.ok and got.codes == (), got
    assert R.check(rec, **check).ok
    for of in ("root", "domain", "opened"):
        row = win.registered[f"embed/{of}"]
        assert row.program == pd and list(row.salts) == RV.supplied_salts(pd, f"embed/{of}", 1), of


def test_a_tables_root_and_domain_marks_give_the_rows_the_record_commits():
    """`registered.root/v1` and `registered.domain/v1` with the table's registered name as `value`, read off the record's
    `registered` entry of that name, give the 64 bytes `supplied` commits for the table's root and domain id: as a
    `Digest`'s 512 leaves packed into its 32 words (`row/v2`), the same row, and under the program's salts the same root."""
    table = Table()
    reg = I.Registered({"embed": 64}, (I.Table("embed", "embed", 0, 16, 4),), (),
                       (I.TableRead(EID, "root", "embed", "root", 0, 3), I.TableRead(EID, "domain", "embed", "domain", 0, 3)),
                       (I.TableOpening(EID, "embed", "tok", ("row", "salt", "path"), "o1", 0, 3),))
    prog = "cd" * 64
    committed = I.supplied(reg, {"embed": table.entry()}, prog)
    rec = {"registered": [table.entry()]}
    for of, want in (("root", table.tree.root), ("domain", table.dom.domain_id)):
        name = I.supplied_param(reg.readable["embed"], of)
        mark = {"supplied": f"registered.{of}/v1", "args": {"value": "embed"}, "layout": "digest", "bits": 512}
        (row,) = G.of_mark({}, {}, name, mark)
        leaves = G.row_leaves_of(row, lambda k, n: G._source_bytes(rec, name, k, n, [name], [], None), 1, True)
        words = G.bit_words(leaves, 1, 16)
        assert leaves == G.digest_leaves(want) and words == I.digest_words(want) == list(committed[name].rows[0]), of
        mine = RV.Registered.commit(prog, name, [words], RV.supplied_salts(prog, name, 1), schema=V2)
        assert mine.root == committed[name].root and mine.entry() == committed[name].entry(), of


def _anchor(rec: dict, of: str, value: str) -> list[int]:
    """The leaves `registered.<of>/v1` of ``value`` gives a `Digest` port, read off ``rec``."""
    mark = {"supplied": f"registered.{of}/v1", "args": {"value": value}, "layout": "digest", "bits": 512}
    (row,) = G.of_mark({}, {}, f"{value}/{of}", mark)
    return G.row_leaves_of(row, lambda k, n: G._source_bytes(rec, f"{value}/{of}", k, n, [], [], None), 1, True)


def _opened(table: Table, tok: int, cases: list[tuple[list[int], list[int], list[int]]]) -> list[tuple[list[int], int]]:
    """`embedding.opening(4, 16)` at ``tok`` on each ``(root, domain, row)`` leaves, the salt and path ``tok``'s: (h, ok)."""
    args = E.arguments(4, tok, table.tree.root, table.dom.domain_id, table.rows[tok].tolist(), table.salts[tok], table.path(tok))
    got = evaluate_batch(E.opening(4, 16), np.array([[tok]] * len(cases), dtype=object),
                         *(np.array([c[k] for c in cases], dtype=object) for k in range(3)),
                         *(np.array([args[k]] * len(cases), dtype=object) for k in (4, 5)))
    return [([int(w) for w in g[:16]], int(g[16])) for g in np.asarray(got).reshape(len(cases), -1)]


def _anchored() -> tuple[Table, dict, int, list[int]]:
    table = Table()
    rec = {"registered": [table.entry(), {"value": "other", "root": "11" * 64, "domain": "22" * 64}]}
    tok = 2
    return table, rec, tok, [(int(w) >> j) & 1 for w in table.rows[tok].tolist() for j in range(16)]


@pytest.mark.slow
def test_a_tables_anchors_from_their_supplied_rows_open_its_honest_row_and_not_under_a_swapped_domain():
    """The opening (`embedding.opening`) evaluated with its root and domain ports the leaves the table's
    `registered.root/v1` and `registered.domain/v1` marks give: an honest row opens, ``ok`` 1 and the row its own; the
    same read with the domain swapped for another value's is ``ok`` 0. Slow (a minute or two tracing the opening's
    Merkle path): only an evaluation holds the supplied anchors to what the opening's circuit expects as root and domain."""
    table, rec, tok, row = _anchored()
    root, domain = _anchor(rec, "root", "embed"), _anchor(rec, "domain", "embed")
    got = _opened(table, tok, [(root, domain, row), (root, _anchor(rec, "domain", "other"), row)])
    assert [ok for _, ok in got] == [1, 0] and got[0][0] == table.rows[tok].tolist()


@pytest.mark.slow
@pytest.mark.skipif(descriptor_id(E.opening(4, 16)).startswith("EmbeddingRead_v3"),
                    reason="the opening gates its row on ok from EmbeddingRead_v4")
def test_an_in_range_token_with_a_wrong_row_reads_the_fill_row_and_ok_0():
    """At an in-range token a row other than the table's, under the supplied anchors, gives the fill row and ``ok`` 0.
    Slow as the test above, whose opening it evaluates."""
    table, rec, tok, row = _anchored()
    row[0] ^= 1
    got = _opened(table, tok, [(_anchor(rec, "root", "embed"), _anchor(rec, "domain", "embed"), row)])
    assert got == [([E.FILL] * 16, 0)]
