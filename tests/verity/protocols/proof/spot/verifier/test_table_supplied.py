"""A table's supplied rows (`Window.supply`: its root, its domain id and the opening verdict's 1) are the rows R7 recomputes
from the program's digest: with the anchors' values from C's marks (`registered.root/v1`, `registered.domain/v1` of the
table's registered value) read off the table's pin, never the record's entry, both verifiers accept the window's own rows, and each row's public salts are the
program's (`registered_value.supplied_salts`), the digest `Window.give` and the verifier's `Supplied.root` salt with, never
the partition's. It runs Lean's registration check beside
the Python twin (minutes, with the executable built): no cheaper test holds a window's own committed rows to Lean's R7."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity.circuits.codec import descriptor_chunks, descriptor_id, program_digest
from verity.circuits.evaluation.batch import evaluate_batch
from verity.commitments import docket_digest as DD, domains as DM
from verity.commitments import registered_value as RV
from verity.commitments import row_table as RT
from verity.commitments.gates import embedding as E
from verity.commitments.periods import interface as I
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2, bit_row
from verity.protocols.proof.spot.prover import docket as DK
from verity.protocols.proof.spot.verifier.one_stage import docket as R, registered as G, verifier as V

EMB = I.resolve("EmbeddingRead_v2{V=4,H=16}")
HELD = I.resolve("EmbeddingRead_v4{V=20,H=8}")
EID = descriptor_id(EMB)
RUN = "ab" * 64
DESIGN = {"law": "stratified", "k": 2}
SCHEME = "frame-v3-sha512/hm96-sha512"
WEIGHTS_RECORD = "cd" * 64


class Table:
    """A ``v``-row table of ``h`` words a row under `hm96-sha512/row-seg/v1`, as core registers it: `EMB`'s by default."""

    def __init__(self, v: int = 4, h: int = 16) -> None:
        rng = np.random.default_rng(7)
        self.rows = rng.integers(0, 1 << 16, (v, h), dtype=np.uint16)
        self.salts = [bytes(rng.integers(0, 256, E.SALT_BYTES, dtype=np.uint8)) for _ in range(v)]
        self.dom = RV.domain(WEIGHTS_RECORD, "embed", h, v)
        self.tree = RT.commit(self.dom, [bit_row((r.tolist(), 16)) for r in self.rows], self.salts)
        self.name = self.value = "embed"
        self.root, self.words, self.leaves, self.word_bits = self.tree.root, h, v, 16

    def entry(self) -> dict:
        return {"value": "embed", "root": self.tree.root.hex(), "leaves": self.leaves, "words": self.words,
                "domain": self.dom.domain_id.hex()}

    def held(self) -> dict:
        return {"embed": {"root": self.tree.root.hex(), "domain": self.dom.domain_id.hex()}}

    def path(self, tok: int) -> list[bytes]:
        p = self.tree.path(tok)
        return [p[k:k + 64] for k in range(0, len(p), 64)]


def _window(out, emb=EMB) -> tuple[I.Window, Table, dict]:
    """Three reads of one table through `Window.supply`, `close` and `write`, as `one_linear.py` writes a window, by
    ``emb``, an `EmbeddingRead_v2` or `EmbeddingRead_v4` binding."""
    v, h = emb.bindings["V"], emb.bindings["H"]
    eid, ref = descriptor_id(emb), E.reference_opening if emb.base_id == "EmbeddingRead_v4" else E.reference
    table = Table(v, h)
    reg = I.Registered({"embed": v * h}, (I.Table("embed", "embed", 0, h, v),), (),
                       (I.TableRead(eid, "root", "embed", "root", 0, 3), I.TableRead(eid, "domain", "embed", "domain", 0, 3)),
                       (I.TableOpening(eid, "embed", "tok", ("row", "salt", "path"), "o1", 0, 3),))
    win = I.Window(I.Circuit("embedding", ((emb, 3),), registered=reg), RUN, DESIGN, 2, registered={"embed": table})
    win.supply()
    toks = [v - 1, v - 1, 1 % v]
    got = [ref(v, t, table.tree.root, table.dom.domain_id, table.rows[t].tolist(), table.salts[t], table.path(t))
           for t in toks]

    def digest(d: bytes) -> np.ndarray:
        return np.frombuffer(d, ">u8").astype(np.uint64)

    ins = [np.array(toks, dtype=np.uint32), np.stack([digest(table.tree.root)] * 3), np.stack([digest(table.dom.domain_id)] * 3),
           table.rows[toks], np.stack([np.frombuffer(table.salts[t], ">u8").astype(np.uint64) for t in toks]),
           np.stack([np.stack([digest(s) for s in table.path(t)]) for t in toks])]
    win.record(emb, ins, [np.array([h for h, _ in got], dtype=np.uint16), np.array([ok for _, ok in got], dtype=np.uint16)])
    out.mkdir()
    win.write(win.close(), out, {"table": "salt"}, registrar=DK)
    return win, table, json.loads((out / "registration.json").read_text())


def _pinned(out, emb=EMB) -> tuple[I.Window, Table, dict, dict]:
    """`_window`'s window with its record and check: the record registers the anchors' entries and the table's entry
    `Window.write` registers, the check holds a weights' registration of record pinning the table (`embed`) at its root,
    leaves, words and domain, and names no source for the anchors, whose rows are C's marks'."""
    win, table, window = _window(out, emb)
    prog, query = win.window.program, win.window.query
    assert G.marks(json.loads(b"".join(descriptor_chunks(prog)))) == {
        f"embed/{of}": {"supplied": f"registered.{of}/v1", "args": {"value": "embed"}, "layout": "digest", "bits": 512}
        for of in ("root", "domain")}
    values = {f"embed/{of}": win.registered[f"embed/{of}"] for of in ("root", "domain")}
    written = {e["value"]: e for e in window["registered"]}
    assert written["embed"] == table.entry()
    rec = DK.record(prog, query, DESIGN, SCHEME, window["roots"], {"kind": "served", "row": 7, "run": "r-table"},
                    leaf_layer=window["leaf_layer"], registered=[v.entry() for v in values.values()] + [written["embed"]])
    body = json.dumps({"format": DM.REGISTRATION, "program": WEIGHTS_RECORD, "deployment": {"kind": "test"},
                       "registered": [table.entry()], "previous": None}).encode()
    check = {"program": prog, "query": query, "sampling_design": DESIGN, "scheme": SCHEME,
             "ports": {r["port"]: {"leaves": r["leaves"], "domain": r["domain"]} for r in window["roots"]},
             "leaf_layer": window["leaf_layer"], "registered": {v: x.words for v, x in values.items()},
             "schemas": {v: V2 for v in values}, "registration": {"record": body.hex(), "head": R.DEVELOPMENT}}
    return win, table, rec, check


def _docket(check: dict) -> dict:
    """``check`` as `docket.check` takes it: the registration's record and its digest (`weights_of_record`)."""
    reg = check["registration"]
    body = None if reg is None else bytes.fromhex(reg["record"])
    return {**{k: v for k, v in check.items() if k != "registration"},
            "weights_of_record": None if body is None else (body, DD.digest(json.loads(body)))}


def test_a_tables_anchors_from_cs_marks_and_its_pin_are_the_rows_r7_recomputes_from_the_programs_digest(tmp_path):
    """C marks the table's root and domain anchors with `registered.root/v1` and `registered.domain/v1` of the table's
    registered value, the weights' registration of record pins the table, the record registers the table's entry
    `Window.write` writes, and no scheme or parameter supplies rows for the anchors: R7 recomputes their rows from the
    pin, the rows `interface.supplied` commits, in both verifiers, salted with the program's digest, through a read at a
    binding the verifier holds gated (`HELD`): an accepted run."""
    win, table, rec, check = _pinned(tmp_path / "window", HELD)
    pd = program_digest(check["program"])
    for of in ("root", "domain", "opened"):
        row = win.registered[f"embed/{of}"]
        assert row.program == pd and list(row.salts) == RV.supplied_salts(pd, f"embed/{of}", 1), of
    got = V.check(rec, **check)
    assert got.ok and got.codes == (), got
    assert R.check(rec, **_docket(check)).ok


@pytest.mark.parametrize("emb", ["EmbeddingRead_v2{V=4,H=16}", "EmbeddingRead_v4{V=2,H=1}", "EmbeddingRead_v4{V=4,H=16}"])
def test_a_circuit_reaching_an_opener_outside_a_held_gated_template_is_refused(tmp_path, emb):
    """The same window read through the ungated read, whose rows return whether or not the opening held, or through the
    gated read at a binding the verifier doesn't hold: refused by both verifiers by `R9-ungated-opener` alone, naming
    the read."""
    win, table, rec, check = _pinned(tmp_path / "window", I.resolve(emb))
    got, mine = V.check(rec, **check), R.check(rec, **_docket(check))
    assert got.codes == mine.codes == ("R9-ungated-opener",), (got, mine)
    assert mine.detail["R9-ungated-opener"]["templates"] == [emb], mine


def _entry(rec: dict, value: str, **change) -> dict:
    """``rec`` with its entry of ``value`` changed by ``change``, or left out for none."""
    es = [{**e, **change} if e["value"] == value else e for e in rec["registered"]] if change else [
        e for e in rec["registered"] if e["value"] != value]
    return {**rec, "registered": es}


@pytest.mark.parametrize("name, change, code", [
    ("the table's entry at another root", lambda rec, check: (_entry(rec, "embed", root="11" * 64), check), "R7-pin"),
    ("the table's entry in another domain only", lambda rec, check: (_entry(rec, "embed", domain="11" * 64), check),
     "R7-pin"),
    ("no pin of the table", lambda rec, check: (rec, {**check, "registration": None}), "R7-anchor-unpinned"),
    ("no pin of the table: a mark is no source", lambda rec, check: (rec, {**check, "registration": None}),
     "R7-supplied-source"),
    ("the record leaves the table out", lambda rec, check: (_entry(rec, "embed"), check), "R7-values"),
], ids=lambda x: x if isinstance(x, str) else "")
def test_a_table_input_is_held_to_the_pin_never_the_records_entry(tmp_path, name, change, code):
    """The window's own record with its table's entry off its pin, left out, or with no pin of the table: refused by
    both verifiers by the code."""
    win, table, rec, check = _pinned(tmp_path / "window")
    rec, check = change(rec, check)
    got, mine = V.check(rec, **check), R.check(rec, **_docket(check))
    assert code in got.codes and code in mine.codes, (got, mine)


def test_a_scheme_that_also_supplies_a_marked_anchor_is_refused(tmp_path):
    """A table's anchor that C marks takes its rows from the mark alone: a scheme that also supplies rows for it is refused
    by both verifiers, one source for a supplied value."""
    win, table, rec, check = _pinned(tmp_path / "window")
    both = {**check, "supplied": {"embed/root": [[{"bytes": table.tree.root.hex()}]],
                                  "embed/domain": [[{"bytes": table.dom.domain_id.hex()}]]}}
    for verifier, kw in ((V, both), (R, _docket(both))):
        with pytest.raises(ValueError, match="C's mark names its function, and the scheme supplies rows for it too"):
            verifier.check(rec, **kw)


def test_a_tables_root_and_domain_marks_give_the_rows_the_record_commits():
    """`registered.root/v1` and `registered.domain/v1` with the table's registered name as `value`, read off the table
    the verifier holds of that name (`held`), give the 64 bytes `supplied` commits for the table's root and domain id: as a
    `Digest`'s 512 leaves packed into its 32 words (`row/v2`), the same row, and under the program's salts the same root."""
    table = Table()
    reg = I.Registered({"embed": 64}, (I.Table("embed", "embed", 0, 16, 4),), (),
                       (I.TableRead(EID, "root", "embed", "root", 0, 3), I.TableRead(EID, "domain", "embed", "domain", 0, 3)),
                       (I.TableOpening(EID, "embed", "tok", ("row", "salt", "path"), "o1", 0, 3),))
    prog = "cd" * 64
    committed = I.supplied(reg, {"embed": table.entry()}, prog)
    for of, want in (("root", table.tree.root), ("domain", table.dom.domain_id)):
        name = I.supplied_param(reg.readable["embed"], of)
        mark = {"supplied": f"registered.{of}/v1", "args": {"value": "embed"}, "layout": "digest", "bits": 512}
        (row,) = G.of_mark({}, {}, name, mark)
        leaves = G.row_leaves_of(row, lambda k, n: G._source_bytes({}, name, k, n, [name], [], None, table.held()), 1, True)
        words = G.bit_words(leaves, 1, 16)
        assert leaves == G.digest_leaves(want) and words == I.digest_words(want) == list(committed[name].rows[0]), of
        mine = RV.Registered.commit(prog, name, [words], RV.supplied_salts(prog, name, 1), schema=V2)
        assert mine.root == committed[name].root and mine.entry() == committed[name].entry(), of


def _anchor(held: dict, of: str, value: str) -> list[int]:
    """The leaves `registered.<of>/v1` of ``value`` gives a `Digest` port, read off the tables the verifier holds."""
    mark = {"supplied": f"registered.{of}/v1", "args": {"value": value}, "layout": "digest", "bits": 512}
    (row,) = G.of_mark({}, {}, f"{value}/{of}", mark)
    return G.row_leaves_of(row, lambda k, n: G._source_bytes({}, f"{value}/{of}", k, n, [], [], None, held), 1, True)


def _opened(table: Table, tok: int, cases: list[tuple[list[int], list[int], list[int]]]) -> list[tuple[list[int], int]]:
    """`embedding.opening(4, 16)` at ``tok`` on each ``(root, domain, row)`` leaves, the salt and path ``tok``'s: (h, ok)."""
    args = E.arguments(4, tok, table.tree.root, table.dom.domain_id, table.rows[tok].tolist(), table.salts[tok], table.path(tok))
    got = evaluate_batch(E.opening(4, 16), np.array([[tok]] * len(cases), dtype=object),
                         *(np.array([c[k] for c in cases], dtype=object) for k in range(3)),
                         *(np.array([args[k]] * len(cases), dtype=object) for k in (4, 5)))
    return [([int(w) for w in g[:16]], int(g[16])) for g in np.asarray(got).reshape(len(cases), -1)]


def _anchored() -> tuple[Table, dict, int, list[int]]:
    table = Table()
    rec = {**table.held(), "other": {"root": "11" * 64, "domain": "22" * 64}}
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
def test_an_in_range_token_with_a_wrong_row_reads_the_fill_row_and_ok_0():
    """At an in-range token a row other than the table's, under the supplied anchors, gives the fill row and ``ok`` 0.
    Slow as the test above, whose opening it evaluates."""
    table, rec, tok, row = _anchored()
    row[0] ^= 1
    got = _opened(table, tok, [(_anchor(rec, "root", "embed"), _anchor(rec, "domain", "embed"), row)])
    assert got == [([E.FILL] * 16, 0)]
