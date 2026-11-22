"""A table's supplied rows (`Window.supply`: its root, its domain id and the opening verdict's 1) are the rows R7 recomputes
from the program's digest: a docket whose `supplied` names the table's root and domain id by `bytes` accepts the window's
own rows in both verifiers, and each row's public salts are the program's (`registered_value.supplied_salts`), the digest
`Window.give` and the verifier's `Supplied.root` salt with, never the partition's. It runs Lean's registration check beside
the Python twin (minutes, with the executable built): no cheaper test holds a window's own committed rows to Lean's R7."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from verity.circuits.codec import descriptor_id, program_digest
from verity.commitments import registered_value as RV
from verity.commitments import row_table as RT
from verity.commitments.gates import embedding as E
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.periods import interface as I
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2, bit_row
from verity.protocols.proof.spot.prover import docket as DK
from verity.protocols.proof.spot.verifier.one_stage import docket as R, verifier as V

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
