"""The round's spot check as a C-Flock statement: `PouwSegCheck_v1{KS}` (verity/core/protocols/pouw/circuit/segment.py, Boolean), one
instance, the drawn thread of the drawn call's tile at the drawn segment j, staged (`Stager`: `class_statement.stage_unit`
split in two) with its own rows; its output one hm96-sha512/row/v2 row under a public salt.

Its rows, each hm96-sha512/row/v2:
    xa0..xa3  a[r]   the thread's row r of A at segment j (32 int8, 256 bits): A's committed leaf (row, j), the committer's salt
    xu        u      U at those 4 rows (4 x 32 int8): the coin's noise, salt public (`public_salt`)
    wb0..wb7  b[c]   the thread's column c's row of B at segment j: B's committed leaf (column, j), the committer's salt
    wf        f      F's sign bits at those 8 rows and segment j (256 bits), public
    prev      prev   the thread's 32 prefix words after segment j - 1 (zero at j = 0): witness, the committer's salt
    st        st     the chain keys of segments j - 1 and j (y then z, 16 words each; derive.seg_keys), public
    rc        rc     the thread's receipt row, its KS words: the call's receipt tree's leaf tile NCT + thread, the committer's salt
    sel       sel    one-hot j (KS bits), public
    o0        ok     1 exactly when receipt words j - 1 and j are those of the committed operands' noised product, public salt
so the challenger holds xa, wb and rc to the trees committed before and after the coin, recomputes xu, wf, st and sel from
the coin and the draw, and holds o0 to content 1.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import commit as CM
import derive
from verity.core.protocols.pouw.circuit import segment as SG
from verity.ml.circuits.defs import bind
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import circuit as FC
from verity_flock import class_statement as CS

SALT_TAG = b"verity/e2e-slice/public-salt/v2\0"
PUBLIC = ("xu", "wf", "st", "sel", "o0")


def in_rows(ks: int) -> list[tuple[str, str, int, list[str]]]:
    return ([(f"xa{r}", V2, 256, [f"a[{r}]"]) for r in range(4)] + [("xu", V2, 1024, ["u"])]
            + [(f"wb{c}", V2, 256, [f"b[{c}]"]) for c in range(8)] + [("wf", V2, 256, ["f"]), ("prev", V2, 1024, ["prev"]),
                                                                       ("st", V2, 1024, ["st"]), ("rc", V2, 32 * ks, ["rc"]),
                                                                       ("sel", V2, ks, ["sel"])])


def definition(ks: int):
    return bind(SG.PouwSegCheck, KS=ks)


def public_salt(sigma: bytes, stage: int, call: int, port: str) -> bytes:
    return hashlib.shake_256(SALT_TAG + sigma + bytes([stage]) + call.to_bytes(4, "big") + port.encode()).digest(CM.SALT)


def _u16(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b, np.uint8).reshape(1, -1).view("<u2")


def _bits(v: np.ndarray) -> np.ndarray:
    return _u16(np.packbits(np.asarray(v, np.uint8).reshape(-1), bitorder="little"))


def public_rows(U: np.ndarray, F: np.ndarray, rseed: int, draw: dict, ks: int) -> dict:
    """xu, wf, st and sel from the coin's noise at the thread's rows (U: its 4 rows, 4 x 32) and columns (F: its 8 columns'
    32 entries at segment j, 8 x 32 of ±1), the receipt key and the draw (``tile`` the call's tile index)."""
    j, t, th = draw["seg"], draw["tile_index"], draw["thread"]
    y, z = derive.seg_keys(rseed, t, ks, [max(j - 1, 0), j], [th])
    st = np.concatenate([y[:, 0], z[:, 0]], axis=1).astype("<u4")
    return {"xu": _u16(np.asarray(U, np.int64).astype(np.int8).view(np.uint8)), "wf": _bits(np.asarray(F) < 0),
            "st": _u16(st.view(np.uint8)), "sel": _bits(np.arange(ks) == j)}


def ok_row() -> np.ndarray:
    return np.ones((1, 1), np.uint16)


class Stager:
    """`class_statement.stage_unit` split in two: the lowering and the composed circuit once (``circuit``, its pin), then a
    round's statement (`stage`) from the lowering's own evaluation and `circuit.write`."""

    def __init__(self, circuit: Path, ks: int):
        self.ks = ks
        self.rows = in_rows(ks)
        self.name = f"unit/e2e-slice-segcheck-ks{ks}"
        CS.cache_slot_circuits()
        self.low, out_words = CS.unit_lowering(definition(ks), self.name, [(r, ps) for r, _s, _b, ps in self.rows])
        if out_words != [1]:
            raise ValueError(f"{self.name}: outputs {out_words} words, not one")
        port = lambda nm: SimpleNamespace(name=nm, dtype="u16")  # noqa: E731
        content = json.dumps([definition(ks).id, [list(r[:3]) + [list(r[3])] for r in self.rows], [[V2, [0]]]]).encode()
        sub = SimpleNamespace(id=self.name, template=SimpleNamespace(id="unit-class"), inputs=[port(r) for r, *_ in self.rows],
                              outputs=[port("o0")],
                              program_digests=lambda statics: {self.name: hashlib.sha256(content).hexdigest()})
        self.iset = SimpleNamespace(subcircuit=sub, name=self.name.replace("/", "-"), content_digest=hashlib.sha256(content).hexdigest())
        self.text = FC.compose(self.low, sub, port_bits=[b for _r, _s, b, _ps in self.rows], v2_out_rows=(("o0",),))
        self.meta = json.loads(self.text.split("\n", 2)[1][5:])
        self.ports = FC._meta_ports(self.meta)
        self.pin = FC.digest(self.text)
        self.circuit = circuit
        circuit.parent.mkdir(parents=True, exist_ok=True)
        circuit.write_text(self.text)

    def stats(self) -> dict:
        c = self.low.layout.counts
        return {"pin": self.pin, "ands": c["ands"], "unit_nnz": c["nnz_a"] + c["nnz_b"], "k_log": self.meta["k_log"],
                "circuit_bytes": len(self.text), "ports": [p.name for p in self.ports], "definition": definition(self.ks).id}

    def stage(self, out: Path, rows: dict, salts: dict, sigma: bytes, stage: int, call: int, salt_key: bytes) -> int:
        """The spot check's inst.bin and pub.bin under ``out`` from its rows (``{port: (1, words) u16}``, every input row);
        ``salts`` the committed rows' salts (xa*, wb*, rc), the public rows' from the coin (`public_salt`), prev's from
        ``salt_key``. Returns the statement's output, ok, as the lowering computes it (the row o0 holds it)."""
        flat = np.concatenate([np.asarray(rows[r], np.uint16).reshape(1, -1) for r, *_ in self.rows], axis=1)
        got, _cut, sat = self.low.evaluate(flat.astype(np.uint64))
        if int(sat) != 1:
            raise AssertionError(f"{self.name}: the unit is unsatisfied on its rows")
        o0 = got.astype(np.uint16)[:, :1]
        pub = {p: [public_salt(sigma, stage, call, p)] for p in PUBLIC}
        all_salts = {**FC.stage_salts(self.iset, 0, 1, self.ports, salt_key), **salts, **pub}
        out.mkdir(parents=True, exist_ok=True)
        FC.write(self.text, self.iset, 0, 1, out / "inst.bin", out / "pub.bin", salt_key, rows=rows, outs={"o0": o0},
                 salts=all_salts)
        return int(o0[0, 0])


def ports(text: str) -> list[str]:
    """The circuit's row ports in META order (the public file's order)."""
    return [p.name for p in FC._meta_ports(json.loads(text.split("\n", 2)[1][5:]))]


def read_public(pub: Path, names: list[str]) -> tuple[dict, dict]:
    """The public file's header and every row's b || c by port: {port: [bc, ...]} (instance-major, ``names`` in META order)."""
    raw = pub.read_bytes()
    head_b, body = raw.split(b"\n", 1)
    head = json.loads(head_b)
    n = head["instances"]
    if len(body) != 128 * n * len(names):
        raise ValueError(f"public file: {len(body)} bytes of b || c, not {n} instances x {len(names)} rows")
    return head, {p: [body[128 * (i * len(names) + q):128 * (i * len(names) + q + 1)] for i in range(n)] for q, p in enumerate(names)}
