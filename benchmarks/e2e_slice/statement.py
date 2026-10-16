"""The round's spot check as a C-Flock statement: `PouwSegCheck_v2{S, FOLD}` (verity/core/protocols/pouw/circuit/segment.py,
Boolean), one instance, the drawn thread of the drawn call's tile at the drawn segment j, staged (`Stager`:
`class_statement.stage_unit` split in two) with its own rows; its output one hm96-sha512/row/v2 row under a public salt.

Its rows, each hm96-sha512/row/v2:
    xa     a      A's tile leaf (64 rows x 32 int8, 16384 bits) holding the thread's 4 rows at segment j: committed, the
                  committer's salt
    asel   asel   the one-hot offset of the thread's rows in it (16 bits), public (`public_salt`)
    xu     u      U at those 4 rows (4 x 32 int8): the coin's noise, public
    wb     b      B's tile leaf holding the thread's 8 columns' rows at segment j: committed
    wsel   bsel   their one-hot offset (8 bits), then j's one-hot position in its window (jsel, `segment.jsel_bits`(S)
           jsel   bits) and whether the window is the first (first): one row, whole u16 words, public
           first
    wf     f      F's sign bits at those 8 rows and segment j (256 bits), public; under the dither
                  (`PouwSegCheckDither_v2`) the same coin bits are b
    p      p      the window's S + 1 absorbed prefix-word sets (T_(w S - 1) .. T_(w S + S - 1)): witness, the committer's salt
    st     st     the window's fold keys (derive.seg_keys at (tile, thread, w, KS / S); duo y then z, XOR-in y), public
    rc     rc     the receipt leaf holding the thread's window word (RW words): committed
    rsel   rsel   the word's one-hot position in it (RW bits), public
    o0     ok     1 exactly when the window's word is that of the committed operands' noised product with segment j's step
                  holding, public
so the challenger holds xa, wb and rc to the trees committed before and after the coin, recomputes the public rows from the
coin and the draw, and holds o0 to content 1.
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

SALT_TAG = b"verity/e2e-slice/public-salt/v3\0"
PUBLIC = ("asel", "xu", "wsel", "wf", "st", "rsel", "o0")


def in_rows(S: int, fold: str) -> list[tuple[str, str, int, list[str]]]:
    fb = SG.FOLDS[fold][2]
    return [("xa", V2, 16384, ["a"]), ("asel", V2, 16, ["asel"]), ("xu", V2, 1024, ["u"]), ("wb", V2, 16384, ["b"]),
            ("wsel", V2, 9 + SG.jsel_bits(S), ["bsel", "jsel", "first"]), ("wf", V2, 256, ["f"]),
            ("p", V2, 1024 * (S + 1), ["p"]), ("st", V2, fb, ["st"]), ("rc", V2, 32 * SG.RW, ["rc"]),
            ("rsel", V2, SG.RW, ["rsel"])]


def definition(S: int, fold: str = "duo", dither: bool = False):
    """`PouwSegCheck_v2{S, FOLD}`, or under decision 29's dither `PouwSegCheckDither_v2{S, FOLD}` (wf's bits are then b)."""
    return bind(SG.PouwSegCheckDither if dither else SG.PouwSegCheck, S=S, FOLD=fold)


def public_salt(sigma: bytes, stage: int, call: int, port: str) -> bytes:
    return hashlib.shake_256(SALT_TAG + sigma + bytes([stage]) + call.to_bytes(4, "big") + port.encode()).digest(CM.SALT)


def _u16(b: np.ndarray) -> np.ndarray:
    b = np.ascontiguousarray(b, np.uint8).reshape(-1)
    return np.concatenate([b, np.zeros(len(b) % 2, np.uint8)]).reshape(1, -1).view("<u2")


def _bits(v: np.ndarray) -> np.ndarray:
    return _u16(np.packbits(np.asarray(v, np.uint8).reshape(-1), bitorder="little"))


def thread_place(th: int, S: int, ks: int, tile: int, j: int) -> dict:
    """Where thread ``th`` of ``tile`` reads at segment j: its A and B leaves' offset indices (asel, bsel), its window w, j's
    place in it, and its receipt word's leaf and place."""
    warp, lane = th // 32, th % 32
    wm, wn, g, q = warp // 4, warp % 4, lane >> 2, lane & 3
    nw = ks // S
    word = (tile * nw + j // S) * derive.NCT + th
    return {"ia": 8 * (wm % 2) + g, "ib": 4 * (wn % 2) + q, "a_block": wm // 2, "b_block": wn // 2, "w": j // S, "i": j % S,
            "nw": nw, "r_leaf": word // SG.RW, "r_word": word % SG.RW}


def public_rows(U: np.ndarray, F: np.ndarray, rseed: int, draw: dict, S: int, ks: int, fold: str) -> dict:
    """The public rows from the coin's noise at the thread's rows (U: its 4 rows, 4 x 32) and columns (F: its 8 columns'
    32 entries at segment j, 8 x 32 of ±1), the receipt key and the draw (``tile_index`` the call's tile index)."""
    j, t, th = draw["seg"], draw["tile_index"], draw["thread"]
    pl = thread_place(th, S, ks, t, j)
    st = np.array(SG.fold_keys(fold, rseed, t, th, pl["w"], pl["nw"]), "<u4")
    return {"asel": _bits(np.arange(16) == pl["ia"]), "xu": _u16(np.asarray(U, np.int64).astype(np.int8).view(np.uint8)),
            "wsel": _bits(np.concatenate([np.arange(8) == pl["ib"], np.arange(SG.jsel_bits(S)) == pl["i"], [pl["w"] == 0]])),
            "wf": _bits(np.asarray(F) < 0), "st": _u16(st.view(np.uint8)), "rsel": _bits(np.arange(SG.RW) == pl["r_word"])}


def ok_row() -> np.ndarray:
    return np.ones((1, 1), np.uint16)


class Stager:
    """`class_statement.stage_unit` split in two: the lowering and the composed circuit once (``circuit``, its pin), then a
    round's statement (`stage`) from the lowering's own evaluation and `circuit.write`."""

    def __init__(self, circuit: Path, ks: int, dither: bool = False, S: int = 1, fold: str = "duo"):
        if ks % S:
            raise ValueError(f"{ks} segments: not whole windows of {S}")
        self.ks, self.dither, self.S, self.fold = ks, dither, S, fold
        self.D = definition(S, fold, dither)
        self.rows = in_rows(S, fold)
        self.name = f"unit/e2e-slice-wincheck{'-dither' if dither else ''}-{fold}-s{S}"
        CS.cache_slot_circuits()
        self.low, out_words = CS.unit_lowering(self.D, self.name, [(r, ps) for r, _s, _b, ps in self.rows])
        if out_words != [1]:
            raise ValueError(f"{self.name}: outputs {out_words} words, not one")
        port = lambda nm: SimpleNamespace(name=nm, dtype="u16")  # noqa: E731
        content = json.dumps([self.D.id, [list(r[:3]) + [list(r[3])] for r in self.rows], [[V2, [0]]]]).encode()
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
                "circuit_bytes": len(self.text), "ports": [p.name for p in self.ports], "definition": self.D.id}

    def stage(self, out: Path, rows: dict, salts: dict, sigma: bytes, stage: int, call: int, salt_key: bytes) -> int:
        """The spot check's inst.bin and pub.bin under ``out`` from its rows (``{port: (1, words) u16}``, every input row);
        ``salts`` the committed rows' salts (xa, wb, rc), the public rows' from the coin (`public_salt`), p's from
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
