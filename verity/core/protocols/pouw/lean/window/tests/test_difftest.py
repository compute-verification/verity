"""The window's Lean against `verity.core.protocols.pouw.window`, on the shared harness (`verity_difftest`): one
`Port` per ported function of `window.py`, served by `pouw-window-difftest` (`PouwWindow/Difftest.lean`).

The inputs are drawn by the generators below from a Hypothesis-controlled `random.Random` (`st.randoms`), so a run is
deterministic and shrinks. The grid's numbers run over bits 0 … 6, n up to 2^60 and the integers at or below 0; K′'s
parameters over realistic ε and δ (down to 2^-40), grid-derived and random work up to n = 2^60, cases where (1 − p)^K = δ
exactly, and every way out of its domain.

The advice's functions run on entries of the layout and every way off it (`_layout_entry`), rows and words mutated off
the row format, records that are their entries' and each way they are not, small side domains of up to 2^8 positions
(the side trees' code is the same at 2^13 and 2^17, which the pinned window exercises), and windows built honestly
(`_window`: one to four calls, the filler, empty entries, records and side indices) and then broken in one of the ways
`window_problem` names. Each problem-returning function must reach every one of its reasons (`REACHES`). The pinned
window of `tests/vectors/window.json` (`window_vectors.py`) and the windows of served runs (`SERVED`, from the evidence
store) are fixed vectors on top of the drawn ones.
"""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import pytest
import verity_difftest as D
from hypothesis import strategies as st

from verity.core.protocols.pouw import window as Wd
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

LEAN = Path(__file__).resolve().parents[1]
EXE = "pouw-window-difftest"


def _frac(q: Fraction) -> list[int]:
    return [q.numerator, q.denominator]


def _q(v: list[int]) -> Fraction:
    return Fraction(v[0], v[1])


def _work(v: dict) -> Wd.Work:
    return Wd.Work(v["level"], v["credited"], v["n"])


# ---- the grid -----------------------------------------------------------------------------------------------------------

def _number(rng: random.Random, bits: int) -> int:
    r = rng.random()
    if r < 0.25:
        return rng.randint(0, 300)
    if r < 0.45:
        return Wd.point(rng.randint(0, 400), bits)
    if r < 0.55:
        return (1 << rng.randint(0, 60)) + rng.choice((-1, 0, 1))
    if r < 0.62:
        return -rng.randint(0, 1 << rng.randint(1, 62))
    return rng.getrandbits(rng.randint(1, 60))


def grid_n(rng: random.Random) -> dict:
    bits = rng.randint(0, 6)
    return {"bits": bits, "n": _number(rng, bits)}


def grid_ell(rng: random.Random) -> dict:
    return {"bits": rng.randint(0, 6), "ell": rng.choice((rng.randint(0, 60), rng.randint(0, 2000), -rng.randint(1, 5)))}


def levels_in(rng: random.Random) -> dict:
    bits = rng.randint(0, 6)
    lo, hi = sorted((_number(rng, bits), _number(rng, bits)))
    if rng.random() < 0.2:
        lo, hi = hi, lo
    return {"bits": bits, "lo": lo, "hi": hi}


# ---- the draw -----------------------------------------------------------------------------------------------------------

def _eps(rng: random.Random) -> Fraction:
    return rng.choice((Fraction(1, 2), Fraction(1, 2), Fraction(1, 1000), Fraction(1), Fraction(1, 3), Fraction(9, 10),
                       Fraction(rng.randint(1, 1000), 1000), Fraction(rng.randint(1, 7), 7)))


def _delta(rng: random.Random) -> Fraction:
    return rng.choice((Fraction(1, 10), Fraction(1, 2 ** 40), Fraction(1, 2 ** rng.randint(1, 40)), Fraction(1, 3),
                       Fraction(1, 10 ** rng.randint(1, 12)), Fraction(rng.randint(1, 999), 1000)))


def _draw_params(rng: random.Random) -> tuple[Fraction, Fraction, int, int, int]:
    eps, delta = _eps(rng), _delta(rng)
    r = rng.random()
    if r < 0.35:
        bits = rng.choice((3, 3, rng.randint(0, 6)))
        w = Wd.bucket(rng.choice((rng.randint(1, 5000), rng.getrandbits(rng.randint(1, 40)) + 1)), bits)
        tile = rng.choice((1, 4096, rng.randint(1, 1 << 20)))
        return eps, delta, w.credited * tile, w.n, tile
    if r < 0.55:
        n, w = rng.getrandbits(rng.randint(1, 60)) + 1, rng.randint(1, 1 << rng.randint(0, 20))
        return eps, delta, rng.randint(1, n * w), n, w
    if r < 0.65:
        n, w = rng.getrandbits(rng.randint(20, 60)) + 1, rng.randint(1, 1 << 10)
        return eps, delta, rng.randint(1, 1 << rng.randint(0, 20)), n, w
    if r < 0.8:
        # (1 − p)^K = δ exactly: p = ε·credited/(n·w), δ its complement's K-th power
        n, w = rng.randint(1, 40), 1
        credited = rng.randint(1, n)
        p = eps * credited / n
        if p == 1:
            return eps, delta, credited, n, w
        return eps, (1 - p) ** rng.randint(1, 30), credited, n, w
    n, w = rng.randint(1, 1 << 30), rng.randint(1, 100)
    bad = rng.choice(("eps0", "eps>1", "eps<0", "delta0", "delta1", "delta>1", "delta<0", "n0", "n<0", "w0", "w<0",
                      "credited0", "credited<0", "credited>nw"))
    credited = rng.randint(1, n * w)
    return (Fraction(0) if bad == "eps0" else Fraction(3, 2) if bad == "eps>1" else Fraction(-1, 2) if bad == "eps<0" else eps,
            Fraction(0) if bad == "delta0" else Fraction(1) if bad == "delta1" else Fraction(5, 4) if bad == "delta>1"
            else Fraction(-1, 10) if bad == "delta<0" else delta,
            0 if bad == "credited0" else -credited if bad == "credited<0" else n * w + rng.randint(1, 9)
            if bad == "credited>nw" else credited,
            0 if bad == "n0" else -n if bad == "n<0" else n, 0 if bad == "w0" else -w if bad == "w<0" else w)


def k_prime_in(rng: random.Random) -> dict:
    eps, delta, credited, n, w = _draw_params(rng)
    return {"eps": _frac(eps), "delta": _frac(delta), "credited": credited, "n": n, "w_max": w}


def _some_work(rng: random.Random, bits: int) -> Wd.Work:
    return Wd.Work.at(rng.randint(0, 400), bits) if rng.random() < 0.6 else Wd.bucket(rng.randint(1, 1 << 40), bits)


def sampling_in(rng: random.Random) -> dict:
    eps, delta, bits = _eps(rng), _delta(rng), rng.randint(0, 6)
    if rng.random() < 0.15:
        eps = rng.choice((Fraction(0), Fraction(2)))
    return {"eps": _frac(eps), "delta": _frac(delta), "work": asdict(_some_work(rng, bits)),
            "tile_work": rng.choice((1, 4096, rng.randint(1, 1 << 20), 0, -3)), "bits": bits}


def work_to_json_in(rng: random.Random) -> dict:
    return {"work": asdict(_some_work(rng, rng.randint(0, 6))), "tile_work": rng.choice((1, 4096, rng.randint(-5, 1 << 20)))}


# ---- the advice --------------------------------------------------------------------------------------------------------

VECTORS = json.loads((Path(Wd.__file__).parent / "tests" / "vectors" / "window.json").read_text())


def _digest(rng: random.Random) -> bytes:
    return rng.choice((bytes(64), rng.randbytes(64), rng.randbytes(64)))


def _call(rng: random.Random, lo: int = 0, tile0: int = 0) -> Wd.Entry:
    m = Wd.ROW_STEP * rng.randint(1, rng.choice((4, 8, 1 << 9)))
    cols = rng.randint(1, rng.choice((3, 1 << 11)))
    n_a, n_b = rng.randint(1, min(m, 40)), rng.randint(1, min(Wd.TILE * cols, 40))
    return Wd.Entry("call", lo, lo + n_a * n_b, n_a, n_b, _digest(rng), _digest(rng), _digest(rng), _digest(rng), tile0, cols,
                    m)


def _filler(rng: random.Random, lo: int = 0, size: int | None = None) -> Wd.Entry:
    size = rng.randint(1, 50) if size is None else size
    n_a = rng.randint(1, min(size, 1 << 13))
    n_b = -(-size // n_a)
    return Wd.Entry("filler", lo, lo + size, n_a, n_b, _digest(rng), _digest(rng))


def _layout_entry(rng: random.Random) -> Wd.Entry:
    """An entry of the layout, or one off it in one of the ways `layout_problem` names (all of them u64s)."""
    e = rng.choice((_call, _call, _filler, lambda r: Wd.Entry("none")))(rng)
    if e.kind != "none" and rng.random() < 0.3:
        lo = rng.choice((rng.randint(0, 1 << 20), (1 << 40) - rng.randint(1, 1000)))
        e = Wd.Entry(e.kind, lo, lo + e.hi, e.n_a, e.n_b, e.root_a, e.root_b, e.root_h3, e.record, e.tile0, e.cols, e.m)
    bad = rng.choice(("none",) * 6 + ("range", "zero", "side", "h3", "record", "pass", "m", "cols", "tile0"))
    big = rng.choice(((1 << 40) + rng.randint(0, 9), (1 << 64) - 1))
    d = e.__dict__ | {}
    if bad == "range":
        d[rng.choice(("lo", "hi"))] = big
    elif bad == "zero" and e.kind == "none":
        d[rng.choice(("lo", "hi", "n_a", "n_b", "tile0", "cols", "m"))] = rng.randint(1, 9)
    elif bad == "zero":
        d = Wd.Entry("none", root_a=rng.randbytes(64)).__dict__ | {}
    elif bad == "side":
        side = rng.choice(("a", "b"))
        d[f"n_{side}"] = rng.choice((0, (1 << Wd.DEPTHS[side]) + rng.randint(1, 5), big))
    elif bad == "h3" and e.kind == "filler":
        d["root_h3"] = rng.randbytes(64)
    elif bad == "record" and e.kind == "filler":
        d["record"] = rng.randbytes(64)
    elif bad == "pass" and e.kind == "filler":
        d[rng.choice(("tile0", "cols", "m"))] = rng.randint(1, 9)
    elif bad == "m" and e.kind == "call":
        d["m"] = rng.choice((0, e.n_a - 1 if e.n_a > 1 else 0, e.m + rng.randint(1, 15), (1 << 13) + 16, big))
    elif bad == "cols" and e.kind == "call":
        d["cols"] = rng.choice((0, (1 << 11) + 1, max(0, -(-e.n_b // 64) - 1), big))
    elif bad == "tile0" and e.kind == "call":
        d["tile0"] = (1 << 40) - rng.randint(0, 3)
    return Wd.Entry(**d)


def _row(rng: random.Random) -> str:
    return _layout_entry(rng).row().hex()


def _mutated_row(rng: random.Random) -> bytes:
    row = bytearray(_layout_entry(rng).row())
    bad = rng.choice(("none", "none", "length", "kind", "pad", "pad2", "random"))
    if bad == "length":
        row = row[:rng.randint(0, len(row) - 1)] if rng.random() < 0.5 else row + rng.randbytes(rng.randint(1, 3))
    elif bad == "kind":
        row[0:8] = rng.choice((3, 255, 1 << 40, (1 << 64) - 1)).to_bytes(8, "little")
    elif bad == "pad":
        row[rng.randint(40, 47)] = rng.randint(1, 255)
    elif bad == "pad2":
        row[rng.randint(264, 271)] = rng.randint(1, 255)
    elif bad == "random":
        row = bytearray(rng.randbytes(Wd.ENTRY_BYTES))
        row[0:8] = rng.randint(0, 2).to_bytes(8, "little")
    return bytes(row)


def entry_json_in(rng: random.Random) -> dict:
    x = _layout_entry(rng).to_json()
    bad = rng.choice(("none",) * 5 + ("kind", "negative", "u64", "digest", "odd", "nothex"))
    if bad == "kind":
        x["kind"] = rng.choice(("Call", "", "fill"))
    elif bad == "negative":
        x[rng.choice(("lo", "hi", "n_a", "n_b", "tile0", "cols", "m"))] = -rng.randint(1, 1 << 40)
    elif bad == "u64":
        x[rng.choice(("lo", "hi", "n_a", "n_b", "tile0", "cols", "m"))] = (1 << 64) + rng.randint(0, 9)
    elif bad == "digest":
        x[rng.choice(("root_a", "root_b", "root_h3", "record"))] = rng.randbytes(rng.choice((0, 32, 63, 65))).hex()
    elif bad == "odd":
        x[rng.choice(("root_a", "record"))] += "a"
    elif bad == "nothex":
        x["root_b"] = "zz" + x["root_b"][2:]
    return {"entry": x}


def row_in(rng: random.Random) -> dict:
    return {"row": _mutated_row(rng).hex()}


def words_in(rng: random.Random) -> dict:
    words = list(_layout_entry(rng).words())
    bad = rng.choice(("none", "none", "none", "length", "big", "negative"))
    if bad == "length":
        words = words[:-1] if rng.random() < 0.5 else words + [0]
    elif bad == "big":
        words[rng.randrange(len(words))] = 1 << 16
    elif bad == "negative":
        words[rng.randrange(len(words))] = -1
    elif rng.random() < 0.3:
        words[0], words[20] = rng.randint(0, 3), rng.randint(0, 1 << 16 - 1)
    return {"words": words}


def layout_in(rng: random.Random) -> dict:
    return {"row": _row(rng)}


def bc_in(rng: random.Random) -> dict:
    return {"row": _row(rng), "salt": rng.randbytes(rng.choice((192, 192, 192, 0, 191, 193, 64))).hex()}


def _domain(rng: random.Random) -> dict:
    return {"binding": rng.randbytes(64).hex(), "count": 1 << rng.randint(0, 8)}


def _dom(x: dict) -> CommitmentDomain:
    return CommitmentDomain(bytes.fromhex(x["binding"]), Wd.OWNER, RangeIndexedDomain(0, x["count"]), hash="sha512")


def leaf_in(rng: random.Random) -> dict:
    d = _domain(rng)
    if rng.random() < 0.1:
        d["binding"] = rng.randbytes(rng.choice((0, 10, 63))).hex()
    return {"domain": d, "e": rng.randint(0, d["count"]), "bc": rng.randbytes(rng.choice((128,) * 6 + (0, 64, 127, 129))).hex()}


def tile_in_in(rng: random.Random) -> dict:
    e = _layout_entry(rng)
    t = rng.choice((rng.randint(min(e.lo, e.hi) - 2, max(e.lo, e.hi) + 2), e.lo, e.hi - 1, e.hi, -rng.randint(1, 9)))
    return {"row": e.row().hex(), "t": t}


def tile_at_in(rng: random.Random) -> dict:
    if rng.random() < 0.5:
        w = _window(rng, broken=False)
        rows = w["rows"]
        t = rng.randint(-2, w["work"]["n"] + 2)
    else:
        rows = [_row(rng) for _ in range(rng.randint(0, 5))]
        t = rng.randint(-2, 200)
    return {"rows": rows, "t": t}


def tile_read_in(rng: random.Random) -> dict:
    e = _layout_entry(rng)
    i = rng.choice((rng.randint(0, max(e.m - 1, 0)), e.m - 1, e.m, -1, rng.randint(0, 1 << 14)))
    j = rng.choice((rng.randint(0, max(64 * e.cols - 1, 0)), 64 * e.cols - 1, 64 * e.cols, -1))
    return {"row": e.row().hex(), "i": i, "j": j}


_TEXT = ("call", "keep", "\"", "\\", "\n", "\t", "\x00", "\x1f", "\x7f", "é", "\u00a0", "\u2028", "😀", "日本", "~", " ")


def _json(rng: random.Random, depth: int = 0) -> Any:
    r = rng.random()
    if depth > 2 or r < 0.3:
        return rng.choice((None, True, False, rng.randint(-5, 5), rng.getrandbits(70) - (1 << 69),
                           "".join(rng.choice(_TEXT) for _ in range(rng.randint(0, 4)))))
    if r < 0.6:
        return [_json(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return {"".join(rng.choice(_TEXT) for _ in range(rng.randint(0, 3))): _json(rng, depth + 1) for _ in range(rng.randint(0, 3))}


def record_digest_in(rng: random.Random) -> dict:
    return {"record": _json(rng) if rng.random() < 0.6 else _record(rng, _call(rng))}


def _record(rng: random.Random, e: Wd.Entry) -> dict:
    n = rng.randint(max(1, 64 * (e.cols - 1) + 1), 64 * e.cols)
    n = max(n, e.n_b)
    return {"call": rng.randint(0, 9), "shape": [e.m, rng.choice((32, 4096)), n], "tile0": e.tile0,
            "keep": {"a": sorted(rng.sample(range(e.m), e.n_a)), "b": sorted(rng.sample(range(n), e.n_b))}}


def _bound(e: Wd.Entry, record: Any) -> tuple[Wd.Entry, Any]:
    return Wd.Entry(**(e.__dict__ | {"record": Wd.record_digest(record)})), record


def record_problem_in(rng: random.Random) -> dict:
    e = _call(rng)
    record = _record(rng, e)
    bad = rng.choice(("none",) * 4 + ("digest", "shape", "notlist", "short", "tile0", "keep", "cols", "m", "n", "count",
                                      "order", "dup", "range", "neg", "bool", "type", "keepstr", "keepmissing", "boolshape"))
    if bad == "digest":
        return {"row": e.row().hex(), "record": record}
    if bad == "shape":
        del record[rng.choice(("shape", "tile0", "keep"))]
    elif bad == "notlist":
        record["shape"] = rng.choice((5, "abc", None))
    elif bad == "short":
        record["shape"] = record["shape"][:2]
    elif bad == "tile0":
        record["tile0"] = rng.choice((e.tile0 + 1, str(e.tile0), None))
    elif bad == "cols":
        record["shape"][2] += 64
    elif bad == "m":
        record["shape"][0] = rng.choice((e.m + 16, str(e.m)))
    elif bad == "n":
        record["shape"][2] = rng.choice(("64", None, [1]))
    elif bad == "count":
        side = rng.choice("ab")
        record["keep"][side] = record["keep"][side][:-1]
    elif bad == "order" and e.n_a > 1:
        record["keep"]["a"] = record["keep"]["a"][::-1]
    elif bad == "dup" and e.n_b > 1:
        record["keep"]["b"][1] = record["keep"]["b"][0]
    elif bad == "range":
        side = rng.choice("ab")
        record["keep"][side][-1] = record["shape"][0 if side == "a" else 2] + rng.randint(0, 3)
    elif bad == "neg":
        record["keep"]["a"][0] = -1
    elif bad == "bool":
        record["keep"]["b"][0] = rng.choice((True, False))
    elif bad == "type":
        record["keep"]["a"][0] = rng.choice(("0", None, [0]))
    elif bad == "keepstr":
        record["keep"] = rng.choice(("ab", ["a", "b"], 3))
    elif bad == "keepmissing":
        del record["keep"][rng.choice("ab")]
    elif bad == "boolshape":
        record["shape"][2] = True
    e, record = _bound(e, record)
    return {"row": e.row().hex(), "record": record}


def side_row_in(rng: random.Random) -> dict:
    words = [rng.randint(0, 65535) for _ in range(Wd.SIDE_INDEX_WORDS * rng.randint(0, 4) + rng.choice((0,) * 5 + (1, 7)))]
    index = rng.choice((rng.randint(0, 1 << 17), rng.randint(0, (1 << 32) - 1), (1 << 32) - 1, 1 << 32, -1))
    return {"words": words, "index": index}


def side_index_in(rng: random.Random) -> dict:
    row = Wd.side_row([rng.randint(0, 65535) for _ in range(8 * rng.randint(0, 3))], rng.randint(0, (1 << 32) - 1))
    bad = rng.choice(("none",) * 4 + ("short", "pad", "big", "negative"))
    if bad == "short":
        row = row[-rng.randint(0, 8):]
    elif bad == "pad":
        row[-rng.randint(1, 6)] = rng.randint(1, 65535)
    elif bad == "big":
        row[-rng.choice((7, 8))] = 1 << 16
    elif bad == "negative":
        row[-8] = -1
    return {"row": row}


def side_domain_in(rng: random.Random) -> dict:
    return {"window": rng.randbytes(rng.choice((64,) * 8 + (0, 32, 65))).hex(), "side": rng.choice(("a", "b", "a", "b", "c", ""))}


def pad_node_in(rng: random.Random) -> dict:
    return {"domain": _domain(rng), "level": rng.randint(0, 6), "index": rng.randint(0, 40)}


def side_levels_in(rng: random.Random) -> dict:
    d = _domain(rng)
    if rng.random() < 0.1:
        d["count"] = rng.choice((0, 3, 12))
    n = rng.choice((rng.randint(1, d["count"]) if d["count"] else 1, d["count"], d["count"] + 1, 0, 1))
    return {"domain": d, "leaves": [rng.randbytes(64).hex() for _ in range(n)]}


def side_path_in(rng: random.Random) -> dict:
    x = side_levels_in(rng)
    return x | {"index": rng.randint(0, max(len(x["leaves"]) + 1, x["domain"]["count"]))}


def _window(rng: random.Random, broken: bool = True) -> dict:
    """An honest window (rows, work, bound, root_h3, bits, records, indices) and, when `broken`, one break of it."""
    bits = rng.choice((3, 3, 3, rng.randint(0, 6)))
    root_h3 = rng.randbytes(64)
    entries, records, indices, at, in_pass = [], [], [], 0, rng.randint(0, 5)
    for _ in range(rng.randint(1, 4)):
        e = _call(rng, at, in_pass)
        e = Wd.Entry(**(e.__dict__ | {"root_h3": root_h3}))
        e, record = _bound(e, _record(rng, e))
        entries.append(e)
        records.append(record)
        indices.append({s: list(record["keep"][s]) for s in "ab"})
        at, in_pass = e.hi, e.tile0 + Wd.pass_tiles(e) + rng.choice((0, 0, rng.randint(1, 5)))
    work = Wd.bucket(at, bits)
    filler = _filler(rng, at, work.n - at)
    entries.append(filler)
    indices.append({"a": [0] * filler.n_a, "b": [0] * filler.n_b})
    entries += [Wd.Entry("none")] * rng.randint(0, 3)
    records += [None] * (len(entries) - len(records))
    indices += [rng.choice((None, {"a": [], "b": []}))] * (len(entries) - len(indices))
    rows = [e.row() for e in entries]
    calls, bound = len(records) - records.count(None), len(entries)
    out = {"rows": rows, "work": asdict(work), "bound": bound, "root_h3": root_h3.hex(), "bits": bits,
           "records": records if rng.random() < 0.7 else None, "indices": None}
    if out["records"] is not None and rng.random() < 0.7:
        out["indices"] = indices
    bad = rng.choice(("none",) * 3 + ("bound", "row", "layout", "kinds", "nofiller", "gap", "empty", "size", "h3", "overlap",
                                      "fillerbig", "end", "level", "records", "pastcalls", "record", "indices", "fillerindex",
                                      "noindices", "h3len", "indicesonly", "shortindices")) if broken else "none"
    e = rng.randrange(calls)
    def put(i: int, **kw) -> None:
        rows[i] = Wd.Entry(**(Wd.Entry.from_row(rows[i]).__dict__ | kw)).row()
    if bad == "bound":
        out["bound"] = bound + rng.choice((-1, 1))
    elif bad == "row":
        rows[rng.randrange(bound)] = _mutated_row(rng) if rng.random() < 0.5 else rows[0][:-1]
    elif bad == "layout":
        rows[rng.randrange(bound)] = _layout_entry(rng).row()
    elif bad == "kinds":
        i, j = rng.sample(range(bound), 2) if bound > 1 else (0, 0)
        rows[i], rows[j] = rows[j], rows[i]
    elif bad == "nofiller":
        rows[calls] = Wd.Entry("none").row()
    elif bad == "gap":
        put(e, lo=entries[e].lo + 1, hi=entries[e].hi + 1)
    elif bad == "empty":
        put(e, hi=entries[e].lo)
    elif bad == "size":
        put(e, n_b=entries[e].n_b + 1)
    elif bad == "h3":
        put(e, root_h3=rng.randbytes(64))
    elif bad == "overlap" and calls > 1:
        put(calls - 1, tile0=max(0, entries[calls - 2].tile0 + Wd.pass_tiles(entries[calls - 2]) - 1))
    elif bad == "fillerbig":
        put(calls, n_a=1, n_b=max(1, filler.hi - filler.lo - 1))
    elif bad == "end":
        out["work"] = asdict(Wd.Work(work.level, work.credited, work.n + rng.choice((-1, 1))))
    elif bad == "level":
        lv = Wd.Work.at(work.level + 1, bits)
        out["work"] = asdict(Wd.Work(lv.level, lv.credited, work.n))
    elif bad == "records" and out["records"]:
        out["records"] = out["records"][:-1]
    elif bad == "pastcalls" and out["records"]:
        out["records"] = out["records"][:calls] + [records[0]] + out["records"][calls + 1:]
    elif bad == "record" and out["records"]:
        out["records"] = list(out["records"])
        out["records"][e] = dict(records[e]) | {"call": -1}
    elif bad == "indices" and out["indices"]:
        side = rng.choice("ab")
        out["indices"][e] = dict(indices[e]) | {side: indices[e][side][:-1] + [indices[e][side][-1] + 1]}
    elif bad == "fillerindex" and out["indices"]:
        out["indices"][calls] = {"a": [1] * filler.n_a, "b": [0] * filler.n_b}
    elif bad == "noindices" and out["indices"]:
        out["indices"][rng.randint(0, calls)] = None
    elif bad == "shortindices" and out["indices"]:
        out["indices"] = out["indices"][:rng.randint(0, calls)]
    elif bad == "h3len":
        out["root_h3"] = rng.randbytes(rng.choice((0, 32))).hex()
    elif bad == "indicesonly":
        out["records"], out["indices"] = None, indices
    out["rows"] = [r.hex() for r in rows]
    return out


def window_in(rng: random.Random) -> dict:
    return _window(rng)


def session_cap_in(rng: random.Random) -> dict:
    return {"k_log": rng.choice((rng.randint(1, 32), rng.randint(-2, 40))), "per_block": rng.choice((1, 1, 2, 4, 0, -1))}


def sessions_in(rng: random.Random) -> dict:
    return {"k": rng.choice((rng.randint(1, 40_000), 0, -3)), "cap": rng.choice((rng.randint(1, 1 << 16), 1, 0, -1))}


def _registration(rng: random.Random) -> tuple[dict, list[int]]:
    n = rng.choice((rng.randint(1, 200), 1 << 40))
    k = rng.randint(1, min(n, 12))
    drawn = sorted(rng.sample(range(n), k)) if n < 1 << 20 else sorted(set(rng.randrange(n) for _ in range(k)))
    k = len(drawn)
    roots = [{"port": "entries", "root": rng.randbytes(64).hex(), "leaves": 4, "domain": rng.randbytes(64).hex()}]
    if rng.random() < 0.3:
        roots.append({"port": "h3", "root": rng.randbytes(64).hex(), "domain": rng.randbytes(64).hex()})
    return {"population": n, "law": {"law": "subset", "k": k}, "roots": roots}, drawn


def instance_ts_in(rng: random.Random) -> dict:
    reg, drawn = _registration(rng)
    k = reg["law"]["k"]
    cap = rng.choice((1, 2, 3, 8, rng.randint(1, 16)))
    sigma: Any = rng.randint(0, -(-k // cap) - 1)
    bad = rng.choice(("none",) * 5 + ("population", "law", "k", "drawn", "sigma", "cap"))
    if bad == "population":
        reg["population"] = rng.choice((0, -1, (1 << 40) + 1, True, "9", None))
    elif bad == "law":
        reg["law"] = rng.choice(({"law": "bernoulli", "k": k}, None, "subset", {"k": k}))
    elif bad == "k":
        reg["law"]["k"] = rng.choice((0, -1, True, str(k)))
    elif bad == "drawn":
        how = rng.choice(("short", "unsorted", "dup", "range", "bool", "neg"))
        if how == "short":
            drawn = drawn[:-1]
        elif how == "unsorted" and k > 1:
            drawn = drawn[::-1]
        elif how == "dup" and k > 1:
            drawn[1] = drawn[0]
        elif how == "range":
            drawn[-1] = reg["population"] if isinstance(reg["population"], int) else 10 ** 6
        elif how == "bool":
            drawn[0] = True
        elif how == "neg":
            drawn[0] = -1
    elif bad == "sigma":
        sigma = rng.choice((-1, -(-k // cap), True, False, "0", None))
    elif bad == "cap":
        cap = rng.choice((0, -2))
    return {"registration": reg, "drawn": drawn, "sigma": sigma, "cap": cap}


def public_inputs_in(rng: random.Random) -> dict:
    x = instance_ts_in(rng) | {"domain_h3": rng.randbytes(rng.choice((64,) * 8 + (0, 32))).hex()}
    roots = x["registration"]["roots"]
    bad = rng.choice(("none",) * 6 + ("two", "zero", "short", "odd", "notdict", "nonroot"))
    if bad == "two":
        roots.append(dict(roots[0]))
    elif bad == "zero":
        roots[0]["port"] = "Entries"
    elif bad == "short":
        roots[0][rng.choice(("root", "domain"))] = rng.randbytes(32).hex()
    elif bad == "odd":
        roots[0]["root"] += "0"
    elif bad == "notdict":
        roots.insert(0, "entries")
    elif bad == "nonroot":
        x["registration"]["roots"] = rng.choice((None, [], "", {}))
    return x


# ---- the references ------------------------------------------------------------------------------------------------------

def _entry_row(v: dict) -> Wd.Entry:
    return Wd.Entry.from_row(bytes.fromhex(v["row"]))


def _reason(fn: str, why: str | None, keys: dict[str, str]) -> None:
    """Note which of `fn`'s reasons `why` is (by the first of `keys`' fragments it holds), or accept."""
    SEEN[fn].add("accept" if why is None else next((k for k, frag in keys.items() if frag in why), f"other: {why}"))


LAYOUT = {"range": "its range [", "zero": "zero row", "side": "rows, not 1 to", "h3": "-h3 root is not zero",
          "record": "record is not zero", "pass": "tile0, cols and m are not zero", "m": "its m ", "cols": "its cols ",
          "tile0": "its pass tiles"}
RECORD = {"digest": "not the one the entry binds", "place": "its tile0, cols and m", "keep": "keep is not",
          "shape": "has no shape"}
WINDOW = {"bound": "entries, not the bound", "row": "not an entry row", "layout": "not the layout", "kinds": "the calls, then",
          "gap": "does not start where", "size": "tiles, not its", "h3": "-h3 root is not the pass's",
          "overlap": "before the last call's end", "fillerbig": "the filler's range is", "end": "the ranges end at",
          "level": "real tiles are level", "records": "records, not one per call", "indices": "rows' indices are not",
          "record": "record"}


def layout_problem(v: dict) -> str | None:
    why = Wd.layout_problem(_entry_row(v))
    _reason("layout_problem", why, LAYOUT)
    return why


def record_problem(v: dict) -> str | None:
    why = Wd.record_problem(_entry_row(v), v["record"])
    _reason("record_problem", why, RECORD)
    return why


def window_problem(v: dict) -> str | None:
    try:
        why = Wd.window_problem([bytes.fromhex(r) for r in v["rows"]], _work(v["work"]), v["bound"], bytes.fromhex(v["root_h3"]),
                                v["bits"], v["records"], v["indices"])
    except ValueError:
        SEEN["window_problem"].add("refused")
        raise
    _reason("window_problem", why, WINDOW)
    return why


def side_levels(v: dict) -> list[list[str]]:
    return [[x.hex() for x in lv] for lv in Wd.side_levels(_dom(v["domain"]), [bytes.fromhex(x) for x in v["leaves"]])]


def side_path(v: dict) -> list[str]:
    dom = _dom(v["domain"])
    return [x.hex() for x in Wd.side_path(dom, Wd.side_levels(dom, [bytes.fromhex(x) for x in v["leaves"]]), v["index"])]


def public_inputs(v: dict) -> str:
    return Wd.public_inputs(v["registration"], v["drawn"], v["sigma"], v["cap"], bytes.fromhex(v["domain_h3"])).hex()


def layout_constants(_: dict) -> dict:
    return {"T_BITS": Wd.T_BITS, "DEPTHS": Wd.DEPTHS, "PORT": Wd.PORT, "ENTRY_SCHEMA": Wd.ENTRY_SCHEMA,
            "ENTRY_BYTES": Wd.ENTRY_BYTES, "TILE": Wd.TILE, "ROW_STEP": Wd.ROW_STEP, "SIDE_DOMAIN": Wd.SIDE_DOMAIN,
            "OWNER": Wd.OWNER, "SIDE_INDEX_WORDS": Wd.SIDE_INDEX_WORDS, "PUBLIC_INPUTS_FORMAT": Wd.PUBLIC_INPUTS_FORMAT,
            "M_MAX": Wd.M_MAX, "PUBLIC_PORTS": [dict(p) for p in Wd.PUBLIC_PORTS]}


# ---- the ports ----------------------------------------------------------------------------------------------------------

#: What each port's run must reach, so that none is vacuous for one of its checks or regimes: K′'s regimes, and every reason
#: of each function that returns one, and its accept.
REACHES = {"k_prime": {"refused", "capped", "k > 10,000", "n > 2^50", "exact"},
           "layout_problem": {"accept", *LAYOUT}, "record_problem": {"accept", *RECORD},
           "window_problem": {"accept", "refused", *WINDOW}}
SEEN: dict[str, set[str]] = {fn: set() for fn in REACHES}


def k_prime(v: dict) -> int:
    eps, delta, credited, n, w = _q(v["eps"]), _q(v["delta"]), v["credited"], v["n"], v["w_max"]
    try:
        k = Wd.k_prime(eps, delta, credited, n, w)
    except ValueError:
        SEEN["k_prime"].add("refused")
        raise
    q = 1 - eps * credited / (n * w)
    SEEN["k_prime"].update(x for x, hit in (("capped", k == n > 1), ("k > 10,000", k > 10_000), ("n > 2^50", n > 1 << 50),
                                            ("exact", 0 < q and k < delta.denominator.bit_length() and q ** k == delta))
                            if hit)
    return k


def _port(fn: str, reference: Callable[[dict], Any], gen: Callable[[random.Random], dict], **kw) -> D.Port:
    return D.Port(EXE, fn, reference=reference, inputs=st.randoms(use_true_random=False).map(gen), package=LEAN,
                  rejects=(ValueError,), **kw)


PORTS = {
    "on_grid": (_port("on_grid", lambda v: Wd.on_grid(v["n"], v["bits"]), grid_n, need_rejects=False), 500),
    "floor_point": (_port("floor_point", lambda v: Wd.floor_point(v["n"], v["bits"]), grid_n), 500),
    "pad": (_port("pad", lambda v: Wd.pad(v["n"], v["bits"]), grid_n), 500),
    "level": (_port("level", lambda v: Wd.level(v["n"], v["bits"]), grid_n), 500),
    "bucket": (_port("bucket", lambda v: asdict(Wd.bucket(v["n"], v["bits"])), grid_n), 500),
    "point": (_port("point", lambda v: Wd.point(v["ell"], v["bits"]), grid_ell), 500),
    "work_at": (_port("work_at", lambda v: asdict(Wd.Work.at(v["ell"], v["bits"])), grid_ell), 500),
    "levels": (_port("levels", lambda v: Wd.levels(v["lo"], v["hi"], v["bits"]), levels_in), 500),
    "k_prime": (_port("k_prime", k_prime, k_prime_in), 3000),
    "sampling": (_port("sampling", lambda v: Wd.sampling(_q(v["eps"]), _q(v["delta"]), _work(v["work"]), v["tile_work"],
                                                         v["bits"]), sampling_in), 1000),
    "work_to_json": (_port("work_to_json", lambda v: _work(v["work"]).to_json(v["tile_work"]), work_to_json_in,
                           need_rejects=False), 300),
    "entry_row": (_port("entry_row", lambda v: Wd.Entry.from_json(v["entry"]).row().hex(), entry_json_in), 500),
    "entry_words": (_port("entry_words", lambda v: Wd.Entry.from_json(v["entry"]).words(), entry_json_in), 300),
    "entry_from_row": (_port("entry_from_row", lambda v: _entry_row(v).to_json(), row_in), 500),
    "entry_from_words": (_port("entry_from_words", lambda v: Wd.Entry.from_words(v["words"]).to_json(), words_in), 500),
    "layout_problem": (_port("layout_problem", layout_problem, layout_in, need_rejects=False), 1500),
    "pass_tiles": (_port("pass_tiles", lambda v: Wd.pass_tiles(_entry_row(v)), layout_in, need_rejects=False), 200),
    "tiles": (_port("tiles", lambda v: Wd.tiles(_entry_row(v)), layout_in, need_rejects=False), 200),
    "entry_digest": (_port("entry_digest", lambda v: Wd.entry_digest(_entry_row(v)).hex(), layout_in,
                           need_rejects=False), 200),
    "entry_bc": (_port("entry_bc", lambda v: Wd.entry_bc(_entry_row(v), bytes.fromhex(v["salt"])).hex(), bc_in), 300),
    "entry_leaf": (_port("entry_leaf", lambda v: Wd.entry_leaf(_dom(v["domain"]), v["e"], bytes.fromhex(v["bc"])).hex(),
                         leaf_in), 300),
    "tile_in": (_port("tile_in", lambda v: Wd.tile_in(_entry_row(v), v["t"]), tile_in_in), 500),
    "tile_at": (_port("tile_at", lambda v: Wd.tile_at([Wd.Entry.from_row(bytes.fromhex(r)) for r in v["rows"]], v["t"]),
                      tile_at_in), 500),
    "tile_read": (_port("tile_read", lambda v: Wd.tile_read(_entry_row(v), v["i"], v["j"]), tile_read_in), 500),
    "record_digest": (_port("record_digest", lambda v: Wd.record_digest(v["record"]).hex(), record_digest_in,
                            need_rejects=False), 500),
    "record_problem": (_port("record_problem", record_problem, record_problem_in, need_rejects=False), 2000),
    "side_row": (_port("side_row", lambda v: Wd.side_row(v["words"], v["index"]), side_row_in), 300),
    "side_index": (_port("side_index", lambda v: Wd.side_index(v["row"]), side_index_in), 300),
    "side_domain": (_port("side_domain", lambda v: Wd.side_domain(bytes.fromhex(v["window"]), v["side"]).domain_id.hex(),
                          side_domain_in), 200),
    "pad_node": (_port("pad_node", lambda v: Wd.pad_node(_dom(v["domain"]), v["level"], v["index"]).hex(), pad_node_in,
                       need_rejects=False), 200),
    "side_levels": (_port("side_levels", side_levels, side_levels_in), 300),
    "side_path": (_port("side_path", side_path, side_path_in), 300),
    "window_problem": (_port("window_problem", window_problem, window_in), 3000),
    "session_cap": (_port("session_cap", lambda v: Wd.session_cap(v["k_log"], v["per_block"]), session_cap_in), 300),
    "sessions": (_port("sessions", lambda v: Wd.sessions(v["k"], v["cap"]), sessions_in), 300),
    "instance_ts": (_port("instance_ts", lambda v: Wd.instance_ts(v["registration"], v["drawn"], v["sigma"], v["cap"]),
                          instance_ts_in), 1000),
    "public_inputs": (_port("public_inputs", public_inputs, public_inputs_in), 500),
    "layout_constants": (D.Port(EXE, "layout_constants", reference=layout_constants, inputs=st.just({}), package=LEAN,
                                need_rejects=False), 1),
    "constants": (D.Port(EXE, "constants", reference=lambda _: {
        "PUBLIC": [list(p) for p in Wd.PUBLIC], "REGISTERED": list(Wd.REGISTERED), "KINDS": list(Wd.KINDS),
        "GRID_BITS": Wd.GRID_BITS}, inputs=st.just({}), package=LEAN, need_rejects=False), 1),
}

server = D.server_fixture(PORTS["constants"][0])


#: What the recorded served windows of the current registration registered (served_zk runs r20261006-070934-e5f4,
#: -081412-94b4, -195343-75a1, -202341-0806, -214314-06f0 and -225123-3136): run, the work item, the law's K′ and the
#: entries root's domain, each at ε 1/2, δ 1/10, 3 grid bits and w = 4096
SERVED = [
    ("e5f4", {"credited": 33554432, "credited_tiles": 8192, "level": 87, "n": 9216}, 4,
     "033a13db83abc925f5f87e5a3c586d3ec3d9cc7ae21245a150aaff0388ae3204d480216a98dbf8e7ea9d369a3b7eeb7e1d7283897e157c5a93c3d47fde9b27e1"),
    ("94b4", {"credited": 805306368, "credited_tiles": 196608, "level": 123, "n": 212992}, 4,
     "53bd0b2f99ea77a2ee9c6443ef44e068803aa79dbb0a1d91b117e77ae31328b9748c14dc9e39bdb4e326ab1ff3f7c4f4bc018ab6137ca3a71db8743ea32f7004"),
    ("75a1", {"credited": 805306368, "credited_tiles": 196608, "level": 123, "n": 212992}, 4,
     "369b2d7c40a7cada17ddbcd6c66e53cd4a49edf594cd540ff05a0dd68685a31c751221e846991a30edbaeaa96ab9e919fdbfbd61a9c54d6e74baa4743bdeee16"),
    ("0806", {"credited": 33554432, "credited_tiles": 8192, "level": 87, "n": 9216}, 4,
     "9d7ed3025d59d0b96637d6d332b6dfb3a2ff5f87fa0c361cfd652e8d33c896ea9efb24a2265fb360134e207efc366d006a6ffd49bccc84c1193a776ca517b9c2"),
    ("06f0", {"credited": 1476395008, "credited_tiles": 360448, "level": 130, "n": 393216}, 4,
     "47205c057ee419bc38cd24a5f91ea7b97d322bdce9be5c2b9631e4afd531a1449aa8601ae0c8f02269a984eee3a47ec0c8c22b2891e21cdb849b74dffd5f60cc"),
    ("3136", {"credited": 1476395008, "credited_tiles": 360448, "level": 130, "n": 393216}, 4,
     "96e26f4cbffb87a3436eaac2bf9f63bf175bb7be18c0d332a33af5c44b17f9eb49c22caafaedd132db8f56eb1393d5dbe857cecf065df4b2251d870698388e5e"),
]


@pytest.mark.parametrize("run,work,k,domain", SERVED, ids=[s[0] for s in SERVED])
def test_the_served_windows(run: str, work: dict, k: int, domain: str, server) -> None:
    """Lean and Python agree on each recorded window's level, work, sampling and side domains, and give what it
    registered."""
    tw, eps, delta = 4096, [1, 2], [1, 10]
    at = json.loads(D.replay(PORTS["work_at"][0], {"ell": work["level"], "bits": 3}, server)[1])
    assert json.loads(D.replay(PORTS["work_to_json"][0], {"work": at, "tile_work": tw}, server)[1]) == work
    assert json.loads(D.replay(PORTS["k_prime"][0], {"eps": eps, "delta": delta, "credited": work["credited"],
                                                     "n": at["n"], "w_max": tw}, server)[1]) == k
    sampling = json.loads(D.replay(PORTS["sampling"][0], {"eps": eps, "delta": delta, "work": at, "tile_work": tw,
                                                          "bits": 3}, server)[1])
    assert sampling == {"delta": "1/10", "eps": "1/2", "grid_bits": 3, "law": {"k": k, "law": "subset"}}
    for side in ("a", "b"):
        D.replay(PORTS["side_domain"][0], {"window": domain, "side": side}, server)


@pytest.mark.parametrize("fn", sorted(PORTS))
def test_the_lean_is_the_python(fn: str, server) -> None:
    port, examples = PORTS[fn]
    SEEN.get(fn, set()).clear()
    D.run(port, examples=examples, server=server)
    if fn in REACHES:
        assert SEEN[fn] == REACHES[fn]
