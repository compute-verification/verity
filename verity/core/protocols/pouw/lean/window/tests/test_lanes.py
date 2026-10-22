"""`PouwWindow.Lanes`, a drawn H1 leaf's lane and combine statements port by port (option A), held to the Python on the
shared harness (`verity_difftest`): each port's fields to `h1zy.statement_ports` (their bits from the Definitions'
signatures), and each port's row to `h1tile.Layout`'s rows and columns as #3116's `pouw.leaf_lanes` reads them (`reads`,
`known`), which `_source` restates until that module is on the tree.

The drawn inputs are tiles on and off a leaf's (R / 4 and C / 8 powers of two, at least two lanes), K on and off the
32-wide step, lane indices in and out of range and paths that are and aren't a statement's. The fixed ones are the served
32 × 32 leaf at K = 448 (the least K `h1zy.proved_k` takes), 4096 and 14336 (Llama-8B's), and the sidecar's 8 × 16 leaf at
K = 64 (`h1zy.circuit-check.py`), every lane and the combine, over both H1 tiles.
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import pytest
import verity_difftest as D
from hypothesis import strategies as st

from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import h1zy as Z
from verity.core.protocols.pouw.circuit import segment as SG
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import gemm as BG

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

LEAN = Path(__file__).resolve().parents[1]
EXE = "pouw-window-difftest"
SERVED = [(32, 32, 448), (32, 32, 4096), (32, 32, 14336)]
SIDECAR = (8, 16, 64)
TILES = (H.PouwH1Tile, H.PouwH1Tile3)


def _tile(r: int, c: int, k: int, defn: Any = H.PouwH1Tile) -> Any:
    exact = bind(BG.GemmAccumulatorS8, K=k, **BG.GEMM_SM120_S8)
    return bind(defn, EXACT=exact, R=r, C=c, PAIRS=tuple((i, j) for i in range(r) for j in range(c)))


def _statement(x: dict, defn: Any = H.PouwH1Tile) -> tuple[Any, H.Layout, int | None]:
    """The lane (``x["lane"]`` its index) or the combine (``lane`` None or absent) of ``x``'s leaf, bound; a reject where
    `h1zy` refuses the tile or the index is none of its lanes (`leaf_lanes.lane_rows`)."""
    ln = x.get("lane")
    try:
        d = bind(Z.PouwH1Combine if ln is None else Z.PouwH1LaneZy, TILE=_tile(x["r"], x["c"], x["k"], defn))
    except ValueError as e:
        raise D.Reject(str(e)) from e
    lay = H.Layout(x["r"], x["c"], x["k"])
    if ln is not None and not 0 <= ln < lay.lanes:
        raise D.Reject(f"an {lay.r} × {lay.c} leaf has {lay.lanes} lanes, not lane {ln}")
    return d, lay, ln


def _source(lay: H.Layout, ln: int | None, port: str) -> dict:
    """Where the verifier opens ``port``'s row, as `leaf_lanes.reads` and `known` state it: a read of a registered value's
    row, or a known row (statement i's ``o0`` the value ``ok<i>``, i = lane, then the combine)."""
    if port == "o0":
        return {"known": f"ok{lay.lanes if ln is None else ln}"}
    if ln is None:
        return {"read": "receipt", "row": 0} if port == "receipt" else {"read": "partials", "row": int(port[1:])}
    if port[0] in "ab" and port[1:].isdigit():
        return {"read": port[0], "row": (lay.rows if port[0] == "a" else lay.cols)(ln)[int(port[1:])]}
    return {"admit": {"known": "admit"}, "pub": {"known": f"pub{ln}"}, "zy": {"read": "zy", "row": 0},
            "partials": {"read": "partials", "row": ln}}[port]


def _field(at: int, bits: int) -> dict:
    return {"bit": at, "bits": bits, "word": at // 16, "words": -(-(at + bits) // 16) - at // 16}


def ports(x: dict, defn: Any = H.PouwH1Tile) -> list[dict]:
    d, lay, ln = _statement(x, defn)
    out = []
    for name, fields in Z.statement_ports(d):
        bits = sum(b for _p, _at, b in fields)
        out.append({"name": name, "source": _source(lay, ln, name), "output": name == "o0", "bits": bits,
                    "words": -(-bits // 16), "fields": [{"path": p, **_field(at, b)} for p, at, b in fields]})
    return out


def field_of(x: dict) -> dict | None:
    for p in ports(x):
        for f in p["fields"]:
            if f["path"] == x["path"]:
                return {"port": p["name"], "source": p["source"], **{k: f[k] for k in ("bit", "bits", "word", "words")}}
    return None


# ---- the drawn inputs ---------------------------------------------------------------------------------------------------

def _leaf(rng: random.Random) -> dict:
    r = rng.choice((4, 8, 8, 16, 32, 32, 64, rng.choice((0, 3, 12, 20))))
    c = rng.choice((8, 16, 16, 32, 32, 64, 128, rng.choice((0, 12, 24, 40))))
    k = rng.choice((0, 32, 64, 384, 448, 896, 4096, 14336, 32 * rng.randint(1, 512), rng.randint(1, 20000)))
    return {"r": r, "c": c, "k": k}


def _lanes(x: dict) -> int:
    return max(1, (x["r"] // 4) * (x["c"] // 8))


def lane_in(rng: random.Random) -> dict:
    x = _leaf(rng)
    return {**x, "lane": rng.choice((0, rng.randrange(_lanes(x)), _lanes(x) - 1, _lanes(x), -1, rng.randint(0, 70000)))}


def combine_in(rng: random.Random) -> dict:
    return _leaf(rng)


def field_in(rng: random.Random) -> dict:
    x = lane_in(rng) if rng.random() < 0.6 else {**combine_in(rng), "lane": None}
    try:
        paths = [f["path"] for p in ports(x) for f in p["fields"]]
    except D.Reject:
        paths = ["salt"]
    return {**x, "path": rng.choice(paths + ["a[4].codes", "b[8].scale", "partials[9999]", "zy", "receipt.z", "nope"])}


def ks_in(rng: random.Random) -> dict:
    p = 128 ** rng.randint(0, 3)
    return {"ks": rng.choice((rng.randint(0, 600), p - 1, p, p + 1, rng.randint(0, 1 << 24), -1))}


def _ks(x: dict) -> int:
    if x["ks"] < 0:
        raise D.Reject("ks is a natural number")
    return SG.n_digits(x["ks"])


def _port(fn: str, reference, gen) -> D.Port:
    return D.Port(EXE, fn, reference=reference, inputs=st.randoms(use_true_random=False).map(gen), package=LEAN)


PORTS = {
    "lane_ports": (_port("lane_ports", ports, lane_in), 300),
    "combine_ports": (_port("combine_ports", ports, combine_in), 150),
    "field_of": (_port("field_of", field_of, field_in), 300),
    "n_digits": (_port("n_digits", _ks, ks_in), 300),
}

server = D.server_fixture(PORTS["n_digits"][0])


@pytest.mark.parametrize("fn", sorted(PORTS))
def test_the_python_is_the_lean(fn: str, server) -> None:
    port, examples = PORTS[fn]
    D.run(port, examples=examples, server=server)


@pytest.mark.parametrize("r,c,k", SERVED + [SIDECAR], ids=lambda v: str(v))
def test_the_leaves(r: int, c: int, k: int, server) -> None:
    """Every lane's ports and the combine's are Python's at the served leaf and the sidecar's, the same over both H1 tiles
    (the statements read a tile's signature, which its form doesn't change)."""
    for ln in [*range(H.Layout(r, c, k).lanes), None]:
        x = {"r": r, "c": c, "k": k, "lane": ln}
        assert ports(x, TILES[0]) == ports(x, TILES[1])
        D.replay(PORTS["combine_ports" if ln is None else "lane_ports"][0], x, server)


@pytest.mark.parametrize("r,c,k", SERVED, ids=lambda v: str(v))
def test_the_served_rows_are_proved_rows(r: int, c: int, k: int, server) -> None:
    """At the served leaf every input row of the 33 statements, in Lean's layout, is whole 16-bit words (C-Flock's
    `unit_lowering`) and a row the Lean verifier's v2 rule takes (`h1zy.row_fits`), and the combine has one row past one
    SHA-512 block (C-Flock's), so a proof can take every statement."""
    lanes = H.Layout(r, c, k).lanes
    for ln in [*range(lanes), None]:
        x = {"r": r, "c": c, "k": k, "lane": ln}
        got = json.loads(D.replay(PORTS["combine_ports" if ln is None else "lane_ports"][0], x, server)[1])
        rows = [p for p in got if not p["output"]]
        assert all(p["bits"] % 16 == 0 and Z.row_fits(p["bits"]) for p in rows), [p["name"] for p in rows]
        assert ln is not None or max(2 * p["words"] for p in rows) >= 128
