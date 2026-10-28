"""`PouwWindow.Lanes`, a drawn H1 leaf's lane and combine statements port by port (option A), held to the Python on the
shared harness (`verity_difftest`): each port's fields to `h1zy.statement_ports` (their bits from the Definitions'
signatures), each port's row to `pouw.leaf_lanes`' (`reads`, `known`), each known port's row to `leaf_lanes.known_rows`
(its words, and each field's bits where Lean's layout puts it), and the unit's input wires (`wires?`, `wireOf?`) to
the circuits C-Flock lowers the statements to (`class_statement.unit_lowering` on `h1zy.statement_rows`, and the circuit
file's META as `UnitStager` writes it, as #3131's h1_lanes stages them).

The drawn inputs are tiles on and off a leaf's (R / 4 and C / 8 powers of two, at least two lanes), K on and off the
32-wide step, lane indices in and out of range, paths that are and aren't a statement's and wires in, between and past
the ports. The fixed ones are the served 32 × 32 leaf at K = 448 (h1_lanes' default), 4096 and 14336
(Llama-8B's), and the sidecar's 8 × 16 leaf at K = 64 (`h1zy.circuit-check.py`), every lane and the combine, over both H1
tiles; the lowered ones are the combine at the sidecar's leaf, at 8 × 16 and the served leaf at K = 448, the sidecar's
lane (whose 8-bit admit row C-Flock refuses), and with `VERITY_POUW_WINDOW_LOWERED_LANES` a 16 × 16 lane at K = 64 and a
served lane at K = 448.

A J10-R leaf's statements (`j10r_lane_ports`, `j10r_combine_ports`: `PouwH1LaneZy_v2`, `PouwH1LaneZyChunk_v2` and
`PouwH1Combine_v2` over `PouwH1Tile_v5`, as `leaf_lanes` reads them with ``j10r``) are sampled the same way at
n = `weights.chunks`(K), chunk indices in and out of range, and fixed at the served leaf in 1, 2, 4 and 16 chunks (K = 448,
4096, 8192, 16384 and 65536) and the sidecar's in 2, every statement; the combine's lowering at 8 × 16, K = 448 and 8192.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import pytest
import verity_difftest as D
from hypothesis import strategies as st
from verity_flock import class_statement as CS

from verity.core.protocols.pouw import leaf_lanes as LL
from verity.core.protocols.pouw import weights as W
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import h1zy as Z
from verity.core.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.pouw.circuit import segment as SG
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

LEAN = Path(__file__).resolve().parents[1]
EXE = "pouw-window-difftest"
SERVED = [(32, 32, 448), (32, 32, 4096), (32, 32, 14336)]
SIDECAR = (8, 16, 64)
TILES = (H.PouwH1Tile, H.PouwH1Tile3)
LOWERED = [(*SIDECAR, None), (8, 16, 448, None), (32, 32, 448, None), (*SIDECAR, 0)]
LOWERED_LANES = [(16, 16, 64, 3), (32, 32, 448, 7)]
#: a J10-R leaf at K in one chunk (448), at the chunk limit (4096) and past it, in 2, 4 and 16 chunks (8192, 16384 and
#: 65536; `weights.chunks`), and the sidecar's 8 × 16 leaf in 2
J10R_SERVED = [(32, 32, 448), (32, 32, 4096), (32, 32, 8192), (32, 32, 16384), (32, 32, 65536)]
J10R_SIDECAR = (8, 16, 8192)
J10R_LOWERED = [(8, 16, 448), (8, 16, 8192)]


def _tile(r: int, c: int, k: int, defn: Any = H.PouwH1Tile) -> Any:
    exact = bind(BG.GemmAccumulatorS8, K=k, **BG.GEMM_SM120_S8)
    return bind(defn, EXACT=exact, R=r, C=c, PAIRS=tuple((i, j) for i in range(r) for j in range(c)))


def _j10r_statement(x: dict) -> tuple[Any, H.Layout, tuple]:
    """A J10-R leaf's chunk ``x["chunk"]`` of lane ``x["lane"]`` in ``x["n"]`` (`PouwH1LaneZy_v2` at n = 1,
    `PouwH1LaneZyChunk_v2` past it) or its combine (`PouwH1Combine_v2`, ``lane`` None), over `PouwH1Tile_v5`, bound; a
    reject where `h1zy` refuses them. The drawn ``n`` is `weights.chunks`(K) wherever the tile is a leaf's, the lanes'
    chunks `leaf_lanes` reads (and Lean, which takes n, then states the same statements)."""
    ln, n, ch, k = x.get("lane"), x["n"], x.get("chunk"), x["k"]
    try:
        tile = _tile(x["r"], x["c"], k, H.PouwH1Tile5)
        if ln is None:
            d = bind(Z.PouwH1Combine2, TILE=tile)
        elif n == 1:
            if ch != 0:
                raise D.Reject(f"a lane in one chunk has chunk 0 only, not chunk {ch!r}")
            d = bind(Z.PouwH1LaneZy2, TILE=tile)
        else:
            d = bind(Z.PouwH1LaneZyChunk2, TILE=tile, CHUNKS=n, CHUNK=ch)
    except ValueError as e:
        raise D.Reject(str(e)) from e
    assert n == W.chunks(k), f"the drawn n is chunks(K) = {W.chunks(k)} at a leaf's K = {k}, not {n}"
    lay = H.Layout(x["r"], x["c"], k)
    if ln is not None and not 0 <= ln < lay.lanes:
        raise D.Reject(f"an {lay.r} × {lay.c} leaf has {lay.lanes} lanes, not lane {ln}")
    kind = LL.COMBINE if ln is None else LL.LANE if n == 1 else LL.CHUNK
    return d, lay, (kind, ln, None if n == 1 else ch, k)


def _statement(x: dict, defn: Any = H.PouwH1Tile) -> tuple[Any, H.Layout, tuple]:
    """The lane (``x["lane"]`` its index) or the combine (``lane`` None or absent) of ``x``'s leaf, bound, with the
    statement as `leaf_lanes` keys it (kind, lane, chunk, J10-R's K); a reject where `h1zy` refuses the tile or the index
    is none of its lanes (`leaf_lanes.lane_rows`). A J10-R leaf's where ``x["n"]`` is given (`_j10r_statement`)."""
    if x.get("n") is not None:
        return _j10r_statement(x)
    ln = x.get("lane")
    try:
        d = bind(Z.PouwH1Combine if ln is None else Z.PouwH1LaneZy, TILE=_tile(x["r"], x["c"], x["k"], defn))
    except ValueError as e:
        raise D.Reject(str(e)) from e
    lay = H.Layout(x["r"], x["c"], x["k"])
    if ln is not None and not 0 <= ln < lay.lanes:
        raise D.Reject(f"an {lay.r} × {lay.c} leaf has {lay.lanes} lanes, not lane {ln}")
    return d, lay, (LL.COMBINE if ln is None else LL.LANE, ln, None, None)


def _sources(lay: H.Layout, at: tuple) -> dict[str, dict]:
    """Where the verifier opens each port's row (`leaf_lanes.reads`, `known`) for the statement ``at`` (`_statement`'s):
    a read of a registered value's row, or a known row (statement i's ``o0`` the value ``ok<i>``, i = lane, then the
    combine; a J10-R leaf's lanes' and chunks' ``o0`` row i of ``oks``)."""
    kind, ln, ch, j10r = at
    out = {p: {"known": v} for p, v in LL.known(lay.r, lay.c, kind, ln, ch, j10r=j10r).items()}
    return {**out, **{p: {"read": v, "row": r} for p, (v, r) in LL.reads(lay.r, lay.c, kind, ln, ch, j10r=j10r).items()}}


def _bits_n(words: list[int], lo: int, n: int) -> int:
    """Bits ``[lo, lo + n)`` of a row as a number, least significant first (`Accounting.TS.Lanes.bitsN`)."""
    return sum(((words[(lo + i) // 16] >> ((lo + i) % 16)) & 1) << i for i in range(n))


def _field(at: int, bits: int) -> dict:
    return {"bit": at, "bits": bits, "word": at // 16, "words": -(-(at + bits) // 16) - at // 16}


def ports(x: dict, defn: Any = H.PouwH1Tile) -> list[dict]:
    d, lay, at = _statement(x, defn)
    src, out = _sources(lay, at), []
    for name, fields in Z.statement_ports(d):
        bits = sum(b for _p, _at, b in fields)
        out.append({"name": name, "source": src[name], "output": name == "o0", "bits": bits,
                    "words": -(-bits // 16), "fields": [{"path": p, **_field(at, b)} for p, at, b in fields]})
    return out


def field_of(x: dict) -> dict | None:
    for p in ports(x):
        for f in p["fields"]:
            if f["path"] == x["path"]:
                return {"port": p["name"], "source": p["source"], **{k: f[k] for k in ("bit", "bits", "word", "words")}}
    return None


def wires(x: dict) -> list[dict]:
    """Each input port's first wire and wires: the input rows one after another, as `unit_lowering` lays them out, which
    refuses a row that isn't whole u16 words."""
    out, at = [], 0
    for p in ports(x):
        if p["output"]:
            continue
        if p["bits"] % 16:
            raise D.Reject(f"row {p['name']} is {p['bits']} bits, not whole u16 words")
        out.append({"port": p["name"], "source": p["source"], "wire": at, "wires": p["bits"]})
        at += p["bits"]
    return out


def _place(x: dict, port: str, bit: int) -> dict:
    """Bit ``bit`` of ``port``'s row, with the field holding it (`h1zy.statement_ports`)."""
    p = next(p for p in ports(x) if p["name"] == port)
    f = next(f for f in p["fields"] if f["bit"] <= bit < f["bit"] + f["bits"])
    return {"port": port, "source": p["source"], "bit": bit, "field": f["path"], "field_bit": bit - f["bit"]}


def wire_of(x: dict) -> dict | None:
    ws = wires(x)
    if x["wire"] < 0:
        raise D.Reject("a wire is a natural number")
    hit = [w for w in ws if w["wire"] <= x["wire"] < w["wire"] + w["wires"]]
    return _place(x, hit[0]["port"], x["wire"] - hit[0]["wire"]) if hit else None


_LOWERINGS: dict[tuple, Any] = {}


def _lowering(x: dict, stage: bool = False) -> tuple[Any, dict | None]:
    """C-Flock's lowering of ``x``'s statement (`class_statement.unit_lowering` on `h1zy.statement_rows`) and, with
    ``stage``, its circuit file's META (`UnitStager`: each input row ``hm96-sha512/row/v2`` of its bits, ``ok`` one v2 row,
    as #3131's h1_lanes stages them); a reject where C-Flock refuses a row that isn't whole u16 words. Each is built once
    (a lane's lowering is about 2M ANDs), a staged one serving an unstaged ask."""
    key = (x["r"], x["c"], x["k"], x.get("lane"), x.get("n"), x.get("chunk"))
    got = _LOWERINGS.get(key)
    if got is None or (stage and not isinstance(got, D.Reject) and got[1] is None):
        d, _lay, (_kind, ln, _ch, _j10r) = _statement(x)
        rows = Z.statement_rows(d)
        name = f"unit/pouw-h1-{'combine' if ln is None else 'lane'}"
        try:
            if stage:
                bits = {p: sum(b for _f, _at, b in fs) for p, fs in Z.statement_ports(d)}
                s = CS.UnitStager(d, name, [(p, V2, bits[p], ps) for p, ps in rows], [(V2, (0,))])
                got = (s.low, s.meta)
            else:
                got = (CS.unit_lowering(d, name, rows)[0], None)
        except ValueError as e:
            if "not whole u16 words" not in str(e):
                raise
            got = D.Reject(str(e))
        _LOWERINGS[key] = got
    if isinstance(got, D.Reject):
        raise D.Reject(str(got))
    return got


def _columns(x: dict) -> tuple[dict[int, int], list[tuple[str, int]]]:
    """The lowered unit's input columns, each row word's (C-Flock's u16 input ``in<j>`` at `Layout.in_port_cols[j]`, on
    the row word META's ``leaves_in`` names, or the lowering's own leaf map unstaged), and its row ports' words in order
    (the lowering's own signature)."""
    low, meta = _lowering(x)
    (leaves,) = meta["leaves_in"] if meta else low.leaves["in"]
    assert low.layout.in_bits == [16] * len(leaves)
    sig = [(r, t.n) for r, t in low.definition.signature()[0]]
    return {g: col for g, col in zip(leaves, low.layout.in_port_cols, strict=True)}, sig


def lowered_wires(x: dict) -> list[dict]:
    """`wires` as the lowered unit lays its input rows out: each port's first input column and its columns, which must be
    one run of 16 per row word."""
    col, sig = _columns(x)
    _d, lay, at = _statement(x)
    src, out, g = _sources(lay, at), [], 0
    for port, words in sig:
        run = [col[g + i] + t for i in range(words) for t in range(16)]
        assert run == list(range(run[0], run[0] + 16 * words)), port
        out.append({"port": port, "source": src[port], "wire": run[0], "wires": 16 * words})
        g += words
    return out


def lowered_wire_of(x: dict) -> dict | None:
    """`wire_of` as the lowered unit reads input column ``wire``: the row word its u16 input is on and that input's bit."""
    col, sig = _columns(x)
    if x["wire"] < 0:
        raise D.Reject("a wire is a natural number")
    hit = [g for g, c in col.items() if c <= x["wire"] < c + 16]
    if not hit:
        return None
    g, at = hit[0], 0
    for port, words in sig:
        if g < at + words:
            return _place(x, port, 16 * (g - at) + x["wire"] - col[g])
        at += words
    raise AssertionError(f"row word {g} is past the rows {sig}")


# ---- the drawn inputs ---------------------------------------------------------------------------------------------------

def _leaf(rng: random.Random) -> dict:
    r = rng.choice((4, 8, 8, 16, 32, 32, 64, rng.choice((0, 3, 12, 20))))
    c = rng.choice((8, 16, 16, 32, 32, 64, 128, rng.choice((0, 12, 24, 40))))
    k = rng.choice((0, 32, 64, 384, 448, 896, 4096, 14336, 32 * rng.randint(1, 512), rng.randint(1, 20000),
                    (1 << 17) - 32, 1 << 17, 32 * rng.randint(4090, 4200)))
    return {"r": r, "c": c, "k": k}


def _lanes(x: dict) -> int:
    return max(1, (x["r"] // 4) * (x["c"] // 8))


def lane_in(rng: random.Random) -> dict:
    x = _leaf(rng)
    return {**x, "lane": rng.choice((0, rng.randrange(_lanes(x)), _lanes(x) - 1, _lanes(x), -1, rng.randint(0, 70000)))}


def combine_in(rng: random.Random) -> dict:
    return _leaf(rng)


def _tiles_k(k: int) -> bool:
    return k % 32 == 0 and 64 <= k < 1 << 17


def _j10r_leaf(rng: random.Random) -> dict:
    """A leaf as `_leaf` draws it, or at a served J10-R K, with ``n`` its lanes' chunks: `weights.chunks`(K) at every K a
    leaf's tile takes (a K there that `chunks` refuses is sampled again), any small n at one it doesn't."""
    while True:
        x = _leaf(rng)
        if rng.random() < 0.5:
            x["k"] = rng.choice((8192, 12288, 16384, 65536, 512 * rng.randint(9, 255)))
        try:
            return {**x, "n": W.chunks(x["k"])}
        except ValueError:
            if not _tiles_k(x["k"]):
                return {**x, "n": rng.choice((0, 1, 2, 3))}


def j10r_lane_in(rng: random.Random) -> dict:
    x = _j10r_leaf(rng)
    n, lanes = x["n"], _lanes(x)
    return {**x, "lane": rng.choice((0, rng.randrange(lanes), lanes - 1, lanes, -1)),
            "chunk": rng.choice((0, rng.randrange(max(1, n)), n - 1, n, -1))}


def j10r_combine_in(rng: random.Random) -> dict:
    return _j10r_leaf(rng)


def _statement_in(rng: random.Random) -> dict:
    u = rng.random()
    if u < 0.3:
        return lane_in(rng)
    if u < 0.5:
        return {**combine_in(rng), "lane": None}
    return j10r_lane_in(rng) if u < 0.8 else {**j10r_combine_in(rng), "lane": None}


def field_in(rng: random.Random) -> dict:
    x = _statement_in(rng)
    try:
        paths = [f["path"] for p in ports(x) for f in p["fields"]]
    except D.Reject:
        paths = ["salt"]
    return {**x, "path": rng.choice(paths + ["a[4].codes", "b[8].scale", "partials[9999]", "zy", "receipt.z", "nope",
                                             "carry_in.words", "oks[99999]", "call"])}


def wires_in(rng: random.Random) -> dict:
    return _statement_in(rng)


def wire_in(rng: random.Random) -> dict:
    x = wires_in(rng)
    try:
        ws = wires(x)
    except D.Reject:
        ws = []
    n = sum(w["wires"] for w in ws)
    edges = [e for w in ws for e in (w["wire"], w["wire"] + w["wires"] - 1)]
    return {**x, "wire": rng.choice(edges + [0, n - 1, n, n + 1, rng.randrange(max(1, n)), -1, rng.randint(0, 1 << 40)])}


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
    "j10r_lane_ports": (_port("j10r_lane_ports", ports, j10r_lane_in), 300),
    "j10r_combine_ports": (_port("j10r_combine_ports", ports, j10r_combine_in), 150),
    "field_of": (_port("field_of", field_of, field_in), 300),
    "n_digits": (_port("n_digits", _ks, ks_in), 300),
    "wires": (_port("wires", wires, wires_in), 300),
    "wire_of": (_port("wire_of", wire_of, wire_in), 300),
}
LOWERED_WIRES = D.Port(EXE, "wires", reference=lowered_wires, inputs=st.nothing(), package=LEAN)
LOWERED_WIRE_OF = D.Port(EXE, "wire_of", reference=lowered_wire_of, inputs=st.nothing(), package=LEAN)

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
    `unit_lowering`; the Lean verifier's v2 rule takes a row of any length, #3148), and the combine has a row that
    compresses before its last SHA-512 block, at least 112 bytes (C-Flock's, `h1zy.LEAST_K`), so a proof can take every
    statement."""
    lanes = H.Layout(r, c, k).lanes
    for ln in [*range(lanes), None]:
        x = {"r": r, "c": c, "k": k, "lane": ln}
        got = json.loads(D.replay(PORTS["combine_ports" if ln is None else "lane_ports"][0], x, server)[1])
        rows = [p for p in got if not p["output"]]
        assert all(p["bits"] % 16 == 0 for p in rows), [p["name"] for p in rows]
        assert ln is not None or max(2 * p["words"] for p in rows) + 17 > 128


@pytest.mark.parametrize("r,c,k", SERVED[:1] + [SIDECAR], ids=lambda v: str(v))
def test_the_known_rows_are_the_known_ports(r: int, c: int, k: int, server) -> None:
    """Each statement's known rows (`leaf_lanes.known_rows`) are its known ports in Lean's layout: the same ports and
    values, each row its port's words (lane l's ``pub`` 101, ``admit`` the leaf's R bits, ``o0`` one) with no bit set past
    its port's, and each field read least significant bit first where Lean puts it (`fieldOf?`, as
    `Accounting.TS.Lanes.decLane` reads it) the value it was built of: the salt and roots with their big-endian 32-bit
    halves least significant first, the position with its halves swapped, the index, the admit bits and ok = 1."""
    rng = random.Random(r * c * k)
    salt, roots, position = rng.randbytes(64), [rng.randbytes(64), rng.randbytes(64)], rng.getrandbits(64)
    admit = [rng.getrandbits(1) for _ in range(r)]

    def halves_up(b: bytes) -> int:
        return sum(h << 32 * i for i, h in enumerate(SD.halves(b)))

    want = {"salt": halves_up(salt), "position": (position & 0xFFFFFFFF) << 32 | position >> 32,
            "roots": halves_up(roots[0] + roots[1]), "admit": sum(b << i for i, b in enumerate(admit)), "ok": 1}
    for ln in [*range(H.Layout(r, c, k).lanes), None]:
        x = {"r": r, "c": c, "k": k, "lane": ln}
        got = json.loads(D.replay(PORTS["combine_ports" if ln is None else "lane_ports"][0], x, server)[1])
        lean = {p["name"]: p for p in got if "known" in p["source"]}
        mine = LL.known_rows(r, c, LL.COMBINE if ln is None else LL.LANE, ln, admit=admit, salt=salt, position=position,
                             roots=roots)
        assert {p: v for p, (v, _) in mine.items()} == {p: q["source"]["known"] for p, q in lean.items()}
        assert ln is None or (lean["pub"]["words"], lean["admit"]["bits"]) == (101, r)
        assert lean["o0"]["words"] == 1
        for p, (_v, words) in mine.items():
            port = lean[p]
            assert len(words) == port["words"] and _bits_n(words, port["bits"], 16 * len(words) - port["bits"]) == 0, p
            for f in port["fields"]:
                assert _bits_n(words, f["bit"], f["bits"]) == {**want, "lane": ln}[f["path"]], (ln, p, f["path"])


def _j10r_statements(r: int, c: int, k: int) -> list[dict]:
    """A J10-R leaf's statements (`leaf_lanes.statements`) as the difftest's inputs: each lane's chunks, then the
    combine."""
    n = W.chunks(k)
    return [{"r": r, "c": c, "k": k, "n": n, "lane": s[1], "chunk": s[2] if len(s) > 2 else 0} if s[0] != LL.COMBINE
            else {"r": r, "c": c, "k": k, "n": n, "lane": None} for s in LL.statements(r, c, j10r=k)]


def _j10r_fn(x: dict) -> D.Port:
    return PORTS["j10r_combine_ports" if x["lane"] is None else "j10r_lane_ports"][0]


@pytest.mark.parametrize("r,c,k", J10R_SERVED + [J10R_SIDECAR], ids=lambda v: str(v))
def test_the_j10r_leaves(r: int, c: int, k: int, server) -> None:
    """Every statement of a J10-R leaf, each lane's chunks (`weights.chunks`(K) of them: 1 to 16 at the served K) and
    the combine over their oks, is Python's (`h1zy.statement_ports` of `PouwH1LaneZy_v2`, `PouwH1LaneZyChunk_v2` and
    `PouwH1Combine_v2`, the rows `leaf_lanes` reads), and every input row is whole 16-bit words, the combine's at least
    112 bytes (`h1zy.LEAST_K`), so a proof can take each (but the sidecar's lanes', whose admit row is 8 bits)."""
    for x in _j10r_statements(r, c, k):
        got = json.loads(D.replay(_j10r_fn(x), x, server)[1])
        rows = [p for p in got if not p["output"]]
        assert r % 16 or all(p["bits"] % 16 == 0 for p in rows), (x, [p["name"] for p in rows if p["bits"] % 16])
        assert x["lane"] is not None or max(2 * p["words"] for p in rows) + 17 > 128
        assert got[-1]["source"] == ({"known": f"ok{H.Layout(r, c, k).lanes * x['n']}"} if x["lane"] is None else
                                     {"read": LL.OKS, "row": x["n"] * x["lane"] + x["chunk"]})


@pytest.mark.parametrize("r,c,k", J10R_LOWERED, ids=lambda v: str(v))
def test_the_j10r_combines_lowered_wires(r: int, c: int, k: int, server) -> None:
    """Lean's wire map of a J10-R combine (v1's rows, then each lane's and chunk's ok row) is the circuit C-Flock lowers
    `PouwH1Combine_v2` to."""
    _lowered_wires({"r": r, "c": c, "k": k, "n": W.chunks(k), "lane": None}, server)


@pytest.mark.parametrize("r,c,k", [(32, 32, 448), (32, 32, 8192), J10R_SIDECAR], ids=lambda v: str(v))
def test_the_j10r_known_rows_are_the_known_ports(r: int, c: int, k: int, server) -> None:
    """A J10-R leaf's known rows (`leaf_lanes.known_rows` with the unit's call and blocks) are its known ports in Lean's
    layout: each chunk's ``pub`` (113 words: v1's, then the call and the two blocks, each a u64 as the position's leaves,
    then the index), the last chunk's ``admit``, and only the combine's ``o0``; each field where Lean puts it the value
    it was built of."""
    rng = random.Random(r * c * k + 1)
    salt, roots, position = rng.randbytes(64), [rng.randbytes(64), rng.randbytes(64)], rng.getrandbits(64)
    call, blocks = rng.getrandbits(64), [rng.getrandbits(64), rng.getrandbits(64)]
    admit = [rng.getrandbits(1) for _ in range(r)]

    def halves_up(b: bytes) -> int:
        return sum(h << 32 * i for i, h in enumerate(SD.halves(b)))

    def u64(v: int) -> int:
        return (v & 0xFFFFFFFF) << 32 | v >> 32

    want = {"salt": halves_up(salt), "position": u64(position), "roots": halves_up(roots[0] + roots[1]),
            "call": u64(call), "blocks": u64(blocks[0]) | u64(blocks[1]) << 64,
            "admit": sum(b << i for i, b in enumerate(admit)), "ok": 1}
    for x in _j10r_statements(r, c, k):
        got = json.loads(D.replay(_j10r_fn(x), x, server)[1])
        lean = {p["name"]: p for p in got if "known" in p["source"]}
        ln, n = x["lane"], x["n"]
        kind = LL.COMBINE if ln is None else LL.LANE if n == 1 else LL.CHUNK
        mine = LL.known_rows(r, c, kind, ln, None if n == 1 or ln is None else x["chunk"], admit=admit, salt=salt,
                             position=position, roots=roots, j10r=k, call=call, blocks=blocks)
        assert {p: v for p, (v, _) in mine.items()} == {p: q["source"]["known"] for p, q in lean.items()}
        assert set(lean) == ({"o0"} if ln is None else {"pub", "admit"} if x["chunk"] == n - 1 else {"pub"})
        assert ln is None or lean["pub"]["words"] == 113
        for p, (_v, words) in mine.items():
            port = lean[p]
            assert len(words) == port["words"] and _bits_n(words, port["bits"], 16 * len(words) - port["bits"]) == 0, p
            for f in port["fields"]:
                assert _bits_n(words, f["bit"], f["bits"]) == {**want, "lane": ln}[f["path"]], (x, p, f["path"])


def _lowered_wires(x: dict, server) -> None:
    """Lean's input wires are the lowered unit's, port by port and at each port's first and last wire, past the rows and
    at a wire far out."""
    kind, text = D.replay(LOWERED_WIRES, x, server)
    ws = json.loads(text) if kind == "output" else []
    n = sum(w["wires"] for w in ws)
    for wire in sorted({e for w in ws for e in (w["wire"], w["wire"] + w["wires"] - 1)} | {0, n, 1 << 40}):
        D.replay(LOWERED_WIRE_OF, {**x, "wire": wire}, server)


@pytest.mark.parametrize("r,c,k,lane", LOWERED, ids=lambda v: str(v))
def test_the_lowered_wires(r: int, c: int, k: int, lane: int | None, server) -> None:
    """Lean's wire map is the circuit C-Flock lowers the statement to (`class_statement.unit_lowering`): its row ports
    and their words, each u16 input's column and row word. The sidecar's lane is C-Flock's refusal and Lean's (its admit
    row is 8 bits)."""
    _lowered_wires({"r": r, "c": c, "k": k, "lane": lane}, server)


def _circuit_ports(x: dict, server) -> None:
    """The staged circuit file's own port records are Lean's wires: META ``ports`` (the input rows' names and words, then
    ``o0``, one word), ``row_schemas`` (each row ``hm96-sha512/row/v2`` of exactly its wires) and ``leaves_in`` (row word
    ``j`` the unit's input leaf ``j``)."""
    _low, meta = _lowering(x, stage=True)
    got = json.loads(D.replay(LOWERED_WIRES, x, server)[1])
    n = meta["input_ports"]
    assert [p[:2] for p in meta["ports"][:n]] == [[w["port"], w["wires"] // 16] for w in got]
    assert [p[:2] for p in meta["ports"][n:]] == [["o0", 1]]
    assert meta["row_schemas"][:n] == [[V2, w["wires"]] for w in got]
    assert meta["leaves_in"] == [list(range(sum(w["wires"] for w in got) // 16))]
    _lowered_wires(x, server)


@pytest.mark.slow
@pytest.mark.parametrize("r,c,k", [(8, 16, 448), (32, 32, 448)], ids=lambda v: str(v))
def test_the_circuit_files_ports(r: int, c: int, k: int, server) -> None:
    """The combine's circuit file at 8 × 16 and the served leaf (K = 448) holds its rows as Lean's wires say: what a
    decoder of the proved statement reads, which `unit_lowering` alone doesn't write. Slow: the first `UnitStager` builds
    C-Flock's SHA-512 and hm96 slot circuits, about 14 s."""
    _circuit_ports({"r": r, "c": c, "k": k, "lane": None}, server)


@pytest.mark.slow
def test_the_sidecar_combine_has_no_circuit_file() -> None:
    """At the sidecar's leaf (K = 64) the combine's rows are 16 words, under one SHA-512 block, which `circuit.compose`
    refuses: the leaf's wire map there is the lowered unit's only. Slow as `test_the_circuit_files_ports`."""
    with pytest.raises(ValueError, match="no row has a compression before its last block"):
        _lowering({"r": SIDECAR[0], "c": SIDECAR[1], "k": SIDECAR[2], "lane": None}, stage=True)


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("VERITY_POUW_WINDOW_LOWERED_LANES"),
                    reason="a lane's lowering is about 2M ANDs: about 150 s at 16 × 16 and 460 s at the served leaf, "
                           "32 GB at most; set VERITY_POUW_WINDOW_LOWERED_LANES=1")
@pytest.mark.parametrize("r,c,k,lane", LOWERED_LANES, ids=lambda v: str(v))
def test_the_lanes_circuit_files_ports(r: int, c: int, k: int, lane: int, server) -> None:
    """A lane's circuit file holds its rows as Lean's wires say, at the least leaf whose admit row is whole words (16 × 16,
    K = 64) and at the served leaf (K = 448): what the combine's can't check, the lane's own rows (A, B, admit, zy,
    partials, pub) and their order as C-Flock lowers them."""
    try:
        _circuit_ports({"r": r, "c": c, "k": k, "lane": lane}, server)
    finally:
        _LOWERINGS.clear()
