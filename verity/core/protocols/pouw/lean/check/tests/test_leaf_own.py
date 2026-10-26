"""`pouw-leaf-own-window` (`PouwCheck.LeafOwn`): a window unit's leaf statements read their values at the unit's
placement in the window's tables, which the verifier takes from the window's committed descriptor (`credit.layout_of`'s
reading, `pouw_credit_cases.window_desc`'s encoding) and its round log alone.

The placement here is the spec's arithmetic, written again (`placement`): unit u, tile t = ti·(n/C) + tj of call c, the
j-th call of its K in its round s, reads its A rows from row M·j + R·ti of ``<prefix>a/<K>@<s>``, its B rows from row
b0 + C·tj of ``<prefix>b/<K>`` (b0 the call's linear's first row there), and its z and receipt rows at row t of
``<prefix>zy@<c>`` and ``<prefix>receipt@<c>``; the partials are the leaf's own. The file is `leaf_lanes.registered_file`
at that placement. The check accepts it for each statement of several units of a window whose calls span two rounds and
two linears of one K, and refuses a file of another unit, one read at another round count, and a unit past the window.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import pouw_credit_cases as cc
import pytest
import verity_difftest as D

from verity.core.protocols.pouw import leaf_lanes as LL
from verity.core.protocols.sampled_proofs.one_stage import registered as G

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

LEAN = Path(__file__).resolve().parents[1]
EXE = "pouw-leaf-own-window"
PREFIX = "h1/w-2/"
K = 448
R = C = 32
#: two linears of one K, each called once a round over two rounds: q_proj n = 2C, k_proj n = C, M = 2R
LAYOUT = {"calls": [["q_proj", 0], ["k_proj", 0], ["q_proj", 1], ["k_proj", 1]],
          "linears": [["q_proj", K, 2 * C], ["k_proj", K, C]], "m": 2 * R, "tile": [R, C]}
STEPS = 2


def placement(lay: dict, pre: str, steps: int, u: int) -> tuple[int, int, dict[str, tuple[str, int]]]:
    """Unit u's call, tile and placement of `lay` over `steps` rounds (the module docstring's arithmetic)."""
    (r, c), m = lay["tile"], lay["m"]
    kn = {name: (k, n) for name, k, n in lay["linears"]}
    sizes = [(m // r) * (kn[lin][1] // c) for lin, _blk in lay["calls"]]
    call = next(i for i in range(len(sizes)) if u < sum(sizes[:i + 1]))
    t = u - sum(sizes[:call])
    lin = lay["calls"][call][0]
    k, n = kn[lin]
    each = len(lay["calls"]) // steps
    s = call // each
    j = sum(kn[x][0] == k for x, _blk in lay["calls"][s * each:call])
    b0 = sum(nn for name, kk, nn in lay["linears"][:[x[0] for x in lay["linears"]].index(lin)] if kk == k)
    ti, tj = divmod(t, n // c)
    return call, t, {LL.A: (f"{pre}a/{k}@{s}", m * j + r * ti), LL.B: (f"{pre}b/{k}", b0 + c * tj),
                     LL.ZY: (f"{pre}zy@{call}", t), LL.RECEIPT: (f"{pre}receipt@{call}", t)}


def _entry(rng: random.Random, value: str) -> dict:
    return {"value": value, "root": rng.randbytes(64).hex(), "leaves": 64, "words": 8, "domain": rng.randbytes(64).hex()}


def _known(rng: random.Random, name: str) -> dict:
    return {"value": name, "root": rng.randbytes(64).hex(), "leaves": 1, "words": 1, "word_bits": 16,
            "domain": rng.randbytes(64).hex(), G.SCHEMA_KEY: G.SCHEMA_V2, "reads": [0]}


def _window(rng: random.Random) -> tuple[dict, dict]:
    """The window's registration (each of its tables once, every call's) and a leaf's produced record."""
    tables = sorted({tb for u in range(12) for tb, _ in placement(LAYOUT, PREFIX, STEPS, u)[2]
                     .values()})
    reg = {"format": "verity/registration/v2", "program": rng.randbytes(64).hex(),
           "registered": [_entry(rng, v) for v in tables]}
    prod = {"format": "verity/registration/produced/v1",
            "produced": [{"value": LL.PARTIALS, "root": rng.randbytes(64).hex(), "leaves": 32, "words": 8,
                          "domain": rng.randbytes(64).hex()}]}
    return reg, prod


def _file(rng: random.Random, reg: dict, prod: dict, place: dict, ln: int | None) -> dict:
    kind = LL.COMBINE if ln is None else LL.LANE
    ent, roots = LL.verifier_entries(reg, prod, LL.lanes(R, C), 2 * LL.words(K))
    known = {nm: _known(rng, nm) for nm in ((LL.PUB, LL.ADMIT, LL.OK) if ln is not None else (LL.OK,))}
    return LL.registered_file(LL.reads(R, C, kind, ln), ent, roots, {v: G.SCHEMA_V2 for v in ent}, known, True, place)


def _own(tmp: Path, reg: dict, prod: dict, file: dict, u: int, ln: int | None, steps: int = STEPS) -> tuple[int, dict]:
    """`pouw-leaf-own-window`'s exit code and verdict on unit u's statement ln's file."""
    desc = cc.desc_bytes(cc.window_desc(random.Random(7), LAYOUT, PREFIX))
    paths = {}
    for name, body in (("desc", desc), ("rounds", json.dumps([{"round": s} for s in range(steps)]).encode()),
                       ("reg", json.dumps(reg).encode()), ("prod", json.dumps(prod).encode()),
                       ("file", json.dumps(file).encode())):
        paths[name] = tmp / name
        paths[name].write_bytes(body)
    sess = tmp / "session"
    sess.mkdir(exist_ok=True)
    (sess / "session.json").write_text(json.dumps({"format": "flock-session/v1", "unit_draw": {
        "law": "subset", "population": 1, "k": 1, "units": [0]}}))
    exe = D.build(LEAN, EXE)
    p = subprocess.run([str(exe), *(str(paths[n]) for n in ("desc", "rounds", "reg", "prod", "file")), str(u),
                        "combine" if ln is None else str(ln), str(sess)], capture_output=True, text=True)
    line = next(x for x in p.stdout.splitlines() if x.startswith("OWN "))
    return p.returncode, json.loads(line[4:])


@pytest.mark.parametrize("u", [0, 3, 5, 9, 11])
@pytest.mark.parametrize("ln", [0, 17, 31, None])
def test_the_unit_s_file_at_its_placement_passes(u: int, ln: int | None, tmp_path: Path) -> None:
    """Each statement's file at its unit's placement in the window's tables is accepted, its reads at the tables' base
    rows plus the leaf's rows."""
    rng = random.Random(u * 64 + (ln or 0))
    reg, prod = _window(rng)
    _call, _t, place = placement(LAYOUT, PREFIX, STEPS, u)
    file = _file(rng, reg, prod, place, ln)
    assert _own(tmp_path, reg, prod, file, u, ln) == (0, {"accepted": True})


@pytest.mark.parametrize("ln", [4, None])
def test_another_unit_s_file_is_refused(ln: int | None, tmp_path: Path) -> None:
    """A file at another unit's placement (another tile of its call, a call of the other round, the other linear) is
    refused by name, and so is a lane's read as though the window had one round (its A table another round's, from
    another row); a unit past the window is refused."""
    rng = random.Random(5)
    reg, prod = _window(rng)
    for u, other in ((11, 10), (11, 5), (9, 11)):
        file = _file(rng, reg, prod, placement(LAYOUT, PREFIX, STEPS, other)[2], ln)
        code, got = _own(tmp_path, reg, prod, file, u, ln)
        assert code == 1 and not got["accepted"] and "registered reads: port" in got["why"], (u, other, got)
    file = _file(rng, reg, prod, placement(LAYOUT, PREFIX, STEPS, 11)[2], ln)
    if ln is not None:
        code, got = _own(tmp_path, reg, prod, file, 11, ln, steps=1)
        assert code == 1 and "registered reads: port a0" in got["why"], got
    code, got = _own(tmp_path, reg, prod, file, 12, ln)
    assert code == 1 and "the window has 12 units, not a unit 12" in got["why"], got
