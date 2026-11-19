"""Pearl+'s tile core on a tile part filler (TR real rows of its TM: `pearl_plus_tile`'s TR) gives the verdict Lean's tile
rule gives (`pouw-pearlplus`: `PouwPearlPlus/Rule.lean`'s `rowsFailure`, whose filler rows are a call's rows past its
``real_rows`` and must be zero, and `UnitRecord.credited`, which leaves them out of the bound and of y).

The pass is `test_pearl_c_vllm.proved_pass`'s at k = 256 on 16 x 2 tiles: its first call has 20 real rows of 32, so its
tiles at rows 16-31 have TR = 4 and 12 filler rows. Lean opens tiles of the pass (``--draws``), and on each the core's
Python twin (`pearl_plus_tile.reference`) on the tile's ports (`credited.credited_inputs`, TR the call's real rows in the
tile) is ok exactly where Lean's tile verdict is; on one filler tile the core itself, on words, is too. A pass whose
prover committed 1.0 or -0 in the first filler row (row 20) is refused by both on every tile holding that row, Lean
naming it. Two disagreements that are not filler's are asserted by name where Lean opens such a tile and kept out of the
comparison: a tile holding an excluded row (the block admits every real row, Lean holds an excluded one to failing the
per-row rules) and a tile over its inert-work bound (Lean's ``uncredited`` verdict is ok with its reason, the block's
ok 0). Voluntary rows and a kept width |K| < K, the others, aren't in the pass."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import verity_difftest as D
from verity.circuits.boolean.trace import word_of
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import inert_bound
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_tile as T
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c.credited import credited_inputs

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "verity" / "protocols" / "resource_accounting" / "compute" / "pouw" / "verifier" / "lean" / "pearlplus"
SCHEME = "pearl-c-h100-v1-h3"
SOURCE = b"verifier" * 4
#: 16 rows a tile, so the first call's second row tile is 4 real rows and 12 filler; 2 columns keep the core on words
#: about 40 s a tile
TILE = (16, 2)
#: the core's inert-work bound needs two whole promotion windows of 128 (`inert_bound._promoted`)
K = 256
#: about 8 % of the pass's work is its filler tiles', so 64 draws open one on each pass here
DRAWS = 64
FILLER_ROW = 20
#: FP32 words of the BF16 values 1.0 and -0
NONZERO = [pytest.param(0x3F800000, id="1.0"), pytest.param(0x80000000, id="-0")]


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


V = _module("test_pearl_c_vllm", ROOT / "tests" / "benchmarks" / "pouw" / "test_pearl_c_vllm.py")
pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]


@pytest.fixture(scope="module")
def exe() -> Path:
    return D.build(PACKAGE, "pouw-pearlplus")


def lean(exe: Path, path: Path) -> dict:
    """Lean's verdict on the pass at `path`, read as a test instance's, under its coins and `SOURCE`."""
    p = subprocess.run([str(exe), "--coins", str(path.parent / "coins.json"), "--draws", str(DRAWS), "--source", SOURCE.hex(),
                        "--test-instance", str(path)], capture_output=True, text=True)
    assert p.returncode in (0, 1) and not p.stderr, p.stderr
    got = json.loads(p.stdout)
    assert p.returncode == (got["verdict"] != "accept")
    return got


def statics(c) -> dict:
    d = c.scheme.device
    return dict(K=c.k, TM=c.tm, TR=c.real, TN=c.tn, N=c.n, RHO=inert_bound._rho(d.name, None),
                RHOR=inert_bound.row_rho(c.scheme, c.k), DEV=d.name)


def reference(c) -> tuple[bytes | None, int]:
    s = statics(c)
    return T.reference(s["DEV"], c.k, c.n, c.a, c.b, c.y, c.cert, c.salt, c.root, c.unit, c.rows, list(c.ref.tile.cols),
                       c.col, s["RHO"], s["RHOR"])


def _flat(x) -> list[int]:
    if isinstance(x, (bytes, bytearray)):
        return list(x)
    if isinstance(x, int):
        return [x]
    return [w for v in x for w in _flat(v)]


def on_words(c) -> tuple[bytes, int]:
    out = list(evaluate(word_of(bind(T.PearlPlusTileCoreV2, **statics(c))), *(_flat(a) for a in c.core_args())))
    return bytes(out[:64]), out[64]


def compared(exe: Path, path: Path, records) -> tuple[dict, dict[int, tuple]]:
    """Lean's verdict on the pass, and each tile it opened as (Lean's ok, its reason, the block's ok, its ports), held to
    agree on every tile but the two named disagreements."""
    got = lean(exe, path)
    tiles = {}
    for v in got["verdicts"]:
        if v["tile"] in tiles:
            continue
        c = credited_inputs(path.parent, v["tile"])
        tiles[v["tile"]] = (v["ok"], v["reason"], reference(c)[1], c)
    for tid, (ok, why, block, c) in tiles.items():
        rec = records[c.ref.matmul]
        if why is not None and "exceeds" in why:
            assert (ok, block) == (True, 0), f"tile {tid}: over its inert-work bound ({why})"
        elif set(rec.excluded_a) & set(c.rows[:c.real]):
            assert (ok, why, block) == (True, None, 0), f"tile {tid}: an excluded row's tile"
        else:
            assert ok == bool(block), f"tile {tid} (TR {c.real} of {c.tm}): Lean {ok} {why!r}, the block {block}"
    return got, tiles


def test_a_filler_tiles_verdict_is_leans(exe, tmp_path):
    """On an honest pass Lean accepts, and every tile it opens the block passes, its filler tiles (TR = 4) among them;
    on one, the core on words is ok 1 with the pass's leaf as digest. Slow (a pass, Lean's 64 draws and the core on words
    at 16 x 2, about two minutes): the only test that runs Lean's tile rule and the block's on the same filler tile."""
    path, _s, epoch, prover, _tr, _splits, records = V.proved_pass(tmp_path, SCHEME, tile=TILE, k=K)
    assert records[0].real_rows == FILLER_ROW and records[0].excluded_a == (5,)
    got, tiles = compared(exe, path, records)
    assert got["verdict"] == "accept", got["reason"]
    filler = sorted(t for t, (_ok, why, _b, c) in tiles.items() if c.real < c.tm and why is None)
    assert filler, f"Lean's {DRAWS} draws opened no filler tile within its bound: {sorted(tiles)}"
    c = tiles[filler[0]][3]
    assert c.real == 4 and len(c.y) == 4 and not any(w for x in c.a[4:] for w in x)
    assert tiles[filler[0]][:3] == (True, None, 1)
    assert on_words(c) == (prover.digest(filler[0]), 1)


@pytest.mark.parametrize("word", NONZERO)
def test_a_filler_row_that_is_not_zero_is_refused_by_both(exe, tmp_path, word):
    """A prover that committed `word` in filler row 20 at column 0: Lean refuses the pass, and on every tile it opens
    holding that row its verdict is ``filler row 20 is not zero`` and the block's ok is 0, on words too; the other
    tiles agree as on an honest pass. Slow, as the honest pass's test, and the only test of the block against Lean on
    a filler row that is not zero."""
    path, _s, _epoch, prover, _tr, _splits, records = V.proved_pass(tmp_path, SCHEME, tile=TILE, k=K, filler=word)
    got, tiles = compared(exe, path, records)
    assert got["verdict"] == "reject"
    bad = sorted(t for t, (*_, c) in tiles.items() if FILLER_ROW in c.rows)
    assert bad, f"Lean's {DRAWS} draws opened no tile holding row {FILLER_ROW}: {sorted(tiles)}"
    for t in bad:
        ok, why, block, c = tiles[t]
        assert (ok, block) == (False, 0) and f"filler row {FILLER_ROW} is not zero" in why, (ok, why, block)
        assert c.a[FILLER_ROW - c.rows[0]][0] == word
    assert on_words(tiles[bad[0]][3]) == (prover.digest(bad[0]), 0)
