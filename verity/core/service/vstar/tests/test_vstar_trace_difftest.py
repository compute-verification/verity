"""The Lean trace assigner (`VStar.Trace.assign`, served by ``vstar-trace``) against the witness the deployed C-Flock prover
assigns (``flock-trace``: `Stmt::row_inputs`, `Stmt::unit_inputs`, `Stmt::converge`, `Stmt::witness`, M0's zero mask), on
V*'s staged parts: each block's SHA-512 and each block's ranges' (the ``sha512x3`` compressions, the ``hm96`` slots, the
units, the mask) must be equal.

The drawn cases stage a level, a part of the algebra and a cap with the stager's own code on small consistent sessions
(`test_vstar_in_difftest`'s, whose ``inst.bin`` `VStar.In.partIn` is held to). With ``VERITY_VSTAR_IN_DEPLOYED=1``,
`rec-coins`' recorded (16,16,8) session is staged with `rec_vstage.stage` and every staged statement is compared. Skips
unless ``FLOCK_TRACE`` names a build of ``flock-trace`` (or ``FLOCK_CIRCUIT`` one of ``flock-circuit`` beside it)."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
import verity_difftest as D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_vstar_in_difftest as IN  # noqa: E402

from verity_flock import rec_vstage as S  # noqa: E402

LEAN = IN.LEAN


def _rust() -> Path | None:
    if os.environ.get("FLOCK_TRACE"):
        return Path(os.environ["FLOCK_TRACE"])
    if os.environ.get("FLOCK_CIRCUIT"):
        return Path(os.environ["FLOCK_CIRCUIT"]).with_name("flock-trace")
    return None


RUST = _rust()
pytestmark = [IN.pytestmark, pytest.mark.skipif(not (RUST and RUST.exists()), reason="FLOCK_TRACE names no flock-trace build")]


@pytest.fixture(scope="module")
def lean() -> Path:
    return D.build(LEAN, "vstar-trace")


def trace(argv: list[str]) -> dict:
    p = subprocess.run(argv, capture_output=True, text=True)
    assert p.returncode == 0, (argv[0], p.stderr[-2000:])
    return json.loads(p.stdout)


def agree(lean: Path, d: Path) -> dict:
    """Both assigners on the staged statement ``d``: the deployed prover's digests, which Lean's must equal."""
    rust = trace([str(RUST), "--instances", str(d / "inst.bin"), "--circuit", str(d / "circuit.txt")])
    ours = trace([str(lean), str(d / "circuit.txt"), str(d / "inst.bin")])
    assert ours["m"] == rust["m"] and ours["k_log"] == rust["k_log"] and ours["instances"] == rust["instances"], (ours, rust)
    bad = [(b, r) for b, (x, y) in enumerate(zip(ours["ranges"], rust["ranges"], strict=True))
           for r, (u, v) in enumerate(zip(x, y, strict=True)) if u != v]
    assert not bad and ours["blocks"] == rust["blocks"], f"{d}: (block, range) where the traces differ: {bad[:12]}"
    return rust


def stage(seed: int, kind: str, levels: list, d: Path) -> Path:
    """One statement of a drawn session staged with the stager's own code (`test_vstar_in_difftest.staged`'s), kept in ``d``."""
    lv = [(H, c, max(1, min(q, 2 ** c // (2 if i == 0 else 1)))) for i, (H, c, q) in enumerate(levels)]
    s = IN.session(random.Random(seed), lv, [1, 2], 3)
    level = seed % len(lv)
    if kind == "level":
        IN.stage_level(s, level, d)
        return d
    if kind == "inner":
        IN.stage_part(s, IN.ALG_S, [2], d)
        return d / "part"
    IN.stage_cap(s, level, d)
    return d


@pytest.mark.parametrize("kind", ["level", "inner", "cap"])
def test_the_trace_is_the_deployed_provers(lean, tmp_path, kind):
    """Each kind of part on drawn sessions: Lean's trace is the deployed prover's witness, block for block and range for range."""
    g = random.Random(IN.SEED)
    for i in range(3):
        levels = [(g.randint(0, 2), g.randint(1, 2), g.randint(1, 2)) for _ in range(g.randint(1, 2))]
        agree(lean, stage(g.getrandbits(32), kind, levels, tmp_path / f"{kind}{i}"))


def test_a_salt_bit_moves_only_its_hm96_slots(lean, tmp_path):
    """The comparison is not vacuous: one bit of one table row's salt moves, on both sides alike, only the ``hm96`` range
    of the blocks whose instances read that row."""
    d = stage(7, "level", [(1, 2, 2)], tmp_path / "honest")
    honest = agree(lean, d)
    data = bytearray((d / "inst.bin").read_bytes())
    nl = data.index(b"\n") + 1
    head = json.loads(data[:nl])
    meta = json.loads((d / "circuit.txt").read_text().split("\n", 2)[1][5:])
    at = nl + sum(2 * w * head["shared_rows"][nm] for nm, w, *_ in meta["ports"])
    data[at] ^= 1
    (d / "inst.bin").write_bytes(bytes(data))
    moved = agree(lean, d)
    hm = next(i for i, r in enumerate(meta["ranges"]) if r.get("net") == "hm96")
    changed = {(b, r) for b, (x, y) in enumerate(zip(honest["ranges"], moved["ranges"], strict=True))
               for r, (u, v) in enumerate(zip(x, y, strict=True)) if u != v}
    assert changed and {r for _, r in changed} == {hm}, changed


COINS = IN.COINS


@pytest.mark.slow
@IN.DEPLOYED
@IN._fetched(COINS, "rec-coins")
def test_the_recorded_plain_leaf_sessions_traces(lean, tmp_path):
    """`rec-coins`' recorded (16,16,8) session at 64 instances staged with `rec_vstage.stage`: every staged statement's trace
    (each level, its cap, each part of the algebra) is the deployed prover's."""
    out = tmp_path / "out"
    S.stage(COINS / "inner-honest", json.loads((COINS / "fold-honest.json").read_text()), out, levels=None,
            salts=COINS / "acc-salts.bin", pts=json.loads((COINS / "points-honest.json").read_text()), evaluate=False,
            part_jobs=1, version=0)
    dirs = sorted(p.parent for p in out.rglob("inst.bin") if (p.parent / "circuit.txt").exists())
    checked = []
    for d in dirs:
        agree(lean, d)
        checked.append(str(d.relative_to(out)))
    print(json.dumps({"checked": checked}))
    assert any(c.startswith("L") for c in checked) and "cap" in checked and any(c.startswith("alg/") for c in checked)
