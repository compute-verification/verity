"""The Lean trace assigner (`VStar.Trace.assign`, served by ``vstar-trace``) against the witness the deployed C-Flock prover
assigns (``flock-trace``: `Stmt::row_inputs`, `Stmt::unit_inputs`, `Stmt::converge`, `Stmt::witness`, M0's zero mask), on
V*'s staged parts: each block's SHA-512 and each block's ranges' (the ``sha512x3`` compressions, the ``hm96`` slots, the
units, the mask) must be equal, and every port's row whole 16-bit words (`VStar.Trace.fullRows`, which
`RecDeploy.VStarTrace.TraceOk` reads for the row width).

The drawn cases stage a level, a part of the algebra and a cap with the stager's own code on small consistent sessions
(`test_vstar_in_difftest`'s, whose ``inst.bin`` `VStar.In.partIn` is held to). With ``VERITY_VSTAR_IN_DEPLOYED=1``,
`rec-coins`' recorded (16,16,8) session is staged with `rec_vstage.stage` and every staged statement is compared.

``vstar-trace --sat`` holds the same trace to the one table ``verify --zk`` sets up on each statement's verifier files
(`RecDeploy.TablesSat`: every row of the level-0 matrices, the pin, every region bit, in every block): on drawn sessions,
with flips that break it, and on `rec-coins`' staged session, the witness run `RecDeploy.HonestWitSatisfies` cites. Skips
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

from verity_flock import circuit as FC  # noqa: E402
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
    assert ours["full_rows"] is True, f"{d}: a port's row is not whole 16-bit words (the port lemma's row width)"
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


#: the name `stage_part` gives its unit, which claims the verifier's part at that shape (`RecResiduals.check`)
CLAIMED = "rec-residuals/m22-k6-c2-p0"


def unclaim(d: Path) -> Path:
    """``d``'s circuit with its unit renamed to one no verifier check claims: ``ALG_S`` is not the part at the shape its
    name gives, so the parse refuses it as the verifier does (`test_a_part_the_verifier_refuses_has_no_trace`); renamed, it
    is the same gates and ports, which neither assigner reads the name of. The instance file is restaged for the renamed
    circuit: its header's ``circuit_sha512`` and unit classes, as `inner_registered`'s restaging writes them."""
    f = d / "circuit.txt"
    text = f.read_text()
    assert CLAIMED in text
    text = text.replace(CLAIMED, "vstar-trace/alg-p0")
    f.write_text(text)
    inst = d / "inst.bin"
    line, body = inst.read_bytes().split(b"\n", 1)
    head = json.loads(line)
    head["circuit_sha512"] = FC.digest(text)
    if "units" in head:
        head["units"] = dict(head["units"], classes=[FC.class_digest(text)] * len(head["units"]["classes"]))
    inst.write_bytes(json.dumps(head, sort_keys=True).encode() + b"\n" + body)
    return d


@pytest.mark.parametrize("kind", ["level", "inner", "cap"])
def test_the_trace_is_the_deployed_provers(lean, tmp_path, kind):
    """Each kind of part on drawn sessions: Lean's trace is the deployed prover's witness, block for block and range for range."""
    g = random.Random(IN.SEED)
    for i in range(3):
        levels = [(g.randint(0, 2), g.randint(1, 2), g.randint(1, 2)) for _ in range(g.randint(1, 2))]
        d = stage(g.getrandbits(32), kind, levels, tmp_path / f"{kind}{i}")
        agree(lean, unclaim(d) if kind == "inner" else d)


def test_a_part_the_verifier_refuses_has_no_trace(lean, tmp_path):
    """A unit that claims a ``rec-residuals`` part it is not: Lean's parse, the verifier's, refuses it, so no trace."""
    d = stage(3, "inner", [(1, 1, 1)], tmp_path / "inner")
    p = subprocess.run([str(lean), str(d / "circuit.txt"), str(d / "inst.bin")], capture_output=True, text=True)
    assert p.returncode == 2 and "rec-residuals" in p.stderr and not p.stdout, (p.returncode, p.stderr[-2000:])


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


def sat(lean: Path, d: Path, *extra: str) -> dict:
    """``vstar-trace --sat`` on the staged statement ``d``: Lean's trace against the tables ``verify --zk`` sets up on the
    verifier's files (`RecDeploy.TablesSat`)."""
    return trace([str(lean), "--sat", "--circuit", str(d / "circuit.txt"), "--public", str(d / "pub.bin"),
                  "--instances", str(d / "inst.bin"), "--partition", str(d / "partition.json"),
                  "--program", str(d / "program.json"), *extra])


def satisfied(r: dict) -> bool:
    return (r["ok"] is True and r["fits"] is True and r["shape_ok"] is True and r["trace_m"] == r["m"]
            and r["rows_bad"] == r["pin_bad"] == r["region_bad"] == 0)


#: a mask seed: `--zk`'s mask is uniform, and every table must hold whatever it is
MASK_SEED = "20261027"


@pytest.mark.parametrize("kind", ["level", "cap"])
def test_the_trace_satisfies_the_verifiers_tables(lean, tmp_path, kind):
    """Each kind of part on drawn sessions: the trace holds in every block of the one table ``verify --zk`` sets up (every
    row, the pin, every region bit), under the zero mask and a random one, and is no longer a witness once bits move: 64
    flipped bits break a row, the flipped pin the pin, and one salt bit of the prover's rows the verifier's public rows."""
    g = random.Random(IN.SEED + 1)
    for i in range(2):
        levels = [(g.randint(0, 2), g.randint(1, 2), g.randint(1, 2)) for _ in range(g.randint(1, 2))]
        d = stage(g.getrandbits(32), kind, levels, tmp_path / f"{kind}{i}")
        honest = sat(lean, d)
        assert satisfied(honest) and honest["regions"] > 0 and honest["region_bits"] > 0, honest
        assert satisfied(sat(lean, d, "--mask", MASK_SEED))
        flips = random.Random(i).sample(range(2 ** honest["m"]), 64)
        moved = sat(lean, d, "--flip", ",".join(map(str, flips)))
        assert moved["ok"] is False and moved["rows_bad"] > 0, moved
        pin = sat(lean, d, "--flip", str(honest["pin"]))
        assert pin["ok"] is False and pin["pin_bad"] == 1, pin
    data = bytearray((d / "inst.bin").read_bytes())
    nl = data.index(b"\n") + 1
    head = json.loads(data[:nl])
    meta = json.loads((d / "circuit.txt").read_text().split("\n", 2)[1][5:])
    data[nl + sum(2 * w * head["shared_rows"][nm] for nm, w, *_ in meta["ports"])] ^= 1
    (d / "inst.bin").write_bytes(bytes(data))
    salted = sat(lean, d)
    assert salted["ok"] is False and salted["rows_bad"] + salted["region_bad"] > 0, salted


COINS = IN.COINS


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> tuple[Path, list[Path]]:
    """`rec-coins`' recorded (16,16,8) session at 64 instances staged with `rec_vstage.stage`, once for the module."""
    out = tmp_path_factory.mktemp("recorded") / "out"
    S.stage(COINS / "inner-honest", json.loads((COINS / "fold-honest.json").read_text()), out, levels=None,
            salts=COINS / "acc-salts.bin", pts=json.loads((COINS / "points-honest.json").read_text()), evaluate=False,
            part_jobs=1, version=0)
    return out, sorted(p.parent for p in out.rglob("inst.bin") if (p.parent / "circuit.txt").exists())


@pytest.mark.slow
@IN.DEPLOYED
@IN._fetched(COINS, "rec-coins")
def test_the_recorded_plain_leaf_sessions_traces(lean, recorded):
    """Every statement of `rec-coins`' staged session (each level, its cap, each part of the algebra): its trace is the
    deployed prover's."""
    out, dirs = recorded
    checked = []
    for d in dirs:
        agree(lean, d)
        checked.append(str(d.relative_to(out)))
    print(json.dumps({"checked": checked}))
    assert all(any(c.startswith(k) for c in checked) for k in ("L", "cap", "alg/")), checked


@pytest.mark.slow
@IN.DEPLOYED
@IN._fetched(COINS, "rec-coins")
def test_the_recorded_plain_leaf_sessions_satisfy(lean, recorded):
    """Every statement of `rec-coins`' staged session: its trace satisfies the table ``verify --zk`` sets up on its
    verifier's files, under the zero mask and a random one (`RecDeploy.HonestWitSatisfies`' witness run)."""
    out, dirs = recorded
    checked = {}
    for d in dirs:
        for mask in ("zero", MASK_SEED):
            r = sat(lean, d, "--mask", mask)
            print(json.dumps({"sat": str(d.relative_to(out)), "mask": mask, **r}), flush=True)
            assert satisfied(r), (d, r)
        checked[str(d.relative_to(out))] = {k: r[k] for k in ("m", "k_log", "blocks", "regions", "region_bits")}
    print(json.dumps({"satisfied": checked}))
    assert all(any(c.startswith(k) for c in checked) for k in ("L", "cap", "alg/")), checked
