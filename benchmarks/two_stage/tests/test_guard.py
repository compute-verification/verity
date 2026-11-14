"""SPOT[Flock]'s two-stage Lean model on a real run of the toy: the conjuncts of `SpotFlock.stmtOk`
(`security/Proofs/Flock/SpotFlock/Code.lean`) that read the run accept an honest two-stage run at a `Held` fixed before
the run, and refuse the runs they must refuse.

What it shows: `stmtOk` is its conjuncts on the registration alone (`regOk`: `partition` and `two`) and its conjuncts on
the run (`runOkAt`), a kernel-checked equation (`TwoGuard.stmtOk_split`). On the honest run's messages (the boundary
record x, the interior record m₁, its session statements m₂, one per template in template order, the two stages' coins
c₀ and c₁), `runOkAt h x m₁ m₂ c₀ c₁` evaluates to true in Lean, at an `h` that `TwoGuard.lean` builds only from what the
auditor registers before the run: the boundary registration's program, query, work table and selection rule, the counts,
the templates' names, links, wires and descriptor, the boundary and interior registrations' ports (`held.json`), the
templates' reference statements, made from the interior window with no run in it and admitted only after the script
checks each is in Flock's masked scope (`zkOk`), and the templates' reference reads as the params name them
(`ref-<t>.reads.json`, `Held.refReads`). Nothing in `h` reads the run's messages or coins. `two`'s conjuncts on the
members' files (`twoFilesOk`: the layout fits them, and each reference statement reads as a member's) evaluate to true
too. `runOkAt` refuses the altered run, and the honest run with one byte of a session statement's registered reads
flipped (`guard.flip_registered`). Its records check (`recordsOk`) refuses the honest run with FLIP_B's own port `o0`,
which no link or wire names, dropped from round 1's record and registered as a value of its name (`guard.drop_port`),
and with a value the registration registers registered twice (`guard.register`), where the same run with it registered
once passes.

What it doesn't show: that `stmtOk` accepts, so not that a guarantee over it (`SpotKSTwo`) is non-vacuous. `regOk` also
asks for `partition`'s `membersPart` and `two`'s `scopeOk` over `bitsProgs`, which read only `h` and unroll the members'
circuits; `regOk h = true` is a fact about the registration, to be proved rather than decided on a run, and this evaluates
neither. The result lists each member's sizes.
Nor that each session accepts: `stmtOk` reads the session statements and their registered reads, never a session's
transcript. Each session's acceptance is the Lean verifier's verdict in the run that made the fixture (`verdicts.json`,
asserted here), whose statements these are byte for byte (`guard.session_statement` rebuilds each from its command line
and archive).

Opt-in: ``VERITY_TWO_STAGE_GUARD`` names the command that runs `TwoGuard.lean` (`twoguard`), which `guard_node.py
--pytest` makes on a node's warm Lean tree and runs this with. The fixture is a run's `guard/` export (`guard.py`) for
both cases."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from verity_config.settings import of

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import guard as G  # noqa: E402

EXE = of(__file__).VERITY_TWO_STAGE_GUARD
FIXTURE = HERE / "fixtures" / "guard.tar.xz"
#: a value no port of the toy's is named, which the registered-value variants register
VALUE = "TwoStageToyGuard/registered"
#: `toy.FLIP_B`'s name, the template whose own port `o0` no link or wire names; `toy` needs the workspace, which
#: `guard_node.py --pytest`'s environment doesn't have
FLIP_B = "TwoStageToyFlipB"
pytestmark = pytest.mark.skipif(not EXE, reason="VERITY_TWO_STAGE_GUARD names the command that runs TwoGuard.lean "
                                "(guard_node.py --pytest makes it)")


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("two-stage-guard")
    with tarfile.open(FIXTURE) as tar:
        tar.extractall(out, filter="data")
    return out


def check(run: Path, m2: Path | None = None) -> dict:
    argv = [EXE, str(run), "check", *([str(m2)] if m2 else [])]
    p = subprocess.run(argv, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("GUARD-RESULT ")]
    assert lines, f"TwoGuard.lean printed no result (exit {p.returncode}):\n{p.stdout[-4000:]}\n{p.stderr[-4000:]}"
    result = json.loads(lines[-1].removeprefix("GUARD-RESULT "))
    assert result["held"], f"a reference statement is out of Flock's masked scope:\n{p.stdout[-4000:]}"
    assert p.returncode == (0 if result["runOk"] and result["twoFilesOk"] else 1), p.stdout[-4000:]
    return result


def verdicts(run: Path) -> list[bool]:
    return [v["accepted"] for v in json.loads((run / "verdicts.json").read_text())]


def variant(tmp_path: Path, run: Path, files: dict[str, bytes]) -> Path:
    """``run``'s export with ``files`` in place of its own, the rest linked."""
    d = tmp_path / "run"
    d.mkdir()
    for p in run.iterdir():
        if p.name not in files:
            (d / p.name).symlink_to(p)
    for name, data in files.items():
        (d / name).write_bytes(data)
    return d


def registered(run: Path, tmp_path: Path, times: int) -> dict:
    """``check`` of ``run`` at a registration whose interior registers `VALUE`, its record registering it ``times`` times."""
    held = json.loads((run / "held.json").read_text())
    held["interior_names"]["values"] = [VALUE]
    m1 = G.register((run / "m1.bin").read_bytes(), VALUE, times)
    return check(variant(tmp_path, run, {"held.json": json.dumps(held).encode(), "m1.bin": m1}))


def test_honest_run_passes(runs: Path):
    assert all(verdicts(runs / "honest"))
    result = check(runs / "honest")
    assert result["runOk"] and result["recordsOk"] and result["twoFilesOk"], result


def test_altered_run_fails(runs: Path):
    assert not all(verdicts(runs / "altered"))
    assert not check(runs / "altered")["runOk"]


def test_one_registered_byte_flipped_fails(runs: Path, tmp_path: Path):
    flipped = tmp_path / "m2-flipped.bin"
    flipped.write_bytes(G.flip_registered((runs / "honest" / "m2.bin").read_bytes()))
    assert not check(runs / "honest", flipped)["runOk"]


def test_dropped_port_registered_as_a_value_fails(runs: Path, tmp_path: Path):
    honest = runs / "honest"
    (flip_b,) = [n for n in json.loads((honest / "held.json").read_text())["names"] if n.startswith(f"{FLIP_B}_")]
    port = f"{flip_b}/o0"
    result = check(variant(tmp_path, honest, {"m1.bin": G.drop_port((honest / "m1.bin").read_bytes(), port)}))
    assert not result["recordsOk"] and not result["runOk"], result


def test_value_registered_once_passes(runs: Path, tmp_path: Path):
    result = registered(runs / "honest", tmp_path, 1)
    assert result["recordsOk"] and result["runOk"], result


def test_value_registered_twice_fails(runs: Path, tmp_path: Path):
    result = registered(runs / "honest", tmp_path, 2)
    assert not result["recordsOk"] and not result["runOk"], result
