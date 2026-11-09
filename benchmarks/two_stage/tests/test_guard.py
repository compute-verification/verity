"""SPOT[Flock]'s two-stage Lean model is not vacuous: `SpotFlock.stmtOk` (`Security/Proofs/Flock/SpotFlock/Code.lean`)
accepts an honest two-stage run of the toy at a `Held` fixed before the run, and refuses two runs it must refuse.

What it shows: on the honest run's messages (the boundary record x, the interior record m₁, its four session statements
m₂ in template order, the two stages' coins c₀ and c₁), `stmtOk h x m₁ m₂ c₀ c₁` evaluates to true in Lean, at an `h`
that `TwoGuard.lean` builds only from what the auditor registers before the run: the boundary registration's program,
query, work table and selection rule, the counts, the templates' names, links and descriptor (`held.json`), and the
templates' reference statements, made from the interior window with no run in it and admitted only after the script
checks each is in Flock's masked scope (`zkOk`). Nothing in `h` reads the run's messages or coins. So neither the model's
second-stage registration check (`two`) nor `stmtOk` refuses everything, and a guarantee over `stmtOk` (`SpotKSTwo`) is
not vacuous. It refuses the altered run and the honest run with one byte of a session statement's registered reads
flipped.

What it doesn't show: that each session accepts. `stmtOk` reads the session statements and their registered reads, never
a session's transcript; each session's acceptance is `flock-verify`'s verdict in the run that made the fixture
(`audit.py`), whose statements these are byte for byte (`guard.session_statement` rebuilds each from its command line
and archive).

Opt-in: ``VERITY_TWO_STAGE_GUARD`` names the command that runs `TwoGuard.lean` (`twoguard`), which `guard_node.py
--pytest` makes on a node's warm Lean tree and runs this with. The fixture is a run's `guard/` export (`guard.py`) for both cases."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EXE = os.environ.get("VERITY_TWO_STAGE_GUARD")
FIXTURE = HERE / "fixtures" / "guard.tar.xz"
pytestmark = pytest.mark.skipif(not EXE, reason="VERITY_TWO_STAGE_GUARD names the command that runs TwoGuard.lean "
                                "(guard_node.py --pytest makes it)")


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("two-stage-guard")
    with tarfile.open(FIXTURE) as tar:
        tar.extractall(out, filter="data")
    return out


def stmt_ok(run: Path, m2: Path | None = None) -> dict:
    argv = [EXE, str(run), "check", *([str(m2)] if m2 else [])]
    p = subprocess.run(argv, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("GUARD-RESULT ")]
    assert lines, f"TwoGuard.lean printed no result (exit {p.returncode}):\n{p.stdout[-4000:]}\n{p.stderr[-4000:]}"
    result = json.loads(lines[-1].removeprefix("GUARD-RESULT "))
    assert result["held"], f"a reference statement is out of Flock's masked scope:\n{p.stdout[-4000:]}"
    assert p.returncode == (0 if result["stmtOk"] else 1), p.stdout[-4000:]
    return result


def registered_span(st: bytes) -> tuple[int, int]:
    """Where a session statement's registered reads sit in its `FlockDeploy.Raw` bytes (`guard.raw`'s field order)."""
    i = 0

    def nat() -> int:
        nonlocal i
        i += 8
        return int.from_bytes(st[i - 8:i], "little")

    def skip() -> None:
        nonlocal i
        n = nat()
        i += n

    def opt(f) -> None:
        nonlocal i
        i += 1
        if st[i - 1]:
            f()

    for _ in range(4):
        skip()
    for _ in range(nat()):
        skip(), skip()
    i += 2
    nat()
    for f in (skip, skip, skip, nat, skip, nat, skip):
        opt(f)
    assert st[i] == 1, "the statement has no registered reads"
    i += 1
    n = nat()
    return i, i + n


def test_honest_run_passes(runs: Path):
    assert stmt_ok(runs / "honest")["stmtOk"]


def test_altered_run_fails(runs: Path):
    assert not stmt_ok(runs / "altered")["stmtOk"]


def test_one_registered_byte_flipped_fails(runs: Path, tmp_path: Path):
    m2 = bytearray((runs / "honest" / "m2.bin").read_bytes())
    start = 16
    lo, hi = registered_span(bytes(m2[start:start + int.from_bytes(m2[8:16], "little")]))
    reads = m2[start + lo:start + hi]
    at = reads.index(b'"root"')
    at = reads.index(b'"', at + len(b'"root"') + 1) + 1
    m2[start + lo + at] = ord("1") if reads[at] == ord("0") else ord("0")
    json.loads(m2[start + lo:start + hi])
    flipped = tmp_path / "m2-flipped.bin"
    flipped.write_bytes(m2)
    assert not stmt_ok(runs / "honest", flipped)["stmtOk"]
