"""The recursive audit's event and the premises it takes of V*'s sessions, evaluated on a recorded recursive session
(`flock-witness`, `Flock/Witness.lean`; `Proofs.Flock.Recursive.Witness` proves each check's `.ok` is the model's
predicate): the fetched fixture `rec-coins/16-16-8-n64` (`r20261009-111638-6720`'s inner statement and honest inner session,
and level 3's recorded `pub.bin`, `registered.json` and outer session), with level 3's statement restaged by
`rec_vstage.stage` as `test_lean_rec_coins` restages it. What a recorded honest session meets is asserted; what none meets yet
is asserted refused by name, so the change that closes it has to say so here: the inner description is not registered
(no staging runs `inner_registered register`), and V*'s sessions hide their outputs where `ZkOuterR.hTags` (`TagsOk`)
takes public ones. Staging the level takes about a minute and the session check sets up its 158 MB circuit, so the module
is slow; it catches what reading the theorems can't: a premise or event conjunct that no honest session meets."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from verity_flock import rec_vstage as S

HERE = Path(__file__).resolve().parent
LEAN = HERE.parents[2] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-witness"
SET = HERE / "fixtures" / "rec-coins" / "16-16-8-n64"
LEVEL = 3
#: the recorded level's circuit (`v/L3/circuit.txt`), which its outer session proves (`test_lean_rec_coins`)
CIRCUIT_SHA512 = ("4c47dec94a141028c1bb722b703f0c5d5624b736cfecbedb142cd0512c411e1fc91b0317d2de6a74c5897476217bd89904dabdcfc71e84"
                  "28ff1c6ecaa6c399f9")

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


def _witness(*args: str) -> dict[str, dict]:
    p = subprocess.run([str(BIN), *args], capture_output=True, text=True, check=False)
    checks = [json.loads(line[len("WITNESS "):]) for line in p.stdout.splitlines() if line.startswith("WITNESS ")]
    assert checks, p.stderr
    return {c["check"]: c for c in checks}


@pytest.fixture(scope="module")
def inner() -> dict[str, dict]:
    assert (SET / "u" / "stage" / "circuit.txt").is_file(), (
        f"{SET} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/rec-coins`")
    subprocess.run(["lake", "build", "flock-witness"], cwd=LEAN, check=True, capture_output=True)
    u = SET / "u" / "stage"
    return _witness("inner", "--circuit", str(u / "circuit.txt"), "--public", str(u / "pub-64.bin"), "--partition",
                    str(u / "partition.json"), "--record", str(SET / "inner-honest" / "session" / "session.json"))


@pytest.fixture(scope="module")
def outer(inner, tmp_path_factory) -> dict[str, dict]:
    d = tmp_path_factory.mktemp("rec-witness")
    S.stage(SET / "inner-honest", json.loads((SET / "fold-honest.json").read_text()), d, levels=[LEVEL], algebra=False,
            pts=json.loads((SET / "points-honest.json").read_text()), salts=SET / "acc-salts.bin")
    level = d / f"L{LEVEL}"
    assert hashlib.sha512((level / "circuit.txt").read_bytes()).hexdigest() == CIRCUIT_SHA512
    return _witness("session", "--statement", "verity/flock-circuit", "--circuit", str(level / "circuit.txt"), "--public",
                    str(SET / f"L{LEVEL}" / "pub.bin"), "--partition", str(level / "partition.json"), "--program",
                    str(level / "program.json"), "--registered", str(SET / f"L{LEVEL}" / "registered.json"), "--session",
                    str(SET / f"outer-L{LEVEL}" / "session"))


def test_lean_accepts_the_inner_statement(inner):
    """`AcceptsStmt`: `setupTables` at the inner tags sets the statement up as one table, and `Zk.schedule` takes its m."""
    assert inner["accepts_stmt"]["ok"], inner["accepts_stmt"].get("why")


def test_the_inner_session_is_the_inner_statements(inner):
    assert inner["record_sigma"]["ok"], inner["record_sigma"].get("why")


def test_no_recorded_inner_description_is_registered(inner):
    """`AcceptsDescription`, the event's second conjunct, at no recorded session: the inner statement's header names no
    registered rows. It holds once a staging runs `inner_registered register` on the inner statement."""
    c = inner["accepts_description"]
    assert not c["ok"]
    assert c["why"] == ("registered reads: the inner statement names no registered rows: its description committed afresh "
                        "opens no registered root")


@pytest.mark.parametrize("check", ["scope_z", "scope_zr", "scope_ok", "full_draw", "accepts_zkr"])
def test_the_recorded_level_meets(outer, check):
    """`ScopeZ` and `ScopeZR` (the statement check without and with the registered reads), `VStmt.hscope`, the full draw
    `RecordCustody` reads, and `AcceptsZKR`."""
    assert outer[check]["ok"], outer[check].get("why")


def test_the_recorded_level_hides_its_outputs(outer):
    """`ZkOuterR.hTags` (`TagsOk`) takes public outputs, but every V* session is `verity/flock-circuit`'s, which hides
    them. It holds once the chain's session lemmas take hidden outputs (`TagsOkZHJ`)."""
    c = outer["tags_ok"]
    assert not c["ok"]
    assert c["why"].startswith("verity/flock-circuit: hidden outputs true,"), c["why"]
