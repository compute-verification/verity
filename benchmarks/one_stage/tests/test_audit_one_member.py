"""A window of one member through the records audit on the window's own challenger session, then the compiled check,
`sampled-proofs-one-stage-zk check`, on the case the audit writes (``case-v2.json``). It is `test_audit_two_members`'s
audit (its `_audit` and `binaries`) on a window of one template and one proof block (``ScaledMmS8Coordinate`` at
K = 128) under ``stratified:1``, under FLOCK_CHALLENGER=1 and a local key at the deployed parameters
(``--deployed-key``), so the session takes its coin tree through the audit's relay with no closing round
(`a5.closes_trees`) and the case names the challenger and carries its coins.

The check is expected to accept. One refusal is known and stays until v2's Lean takes a window's registration in the
one-statement form: `OneStageZk.publicHeld` holds the public file, whose ports are bare, to the whole registration,
whose ports a window names ``<member>/<port>`` (`PUBLIC_PORTS`). That refusal alone is an expected failure; any other
outcome fails.

Slow, and past a minute in `check`: one audit as `test_audit_two_members`' window-session test runs it, of one session,
about four minutes on a node with the deployed key. It alone runs the one-statement form of the compiled check on a
real audit's files with the session on the window's challenger, where every other test of that form gives a stand-in
for the prover, the verifier or the coins."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from verity.circuits.evaluation import evaluate_batch
from verity.commitments.windows import interface as IF
from verity.protocols.spot.verifier.one_stage import draw as D, proofs as PF, registration as R

import test_audit_two_members as T
from test_audit_two_members import binaries  # noqa: F401 -- the fixture, under the name the test takes
import member_trees as MT  # beside a5, on the path `test_audit_two_members` put there

#: v2's refusal of a window's case in the one-statement form (module docstring)
PUBLIC_PORTS = "proofs: public file: not the registration's: roots, domains"

pytestmark = pytest.mark.slow


def _window(d: Path) -> Path:
    t, k = T.TEMPLATES[0], T.KS[0]
    win = IF.Window(IF.Circuit("one-member", ((t, 1),)), T.RUN, D.parse_design(T.DESIGN), 4)
    rng = np.random.default_rng(0)
    x, w = (rng.integers(-128, 128, (1, k)).astype(np.int8) for _ in range(2))
    sx, sw = ((np.abs(rng.standard_normal(1)) / 127).astype(np.float32) for _ in range(2))
    ins = [x.view(np.uint8), sx.view(np.uint32), w.view(np.uint8), sw.view(np.uint32)]
    win.record(t, ins, [np.asarray(evaluate_batch(t, *ins, kernel="reference"), dtype=np.uint16)])
    win.write(win.close(salt_key=bytes(32)), d, {"label": "test"}, registrar=R)
    return d


def test_one_member_on_the_windows_challenger_session_is_accepted_by_the_compiled_check(tmp_path, binaries):
    """The audit accepts with the member's ``challenger`` verdict; the release log holds the sample, then the session's
    one tree with no closing round, as the driver's check reads it; and Lean's check on the case accepts, or refuses
    for `PUBLIC_PORTS` alone."""
    rc, row, out, said = T._audit(tmp_path, _window(tmp_path / "window"), binaries, "--deployed-key",
                                  more={"FLOCK_CHALLENGER": "1"})
    assert rc == 0 and row.get("ok") is True, said
    assert row["draw"] == [0] and row["audit"]["accepted"] is True, (row["audit"], said)
    assert row["audit"]["verdicts"] == {f"{v}/m0": True for v in (*T.VERDICTS, "challenger")}
    assert row["coins_session"]["error"] is None, row["coins_session"]

    case = json.loads((out / "window" / "case-v2.json").read_text())
    params, evidence = case["params"], json.loads(bytes.fromhex(case["evidence"]))
    assert sorted(params["challenger"]) == ["key", "label"] and len(params["window_root"]) == 128
    session, = sorted((out / "window" / "session-m0" / "sessions").glob("*/session.json"))
    assert evidence["sessions"] == [PF.file_sha512(session)]
    log = evidence["coins"]["session"]["log"]
    assert MT.trees_reason(log, [(evidence["sessions"][0], session.parent)], closed=False) is None

    d = tmp_path / "case"
    d.mkdir()
    files = {"params.json": json.dumps(params, sort_keys=True, separators=(",", ":")).encode(),
             "commitment.bin": bytes.fromhex(case["commitment"]), "evidence.json": bytes.fromhex(case["evidence"])}
    for name, b in files.items():
        (d / name).write_bytes(b)
    p = subprocess.run([str(PF.v2_executable()), "check", *(str(d / n) for n in files), *case["blobs"]],
                       capture_output=True, text=True)
    answer = json.loads(p.stdout.strip().splitlines()[-1])
    if (p.returncode, answer["verdict"], answer.get("reason")) == (1, "reject", PUBLIC_PORTS):
        pytest.xfail(f"v2 refuses a window's registration in the one-statement form: {PUBLIC_PORTS}")
    assert (p.returncode, answer["verdict"]) == (0, "accept"), (answer, p.stderr[-2000:])
    assert answer["public"]["drawn"] == [0] and answer["public"]["proved"] == [0], answer
