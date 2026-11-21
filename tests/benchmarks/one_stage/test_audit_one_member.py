"""A window of one member through the records audit on the window's own challenger session, then the compiled check,
`spot-one-stage-zk check`, on the case the audit writes (``case-v2.json``). It is `test_audit_two_members`'s
audit (its `_audit` and `binaries`) on a window of one template and one proof block (``ScaledMmS8Coordinate`` at
K = 128) of one instance under ``subset:1``, the plain random subset the one-statement theorem covers (a design by
strata or by work is the several-statement form's), under FLOCK_CHALLENGER=1 and a local key at the deployed parameters
(``--deployed-key``), so the session takes its coin tree through the audit's relay with no closing round
(`a5.closes_trees`) and the case names the challenger and carries its coins.

The check accepts: the public file names its ports bare, the window's docket names them ``<member>/<port>``,
and v2 holds the one to the other at that one member's roots (`OneStageZk.publicPorts`).

Slow, and past a minute in `check`: one audit as `test_audit_two_members`' window-session test runs it, of one session,
about four minutes on a node with the deployed key. It alone runs the one-statement form of the compiled check on a
real audit's files with the session on the window's challenger, where every other test of that form gives a stand-in
for the prover, the verifier or the coins."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from verity.circuits.evaluation import evaluate_batch
from verity.commitments.periods import interface as IF
from verity.protocols.proof.spot.verifier.one_stage import docket as R
from verity.protocols.proof.spot.verifier.one_stage import proofs as PF
from verity.protocols.proof.spot.prover import docket as DK
from verity.commitments import randomness as RND

import test_audit_two_members as T
from test_audit_two_members import binaries  # noqa: F401 -- the fixture, under the name the test takes
from verity.protocols.proof.spot.verifier.one_stage import member_trees as MT  # beside a5, on the path `test_audit_two_members` put there


def fixed(w, seed: int = 0) -> dict[str, bytes]:
    """Salts a test hands in for every port a commit takes in full (`rows.full_ports`), so two commits of one period are
    one commitment."""
    from verity.commitments.periods import rows

    w = getattr(w, "window", w)
    return {p: hashlib.shake_256(f"{seed} {p}".encode()).digest(n * rows.SALT_BYTES) for p, n in rows.full_ports(w).items()}

pytestmark = pytest.mark.slow


#: The sampling design, the plain random subset, and the window's one instance, which its sample is
DESIGN, N = "subset:1", 1
#: the rate the audit is given, which names `DESIGN` over the window's one proof block
RATE = "rate:1/1"


def _window(d: Path) -> Path:
    t, k = T.TEMPLATES[0], T.KS[0]
    win = IF.Window(IF.Circuit("one-member", ((t, N),)), T.RUN, RND.parse_design(DESIGN), 4)
    rng = np.random.default_rng(0)
    x, w = (rng.integers(-128, 128, (N, k)).astype(np.int8) for _ in range(2))
    sx, sw = ((np.abs(rng.standard_normal(N)) / 127).astype(np.float32) for _ in range(2))
    ins = [x.view(np.uint8), sx.view(np.uint32), w.view(np.uint8), sw.view(np.uint32)]
    win.record(t, ins, [np.asarray(evaluate_batch(t, *ins, kernel="reference"), dtype=np.uint16)])
    win.write(win.close(salts_of=fixed(win)), d, {"label": "test"}, registrar=DK)
    return d


def test_one_member_on_the_windows_challenger_session_is_accepted_by_the_compiled_check(tmp_path, binaries):
    """The audit accepts with the member's ``challenger`` verdict; the release log holds the sample, then the session's
    one tree with no closing round, as the driver's check reads it; and Lean's check on the case accepts, crediting the
    drawn unit as proved."""
    window = _window(tmp_path / "window")
    assert T.RA.verifier_window(window, RATE, None)[0].law == RND.parse_design(DESIGN)
    rc, row, out, said = T._audit(tmp_path, window, binaries, "--deployed-key", more={"FLOCK_CHALLENGER": "1"},
                                  design=RATE)
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

    def check(params_: dict) -> tuple[int, dict, str]:
        files["params.json"] = json.dumps(params_, sort_keys=True, separators=(",", ":")).encode()
        (d / "params.json").write_bytes(files["params.json"])
        p = subprocess.run([str(PF.v2_executable()), "check", *(str(d / n) for n in files), *case["blobs"]],
                           capture_output=True, text=True)
        return p.returncode, json.loads(p.stdout.strip().splitlines()[-1]), p.stderr[-2000:]

    rc, answer, err = check(params)
    assert (rc, answer["verdict"]) == (0, "accept"), (json.dumps(answer, sort_keys=True)[-4000:], err)
    assert answer["public"]["drawn"] == answer["public"]["proved"] == [0], answer

    # the public file names the run input ports, which the registration holds only at their one tree each, and both
    # verifiers leave each out only as the registered file reads it there
    rec = json.loads(bytes.fromhex(case["commitment"]))
    registered = params["proofs"]["registered"]
    pub, = (Path(b) for b in case["blobs"] if PF.file_sha512(Path(b)) == params["proofs"]["public"])
    header = json.loads(pub.read_bytes().split(b"\n", 1)[0])
    fresh = [q for q in R.fresh_reads(registered) if q in header["frame_v3"]["roots"]]
    assert fresh, registered
    audit = json.loads((out / "window" / "audit.json").read_text())
    assert audit["checks"]["one_statement_public"]["ok"] is True and R.public_held(rec, header, registered).ok, audit["checks"]

    # with no registered file to read them, both refuse the public file on its roots
    assert R.public_held(rec, header, None).codes == ("roots",)
    rc, answer, err = check({**params, "proofs": {**params["proofs"], "registered": None}})
    assert answer["verdict"] != "accept" and "public file: not the registration's: roots" in answer["reason"], (answer, err)

    # a registered file opening a run input port under another root than its tree's, so a header root over other rows
    # than the registration's: refused by name
    q = fresh[0]
    other = {**registered, "ports": {**registered["ports"], q: {**registered["ports"][q], "root": "0" * 128}}}
    rc, answer, err = check({**params, "proofs": {**params["proofs"], "registered": other}})
    want = f"port {q} opens the committed tree {registered['ports'][q]['value']} at another root than the registration's"
    assert answer["verdict"] != "accept" and want in answer["reason"], (answer, err)
    why = R.registered_trees_held(rec, "the verifier", other, R.one_template(rec))
    assert why is not None and why in answer["reason"], (why, answer["reason"])
    assert R.registered_trees_held(rec, "the verifier", registered, R.one_template(rec)) is None
