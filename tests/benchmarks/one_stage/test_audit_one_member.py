"""A window of one member through the records audit on the window's own challenger session, then the compiled check,
`spot-one-stage-zk check`, on the case the audit writes (``case-v2.json``). It is `test_audit_two_members`'s
audit (its `_audit` and `binaries`) on a window of one template and one proof block (``ScaledMmS8Coordinate`` at
K = 128) of one instance under ``subset:1``, the plain random subset the one-statement theorem covers (a design by
strata or by work is the several-statement form's), under FLOCK_CHALLENGER=1 and a local key at the deployed parameters
(``--deployed-key``), so the session takes its coin tree through the audit's relay with no closing round
(`a5.closes_trees`) and the case names the challenger and carries its coins.

Both verifiers refuse the run, by one reason: the member's own output is read at its template's field tree, which
the one statement holds no tie to yet (`WIRED_TREE_WHY`). The accept returns with that tie.

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


def test_one_member_on_the_windows_challenger_session_is_refused_by_both_at_its_own_output_tree(tmp_path, binaries):
    """The release log holds the sample, then the session's one tree with no closing round, as the driver's check
    reads it. Both verifiers hold the public file's run input ports by the registered file's reads of their one trees
    (`OneStageZk.publicHeld`, `docket.public_held`), and both refuse the run at its member's own output read at its
    template's field tree (`WIRED_TREE_WHY`, `docket.registered_trees_held`): the one statement holds no field tree
    yet, and the accept returns with the tie of a reader's rows to its writer's tree. Both refuse a public file with no
    registered file on its roots, and a registered file opening a run input port under another root than its tree's
    by name (`TREE_OTHER`)."""
    window = _window(tmp_path / "window")
    assert T.RA.verifier_window(window, RATE, None)[0].law == RND.parse_design(DESIGN)
    rc, row, out, said = T._audit(tmp_path, window, binaries, "--deployed-key", more={"FLOCK_CHALLENGER": "1"},
                                  design=RATE)
    assert row["draw"] == [0] and row["audit"]["accepted"] is False, (row.get("audit"), said)
    assert row["coins_session"]["error"] is None, row["coins_session"]
    audit = json.loads((out / "window" / "audit.json").read_text())
    held = audit["checks"]["one_statement_public"]
    assert held["codes"] == ["registered-trees"], held

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

    rec = json.loads(bytes.fromhex(case["commitment"]))
    registered = params["proofs"]["registered"]
    rc, answer, err = check(params)
    assert answer["verdict"] != "accept" and held["detail"]["registered-trees"] in answer["reason"], (answer, held, err)
    assert ("the field tree it opens is neither its own output's nor one C wires into it"
            in answer["reason"]), answer["reason"]

    # the public file names the run input ports, which the registration holds only at their one tree each, and both
    # leave each out only as the registered file reads it there: the executable refused past publicHeld
    pub, = (Path(b) for b in case["blobs"] if PF.file_sha512(Path(b)) == params["proofs"]["public"])
    header = json.loads(pub.read_bytes().split(b"\n", 1)[0])
    fresh = [q for q in R.fresh_reads(registered) if q in header["frame_v3"]["roots"]]
    assert fresh and R.public_held(rec, header, registered).ok, registered

    # with no registered file to read them, both refuse the public file on its roots
    assert R.public_held(rec, header, None).codes == ("roots",)
    rc, answer, err = check({**params, "proofs": {**params["proofs"], "registered": None}})
    assert answer["verdict"] != "accept" and "public file: not the registration's: roots" in answer["reason"], (answer, err)

    # a registered file, its field-tree reads left out, opening a run input port under another root than its tree's,
    # so a header root over other rows than the registration's: refused by both by name
    ports = {n: e for n, e in registered["ports"].items() if not R.field_tree_name(e["value"])}
    q = fresh[0]
    other = {**registered, "ports": {**ports, q: {**ports[q], "root": "0" * 128}}}
    why = R.registered_trees_held(rec, "the verifier", other)
    assert why is not None and "at another root than the registration's" in why, why
    rc, answer, err = check({**params, "proofs": {**params["proofs"], "registered": other}})
    assert answer["verdict"] != "accept" and why in answer["reason"], (why, answer, err)
    assert R.registered_trees_held(rec, "the verifier", {**registered, "ports": ports}) is None
