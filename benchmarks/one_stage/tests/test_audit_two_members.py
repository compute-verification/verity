"""Two sampled members of a tiny committed window through the records audit as a served window's audit runs it
(`served_verdicts.sh`'s ``v3c_audit``): ``records_audit.py audit`` in a process of its own, which samples on its stream
and hands `a5.audit` the members, each proved by Flock's prover (`flock-circuit`, the CPU build) behind the ZK server
and verified by the Lean verifier. Nothing on that path is replaced.

* **The window** (`_window`): two templates of one proof block each, `ScaledMmS8Coordinate` at K = 128 and K = 256 (the
  smallest whose rows give Flock's composer a compression before a row's last block), recorded, committed and written
  through the interface as serving writes one, under ``stratified:1``, the design the audit's rule (`RULE`, a stratified
  rate) names over its two proof blocks. A template is a stratum, so the sample always holds both proof blocks, one a
  member, and the audit runs two sessions.
* **The coins** are a challenger's: ``--challenger local`` starts a development session of the auditor host's own
  service on this machine (`a5_coins.Local`: `live_challenger`'s reserve, admit, run and `serve.py`), pinned on
  this checkout's first-parent line. `test_a5_reserve` opens one the same way under a stub of the Lean verifier's coins;
  here its `serve.py` runs SPOT's Lean verifier itself. Its key is the session's own, so its coins show the path and
  are no auditor's.
* **The sessions' coin trees.** Under FLOCK_CHALLENGER=1 the two sessions run on that same challenger session
  (`a5.window_trees`): each verifier takes its tree through the audit's relay, and each session's tree is closed
  there after its evidence before the next session starts. The audit then writes its case in v2's per-member form
  with the challenger's key, the window's root and the coins, and v2 refuses it for its form alone
  (`members.ALL_PASSED`). v2 reads a key at the deployed parameters only, so that audit's local key is made at them
  (``--deployed-key``), over the machine's cores.
* **The binaries.** The Lean verifier and the ZK server's Lean are what check's lean-build step builds before this
  suite (`AFTER_LEAN`, the suite's ``artifacts``), and the tests skip without them. `flock-circuit` is built by
  `check_build.sh bin` on the Flock checkout check's flock-circuit-build step leaves on a machine. That build fetches
  nothing: on a machine with no checkout, not its Rust toolchain or not its crates the tests skip, saying so. The
  stream's and the challenger's Lean, SPOT's verifier, is its own package's, which the audit builds with `lake` on
  first use (`one_stage.verifier.executable`): a checkout's first audit pays that build.

Slow, and past a minute in `check`: about 150 s on a node, about 210 s where the audit builds the coins' Lean first,
and about 5 GB at its peak, which the suite's ``worker_gb`` declares. Of it the changed member's audit is about 50 s:
it alone shows a false member stopped (the ZK server lets no proof of it out, and the Lean verifier refuses its
session), where the audit's own negatives change the record around honest sessions. The audit on the window's
challenger session is a third run of the window, and its key about fifteen core-minutes: it alone runs Flock's own
`serve --tree-from` client against the relay, the challenger's service on a session held open through the proving,
and Lean's whole per-member check (`OneStageZk.membersRun`, then v1's check at the challenger's coins) on a window's
real files, where every other test of those gives one of them a stand-in. A pass is reused while the suite's
inputs are unchanged, so it reruns only on a change to what `benchmarks/one_stage` reads. Every other test of the audit
gives `a5.audit`, the prover or the verifier a stand-in, so none sees what only the real ones refuse: a rule the
stream's service doesn't take, a statement the prover can't stage or the Lean verifier won't set up, a session the ZK
server stops, or a record a later step can't read. A served window's audit finds those hours in."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.commitments import coin_tree as CT
from verity.protocols.spot.device import challenger as C
from verity.protocols.spot.verifier import service as S
from verity.protocols.spot.verifier.one_stage import draw as D, members as M, proofs as PF, registration as R
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.definitions.ops import gemm as G
from verity.commitments.windows import interface as IF

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "prove_path"))
sys.path.insert(0, str(HERE))
import member_trees as MT  # noqa: E402
import records_audit as RA  # noqa: E402

LEAN = ROOT / "verity" / "protocols" / "flock" / "verifier" / ".lake" / "build" / "bin"
BUILD = ROOT / "verity" / "protocols" / "flock" / "prover" / "check_build.sh"
#: `check_build.sh bin`'s exit where the machine has no Flock checkout, not its toolchain or not its crates
NOTHING_TO_BUILD_ON = 3
KS = (128, 256)
TEMPLATES = tuple(bind(G.ScaledMmS8Coordinate, K=k, **G.GEMM_SM120_S8) for k in KS)
DESIGN = "stratified:1"
RULE = "stratified-rate:1/2"
RUN = "ab" * 64
#: what the steps after a served window's audit read of its record (`served_verdicts.sh`'s hand-off, `records_audit.period`)
RECORD = ("summary.json", "window/audit.json", "window/registration.json", "window/program.json", "window/case.json")
#: a member's verdicts in the audit record: the prover's own verifier's, Lean's of the session, and Lean's of its sample
VERDICTS = ("m0-verifier", "lean-verify", "lean-draw")

pytestmark = pytest.mark.slow


def _window(d: Path, flipped: bool = False) -> Path:
    """The window (module docstring) written to ``d``. ``flipped``: the K = 256 proof block's committed output has bit 0
    of its first word flipped, and is committed as consistently as an honest one."""
    win = IF.Window(IF.Circuit("two-members", tuple((t, 1) for t in TEMPLATES)), RUN, D.parse_design(DESIGN), 4)
    rng = np.random.default_rng(0)
    for t, k in zip(TEMPLATES, KS):
        x, w = (rng.integers(-128, 128, (1, k)).astype(np.int8) for _ in range(2))
        sx, sw = ((np.abs(rng.standard_normal(1)) / 127).astype(np.float32) for _ in range(2))
        ins = [x.view(np.uint8), sx.view(np.uint32), w.view(np.uint8), sw.view(np.uint32)]
        out = np.asarray(evaluate_batch(t, *ins, kernel="reference"), dtype=np.uint16)
        if flipped and k == KS[-1]:
            out.reshape(-1)[0] ^= 1
        win.record(t, ins, [out])
    win.write(win.close(salt_key=bytes(32)), d, {"label": "test"}, registrar=R)
    return d


@pytest.fixture(scope="module")
def binaries(tmp_path_factory) -> dict[str, Path]:
    """The prover, the Lean verifier and the ZK server's Lean (module docstring)."""
    lean = {"lean": LEAN / "flock-verify", "firewall": LEAN / "flock-firewall"}
    if not all(p.is_file() for p in lean.values()):
        pytest.skip(f"no {' or '.join(p.name for p in lean.values())} under {LEAN}: check's lean-build step builds them")
    prover = tmp_path_factory.mktemp("flock-circuit") / "flock-circuit"
    p = subprocess.run(["bash", str(BUILD), "bin", str(prover)], capture_output=True, text=True)
    if p.returncode == NOTHING_TO_BUILD_ON:
        pytest.skip(p.stderr.strip())
    assert p.returncode == 0 and prover.is_file(), f"check_build.sh bin: {p.stdout[-2000:]}\n{p.stderr[-4000:]}"
    return {"bin": prover, **lean}


def _audit(tmp: Path, window: Path, binaries: dict[str, Path], *flags: str,
           more: dict[str, str] | None = None) -> tuple[int, dict, Path, str]:
    """``records_audit.py audit`` of ``window`` into ``tmp/audit`` as `v3c_audit` runs it, on a local challenger's coins,
    with ``more`` added to its environment -> (its exit code, its summary, its out directory, the end of its output for
    a failure's message)."""
    out, main = tmp / "audit", tmp / "main.txt"
    main.write_text(subprocess.run(["git", "-C", str(ROOT), "rev-list", "--first-parent", "HEAD"], capture_output=True,
                                   text=True, check=True).stdout)
    m0 = os.pathsep.join(str(ROOT / p) for p in ("", "benchmarks/numerical/python",
                                                 "verity/protocols/flock/prover/python", "adapters/vllm"))
    argv = [sys.executable, str(ROOT / "benchmarks" / "prove_path" / "records_audit.py"), "audit", "--window", str(window),
            "--law", RULE, "--circuit", hashlib.sha512((window / "circuit.json").read_bytes()).hexdigest(), "--run", RUN,
            "--challenger", "local", "--main", str(main), "--m0-pythonpath", m0, *(x for k, p in binaries.items()
                                                                                    for x in (f"--{k}", str(p))),
            "--ports", "2", "--out", str(out), *flags]
    # no setting of the prover's or the sessions' that the caller's environment happens to hold (FC_*, FLOCK_CHALLENGER)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FC_", "FLOCK_"))}
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                       env={**env, "RAYON_NUM_THREADS": "4", "RUST_MIN_STACK": str(64 << 20), **(more or {})})
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").is_file() else {}
    return p.returncode, summary, out, f"{p.stdout[-6000:]}\n{p.stderr[-6000:]}"


def test_two_sampled_members_pass_the_audit_on_the_real_prover_and_verifier(tmp_path, binaries):
    """Both members are sampled, each one's session is accepted by the prover's own verifier and by Lean's, the audit
    record accepts with every negative refused, and the record holds the files the later steps read. The stream's rule is
    the rate the audit is given (``--law``), which the service takes where it and the audit refuse the design's fixed
    count by name; over the window's two proof blocks that rate names the design the window was committed under
    (`records_audit.verifier_window`), and the stream sampled by it accepts its outcome on the challenger's coins. The
    audit record names whose coins those are, the key and the session the challenger's ledger opens on the run's root,
    and says it wrote no case for the compiled check; sampled proofs' answer on the audit accepts with both in its
    public outputs."""
    window = _window(tmp_path / "window")
    assert RA.verifier_window(window, RULE, None)[0].law == D.parse_design(DESIGN) and S.Law(RULE).text == RULE
    with pytest.raises(ValueError, match="a fixed count"):
        RA.verifier_window(window, DESIGN, None)
    rc, row, out, said = _audit(tmp_path, window, binaries)
    assert rc == 0 and row.get("ok") is True, said
    assert row["refused"] is None and row["population"] == 2 and row["draw"] == [0, 1], row
    assert [(m["template"], m["n"]) for m in row["members"]] == [(descriptor_id(t), 1) for t in TEMPLATES]
    assert row["audit"]["accepted"] is True, row["audit"]
    assert row["audit"]["verdicts"] == {f"{v}/m{k}": True for v in VERDICTS for k in (0, 1)}
    assert row["negatives"] and all(row["negatives"].values()), row["negatives"]
    outcome = row["coins"]["outcome"]
    assert (outcome["accepted"], outcome["codes"], row["coins"]["coins_reason"]) == (True, [], None), row["coins"]
    assert row["coins_session"]["error"] is None and row["coins_session"]["trusted"]["pinned_session"] is None
    # whose coins: the one session the local challenger's own ledger opens on the run's root, under that session's key,
    # as the coins' check verified them; sampled proofs' answer carries them, and that the audit, of two members, wrote
    # no case for the compiled check
    opened, = (e for e in map(json.loads, (out / "challenger" / "windows.jsonl").read_text().splitlines())
               if e["kind"] == "open")
    whose = {"key_sha512": opened["body"]["key_sha512"], "session": opened["session"]}
    assert opened["body"]["window"] == {"root": RUN} and len(whose["key_sha512"]) == 128
    assert row["coins"]["verified"] == whose
    record = json.loads((out / "window" / "audit.json").read_text())
    assert record["coins"]["verified"] == whose
    assert record["case_v2"]["written"] is False and "this audit drew 2 members" in record["case_v2"]["why"]
    answer = _answer(out)
    assert (answer["verdict"], answer["public"]["coins"]) == ("accept", whose), answer
    assert answer["public"]["case_v2"] == record["case_v2"]
    assert row["fixed_before_draw"]["ok"] is True
    for k in range(len(TEMPLATES)):
        session, = sorted((out / "window" / f"session-m{k}" / "sessions").glob("*/session.json"))
        assert json.loads(session.read_text())["verdict"]["accepted"] is True
    missing = [f for f in RECORD if not (out / f).is_file()]
    assert not missing, missing
    # v2's case is one table's session, so an audit of two members writes why it has none, which SPOT's row reads
    assert not (out / "window" / "case-v2.json").exists()
    assert "this audit drew 2 members" in (out / "window" / "case-v2-error.txt").read_text()


def test_two_members_sessions_close_on_the_windows_challenger_session_and_v2_refuses_the_form_alone(tmp_path, binaries):
    """Under FLOCK_CHALLENGER=1 the audit's two sessions take their coin trees from the session the sample's coins came
    from, and each is closed there after its evidence: the audit accepts with each member's ``challenger`` verdict, the
    release log holds the sample, then each session's tree, its rounds and its closing round, in the sessions' order,
    and the per-member case names the challenger and carries its coins. On that case Lean's check passes everything
    before the per-member form's refusal and answers `MULTI_REFUSED`, with exit 1; with one proof file of the first
    session changed after its closing round, it refuses the session's coins as the driver's check of the log does."""
    window = _window(tmp_path / "window")
    rc, row, out, said = _audit(tmp_path, window, binaries, "--deployed-key", more={"FLOCK_CHALLENGER": "1"})
    assert rc == 0 and row.get("ok") is True, said
    assert row["audit"]["accepted"] is True, (row["audit"], said)
    assert row["audit"]["verdicts"] == {f"{v}/m{k}": True for v in (*VERDICTS, "challenger") for k in (0, 1)}
    assert row["coins_session"]["error"] is None, row["coins_session"]
    case = json.loads((out / "window" / "case-v2-members.json").read_text())
    params, evidence = case["params"], json.loads(bytes.fromhex(case["evidence"]))
    assert sorted(params["challenger"]) == ["key", "label"] and len(params["window_root"]) == 128
    assert sorted(evidence["coins"]) == ["ledger", "main", "pins", "segment", "session"]
    sessions = [s for st in evidence["proofs"]["statements"] for s in st["sessions"]]
    dirs = [f.parent for k in range(len(TEMPLATES))
            for f in sorted((out / "window" / f"session-m{k}" / "sessions").glob("*/session.json"))]
    assert [s["name"] for s in sessions] == evidence["sessions"] == [PF.file_sha512(d / "session.json") for d in dirs]
    log = evidence["coins"]["session"]["log"]
    closing = [x for x in map(json.loads, log) if x.get("event") == "tree-open" and x.get("round") == C.CLOSING_ROUND
               and x.get("stream") == CT.SESSION]
    assert [(x["n"], x["after"]) for x in closing] == [(0, MT.closing_after(d).hex()) for d in dirs]
    named = list(zip(evidence["sessions"], dirs))
    assert MT.trees_reason(log, named) is None and MT.trees_reason(log, named[::-1]) == C.MEMBERS_ORDER

    assert M.case_refusal(case) == (M.ALL_PASSED, True)

    rep = sorted(dirs[0].glob("*.bin"))[0]
    changed = tmp_path / "changed" / rep.name
    changed.parent.mkdir()
    changed.write_bytes(rep.read_bytes() + b"\0")
    sessions[0]["reps"][rep.name] = PF.file_sha512(changed)
    other = {**case, "blobs": [str(changed) if Path(b) == rep else b for b in case["blobs"]],
             "evidence": json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode().hex()}
    assert str(rep) in case["blobs"] and other["blobs"] != case["blobs"]
    why = f"session {evidence['sessions'][0]}'s coins are not the challenger's: {C.NO_CLOSING}"
    assert M.case_refusal(other) == (f"proofs: {why}", True)


def test_a_member_with_one_bit_changed_is_refused(tmp_path, binaries):
    """One bit of one member's committed output flipped, committed as consistently as an honest window (the prover's
    self-check off, ``--forged``): every check before the sample passes, both members are sampled, the changed member's
    session ends with no proof and each of its verdicts refuses, the other member's accept, and the audit record
    rejects."""
    window = _window(tmp_path / "window", flipped=True)
    rc, row, out, said = _audit(tmp_path, window, binaries, "--forged")
    assert rc == 0 and row.get("ok") is True, said
    assert row["forged"]["checks"] == {"checks_before_draw": True, "rejected": True, "lean_refused": True}, row["forged"]
    assert row["draw"] == [0, 1] and row["audit"]["accepted"] is False
    assert _answer(out)["verdict"] == "reject"
    assert row["audit"]["verdicts"] == {f"{v}/m{k}": k == 0 for v in VERDICTS for k in (0, 1)}, row["audit"]
    assert sorted(row["audit"]["why"]) == sorted(f"{v}/m1" for v in VERDICTS), row["audit"]["why"]
    for k, proofs in ((0, True), (1, False)):
        session, = sorted((out / "window" / f"session-m{k}" / "sessions").glob("*/session.json"))
        record = json.loads(session.read_text())
        assert (record["verdict"]["accepted"], bool(record["proof_sha512"])) == (proofs, proofs), session


def _answer(out: Path) -> dict:
    """Sampled proofs' answer on the audit under ``out`` (`served_integrity.answer`, as the served run's record and the
    certifier's period read it), its registration the one the audit registered."""
    import served_integrity as SI  # beside a5, on the path `records_audit` put there

    return SI.answer(out, json.loads((out / "window" / "registration.json").read_text()))
