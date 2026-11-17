"""What the node side keeps and records: the accepted timed session, the result's files, a period or a quarter, and a
statement on the node or from the store."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pytest
from steady_standins import (
    ART,
    BURSTY,
    CIRCUIT,
    FULL,
    IN_ART,
    PIN,
    POINT,
    QUARTER,
    SPEC,
    TIMED,
    Node,
    stored,
)
from test_steady import END, SESSIONS, START

from verity_bench import settings
from verity_bench import steady as S
from verity_bench import steady_node as N
from verity_bench import steady_spec as SP


def filed(step: Path, name: str, label: str, *, mtime: int, warm: bool = False, accepted: bool = True,
          circuit: str = PIN, said: int = 1, proof: bytes = b"rep0") -> None:
    """A session a verifier filed under `step` (its four files), its verifier's SESSION lines (`said` of them carry
    its timings) and its row of sessions.jsonl."""

    session = step / "verifier-0" / name
    session.mkdir(parents=True)
    (session / "circuit.rep0.bin").write_bytes(proof)
    (session / "circuit.rep1.bin").write_bytes(b"rep1")
    (session / "verifier.json").write_text("{}")
    timings = {"handle_s": 1.5, "verify_s": float(mtime)}
    (session / "session.json").write_text(json.dumps({
        "verdict": {"accepted": accepted, **timings, "reps": [{"ok": accepted}, {"ok": True}]}, "aborted": False,
        "proof_sha512": [{"stream": f"circuit/rep{i}", "bytes": 4, "sha512": hashlib.sha512(b"rep%d" % i).hexdigest()}
                         for i in (0, 1)]}))
    for path in session.iterdir():
        os.utime(path, (mtime, mtime))
    os.utime(session, (mtime, mtime))
    with (step / "verifier-0.log").open("a") as log:
        log.writelines(f"SESSION\t{json.dumps({'label': label, 'accepted': accepted, **timings})}\n" for _ in range(said))
    with (step / "sessions.jsonl").open("a") as rows:
        rows.write(json.dumps({"error": None, "live": {"label": label, "accepted": accepted, "warm": warm,
                                                       "circuit_sha512": circuit}}) + "\n")


def test_one_accepted_timed_session_is_kept_for_the_lean_verifier_with_its_label_and_proofs(tmp_path):
    """The kept proof is what the Lean verifier is handed: a warm, refused or other-circuit session, or one whose
    files aren't the proofs it recorded, would be a verdict on something the row didn't time."""

    step, kept = tmp_path / "rate" / "gemm64-k2", tmp_path / "out" / "gemm64-k2"
    step.mkdir(parents=True)
    kept.mkdir(parents=True)
    filed(step, "s1", "gemm64/k2/3", mtime=100)
    filed(step, "s2", "gemm64/k2/4", mtime=200, warm=True)
    assert N.keep_session(step, kept, PIN) == {
        "session": "gemm64-k2/kept-session", "label": "gemm64/k2/3",
        "proof_sha512": json.loads((step / "verifier-0" / "s1" / "session.json").read_text())["proof_sha512"]}
    assert sorted(p.name for p in (kept / N.KEPT_SESSION).iterdir()) == sorted(N.SESSION_FILES)


@pytest.mark.parametrize("change, says", [
    ({"warm": True}, "its row in sessions.jsonl isn't a timed, accepted session without an error"),
    ({"accepted": False}, "it wasn't accepted with every repetition"),
    ({"circuit": "0" * 128}, "its circuit isn't the row's pin"),
    ({"said": 2}, "2 of the verifiers' 2 SESSION lines carry its timings"),
    ({"proof": b"rep9"}, "circuit.rep0.bin isn't the proof the session recorded for circuit/rep0"),
])
def test_no_session_is_kept_where_none_is_an_accepted_timed_one_of_the_pin_and_the_record_says_why(tmp_path, change,
                                                                                                   says):
    step, kept = tmp_path / "rate" / "gemm64-k2", tmp_path / "out" / "gemm64-k2"
    step.mkdir(parents=True)
    kept.mkdir(parents=True)
    filed(step, "s1", "gemm64/k2/3", mtime=100, **change)
    record = N.keep_session(step, kept, PIN)
    assert record["session"] is None and record["why"] == f"verifier-0/s1: {says}"
    assert not (kept / N.KEPT_SESSION).exists()
    assert N.keep_session(tmp_path / "none", kept, PIN) == {"session": None, "why": "the verifiers filed no session"}


def test_the_result_lists_every_file_plan_judges_the_run_by(tmp_path, monkeypatch):
    """The store keeps the files a run's result lists (`artifacts` and `measurement_files`) and no other, so a verdict
    recomputed from the store reads only those: a result that listed only its records left a passed run "cannot tell"."""

    import shutil

    node = Node(tmp_path, monkeypatch)
    assert node.point() == 0
    result = node.read("result.json")
    listed = sorted({*result["artifacts"], *result["measurement_files"]})
    assert {S.RATES, S.LEASES, START, END, SESSIONS, N.VERDICT_FILE, N.POINT_FILE, N.TIMED_FILE} <= set(listed)
    assert all((node.run_dir / path).is_file() for path in listed)
    for path in listed:
        (tmp_path / "stored" / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(node.run_dir / path, tmp_path / "stored" / path)
    spec = SP.load_spec(node.root / node.spec, root=node.root)
    item, files = SP.point(spec, POINT), S.files_in(tmp_path / "stored")
    assert SP.point_why(SP.own_row(files, item), files, spec, item, own=True) is None


def test_the_node_side_times_a_row_with_a_period_at_the_points_count_and_records_the_period(tmp_path, monkeypatch):
    period = {"row0": {"period": 4}}
    node = Node(tmp_path, monkeypatch, spec=period, gaps=(BURSTY * 4)[:27])
    assert node.point() == 0
    assert node.timed[node.timed.index("--sessions") + 1] == "28", "not the spec's 16"
    [row] = node.read(S.RATES)
    assert row[S.PERIOD] == 4 and row["sessions"] == 28, "the field `level` reads"
    assert node.read(N.POINT_FILE)["point"] | {"period": 4, "sessions": 28} == node.read(N.POINT_FILE)["point"]

    short = Node(tmp_path / "short", monkeypatch, spec=period)
    assert short.point() == SP.NOT_PASSED, "a script that timed another count than the point's"
    assert short.read(N.VERDICT_FILE)["why"].startswith("fail: it timed 16 sessions after 2 warm ones a prover; the "
                                                        "point's are 28 and 2")
    plain = Node(tmp_path / "plain", monkeypatch)
    assert plain.point() == 0 and S.PERIOD not in plain.read(S.RATES)[0], "a row without a period records none"


def test_the_node_side_times_a_quarter_of_a_statement_named_beside_its_full_row(tmp_path, monkeypatch):
    rows = {"row": [FULL, QUARTER, SPEC["row"][1]]}
    node = Node(tmp_path, monkeypatch, spec=rows, point_id="gemm64-n2-k2", units=2, row=TIMED[2])
    assert node.point() == 0
    stager = node.ran[0]
    assert stager[2:4] == ["gemm64", "2"] and Path(stager[4]).parts[-3:] == ("gemm64", "c2", "stage")
    verdict = node.read(N.VERDICT_FILE)
    assert (verdict["point"], verdict["passed"], verdict["why"]) == ("gemm64-n2-k2", True, None)

    full = Node(tmp_path / "full", monkeypatch, spec=rows, point_id="gemm64-n2-k2", units=2, row=TIMED[8])
    assert full.point() == SP.NOT_PASSED, "a script that timed the full statement for the quarter"
    assert full.read(N.VERDICT_FILE)["why"] == "fail: its statement holds 8 blocks; the point's `n` is 2"


def test_a_statement_already_on_the_node_is_proved_from_a_copy_checked_against_the_pin(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch)
    (staged := tmp_path / "staged").mkdir()
    for name, text in {"circuit.txt": CIRCUIT, "pin.txt": PIN, "stage.json": json.dumps({"n": 1})}.items():
        (staged / name).write_text(text)
    path = node.root / node.spec
    path.write_text(path.read_text().replace("warm = 2\n", f'warm = 2\nstaged = "{staged}"\n', 1))
    assert node.point() == 0, "`staged` is no part of the identity"
    assert [argv[0] for argv in node.ran] == ["gpu-lease", sys.executable] and node.ran[1][1] == SP.TIMER
    assert node.read(N.POINT_FILE)["statement"]["from"] == str(staged)

    (staged / "circuit.txt").write_text("another circuit\n")
    node.ran.clear()
    assert node.point() == N.REFUSED and node.ran == []
    assert "another circuit takes no GPU time" in node.read("result.json")["validation"]["detail"]


def test_a_statement_in_the_store_is_fetched_into_the_scratch_and_leaves_no_copy_on_the_node(tmp_path, monkeypatch):
    """The fetch reads the artifact's one directory through a store of its own under the scratch, which is removed
    before the statement is hashed; the statement is then held to the pin like any other, and goes with the scratch."""

    node = Node(tmp_path, monkeypatch)
    stored(node)
    assert node.point() == 0, "`staged` is no part of the identity"
    fetch, lease_argv, timer = node.ran
    scratch = Path(lease_argv[-1])
    assert fetch == [sys.executable, "-m", "research", "data", "fetch", ART, "--store", str(scratch / "store"),
                     "--path", IN_ART, "--to", str(scratch / "fetched")]
    assert node.asked[0] == [*fetch[:-2], "--list", "--json"], "the listing is asked of the same store, before the fetch"
    assert node.seen == {"store": False, "files": ["circuit.txt", "inst.bin", "pin.txt", "stage.json"]}, (
        "at the timed step the fetch's store is gone and the statement is where the timing script looks for it")
    assert not scratch.exists() and list(node.scratch_root.iterdir()) == []
    statement = node.read(N.POINT_FILE)["statement"]
    assert (statement["from"], statement["pin"], statement["n"]) == (f"{ART}/{IN_ART}", PIN, 1)
    assert "out/fetch.log" in node.read("result.json")["artifacts"]


#: a statement in the store that is refused: how the stand-in differs, and what the refusal says
NOT_FETCHED = [
    ({"listed": ""}, "the store lists no file of the `staged` of row 'gemm64'.*`research run --store-read`"),
    ({"listed": "[]"}, "the store lists no file of the `staged` of row 'gemm64'"),
    ({"listed": "not a listing"}, r"the store's listing of the `staged` of row 'gemm64'.*bytes \(JSONDecodeError"),
    ({"listed": json.dumps([{"path": "out/row1/x"}])}, r"isn't each file with its bytes \(KeyError"),
    ({"listed": json.dumps([{"path": "out/row1/x", "bytes": 1 << 62}])}, "held twice while it is fetched, and the disk"),
    ({"fetch_exit": 1}, r"fetching the `staged` of row 'gemm64', art:5+/out/row1, failed \(exit 1\): out/fetch.log"),
    ({"pin.txt": None}, "holds no staged statement in the store"),
    ({"circuit.txt": "another circuit\n"}, "another circuit takes no GPU time"),
    ({"stage.json": json.dumps({"n": 2})}, "the statement holds 2 units"),
]


@pytest.mark.parametrize("change, says", NOT_FETCHED, ids=[says[:40] for _, says in NOT_FETCHED])
def test_a_statement_the_store_does_not_give_whole_is_refused_before_any_card(tmp_path, monkeypatch, change, says):
    node = Node(tmp_path, monkeypatch)
    node.fetch_exit = change.get("fetch_exit", 0)
    stored(node, **{key: value for key, value in change.items() if key != "fetch_exit"})
    assert node.point() == N.REFUSED and not any(argv[0] == "gpu-lease" for argv in node.ran)
    assert re.search(says, node.read("result.json")["validation"]["detail"])
    assert list(node.scratch_root.iterdir()) == [], "what was fetched goes with the scratch"


def test_a_statement_from_the_store_is_never_fetched_into_the_nodes_memory(tmp_path, monkeypatch):
    """The scratch's default is the node's memory, where a statement staged in the run may be: one from the store is
    refused there before the store is asked anything."""

    assert settings.REGISTRY.setting("VERITY_BENCH_STEADY_SCRATCH").default == str(N.MEMORY)
    node = Node(tmp_path, monkeypatch)
    stored(node)
    monkeypatch.setattr(N, "MEMORY", node.scratch_root.resolve())
    assert node.point() == N.REFUSED and node.ran == [] and node.asked == []
    assert re.search("is fetched from the store onto the node's disk, and the scratch .* is in its memory: the spec's "
                     "setup exports VERITY_BENCH_STEADY_SCRATCH", node.read("result.json")["validation"]["detail"])
    assert list(node.scratch_root.iterdir()) == []
