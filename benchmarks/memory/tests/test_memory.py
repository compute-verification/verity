"""`verity bench memory` (`run.py`): the number is the γ at the specification's rate that an accepted session's
verifier recorded, printed as text and as one line of JSON; a session that wasn't accepted, or a run with no verifier's
record, prints no number and exits 1. The three ways in share that printer: a recorded run, the loopback (which says
it certifies nothing) and the node run (a gate run of one honest session, read from its fire run). The two references
are read from their runs, and one that can't be read never takes the number down. Under a research run an accepted
session's amounts are also written through the harness kit's writer, and what is printed stays the same to the byte.

Nothing here runs a session, reaches the evidence store or waits: the records are built in the test, `FakeStore`
stands in for the store, and the processes the command starts and its waits are stand-ins handed to `main`."""
from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.memory import run as M  # noqa: E402
from benchmarks.remote_session import summary  # noqa: E402

RUN, FIRE, PROVER = "r20000101-000000-aaaa", "r20000101-000001-bbbb", "r20000101-000002-cccc"
POUS, WIPE = "r20000101-000003-dddd", "r20000101-000004-eeee"
COMMIT = "c" * 40
PACED = M.SETTINGS.setting("PACED").default
#: gate_run.sh's variables for a host run of sessions with one coin a window
WINDOWED = ("SESSIONS", "WINDOWS", "WINDOW_MS", "SAMPLES")
GIB = 1 << 30
SESSION = HERE / "fixtures" / "session.json"
#: the bench's own run, and the host stamp the kit's writer stamps each measurement with
BENCH = "r20000101-000005-ffff"
STAMP = {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": "test", "commit": "d" * 40}
#: the result's measurements, in the order they are written
SHARED = ["metric", "verified", "capacity at spec", "free at spec", "capacity at measured", "free at measured", "error"]


def a_session(k: int = 0, *, accepted: bool = True, gamma=0.25, failure: str | None = None, verdict="lean") -> dict:
    """One session as the verifier records it (`fixtures/session.json`: the fields `summary.py` and the bench read),
    with this one's index, verdict and γ."""
    s = json.loads(SESSION.read_text())
    s["ledger"]["spec"]["gamma"] = gamma
    return {**s, "k": k, "accepted": accepted, "verdict": verdict,
            "failure": None if failure is None else {"t": 0, "failure": failure, "at": None}}


def with_amounts(s: dict, *, span_ns: int = 4 * 10 ** 9, error: float = 2.0 ** -30) -> dict:
    """Session `s` with what a verifier's ledger holds beside γ: the session's span, each reading's read rate (2 TB/s
    at the data sheet's, 1.6 TB/s measured), γ and certified bytes, the same bytes at both, and the declared error."""
    ledger = s["ledger"]
    whole = {"spec": 2e12 * span_ns / 1e9, "measured": 1.6e12 * span_ns / 1e9}
    certified = (1 - ledger["spec"]["gamma"]) * whole["spec"]
    ledger["inputs"]["span_ns"] = span_ns
    ledger["spec"].update(capacity_bytes_per_s=2e12, certified_bytes=certified)
    ledger["measured"] = {"capacity_bytes_per_s": 1.6e12, "gamma": 1 - certified / whole["measured"],
                          "certified_bytes": certified}
    ledger["declared"] = {"gamma": {r: {"error": {"windows": error, "horizon": error / 2}} for r in whole}}
    return s


def a_record(*sessions: dict, blocks: int = GIB // 4096, binding: dict | None = None) -> dict:
    """A verifier's record of `sessions` over a store of `blocks` blocks."""
    return {"registration": {"blocks": blocks}, "sessions": list(sessions), "calibration": {}, "rtt": {},
            "allowance_ns": 1, "process_cpu_ns": 1, "wire": {}, **({"binding": binding} if binding else {})}


class FakeStore:
    """The evidence store as `run.EvidenceStore` offers it, over runs laid out under a directory."""

    def __init__(self, root: Path) -> None:
        self.root, self.attempts, self.labels, self.fetched, self.asked = root, {}, {}, [], []

    def add(self, run: str, files: dict | None = None, *, rc: int = 0, dirty: bool = False, env: dict | None = None,
            sku: str | None = None) -> None:
        self.attempts[run] = {"source": {"commit": COMMIT, "dirty": dirty}, "status": {"rc": rc},
                              "params": {"env": env or {}}, "conditions": {"hw.sku": sku},
                              "execution": {"start_utc": "2000-01-02T03:04:05.678Z"}}
        for path, body in (files or {}).items():
            p = self.root / run / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body if isinstance(body, str) else json.dumps(body))

    def attempt(self, run: str) -> dict | None:
        self.asked.append(run)
        return self.attempts.get(run)

    def files(self, run: str, globs: list[str]) -> Path | None:
        if run not in self.attempts:
            raise M.Unreadable(f"{run} is not in the evidence store this machine reads")
        self.fetched.append(run)
        tree = self.root / run
        held = [str(p.relative_to(tree)) for p in tree.rglob("*") if p.is_file()] if tree.is_dir() else []
        return tree if any(fnmatch.fnmatchcase(p, g) for p in held for g in globs) else None

    def label(self, run: str, key: str) -> str | None:
        return self.labels.get((run, key))


@pytest.fixture
def store(tmp_path: Path, monkeypatch) -> FakeStore:
    """A store holding the two references' runs, which `references.json` is pointed at; no `research run` around
    the command; and the kit's writer stamping with `STAMP` rather than asking this machine."""
    for name in ("RESEARCH_RUN_DIR", "RESEARCH_RUN_ID", "RESEARCH_SOURCE_SHA", "PACED", "UNPACED", "ROLE", "SERVE",
                 "HOST_GIB", *WINDOWED):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(M.kit, "host_stamp", lambda: dict(STAMP))
    s = FakeStore(tmp_path / "store")
    s.add(POUS, {"out/summary.json": {"certificate": {"certified": True, "s_bits": 6 * 8 * GIB},
                                      "held": {"total_bytes": 8 * GIB, "device": "CARD A"}}})
    s.add(WIPE, sku="CARD B")
    s.labels[WIPE, "label"] = "full-fill wipe certified: foreign <= 8,589,934,592 bits (1.0737 GB) at eps 2^-128, M = 8,589,934,592 B"
    references = tmp_path / "references.json"
    references.write_text(json.dumps({"PoUS state bound": POUS, "wipe residue": WIPE}))
    monkeypatch.setattr(M, "REFERENCES", references)
    return s


def bench(store: FakeStore, *argv: str, **kw) -> int:
    def refuse(*a, **k):
        raise AssertionError(f"nothing is started or waited for here: {a}")

    return M.main(list(argv), store_of=lambda: store, **{"run": refuse, "wait": refuse, **kw})


def printed(capsys) -> tuple[list[str], dict | None, str]:
    """What the command printed: its text lines, its JSON document (None with nothing printed) and its stderr."""
    out, err = capsys.readouterr()
    lines = out.splitlines()
    return lines[:-1], json.loads(lines[-1]) if lines else None, err


def test_an_accepted_session_prints_its_gamma_at_the_data_sheets_rate_as_text_and_as_json(store, capsys):
    store.add(RUN, {"out/verifier.json": a_record(a_session(gamma=0.15625))})
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[0] == ("memory: 0.156 of the node's memory read rate is left uncertified by this session, at the data "
                       "sheet's read bandwidth (2.0000 TB/s)")
    assert text[1] == (f"  session 0 of {RUN}, commit {COMMIT}: accepted by the Lean verifier, its verifier not on the "
                       f"auditor's verifier host")
    assert text[2] == "  store: 1.07 GB, 0.125 of the node's 8.59 GB of GPU memory, bound to no checkpoint"
    assert list(doc)[:4] == ["value", "unit", "commit", "run"]
    assert (doc["value"], doc["unit"], doc["commit"], doc["run"]) == (0.15625, M.UNIT, COMMIT, RUN)
    assert doc["loopback"] is False and doc["host_run"] is None
    assert not any("loopback" in line or "certifies nothing" in line for line in text)


def test_the_number_is_the_session_harnesss_gamma_spec_of_the_same_records(tmp_path):
    """The bench reads the verifier's record alone; `summary.py`, which also reads the prover's, gives the same number
    for the session."""
    (tmp_path / "verifier.json").write_text(json.dumps(a_record(a_session(gamma=0.3125))))
    (tmp_path / "prover.json").write_text(json.dumps({"sessions": [{"mode": "honest", "opened": [{"bytes": 10}]}]}))
    (row,) = summary.summarize(tmp_path)["honest"]
    got = M.session(tmp_path / "verifier.json", None)
    assert got["value"] == row["gamma_spec"] == 0.3125 and got["session"] == row["k"]
    assert got["store_bytes"] == summary.summarize(tmp_path)["bytes"]


@pytest.mark.parametrize("record, why", [
    (a_record(a_session(accepted=False, failure="receipt")), "session 0 was not accepted (receipt)"),
    (a_record(a_session(accepted=False)), "session 0 was not accepted (no reason recorded)"),
    (a_record(), "the verifier recorded no session"),
    (a_record(a_session(gamma=None)), "session 0 was accepted and its ledger holds no γ"),
    (a_record(a_session(gamma=True)), "session 0 was accepted and its ledger holds no γ"),
    ({"sessions": "none"}, "the verifier's record doesn't read as one"),
    ("not json", "the verifier's record doesn't read as one"),
])
def test_a_session_that_was_not_accepted_prints_no_number_and_exits_1(store, capsys, record, why):
    store.add(RUN, {"out/verifier.json": record})
    assert bench(store, "--from", RUN) == 1
    out, err = capsys.readouterr()
    assert out == "" and f"memory: no number: {why}" in err


def test_a_run_that_left_no_verifiers_record_prints_no_number_and_exits_1(store, capsys):
    store.add(RUN, {"out/probe.json": {}}, rc=3)
    assert bench(store, "--from", RUN) == 1
    out, err = capsys.readouterr()
    assert out == "" and f"memory: no number: {RUN} left no verifier's record (it ended rc 3)" in err


def test_a_run_that_is_not_in_the_store_exits_2(store, capsys):
    assert bench(store, "--from", RUN) == 2
    out, err = capsys.readouterr()
    assert out == "" and f"{RUN} is not in the evidence store" in err


def test_a_run_that_holds_several_sessions_needs_one_named(store, capsys):
    record = a_record(a_session(0, gamma=0.75), a_session(1, accepted=False, failure="receipt"), a_session(2, gamma=0.5))
    store.add(RUN, {"out/verifier.json": record})
    assert bench(store, "--from", RUN) == 2
    out, err = capsys.readouterr()
    assert out == "" and "holds sessions 0 (accepted), 1 (not accepted), 2 (accepted): name one with --session K" in err
    assert bench(store, "--from", RUN, "--session", "2") == 0
    assert printed(capsys)[1]["value"] == 0.5
    assert bench(store, "--from", RUN, "--session", "1") == 1
    assert "session 1 was not accepted (receipt)" in capsys.readouterr().err
    assert bench(store, "--from", RUN, "--session", "5") == 2
    assert "no session 5" in capsys.readouterr().err


def test_a_gate_runs_number_is_read_from_the_host_run_its_fire_run_names(store, capsys):
    """A fire run holds the verifier host's run under out/host/, named in out/host-run.txt; a host run it doesn't
    name is not read."""
    binding = {"name": "SOME_ROLE", "bound": True}
    store.add(FIRE, {"out/host-run.txt": "v2\n",
                     "out/host/v1/out/verifier.json": a_record(a_session(gamma=0.875)),
                     "out/host/v2/out/verifier.json": a_record(a_session(gamma=0.625), binding=binding)})
    assert bench(store, "--from", FIRE) == 0
    text, doc, _ = printed(capsys)
    assert doc["value"] == 0.625 and doc["model"] == "SOME_ROLE" and doc["host_run"] == "v2"
    assert text[1].endswith("accepted by the Lean verifier, on the auditor's verifier host (its run v2)")
    assert text[2].endswith("bound to SOME_ROLE's pinned checkpoint")


def test_a_binding_that_did_not_bind_and_a_record_with_no_lean_verdict_are_said(store, capsys):
    record = a_record(a_session(verdict=None), binding={"name": "SOME_ROLE", "bound": False})
    store.add(RUN, {"out/verifier.json": record}, dirty=True)
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[1].startswith(f"  session 0 of {RUN}, commit {COMMIT}-dirty: accepted (its record names no Lean verdict)")
    assert text[2].endswith("bound to no checkpoint") and doc["model"] is None and doc["commit"] == f"{COMMIT}-dirty"


@pytest.mark.parametrize("gamma", [1.0, 1.25])
def test_a_share_of_one_or_more_says_it_certifies_nothing(store, capsys, gamma):
    """An accepted session whose ledger leaves the whole read rate uncertified has a number, and the text says what it
    is worth."""
    store.add(RUN, {"out/verifier.json": a_record(a_session(gamma=gamma))})
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[1] == f"  {M.NOTHING}" and "certifies nothing" in text[1] and doc["value"] == gamma


def test_a_recorded_loopback_is_printed_as_one(store, capsys):
    """A run this command's `--loopback` left holds its document, which says so: read back with `--from`, it is still
    a loopback, in the text and in the JSON, with no node's memory beside its test store."""
    store.add(RUN, {"out/verifier.json": a_record(a_session(gamma=0.5), blocks=64),
                    "out/memory.json": {"value": 0.5, "run": RUN, "loopback": True}})
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[1] == f"  {M.LOOPBACK}" and doc["loopback"] is True
    assert text[2] == f"  session 0 of {RUN}, commit {COMMIT}: accepted by the Lean verifier"
    assert text[3] == "  store: 262,144 bytes, a test store, bound to no checkpoint"


@pytest.mark.parametrize("body", ["{", "[1]", "7"])
def test_a_run_whose_own_document_cannot_be_read_is_not_printed(store, capsys, body):
    """Whether the run was a loopback is then unknown, so nothing is printed as a node's number."""
    store.add(RUN, {"out/verifier.json": a_record(a_session()), "out/memory.json": body})
    assert bench(store, "--from", RUN) == 2
    out, err = capsys.readouterr()
    assert out == "" and "out/memory.json doesn't read as this command's document" in err


def test_a_run_holding_another_runs_document_is_not_printed(store, capsys):
    """The document a run holds marks that run's session only when it is about that run. One about another run leaves
    this run's session unmarked, and it may be a loopback's, so nothing is printed as a node's number."""
    store.add(RUN, {"out/verifier.json": a_record(a_session(gamma=0.5), blocks=64),
                    "out/memory.json": {"value": 0.125, "run": FIRE, "loopback": False}})
    assert bench(store, "--from", RUN) == 2
    out, err = capsys.readouterr()
    assert out == "" and f"{RUN}'s out/memory.json is the document of {FIRE}" in err


def test_the_references_are_read_from_their_runs_with_date_and_card(store, capsys):
    store.add(RUN, {"out/verifier.json": a_record(a_session())})
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[3] == f"  beside it, PoUS state bound: 0.750 (6.44 GB of 8.59 GB; {POUS}, 2000-01-02, CARD A)"
    assert text[4] == f"  beside it, wipe residue: 0.125 (1.07 GB of 8.59 GB; {WIPE}, 2000-01-02, CARD B)"
    assert [(r["name"], r["value"], r["run"], r["date"], r["card"]) for r in doc["beside"]] == [
        ("PoUS state bound", 0.75, POUS, "2000-01-02", "CARD A"), ("wipe residue", 0.125, WIPE, "2000-01-02", "CARD B")]
    assert all(s in doc["beside"][0]["unit"] for s in ("S0 = floor(rho*|C|)", "not the device memory credited"))


def test_a_reference_that_cannot_be_read_says_why_and_the_number_still_prints(store, capsys):
    store.add(RUN, {"out/verifier.json": a_record(a_session(gamma=0.25))})
    store.add(POUS, {"out/summary.json": {"certificate": {"certified": False, "s_bits": 0}, "held": {"total_bytes": 8}}})
    store.labels[WIPE, "label"] = "a wipe, no bound stated"
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert doc["value"] == 0.25 and [r["value"] for r in doc["beside"]] == [None, None]
    assert text[3] == f"  beside it, PoUS state bound: not read ({POUS}'s record certifies nothing)"
    assert text[4] == f"  beside it, wipe residue: not read ({WIPE}'s label doesn't state a residue: 'a wipe, no bound stated')"


def test_a_store_that_cannot_be_opened_for_the_references_leaves_the_loopbacks_number(store, capsys, monkeypatch):
    def launch(argv, **kw):
        out = Path(kw["env"]["RESEARCH_RUN_DIR"]) / "out"
        out.mkdir(parents=True)
        (out / "verifier.json").write_text(json.dumps(a_record(a_session(gamma=4.5))))
        return SimpleNamespace(returncode=0)

    def no_store():
        raise M.Unreadable("the evidence store's client isn't installed here")

    monkeypatch.setenv("RESEARCH_SOURCE_SHA", COMMIT)
    assert M.main(["--loopback"], store_of=no_store, run=launch) == 0
    text, doc, _ = printed(capsys)
    assert doc["value"] == 4.5 and all(r["value"] is None for r in doc["beside"])
    assert text[4] == "  beside it, PoUS state bound: not read (the evidence store's client isn't installed here)"


def test_every_reference_of_the_tree_has_a_reader():
    assert set(json.loads((HERE.parent / "references.json").read_text())) == set(M.READERS)


def test_the_loopback_runs_both_processes_through_run_sh_and_says_it_certifies_nothing(store, capsys, monkeypatch):
    """`--loopback` is `run.sh` with ROLE=both and the CPU reference as the prover, on this process's cores and
    Python; its number is its verifier's, and the text and the JSON both say no node was audited."""
    seen = []

    def launch(argv, **kw):
        seen.append((argv, kw["cwd"], dict(kw["env"])))
        out = Path(kw["env"]["RESEARCH_RUN_DIR"]) / "out"
        out.mkdir(parents=True)
        (out / "verifier.json").write_text(json.dumps(a_record(a_session(gamma=7.5), blocks=64)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("RESEARCH_SOURCE_SHA", COMMIT)
    monkeypatch.setenv("SERVE", "node")
    monkeypatch.setattr(M.os, "sched_getaffinity", lambda pid: {9, 3, 7, 5})
    assert bench(store, "--loopback", run=launch) == 0
    ((argv, cwd, env),) = seen
    assert argv == ["bash", str(ROOT / "benchmarks/remote_session/run.sh")] and cwd == ROOT
    assert (env["ROLE"], env["PLAN"], env["CPU"], env["PY"]) == ("both", "honest", "64", sys.executable)
    assert (env["VERIFIER_CPUS"], env["PROVER_CPUS"]) == ("3,5", "7,9") and "SERVE" not in env
    assert not Path(env["RESEARCH_RUN_DIR"]).exists(), "outside a research run its records are scratch"
    text, doc, _ = printed(capsys)
    assert text[0].startswith("memory: 7.500 of the node's memory read rate")
    assert text[1] == f"  {M.LOOPBACK}" and "certifies nothing" in text[1]
    assert text[2] == f"  session 0 of no recorded run, commit {COMMIT}: accepted by the Lean verifier"
    assert text[3] == "  store: 262,144 bytes, a test store, bound to no checkpoint", "no node's memory beside a test store"
    assert (doc["loopback"], doc["value"], doc["commit"], doc["run"]) == (True, 7.5, COMMIT, None)


def test_a_loopback_under_a_research_run_keeps_its_records_and_its_document_in_the_runs_out(store, capsys, monkeypatch,
                                                                                         tmp_path):
    def launch(argv, **kw):
        assert kw["env"]["RESEARCH_RUN_DIR"] == str(tmp_path / "rundir")
        (tmp_path / "rundir" / "out").mkdir(parents=True)
        (tmp_path / "rundir" / "out" / "verifier.json").write_text(json.dumps(a_record(a_session())))
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path / "rundir"))
    monkeypatch.setenv("RESEARCH_RUN_ID", RUN)
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", COMMIT)
    assert bench(store, "--loopback", run=launch) == 0
    _, doc, _ = printed(capsys)
    assert doc["run"] == RUN and json.loads((tmp_path / "rundir" / "out" / "memory.json").read_text()) == doc
    assert (tmp_path / "rundir" / "out" / "verifier.json").is_file()


def test_a_run_that_ran_the_loopback_keeps_its_mark_whatever_it_prints_afterwards(store, capsys, monkeypatch, tmp_path):
    """A run's `out/memory.json` is how `--from` knows its session for a loopback. A second command in the same run
    that prints another run's number must not replace it, or the loopback's number would read as a node's."""
    out = tmp_path / "rundir" / "out"

    def launch(argv, **kw):
        out.mkdir(parents=True)
        (out / "verifier.json").write_text(json.dumps(a_record(a_session(gamma=0.5))))
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path / "rundir"))
    monkeypatch.setenv("RESEARCH_RUN_ID", RUN)
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", COMMIT)
    assert bench(store, "--loopback", run=launch) == 0
    _, first, _ = printed(capsys)
    store.add(FIRE, {"out/verifier.json": a_record(a_session(gamma=0.125))})
    assert bench(store, "--from", FIRE) == 0
    _, second, err = printed(capsys)
    assert (second["run"], second["loopback"], second["value"]) == (FIRE, False, 0.125)
    assert "its out/memory.json stays the loopback's document" in err
    assert json.loads((out / "memory.json").read_text()) == first
    store.add(RUN, {f"out/{p.name}": p.read_text() for p in out.iterdir()})
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert doc["loopback"] is True and text[1] == f"  {M.LOOPBACK}"
    (out / "memory.json").write_text("not json")
    assert bench(store, "--from", FIRE) == 2
    printed_out, err = capsys.readouterr()
    assert printed_out == "" and "out/memory.json doesn't read as this command's document" in err


@pytest.mark.parametrize("rc, record, why", [
    (3, None, "the loopback's processes failed (run.sh exited 3)"),
    (0, a_record(a_session(accepted=False, failure="late")), "session 0 was not accepted (late)"),
    (0, None, "the verifier's record doesn't read as one"),
])
def test_a_loopback_whose_session_failed_prints_no_number_and_exits_1(store, capsys, monkeypatch, rc, record, why):
    def launch(argv, **kw):
        if record is not None:
            out = Path(kw["env"]["RESEARCH_RUN_DIR"]) / "out"
            out.mkdir(parents=True)
            (out / "verifier.json").write_text(json.dumps(record))
        return SimpleNamespace(returncode=rc)

    monkeypatch.setenv("RESEARCH_SOURCE_SHA", COMMIT)
    assert bench(store, "--loopback", run=launch) == 1
    out, err = capsys.readouterr()
    assert out == "" and f"memory: no number: {why}" in err


@pytest.mark.parametrize("porcelain, commit", [("", "a" * 40), (" M file\n", "a" * 40 + "-dirty")])
def test_the_loopbacks_commit_outside_a_research_run_is_gits_head_marked_when_the_tree_has_changes(
        monkeypatch, porcelain, commit):
    def git(argv, **kw):
        assert argv[0] == "git" and kw["cwd"] == ROOT
        return SimpleNamespace(returncode=0, stdout="a" * 40 + "\n" if argv[1] == "rev-parse" else porcelain)

    monkeypatch.delenv("RESEARCH_SOURCE_SHA", raising=False)
    assert M.source_commit(git) == commit
    assert M.source_commit(lambda argv, **kw: SimpleNamespace(returncode=128, stdout="")) is None


def test_one_core_is_both_processes(monkeypatch):
    monkeypatch.setattr(M.os, "sched_getaffinity", lambda pid: {2})
    assert M.cores() == ("2", "2")


def gate_run(home: Path, seen: list, *, rc: int = 0):
    """`gate_run.sh` as the node run starts it: records its command line and environment, and leaves the state file a
    launched gate run leaves."""
    def launch(argv, **kw):
        seen.append((argv, dict(kw["env"])))
        if rc == 0 and "--dry-run" not in argv:
            state = home / ".research" / "gate-runs" / f"{argv[-1]}.env"
            state.parent.mkdir(parents=True)
            state.write_text(f"COMMIT={COMMIT}\nprobe=r20000101-000009-ffff\nprover={PROVER}\nfire={FIRE}\n")
        return SimpleNamespace(returncode=rc)

    return launch


def test_the_node_run_is_a_gate_run_of_one_honest_session_read_from_its_fire_run(store, capsys, monkeypatch, tmp_path):
    """The node run starts `gate_run.sh PERIOD` with the Llama role, one honest session and one paced word, no host
    memory in the store and nothing of a session of one coin a window, then waits for the fire run's record, a minute
    a look, and prints from it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("COMMIT", "d" * 40)
    seen, waits = [], []

    def wait(seconds):
        waits.append(seconds)
        if len(waits) == 2:
            store.add(FIRE, {"out/host-run.txt": "v1", "out/host/v1/out/verifier.json": a_record(a_session(gamma=0.375))})

    assert bench(store, "some-period", run=gate_run(tmp_path, seen), wait=wait) == 0
    ((argv, env),) = seen
    assert argv == ["bash", str(ROOT / "benchmarks/remote_session/gate_run.sh"), "some-period"]
    assert (env["ROLE"], env["PLAN"], env["PACED"], env["COMMIT"]) == ("LLAMA32_1B_BF16", "honest", PACED, "d" * 40)
    assert not {"HOST_GIB", "UNPACED", *WINDOWED} & set(env)
    assert len(PACED.split()) == 1 and waits == [60, 60]
    text, doc, err = printed(capsys)
    assert (doc["value"], doc["run"], doc["commit"], doc["loopback"]) == (0.375, FIRE, COMMIT, False)
    assert f"fire {FIRE}, prover {PROVER}" in err and f"verity bench memory --from {FIRE}" in err
    assert PROVER not in store.fetched, "the prover's record is never read"


def test_the_node_runs_role_and_pace_can_be_named_and_a_pace_of_several_sessions_is_refused(store, capsys, monkeypatch,
                                                                                         tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ROLE", "OTHER_ROLE")
    monkeypatch.setenv("PACED", "1:60000:33273:2")
    seen = []
    store.add(FIRE, {"out/verifier.json": a_record(a_session())})
    assert bench(store, "p", run=gate_run(tmp_path, seen)) == 0
    assert (seen[0][1]["ROLE"], seen[0][1]["PACED"], seen[0][1]["PLAN"]) == ("OTHER_ROLE", "1:60000:33273:2", "honest")
    capsys.readouterr()
    monkeypatch.setenv("PACED", "1:1000:2773:7 1:1000:2773:7")
    assert bench(store, "q") == 2
    out, err = capsys.readouterr()
    assert out == "" and "the bench runs one session, one word" in err


@pytest.mark.parametrize("given, want", [
    ({"HOST_GIB": "8", "UNPACED": "7:60000:33273"},
     {"HOST_GIB": "8", "SESSIONS": "1", "WINDOWS": "7", "WINDOW_MS": "60000", "SAMPLES": "33273"}),
    ({"HOST_GIB": "8"}, {"HOST_GIB": "8", "PACED": PACED}),
    ({"UNPACED": "2:1000.5:2773"}, {"SESSIONS": "1", "WINDOWS": "2", "WINDOW_MS": "1000.5", "SAMPLES": "2773"}),
    ({"UNPACED": ""}, {"PACED": PACED}),
], ids=["both", "host-memory", "one-coin-a-window", "an-empty-word"])
def test_the_node_run_can_put_host_memory_in_the_store_and_ask_for_one_coin_a_window(store, capsys, monkeypatch,
                                                                                    tmp_path, given, want):
    """HOST_GIB passes through to `gate_run.sh`, which sends it to the node. UNPACED's word is one session of that
    many windows, each that long with that many openings, as `gate_run.sh`'s variables for the host's run, and then no
    PACED goes with it; without it the session is PACED's, and no variable of the other form is set."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for name, value in given.items():
        monkeypatch.setenv(name, value)
    seen = []
    store.add(FIRE, {"out/host-run.txt": "v1", "out/host/v1/out/verifier.json": a_record(a_session(gamma=0.15))})
    assert bench(store, "p", run=gate_run(tmp_path, seen)) == 0
    ((argv, env),) = seen
    assert {k: env[k] for k in ("HOST_GIB", "PACED", *WINDOWED) if k in env} == want
    assert (env["ROLE"], env["PLAN"]) == ("LLAMA32_1B_BF16", "honest")
    assert printed(capsys)[1]["value"] == 0.15


@pytest.mark.parametrize("given, why", [
    ({"UNPACED": "7:60000:33273", "PACED": "1:1000:2773:7"},
     "UNPACED and PACED are both set: the bench runs one session, in one form, so unset one"),
    ({"UNPACED": "7:60000:33273 7:60000:33273"}, "one word WINDOWS:WINDOW_MS:SAMPLES"),
    ({"UNPACED": "7:60000"}, "one word WINDOWS:WINDOW_MS:SAMPLES"),
    ({"UNPACED": "1:60000:33273:2"}, "one word WINDOWS:WINDOW_MS:SAMPLES"),
    ({"UNPACED": "0:60000:33273"}, "one word WINDOWS:WINDOW_MS:SAMPLES"),
    ({"UNPACED": "7:60000.0005:33273"}, "one word WINDOWS:WINDOW_MS:SAMPLES"),
    ({"UNPACED": "7:a minute:33273"}, "one word WINDOWS:WINDOW_MS:SAMPLES"),
], ids=["both-forms", "two-sessions", "two-numbers", "a-paced-word", "no-window", "a-fourth-place", "not-numbers"])
@pytest.mark.parametrize("dry", [(), ("--dry-run",)], ids=["run", "dry-run"])
def test_both_forms_or_a_word_that_is_not_one_session_of_windows_starts_nothing_and_exits_2(store, capsys, monkeypatch,
                                                                                           given, why, dry):
    for name, value in given.items():
        monkeypatch.setenv(name, value)
    assert bench(store, "p", *dry) == 2
    out, err = capsys.readouterr()
    assert out == "" and why in err and store.asked == []


def test_a_gate_run_that_did_not_launch_prints_no_number_and_waits_for_nothing(store, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert bench(store, "p", run=gate_run(tmp_path, [], rc=3)) == 1
    out, err = capsys.readouterr()
    assert out == "" and "memory: no number: the gate run did not launch (gate_run.sh exited 3)" in err
    assert store.asked == []


def test_a_fire_run_that_never_reaches_the_store_prints_no_number(store, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(M, "WAIT_MIN", 3)
    waits = []
    assert bench(store, "p", run=gate_run(tmp_path, []), wait=waits.append) == 1
    out, err = capsys.readouterr()
    assert out == "" and waits == [60, 60, 60] and f"{FIRE} left no record in the store within 3 minutes" in err


def test_a_fire_run_whose_host_run_failed_prints_no_number(store, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    store.add(FIRE, {"out/probe.json": {}}, rc=3)
    assert bench(store, "p", run=gate_run(tmp_path, [])) == 1
    out, err = capsys.readouterr()
    assert out == "" and f"{FIRE} left no verifier's record (it ended rc 3)" in err


@pytest.mark.parametrize("rc", [0, 3])
def test_a_dry_node_run_passes_the_flag_on_prints_no_number_and_exits_as_the_gate_run_did(store, capsys, monkeypatch,
                                                                                       tmp_path, rc):
    monkeypatch.setenv("HOME", str(tmp_path))
    seen = []
    assert bench(store, "p", "--dry-run", run=gate_run(tmp_path, seen, rc=rc)) == rc
    assert seen[0][0][-2:] == ["--dry-run", "p"]
    out, err = capsys.readouterr()
    assert out == "" and "there is no number" in err and store.asked == []


@pytest.mark.parametrize("argv", [[], ["p", "--loopback"], ["--from", RUN, "--loopback"], ["p", "--from", RUN]])
def test_the_command_takes_exactly_one_way_in(store, capsys, argv):
    with pytest.raises(SystemExit) as e:
        bench(store, *argv)
    assert e.value.code == 2 and "one of PERIOD (the node run), --from RUN or --loopback" in capsys.readouterr().err


def a_run(monkeypatch, tmp_path: Path) -> Path:
    """A `research run`, `BENCH`, around the command from here on: its directory."""
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path / "rundir"))
    monkeypatch.setenv("RESEARCH_RUN_ID", BENCH)
    return tmp_path / "rundir"


def results(held: Path) -> list[Path]:
    """Every `result.json` in the run's directory `held` and under it."""
    return sorted(held.rglob("result.json"))


def written(held: Path) -> dict[str, dict]:
    """The measurements of the run's result, by name, in the order they were written. The result is the run
    directory's own `result.json`, the one file `research` reads a run's result from, and the run holds no other:
    none under its `out/`."""
    assert results(held) == [held / "result.json"]
    result = json.loads((held / "result.json").read_text())
    assert (result["schema"], result["run_id"]) == ("research/result/v0.1", BENCH)
    return {m["name"]: m for m in result["measurements"]}


def test_under_a_research_run_an_accepted_session_also_writes_its_amounts_and_prints_the_same(store, capsys, monkeypatch,
                                                                                            tmp_path):
    """The fields every resource's benchmark reports, through the harness kit's writer: each is the verifier's record's
    and names the record's key it was read from. Free over capacity is the reading's γ, at the first reading the
    printed number, and verified and free are the capacity. What the command prints, and its `out/memory.json`, are
    the same to the byte with the result written; `out/` holds the document alone."""
    store.add(RUN, {"out/verifier.json": a_record(with_amounts(a_session()))})
    assert bench(store, "--from", RUN) == 0
    alone = capsys.readouterr()
    held = a_run(monkeypatch, tmp_path)
    assert bench(store, "--from", RUN) == 0
    assert capsys.readouterr() == alone
    line = alone.out.splitlines()[-1]
    doc = json.loads(line)
    assert (held / "out" / "memory.json").read_text() == line + "\n"
    assert sorted(p.name for p in (held / "out").iterdir()) == ["memory.json"]
    assert list(doc) == ["value", "unit", "commit", "run", "loopback", "host_run", "session", "verdict",
                         "rate_bytes_per_s", "store_bytes", "node_gpu_bytes", "model", "beside"]
    got = written(held)
    assert list(got) == SHARED
    assert {n: m["value"] for n, m in got.items()} == {
        "metric": 0.25, "verified": 6e12, "capacity at spec": 8e12, "free at spec": 2e12,
        "capacity at measured": 6.4e12, "free at measured": 4e11, "error": 2.0 ** -30}
    assert got["metric"]["value"] == doc["value"] and got["metric"]["unit"] == doc["unit"] == M.UNIT
    for reading, gamma in (("spec", doc["value"]), ("measured", 0.0625)):
        capacity, free = got[f"capacity at {reading}"]["value"], got[f"free at {reading}"]["value"]
        assert free / capacity == gamma and got["verified"]["value"] + free == capacity
    # a capacity's unit says where its rate comes from
    assert got["capacity at spec"]["unit"] == (
        "bytes over the session's span, at the data sheet's read bandwidth: the verifier's own figure for the node "
        "(its timing.py), not the registered hardware's")
    assert got["capacity at measured"]["unit"] == f"{M.BYTES}, at the measured read rate: {M.OWN_RATE}"
    assert {got[n]["unit"] for n in ("verified", "free at spec", "free at measured")} == {M.BYTES}
    assert got["error"]["unit"] == M.ERROR and "PRF advantages" in M.ERROR and "are not in it" in M.ERROR
    rate, span = "ledger.{}.capacity_bytes_per_s", "ledger.inputs.span_ns"
    assert {n: m["inputs"] for n, m in got.items()} == {
        "metric": [RUN, "ledger.spec.gamma"], "verified": [RUN, "ledger.spec.certified_bytes"],
        "capacity at spec": [RUN, rate.format("spec"), span],
        "free at spec": [RUN, "ledger.spec.gamma", rate.format("spec"), span],
        "capacity at measured": [RUN, rate.format("measured"), span],
        "free at measured": [RUN, "ledger.measured.gamma", rate.format("measured"), span],
        "error": [RUN, "ledger.declared.gamma.spec.error.windows"]}
    for m in got.values():
        assert (m["run"], m["commit"], m["host"]) == (BENCH, STAMP["commit"], STAMP)
        assert (m["target"], m["verdict"], m["ci95"]) == (None, None, None)


@pytest.mark.parametrize("drop, left_out", [
    (("inputs", "span_ns"), SHARED[2:6]),
    (("spec", "certified_bytes"), ["verified"]),
    (("spec", "capacity_bytes_per_s"), SHARED[2:4]),
    (("measured", "gamma"), SHARED[4:6]),
    (("measured",), SHARED[4:6]),
    (("declared", "gamma", "spec", "error"), ["error"]),
    (("declared",), ["error"]),
])
def test_an_amount_the_record_does_not_hold_is_left_out_of_the_result(store, monkeypatch, tmp_path, drop, left_out):
    """A reading the verifier refused holds its rate and why, and no γ; a record may hold no span or no declared
    error. What it holds is still written, the metric always."""
    s = with_amounts(a_session())
    at = s["ledger"]
    for k in drop[:-1]:
        at = at[k]
    del at[drop[-1]]
    store.add(RUN, {"out/verifier.json": a_record(s)})
    held = a_run(monkeypatch, tmp_path)
    assert bench(store, "--from", RUN) == 0
    assert list(written(held)) == [n for n in SHARED if n not in left_out]


def test_a_record_that_holds_only_the_share_writes_the_metric_alone(store, monkeypatch, tmp_path):
    store.add(RUN, {"out/verifier.json": a_record(a_session(gamma=0.375))})
    held = a_run(monkeypatch, tmp_path)
    assert bench(store, "--from", RUN) == 0
    assert {n: m["value"] for n, m in written(held).items()} == {"metric": 0.375}


def test_a_share_of_one_or_more_writes_nothing_verified_and_the_whole_capacity_free(store, monkeypatch, tmp_path):
    """The verifier's certified bytes are then at most none. The metric stays the printed share."""
    store.add(RUN, {"out/verifier.json": a_record(with_amounts(a_session(gamma=1.25)))})
    held = a_run(monkeypatch, tmp_path)
    assert bench(store, "--from", RUN) == 0
    got = {n: m["value"] for n, m in written(held).items()}
    assert got == {"metric": 1.25, "verified": 0.0, "capacity at spec": 8e12, "free at spec": 8e12,
                   "capacity at measured": 6.4e12, "free at measured": 6.4e12, "error": 2.0 ** -30}


def test_the_node_runs_session_writes_its_amounts_from_the_fire_run(store, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    held = a_run(monkeypatch, tmp_path)
    store.add(FIRE, {"out/host-run.txt": "v1",
                     "out/host/v1/out/verifier.json": a_record(with_amounts(a_session(gamma=0.5)))})
    assert bench(store, "some-period", run=gate_run(tmp_path, [])) == 0
    _, doc, _ = printed(capsys)
    got = written(held)
    assert list(got) == SHARED and got["metric"]["value"] == doc["value"] == 0.5
    assert (got["verified"]["value"], got["free at spec"]["value"]) == (4e12, 4e12)
    assert all(m["inputs"][0] == FIRE for m in got.values())


def test_a_loopback_writes_no_result(store, capsys, monkeypatch, tmp_path):
    """It certifies nothing: neither the loopback itself nor a recorded one printed with `--from` writes amounts."""
    held = a_run(monkeypatch, tmp_path)

    def launch(argv, **kw):
        (held / "out").mkdir(parents=True)
        (held / "out" / "verifier.json").write_text(json.dumps(a_record(with_amounts(a_session()))))
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("RESEARCH_SOURCE_SHA", COMMIT)
    assert bench(store, "--loopback", run=launch) == 0
    assert (held / "out" / "memory.json").is_file() and results(held) == []
    store.add(RUN, {"out/verifier.json": a_record(with_amounts(a_session())),
                    "out/memory.json": {"value": 0.25, "run": RUN, "loopback": True}})
    assert bench(store, "--from", RUN) == 0
    assert printed(capsys)[1]["loopback"] is True and results(held) == []


def test_a_session_that_was_not_accepted_writes_no_result_and_leaves_the_runs_own(store, capsys, monkeypatch, tmp_path):
    held = a_run(monkeypatch, tmp_path)
    store.add(RUN, {"out/verifier.json": a_record(with_amounts(a_session(accepted=False, failure="receipt")))})
    assert bench(store, "--from", RUN) == 1 and not held.exists()
    store.add(FIRE, {"out/verifier.json": a_record(with_amounts(a_session()))})
    assert bench(store, "--from", FIRE) == 0
    kept = (held / "result.json").read_bytes()
    capsys.readouterr()
    assert bench(store, "--from", RUN) == 1
    assert capsys.readouterr().out == "" and (held / "result.json").read_bytes() == kept and len(results(held)) == 1


def test_a_second_session_printed_in_one_run_keeps_the_first_ones_result_and_says_so(store, capsys, monkeypatch,
                                                                                    tmp_path):
    """A result holds one measurement a name. The second session's number still prints, the command still exits 0,
    and its `out/memory.json` is the second's, as before."""
    held = a_run(monkeypatch, tmp_path)
    store.add(RUN, {"out/verifier.json": a_record(with_amounts(a_session()))})
    store.add(FIRE, {"out/verifier.json": a_record(with_amounts(a_session(gamma=0.5)))})
    assert bench(store, "--from", RUN) == 0
    kept = (held / "result.json").read_bytes()
    capsys.readouterr()
    assert bench(store, "--from", FIRE) == 0
    _, doc, err = printed(capsys)
    assert doc["value"] == 0.5 and json.loads((held / "out" / "memory.json").read_text()) == doc
    assert f"memory: the result ({', '.join(SHARED)}) is not written: " in err
    assert "already holds a measurement named metric" in err
    assert (held / "result.json").read_bytes() == kept and written(held)["metric"]["inputs"][0] == RUN


class Local:
    """`research.store.local.LocalStore` as `run.EvidenceStore` uses it: attempts, the withdrawn label, a record's listing and
    fetches. Its record holds every path but `nothing`, and `lost` is a file whose bytes the store no longer has."""

    def __init__(self, root: Path, withdrawn: dict) -> None:
        self.root, self.gone, self.fetched = root, withdrawn, []

    def get_attempt(self, run: str) -> dict | None:
        return {"outputs": {"run_record": f"art:{run}"}}

    def withdrawn(self, target: str) -> str | None:
        return self.gone.get(target)

    def listing(self, art: str, paths: list[str]) -> list[dict]:
        return [{"path": p} for p in paths if p != "nothing"]

    def fetch(self, art: str, paths: list[str]) -> Path:
        self.fetched.append(art)
        if paths == ["lost"]:
            raise OSError(f"{art}: blob is not local and not on the remote")
        return self.root


@pytest.mark.parametrize("target", [f"art:{RUN}", RUN])
def test_a_withdrawn_run_is_refused_before_any_of_it_is_fetched(tmp_path, target):
    """The store's own adapter, over a stand-in for its client: a `withdrawn` label on the run or on its record
    refuses the run, and its files are never asked for."""
    pulled = []
    s = M.EvidenceStore.__new__(M.EvidenceStore)
    s.store, s.error = Local(tmp_path, {target: "'held a secret', by someone"}), OSError
    s.sync = SimpleNamespace(read_through=lambda store, targets, warn: pulled.append(targets))
    with pytest.raises(M.Unreadable, match="was withdrawn .*held a secret.*: nothing of it is fetched"):
        s.files(RUN, ["out/verifier.json"])
    assert s.store.fetched == [] and pulled == [[RUN, f"art:{RUN}"]], "the remote's labels are pulled first"
    s.store.gone = {}
    assert s.files(RUN, ["out/verifier.json"]) == tmp_path
    assert s.files(RUN, ["nothing"]) is None and s.store.fetched == [f"art:{RUN}"], "a file the record lacks isn't asked for"
    with pytest.raises(M.Unreadable, match="record can't be fetched: .*blob is not local"):
        s.files(RUN, ["lost"])
