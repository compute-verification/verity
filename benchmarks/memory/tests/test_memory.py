"""`verity bench memory` (`run.py`): the number is the γ at the specification's rate that an accepted session's
verifier recorded, printed as text and as one line of JSON; a session that wasn't accepted, or a run with no verifier's
record, prints no number and exits 1. The number is printed with its error, the ledger's declared one, and a session
whose error is 1 or more prints "no bound at this period" with it and no number. The four ways in share that printer: a recorded run, the loopback (which says
it certifies nothing), the node run (a gate run of one honest session, read from its fire run) and `--local` (a served
session on this machine, which says it is a measurement and not an audit). The two references
are read from their runs, and one that can't be read never takes the number down. Under a research run an accepted
session's amounts are also written through the harness kit's writer, and what is printed stays the same to the byte.

Nothing here runs a session, reaches the evidence store or waits: the records are built in the test, `FakeStore`
stands in for the store, and the processes the command starts and its waits are stand-ins handed to `main`."""
from __future__ import annotations

import fnmatch
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.memory import run as M  # noqa: E402
from benchmarks.remote_session import summary  # noqa: E402
from verity import protocol  # noqa: E402
from verity_bench import kit  # noqa: E402

RUN, FIRE, PROVER = "r20000101-000000-aaaa", "r20000101-000001-bbbb", "r20000101-000002-cccc"
POUS, WIPE = "r20000101-000003-dddd", "r20000101-000004-eeee"
COMMIT = "c" * 40
PACED = M.SETTINGS.setting("PACED").default
#: gate_run.sh's variables for a host run of sessions with one coin a window
WINDOWED = ("SESSIONS", "WINDOWS", "WINDOW_MS", "SAMPLES")
GIB = 1 << 30
#: one card's HBM, as its card's sheet states it (`store.dram`)
GPU_HBM = {"h200-sxm-141gb-x1": 151397597184, "rtx-pro-6000-server-x8": 103079215104}
SESSION = HERE / "fixtures" / "session.json"
UNDECIDED = ("undecided: the Lean's answer for the γ terms isn't written yet "
             "(memory-accounting/lean-decides-memory-verdicts)")
#: the bench's own run, and the host stamp the kit's writer stamps each measurement with
BENCH = "r20000101-000005-ffff"
STAMP = {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": "test", "commit": "d" * 40}
#: the result's measurements, in the order they are written
SHARED = ["metric", "verified", "capacity at spec", "free at spec", "capacity at measured", "free at measured", "error"]
#: what a node's session writes: the shared fields, and its number again under the Metrics tab's id
WRITTEN = ["metric", M.METRIC, *SHARED[1:]]


def a_session(k: int = 0, *, accepted: bool = True, gamma=0.25, failure: str | None = None, verdict="lean") -> dict:
    """One session as the verifier records it (`fixtures/session.json`: the fields `summary.py` and the bench read),
    with this one's index, verdict and γ."""
    s = json.loads(SESSION.read_text())
    s["ledger"]["spec"]["gamma"] = gamma
    return {**s, "k": k, "accepted": accepted, "verdict": verdict,
            "failure": None if failure is None else {"t": 0, "failure": failure, "at": None}}


def an_undecided_session() -> dict:
    """An accepted session whose γ at the data sheet's rate is undecided: its Lean answer isn't written."""
    s = a_session()
    s["ledger"]["spec"]["undecided"] = UNDECIDED
    return s


def a_python_ledger_session(gamma: float = 0.199, error: float = 2.0 ** -40) -> dict:
    """An accepted session whose γ the Lean leaves undecided, with the Python ledger's reading the verifier records
    beside it (`python_ledger`): γ at the data sheet's rate and the declared error."""
    s = an_undecided_session()
    del s["ledger"]["spec"]["gamma"]
    s["ledger"]["spec"]["python_ledger"] = {"gamma": gamma}
    s["ledger"]["declared"] = {"undecided": UNDECIDED, "python_ledger": {"gamma": {"spec": {"error": {"windows": error}}}}}
    return s


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


def undecided(s: dict) -> dict:
    """Session `s` as its verifier records it where the Lean leaves γ's terms undecided: each reading and the
    declaration undecided, the reading they would have been beside them (`python_ledger`)."""
    ledger = s["ledger"]
    for r in ("spec", "measured", "declared"):
        ledger[r] = {**({"capacity_bytes_per_s": ledger[r]["capacity_bytes_per_s"]} if r != "declared" else {}),
                     "undecided": UNDECIDED, "python_ledger": ledger[r]}
    return s


def a_record(*sessions: dict, blocks: int = GIB // 4096, binding: dict | None = None) -> dict:
    """A verifier's record of `sessions` over a store of `blocks` blocks."""
    return {"registration": {"blocks": blocks}, "sessions": list(sessions), "calibration": {}, "rtt": {},
            "allowance_ns": 1, "process_cpu_ns": 1, "wire": {}, **({"binding": binding} if binding else {})}


#: `M.PACE` as the verifier's `pace_of` records it
PACE = {"k": 1, "t_ms": 1000.0, "draws_t": 2773, "windows": 7, "groups": 12, "draws": 232, "pass_ms": 85.0}


def as_local(record: dict) -> dict:
    """`record` as a `--local` session's verifier records it: its own challenger's coins, the registration's 16 sketch
    rows, the one paced session form (`M.PACE`, as `pace_of` reads it at an 85 ms pass) and the prover its error is
    priced against."""
    record["registration"]["rows"] = 16
    for s in record["sessions"]:
        s.update(challenger={"challenger": "local"}, samples=232, delta=2.0 ** -101, windows=96, window_ns=1,
                 allowance_ns=2 * 10 ** 6, pace=PACE)
        s["ledger"]["sketch_failure"] = {"queries_log2": 80}
    return record


#: the `ran with` detail of such a record, under `--local` on an H200
LOCAL_DETAIL = ("2773 openings a window; 7 windows of 1,000 ms; 16 sketch rows, not the deployed registration's 64; the "
                "sample's term a window 2^-101.0; lateness allowed 2 ms; the error against a prover of 2^80 SHA-512 "
                "compressions on any hardware; the share against hardware/nodes/h200-sxm-141gb-x1.toml at one card")


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
    for name in ("RESEARCH_RUN_DIR", "OUT", "RESEARCH_RUN_ID", "RESEARCH_SOURCE_SHA", "PACED", "UNPACED", "ROLE", "SERVE",
                 "HOST_GIB", "LEAF_CTAS", "PLAN", "LOCAL_WORK", "VLLM_PY", "GPU", "GPU_LEASE_DIR", "ARENA_GIB", "UTIL",
                 "PACE_PASS_MS", "ENGINE_ARGS", "NODE_ARGS", "VERIFIER_ARGS", "JITTER_MS", "QUANTILE", "ENGINE_CPUS",
                 "VERIFIER_CPUS", "PROVER_CPUS", *WINDOWED):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(M.kit, "host_stamp", lambda: dict(STAMP))
    s = FakeStore(tmp_path / "store")
    s.add(POUS, {"out/summary.json": {"certificate": {"certified": True, "s_bits": 6 * 8 * GIB, "eps": "1/100"},
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
    """What the command printed: its text lines, its JSON document (None with nothing printed) and its stderr. The
    kit's `ran with` line after the first, and `a measurement, not an audit` below it where the document's `audit` is
    false (`kit.report`), are held to the document's `ran_with` here and left out of the lines returned."""
    out, err = capsys.readouterr()
    lines = out.splitlines()
    if not lines:
        return [], None, err
    text, doc = lines[:-1], json.loads(lines[-1])
    said = doc["ran_with"]
    kits = [kit.NOT_STATED if said is None else kit.ran_with(said["coins"], said["parameters"], said["detail"])[0],
            *([] if doc["audit"] else ["a measurement, not an audit"])]
    assert text[1:1 + len(kits)] == kits, text[:4]
    return text[:1] + text[1 + len(kits):], doc, err


def test_an_accepted_session_prints_its_gamma_at_the_data_sheets_rate_as_text_and_as_json(store, capsys):
    store.add(RUN, {"out/verifier.json": a_record(a_session(gamma=0.15625))})
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[0] == ("memory: 0.156 of the node's memory read rate is left uncertified by this session, at the data "
                       "sheet's read bandwidth (2.0000 TB/s); its error reads δ + Σ_t adv_t (SHA-512 stream as PRF: "
                       "not priced, not assumed), and the verifier's record holds no δ for it")
    assert text[1] == (f"  session 0 of {RUN}, commit {COMMIT}: accepted by the Lean verifier, its verifier not on the "
                       f"auditor's verifier host")
    assert text[2] == ("  store: 1.07 GB, 0.125 of the node's 8.59 GB of GPU memory; the verifier held no checkpoint: "
                       "the store's bytes are the developer's choice")
    assert doc["model"] is None and doc["bound"] is False
    assert list(doc)[:4] == ["value", "unit", "commit", "run"]
    assert (doc["value"], doc["unit"], doc["commit"], doc["run"]) == (0.15625, M.UNIT, COMMIT, RUN)
    assert doc["loopback"] is False and doc["host_run"] is None and doc["error"] is None and doc["no_bound"] is None
    assert not any("loopback" in line or "certifies nothing" in line for line in text)


def test_the_number_is_printed_with_the_error_its_ledger_declares(store, capsys):
    store.add(RUN, {"out/verifier.json": a_record(with_amounts(a_session(), error=4.654e-10))})
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[0].startswith("memory: 0.250 of the node's memory read rate is left uncertified by this session")
    assert text[0].endswith("not priced, not assumed), δ = 4.65e-10 (2^-31.0) in the verifier's ledger")
    assert (doc["value"], doc["error"], doc["no_bound"]) == (0.25, 4.654e-10, None)


@pytest.mark.parametrize("error", [1.0, 38.7])
def test_a_session_whose_error_is_1_or_more_prints_no_bound_and_no_number(store, capsys, monkeypatch, tmp_path, error):
    """A δ of 1 or more bounds nothing, so no share is printed with it, in the text or in the document, and no amount
    is written that would read as verified or free: the capacities and the error are."""
    store.add(RUN, {"out/verifier.json": a_record(with_amounts(a_session(), error=error))})
    held = a_run(monkeypatch, tmp_path)
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    why = f"{protocol.NO_BOUND}: the row's error, {error:.4g}, is 1 or more"
    assert protocol.NO_BOUND == "no bound at this period"
    assert text[0] == (f"memory: {why}: this session certifies no share of the node's memory read rate, at the data "
                       f"sheet's read bandwidth (2.0000 TB/s); its error reads {M.ERROR_FORM}, δ = {M.power(error)} in "
                       f"the verifier's ledger")
    assert "0.250" not in text[0] and (doc["value"], doc["error"], doc["no_bound"]) == (None, error, why)
    got = written(held)
    assert {n: m["value"] for n, m in got.items()} == {
        "metric": None, M.METRIC: None, "verified": None, "capacity at spec": 8e12, "free at spec": None,
        "capacity at measured": 6.4e12, "free at measured": None, "error": error}
    assert {n: m.get("why") for n, m in got.items()} == {
        "metric": why, M.METRIC: why, "verified": why, "capacity at spec": None, "free at spec": why,
        "capacity at measured": None, "free at measured": why, "error": None}


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
    (a_record(an_undecided_session()), f"session 0 was accepted and its γ at the specification's rate is {UNDECIDED}"),
    (a_record(a_session(verdict=None)), "session 0 is recorded accepted, but its record names no Lean verdict (None)"),
    (a_record(a_session(verdict="python")),
     "session 0 is recorded accepted, but its record names no Lean verdict ('python')"),
    ({"sessions": "none"}, "the verifier's record doesn't read as one"),
    ("not json", "the verifier's record doesn't read as one"),
])
def test_a_session_that_was_not_accepted_prints_no_number_and_exits_1(store, capsys, record, why):
    store.add(RUN, {"out/verifier.json": record})
    assert bench(store, "--from", RUN) == 1
    out, err = capsys.readouterr()
    assert out == "" and f"memory: no number: {why}" in err


def test_a_run_that_left_no_verifiers_record_prints_no_number_and_exits_3(store, capsys):
    store.add(RUN, {"out/probe.json": {}}, rc=3)
    assert bench(store, "--from", RUN) == 3
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
    assert doc["value"] == 0.625 and doc["model"] == "SOME_ROLE" and doc["bound"] is True and doc["host_run"] == "v2"
    assert text[1].endswith("accepted by the Lean verifier, on the auditor's verifier host (its run v2)")
    assert text[2].endswith("bound to SOME_ROLE's pinned checkpoint")


def test_a_binding_that_did_not_bind_is_said(store, capsys):
    record = a_record(a_session(), binding={"name": "SOME_ROLE", "bound": False})
    store.add(RUN, {"out/verifier.json": record}, dirty=True)
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[1].startswith(f"  session 0 of {RUN}, commit {COMMIT}-dirty: accepted by the Lean verifier")
    assert text[2].endswith(f"; {M.UNBOUND}") and doc["model"] is None and doc["bound"] is False
    assert doc["commit"] == f"{COMMIT}-dirty"


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


def test_a_recorded_local_session_is_printed_as_one_and_writes_no_node_result(store, capsys, monkeypatch, tmp_path):
    """A run `--local` left holds its document, which says so: read back with `--from`, inside another research run,
    it is still a `--local` measurement, in the text and in the JSON, and writes its one-machine row, never the
    node's."""
    held = a_run(monkeypatch, tmp_path)
    store.add(RUN, {"out/verifier.json": a_record(with_amounts(a_session())), M.NODE_FILE: {},
                    "out/memory.json": {"value": 0.25, "run": RUN, "loopback": False, "local": M.LOCAL}})
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[1] == f"  {M.LOCAL}" and (doc["local"], doc["loopback"]) == (M.LOCAL, False)
    got = written(held)
    assert M.METRIC not in got and got[M.LOCAL_METRIC]["value"] == doc["value"]


def test_a_local_run_that_died_before_its_document_is_not_printed_as_a_node_session(store, capsys, work, monkeypatch,
                                                                                    tmp_path):
    """The records a real `--local` run leaves, read back as they would be had it died after the session and before
    writing its document: its node file marks it, so it is refused, by run id and by directory alike."""
    held = a_run(monkeypatch, tmp_path)
    assert bench(store, "--local", run=Machine()) == 0
    capsys.readouterr()
    (held / "out" / "memory.json").unlink()
    monkeypatch.delenv("RESEARCH_RUN_DIR")
    monkeypatch.delenv("RESEARCH_RUN_ID")
    store.add(RUN, {f: (held / f).read_text() for f in ("out/verifier.json", M.NODE_FILE)})
    assert bench(store, "--from", RUN) == 2
    out, err = capsys.readouterr()
    assert out == "" and f"{RUN} holds --local's {M.NODE_FILE} but no out/memory.json saying what it was" in err
    assert M.main(["--from", str(held)], store_of=lambda: store) == 2
    assert f"holds --local's {M.NODE_FILE}" in capsys.readouterr().err


@pytest.mark.parametrize("marks", [{}, {"loopback": "no"}, {"loopback": False, "local": 1},
                                   {"loopback": False, "local": ""}, {"loopback": True, "local": "local: x"}])
def test_a_run_whose_document_does_not_say_what_it_was_is_not_printed(store, capsys, marks):
    store.add(RUN, {"out/verifier.json": a_record(a_session()), "out/memory.json": {"value": 0.25, "run": RUN, **marks}})
    assert bench(store, "--from", RUN) == 2
    out, err = capsys.readouterr()
    assert out == "" and f"{RUN}'s out/memory.json doesn't say what its session was" in err


def test_a_run_that_ran_local_keeps_its_document_when_it_prints_another_session(store, capsys, monkeypatch, tmp_path):
    held = a_run(monkeypatch, tmp_path)
    (held / "out").mkdir(parents=True)
    mine = {"value": 0.25, "run": BENCH, "loopback": False, "local": M.LOCAL}
    (held / "out" / "memory.json").write_text(json.dumps(mine))
    store.add(RUN, {"out/verifier.json": a_record(with_amounts(a_session()))})
    assert bench(store, "--from", RUN) == 0
    assert json.loads((held / "out" / "memory.json").read_text()) == mine
    assert "this run ran --local, and its out/memory.json stays --local's document" in capsys.readouterr().err


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
    assert text[3] == (f"  beside it, PoUS state bound: 0.750 (6.44 GB of 8.59 GB, through the device model, not the "
                       f"answer; error 0.01; {POUS}, 2000-01-02, CARD A)")
    assert text[4] == (f"  beside it, wipe residue: 0.125 (1.07 GB of 8.59 GB; error 2.94e-39; {WIPE}, 2000-01-02, "
                       f"CARD B)")
    assert [(r["name"], r["value"], r["error"], r["run"], r["date"], r["card"]) for r in doc["beside"]] == [
        ("PoUS state bound", 0.75, 0.01, POUS, "2000-01-02", "CARD A"),
        ("wipe residue", 0.125, 2.0 ** -128, WIPE, "2000-01-02", "CARD B")]
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


def test_a_reference_states_no_number_where_its_error_is_1_or_more_and_says_where_its_record_holds_none(store, capsys):
    store.add(RUN, {"out/verifier.json": a_record(a_session(gamma=0.25))})
    store.add(POUS, {"out/summary.json": {"certificate": {"certified": True, "s_bits": 6 * 8 * GIB, "eps": "3/2"},
                                          "held": {"total_bytes": 8 * GIB, "device": "CARD A"}}})
    store.labels[WIPE, "label"] = "full-fill wipe certified: foreign <= 8,589,934,592 bits, M = 8,589,934,592 B"
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    why = f"{protocol.NO_BOUND}: the row's error, 1.5, is 1 or more"
    assert text[3] == f"  beside it, PoUS state bound: {why} ({POUS}, 2000-01-02, CARD A)"
    assert text[4] == (f"  beside it, wipe residue: 0.125 (1.07 GB of 8.59 GB; no error in its record; {WIPE}, "
                       f"2000-01-02, CARD B)")
    assert [(r["value"], r["error"], r.get("no_bound")) for r in doc["beside"]] == [(None, 1.5, why), (0.125, None, None)]


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


@pytest.mark.parametrize("rc, record, why, code", [
    (3, None, "the loopback's processes failed (run.sh exited 3)", 3),
    (0, a_record(a_session(accepted=False, failure="late")), "session 0 was not accepted (late)", 1),
    (0, None, "the verifier's record doesn't read as one", 1),
])
def test_a_loopback_whose_session_failed_prints_no_number(store, capsys, monkeypatch, rc, record, why, code):
    def launch(argv, **kw):
        if record is not None:
            out = Path(kw["env"]["RESEARCH_RUN_DIR"]) / "out"
            out.mkdir(parents=True)
            (out / "verifier.json").write_text(json.dumps(record))
        return SimpleNamespace(returncode=rc)

    monkeypatch.setenv("RESEARCH_SOURCE_SHA", COMMIT)
    assert bench(store, "--loopback", run=launch) == code
    text, doc, _ = printed(capsys)
    assert text[0].startswith(f"memory: no number: {why}") and len(text) == 1
    assert (doc["value"], doc["no_number"], doc["loopback"], doc["commit"]) == (None, text[0][19:], True, COMMIT)
    assert doc["ran_with"] == {"coins": "local challenger", "parameters": "development", "detail": None}


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
    assert not {"HOST_GIB", "LEAF_CTAS", "UNPACED", "BIND", *WINDOWED} & set(env)
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
    ({"HOST_GIB": "8", "UNPACED": "7:60000:33273", "LEAF_CTAS": "160"},
     {"HOST_GIB": "8", "LEAF_CTAS": "160", "SESSIONS": "1", "WINDOWS": "7", "WINDOW_MS": "60000", "SAMPLES": "33273"}),
], ids=["both", "host-memory", "one-coin-a-window", "an-empty-word", "leaf-pass"])
def test_the_node_run_can_put_host_memory_in_the_store_and_ask_for_one_coin_a_window(store, capsys, monkeypatch,
                                                                                    tmp_path, given, want):
    """HOST_GIB and LEAF_CTAS pass through to `gate_run.sh`, which sends them to the node.
    UNPACED's word is one session of that many windows, each that long with that many openings, as `gate_run.sh`'s
    variables for the host's run, and then no PACED goes with it; without it the session is PACED's, and no variable of
    the other form is set."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for name, value in given.items():
        monkeypatch.setenv(name, value)
    seen = []
    store.add(FIRE, {"out/host-run.txt": "v1", "out/host/v1/out/verifier.json": a_record(a_session(gamma=0.15))})
    assert bench(store, "p", run=gate_run(tmp_path, seen)) == 0
    ((argv, env),) = seen
    assert {k: env[k] for k in ("HOST_GIB", "LEAF_CTAS", "PACED", *WINDOWED) if k in env} == want
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
    assert bench(store, "p", run=gate_run(tmp_path, [], rc=3)) == 3
    out, err = capsys.readouterr()
    assert out == "" and "memory: no number: the gate run did not launch (gate_run.sh exited 3)" in err
    assert store.asked == []


def test_a_fire_run_that_never_reaches_the_store_prints_no_number(store, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(M, "WAIT_MIN", 3)
    waits = []
    assert bench(store, "p", run=gate_run(tmp_path, []), wait=waits.append) == 3
    out, err = capsys.readouterr()
    assert out == "" and waits == [60, 60, 60] and f"{FIRE} left no record in the store within 3 minutes" in err


def test_a_fire_run_whose_host_run_failed_prints_no_number(store, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    store.add(FIRE, {"out/probe.json": {}}, rc=3)
    assert bench(store, "p", run=gate_run(tmp_path, [])) == 3
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


@pytest.mark.parametrize("argv", [[], ["p", "--loopback"], ["--from", RUN, "--loopback"], ["p", "--from", RUN],
                                  ["--local", "--loopback"], ["p", "--local"]])
def test_the_command_takes_exactly_one_way_in(store, capsys, argv):
    with pytest.raises(SystemExit) as e:
        bench(store, *argv)
    assert e.value.code == 2 and ("one of PERIOD (the node run), --from RUN, --loopback or --local"
                                  in capsys.readouterr().err)


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
    assert list(doc) == ["value", "unit", "commit", "run", "loopback", "local", "machine", "gamma_decided_by",
                         "host_run", "no_bound", "session", "verdict", "error", "rate_bytes_per_s", "store_bytes",
                         "node_gpu_bytes", "model", "bound", "parameters", "beside", "ran_with", "audit"]
    got = written(held)
    assert list(got) == WRITTEN
    assert {n: m["value"] for n, m in got.items()} == {
        "metric": 0.25, M.METRIC: 0.25, "verified": 6e12, "capacity at spec": 8e12, "free at spec": 2e12,
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
    # and says it in the words every δ of the session's is read in, never as a bare error
    assert "δ + Σ_t adv_t (SHA-512 stream as PRF: not priced, not assumed), this number its δ" in M.ERROR
    rate, span = "ledger.{}.capacity_bytes_per_s", "ledger.inputs.span_ns"
    assert {n: m["inputs"] for n, m in got.items()} == {
        "metric": ["measured", RUN, "ledger.spec.gamma"], M.METRIC: ["measured", RUN, "ledger.spec.gamma"],
        "verified": ["measured", RUN, "ledger.spec.certified_bytes"],
        "capacity at spec": ["measured", RUN, rate.format("spec"), span],
        "free at spec": ["measured", RUN, "ledger.spec.gamma", rate.format("spec"), span],
        "capacity at measured": ["measured", RUN, rate.format("measured"), span],
        "free at measured": ["measured", RUN, "ledger.measured.gamma", rate.format("measured"), span],
        "error": ["measured", RUN, "ledger.declared.gamma.spec.error.windows"]}
    for n, m in got.items():
        assert (m["mark"], m["run"], m["commit"], m["host"]) == ("measured", BENCH, STAMP["commit"], STAMP)
        assert (m["target"], m["verdict"], m["ci95"]) == (
            (0.14, "past target; the verifier held no checkpoint; its error is above 2^-100", None) if n == M.METRIC
            else (None, None, None))
        assert m.get("bound", "absent") == (False if n == M.METRIC else "absent")
    assert got[M.METRIC]["unit"] == M.UNIT


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
    assert list(written(held)) == [n for n in WRITTEN if n not in left_out]


def test_a_record_that_holds_only_the_share_writes_the_metric_alone(store, monkeypatch, tmp_path):
    store.add(RUN, {"out/verifier.json": a_record(a_session(gamma=0.375))})
    held = a_run(monkeypatch, tmp_path)
    assert bench(store, "--from", RUN) == 0
    assert {n: m["value"] for n, m in written(held).items()} == {"metric": 0.375, M.METRIC: 0.375}


def test_a_share_of_one_or_more_writes_nothing_verified_and_the_whole_capacity_free(store, monkeypatch, tmp_path):
    """The verifier's certified bytes are then at most none. The metric stays the printed share."""
    store.add(RUN, {"out/verifier.json": a_record(with_amounts(a_session(gamma=1.25)))})
    held = a_run(monkeypatch, tmp_path)
    assert bench(store, "--from", RUN) == 0
    got = {n: m["value"] for n, m in written(held).items()}
    assert got == {"metric": 1.25, M.METRIC: 1.25, "verified": 0.0, "capacity at spec": 8e12, "free at spec": 8e12,
                   "capacity at measured": 6.4e12, "free at measured": 6.4e12, "error": 2.0 ** -30}


def test_the_node_runs_session_writes_its_amounts_from_the_fire_run(store, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    held = a_run(monkeypatch, tmp_path)
    store.add(FIRE, {"out/host-run.txt": "v1",
                     "out/host/v1/out/verifier.json": a_record(with_amounts(a_session(gamma=0.5)))})
    assert bench(store, "some-period", run=gate_run(tmp_path, [])) == 0
    _, doc, _ = printed(capsys)
    got = written(held)
    assert list(got) == WRITTEN and got["metric"]["value"] == doc["value"] == 0.5
    assert (got["verified"]["value"], got["free at spec"]["value"]) == (4e12, 4e12)
    assert all(m["inputs"][:2] == ["measured", FIRE] for m in got.values())


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
    assert f"memory: the result ({', '.join(WRITTEN)}) is not written: " in err
    assert "already holds a measurement named metric" in err
    assert (held / "result.json").read_bytes() == kept and written(held)["metric"]["inputs"][1] == RUN


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


@pytest.mark.parametrize("gamma, verdict", [(0.14, "within target"), (0.1401, "past target"), (0.199, "past target")])
def test_the_metrics_tabs_row_holds_the_number_against_its_target(gamma, verdict):
    """The Metrics tab's row is `benchmarks/memory/metrics.toml`'s, its unit the short form of the measurement's, and its
    value is the printed number with the goal's target beside it, a share at the target being within it."""
    rows = {m["id"]: m for m in tomllib.loads((ROOT / "benchmarks" / "memory" / "metrics.toml").read_text())["metric"]}
    row = rows[M.METRIC]
    assert (row["owner"], row["better"]) == ("memory-accounting", "lower") and M.UNIT.startswith(f"{row['unit']} ")
    assert "benchmarks/memory/run.py --from" in row["command"]
    within = {"declared": {"gamma": {"spec": {"error": {"windows": 2.0 ** -100}}}}}
    doc = {"value": gamma, "unit": M.UNIT, "run": RUN, "bound": True}
    (m,) = [m for m in M.amounts(doc, within) if m["name"] == M.METRIC]
    assert (m["value"], m["target"], m["verdict"]) == (gamma, M.TARGET, verdict) and M.TARGET == 0.14
    assert "extra" not in m


@pytest.mark.parametrize("bound", [False, None], ids=["unbound", "older record"])
def test_an_unbound_sessions_row_is_written_and_says_its_verifier_held_no_checkpoint(store, monkeypatch, tmp_path, bound):
    """A session whose verifier bound its store to no checkpoint still writes the Metrics tab's row; the entry carries
    `"bound": false` and its verdict names the premise, which the console and `research metrics` show as written. A
    document with no `bound` (a record from before it) reads as unbound."""
    within = {"declared": {"gamma": {"spec": {"error": {"windows": 2.0 ** -101}}}}}
    doc = {"value": 0.127, "unit": M.UNIT, "run": RUN, **({} if bound is None else {"bound": bound})}
    (m,) = [m for m in M.amounts(doc, within) if m["name"] == M.METRIC]
    assert m["verdict"] == "within target; the verifier held no checkpoint" and m["extra"] == {"bound": False}
    store.add(RUN, {"out/verifier.json": a_record(with_amounts(a_session(), error=2.0 ** -100),
                                                  binding={"name": "SOME_ROLE", "bound": True})})
    held = a_run(monkeypatch, tmp_path)
    assert bench(store, "--from", RUN) == 0
    bound_row = written(held)[M.METRIC]
    assert bound_row["verdict"] == "past target" and "bound" not in bound_row


@pytest.mark.parametrize("ledger, why", [
    ({"declared": {"gamma": {"spec": {"error": {"windows": 2.0 ** -31}}}}}, "the whole error, today's"),
    ({"declared": {"gamma": {"spec": {"error": {"windows": 2.0 ** -99}}}}}, "the whole error, just above"),
    ({"spec": {"error": {"windows": 2.0 ** -40}}}, "no declared error: the ledger's own"),
    ({}, "no error at all"),
], ids=["today", "just-above", "ledger-only", "none"])
def test_a_row_whose_error_is_above_2_to_the_minus_100_says_so_in_its_verdict(ledger, why):
    """The goal holds the number at an error of at most 2^-100: a row whose error the record holds above it, the
    declared error where it has one and else the ledger's own, or holds none, names that premise after the
    checkpoint's, so a share under 0.14 never reads "within target" alone."""
    for bound, held_none in ((True, ""), (False, "; the verifier held no checkpoint")):
        doc = {"value": 0.127, "unit": M.UNIT, "run": RUN, "bound": bound}
        (m,) = [m for m in M.amounts(doc, ledger) if m["name"] == M.METRIC]
        assert m["verdict"] == f"within target{held_none}; its error is above 2^-100", why
    assert M.ABOVE_BOUND == "its error is above 2^-100" and M.ERROR_BOUND_LOG2 == -100


def test_a_declared_error_at_the_bound_is_preferred_to_the_ledgers_own_and_holds_no_premise():
    """The declared error (the ledger's with the sketch's and the sample's) decides where the record holds it: a ledger's
    own error above the bound beside a declared one at it adds nothing; the declared one above it, beside a ledger's own
    under it, does."""
    doc = {"value": 0.127, "unit": M.UNIT, "run": RUN, "bound": True}

    def verdict(declared: float, own: float) -> str:
        ledger = {"declared": {"gamma": {"spec": {"error": {"windows": declared}}}}, "spec": {"error": {"windows": own}}}
        return next(m["verdict"] for m in M.amounts(doc, ledger) if m["name"] == M.METRIC)

    assert verdict(2.0 ** -100, 2.0 ** -31) == "within target"
    assert verdict(2.0 ** -31, 2.0 ** -120) == "within target; its error is above 2^-100"


#: a --local store: the arena `arena_of` gives an H200, which the store guard takes
ARENA = 112


def a_local_record(*sessions: dict, arena: int = ARENA) -> dict:
    """A `--local` session's verifier's record over an arena of `arena` GiB (`as_local`)."""
    return as_local(a_record(*sessions, blocks=arena * GIB // 4096))


class Machine:
    """This machine as `--local` asks it, one stand-in for every process it starts: nvidia-smi answers `card` (or fails
    with `smi_rc`, or is not installed, `driver` false), the checkpoint's fetch ends `fetch_rc`, and `run.sh` writes
    `record` as its verifier's, its engine's record with the `--arena-gib` and `--util` it was handed as `engine.py`
    does (unless `engine` is false), and ends `rc`."""

    def __init__(self, *, card: str = "NVIDIA H200, 9.0", smi_rc: int = 0, fetch_rc: int = 0, rc: int = 0,
                 record: dict | None = None, driver: bool = True, engine: bool = True) -> None:
        self.card, self.smi_rc, self.fetch_rc, self.rc, self.driver = card, smi_rc, fetch_rc, rc, driver
        self.engine = engine
        self.record = a_local_record(a_session(gamma=0.199)) if record is None else record
        self.started: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kw):
        self.started.append((argv, dict(kw.get("env") or {})))
        if argv[0] == "nvidia-smi":
            if not self.driver:
                raise FileNotFoundError(2, "No such file or directory", "nvidia-smi")
            return SimpleNamespace(returncode=self.smi_rc, stdout=self.card + "\n" if not self.smi_rc else "",
                                   stderr="" if not self.smi_rc else "NVIDIA-SMI has failed")
        if argv[0] == "git":
            return SimpleNamespace(returncode=0, stdout=COMMIT + "\n" if argv[1] == "rev-parse" else "")
        if argv[1:3] == ["-c", M.FETCH]:
            return SimpleNamespace(returncode=self.fetch_rc)
        out = Path(kw["env"]["RESEARCH_RUN_DIR"]) / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "verifier.json").write_text(json.dumps(self.record))
        if self.engine:
            given = kw["env"].get("ENGINE_ARGS", "").split()
            args = {k: float(given[given.index(f) + 1]) for k, f in (("arena_gib", "--arena-gib"), ("util", "--util"))
                    if f in given}
            args["arena_gib"] = int(args["arena_gib"]) if "arena_gib" in args else None
            (out / "engine.0.json").write_text(json.dumps({"schema": "verity/remote-session/engine/v0", "args": args}))
        return SimpleNamespace(returncode=self.rc)

    def runs(self) -> list[tuple[list[str], dict]]:
        """The `run.sh` processes started."""
        return [(a, e) for a, e in self.started if a[:2] == ["bash", str(M.RS / "run.sh")]]

    def fetches(self) -> list[list[str]]:
        return [a for a, _ in self.started if a[1:3] == ["-c", M.FETCH]]


@pytest.fixture
def work(tmp_path, monkeypatch) -> Path:
    """LOCAL_WORK in the test's directory, a vLLM interpreter of the person's own, and no `gpu-lease` installed nor a
    lease directory of the machine's."""
    monkeypatch.setenv("LOCAL_WORK", str(tmp_path / "work"))
    monkeypatch.setattr(M, "LEASE_DIR", tmp_path / "no-lease-dir")
    monkeypatch.setenv("VLLM_PY", str(tmp_path / "vllm" / "python"))
    monkeypatch.setattr(M.shutil, "which", lambda name: None)
    return tmp_path / "work"


def test_local_runs_the_nodes_session_on_one_card_with_the_verifier_beside_it_and_says_first_it_is_no_audit(
        store, capsys, work, tmp_path):
    """`--local` runs `run.sh` with ROLE=both, SERVE=node and GPUS=1: one engine serving the role's pinned checkpoint
    from an arena of most of the card's memory, the node prover's paced passes over it, and the verifier beside them
    on the one-card node, all kept in LOCAL_WORK. Its number is its verifier's, and the text's first lines and the
    JSON say the verifier and the coins ran on the machine measured; the references are left out with that reason."""
    machine = Machine()
    assert bench(store, "--local", run=machine) == 0
    ((argv, env),) = machine.runs()
    pin = json.loads(M.PIN.read_text())[M.ROLE]
    snap = work / "hf" / "hub" / f"models--{pin['repo'].replace('/', '--')}" / "snapshots" / pin["revision"]
    assert (env["ROLE"], env["SERVE"], env["GPUS"], env["PLAN"], env["MODEL"]) == ("both", "node", "1", "honest",
                                                                                   str(snap))
    assert machine.fetches() == [[str(tmp_path / "vllm" / "python"), "-c", M.FETCH, pin["repo"], pin["revision"],
                                  *sorted(pin["files"])]]
    assert env["ENGINE_ARGS"] == f"--step-hz 2 --align-steps --arena-gib {ARENA} --util 0.88"
    assert env["NODE_ARGS"] == f"--decode-s 60 --warm-openings 2773 --wait-s 1800 --model-dir {snap}"
    assert (env["PACED"], env["PACE_PASS_MS"]) == ("1:1000:2773:7", "0"), "run.sh passes the form, once"
    assert env["VERIFIER_ARGS"] == (f"--node {env['RESEARCH_RUN_DIR']}/{M.NODE_FILE} "
                                    f"--hbm-bytes {GPU_HBM['h200-sxm-141gb-x1']}")
    assert (env["JITTER_MS"], env["QUANTILE"]) == ("50", "max")
    assert (env["NODE_WORKSPACE"], env["HF_HOME"], env["UV_CACHE_DIR"], env["GPU_LEASE_DIR"]) == (
        str(work), str(work / "hf"), str(work / "cache/uv"), str(work / "gpu-lease"))
    assert not [k for k in (*WINDOWED, "ROW", "ROW_ARGS") if k in env] and "UNPACED" not in env, \
        "the windowed form and the row are not passed"
    assert Path(env["RESEARCH_RUN_DIR"]).parent == work / "runs", "outside a research run its records are kept"
    assert json.loads((Path(env["RESEARCH_RUN_DIR"]) / M.NODE_FILE).read_text()) == M.one_card("h200-sxm-141gb-x1")[0]
    text, doc, err = printed(capsys)
    assert text[0].startswith("memory: 0.199 of the node's memory read rate is left uncertified by this session")
    assert doc["audit"] is False and doc["ran_with"]["coins"] == "local challenger"
    assert doc["ran_with"]["detail"].startswith(f"an arena of {ARENA} GiB at vLLM's 0.88 of the card, the store "
                                                f"{ARENA * GIB / 1e9:.2f} GB of its ")
    assert "; paced: 7 credited stretches of 1,000 ms, 12 groups a stretch at 85 ms a pass" in doc["ran_with"]["detail"]
    assert text[1] == f"  {M.LOCAL}" and doc["local"] == M.LOCAL
    assert text[2] == (f"  card: NVIDIA H200, the verifier's node one card of hardware/nodes/h200-sxm-141gb-x1.toml; "
                       f"data sheets sha256 {M.digest(M.SHEETS)[:16]}, verifier sha256 {M.digest(M.VERIFIER)[:16]}")
    assert text[3] == f"  session 0 of no recorded run, commit {COMMIT}: accepted by the Lean verifier"
    assert doc["machine"] == {"card": "NVIDIA H200", "sheet": "h200-sxm-141gb-x1", "arena_gib": ARENA, "util": 0.88,
                              "sheet_sha256": M.digest(M.SHEETS), "verifier_sha256": M.digest(M.VERIFIER)}
    assert (doc["loopback"], doc["value"], doc["commit"]) == (False, 0.199, COMMIT)
    assert [(r["value"], r["why"]) for r in doc["beside"]] == [(None, M.NO_STORE)] * 2 and store.asked == []
    assert f"a {ARENA} GiB arena of its 141 GiB" in err


@pytest.mark.parametrize("card, arena", [("NVIDIA H200, 9.0", 112),
                                         ("NVIDIA RTX PRO 6000 Blackwell Server Edition, 12.0", 72)])
def test_the_arena_is_most_of_the_cards_memory_from_its_sheet(store, work, card, arena):
    """floor(0.88 × the card's GiB) − 12: 112 GiB on an H200, 72 on an RTX PRO 6000."""
    machine = Machine(card=card, record=a_local_record(a_session(gamma=0.199), arena=arena))
    assert bench(store, "--local", run=machine) == 0
    ((_, env),) = machine.runs()
    assert env["ENGINE_ARGS"] == f"--step-hz 2 --align-steps --arena-gib {arena} --util 0.88"


def test_the_arena_vllm_py_gpu_and_cores_are_the_persons_to_set(store, capsys, work, monkeypatch, tmp_path):
    """ARENA_GIB and UTIL size the arena, GPU names the card nvidia-smi is asked about (and `run.sh` leases), the
    person's own cores win, an installed `gpu-lease` keeps its own locks, and a checkpoint already there is not
    fetched."""
    pin = json.loads(M.PIN.read_text())[M.ROLE]
    snap = work / "hf" / "hub" / f"models--{pin['repo'].replace('/', '--')}" / "snapshots" / pin["revision"]
    snap.mkdir(parents=True)
    for f in pin["files"]:
        (snap / f).write_text("")
    for k, v in {"ARENA_GIB": "80", "UTIL": "0.7", "GPU": "3", "PROVER_CPUS": "4-7"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(M.shutil, "which", lambda name: "/usr/local/bin/gpu-lease")
    machine = Machine(record=a_local_record(a_session(gamma=0.199), arena=80))
    assert bench(store, "--local", run=machine) == 0
    ((_, env),) = machine.runs()
    assert env["ENGINE_ARGS"] == "--step-hz 2 --align-steps --arena-gib 80 --util 0.7"
    assert (env["GPU"], env["PROVER_CPUS"]) == ("3", "4-7") and "GPU_LEASE_DIR" not in env
    assert machine.started[0][0][-2:] == ["-i", "3"] and machine.fetches() == []
    assert printed(capsys)[1]["machine"]["arena_gib"] == 80


def test_a_machines_own_lease_directory_stays_the_one_its_lease_counts_in(store, work, monkeypatch, tmp_path):
    """A lease in LOCAL_WORK would take a card no other lane on the machine sees taken: where the machine has its
    lease directory, `--local` leaves GPU_LEASE_DIR to it."""
    (tmp_path / "no-lease-dir").mkdir()
    machine = Machine()
    assert bench(store, "--local", run=machine) == 0
    ((_, env),) = machine.runs()
    assert "GPU_LEASE_DIR" not in env


def test_local_splits_this_process_cores_among_the_engine_the_prover_and_the_verifier(store, work):
    machine = Machine()
    assert bench(store, "--local", run=machine) == 0
    ((_, env),) = machine.runs()
    assert (env["ENGINE_CPUS"], env["PROVER_CPUS"], env["VERIFIER_CPUS"]) == M.thirds()
    have = sorted(M.os.sched_getaffinity(0))
    assert sorted(int(c) for k in ("ENGINE_CPUS", "PROVER_CPUS", "VERIFIER_CPUS") for c in env[k].split(",")) == have \
        or len(have) < 4


@pytest.mark.parametrize("given", [{"PACED": "1:60000:33273:2"}, {"PACE_PASS_MS": "85"}, {"WINDOWS": "7"},
                                   {"UNPACED": "7:60000:33273"}, {"SESSIONS": "2", "SAMPLES": "64"}])
def test_local_refuses_another_session_form(store, capsys, work, monkeypatch, given):
    for k, v in given.items():
        monkeypatch.setenv(k, v)
    machine = Machine()
    assert bench(store, "--local", run=machine) == 2
    assert machine.started == []
    err = capsys.readouterr().err
    assert "--local runs one session form, PACED=1:1000:2773:7 at PACE_PASS_MS=0" in err
    assert all(f"{k}={v}" in err for k, v in given.items())


def test_a_store_below_half_the_cards_memory_prints_no_number(store, capsys, work, tmp_path, monkeypatch):
    """The share is of the card's read rate while its memory is read in timed passes: a session whose store is a
    request's bytes (the served row's 0.02 GB, which left 1.000 uncertified) is not this benchmark's, so a store
    below `STORE_MIN` of the card's memory prints no number and its row says why, whatever its share."""
    monkeypatch.setenv("OUT", str(tmp_path / "mine"))
    record = as_local(a_record(a_session(gamma=0.199), blocks=16_322_560 // 4096))
    assert bench(store, "--local", run=Machine(record=record)) == 1
    text, doc, _ = printed(capsys)
    assert text[0] == ("memory: no number: the store is 0.02 GB, 0.000 of the card's 151.40 GB, below the 0.5 this "
                       "benchmark reads: the session read a request's bytes and not the card's memory, so its share "
                       "is not this benchmark's") and doc["value"] is None
    (m,) = json.loads((tmp_path / "mine" / "result.json").read_text())["measurements"]
    assert (m["name"], m["value"]) == (M.LOCAL_METRIC, None) and "below the 0.5" in m["why"]
    assert bench(store, "--local", run=Machine(record=a_local_record(a_session(gamma=0.199), arena=71))) == 0


def test_a_local_session_under_a_research_run_writes_its_own_metrics_row(store, capsys, work, tmp_path, monkeypatch):
    """Its document is the run's `out/memory.json`, and its number reaches the Metrics tab under its own row, not the
    node's: a measurement, not an audit, of a share the Python ledger computed, which its entry says."""
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path / "rundir"))
    monkeypatch.setenv("RESEARCH_RUN_ID", BENCH)
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", COMMIT)
    machine = Machine(record=a_local_record(undecided(with_amounts(a_session(gamma=0.199)))))
    assert bench(store, "--local", run=machine) == 0
    ((_, env),) = machine.runs()
    assert env["RESEARCH_RUN_DIR"] == str(tmp_path / "rundir")
    _, doc, _ = printed(capsys)
    assert doc["run"] == BENCH and doc["local"] == M.LOCAL and doc["gamma_decided_by"] == M.PYTHON_LEDGER
    assert json.loads((tmp_path / "rundir" / "out" / "memory.json").read_text()) == doc
    got = written(tmp_path / "rundir")
    assert M.METRIC not in got and list(got)[:2] == ["metric", M.LOCAL_METRIC]
    assert list(got) == ["metric", M.LOCAL_METRIC, *SHARED[1:]], "the bytes and the error, from the Python ledger"
    assert got["error"]["value"] == 2.0 ** -30 and got["verified"]["inputs"] == ["measured", BENCH,
                                                                            "ledger.spec.python_ledger.certified_bytes"]
    row = got[M.LOCAL_METRIC]
    assert (row["value"], row["target"]) == (0.199, M.TARGET) and row["value"] == doc["value"]
    assert row["verdict"] == f"past target; {M.HELD_NONE}; {M.ABOVE_BOUND}; {M.PYTHON_LEDGER_PREMISE}"
    assert row["inputs"] == ["measured", BENCH, "ledger.spec.python_ledger.gamma"]
    assert (row["bound"], row["audit"], row["gamma_decided_by"]) == (False, False, M.PYTHON_LEDGER)


def test_a_local_session_with_out_writes_its_document_and_result_there(store, capsys, work, tmp_path, monkeypatch):
    """Outside a research run, OUT names where the document and the result are written, the files the README names."""
    monkeypatch.setenv("OUT", str(tmp_path / "mine"))
    assert bench(store, "--local", run=Machine()) == 0
    _, doc, _ = printed(capsys)
    assert json.loads((tmp_path / "mine" / "memory.json").read_text()) == doc and doc["local"] == M.LOCAL
    names = [m["name"] for m in json.loads((tmp_path / "mine" / "result.json").read_text())["measurements"]]
    assert names[:2] == ["metric", M.LOCAL_METRIC]


def test_the_ran_with_line_names_the_coins_the_form_and_the_prover_the_error_is_priced_against():
    """The kit's `ran with` line, from the record: a local challenger's coins at the development registration's 16
    rows, with the session's form and the prover its error is priced against in the detail; `deployed` only for the
    auditor device's coins over the deployed 64 rows; not stated for a record that names no challenger. The
    record's `delta` is the sample's term a window, which the line doesn't call the error."""
    v = {"registration": {"rows": 16}}
    vs = {"challenger": {"challenger": "local"}, "samples": 2773, "delta": 2 ** -101, "windows": 7,
          "window_ns": 10 ** 9, "allowance_ns": 2 * 10 ** 6, "ledger": {"sketch_failure": {"queries_log2": 80}}}
    p = M.parameters(v, vs)
    assert p == {"coins": "local", "rows": 16, "samples": 2773, "delta": 2 ** -101, "windows": 7, "window_ms": 1000.0,
                 "allowance_ms": 2.0, "budget_log2": 80, "passes": None}
    machine = {"sheet": "h200-sxm-141gb-x1"}
    line, said = M.ran({"parameters": p, "machine": machine})
    assert line == f"ran with: coins local challenger; parameters development ({LOCAL_DETAIL})"
    assert said == {"coins": "local challenger", "parameters": "development", "detail": LOCAL_DETAIL}
    assert M.ran({"parameters": p | {"coins": "gate"}})[1]["parameters"] == "development"
    _, said = M.ran({"parameters": p | {"coins": "gate", "rows": 64}})
    assert (said["coins"], said["parameters"]) == ("deployed challenger", "deployed") and "not the deployed" not in said["detail"]
    assert M.ran({"parameters": M.parameters({}, {})}) is None
    assert M.detail(M.parameters({}, {})).endswith("the error against a prover of SHA-512 compressions not stated")
    assert "; error 2.8e-05 (2^-15.1) against a prover of 2^80" in M.detail(p, error=2.8047495865732136e-05)


def test_a_paced_sessions_line_is_its_records_pace_and_states_no_windows():
    """A paced session, in the shape the node's verifier records one (`PACED=1:1000:2773:7` at the calibration's
    pass): its record's `windows` are the pass groups it ran and its `window_ns` a placeholder of 1, and its ledger
    counts its passes as one number. The parameters state the pace, the pass groups and the passes, and no windows
    or window length; every figure on the line is the record's."""
    pace = {"k": 1, "t_ms": 1000.0, "draws_t": 2773, "windows": 7, "groups": 19, "draws": 146, "pass_ms": 52.906315}
    vs = {"challenger": {"challenger": "local"}, "samples": 146, "delta": 3.944304526105059e-31, "windows": 152,
          "window_ns": 1, "allowance_ns": 190416547, "pace": pace,
          "ledger": {"sketch_failure": {"queries_log2": 80}, "inputs": {"passes": 152}}}
    p = M.parameters({"registration": {"rows": 16}}, vs)
    assert p == {"coins": "local", "rows": 16, "samples": 146, "delta": 3.944304526105059e-31, "pace": pace,
                 "pass_groups": 152, "allowance_ms": 190.416547, "budget_log2": 80, "passes": 152}
    assert M.detail(p, {"sheet": "h200-sxm-141gb-x1", "arena_gib": 112, "util": 0.88}, 2.8047495865882895e-05,
                    120259084288, GPU_HBM["h200-sxm-141gb-x1"]) == (
        "an arena of 112 GiB at vLLM's 0.88 of the card, the store 120.26 GB of its 151.40 GB; paced: 7 credited "
        "stretches of 1,000 ms, 19 groups a stretch at 52.9063 ms a pass, a coin every 1 pass, 2773 openings a "
        "stretch, 146 a group; 152 passes in 152 groups; 16 sketch rows, not the deployed registration's 64; the "
        "sample's term a pass group 2^-101.0; lateness allowed 190 ms; error 2.8e-05 (2^-15.1) against a prover of "
        "2^80 SHA-512 compressions on any hardware; the share against hardware/nodes/h200-sxm-141gb-x1.toml at one "
        "card")
    assert M.parameters({}, vs | {"ledger": {"inputs": {"passes": [12] * 96}}})["passes"] == 1152
    assert M.parameters({}, vs | {"ledger": {"inputs": {"passes": True}}})["passes"] is None


def test_local_states_the_arena_its_engine_recorded_and_not_its_own_arguments(store, capsys, work):
    """The arena and vLLM's share on the line are the engine's record of them (`out/engine.0.json`), not what
    `--local` handed `run.sh`; an engine that left no record leaves them unstated."""
    assert bench(store, "--local", run=Machine()) == 0
    _, doc, _ = printed(capsys)
    assert (doc["machine"]["arena_gib"], doc["machine"]["util"]) == (ARENA, 0.88)
    assert bench(store, "--local", run=Machine(engine=False)) == 0
    _, doc, _ = printed(capsys)
    assert (doc["machine"]["arena_gib"], doc["machine"]["util"]) == (None, None)
    assert doc["ran_with"]["detail"].startswith("an arena its engine's record doesn't state, the store ")


@pytest.mark.parametrize("machine, why, code", [
    (Machine(driver=False), "--local serves on an NVIDIA GPU, and there is no nvidia-smi here, so no NVIDIA driver", 3),
    (Machine(smi_rc=9), "--local serves on an NVIDIA GPU, and nvidia-smi names none here (rc 9: NVIDIA-SMI has failed)",
     3),
    (Machine(card="NVIDIA H100 80GB HBM3, 9.0"), "the tree has no data sheet for NVIDIA H100 80GB HBM3 (hardware/nodes/ "
     "has h200-sxm-141gb-x1, rtx-pro-6000-server-x8 for its cards), and the share is of the card's own sheet", 3),
    (Machine(fetch_rc=1), "unsloth/Llama-3.2-1B@9535bd9b1d1d (config.json, generation_config.json, model.safetensors, "
     "special_tokens_map.json, tokenizer.json, tokenizer_config.json) couldn't be fetched into", 3),
    (Machine(rc=6), "the node's session failed (run.sh exited 6)", 3),
    (Machine(record=a_record(a_session(accepted=False, failure="window 2: a receipt doesn't open"))),
     "session 0 was not accepted (window 2: a receipt doesn't open)", 1),
], ids=["no-driver", "no-gpu", "no-sheet", "checkpoint", "run-sh", "rejected"])
def test_a_local_session_that_cannot_run_exits_3_and_one_not_accepted_exits_1(store, capsys, work, machine, why, code):
    """Both print no number: an accepted session prints its number and exits 0, one the verifier refused exits 1, and
    one that never reached a verdict (this machine lacks what it needs, or a process failed) exits 3."""
    assert bench(store, "--local", run=machine) == code
    text, doc, _ = printed(capsys)
    assert text[0].startswith(f"memory: no number: {why}") and doc["value"] is None and doc["local"] == M.LOCAL


def test_local_without_vllm_py_runs_nothing_and_says_how_to_make_one(store, capsys, work, monkeypatch):
    monkeypatch.delenv("VLLM_PY")
    machine = Machine()
    assert bench(store, "--local", run=machine) == 3
    assert [a[0] for a, _ in machine.started] == ["git"] * len(machine.started)
    text, _, _ = printed(capsys)
    assert text[0].startswith("memory: no number: --local serves with vLLM, and VLLM_PY names no vLLM interpreter")


def test_a_cheating_plan_reaches_run_sh_and_its_refusal_prints_no_number(store, capsys, work, monkeypatch):
    """PLAN reaches `run.sh` under both one-machine ways in, and whatever session it makes, the verifier's refusal is
    the command's, with no number and exit 1."""
    monkeypatch.setenv("PLAN", "skip:0.0625")
    rejected = a_record(a_session(accepted=False, failure="window 0: 4 of 16 openings missing"))
    for way in ("--loopback", "--local"):
        machine = Machine(record=rejected)
        assert bench(store, way, run=machine) == 1
        ((_, env),) = machine.runs()
        assert env["PLAN"] == "skip:0.0625"
        assert printed(capsys)[0][0] == "memory: no number: session 0 was not accepted (window 0: 4 of 16 openings missing)"


def test_a_refused_local_session_writes_its_row_with_no_value_and_why(store, capsys, work, tmp_path, monkeypatch):
    """The Metrics tab's row of a `--local` session that isn't accepted holds no value and the refusal; a loopback
    writes none."""
    monkeypatch.setenv("OUT", str(tmp_path / "mine"))
    rejected = a_record(a_session(accepted=False, failure="late"))
    assert bench(store, "--local", run=Machine(record=rejected)) == 1
    (m,) = json.loads((tmp_path / "mine" / "result.json").read_text())["measurements"]
    assert (m["name"], m["value"], m["why"], m["target"]) == (M.LOCAL_METRIC, None, "session 0 was not accepted (late)",
                                                              M.TARGET)


@pytest.mark.parametrize("way", ["--loopback", "--local"])
def test_where_the_lean_leaves_gamma_undecided_the_one_machine_ways_print_the_python_ledgers_and_say_so_first(
        store, capsys, work, monkeypatch, way):
    """A fresh session prints a number: where the Lean's answer for γ's terms isn't written, `--loopback` and `--local`
    print the Python ledger's γ and error, which the verifier's record carries beside the undecided reading, and the
    text's first lines and the JSON's `gamma_decided_by` say the Lean verifier has not decided them."""
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", COMMIT)
    machine = Machine(record=a_local_record(a_python_ledger_session(gamma=0.199, error=2.0 ** -40)))
    assert bench(store, way, run=machine) == 0
    text, doc, _ = printed(capsys)
    assert text[0].startswith("memory: 0.199 of the node's memory read rate") and "δ = 9.09e-13" in text[0]
    assert text[1] == f"  {M.LOOPBACK if way == '--loopback' else M.LOCAL}"
    assert text[2] == f"  {M.PYTHON_LEDGER_WORDS}" and "has not decided" in text[2]
    assert (doc["value"], doc["error"], doc["gamma_decided_by"]) == (0.199, 2.0 ** -40, M.PYTHON_LEDGER)


def test_no_audit_reads_the_python_ledgers_gamma(store, capsys, tmp_path):
    """The node run and `--from` read a recorded run, which is an audit's: a γ the Lean left undecided prints no number
    there, though the record carries the Python ledger's, and the session harness reads none from it either. A γ the
    Lean decided is `gamma_decided_by` `lean`."""
    record = a_record(a_python_ledger_session())
    store.add(RUN, {"out/verifier.json": record})
    assert bench(store, "--from", RUN) == 1
    out, err = capsys.readouterr()
    assert out == "" and f"session 0 was accepted and its γ at the specification's rate is {UNDECIDED}" in err
    (tmp_path / "verifier.json").write_text(json.dumps(record))
    with pytest.raises(M.NoNumber):
        M.session(tmp_path / "verifier.json", None)
    (tmp_path / "prover.json").write_text(json.dumps({"sessions": [{"mode": "honest", "opened": [{"bytes": 10}]}]}))
    (row,) = summary.summarize(tmp_path)["honest"]
    assert row["gamma_spec"] is None
    store.add(FIRE, {"out/verifier.json": a_record(a_session(gamma=0.125))})
    assert bench(store, "--from", FIRE) == 0
    _, doc, _ = printed(capsys)
    assert doc["gamma_decided_by"] == "lean"


def test_from_a_records_directory_reads_it_with_nothing_of_ours(store, capsys, tmp_path, work):
    """`--from DIR` reads a run's record as `research run` leaves it (`job.json` and `out/`), opening no store: the run
    and commit `job.json` names, and what this command's own document there says (a `--local` session's machine).
    Without `job.json`, the document's run and commit. A share the Python ledger computed prints no number there, as
    under every recorded run's reading."""
    def no_store():
        raise AssertionError("--from DIR opens no store")

    tree = tmp_path / "rec"
    (tree / "out").mkdir(parents=True)
    (tree / "out" / "verifier.json").write_text(json.dumps(as_local(a_record(a_session(gamma=0.125)))))
    (tree / "job.json").write_text(json.dumps({"run": {"id": BENCH}, "computation": {"tree_sha": COMMIT}}))
    machine = {"card": "NVIDIA H200", "sheet": "h200-sxm-141gb-x1", "sheet_sha256": "a", "verifier_sha256": "b"}
    (tree / "out" / "memory.json").write_text(json.dumps({"run": BENCH, "loopback": False, "local": M.LOCAL,
                                                          "machine": machine}))
    assert M.main(["--from", str(tree)], store_of=no_store) == 0
    text, doc, _ = printed(capsys)
    assert (doc["value"], doc["run"], doc["commit"], doc["local"], doc["machine"]) == (0.125, BENCH, COMMIT, M.LOCAL,
                                                                                      machine)
    assert doc["ran_with"]["coins"] == "local challenger" and text[1] == f"  {M.LOCAL}"
    (tree / "job.json").unlink()
    (tree / "out" / "memory.json").write_text(json.dumps({"run": None, "commit": "e" * 40, "loopback": True}))
    assert M.main(["--from", str(tree)], store_of=no_store) == 0
    _, doc, _ = printed(capsys)
    assert (doc["run"], doc["commit"], doc["loopback"]) == (None, "e" * 40, True)
    (tree / "out" / "verifier.json").write_text(json.dumps(as_local(a_record(a_python_ledger_session()))))
    assert M.main(["--from", str(tree)], store_of=no_store) == 1
    out, err = capsys.readouterr()
    assert out == "" and "its γ at the specification's rate is undecided" in err
    (tree / "out" / "verifier.json").unlink()
    assert M.main(["--from", str(tree)], store_of=no_store) == 3
    assert f"{tree} holds no verifier's record" in capsys.readouterr().err


def test_the_verifiers_node_is_the_one_card_of_the_tree_sheets():
    """`--local`'s share is of the card it ran on and its host: the verifier's node is one card of the node sheet,
    read from the tree's data sheets, not the eight-card node `verifier_process.py` defaults to."""
    from verity.protocols.accounting.poub.verifier import ledger, timing
    eight = ledger.Node.stated(timing.RTX_PRO_6000_NODE)
    rtx, rtx_hbm = M.one_card("rtx-pro-6000-server-x8")
    assert rtx["spec_bytes_per_s"] == pytest.approx(eight.spec_bytes_per_s - 7 * timing.GPU_SPEC_BYTES_PER_S)
    assert rtx_hbm == GPU_HBM["rtx-pro-6000-server-x8"]
    h200, h200_hbm = M.one_card("h200-sxm-141gb-x1")
    assert h200["spec_bytes_per_s"] == pytest.approx(4.916736e12 + 6.144e11), "the H200's and its host's, at the sheets"
    assert h200_hbm == GPU_HBM["h200-sxm-141gb-x1"]


def test_a_digest_changes_with_any_file_under_it(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    (tmp_path / "v" / "lean").mkdir(parents=True)
    (tmp_path / "v" / "a.py").write_text("a")
    first = M.digest(tmp_path / "v")
    (tmp_path / "v" / "__pycache__").mkdir()
    (tmp_path / "v" / "__pycache__" / "a.pyc").write_text("cache")
    (tmp_path / "v" / "lean" / ".lake").mkdir()
    (tmp_path / "v" / "lean" / ".lake" / "build").write_text("build")
    assert M.digest(tmp_path / "v") == first, "caches and build trees are left out"
    (tmp_path / "v" / "lean" / "B.lean").write_text("b")
    assert M.digest(tmp_path / "v") != first
