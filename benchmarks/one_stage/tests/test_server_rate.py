"""`server_rate.py`'s rows on a stand-in proof server and device: a row with one refused session, a server process that
exited nonzero, or a prover whose log reports the device out of memory is void (no rate, its reasons in ``void``), and the
run prints a VOID line and exits 1; a clean row keeps its rate and the run exits 0. Each session's record directory is
filed one second after the last (its mtime), so a clean row's rate is exactly one statement a second. A row also says what
prices it: the devices its provers held, the proved statement's proof blocks beside the most its circuit holds, the
verifier's mean seconds over the window, and whether the provers' statement caches had stopped filling (``flat``, with the
quarters' hit rates and the rule), which a prover still building statements in its last quarter fails without voiding the
row; and the CPUs its processes could run on (``cores``: this process's affinity as a count, a list and the NUMA nodes
that hold them, on a made-up node directory). A templated staging is proved with its unit placed, and verified under its
own statement. The Lean verdict is on the row's Program when one is given, and the row keeps what the verdict says of it.
A row whose sessions ran another statement version than it asked for (a prover that ignores ``--wide-leaf``, or one that
runs the wide leaf unasked) or name none is void."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import server_rate as SR  # noqa: E402  (a script beside the suite, importable only once its dir is on the path)

FILED0 = 1_800_000_000  # the first session's filing time, UTC seconds


class Session:
    def __init__(self, label: str, prover: str, d: Path, accepted: bool, cache: dict, verify_s: float, profile: str) -> None:
        self.label, self.prover, self.dir, self.accepted = label, prover, d, accepted
        self.live = ({"accepted": True, "e2e_s": 2.0, "prove_total_s": 1.5, "stmt_cache": cache} if accepted
                     else {"accepted": False, "why": "R7: no proof"}) | {"identity": {"profile": profile} if profile else {}}
        self.served = {"accepted": accepted, "verify_s": verify_s}

    def result(self) -> None:
        return None


class Server:
    """K provers that file session i at FILED0 + i, refuse the sessions in ``refuse``, write ``log`` to the first prover's
    log and close with ``exits``. A prover builds a statement at its sessions in ``built`` and finds it in its cache at
    the others; session i's verifier takes i / 8 seconds. Every session's identity names ``profile``: by default the
    version the server was made with (``wide_leaf``), as a prover that knows ``--wide-leaf`` does; "" names none."""

    def __init__(self, out: Path, provers: int, refuse=(), exits=None, log: str = "", built=(0,), wide_leaf: bool = False,
                 profile: str | None = None) -> None:
        self.out, self.k, self.refuse, self.built = out, provers, set(refuse), set(built)
        self.profile = SR.PROFILES[wide_leaf] if profile is None else profile
        self.exits = exits or {**{f"prover-gpu0-{j}": 0 for j in range(provers)}, "verifier-t": 0}
        (out / "prover-gpu0-0.log").write_text("LIVE\t{}\n" + log)

    @property
    def pids(self) -> dict[str, int]:
        return {}

    def submit(self, st, draw, label: str, verdict=None) -> Session:
        i = int(label.rsplit("/", 1)[1])
        d = self.out / "records" / label.replace("/", "-")
        d.mkdir(parents=True)
        os.utime(d, (FILED0 + i, FILED0 + i))
        mine = range(i % self.k, i + 1, self.k)
        cache = {"lookups": len(mine), "held": sum(1 for j in mine if j in self.built)}
        s = Session(label, f"prover-gpu0-{i % self.k}", d, i not in self.refuse, cache, i / 8, self.profile)
        if verdict is not None:
            s.verdict = lambda: verdict(s)
        return s

    def close(self) -> dict[str, int]:
        return dict(self.exits)

    #: where `ProofServer` placed the verifiers (``FC_VERIFIER_CPUS``); none unplaced
    verifier_cpus: str | None = None


class Device:
    def __init__(self, dev: str) -> None:
        self.samples, self.unparsed = [], []

    def stop(self) -> None:
        pass

    def busy(self, t0: float, t1: float) -> None:
        return None

    def peak_mib(self) -> None:
        return None


def stand_ins(monkeypatch, tmp_path: Path, tmpl_unit: Path | None = None, made: list[dict] | None = None, **server) -> None:
    """``made`` gets the keyword arguments each `ProofServer` was made with."""
    d = tmp_path / "staged"
    d.mkdir()
    (d / "stage.json").write_text(json.dumps({"n": 1, "k_log": 34}))
    monkeypatch.setattr(SR, "staged", lambda z, t, tu: (SimpleNamespace(pin="pin", tmpl_unit=tmpl_unit), 1, d))
    monkeypatch.setattr(SR, "Device", Device)

    def proof_server(binary, out, *, provers, **kw) -> Server:
        if made is not None:
            made.append(kw)
        return Server(Path(out), provers, wide_leaf=kw.get("wide_leaf", False), **server)

    monkeypatch.setattr(SR, "ProofServer", proof_server)


def run(monkeypatch, tmp_path: Path, *args: str, **server) -> tuple[int, dict]:
    stand_ins(monkeypatch, tmp_path, **server)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["server_rate.py", "--bin", "flock-circuit", "--staged", str(tmp_path), "--types", "t",
                                      "--provers", "1", "--out", str(out), "--device", "0", "--sessions", "8", "--warm", "1",
                                      "--ahead", "1", *args])
    rc = SR.main()
    [row] = json.loads((out / "rate.json").read_text())
    return rc, row


def test_a_clean_row_keeps_its_rate(monkeypatch, tmp_path):
    rc, row = run(monkeypatch, tmp_path)
    assert (rc, row["statements_per_s"], row["accepted"], "void" in row) == (0, 1.0, 9, False)


def test_a_row_keeps_its_verifier_cpus_and_their_seconds(monkeypatch, tmp_path):
    for d in ("shared", "placed"):
        (tmp_path / d).mkdir()
    _, row = run(monkeypatch, tmp_path / "shared")
    assert (row["verifier_cpus"], row["verifier_cpu_s_per_statement"], row["verifier_cpu_gone"]) == (None, 0.0, [])
    monkeypatch.setattr(Server, "verifier_cpus", "44-47")
    _, row = run(monkeypatch, tmp_path / "placed")
    assert row["verifier_cpus"] == "44-47"


def test_one_refused_session_voids_the_row_and_the_run(monkeypatch, tmp_path, capsys):
    rc, row = run(monkeypatch, tmp_path, refuse={4})
    assert (rc, row["statements_per_s"], row["accepted"], row["refused_n"]) == (1, None, 8, 1)
    assert row["void"] == ["1 of 9 sessions refused"]
    assert [r["label"] for r in row["refused"]] == ["t/k1/4"]
    assert "VOID t K=1: 1 of 9 sessions refused" in capsys.readouterr().out


def test_a_process_that_exited_nonzero_voids_the_row(monkeypatch, tmp_path):
    rc, row = run(monkeypatch, tmp_path, exits={"prover-gpu0-0": 0, "verifier-t": -15})
    assert (rc, row["statements_per_s"], row["void"]) == (1, None, ["exited nonzero: verifier-t rc -15"])


def test_a_prover_out_of_device_memory_voids_the_row(monkeypatch, tmp_path):
    log = "FFI CUDA error out of memory at prove_circuit.cuh:651\n"
    rc, row = run(monkeypatch, tmp_path, log=log)
    assert (rc, row["statements_per_s"], row["void"]) == (1, None, ["out of device memory: prover-gpu0-0"])


def test_a_row_says_what_prices_it(monkeypatch, tmp_path):
    rc, row = run(monkeypatch, tmp_path)
    assert (rc, row["gpus"], row["stage"]["blocks"], row["stage"]["full"]) == (0, 1, 1, 2)
    # the window is the sessions filed after the first timed one up to the last but one: 2 .. 7, each i / 8 s
    assert row["verify_s"] == 0.5625
    quarter = {"lookups": 2, "held": 0, "hit_rate": 1.0}
    assert (row["flat"], row["cache"]) == (True, {"rule": SR.FLAT_RULE, "provers": {"prover-gpu0-0": {
        "warm": {"lookups": 1, "held": 1}, "timed": 8, "quarters": [quarter] * 4}}})


def test_a_prover_still_building_statements_is_not_flat_and_keeps_its_rate(monkeypatch, tmp_path):
    rc, row = run(monkeypatch, tmp_path, built=(0, 7, 8))
    assert (rc, row["statements_per_s"], row["flat"], "void" in row) == (0, 1.0, False, False)
    assert [q["hit_rate"] for q in row["cache"]["provers"]["prover-gpu0-0"]["quarters"]] == [1.0, 1.0, 1.0, 0.0]


def nodes(tmp_path: Path, **cpulists: str) -> Path:
    """A made-up node directory: each ``nodeN``'s ``cpulist``."""
    for name, cpus in cpulists.items():
        (tmp_path / "node" / name).mkdir(parents=True)
        (tmp_path / "node" / name / "cpulist").write_text(cpus + "\n")
    return tmp_path / "node"


def test_a_row_says_the_cpus_its_processes_could_run_on(monkeypatch, tmp_path):
    monkeypatch.setattr(SR, "NODES", nodes(tmp_path, node7="0-4095"))
    mine = os.sched_getaffinity(0)
    rc, row = run(monkeypatch, tmp_path)
    assert (rc, row["cores"]) == (0, {"count": len(mine), "list": SR.cpu_list(mine), "numa_nodes": [7]})


def test_cores_are_a_count_a_list_as_the_kernel_writes_one_and_the_numa_nodes_that_hold_them(tmp_path):
    d = nodes(tmp_path, node0="0-23", node1="24-47", node2="", node3="72-95", node10="96,98-99")
    assert SR.cores(range(72, 82), d) == {"count": 10, "list": "72-81", "numa_nodes": [3]}
    assert SR.cores({30, 20, 21, 22, 23, 24, 99}, d) == {"count": 7, "list": "20-24,30,99", "numa_nodes": [0, 1, 10]}
    assert SR.cores({97}, d) == {"count": 1, "list": "97", "numa_nodes": []}              # a CPU no node lists
    assert SR.cores({5, 7}, tmp_path / "none") == {"count": 2, "list": "5,7", "numa_nodes": []}   # a machine that lists no nodes


def counters(lookups, held=None):
    """A prover's timed sessions' statement-cache counters, from its cumulative lookups (and statements held, else 1)."""
    return [{"lookups": n, "held": 1 if held is None else h} for n, h in zip(lookups, held or lookups)]


def flat(*provers):
    """`cache_flat` over provers that each built their one statement in a warm session, then ran the given timed ones."""
    return SR.cache_flat([(f"p{j}", warm, c) for j, timed in enumerate(provers)
                          for warm, c in [(True, {"lookups": 1, "held": 1}), *((False, c) for c in timed)]])


def test_flat_is_every_provers_last_two_quarters_within_a_point():
    steady = counters(range(2, 10))
    assert flat(steady, steady)[0] is True
    assert flat(steady, counters(range(2, 10), held=[1, 1, 1, 1, 1, 1, 2, 3]))[0] is False
    # the third quarter built 1 statement in 100 lookups, the last none: 0.99 against 1.0 is within the point; 2 are not
    ok, series = flat(counters([101, 201, 301, 401], held=[1, 1, 2, 2]))
    assert (ok, [q["hit_rate"] for q in series["provers"]["p0"]["quarters"]]) == (True, [1.0, 1.0, 0.99, 1.0])
    assert flat(counters([101, 201, 301, 401], held=[1, 1, 3, 3]))[0] is False


def test_flat_needs_four_timed_sessions_the_counters_and_a_lookup_in_each_of_the_last_two_quarters():
    assert flat(counters(range(2, 5))) == (False, {"rule": SR.FLAT_RULE, "provers": {"p0": {
        "warm": {"lookups": 1, "held": 1}, "timed": 3, "quarters": []}}})
    assert flat([None] * 8)[0] is False
    assert SR.cache_flat([]) == (False, {"rule": SR.FLAT_RULE, "provers": {}})
    # a prover reports its last sessions together, after their lookups: the last quarter added none
    assert flat(counters([2, 3, 4, 4]))[0] is False
    assert flat(counters([2, 3, 4, 5, 6, 7, 9, 9]))[0] is True


def test_statement_blocks_are_its_instances_beside_the_most_its_circuit_holds():
    assert SR.statement_blocks({"n": 2048, "k_log": 24, "vus_per_block": 1}) == {"blocks": 2048, "full": 2048}
    assert SR.statement_blocks({"n": 256, "k_log": 27, "vus_per_block": 4}) == {"blocks": 256, "full": 1024}
    assert SR.statement_blocks({"n": 1}) == {"blocks": 1, "full": None}


def staging(tmp_path: Path, meta: dict, template: list[str] | None = None) -> Path:
    """``tmp_path/t/c4/stage`` holding a circuit file whose head is a header, META and (a templated one's) its template
    section, then a body line."""
    d = tmp_path / "t" / "c4" / "stage"
    d.mkdir(parents=True)
    section = [] if template is None else [f"{SR.FC.TMPL_SECTION} {len(template)}", *template]
    (d / "circuit.txt").write_text("\n".join(["FLOCK-CIRCUIT", "META " + json.dumps(meta), *section, "body"]) + "\n")
    (d / "stage.json").write_text(json.dumps({"n": 4}))
    (d / "pin.txt").write_text("pin\n")
    return d


def place_stand_in(tmp_path: Path) -> str:
    """`flock-tmpl-unit TMPL --layout L --out F`'s stand-in: F holds the layout and the template it was given."""
    b = tmp_path / "flock-tmpl-unit"
    b.write_text('#!/bin/sh\n{ echo "$3"; cat "$1"; } > "$5"\n')
    b.chmod(0o755)
    return str(b)


def test_a_templated_statement_is_proved_with_its_unit_placed(tmp_path):
    """A templated staging (META's statement `verity/flock-circuit/tmpl`) gets its unit placed from its circuit's template
    section and META's layout, which the proof server hands both sides; without `--tmpl-unit-bin` it is refused."""
    d = staging(tmp_path, {"statement": SR.FC.TMPL_STATEMENT, "layout": "iv/hybrid/4"}, ["row 0", "row 1"])
    st, n, got = SR.staged(tmp_path, "t", place_stand_in(tmp_path))
    assert (got, n, st.tmpl_unit) == (d, 4, d / "unit.placed")
    assert st.tmpl_unit.read_text() == "iv/hybrid/4\nrow 0\nrow 1\n"
    with pytest.raises(RuntimeError, match="--tmpl-unit-bin"):
        SR.staged(tmp_path, "t")


def test_a_flat_statement_has_no_unit(tmp_path):
    staging(tmp_path, {"statement": SR.STATEMENT_ID})
    st, _, d = SR.staged(tmp_path, "t")
    assert st.tmpl_unit is None and not (d / "unit.placed").exists()


@pytest.mark.parametrize("wide", [True, False])
def test_a_row_runs_one_statement_version_and_says_which(monkeypatch, tmp_path, wide):
    """``--wide-leaf`` reaches the proof server, whose provers and verifiers all carry it, and the row keeps it; without
    it neither does."""
    made: list[dict] = []
    rc, row = run(monkeypatch, tmp_path, *(["--wide-leaf"] if wide else []), made=made)
    assert (rc, row["wide_leaf"], [kw["wide_leaf"] for kw in made]) == (0, wide, [wide])
    asked = "fast100-wide-leaf" if wide else "fast100"
    assert (row["profile"], "void" in row) == ({"asked": asked, "ran": {asked: 9}}, False)


@pytest.mark.parametrize("wide", [True, False])
def test_a_row_whose_sessions_ran_the_other_statement_version_is_void(monkeypatch, tmp_path, capsys, wide):
    """A prover that ignores ``--wide-leaf`` proves fast100's statement and is accepted, and one that runs the wide
    leaf unasked is accepted too: either way every session's identity names the other version's profile, and the row
    has no rate."""
    asked, ran = ("fast100-wide-leaf", "fast100") if wide else ("fast100", "fast100-wide-leaf")
    rc, row = run(monkeypatch, tmp_path, *(["--wide-leaf"] if wide else []), profile=ran)
    assert (rc, row["statements_per_s"], row["accepted"], row["profile"]) == (1, None, 9, {"asked": asked, "ran": {ran: 9}})
    [why] = row["void"]
    assert why.startswith(f"9 of 9 finished sessions ran another statement version than the row's {asked!r}: 9 at {ran!r}")
    assert ("does not know --wide-leaf" in why) is wide
    assert f"VOID t K=1: {why}" in capsys.readouterr().out


def test_a_row_whose_sessions_name_no_profile_is_void(monkeypatch, tmp_path):
    rc, row = run(monkeypatch, tmp_path, profile="")
    assert (rc, row["statements_per_s"], row["profile"]["ran"]) == (1, None, {"none": 9})
    assert row["void"] == ["9 of 9 finished sessions ran another statement version than the row's 'fast100': 9 at 'none'"]


def test_version_reasons_count_each_other_profile():
    assert SR.version_reasons("fast100", {"fast100": 7}) == []
    assert SR.version_reasons("fast100", {}) == []
    [why] = SR.version_reasons("fast100-wide-leaf", {"fast100-wide-leaf": 4, "fast100": 2, "none": 1})
    assert why.startswith("3 of 7 finished sessions ran another statement version than the row's 'fast100-wide-leaf': "
                          "2 at 'fast100', 1 at 'none'; a flock-circuit that does not know --wide-leaf ignores it")


@pytest.mark.parametrize("tmpl", [False, True])
def test_a_wide_leaf_rows_lean_verdict_is_under_the_versions_statement(monkeypatch, tmp_path, tmpl):
    got = []

    class Lean:
        def __init__(self, lean, d, out, statement, program):
            got.append(statement)

        def close(self):
            pass

    monkeypatch.setattr(SR, "Lean", Lean)
    run(monkeypatch, tmp_path, "--lean", "flock-verify", "--wide-leaf", tmpl_unit=tmp_path / "unit.placed" if tmpl else None)
    assert got == [(SR.FC.TMPL_STATEMENT if tmpl else SR.STATEMENT_ID) + "+wide-leaf"]


def test_a_templated_statements_lean_verdict_is_under_its_statement(monkeypatch, tmp_path):
    got = []

    class Lean:
        def __init__(self, lean, d, out, statement, program):
            got.append(statement)

        def close(self):
            pass

    monkeypatch.setattr(SR, "Lean", Lean)
    run(monkeypatch, tmp_path, "--lean", "flock-verify", tmpl_unit=tmp_path / "unit.placed")
    assert got == [SR.FC.TMPL_STATEMENT]


def test_lean_is_handed_the_rows_program_and_the_row_keeps_what_its_verdict_says_of_it(monkeypatch, tmp_path):
    """`--lean-program`'s descriptor is archived with the circuit and the public file and handed to `verify` by its key
    (`--program`), and the row's Lean verdict keeps what it says of the Program: here, as on a statement under the
    provisional unit cover, its digest as handed and the note that it was not held to the circuit, and no
    `program_sha512`. Without `--lean-program`, `verify` is handed no program and the row keeps its refusal. `verify` and
    `archive-put` are stand-ins: `archive-put` names each file `key-<its name>`."""
    note = "the Program was handed and not held to the circuit: the provisional unit cover binds no Program descriptor"
    refusal = "refused: a session without the verifier's program (--program)"
    cmds = []

    class Child:
        pid = 0

        def __init__(self, cmd, **kw) -> None:
            cmds.append(cmd)
            v = ({"accepted": True, "why": "", "program_handed_sha512": "ab", "program_note": note} if "--program" in cmd
                 else {"accepted": False, "why": refusal})
            self.stdout = io.StringIO(f"VERDICT {json.dumps(v)}\n")

    def put(cmd, **kw):
        cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "".join(f"key-{Path(f).name} {f}\n" for f in cmd[3:]), "")

    monkeypatch.setattr(SR.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(SR.subprocess, "run", put)
    monkeypatch.setattr(SR.subprocess, "Popen", Child)
    monkeypatch.setattr(SR.os, "wait4", lambda pid, options: (pid, 0, SimpleNamespace(ru_utime=0.0, ru_stime=0.0, ru_maxrss=0)))
    for d in ("handed", "none"):
        (tmp_path / d).mkdir()
    program = tmp_path / "program.json"
    lean = ("--lean", "verifier", "--lean-sessions", "1")
    _, row = run(monkeypatch, tmp_path / "handed", *lean, "--lean-program", str(program))
    [v] = row["lean"]
    assert v["accepted"] and (v["program_handed_sha512"], v["program_note"]) == ("ab", note) and "program_sha512" not in v
    archived, verify = cmds
    assert archived[-1] == str(program) and verify[verify.index("--program") + 1] == "key-program.json"
    del cmds[:]
    _, row = run(monkeypatch, tmp_path / "none", *lean)
    [v] = row["lean"]
    assert (v["accepted"], v["why"]) == (False, refusal) and not set(SR.PROGRAM_FIELDS) & set(v)
    archived, verify = cmds
    assert str(program) not in archived and "--program" not in verify


@pytest.mark.parametrize("statement", [SR.STATEMENT_ID + SR.WIDE_LEAF, SR.STATEMENT_ID])
@pytest.mark.parametrize("refuses", [True, False])
def test_a_flock_verify_that_does_not_know_the_statement_is_said_as_that(tmp_path, statement, refuses):
    """A `flock-verify` that refuses the statement's name (exit 2, "unknown statement" on stderr, no VERDICT line) is
    recorded as a refusal of the name, and for a wide-leaf name as a `flock-verify` that does not list it; any other
    exit with no verdict keeps the plain reason."""
    lean = tmp_path / "flock-verify"
    end = "echo 'unknown statement; known: [verity/flock-circuit]' >&2; exit 2" if refuses else "exit 3"
    lean.write_text(f'#!/bin/sh\n[ "$1" = verify ] || exit 0\n{end}\n')
    lean.chmod(0o755)
    d, out = tmp_path / "stage", tmp_path / "out"
    for x in (d, out, tmp_path / "s"):
        x.mkdir()
    v = SR.Lean(str(lean), d, out, statement)
    got = v(SimpleNamespace(label="t/k1/0", dir=tmp_path / "s"))
    v.close()
    assert got["accepted"] is False
    if not refuses:
        assert got["why"] == "no verdict (rc 3)"
        return
    assert got["why"].startswith(f"no verdict (rc 2): this flock-verify does not know the statement {statement!r}")
    assert ("Flock/Tags.lean" in got["why"]) is statement.endswith(SR.WIDE_LEAF)
