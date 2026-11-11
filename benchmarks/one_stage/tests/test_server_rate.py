"""`server_rate.py`'s rows on a stand-in proof server and device: a row with one refused session, a server process that
exited nonzero, or a prover whose log reports the device out of memory is void (no rate, its reasons in ``void``), and the
run prints a VOID line and exits 1; a clean row keeps its rate and the run exits 0. Each session's record directory is
filed one second after the last (its mtime), so a clean row's rate is exactly one statement a second. A row also says what
prices it: the devices its provers held, the proved statement's proof blocks beside the most its circuit holds, the
verifier's mean seconds over the window, and whether the provers' statement caches had stopped filling (``flat``, with the
quarters' hit rates and the rule), which a prover still building statements in its last quarter fails without voiding the
row; and the CPUs its processes could run on (``cores``: this process's affinity as a count, a list and the NUMA nodes
that hold them, on a made-up node directory). A templated staging is proved with its unit placed, and verified under its
own statement."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import server_rate as SR  # noqa: E402  (a script beside the suite, importable only once its dir is on the path)

FILED0 = 1_800_000_000  # the first session's filing time, UTC seconds


class Session:
    def __init__(self, label: str, prover: str, d: Path, accepted: bool, cache: dict, verify_s: float) -> None:
        self.label, self.prover, self.dir, self.accepted = label, prover, d, accepted
        self.live = ({"accepted": True, "e2e_s": 2.0, "prove_total_s": 1.5, "stmt_cache": cache} if accepted
                     else {"accepted": False, "why": "R7: no proof"})
        self.served = {"accepted": accepted, "verify_s": verify_s}

    def result(self) -> None:
        return None


class Server:
    """K provers that file session i at FILED0 + i, refuse the sessions in ``refuse``, write ``log`` to the first prover's
    log and close with ``exits``. A prover builds a statement at its sessions in ``built`` and finds it in its cache at
    the others; session i's verifier takes i / 8 seconds."""

    def __init__(self, out: Path, provers: int, refuse=(), exits=None, log: str = "", built=(0,)) -> None:
        self.out, self.k, self.refuse, self.built = out, provers, set(refuse), set(built)
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
        return Session(label, f"prover-gpu0-{i % self.k}", d, i not in self.refuse, cache, i / 8)

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


def stand_ins(monkeypatch, tmp_path: Path, tmpl_unit: Path | None = None, **server) -> None:
    d = tmp_path / "staged"
    d.mkdir()
    (d / "stage.json").write_text(json.dumps({"n": 1, "k_log": 34}))
    monkeypatch.setattr(SR, "staged", lambda z, t, tu: (SimpleNamespace(pin="pin", tmpl_unit=tmpl_unit), 1, d))
    monkeypatch.setattr(SR, "Device", Device)
    monkeypatch.setattr(SR, "ProofServer", lambda binary, out, *, provers, **kw: Server(Path(out), provers, **server))


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


def test_a_templated_statements_lean_verdict_is_under_its_statement(monkeypatch, tmp_path):
    got = []

    class Lean:
        def __init__(self, lean, d, out, statement):
            got.append(statement)

        def close(self):
            pass

    monkeypatch.setattr(SR, "Lean", Lean)
    run(monkeypatch, tmp_path, "--lean", "flock-verify", tmpl_unit=tmp_path / "unit.placed")
    assert got == [SR.FC.TMPL_STATEMENT]
