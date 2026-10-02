"""A unit class of any partition as a ``verity/flock-circuit`` statement (`verity_flock.class_statement`): on circuit-check's tiny
Serve_v1 under `Q_word`, two table-free classes and the embedding gather compose with one row port per input source, their
laid-out unit rows agree with the reference on synthetic lanes, a source read by several instances is one shared row (the
gather's table, broadcast across its Call's members), and, when ``FLOCK_CIRCUIT`` names a built ``flock-circuit`` binary, a
loopback session accepts each."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("verity.ir.units")
pytest.importorskip("verity_vllm.program.registry")
from circuit_check import targets as TG  # noqa: E402
from verity.ir.defs import bind  # noqa: E402
from verity.ir.program import Program  # noqa: E402

from verity_flock import circuit as FC  # noqa: E402
from verity_flock import class_statement as CS  # noqa: E402
from verity_flock import partition_units as PU  # noqa: E402


@pytest.fixture(scope="module")
def units():
    TG.load_registries()
    from verity_vllm.program.registry import b1

    CS.cache_slot_circuits()
    return PU.partition_units(Program(bind(b1.Serve, C=TG.TINY, LP=3, STEPS=2)), "q-word")


@pytest.mark.slow
def test_table_free_classes_compose_and_prove(units, tmp_path: Path):
    key = PU.derive(PU.SOURCE, PU.DOMAIN, {"program": "test", "seed": 0})
    shared = CS.broadcast_params(units)
    assert shared["GatherBf16x64_v1"] == ({0}, 32)                         # the table, read by all 32 row members of the Call
    picked, gather = [], None
    for s, cl in sorted(units.classes().items(), key=lambda kv: len(kv[1].representative.gates)):
        cc = PU.lower_class(units.activation(cl.definition), cl.representative)
        if cc.circuit is None or cc.circuit.lookups:
            continue
        if cl.definition.id == "GatherBf16x64_v1":
            gather = gather or (s, cl)
        elif len(picked) < 2:
            picked.append((s, cl))
    binary = os.environ.get("FLOCK_CIRCUIT")
    for i, (s, cl) in enumerate([*picked, gather]):
        lanes = CS.class_lanes(units, cl, key, 8, shared)
        d = tmp_path / s[:16]
        st = CS.stage(units, "q-word", s, cl, lanes, d, hashlib.sha256(b"test").digest())
        text = (d / "circuit.txt").read_text()
        assert st["n"] == 8 and st["k_log"] <= FC.K_MAX and st["circuit_sha512"] == FC.digest(text)
        assert '"rule":"verity/partition/v1"' in text                      # the Q_word partition's own digest, not a provisional cover
        assert len(st["ports"]) == len(st["shared_rows"]) and all(w % CS.BLOCK_WORDS == 0 for w in st["ports"])
        if (s, cl) == gather:
            assert st["shared_rows"] == {"p0": 1, "p1": 8}                  # one table row shared by every instance; a row per index
        if binary and Path(binary).exists():
            live = CS.prove(binary, d, st["n"], 7690 + i)
            assert live["accepted"] is True and live["proof_bytes"]


def test_the_slot_circuits_are_built_only_when_a_shape_is_not_in_the_stage_cache(monkeypatch, tmp_path: Path):
    pkg, src = Path(CS.__file__).parent, tmp_path / "src"          # verity_flock's sources, as the stage key's code digest reads them
    for f in pkg.rglob("*.py"):
        (src / f.relative_to(pkg)).parent.mkdir(parents=True, exist_ok=True)
        (src / f.relative_to(pkg)).write_bytes(f.read_bytes())
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: SimpleNamespace(submodule_search_locations=[str(src)])
                        if name == "verity_flock" else find_spec(name, *a))

    def fresh(s):
        (CS._CTX["out"] / s[:16]).mkdir(parents=True, exist_ok=True)
        (CS._CTX["out"] / s[:16] / "circuit.txt").write_text(s)
        return {"shape": s, "stage": {"n": 16}}

    built = []
    monkeypatch.setenv("FLOCK_STAGE_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setitem(CS._CTX, "source_digest", None)
    monkeypatch.setattr(CS, "cache_slot_circuits", lambda: built.append(1))
    monkeypatch.setattr(CS, "_stage_job_fresh", fresh)

    def cached(name):
        CS._CTX.pop("source_digest")                                   # main() reads the code's digest once per process
        assert CS.main(["--model", "tiny", "--partition", "q-word", "--stage-only", "--sample", "2", "--out", str(tmp_path / name)]) == 0
        recs = [json.loads(ln) for ln in (tmp_path / name / "staged.jsonl").read_text().splitlines()]
        assert len(recs) == 2 and len({"stage_cached" in r for r in recs}) == 1
        return "stage_cached" in recs[0]

    assert not cached("first") and built == [1]
    assert cached("again") and built == [1]                            # a cache hit composes nothing
    sha = src / "sha512_circuit.py"
    assert "CARRY_EVERY = 5" in sha.read_text()
    sha.write_text(sha.read_text().replace("CARRY_EVERY = 5", "CARRY_EVERY = 4"))
    assert not cached("sha512-parameter") and built == [1, 1]          # a SHA-512 parameter is in the key
    (src / "class_statement.py").write_text((src / "class_statement.py").read_text() + "\n# another version\n")
    assert not cached("code") and built == [1, 1, 1]                   # and so is the code's version


def test_selftest_jobs_runs_the_case_processes_at_once_and_keeps_their_records_in_case_order(monkeypatch, tmp_path: Path):
    cases = ["gpu_paths_agree", "gpu_proofs_match_cpu", "verify_ahead_matches_serial"]
    mib = dict(zip(cases, (4000, 4500, 4700)))
    st: dict = {"lock": threading.Lock()}

    def run(cmd, **_kw):
        case = cmd[cmd.index("--only") + 1]
        on = "caller" if threading.current_thread() is st["caller"] else "pool"
        with st["lock"]:
            st["argv"][case] = cmd
            st["running"].add(case)
        if on == "pool":
            st["started"].wait()                                   # every case's process is running before any returns,
            st["sampled"].wait()                                   # with its device memory sampled meanwhile,
            if case != cases[-1]:
                st["returned"][cases[cases.index(case) + 1]].wait()   # and they return last case first
        with st["lock"]:
            st["running"].discard(case)
            st["finished"].append(case)
        st["returned"][case].set()
        return CS.subprocess.CompletedProcess(cmd, 0, stdout=f"NEG\t{json.dumps({'case': case, 'pass': True, 'on': on})}\n", stderr="")

    def device():
        with st["lock"]:
            now = {c: mib[c] for c in st["running"]}
        if len(now) == len(cases):
            st["sampled"].set()
        return now

    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setattr(CS.subprocess, "run", run)
    monkeypatch.setattr(CS, "_case_device_mib", device)
    monkeypatch.setattr(CS, "SELFTEST_SAMPLE_S", 0.001)
    monkeypatch.setattr(CS, "cache_slot_circuits", lambda: None)
    monkeypatch.setattr(CS, "_stage_job", lambda s: {"shape": s, "stage": {"n": 16}, "table_reads": {}})
    args = ["--model", "tiny", "--partition", "q-word", "--sample", "1", "--selftest-only", "--selftest", "1", "--selftest-gpu",
            "--selftest-cases", ",".join(cases), "--selftest-binary", "flock-circuit-seeded", "--selftest-case-mib", "6144"]
    argvs, sel = [], {}
    # jobs, the GPU's free MiB, where the cases ran, the order they returned in
    for jobs, free, on, order in (("1", 96000, "caller", cases), ("3", 96000, "pool", cases[::-1]),
                                  ("3", 3 * 6144 - 1, "caller", cases), ("3", None, "caller", cases)):
        monkeypatch.setattr(CS, "_gpu_free_mib", lambda free=free: free)
        st.update(argv={}, running=set(), finished=[], started=threading.Barrier(len(cases)), sampled=threading.Event(),
                  returned={c: threading.Event() for c in cases}, caller=threading.current_thread())
        (tmp_path / "selftests.jsonl").unlink(missing_ok=True)
        assert CS.main([*args, "--selftest-jobs", jobs, "--out", str(tmp_path)]) == 0
        (rec,) = [json.loads(ln) for ln in (tmp_path / "selftests.jsonl").read_text().splitlines()]
        s = sel[jobs, free] = rec["selftest"]
        assert st["finished"] == order
        assert s["all_pass"] is True and s["subset"] == cases and s["did_not_finish"] == []
        assert s["gpu_cases"] == [{"case": c, "pass": True, "on": on} for c in cases]
        assert [(p["case"], p["exit_code"], p["timed_out"], type(p["wall_s"])) for p in s["per_case"]] == [(c, 0, False, float) for c in cases]
        argvs.append(st["argv"])
    assert all(a == argvs[0] for a in argvs)                       # each case its own process, with the same binary and arguments
    assert all(cmd[0] == "flock-circuit-seeded" and "--gpu" in cmd and cmd[-2:] == ["--only", c] for c, cmd in argvs[0].items())
    assert "run" not in sel["1", 96000]
    at_once = sel["3", 96000]
    assert {k: at_once["run"][k] for k in ("jobs_asked", "jobs", "free_mib", "case_bound_mib", "sum_peak_mib")} == \
        {"jobs_asked": 3, "jobs": 3, "free_mib": 96000, "case_bound_mib": 6144, "sum_peak_mib": 13200}
    assert "fallback" not in at_once["run"] and [p["device_peak_mib"] for p in at_once["per_case"]] == [4000, 4500, 4700]
    assert sel["3", 18431]["run"]["jobs"] == 1 and sel["3", 18431]["run"]["fallback"] == "3 x 6144 MiB is more than the 18431 MiB free"
    assert sel["3", None]["run"]["jobs"] == 1 and sel["3", None]["run"]["fallback"] == "the GPU's free memory can't be read"


def test_a_case_process_that_ends_without_its_verdict_is_recorded_as_not_finished(monkeypatch, tmp_path: Path):
    def run(cmd, **_kw):
        if cmd[-1] == "gpu_proofs_match_cpu":
            raise CS.subprocess.TimeoutExpired(cmd, 3600)
        return CS.subprocess.CompletedProcess(cmd, -9, stdout="", stderr="CUDA_ERROR_OUT_OF_MEMORY: out of memory\n")

    monkeypatch.setattr(CS.subprocess, "run", run)
    hung = CS.prove("flock-circuit", tmp_path, 4, 0, selftest="gpu_proofs_match_cpu", gpu=True)
    assert (hung["all_pass"], hung["finished"], hung["timed_out"], hung["exit_code"]) == (False, False, True, None)
    oom = CS.prove("flock-circuit", tmp_path, 4, 0, selftest="gpu_paths_agree", gpu=True)
    assert (oom["all_pass"], oom["finished"], oom["timed_out"], oom["exit_code"]) == (False, False, False, -9)
    assert "out of memory" in oom["stderr_tail"] and "gpu_cases" not in oom


def test_selftest_gpu_runs_the_selftest_on_the_gpu(monkeypatch, tmp_path: Path):
    seen = []

    def run(cmd, **_kw):
        seen.append(cmd)
        return CS.subprocess.CompletedProcess(cmd, 0, stdout='NEG\t{"case": "gpu_paths_agree", "pass": true}\n', stderr="")

    monkeypatch.setattr(CS.subprocess, "run", run)
    gpu = CS.prove("flock-circuit", tmp_path, 4, 0, selftest="gpu_paths_agree", gpu=True)
    assert gpu["all_pass"] is True and gpu["gpu_cases"] == [{"case": "gpu_paths_agree", "pass": True}]
    assert CS.prove("flock-circuit", tmp_path, 4, 0, selftest="honest")["all_pass"] is True
    assert "--gpu" in seen[0] and seen[0][-2:] == ["--only", "gpu_paths_agree"]
    assert "--gpu" not in seen[1]


def test_a_verifier_pod_serves_every_point_of_a_gpu_session_on_the_same_processes(monkeypatch, tmp_path: Path):
    """Each prover process sends its session k to verifier process k mod S from k = 0, so a pod bounded for P points serves
    P provers one after another: each process exits after exactly the sessions the P provers send it."""
    argvs = []

    def popen(argv, stdout, **_kw):
        stdout.write(f"SERVING 0.0.0.0:{9000 + len(argvs)}\n")
        stdout.flush()
        argvs.append(argv)
        return SimpleNamespace(pid=0, kill=lambda: None, wait=lambda: None)

    monkeypatch.setattr(CS.subprocess, "Popen", popen)
    monkeypatch.setenv("FC_VERIFY_SERVERS", "11")
    monkeypatch.setenv("FC_VERIFY_AHEAD", "10")
    monkeypatch.setenv("FC_SERVE_HOST", "10.0.0.7")
    (tmp_path / "circuit.txt").write_text("circuit\n")
    sessions, points = 25, 8
    e = CS.serve_remote("flock-circuit", tmp_path, 2048, tmp_path / "verifiers", sessions, points=points)
    bound = [int(a[a.index("--sessions") + 1]) for a in argvs]
    sent = [sum(1 for _ in range(points) for k in range(sessions) if k % len(argvs) == i) for i in range(len(argvs))]
    assert len(argvs) == 11 and bound == sent and sum(bound) == points * sessions
    doc = json.loads((tmp_path / "verifiers" / CS._served_name(FC.digest("circuit\n"), 2048, sessions)).read_text())
    assert doc == e["doc"] and (doc["sessions"], doc["points"], doc["servers"]) == (sessions, points, 11)
    assert doc["addrs"] == [f"10.0.0.7:{9000 + i}" for i in range(11)]


def test_a_live_statement_is_accepted_only_if_every_session_is_and_prove_exits_0(monkeypatch, tmp_path: Path):
    (tmp_path / "circuit.txt").write_text("circuit")
    st: dict = {}

    class Prove:
        def __init__(self, cmd, stdout, stderr):
            if "-o" in cmd:
                Path(cmd[cmd.index("-o") + 1]).write_text("1024\n")
            stdout.write("".join(f"LIVE\t{json.dumps({'run': i, 'warm': i == 0, 'accepted': v, 'prove_total_s': 1.0 + i})}\n"
                                 for i, v in enumerate(st["verdicts"])))
            self.returncode = st["rc"]

        def wait(self):
            return self.returncode

    monkeypatch.delenv("FC_VERIFIERS", raising=False)
    monkeypatch.setattr(CS, "_serve", lambda *_a, **_kw: ([], ["127.0.0.1:1"]))
    monkeypatch.setattr(CS.subprocess, "Popen", Prove)
    cases = [([True] * 4, 0, True, None), ([True] * 3, 0, False, "3 LIVE records, not warm + runs = 4"),
             ([False, True, True, True], 0, False, "sessions [0] not accepted"), ([True, True, False, True], 0, False, "sessions [2] not accepted"),
             ([True] * 4, 1, False, "prove exited 1")]
    for verdicts, rc, accepted, why in cases:
        st.update(verdicts=verdicts, rc=rc)
        live = CS.prove("flock-circuit", tmp_path, 4, 0, warm=1, runs=3)
        assert (live["accepted"], live["why"]) == (accepted, why), (verdicts, rc)
        assert live["timed_runs"] == len(verdicts) - 1


def test_serve_wait_accepts_only_every_session_served_and_accepted(tmp_path: Path):
    class Exited:
        pid = 0

        def poll(self):
            return 0

        def kill(self):
            pass

        def wait(self):
            return 0

    def served(name: str, verdicts: list[bool]) -> dict:
        d = tmp_path / name
        (d / "sessions").mkdir(parents=True)
        (d / "sessions" / "index.jsonl").write_text("".join(json.dumps({"accepted": v, "verify_s": 0.5}) + "\n" for v in verdicts))
        (d / "addrs.json").write_text("{}")
        return {"file": d / "addrs.json", "srvs": [Exited()], "doc": {"sessions": 3}, "d": d, "t0": 0.0}

    recs = CS.serve_wait([served("all", [True] * 3), served("short", [True] * 2), served("rejected", [True, False, True]),
                          served("none", [])], 1.0)
    assert [(r["served"], r["accepted"]) for r in recs] == [(3, True), (2, False), (3, False), (0, False)]
    assert not any((tmp_path / n / "addrs.json").exists() for n in ("all", "short", "rejected", "none"))
