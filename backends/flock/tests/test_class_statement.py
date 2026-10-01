"""A unit class of any partition as a ``verity/flock-circuit`` statement (`verity_flock.class_statement`): on circuit-check's tiny
Serve_v1 under `Q_word`, two table-free classes and the embedding gather compose with one row port per input source, their
laid-out unit rows agree with the reference on synthetic lanes, a source read by several instances is one shared row (the
gather's table, broadcast across its Call's members), and, when ``FLOCK_CIRCUIT`` names a built ``flock-circuit`` binary, a
loopback session accepts each."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

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
    def hit(s):
        d = tmp_path / "cache" / s[:16]
        d.mkdir(parents=True, exist_ok=True)
        (d / "rec.json").write_text(json.dumps({"shape": s}))
        return d

    built = []
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setattr(CS, "cache_slot_circuits", lambda: built.append(1))
    monkeypatch.setattr(CS, "_stage_job_fresh", lambda s: {"shape": s, "broke": {"at": "test"}})
    args = ["--model", "tiny", "--partition", "q-word", "--stage-only", "--sample", "2"]
    monkeypatch.setattr(CS, "_stage_cache", hit)
    assert CS.main([*args, "--out", str(tmp_path / "hit")]) == 0 and built == []
    assert all(json.loads(ln)["stage_cached"] for ln in (tmp_path / "hit" / "staged.jsonl").read_text().splitlines())
    monkeypatch.setattr(CS, "_stage_cache", lambda s: None)
    assert CS.main([*args, "--out", str(tmp_path / "miss")]) == 0 and built == [1]


def test_selftest_jobs_runs_the_case_processes_at_once_and_keeps_their_records_in_case_order(monkeypatch, tmp_path: Path):
    cases = ["gpu_paths_agree", "gpu_proofs_match_cpu", "verify_ahead_matches_serial"]
    st: dict = {}

    def run(cmd, **_kw):
        case = cmd[cmd.index("--only") + 1]
        st["argv"][case] = cmd
        on = "caller" if threading.current_thread() is st["caller"] else "pool"
        if on == "pool":
            st["started"].wait()                                   # every case's process is running before any returns
            if case != cases[-1]:
                st["returned"][cases[cases.index(case) + 1]].wait()   # and they return last case first
        st["finished"].append(case)
        st["returned"][case].set()
        return CS.subprocess.CompletedProcess(cmd, 0, stdout=f"NEG\t{json.dumps({'case': case, 'pass': True, 'on': on})}\n", stderr="")

    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setattr(CS.subprocess, "run", run)
    monkeypatch.setattr(CS, "cache_slot_circuits", lambda: None)
    monkeypatch.setattr(CS, "_stage_job", lambda s: {"shape": s, "stage": {"n": 16}, "table_reads": {}})
    args = ["--model", "tiny", "--partition", "q-word", "--sample", "1", "--selftest-only", "--selftest", "1", "--selftest-gpu",
            "--selftest-cases", ",".join(cases), "--selftest-binary", "flock-circuit-seeded"]
    argvs = {}
    for jobs, on, order in (("1", "caller", cases), ("3", "pool", cases[::-1])):
        st.update(argv={}, finished=[], started=threading.Barrier(len(cases)), returned={c: threading.Event() for c in cases},
                  caller=threading.current_thread())
        (tmp_path / "selftests.jsonl").unlink(missing_ok=True)
        assert CS.main([*args, "--selftest-jobs", jobs, "--out", str(tmp_path)]) == 0
        (rec,) = [json.loads(ln) for ln in (tmp_path / "selftests.jsonl").read_text().splitlines()]
        assert st["finished"] == order
        assert rec["selftest"]["all_pass"] is True and rec["selftest"]["subset"] == cases
        assert rec["selftest"]["gpu_cases"] == [{"case": c, "pass": True, "on": on} for c in cases]
        argvs[jobs] = st["argv"]
    assert argvs["3"] == argvs["1"]                                # each case its own process, with the same binary and arguments
    assert all(cmd[0] == "flock-circuit-seeded" and "--gpu" in cmd and cmd[-2:] == ["--only", c] for c, cmd in argvs["3"].items())


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
