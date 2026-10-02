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

import numpy as np
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


def _fp_class(dtype: str, K: int):
    import importlib.util

    spec = importlib.util.spec_from_file_location("gemm_fp", Path(__file__).parents[1] / "pod" / "gemm_fp.py")
    gemm_fp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_fp)
    CS.cache_slot_circuits()
    units = PU.partition_units(gemm_fp.program(dtype, K, 4), "q-word")
    ((s, cl),) = units.classes().items()
    act, uc = units.activation(cl.definition), cl.representative
    return units, s, cl, act, uc, CS.sources(act, uc, CS.owners_of(units, cl.definition.id))


@pytest.mark.parametrize("dtype", ["e4m3", "nvf4", "mxf4"])
def test_packed_rows_are_the_packed_tensor_bytes_and_the_unit_is_the_definition(dtype, tmp_path: Path):
    """FLOCK_PACK_WORDS's layout on an sm_120 FP coordinate: four FP4 or two FP8 elements a u16 leaf, so a row's little-endian
    bytes are its tensor's packed bytes (two FP4 a byte, low nibble first; one FP8 or scale a byte), every leaf bit before the
    block padding is read, the unit has the default layout's gates, and on circuit-check's vectors it equals the IR evaluator."""
    from circuit_check import checks as CK
    from circuit_check import vectors as V
    from verity.evaluation.reference import signature_of

    units, s, cl, act, uc, groups = _fp_class(dtype, 128)
    plain = CS.class_lowering(act, uc, "unit-class/plain", groups)
    low = CS.class_lowering(act, uc, "unit-class/packed", groups, packed=True)
    same = ("ands", "xors", "nots", "outputs", "lookups")
    assert {k: low.layout.counts.get(k) for k in same} == {k: plain.layout.counts.get(k) for k in same}
    assert low.layout.counts["rows"] < plain.layout.counts["rows"]
    widths = [[uc.in_widths[t] for t in g] for g in groups]
    assert all(sum(w) % 16 == 0 and CS._slots(w, True)[1] == sum(w) // 16 for w in widths)
    assert low.definition.port_words == tuple(CS._pad(sum(w) // 16, CS.BLOCK_WORDS) for w in widths)

    ins, want, _ids = CS.class_lanes(units, cl, PU.derive(PU.SOURCE, PU.DOMAIN, {"program": "test", "seed": 0}), 4, CS.broadcast_params(units))
    st = CS.stage(units, "q-word", s, cl, (ins, want), tmp_path, hashlib.sha256(b"test").digest(), packed=True)
    assert st["packed"] is True and st["ports"] == list(low.definition.port_words)
    _rows, _outs, per_port = CS.pack(uc, ins, want, low.definition.w_in, low.definition.w_out, groups, low.definition.port_words, True)
    for i, (g, w) in enumerate(zip(groups, widths)):
        for lane in range(4):
            x = [ins[t][lane] for t in g]
            packed = bytes(a | b << 4 for a, b in zip(x[::2], x[1::2])) if w[0] == 4 else bytes(x)
            row = per_port[f"p{i}"][lane].astype("<u2").tobytes()
            assert row == packed + bytes(len(row) - len(packed))

    fn = cl.definition
    params, _ret = signature_of(fn)
    arrays = V.vectors(params, 48, 20260926)
    ref, _raised = CK.reference_outputs(fn, params, arrays)
    leaves = V.flat(arrays)
    lanes = [leaves[:, -1 - k] for k in uc.inputs]
    got = {}
    for name, lw, pk in (("plain", plain, False), ("packed", low, True)):
        rows, _o, _p = CS.pack(uc, lanes, [[0] * len(arrays[0])], lw.definition.w_in, lw.definition.w_out, groups, lw.definition.port_words, pk)
        out, _cut, ok = lw.evaluate(rows.astype(np.uint64))
        got[name] = (out.astype(np.uint64), int(ok))
    assert np.array_equal(got["plain"][0], got["packed"][0]) and got["plain"][1] == got["packed"][1]
    out, ok = got["packed"]
    (w_out,) = uc.out_widths
    word = sum(out[:, h] << np.uint64(16 * h) for h in range(CS._words(w_out)))
    checked = [j for j, r in enumerate(ref) if r is not None and ok >> j & 1]
    assert len(checked) >= 24 and all(int(word[j]) == ref[j][0] for j in checked)


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
    monkeypatch.setattr(CS, "_stage_job", lambda s: {"shape": s, "stage": {"n": 16, "k_log": st["k_log"]}, "table_reads": {}})
    args = ["--model", "tiny", "--partition", "q-word", "--sample", "1", "--selftest-only", "--selftest", "1", "--selftest-gpu",
            "--selftest-cases", ",".join(cases), "--selftest-binary", "flock-circuit-seeded"]
    argvs = {}
    # at k_log 27 the GPU cases run one at a time on the caller, as with --selftest-jobs 1
    for jobs, k_log, on, order in (("1", 26, "caller", cases), ("3", 26, "pool", cases[::-1]), ("3", 27, "caller", cases)):
        st.update(argv={}, finished=[], started=threading.Barrier(len(cases)), returned={c: threading.Event() for c in cases},
                  caller=threading.current_thread(), k_log=k_log)
        (tmp_path / "selftests.jsonl").unlink(missing_ok=True)
        assert CS.main([*args, "--selftest-jobs", jobs, "--out", str(tmp_path)]) == 0
        (rec,) = [json.loads(ln) for ln in (tmp_path / "selftests.jsonl").read_text().splitlines()]
        assert st["finished"] == order
        assert rec["selftest"]["all_pass"] is True and rec["selftest"]["subset"] == cases
        assert rec["selftest"]["gpu_cases"] == [{"case": c, "pass": True, "on": on} for c in cases]
        argvs[jobs, k_log] = st["argv"]
    assert argvs["3", 26] == argvs["1", 26] == argvs["3", 27]      # each case its own process, with the same binary and arguments
    assert all(cmd[0] == "flock-circuit-seeded" and "--gpu" in cmd and cmd[-2:] == ["--only", c] for c, cmd in argvs["3", 26].items())


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


def test_each_selftest_case_keeps_its_stderr_beside_the_class_records(monkeypatch, tmp_path: Path):
    def run(cmd, **_kw):
        case = cmd[cmd.index("--only") + 1] if "--only" in cmd else None
        return CS.subprocess.CompletedProcess(cmd, 0, stdout=f'NEG\t{{"case": "{case}", "pass": false}}\n',
                                              stderr=f"thread 'main' panicked: {case}: out of memory\n")

    monkeypatch.setattr(CS.subprocess, "run", run)
    assert CS.prove("flock-circuit", tmp_path, 16, 0, selftest="gpu_proofs_match_cpu", gpu=True)["all_pass"] is False
    CS.prove("flock-circuit", tmp_path, 16, 0, selftest=True)
    assert (tmp_path / "selftest-gpu_proofs_match_cpu.err").read_text() == "thread 'main' panicked: gpu_proofs_match_cpu: out of memory\n"
    assert (tmp_path / "selftest-all.err").read_text() == "thread 'main' panicked: None: out of memory\n"
