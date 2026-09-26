"""The benchmark entry point (``bench.cell``): planning a cell into its backend's jobs, the reasons a cell cannot run, the protocol
check on two real Flock cells (``data/flock-cells``: art:6d1295ed and art:d1961ba4, the registered bench-result/v1 documents,
both independently replayed) and their mutations, and registration through ``verity_flock.register`` on the runs behind
art:1589ffe1 (``data/flock-runs``)."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from verity_numerical.bench import cell as C
from verity_numerical.bench import drivers

DATA = Path(__file__).parent / "data"
SLICE = DATA / "captured-101" / "slices" / "gemm-coordinate-ampere-bf16-k2048"
BF16 = ("gemm-coordinate/k1536/sm90-mma-bf16+frame-v3/blake3-keyed", "bench-instances-bf16-hopper-v1")
FLOCK = dict(verifier="vy-flock-verifier", verifier_addr="10.0.0.2:7400", rtt_target="10.0.0.2:22", campaign="c", lane="bench-spine")


def _doc(art: str) -> dict:
    return json.loads((DATA / "flock-cells" / f"{art}.json").read_text())


def _cell(statement=BF16[0], input_set=BF16[1], **kw) -> C.Cell:
    return C.plan("C-interactive", statement, input_set, "vy-flock-h100", **{**FLOCK, **kw})


def test_plan_wraps_the_flock_cell_script():
    cell = _cell()
    assert (cell.family, cell.relation, cell.subcircuit, cell.scheme) == ("C-Flock", "bf16-hopper", "gemm-coordinate/k1536/sm90-mma-bf16",
                                                                          "frame-v3/blake3-keyed")
    assert cell.input_set["kind"] == "census" and cell.input_set["dataset"] == "bench-instances-bf16-hopper/v1"
    v, p = cell.jobs
    assert (v["role"], v["pod"], p["role"], p["pod"]) == ("verifier", "vy-flock-verifier", "prover", "vy-flock-h100")
    for j in (v, p):
        argv = j["argv"]
        assert argv[:4] == ["research", "run", "--on", j["pod"]] and "--custody-r2" in argv and argv[argv.index("--cwd") + 1] == "source"
        cmd = argv[argv.index("--") + 1:]
        assert cmd[:2] == ["bash", "backends/flock/pod/30-cell.sh"] and "REL=bf16-hopper" in cmd and "RUNS=5" in cmd
        assert f"POINTS={C.DEFAULT_POINTS}" in cmd
    assert "ROLE=verifier" in v["command"]
    assert {"ROLE=prover", "VERIFIER=10.0.0.2:7400", "RTT_TARGET=10.0.0.2:22", "LOOPBACK=1"} <= set(p["command"])
    assert (Path(C.LW.REPO) / "backends/flock/pod/30-cell.sh").is_file()
    assert C.launch(cell, dry_run=True) == [{"role": "verifier", "argv": v["argv"]}, {"role": "prover", "argv": p["argv"]}]
    assert C.Cell(**json.loads(cell.to_json())) == cell


@pytest.mark.parametrize("kw,why", [
    (dict(statement="rope-head/d64/neox-bf16+frame-v3/blake3-keyed"), "no lowering of RoPEHead_v1"),
    (dict(statement="gemm-coordinate/k2048/sm90-mma-bf16+frame-v3/blake3-keyed"), "K = 2048"),
    (dict(statement="gemm-coordinate/k1536/sm90-mma-bf16+frame-v3/sha-256"), "keyed-BLAKE3"),
    (dict(input_set="bench-instances-fp8-ada-v1"), "is of gemm-coordinate/k1536/sm89-mma-e4m3"),
    (dict(verifier=None), "separate pod"),
    (dict(verifier="vy-flock-h100"), "other than the prover's"),
    (dict(rtt_target=None), "--rtt-target"),
    (dict(lane=None), "producing lane"),
])
def test_cells_that_cannot_run_say_why(kw, why):
    with pytest.raises(C.NotSupported, match=why):
        _cell(**kw)


def test_other_reasons():
    with pytest.raises(C.NotSupported, match="no driver module .*d_fs.py"):
        C.plan("D-fs", "gemm-coordinate/k1536/sm80-mma-bf16+frame-v3/sha-256", "bench-instances-v1", "vy-sp1")
    with pytest.raises(ValueError, match="<subcircuit id>\\+<scheme id>"):
        C.split_statement("gemm-coordinate/k1536/sm80-mma-bf16")
    with pytest.raises(KeyError):
        C.family_of("E-new")
    with pytest.raises(ValueError, match="not a census input set"):
        C.resolve_input_set("no-such-set")


def test_an_input_set_needs_the_lowerings_stage(monkeypatch):
    """A lowering without stage() runs only on the frozen set its pod scripts regenerate, never on an input set."""
    mod = C.LW.registry("C-Flock").module("gemm-coordinate")
    monkeypatch.setattr(mod, "supports", lambda s, c: None)
    monkeypatch.setattr(mod, "relation", lambda s, c: "bf16-ampere-k2048")
    with pytest.raises(C.NotSupported, match="no stage"):
        C.plan("C-interactive", "gemm-coordinate/k2048/sm80-mma-bf16+frame-v3/blake3-keyed", str(SLICE), "vy-p", **FLOCK)
    monkeypatch.setattr(mod, "stage", lambda s, lo, hi, out: out, raising=False)
    cell = C.plan("C-interactive", "gemm-coordinate/k2048/sm80-mma-bf16+frame-v3/blake3-keyed", str(SLICE), "vy-p", **FLOCK)
    assert cell.input_set["kind"] == "set" and cell.relation == "bf16-ampere-k2048"


def test_plan_wraps_the_b_ligero_cell_script(tmp_path, monkeypatch):
    """B-interactive on a captured #101 slice: the K = 2048 fold with the keyed-BLAKE3 xob circuit, the set shipped to the prover
    as a tarball (--send), the verifier on its own pod, the loopback probe on."""
    from verity_numerical.bench.drivers import b_interactive as B
    monkeypatch.setattr(B, "STAGE_DIR", tmp_path)
    kw = dict(verifier="vy-bl-verifier", verifier_addr="10.0.0.3:7000", campaign="c", lane="bligero-real-k", backend_args=["L=4096", "PP=4"])
    cell = C.plan("B-interactive", "gemm-coordinate/k2048/sm80-mma-bf16+frame-v3/blake3-keyed", str(SLICE), "vy-bl-a100", **kw)
    assert (cell.family, cell.relation, cell.input_set["kind"]) == ("B-Ligero", "bf16-ampere-x4-k2048+blake3-xob", "set")
    v, p = cell.jobs
    assert (v["pod"], p["pod"]) == ("vy-bl-verifier", "vy-bl-a100") and "ROLE=verifier" in v["command"]
    tar = Path(p["send"][0])
    assert tar.is_file() and p["argv"][p["argv"].index("--send") + 1] == str(tar)
    assert {"ROLE=prover", "REL=bf16-ampere-x4-k2048+blake3-xob", "VERIFIER=10.0.0.3:7000", "LOOPBACK=1", "L=4096", f"SET={tar.name}",
            "START=1024", "OPERATOR=bligero-real-k"} <= set(p["command"])
    assert (Path(C.LW.REPO) / B.SCRIPT).is_file()
    with pytest.raises(C.NotSupported, match="vllm-v1"):
        C.plan("B-interactive", "gemm-coordinate/k2048/sm80-mma-bf16+vllm-v1/sha-256", str(SLICE), "vy-bl-a100", **kw)
    with pytest.raises(C.NotSupported, match="producing lane"):
        C.plan("B-interactive", "gemm-coordinate/k2048/sm80-mma-bf16+frame-v3/sha-256", str(SLICE), "vy-bl-a100", **{**kw, "lane": None})
    doc = {"workload_fingerprint": {"relation": "GemmCoordinate<2048> over sm80.mma.m16n8k16.bf16 (4 steps per column)",
                                    "statement_relation": "bf16-ampere-x4-k2048+blake3-xob"}}
    assert not any("relation" in p_ for p_ in C.check(cell, doc))


def test_input_set_directories_resolve():
    ref = C.resolve_input_set(str(SLICE))
    assert (ref["kind"], ref["subcircuit"], ref["source"], ref["n"]) == ("set", "gemm-coordinate/k2048/sm80-mma-bf16", "captured", 2)


@pytest.mark.parametrize("art,statement,input_set", [("6d1295ed", *BF16),
                                                     ("d1961ba4", "gemm-coordinate/k1536/sm89-mma-e4m3+frame-v3/blake3-keyed",
                                                      "bench-instances-fp8-ada-v1")])
def test_real_flock_cells_pass_the_check(art, statement, input_set):
    assert C.check(_cell(statement, input_set), _doc(art)) == []


def _meas_drop(doc, name):
    doc["measurements"] = [m for m in doc["measurements"] if m["name"] != name]


MUTATIONS = [
    (lambda d: d["protocol"].update(runs=3), "protocol: warm True, runs 3"),
    (lambda d: d["protocol"].update(contended=True), "contended True"),
    (lambda d: d["sweep"].update(plateau=False), "not a plateau point"),
    (lambda d: d["workload_fingerprint"]["verifier"].update(placement="loopback"), "verifier placement 'loopback'"),
    (lambda d: _meas_drop(d, "live.loopback_round_seconds"), "no loopback probe"),
    (lambda d: d["interaction"].pop("rtt_method"), "no open-connection RTT probe"),
    (lambda d: d["workload_fingerprint"].update(relation="fp8-ada"), "relation 'fp8-ada'"),
    (lambda d: d["workload_fingerprint"]["instances"].update(dataset="bench-instances/v1"), "instances 'bench-instances/v1'"),
]


@pytest.mark.parametrize("mutate,why", MUTATIONS, ids=[w for _m, w in MUTATIONS])
def test_check_catches(mutate, why):
    doc = copy.deepcopy(_doc("6d1295ed"))
    mutate(doc)
    probs = C.check(_cell(), doc)
    assert any(why in p for p in probs), probs


def test_check_catches_one_host(tmp_path):
    p, v = _run(tmp_path / "p", "h1", _doc("6d1295ed")), _run(tmp_path / "v", "h1", None)
    assert any("one host" in x for x in C.check(_cell(), _doc("6d1295ed"), p, v))


def _run(d: Path, host: str, result: dict | None) -> Path:
    (d / "out").mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"computation": {"tree_sha": "t-" + host}, "environment": {"hostname": host}}))
    (d / "out" / "binary.sha256").write_text("ab" * 32 + "  flock-pure-gpu\n")
    if result is not None:
        (d / "result.json").write_text(json.dumps(result))
    return d


RUNS = DATA / "flock-runs"
PROVER_RUN, VERIFIER_RUN = "r20260925-233831-a064", "r20260925-233818-a37e"


def _real_runs(tmp_path: Path, mutate=None) -> tuple[Path, Path]:
    """The runs behind art:1589ffe1 (H100 BF16, the open-connection re-run) as ``verity_flock.register`` reads them: job.json,
    out/binary.sha256, the prover's result.json (``artifacts`` naming its plateau point p4-16384, as bench.py writes it in the run
    directory) and that point's session logs (prove-s0/s1, loopback-prove-s0/s1)."""
    shutil.copytree(RUNS, tmp_path / "runs")
    p, v = tmp_path / "runs" / PROVER_RUN, tmp_path / "runs" / VERIFIER_RUN
    if mutate is not None:
        doc = json.loads((p / "result.json").read_text())
        mutate(doc)
        (p / "result.json").write_text(json.dumps(doc))
    return p, v


def test_register_runs_the_backends_script_and_stamps_the_cell(tmp_path):
    p, v = _real_runs(tmp_path)
    res = C.register(_cell(), str(p), str(v), tmp_path / "work", dry_run=True)
    doc = json.loads(Path(res["meta"]).read_text())
    assert res["problems"] == C.check(_cell(), doc, p, v) == doc["validation"]["cell_problems"] == []
    assert "refused" not in res
    assert doc["cell"]["backend"] == "C-interactive" and doc["cell"]["statement"] == BF16[0]
    assert doc["cell"]["input_set"]["id"] == BF16[1]
    assert (doc["lane"], doc["registered_by"]) == ("bench-spine", "bench-spine")                # the lane that ran it is the producer
    assert doc["cell"]["script_lane"]["registered_by"] in ("flock-backend", "bench-spine")      # before / after flock-backend's --lane
    assert doc["workload_fingerprint"]["software"]["verifier"]["host"] == "5f0e35820d06"        # verity_flock.register's patch
    assert doc["workload_fingerprint"]["software"]["verifier"]["run"] == VERIFIER_RUN
    put = res["put"]
    assert put[:5] == ["research", "data", "put", "--kind", "bench-result/v1"] and put[-1] == "--preserve"
    assert f"run_files=<run_files of {PROVER_RUN}>" in put


@pytest.mark.parametrize("takes", [True, False])
def test_register_passes_the_lane_to_verity_flock_register(tmp_path, monkeypatch, takes):
    """flock-backend a9d13f68 (PR #34) requires ``--lane``; the register before it rejects the flag."""
    drv = drivers.driver("C-interactive")
    monkeypatch.setattr(drv, "takes_lane", lambda: takes)
    (cmd,) = drv.register(_cell(), tmp_path / "p", tmp_path / "v", tmp_path / "m.json")
    assert cmd[1:3] == ["-m", "verity_flock.register"]
    assert (cmd[-2:] == ["--lane", "bench-spine"]) is takes and ("--lane" in cmd) is takes


def test_takes_lane_reads_the_installed_script():
    from verity_flock import register as FR

    assert drivers.driver("C-interactive").takes_lane() is ("--lane" in Path(FR.__file__).read_text())


def test_register_refuses_a_result_off_protocol(tmp_path):
    p, v = _real_runs(tmp_path, lambda d: d["protocol"].update(runs=2))
    res = C.register(_cell(), str(p), str(v), tmp_path / "work", dry_run=True)
    assert res["refused"] and any("runs 2" in x for x in res["problems"])


def test_cli(tmp_path, capsys):
    out = tmp_path / "cell.json"
    assert C.main(["plan", "--backend", "C-interactive", "--statement", BF16[0], "--input-set", BF16[1], "--prover", "vy-p",
                   "--verifier", "vy-v", "--verifier-addr", "h:7400", "--rtt-target", "h:22", "--lane", "bench-spine",
                   "--out", str(out)]) == 0
    assert "research run --on vy-v" in capsys.readouterr().out
    assert C.main(["run", "--cell", str(out), "--dry-run"]) == 0
    assert C.main(["plan", "--backend", "B-fs", "--statement", BF16[0], "--input-set", BF16[1], "--prover", "vy-p"]) == 3
    assert "n/s" in json.loads(capsys.readouterr().out.splitlines()[-1])


def test_every_driver_module_declares_its_backend():
    for b in C.BACKENDS:
        d = drivers.driver(b)
        if d is not None:
            assert (d.BACKEND, d.FAMILY) == (b, C.family_of(b)) and callable(d.jobs) and callable(d.register) and d.RESULT
