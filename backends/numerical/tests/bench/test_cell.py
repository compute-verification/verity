"""The benchmark entry point (``bench.cell``): planning a cell into its backend's jobs, the reasons a cell cannot run, the protocol
check on two real Flock cells (``data/flock-cells``: art:6d1295ed and art:d1961ba4, the registered bench-result/v1 documents,
both independently replayed) and their mutations, and registration through ``verity_flock.register``."""
from __future__ import annotations

import copy
import json
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
    with pytest.raises(ValueError, match="not a census instance set"):
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


def test_register_runs_the_backends_script_and_stamps_the_cell(tmp_path):
    p = _run(tmp_path / "r20260926-010000-aaaa", "prover-host", _doc("6d1295ed"))
    v = _run(tmp_path / "r20260926-010000-bbbb", "verifier-host", None)
    res = C.register(_cell(), str(p), str(v), tmp_path / "work", dry_run=True)
    assert res["problems"] == [] and "refused" not in res
    doc = json.loads(Path(res["meta"]).read_text())
    assert doc["cell"]["backend"] == "C-interactive" and doc["cell"]["statement"] == BF16[0]
    assert doc["cell"]["input_set"]["id"] == BF16[1]
    assert (doc["lane"], doc["registered_by"]) == ("bench-spine", "bench-spine")                # the lane that ran it is the producer
    assert doc["cell"]["script_lane"] == {"lane": "flock-backend", "registered_by": "flock-backend"}
    assert doc["workload_fingerprint"]["software"]["verifier"]["host"] == "verifier-host"   # verity_flock.register's patch
    put = res["put"]
    assert put[:5] == ["research", "data", "put", "--kind", "bench-result/v1"] and put[-1] == "--preserve"
    assert "run_files=<run_files of r20260926-010000-aaaa>" in put


def test_register_refuses_a_result_off_protocol(tmp_path):
    doc = _doc("6d1295ed")
    doc["protocol"]["runs"] = 2
    p = _run(tmp_path / "p", "prover-host", doc)
    v = _run(tmp_path / "v", "verifier-host", None)
    res = C.register(_cell(), str(p), str(v), tmp_path / "work", dry_run=True)
    assert res["refused"] and any("runs 2" in x for x in res["problems"])


def test_cli(tmp_path, capsys):
    out = tmp_path / "cell.json"
    assert C.main(["plan", "--backend", "C-interactive", "--statement", BF16[0], "--input-set", BF16[1], "--prover", "vy-p",
                   "--verifier", "vy-v", "--verifier-addr", "h:7400", "--rtt-target", "h:22", "--out", str(out)]) == 0
    assert "research run --on vy-v" in capsys.readouterr().out
    assert C.main(["run", "--cell", str(out), "--dry-run"]) == 0
    assert C.main(["plan", "--backend", "B-fs", "--statement", BF16[0], "--input-set", BF16[1], "--prover", "vy-p"]) == 3
    assert "n/s" in json.loads(capsys.readouterr().out.splitlines()[-1])


def test_every_driver_module_declares_its_backend():
    for b in C.BACKENDS:
        d = drivers.driver(b)
        if d is not None:
            assert (d.BACKEND, d.FAMILY) == (b, C.family_of(b)) and callable(d.jobs) and callable(d.register) and d.RESULT
