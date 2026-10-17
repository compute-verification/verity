"""The job scripts' helpers on files shaped as the prover and verifier runs write them."""
import json

from verity_numerical.bench import contract

from verity_proof_throughput import cli
from verity_proof_throughput import result as RS

VERDICT = {"session": "l0001-77", "accepted": True, "why": "accepted", "rounds": 61}
ERR = "inputs: 812 ms\nsetup: 2304 ms\n/tmp/b/sessions/l0001-77: 5120 ms\n"


def test_lean_verdict_reads_the_verdict_its_time_and_phases(tmp_path, capsys):
    (tmp_path / "out").write_text("starting\nVERDICT " + json.dumps(VERDICT) + "\n")
    (tmp_path / "err").write_text(ERR)
    (tmp_path / "time").write_text("8.31 7.90 0.40 1048576\n")
    argv = ["lean-verdict", "--session", "l0001-77", "--rc", "0", "--time", str(tmp_path / "time"),
            "--stdout", str(tmp_path / "out"), "--stderr", str(tmp_path / "err")]
    assert cli.main(argv) == 0
    line = json.loads(capsys.readouterr().out)
    assert line == {"session": "l0001-77", "rc": 0, "accepted": True, "why": "accepted", "wall_s": 8.31, "cpu_s": 8.3,
                    "max_rss_kb": 1048576, "inputs_ms": 812, "setup_ms": 2304, "session_ms": 5120}
    assert cli.main([*argv[:4], "1", *argv[5:]]) == 0
    assert json.loads(capsys.readouterr().out)["accepted"] is False


def test_lean_verdict_without_a_verdict_keeps_the_error(tmp_path, capsys):
    (tmp_path / "err").write_text("uncaught exception: refused: plain-leaves statement\n")
    cli.main(["lean-verdict", "--session", "s", "--rc", "1", "--time", str(tmp_path / "t"), "--stdout", str(tmp_path / "o"),
              "--stderr", str(tmp_path / "err")])
    line = json.loads(capsys.readouterr().out)
    assert line["accepted"] is False and "plain-leaves" in line["why"]


def write_point(d, record, hardware):
    (d / "results.jsonl").write_text("noise\n" + json.dumps(record) + "\n")
    (d / "stage-rec.json").write_text(json.dumps({"stage_s": 39.5}))
    (d / "point.json").write_text(json.dumps({"workload": "bf16-k6144", "B": 16, "warm": 1, "runs": 3, "run": "rP"}))
    (d / "gpu-mem.txt").write_text("1200\n18000\n17000\n")
    (d / "hardware.json").write_text(json.dumps(hardware))
    (d / "index.jsonl").write_text("".join(json.dumps({"session": f"l000{i}-1", "accepted": True, "rounds": 61}) + "\n"
                                           for i in range(4)))


def test_point_result_on_the_prover_then_the_verifier(tmp_path, record, hardware, capsys):
    write_point(tmp_path, record, hardware)
    out = tmp_path / "result.json"
    assert cli.main(["point-result", "--workload", "bf16-k6144", "--B", "16", "--dir", str(tmp_path), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert contract.validate(doc) == [] and doc["validation"]["status"] == "passed"
    names = {m["name"]: m["value"] for m in doc["measurements"]}
    assert names["fixed.setup_seconds"] == 39.5 and names["mem.peak_device_bytes"] == 18000 << 20 and names["rounds.protocol"] == 61
    assert (tmp_path / "software.json").exists()

    lean = tmp_path / "lean.jsonl"
    lean.write_text("".join(json.dumps({"session": f"l000{i}-1", "accepted": True, "wall_s": 3.0 + i, "cpu_s": 2.0})
                            + "\n" for i in range(4)))
    (tmp_path / "lean.sha256").write_text("f" * 64)
    assert cli.main(["point-result", "--workload", "bf16-k6144", "--B", "16", "--dir", str(tmp_path), "--lean", str(lean),
                     "--prove-run", "rP", "--bundle", "art:b", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["validation"]["status"] == "passed" and contract.validate(doc) == []
    names = {m["name"]: m["value"] for m in doc["measurements"]}
    assert names["verifier.seconds"] == 5.0   # the timed sessions l0001..l0003; l0000 was the warm one
    fp = doc["workload_fingerprint"]
    assert fp["software"]["verifier"]["sha256"] == "f" * 64 and fp["prove_run"] == "rP" and fp["bundle"] == "art:b"

    drawless = [{"session": f"l000{i}-1", "accepted": False, "why": RS.DRAWLESS, "wall_s": 3.0, "cpu_s": 2.0} for i in range(4)]
    lean.write_text("".join(json.dumps(v) + "\n" for v in drawless))
    cli.main(["point-result", "--workload", "bf16-k6144", "--B", "16", "--dir", str(tmp_path), "--lean", str(lean), "--out", str(out)])
    doc = json.loads(out.read_text())
    assert doc["validation"]["status"] == "passed" and "refused only for having no draw" in doc["validation"]["detail"]

    lean.write_text(json.dumps({"session": "l0000-1", "accepted": False, "why": "bad"}) + "\n" + lean.read_text().split("\n", 1)[1])
    cli.main(["point-result", "--workload", "bf16-k6144", "--B", "16", "--dir", str(tmp_path), "--lean", str(lean), "--out", str(out)])
    assert json.loads(out.read_text())["validation"]["status"] == "failed"


def test_staged_names_the_stage_cache_entry(tmp_path, monkeypatch, capsys):
    rec = {"shape": "ab" * 32, "stage": {"n": 16, "k_log": 25}, "stage_s": 12.0, "stage_cached": "c1"}
    (tmp_path / "r.jsonl").write_text(json.dumps(rec) + "\n")
    monkeypatch.setenv("FLOCK_STAGE_CACHE", str(tmp_path / "cache"))
    assert cli.main(["staged", "--results", str(tmp_path / "r.jsonl"), "--cache-dir"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "cache" / "c1")
    assert cli.main(["staged", "--results", str(tmp_path / "r.jsonl")]) == 0
    assert json.loads(capsys.readouterr().out)["k_log"] == 25


def test_baseline_pins_a_sweep_and_the_table_renders_it(tmp_path, record, hardware, monkeypatch, capsys):
    write_point(tmp_path, record, hardware)
    out = tmp_path / "result.json"
    cli.main(["point-result", "--workload", "bf16-k6144", "--B", "16", "--dir", str(tmp_path), "--out", str(out)])
    doc = json.loads(out.read_text()) | {"run_id": "r20261017-000000-abcd"}
    monkeypatch.setattr(cli, "BASELINE", tmp_path / "baseline.json")
    monkeypatch.setattr(cli, "_load_result", lambda ref: doc)
    assert cli.main(["baseline"]) == 1
    assert cli.main(["baseline", "--lock", "r20261017-000000-abcd", "--commit", "c" * 40]) == 0
    pin = json.loads((tmp_path / "baseline.json").read_text())["bf16-k6144"]
    assert pin["points"] == {"16": "r20261017-000000-abcd"} and pin["commit"] == "c" * 40
    capsys.readouterr()
    assert cli.main(["table", "--baseline", "--json"]) == 0
    assert [r["run"] for r in json.loads(capsys.readouterr().out)] == ["r20261017-000000-abcd"]


def test_the_leased_gpu_keeps_fixed_properties_only():
    csv = ("name,uuid,driver_version,memory.total,compute_cap,clocks.sm,clocks.max.sm,power.limit,temperature.gpu\n"
           "NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU-1234, 580.95.05, 97887, 12.0, 2400, 2430, 600.00, 41\n")
    got = cli.leased_gpu(csv)
    assert got == {"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "compute_cap": "12.0", "clocks.max.sm": "2430",
                   "power.limit": "600.00", "memory.total": "97887", "driver": "580.95.05"}
    assert cli.leased_gpu("name\n") is None
