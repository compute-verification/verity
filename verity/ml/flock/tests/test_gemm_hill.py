"""The BF16 hillclimb's records (`verity/ml/flock/pod/gemm_hill.py`, `gemm_slowdown.py`): the preflight check's verdict as a
point reads it, the readers of records written before the check was named, and a GPU session's points, each citing the
session's one preflight check and refused when its key differs from the check's."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

POD = Path(__file__).resolve().parents[1] / "pod"
sys.path.insert(0, str(POD))
import gemm_hill as GH  # noqa: E402
import gemm_slowdown as GS  # noqa: E402

DEF = "GemmCoordinate_v2{K=64,DOT=HopperBF16WgmmaDot16_v1}"
DEF2048 = "GemmCoordinate_v2{K=2048,DOT=HopperBF16WgmmaDot16_v1}"


def _preflight_record(path: Path, passes=(True, True, True), definition: str = DEF) -> Path:
    cases = [{"case": c, "pass": p, "proofs_equal": p, "transcripts_equal": p} for c, p in zip(GS.PREFLIGHT_CASES, passes)]
    path.write_text(json.dumps({"definition": definition, "live": {"statement_digest": "d" * 64}, "selftest": {"gpu_cases": cases}}) + "\n")
    return path


def test_the_preflight_check_passes_only_when_every_gpu_case_is_byte_equal(tmp_path):
    assert GS.preflight(_preflight_record(tmp_path / "a.jsonl"), shapes_needed=1)["status"] == "passed"
    assert GS.preflight(_preflight_record(tmp_path / "b.jsonl", (True, False, True)), shapes_needed=1)["status"] == "failed"
    assert GS.preflight(_preflight_record(tmp_path / "c.jsonl"), shapes_needed=2)["status"] == "failed"
    assert GS.preflight(tmp_path / "absent.jsonl")["status"] == "not_run"


def test_a_slowdown_row_reads_the_preflight_check_under_its_old_name_too(tmp_path, monkeypatch):
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    cfg = sorted(GS.CONFIGS)[0]
    row = {"configs": {cfg: {"phase": GS.CONFIGS[cfg][0], "slowdown": 2.0, "slowdown_prove_only": 1.5, "prover_s": 4.0, "native_s": 2.0}},
           "native": {"gpu": "fake"}}
    for key in ("preflight", "gate"):
        (tmp_path / f"{key}.json").write_text(json.dumps(row | {key: {"status": "passed"}}))
        out = tmp_path / f"{key}-result.json"
        GS.point(SimpleNamespace(slowdown=str(tmp_path / f"{key}.json"), config=cfg, from_run="r1", result=str(out)))
        assert json.loads(out.read_text())["validation"]["status"] == "passed"


def test_compute_and_metrics_take_the_preflight_record_under_its_old_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(GS, "compute", lambda a: seen.setdefault("compute", a.preflight))
    monkeypatch.setattr(GH, "metrics", lambda a: seen.setdefault("metrics", a.preflight))
    for flag in ("--gate", "--preflight"):
        seen.clear()
        monkeypatch.setattr(sys, "argv", ["gemm_slowdown.py", "compute", "--sweep", "s", flag, "p.jsonl", "--native", "n", "--result", "r"])
        GS.main()
        monkeypatch.setattr(sys, "argv", ["gemm_hill.py", "metrics", "--K", "64", flag, "p.jsonl", "--sweep", "s", "--stamps", "t",
                                          "--dmon", "d", "--out", "o", "--result", "r"])
        GH.main()
        assert seen == {"compute": "p.jsonl", "metrics": "p.jsonl"}


SHA = "a" * 64


def _key(path: Path, fc: dict, sha: str = SHA) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"binaries": {"flock-circuit": sha}, "env": fc, "gpu": {"NVIDIA_VISIBLE_DEVICES": "GPU-1"}}))
    return str(path)


def _session(R: Path, monkeypatch, points: int, point_fc: list[dict]) -> list[dict]:
    """A GPU session's run dir as 74-gemm-hill.sh leaves it before its metrics: the preflight check's record and key, and per
    point its sweep's record, stamps and key; the phases put the job's start at 1000, the preflight check at 1001-1041, point
    i's sweep at 1042 + 40 i for 30 + i s, and the metrics at 1001 + 40 (points + 1). Then each point's metrics and the
    session's outputs, through gemm_hill.main."""
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-session")
    monkeypatch.setenv("HILL_PINNED", "0,1,2,3")
    try:
        from verity_numerical.bench import contract

        monkeypatch.setattr(contract, "probe_hardware", lambda *a, **k: {"gpu": None, "cpu": None})
    except ImportError:
        pass
    (R / "out/preflight/classes").mkdir(parents=True)
    _preflight_record(R / "out/preflight/classes/results.jsonl", definition=DEF2048)
    pk = _key(R / "out/preflight/key.json", {"FC_VERIFY_AHEAD": "10"})
    (R / ".research-source.json").write_text(json.dumps({"commit": "c" * 40, "dirty": False}))
    census = R / "hardware.json"
    census.write_text(json.dumps([{"id": GH.DTYPES["bf16"][2], "flop_per_second": 1.0e15, "device": "RTX PRO 6000"}]))
    phases = ["job\tstart\t1000", "preflight\tstart\t1001", "preflight\tend\t1041"]
    for i in range(points):
        t = 1042 + 40 * i
        phases += [f"sweep-p{i}\tstart\t{t}", f"sweep-p{i}\tend\t{t + 30 + i}"]
    phases.append(f"metrics\tstart\t{1001 + 40 * (points + 1)}")
    (R / "out/phases.tsv").write_text("\n".join(phases) + "\n")
    out = []
    for i in range(points):
        P = R / f"out/p{i}"
        P.mkdir(parents=True)
        live = {"accepted": True, "instances": 2048, "prove_total_s": 1.0, "witness_s": 0.1, "statement_digest": "d" * 64, "dense_m": 35,
                "k_log": 24, "buckets": [], "coins": "live-os"}
        (P / "results.jsonl").write_text(json.dumps({"definition": DEF2048, "shape": "s", "live": live, "stage": {"n": 2048},
                                                     "stage_cached": True, "domain": "total", "stage_s": 5.0}) + "\n")
        t = 1043 + 40 * i
        recs = [{"run": 0, "warm": True, "accepted": True},
                *({"run": 1 + j, "warm": False, "accepted": True, "prove_total_s": 1.0,
                   "verdict": {"verify_s": 2.0, "session_s": 3.0, "coin_seed": "ab"}, "tables": [None]} for j in range(2))]
        (P / "stamps.tsv").write_text(f"{t}.0\tSTART\t[\"prove\"]\n" + "".join(f"{t + 5 + 4 * j}.0\tLIVE\t{json.dumps(x)}\n" for j, x in enumerate(recs))
                                      + f"{t + 20}.0\tEXIT\t0\n")
        monkeypatch.setattr(sys, "argv", ["gemm_hill.py", "metrics", "--K", "2048", "--preflight", str(R / "out/preflight/classes/results.jsonl"),
                                          "--sweep", str(P / "results.jsonl"), "--stamps", str(P / "stamps.tsv"), "--warm", "1", "--runs", "2",
                                          "--dmon", str(R / "out/dmon.tsv"),
                                          "--phases", str(R / "out/phases.tsv"), "--census", str(census), "--root", str(R), "--point", str(i),
                                          "--points", str(points), "--preflight-key", pk, "--point-key", _key(P / "key.json", point_fc[i]),
                                          "--out", str(P / "metrics.json"), "--result", str(P / "result.json"),
                                          "--hillclimb", str(R / f"hillclimb-{i}.json")])
        GH.main()
        out.append(json.loads((R / f"hillclimb-{i}.json").read_text()))
    monkeypatch.setattr(sys, "argv", ["gemm_hill.py", "session", "--dir", str(R), "--points", str(points)])
    GH.main()
    return out


def test_every_point_of_a_gpu_session_cites_its_one_preflight_check_and_shares_its_gpu_held_seconds(tmp_path, monkeypatch):
    pts = _session(tmp_path, monkeypatch, 3, [{"FC_VERIFY_AHEAD": "10"}] * 3)
    record = hashlib.sha256((tmp_path / "out/preflight/classes/results.jsonl").read_bytes()).hexdigest()
    assert [p["point"] for p in pts] == [{"index": i, "of": 3} for i in range(3)]
    assert {(p["coins"], p["scheme"]) for p in pts} == {("live-os", "flock-i-v1+livecoins")}      # the sweep record's coins
    assert all(p["byte_identical"] and p["preflight"]["status"] == "passed" and p["preflight"]["key_matches"] is True
               and p["preflight"]["record_sha256"] == record for p in pts)
    # the session held its GPU from 1000 to the metrics at 1161; the points' sweeps were 30, 31 and 32 s, and the 68 s outside
    # them are shared equally
    assert [p["gpu_held_s_point"] for p in pts] == [30.0, 31.0, 32.0] and {p["gpu_held_s_session"] for p in pts} == {161.0}
    assert abs(sum(p["gpu_held_s_amortized"] for p in pts) - 161.0) < 1e-9
    assert [round(p["gpu_held_s_amortized"] - p["gpu_held_s_point"], 9) for p in pts] == [round(68 / 3, 9)] * 3
    metrics = json.loads((tmp_path / "out/p0/metrics.json").read_text())
    assert metrics["gpu_held_s_job"] is None and metrics["windows"]["job"]["held_s"] == 161.0 and metrics["contract_problems"] == []
    for i in range(3):
        res = json.loads((tmp_path / f"out/p{i}/result.json").read_text())
        assert res["kind"] == "bench-point/v1" and res["validation"]["status"] == "passed"
        assert res["measurement_files"] == [f"out/p{i}/metrics.json", f"out/p{i}/results.jsonl"]
    outs = json.loads((tmp_path / "outputs.json").read_text())["outputs"]
    assert [(o["name"], o["kind"]) for o in outs] == [("preflight", "preflight-record/v0")] + [
        x for i in range(3) for x in ((f"result-{i}", "bench-point/v1"), (f"hillclimb-{i}", "hillclimb-point/v0"))]
    names = [o["name"] for o in outs]
    for j, o in enumerate(outs):
        assert all(v[1:] in names[:j] for v in (o.get("refs") or {}).values())    # every @ref names an earlier output
        assert (tmp_path / o["file"]).is_file()
    assert outs[-1]["refs"] == {"preflight": "@preflight", "result": "@result-2"} and outs[-1]["meta"]["point"] == 2
    sess = json.loads((tmp_path / "result.json").read_text())
    assert (sess["kind"], sess["validation"]["status"], sess["workload_fingerprint"]["points"]) == ("hillclimb-session/v0", "passed", 3)
    m = {x["name"]: x["value"] for x in sess["measurements"]}
    assert (m["hill.points"], m["hill.points_passed"], m["hill.gpu_held_s_session"], m["hill.median_gpu_held_s_point"]) == (3, 3, 161.0, 31.0)
    assert all((tmp_path / f).is_file() for f in sess["measurement_files"] + sess["artifacts"])


def test_a_session_of_no_points_is_its_preflight_check_alone(tmp_path):
    (tmp_path / "out/preflight/classes").mkdir(parents=True)
    for passes, code in (((True, True, True), 0), ((True, False, True), 1)):
        _preflight_record(tmp_path / "out/preflight/classes/results.jsonl", passes, DEF2048)
        with pytest.raises(SystemExit) as e:
            GH.session(SimpleNamespace(dir=str(tmp_path), points=0))
        assert e.value.code == code
        outs = json.loads((tmp_path / "outputs.json").read_text())["outputs"]
        assert [(o["name"], o["kind"], o["file"]) for o in outs] == [("preflight", "preflight-record/v0", "out/preflight/classes/results.jsonl")]
        assert not (tmp_path / "result.json").exists()


def test_a_point_whose_settings_differ_from_its_preflight_checks_is_refused(tmp_path, monkeypatch):
    pts = _session(tmp_path, monkeypatch, 2, [{"FC_VERIFY_AHEAD": "10"}, {"FC_VERIFY_AHEAD": "4"}])
    assert pts[0]["byte_identical"] and pts[0]["preflight"]["key_matches"] is True
    assert not pts[1]["byte_identical"] and pts[1]["preflight"]["key_matches"] is False
    assert pts[1]["why"] == "point 1's env.FC_VERIFY_AHEAD differ from its preflight check's"
    assert json.loads((tmp_path / "out/p1/result.json").read_text())["validation"]["status"] == "failed"
    sess = json.loads((tmp_path / "result.json").read_text())
    assert sess["validation"]["status"] == "failed" and "point 1: failed (point 1's env.FC_VERIFY_AHEAD" in sess["validation"]["detail"]
    assert {x["name"]: x["value"] for x in sess["measurements"]}["hill.points_passed"] == 1


def test_a_point_without_a_key_is_refused_and_the_key_names_what_differs(tmp_path, monkeypatch):
    assert GH._key_differs(None, {"env": {}}) == ["the key"]
    a = {"binaries": {"flock-circuit": SHA}, "env": {"FC_X": "1"}, "gpu": {"CUDA_VISIBLE_DEVICES": "0"}}
    b = {"binaries": {"flock-circuit": "b" * 64}, "env": {"FC_Y": "1"}, "gpu": {"CUDA_VISIBLE_DEVICES": "0"}}
    assert GH._key_differs(a, a) == [] and GH._key_differs(a, b) == ["binaries.flock-circuit", "env.FC_X", "env.FC_Y"]
    sha = tmp_path / "binary.sha256"
    sha.write_text(f"{SHA}  /w/target-0/release/flock-circuit\n{'b' * 64}  /w/target-1/release/flock-circuit\n")
    monkeypatch.setenv("FC_VERIFY_AHEAD", "10")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "GPU-1")
    GH.key(SimpleNamespace(binary_sha=str(sha), out=str(tmp_path / "key.json")))
    k = json.loads((tmp_path / "key.json").read_text())
    assert k["binaries"] == {"/w/target-0/release/flock-circuit": SHA, "/w/target-1/release/flock-circuit": "b" * 64}
    assert k["env"]["FC_VERIFY_AHEAD"] == "10" and all(n.startswith("FC_") for n in k["env"]) and k["gpu"]["NVIDIA_VISIBLE_DEVICES"] == "GPU-1"


def test_a_roll_up_keeps_each_point_of_a_session_once(tmp_path):
    f = tmp_path / "rollup.json"
    sub = GH.subcircuit(2048)
    f.write_text(json.dumps({"schema": GH.ROLLUP_SCHEMA, "subcircuit": sub, "points": []}))
    for i in (0, 1, 1):
        p = tmp_path / f"p{i}.json"
        p.write_text(json.dumps({"schema": GH.POINT_SCHEMA, "subcircuit": sub, "run_id": "r-session", "commit": "c" * 40,
                                 "point": {"index": i, "of": 2}}))
        GH.rollup(SimpleNamespace(point=str(p), attempt=None, run_dir=None, file=str(f)))
    assert [p["point"]["index"] for p in json.loads(f.read_text())["points"]] == [0, 1]


def _fake_store(tmp_path: Path, run: str, n: int, sub: dict) -> dict:
    """A `research` on RESEARCH_BIN (a command, as on node 1) whose store holds attempt `run`: a GPU session's preflight record,
    result.json and n points, the points also in the run's directory with its outputs.json."""
    arts = {"preflight": "art:" + "a" * 64, "result": "art:" + "b" * 64} | {f"hillclimb-{i}": f"art:{i:064x}" for i in range(n)}
    blobs, rd = tmp_path / "blobs", tmp_path / run
    blobs.mkdir()
    rd.mkdir()
    for i in range(n):
        b = json.dumps({"schema": GH.POINT_SCHEMA, "subcircuit": sub, "run_id": run, "commit": "c" * 40, "step": 12,
                        "point": {"index": i, "of": n}})
        (blobs / f"{i:064x}.json").write_text(b)
        (rd / f"hillclimb-{i}.json").write_text(b)
    (rd / "outputs.json").write_text(json.dumps({"outputs": [{"name": f"hillclimb-{i}", "file": f"hillclimb-{i}.json"} for i in range(n)]}))
    exe = tmp_path / "fake_research.py"
    exe.write_text(f"""import hashlib, json, shutil, sys
a, blobs = sys.argv[1:], {str(blobs)!r}
if a[:2] == ["data", "show"] and a[2] == {run!r}:
    sys.exit(print(json.dumps({{"id": a[2], "outputs": {json.dumps(arts)}}})))
if a[:2] == ["data", "show"] and a[2].startswith("art:"):
    b = open(blobs + "/" + a[2][4:] + ".json", "rb").read()
    sys.exit(print(json.dumps({{"id": a[2], "manifest": {{"payload": {{"sha256": hashlib.sha256(b).hexdigest()}}}}}})))
if a[:2] == ["data", "fetch"]:
    i = a[2][4:]
    sys.exit(print(shutil.copy(blobs + "/" + i + ".json", a[a.index("--to") + 1] + f"/hillclimb-{{int(i, 16)}}.json")))
sys.exit(f"research data show: no attempt {{a[2]}} in the store")
""")
    return arts


def test_a_roll_up_from_the_store_cites_each_points_session_and_preflight_arts(tmp_path, monkeypatch):
    """`rollup --attempt` takes every point a session published, from the store or from its run directory when those bytes are
    the arts', and records its own, its session's and its preflight check's art ids; a point rolled up before from its file
    gets them, and nothing is appended twice."""
    sub = GH.subcircuit(2048)
    arts = _fake_store(tmp_path, "r-session", 3, sub)
    monkeypatch.setenv("RESEARCH_BIN", f"{sys.executable} {tmp_path / 'fake_research.py'}")
    want = [{"point": arts[f"hillclimb-{i}"], "session": arts["result"], "preflight": arts["preflight"]} for i in range(3)]
    for run_dir in (None, str(tmp_path / "r-session")):
        f = tmp_path / "rollup.json"
        f.write_text(json.dumps({"schema": GH.ROLLUP_SCHEMA, "subcircuit": sub, "points": []}))
        GH.rollup(SimpleNamespace(point=str(tmp_path / "r-session/hillclimb-0.json"), attempt=None, run_dir=None, file=str(f)))
        assert "arts" not in json.loads(f.read_text())["points"][0]
        for _ in range(2):
            GH.rollup(SimpleNamespace(point=None, attempt="r-session", run_dir=run_dir, file=str(f)))
        pts = json.loads(f.read_text())["points"]
        assert [p["point"]["index"] for p in pts] == [0, 1, 2] and [p["arts"] for p in pts] == want
    with pytest.raises(SystemExit, match="r-elsewhere: not in the store"):
        GH.rollup(SimpleNamespace(point=None, attempt="r-elsewhere", run_dir=None, file=str(f)))
    (tmp_path / "r-session/hillclimb-1.json").write_text("{}")
    with pytest.raises(SystemExit, match="hillclimb-1.json: not the bytes of art:"):
        GH.rollup(SimpleNamespace(point=None, attempt="r-session", run_dir=str(tmp_path / "r-session"), file=str(f)))
