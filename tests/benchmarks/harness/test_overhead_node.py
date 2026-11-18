"""`verity bench overhead`'s node side end to end, with stand-ins for bench.py, nsys and verify.py."""

from __future__ import annotations

import hashlib
import json
import sys

import pytest
from overhead_standins import (
    ARM,
    CUBIN,
    CUBIN_ART,
    H200,
    HOST,
    LINEARS,
    SHA,
    STAMP,
    STRAY,
    UUID,
    Node,
    between,
    brute_force,
    node_points,
    node_run,
)
from overhead_standins import scheme as scheme
from overhead_standins import stamped as stamped
from verity_bench import overhead as O
from verity_bench import overhead_node

# ---- the node side ----------------------------------------------------------------------------------------------------------------


def test_the_node_sweeps_for_the_points_in_rounds_of_bench_py(tmp_path):
    node = Node(tmp_path)
    points = json.loads(node_points(tmp_path, node).read_text())
    want = brute_force()
    assert (points["c"], points["l"]) == (want.c, want.lat) and points["error"] is None
    assert points["schema"] == O.POINTS_SCHEMA and points["families"] == list(O.FP8)
    assert set(map(int, points["times_ms"])) >= set(O.powers()) | {want.c, want.lat}
    for argv in node.calls:
        assert argv[1] == str(overhead_node.HARNESS / "bench.py") and "--bridge" in argv
        assert between(argv, "--expect-uuid") == [UUID] and between(argv, "--families") == list(O.FP8)
        assert len(between(argv, "--shapes")) <= overhead_node.CHUNK * len(LINEARS)
    assert len(points["rounds"]) == len(node.calls)
    result = json.loads((tmp_path / "r20261104-020000-0001" / "result.json").read_text())
    assert result["validation"]["status"] == "passed" and result["validation"]["evidence"] == ["out/points.json"]
    assert {m["name"]: m["value"] for m in result["measurements"]} == {"C": want.c, "L": want.lat}
    for m in result["measurements"]:
        assert (m["unit"], m["run"], m["commit"], m["host"]) == ("m", "r20261104-020000-0001", SHA, STAMP)
        assert m["inputs"] == ["measured", "out/points.json"] and m["mark"] == "measured"
        assert m["ci95"] is None and m["verdict"] is None


def test_the_node_refuses_a_sweep_without_every_library_the_card_chooses_among(tmp_path):
    out, run_dir = tmp_path / "out", tmp_path
    node = Node(tmp_path, bridge_errors={"deepgemm": "ImportError: no DeepGEMM"})
    rc = overhead_node.main(["points", "--card", "h200", "--uuid", UUID, "--out", str(out), "--run-dir", str(run_dir),
                             "--run-id", "r1"], run=node)
    assert rc == 1 and len(node.calls) == 1 and "--bridge" in node.calls[0]
    assert json.loads((out / "points.json").read_text())["error"] == "the H200's deepgemm didn't load (ImportError: no DeepGEMM)"
    assert O.libraries_why(H200, {}) == ("bench.py ran without --bridge: the H200's cublaslt, cutlass, deepgemm are the "
                                        "stock kernel's libraries")
    assert O.libraries_why(O.CARDS["rtx-pro-6000"], {}) is None


def test_the_node_refuses_a_sweep_on_another_card(tmp_path):
    out = tmp_path / "r" / "out"
    rc = overhead_node.main(["points", "--card", "h200", "--uuid", UUID, "--out", str(out), "--run-dir",
                             str(tmp_path / "r"), "--run-id", "r20261104-020000-0001"],
                            run=Node(tmp_path, card="NVIDIA RTX PRO 6000 Blackwell Server Edition"))
    doc = json.loads((out / "points.json").read_text())
    assert rc == 1 and "not the H200" in doc["error"] and doc.get("c") is None
    assert json.loads((tmp_path / "r" / "result.json").read_text())["validation"]["status"] == "failed"


def test_the_result_holds_a_callers_whole_entries_as_given_and_its_own_through_the_kit(tmp_path):
    """Serving cost's benchmark hands `write_result` entries it wrote whole, with fields the kit's writer has no argument
    for: they are held as given, and only the entries given by `measured` are the kit's."""

    run_dir, out = tmp_path / "r1", tmp_path / "r1" / "out"
    out.mkdir(parents=True)
    whole = {"name": "cost", "value": None, "unit": "tokens/s", "run": "another", "rung": 3, "inputs": ["a load"]}
    overhead_node.write_result(run_dir, "r1", out, "record.json", True, "said", {"door": "cost"}, [whole])
    assert json.loads((run_dir / "result.json").read_text())["measurements"] == [whole]
    overhead_node.write_result(run_dir, "r1", out, "record.json", True, "said", {"door": "cost"}, [whole],
                               measured=[{"name": "C", "value": 512, "unit": "m"}])
    held, own = json.loads((run_dir / "result.json").read_text())["measurements"]
    assert held == whole and (own["name"], own["value"], own["run"], own["host"]) == ("C", 512, "r1", STAMP)
    assert (own["mark"], own["inputs"]) == ("measured", ["measured", "out/record.json"])


def test_a_timed_run_gives_gamma_over_the_fastest_and_the_least_sm_stock_kernel_with_its_controls(tmp_path, scheme):
    """At each point the arm takes 1.5 times the busy SM-time of the fastest stock kernel by wall time and 1.875 times the
    one of least SM-time, and the host's enqueue, a tenth of each side's SM-time, counts above stock's: γ is 0.55 and
    0.9625; the weight side bounds it at 0.66, and the twin, the A/A control, is 0. Every row's verdict and ε stand beside
    γ, and nothing is refused."""

    points = node_points(tmp_path, Node(tmp_path))
    node = Node(tmp_path)
    rc, run_dir = node_run(tmp_path, node, points, "pearl-plus", "art:" + "b" * 64)
    doc = json.loads((run_dir / "out" / "overhead.json").read_text())
    assert rc == 0 and doc["whys"] == [] and set(doc["points"]) == {"C", "L"}
    pts = json.loads(points.read_text())
    for name, p in doc["points"].items():
        assert p["m"] == pts[name.lower()] and p["whys"] == [] and p["missing"] == []
        assert p["gamma"]["gamma"] == pytest.approx(0.55, rel=1e-4) and p["gamma"]["n"] == 20
        assert p["gamma_least_sm"]["gamma"] == pytest.approx((1.5 + 0.15 - 0.08) / 0.8 - 1, rel=1e-4)
        assert p["gamma_weight_side_bound"]["gamma"] == pytest.approx(0.66, rel=1e-4)
        assert p["aa"]["gamma"] == pytest.approx(0.0, abs=1e-4)
        assert p["gamma_occupancy"]["gamma"] == pytest.approx(0.55, rel=1e-4)
        assert {v["stock_fastest"]["kernel"] for v in p["linears"].values()} == {"k_fast"}
        assert {v["stock_least_sm"]["kernel"] for v in p["linears"].values()} == {"k_lean"}
        assert all(v["accepted"] for v in p["verified"].values()) and len(p["verified"]) == len(LINEARS)
        assert p["epsilon"]["gate_up"] == {"epsilon": 2.0 ** -40 * 7, "delta": 1e-9, "basis": "stand-in"}
        assert p["gate"] == {"gamma_max": O.GAMMA_MAX, "passes": True} and p["perplexity"]["value"] is None
        fast = p["linears"]["o"]["stock_fastest"]
        assert fast["sm_s"] == pytest.approx(pts[name.lower()] * 8192 * 8192 * 1e-12 * 1.02, rel=1e-3)
        assert fast["cpu_sm_s"] == pytest.approx(HOST * fast["sm_s"], rel=1e-3)
    sent, sha = str(run_dir / "inputs" / "arm.cubin"), hashlib.sha256(CUBIN).hexdigest()
    assert doc["arm"] == {"name": ARM, "cubin_art": CUBIN_ART, "cubin_sha256": sha, "binaries": [sent],
                          "verifier": "verify.py", "gated": [{"path": sent, "sha256": sha, "ok": True}],
                          "commits_y": False}
    prof, *exports, verify = node.calls
    assert prof[:2] == ["env", f"{overhead_node.CUBIN_ENV}={sent}"]
    prof = prof[2:]
    assert prof[:len(overhead_node.NSYS_PROFILE) + 1] == ["nsys", *overhead_node.NSYS_PROFILE]
    assert "--capture-range=cudaProfilerApi" in prof
    bench = prof[prof.index("-o") + 2:]
    assert {"--traced", "--twins", "--bridge"} <= set(bench) and between(bench, "--families") == ["fp8-e4m3-rowwise"]
    assert between(bench, "--methods") == ["graph", "burst"] and between(bench, "--arm") == [scheme.arm]
    assert len(exports) == 1 + 2 * len(LINEARS) == len(doc["reports"]) and all(e["exit"] == 0 for e in doc["reports"])
    assert doc["reports"][0] == {"report": "prof.1.nsys-rep", "exit": 0, "items": 0, "stray_kernels": 0}
    assert {(e["items"], e["stray_kernels"]) for e in doc["reports"][1:]} == {(5 * 20, 0)}
    assert doc["traced_prime"]["ready"]
    assert all(e[1] == "export" and e[e.index("-o") + 1].startswith(str(tmp_path / "scratch")) for e in exports)
    assert verify[2:5] == [str(run_dir / "out" / "bench.json"), "--every-row", "--no-ledger"] and SHA in verify
    assert not list((run_dir / "out").glob("*.nsys-rep")) and not list((run_dir / "out").glob("*.sqlite"))
    result = json.loads((run_dir / "result.json").read_text())
    assert result["validation"]["status"] == "passed"
    written = {m["name"]: m for m in result["measurements"]}
    assert set(written) == {"benchmarks/overhead.gamma-c", "benchmarks/overhead.gamma-l"}
    for name, p in doc["points"].items():
        m = written[f"benchmarks/overhead.gamma-{name.lower()}"]
        assert (m["value"], m["ci95"], m["unit"]) == (p["gamma"]["gamma"], p["gamma"]["ci95"], "ratio")
        assert (m["target"], m["verdict"], m["mark"]) == (None, None, "measured")
        assert m["inputs"] == ["measured", "out/overhead.json"]
        assert (m["run"], m["commit"], m["host"]) == (run_dir.name, SHA, STAMP)


def test_a_profiling_pass_keeps_its_reports_in_out_and_is_not_of_record(tmp_path, scheme):
    points = node_points(tmp_path, Node(tmp_path))
    rc, run_dir = node_run(tmp_path, Node(tmp_path), points, "pearl-plus", "art:" + "b" * 64, extra=["--keep-reports"])
    doc = json.loads((run_dir / "out" / "overhead.json").read_text())
    kept = run_dir / "out" / "nsys"
    assert rc == 0 and doc["whys"] == [overhead_node.KEEP_REPORTS_WHY]
    assert [r["report"] for r in doc["reports"]] == sorted(p.name for p in kept.glob("*.nsys-rep")) != []
    assert not list((tmp_path / "scratch").rglob("*.nsys-rep"))
    assert all(p["gamma"]["gamma"] == pytest.approx(0.55, rel=1e-4) for p in doc["points"].values())


def test_stocks_gamma_is_the_aa_control_of_its_twinned_kernel_against_its_original(tmp_path):
    points = node_points(tmp_path, Node(tmp_path))
    node = Node(tmp_path)
    rc, run_dir = node_run(tmp_path, node, points, "stock", "art:" + "b" * 64)
    doc = json.loads((run_dir / "out" / "overhead.json").read_text())
    assert rc == 0 and len(node.calls) == 2 + 2 * len(LINEARS) and "--arm" not in node.calls[0]
    for p in doc["points"].values():
        assert p["gamma"] == p["aa"] and p["gamma"]["gamma"] == pytest.approx(0.0, abs=1e-4) and p["whys"] == []
        assert p["epsilon"] == {"why": "stock has no ε: it is the A/A control"} and p["verified"] == {}


def test_a_timed_run_says_why_each_point_isnt_of_record(tmp_path, scheme):
    """Too few reps, a rejected row, a noisy arm: each point carries why, and the run still records its γ."""

    points = node_points(tmp_path, Node(tmp_path))
    rc, run_dir = node_run(tmp_path, Node(tmp_path, reps=12, accept=False, spread=0.2), points, "pearl-plus")
    doc = json.loads((run_dir / "out" / "overhead.json").read_text())
    assert rc == 0
    for p in doc["points"].values():
        whys = " ".join(p["whys"])
        assert "12 reps, fewer than 20" in whys and "didn't accept" in whys
    rc, run_dir = node_run(tmp_path, Node(tmp_path, spread=0.2), points, "pearl-plus", run="r20261104-030000-0003")
    assert all("wider than ±5%" in " ".join(p["whys"]) for p in
               json.loads((run_dir / "out" / "overhead.json").read_text())["points"].values())
    rc, run_dir = node_run(tmp_path, Node(tmp_path), points, "pearl-plus", run="r20261104-030000-0004", cubin=None)
    assert json.loads((run_dir / "out" / "overhead.json").read_text())["whys"] == [
        "the run was sent no cubin art (`run --cubin ART`): a cubin built in the run is development only"]
    rc, run_dir = node_run(tmp_path, Node(tmp_path, loads=False), points, "pearl-plus", run="r20261104-030000-0005")
    assert json.loads((run_dir / "out" / "overhead.json").read_text())["whys"] == [
        (f"no binary the SASS gate passed is the cubin the run was sent ({CUBIN_ART}): the arm must load "
         "$VERITY_BENCH_CUBIN and list it in its binaries")]


def test_a_timed_run_refuses_a_capture_range_that_opened_after_its_items(tmp_path, scheme):
    """As nsys does when a start outlasts its wait: the first shape's items in no report, and the next report collected
    from then to its own stop, the next shape's timed pass included."""

    points = node_points(tmp_path, Node(tmp_path))
    rc, run_dir = node_run(tmp_path, Node(tmp_path, late=True), points, "pearl-plus")
    doc = json.loads((run_dir / "out" / "overhead.json").read_text())
    assert rc == 1 and doc["whys"] == [
        (f"prof.2.nsys-rep traced {STRAY} kernels outside its 100 traced items, more than a flush each: its capture range "
         "ran past the traced pass, so a timed pass may have run traced"),
        "nsys recorded 700 of the 800 traced items (item 0 the first missing): a capture range opened after its items"]


def test_a_timed_run_refuses_points_that_arent_a_finished_sweep_of_its_card(tmp_path):
    bad = tmp_path / "points.json"
    bad.write_text(json.dumps({"schema": O.POINTS_SCHEMA, "card": "rtx-pro-6000", "c": 512, "l": 64,
                               "families": list(O.FP8), "model": O.KERNEL_MODEL}))
    node = Node(tmp_path)
    rc, run_dir = node_run(tmp_path, node, bad, "stock")
    assert rc == 1 and node.calls == []
    assert "isn't a finished" in json.loads((run_dir / "result.json").read_text())["validation"]["detail"]


# ---- phases ------------------------------------------------------------------------------------------------------------------

POINTS_FIELDS = {"schema", "card", "model", "precision", "families", "rule", "rounds", "identity", "c", "l",
                 "best_flop_per_ms", "times_ms", "tflops", "linears", "error"}
RUN_FIELDS = {"schema", "scheme", "family", "card", "model", "points_art", "points_sha256", "cpu_sm_s_per_core_s", "points",
              "whys", "bench_exit", "identity", "clocks", "bridge", "sass_gate_ok", "arm", "reports", "traced_prime",
              "verify_exit"}


@pytest.fixture
def ticks(monkeypatch):
    """A clock a second a reading, from 2026-11-18T00:00:00Z."""

    t = iter(range(1_794_960_000, 1_794_960_000 + 10_000))
    monkeypatch.setattr(overhead_node, "CLOCK", lambda: float(next(t)))


def test_a_sweep_records_its_phases_beside_every_field_it_had(tmp_path, ticks, capsys):
    node = Node(tmp_path)
    out, run_dir = tmp_path / "r1" / "out", tmp_path / "r1"
    assert overhead_node.main(["points", "--card", "h200", "--uuid", UUID, "--out", str(out), "--run-dir", str(run_dir),
                               "--run-id", "r1", "--build-s", "12.5"], run=node) == 0
    points = json.loads((out / "points.json").read_text())
    assert set(points) == POINTS_FIELDS | {"phases"}
    rounds = [f"round {i:02d} (m = {', '.join(map(str, r['ms']))})" for i, r in enumerate(points["rounds"])]
    assert points["phases"] == [{"name": "build", "seconds": 12.5}, *({"name": n, "seconds": 1.0} for n in rounds)]
    log = capsys.readouterr().out
    assert f"2026-11-18T00:00:00Z phase {rounds[0]} start" in log and f"2026-11-18T00:00:01Z phase {rounds[0]} end" in log


def test_a_timed_run_records_its_phases_beside_every_field_it_had(tmp_path, scheme, ticks):
    points = node_points(tmp_path, Node(tmp_path))
    rc, run_dir = node_run(tmp_path, Node(tmp_path), points, "pearl-plus", "art:" + "b" * 64, extra=["--build-s", "3"])
    doc = json.loads((run_dir / "out" / "overhead.json").read_text())
    assert rc == 0 and set(doc) == RUN_FIELDS | {"phases"}
    exports = [f"nsys export {i:02d} ({r['report']})" for i, r in enumerate(doc["reports"])]
    assert [p["name"] for p in doc["phases"]] == ["build", "bench.py under nsys", *exports, "verify.py, every row",
                                                  "γ at each point"]
    assert doc["phases"][0] == {"name": "build", "seconds": 3.0} and {p["seconds"] for p in doc["phases"][1:]} == {1.0}


def test_a_refused_run_records_the_phases_it_ran(tmp_path):
    bad = tmp_path / "points.json"
    bad.write_text(json.dumps({"schema": O.POINTS_SCHEMA, "card": "rtx-pro-6000", "c": 512, "l": 64}))
    rc, run_dir = node_run(tmp_path, Node(tmp_path), bad, "stock", extra=["--build-s", "2"])
    assert rc == 1 and json.loads((run_dir / "out" / "overhead.json").read_text())["phases"] == [
        {"name": "build", "seconds": 2.0}]


def test_a_bench_process_is_split_by_its_own_stages_and_what_is_left_outside_its_shapes():
    phases = overhead_node.Phases(None, clock=iter([0.0, 100.0]).__next__)
    with phases("round 00"):
        pass
    phases.bench("round 00", {"shapes": [{"wall_s": {"warmup": 10.0, "timed": 30.0}},
                                         {"wall_s": {"warmup": 5.0, "timed": 25.0, "traced": 5.0}}]})
    assert phases.list == [{"name": "round 00", "seconds": 100.0}, {"name": "round 00: warmup", "seconds": 15.0},
                           {"name": "round 00: timed", "seconds": 55.0}, {"name": "round 00: traced", "seconds": 5.0},
                           {"name": "round 00: outside its shapes", "seconds": 25.0}]


def test_the_runner_keeps_the_log_as_written_and_stamps_each_shape_it_starts(tmp_path, ticks, capsys):
    log = tmp_path / "round" / "bench.log"
    log.parent.mkdir()
    rc = overhead_node._run([sys.executable, "-u", "-c", "import sys; print('sass gate: pass'); print('shape m1-n2-k3', "
                             "file=sys.stderr, flush=True); print('exit 0')"], log)
    assert rc == 0 and log.read_text().splitlines() == ["sass gate: pass", "shape m1-n2-k3", "exit 0"]
    assert capsys.readouterr().out == "2026-11-18T00:00:00Z round/bench.log: shape m1-n2-k3\n"
