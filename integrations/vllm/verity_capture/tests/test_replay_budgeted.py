"""`replay_budgeted` on the synthetic toy Program (fixtures as in `test_replay_synthetic`): the budgeted experiment reuses
`Replayer.evaluate_instance` / `evaluate_child` case by case (identical entries), pools the T-indexed attention family under a
budget with boundary cases first, keeps tier A's k-per-stratum rule for every other family, reports unrecorded expected values as
`unrecorded` (never substituted), is deterministic per seed and JSON-serialisable."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_capture import compare as cmp
from verity_capture import replay
from verity_capture import replay_budgeted as rb
from verity_capture.fold import fold
from verity_capture.tests import test_fold_synthetic as tfs
from verity_capture.tests.test_compare_synthetic import ToyWeights, _evaluate_program, _write_snapshots

REQ = tfs.REQ


def _computed_tokens(r, weights) -> list[int]:
    t = _evaluate_program(r, weights)
    return [int(t[g]) for g in r.program.circuit.root_outputs()]


@pytest.fixture(scope="module")
def consistent_toy():
    if not tfs._real_memory_available():
        tfs.install_stub_memory()
    original = list(tfs.SAMPLED)
    try:
        r = fold(tfs.toy_log(), tfs.toy_profile(), strict=True)
        weights = ToyWeights(r.params[-1][1].leaves)
        t0 = _computed_tokens(r, weights)[0]
        tfs.SAMPLED[:] = [t0, original[1]]
        r = fold(tfs.toy_log(), tfs.toy_profile(), strict=True)
        t1 = _computed_tokens(r, weights)[1]
        tfs.SAMPLED[:] = [t0, t1]
        r = fold(tfs.toy_log(), tfs.toy_profile(), strict=True)
        t = _evaluate_program(r, weights)
        yield r, weights, t
    finally:
        tfs.SAMPLED[:] = original


@pytest.fixture(scope="module")
def snap_dir(consistent_toy, tmp_path_factory) -> Path:
    r, _, t = consistent_toy
    d = tmp_path_factory.mktemp("snaps")
    _write_snapshots(r, t, d)
    return d


@pytest.fixture(scope="module")
def snapshots(snap_dir):
    return cmp.SnapshotStore(snap_dir)


def _attention_cases(rep: dict) -> list[dict]:
    fam = next(f for f in rep["families"] if f.startswith("Attention"))
    return rep["families"][fam]["cases"]


def test_budgeted_passes_and_reuses_tier_a_evaluation(consistent_toy, snapshots):
    r, weights, _ = consistent_toy
    rep = rb.run_budgeted(r, snapshots, weights, budget=4, buckets=2, k=2, seed=0)
    assert rep["status"] == "pass", rep.get("status_reason")
    assert rep["version"] == rb.BUDGETED_VERSION and rep["schema"] == rb.SCHEMA and rep["evaluations"] > 0
    assert all(e["status"] == "pass" for f in rep["families"].values() for e in f["cases"])
    # each case IS Replayer.evaluate_instance's result for that instance (same procedure, same numbers)
    ref = replay.Replayer(r, snapshots, weights, chain=False)
    for f in rep["families"].values():
        for e in f["cases"]:
            m = ref.by_node[e["node"]]
            direct = ref.evaluate_instance(m)
            assert (e["status"], e["n_diff"], e["n_returns"], e["n_args"], e["granularity"]) == \
                   (direct["status"], direct["n_diff"], direct["n_returns"], direct["n_args"], direct["granularity"]), e["name"]
    # the attention family is pooled (T in the spec id); every other family keeps tier A's k-per-(spec, regime) count
    fams = rep["families"]
    att = [f for f in fams if f.startswith("Attention")]
    assert att and fams[att[0]]["pooled"] is True and set(rep["plan"]["pooled_families"]) == set(att)
    for f, v in fams.items():
        if f in att:
            assert v["evaluations"] <= 4 and v["budget"] == 4
        else:
            assert v["pooled"] is False and v["evaluations"] == v["tier_a_equivalent_evaluations"] > 0, (f, v)
    assert rep["tier_a_equivalent_evaluations"] >= rep["evaluations"]
    assert rep["limitations"]["unrecorded_cases"] == 0 and rep["unrecorded"] == []


def test_boundary_cases_first_and_budget_respected(consistent_toy, snapshots):
    r, weights, _ = consistent_toy
    Ts = sorted({rb.T_of(m["spec"]) for m in r.instances if m["spec"].startswith("Attention")})
    assert len(Ts) >= 2
    rep = rb.run_budgeted(r, snapshots, weights, budget=2, buckets=3, k=1, seed=5)
    cases = _attention_cases(rep)
    assert len(cases) == 2
    whys = {e["case"]["why"]: e["case"]["T"] for e in cases}
    assert whys == {"boundary:min_T": Ts[0], "boundary:max_T": Ts[-1]}
    fam = next(f for f in rep["families"] if f.startswith("Attention"))
    lim = rep["families"][fam]["specs_not_evaluated"]
    assert lim["count"] == len(Ts) - 2 and set(lim["T_values_not_evaluated"]) == set(Ts[1:-1])
    assert rep["limitations"]["specs_not_evaluated"][fam] == lim
    # a larger budget fills the strata round-robin and reports every evaluated T
    big = rb.run_budgeted(r, snapshots, weights, budget=100, buckets=3, k=1, seed=5)
    fam_b = big["families"][fam]
    assert fam_b["evaluations"] == fam_b["instances_at_snapshot_steps"] and fam_b["evaluated_T"] == Ts
    assert fam_b["specs_not_evaluated"]["count"] == 0
    assert {e["case"]["bucket"] for e in fam_b["cases"]} <= set(range(len(fam_b["T"]["buckets"])))
    assert all(e["case"]["why"].startswith(("boundary:", "stratified:")) for e in fam_b["cases"])


def test_new_specializations_before_repeats_across_strata(consistent_toy, snapshots):
    """v0.1.2: a stratified draw repeats a specialization only when every attention specialization at a snapshot step has
    already been drawn (by a boundary or a stratified case) -- a thin stratum never spends the budget on repeats while another
    stratum still holds an unvisited specialization.  Checked for every budget on the toy."""
    r, weights, _ = consistent_toy
    avail = {m["spec"] for m in r.instances if m["spec"].startswith("Attention") and m["step"] in snapshots.steps}
    assert len(avail) >= 2
    for B in range(1, len(avail) + 4):
        cases = _attention_cases(rb.run_budgeted(r, snapshots, weights, budget=B, buckets=2, k=1, seed=3))
        assert len(cases) == min(B, sum(1 for m in r.instances if m["spec"].startswith("Attention") and m["step"] in snapshots.steps))
        drawn: set[str] = set()
        for c in cases:
            if c["case"]["why"].startswith("stratified:") and c["spec"] in drawn:
                assert drawn == avail, (B, c["spec"], sorted(avail - drawn))
            drawn.add(c["spec"])
        if len(cases) >= len(avail) + 2:      # at most 2 boundary duplicates (min_T/first_step, max_T/last_step) -> every spec drawn
            assert drawn == avail, (B, sorted(avail - drawn))


def test_unrecorded_expected_value_is_reported_not_substituted(consistent_toy, snap_dir):
    r, weights, _ = consistent_toy
    store = cmp.SnapshotStore(snap_dir)
    plan = rb.select_cases(r, sorted(store.steps), budget=2, buckets=2, k=1, seed=0)
    fam = next(f for f in plan["families"] if f.startswith("Attention"))
    victim = plan["families"][fam]["cases"][0]
    ev = next(m for m in r.instances if m["node"] == victim["node"])["event"]
    store.by_event.pop(ev)                       # the case's own output snapshot was never recorded
    rep = rb.run_budgeted(r, store, weights, budget=2, buckets=2, k=1, seed=0)
    e = next(c for c in rep["families"][fam]["cases"] if c["node"] == victim["node"])
    assert e["status"] == "unrecorded" and "no snapshot" in e["reason"] and "n_diff" not in e
    # the event's snapshots also fed downstream operands: every case that needed them is `unrecorded` too, none is "pass"
    unrec = rep["unrecorded"]
    assert rep["status"] == "partial" and rep["limitations"]["unrecorded_cases"] == len(unrec) >= 1
    assert victim["node"] in {u["node"] for u in unrec} and all("no snapshot" in u["reason"] for u in unrec)
    assert rep["summary"].get("unrecorded") == len(unrec) and rep["summary"].get("pass", 0) >= 1
    all_cases = [c for v in rep["families"].values() for c in v["cases"]]
    assert {c["status"] for c in all_cases} == {"pass", "unrecorded"}
    # the same cases with the complete evidence all pass: nothing else changed
    full = rb.run_budgeted(r, cmp.SnapshotStore(snap_dir), weights, budget=2, buckets=2, k=1, seed=0)
    assert full["status"] == "pass" and [c["node"] for v in full["families"].values() for c in v["cases"]] == [c["node"] for c in all_cases]


def test_child_units_for_oversize_instances(consistent_toy, snapshots):
    r, weights, _ = consistent_toy
    rep = rb.run_budgeted(r, snapshots, weights, budget=3, buckets=2, k=2, seed=3, max_arg_leaves=100)
    assert rep["status"] == "pass", rep.get("status_reason")
    gran = {f: {e["granularity"] for e in v["cases"]} for f, v in rep["families"].items()}
    assert gran["Embedding_v1"] == {"child:GatherBf16x16_v1"}
    assert gran["Gemm_v1"] == {"child:GemmCoordinate_v1{K=16}"}
    assert gran["RMSNormFusedCuda_v2"] == {"instance"}
    assert all(e["n_returns"] == 1 for f in ("Embedding_v1", "Gemm_v1") for e in rep["families"][f]["cases"])


def test_deterministic_json_and_summary(consistent_toy, snapshots):
    r, weights, _ = consistent_toy
    a = rb.run_budgeted(r, snapshots, weights, budget=3, buckets=2, k=2, seed=11)
    b = rb.run_budgeted(r, snapshots, weights, budget=3, buckets=2, k=2, seed=11)
    strip = lambda rep: [(f, e["node"], e["status"], e["case"]["why"]) for f, v in rep["families"].items() for e in v["cases"]]
    assert strip(a) == strip(b)
    c = rb.run_budgeted(r, snapshots, weights, budget=3, buckets=2, k=2, seed=12)
    assert {x[1] for x in strip(c)} != set() and strip(c) != [] and all(x[2] == "pass" for x in strip(c))
    s = json.dumps(cmp._jsonable(a))
    d = json.loads(s)
    assert d["status"] == "pass" and d["not_a_gate"].startswith("budgeted experiment") and "limitations" in d
    line = rb.summary_line(a)
    assert "budgeted v" in line and "tier-A-equivalent" in line
    # a wall budget of zero evaluates nothing and says so
    z = rb.run_budgeted(r, snapshots, weights, budget=3, buckets=2, k=2, seed=11, budget_s=0.0)
    assert z["status"] == "skipped" and z["limitations"]["stopped_early"]


def test_t_buckets():
    assert rb.t_buckets([5, 1, 3, 3], 2) == [(1, 1), (3, 5)]          # 3 distinct T, floor split: 1 + 2
    assert rb.t_buckets([1, 2, 3, 4, 5, 6], 3) == [(1, 2), (3, 4), (5, 6)]
    assert rb.t_buckets([7], 4) == [(7, 7)]
    assert rb.t_buckets([], 4) == []
    assert rb.bucket_of(4, [(1, 2), (3, 4), (5, 6)]) == 1
    assert rb.parse_steps("0-2,5") == [0, 1, 2, 5] and rb.parse_steps("all") is None and rb.parse_steps(None) is None


def test_memory_preflight(tmp_path, monkeypatch):
    """v0.1.3: the run's own fold peak + 3.3 x snapshots.json is compared with a known limit; over -> `skipped` report, exit 2,
    nothing folded; `--force` or an unknown limit -> the projection is only recorded.  Numbers are the measured C1024 case."""
    run = tmp_path / "run"
    (run / "capture").mkdir(parents=True)
    (run / "fold_summary.json").write_text(json.dumps({"max_rss_gb": 52.694}))
    idx = run / "capture" / "snapshots.json"
    idx.write_bytes(b"\0" * 3_000_000)                       # 3 MB stands in for the 3.02 GB (ratio-only arithmetic)
    pre = rb.preflight_memory(run, run / "capture", limit_gb=61.0)
    assert pre["fold_peak_gb_legacy"] == 52.694 and pre["verdict"] == "fits"          # 52.7 + 0.01 < 61 with a 3 MB index
    pre = rb.preflight_memory(run, run / "capture", limit_gb=52.0)
    assert pre["verdict"] == "over"
    assert rb.preflight_memory(run, run / "capture", limit_gb=None)["verdict"] == "no limit known"
    assert rb.preflight_memory(None, tmp_path, limit_gb=61.0)["verdict"] == "unknown"      # no records at all
    # the 3.3x ratio itself, at the C256 index size the ratio was measured on
    big = rb.preflight_memory(None, run / "capture", limit_gb=None)
    assert abs(big["projected_peak_gb"] - 0.003 * rb.INDEX_TRANSIENT_PER_BYTE) <= 0.005      # projection is rounded to 0.01 GB
    # main(): over the limit -> exit 2 + skipped report, without touching the (absent) log
    monkeypatch.setenv("COV_MEM_LIMIT_GB", "52")
    out = tmp_path / "r.json"
    rc = rb.main(["--log", str(run / "capture" / "missing.jsonl.gz"), "--snapshots", str(run / "capture"), "--profile", "x",
                  "--run", str(run), "--out", str(out)])
    assert rc == 2
    rep = json.loads(out.read_text())
    assert rep["status"] == "skipped" and rep["evaluations"] == 0 and rep["preflight"]["verdict"] == "over"
    assert "52" in rep["reason"] and rep["version"] == rb.BUDGETED_VERSION
    assert rb.cgroup_memory_limit_gb() == 52.0
    monkeypatch.setenv("COV_MEM_LIMIT_GB", "not-a-number")
    assert rb.cgroup_memory_limit_gb() is None or isinstance(rb.cgroup_memory_limit_gb(), float)   # falls through to the cgroup files


def test_filter_snapshot_index_matches_full_store(consistent_toy, snap_dir, snapshots, tmp_path):
    """v0.1.3 `--index-steps`: the streamed filter keeps exactly the entries of the wanted steps, the unchanged SnapshotStore
    opened on the filtered dir serves them identically (same Snapshot fields, same words), and a budgeted run on it samples
    only the kept steps and still passes (every operand producer of a step-0 case lives at step 0)."""
    r, weights, _ = consistent_toy
    full = snapshots
    assert len(full.steps) >= 2
    keep = {full.steps[0]}
    dst = tmp_path / "index_0"
    info = rb.filter_snapshot_index(snap_dir, keep, dst, chunk=4096)      # tiny chunk: exercises the refill paths
    assert info["entries_total"] == len(full.entries) + len(full.skipped)
    assert info["entries_kept"] == sum(1 for s in full.entries if s.step in keep) + sum(1 for e in full.skipped if e.get("step") in keep or e.get("step") is None)
    part = cmp.SnapshotStore(dst)
    assert part.steps == sorted(keep) and part.meta == full.meta
    want = {(s.event, s.arg): s for s in full.entries if s.step in keep}
    got = {(s.event, s.arg): s for s in part.entries}
    assert got.keys() == want.keys()
    for key, s in got.items():
        w = want[key]
        assert (s.view, s.step, s.spanned, s.lo, s.sha256) == (w.view, w.step, w.spanned, w.lo, w.sha256)
        assert s.file.resolve() == w.file.resolve() and (part.words(s) == full.words(w)).all()
    rep = rb.run_budgeted(r, part, weights, budget=3, buckets=2, k=2, seed=0)
    assert rep["snapshot_steps"] == sorted(keep)
    cases = [e for v in rep["families"].values() for e in v["cases"]]
    assert cases and all(e["step"] in keep for e in cases) and all(e["status"] == "pass" for e in cases)
    assert rep["status"] == "pass"
    # unsupported layout -> ValueError (the CLI then falls back to the full index)
    odd = tmp_path / "odd"
    odd.mkdir()
    (odd / "snapshots.json").write_text("[]")
    with pytest.raises(ValueError):
        rb.filter_snapshot_index(odd, keep, tmp_path / "odd_out")
