"""Pure-Python tests of the telemetry package (run anywhere with Python 3.11+; the Linux-only fixtures live in fixtures/run_fixtures.sh).

    uv run pytest tools/research/tests -q      (from the verity root)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent / "src"          # tools/research/src (research is installed; kept for the subprocess PYTHONPATH below)

from research.telemetry import CLASSIFIER_VERSION, events, job, policy  # noqa: E402
from research.telemetry.classify import classify  # noqa: E402

# The r17 campaign's RULES.md seed lives in veritor and keeps moving; compare against it only when asked to.
SEED = Path(os.environ["RESEARCH_SEED_RULES"]) if os.environ.get("RESEARCH_SEED_RULES") else None


# ---- (b) the append helper ------------------------------------------------------------------------------------------------------------------

def _writer(path: str, n: int, tag: int) -> None:
    for i in range(n):
        events.append(path, {"kind": "mark", "stage": "t", "name": f"w{tag}", "i": i, "pad": "x" * (100 + (i * 37) % 900)})


def test_concurrent_appends_from_many_processes_never_interleave(tmp_path):
    p = tmp_path / "events.jsonl"
    ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context()
    procs = [ctx.Process(target=_writer, args=(str(p), 300, t)) for t in range(6)]
    for x in procs:
        x.start()
    for x in procs:
        x.join()
    rep = events.ReadReport()
    recs = list(events.iter_records(p, rep))
    assert len(recs) == 1800 and rep.malformed == 0 and not rep.partial
    streams = {r["stream"] for r in recs}
    assert len(streams) == 6, "per-process stream ids"
    for s in streams:   # per-stream seq strictly increasing in file order
        seqs = [r["seq"] for r in recs if r["stream"] == s]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_oversized_record_truncates_fields_not_bytes_and_is_not_reader_partial(tmp_path):
    p = tmp_path / "events.jsonl"
    rec = events.append(p, {"kind": "mark", "stage": "s", "name": "big", "small": 1, "blob": "y" * 10000, "blob2": "z" * 10000})
    assert rec["original_bytes"] > events.MAX_RECORD_BYTES
    assert rec["kind"] == "mark" and rec["stage"] == "s" and rec["name"] == "big" and rec["small"] == 1, "identity + small fields survive"
    assert set(rec["truncated_fields"]) >= {"blob2"} and len(json.dumps(rec)) <= events.MAX_RECORD_BYTES
    rep = events.read_all(tmp_path, "events")
    assert rep.partial == [] and rep.records[0]["truncated_fields"], "writer-side truncation is a complete line, not a reader-side partial"


def test_truncated_last_line_is_reported_as_partial_and_prior_records_survive(tmp_path):
    p = tmp_path / "events.jsonl"
    for i in range(3):
        events.append(p, {"kind": "mark", "stage": "s", "name": str(i)})
    with open(p, "ab") as f:
        f.write(b'{"kind": "mark", "stage": "s", "na')   # the writer died mid-record
    rep = events.ReadReport()
    recs = list(events.iter_records(p, rep))
    assert [r["name"] for r in recs] == ["0", "1", "2"]
    assert len(rep.partial) == 1 and "truncated last line" in rep.partial[0]["reason"] and rep.malformed == 0
    # a malformed INNER line is counted, not fatal
    with open(p, "ab") as f:
        f.write(b'\n{"kind": "mark", "stage": "s", "name": "4"}\n')
    rep2 = events.ReadReport()
    recs2 = list(events.iter_records(p, rep2))
    assert [r["name"] for r in recs2] == ["0", "1", "2", "4"] and rep2.malformed == 1 and not any("truncated" in x["reason"] for x in rep2.partial)


def test_monotonic_durations_are_refused_across_hosts():
    a = {"stream": "hostA:1:2", "t_mono": 10.0}
    b = {"stream": "hostB:1:2", "t_mono": 20.0}
    assert events.duration(a, b) is None
    assert events.duration(a, {"stream": "hostA:9:9", "t_mono": 12.5}) == 2.5


# ---- (h) nested / repeated / overlapping spans and holds ------------------------------------------------------------------------------------------

def _span(kind, sid, stage, t, parent=None, stream="h:1:1", **kv):
    return {"kind": kind, "id": sid, "stage": stage, "parent": parent, "t_mono": t, "t_utc": f"T{t:08.3f}", "stream": stream, **kv}


def test_span_accounting_nested_repeated_overlapping_and_open():
    recs = [
        _span("span_start", "row", "row", 0.0),
        _span("span_start", "load1", "match.load", 1.0, parent="row"),
        _span("span_end", "load1", "match.load", 3.0, parent="row", outcome="ok"),
        _span("span_start", "load2", "match.load", 3.0, parent="row"),          # repeated stage name, distinct invocation
        _span("span_end", "load2", "match.load", 4.0, parent="row", outcome="ok"),
        _span("span_start", "fold", "match.fold", 4.0, parent="row"),
        _span("span_start", "x09", "match.x09", 5.0, parent="row"),             # overlaps fold (parallel legs)
        _span("span_end", "fold", "match.fold", 8.0, parent="row", outcome="ok"),
        _span("span_end", "x09", "match.x09", 9.0, parent="row", outcome="ok"),
        _span("span_start", "gm", "match.gm", 9.0, parent="row"),               # dies open
        {"kind": "mark", "stage": "match.gm", "name": "last", "t_mono": 9.5, "t_utc": "T9.5", "stream": "h:1:1"},
        _span("span_end", "row", "row", 10.0, outcome="fail", rc=137),
    ]
    acc = events.spans(recs)
    by = {s["id"]: s for s in acc["spans"]}
    assert by["row"]["inclusive_s"] == 10.0
    # children: load1 [1,3], load2 [3,4], fold [4,8], x09 [5,9], gm [9, last obs 9.5 ... until row end 10) -> union = [1,10) = 9 -> exclusive 1
    assert by["row"]["exclusive_s"] == pytest.approx(1.0)
    assert by["gm"]["outcome"] == "interrupted" and by["gm"]["end_mono"] is None and by["gm"]["observed_at_least_s"] == pytest.approx(1.0)
    assert acc["by_stage"]["match.load"]["calls"] == 2 and acc["by_stage"]["match.load"]["inclusive_sum_s"] == 3.0
    # overlapping siblings are not summed into elapsed: fold+x09 summed 8, union 5
    assert acc["total_union_s"] == 10.0 and acc["overlap_s"] == 0.0        # one root
    inner = events.spans([r for r in recs if r["id"] in ("fold", "x09") and r["kind"].startswith("span")] if False else
                         [dict(r, parent=None) for r in recs if r.get("id") in ("fold", "x09")])
    assert inner["total_union_s"] == 5.0 and inner["overlap_s"] == 3.0


def test_holds_are_intervals_whose_union_is_the_waiting_time():
    recs = [
        {"kind": "hold_start", "id": "h1", "reason": "release", "t_mono": 0.0, "t_utc": "a", "stream": "h:1:1"},
        {"kind": "hold_start", "id": "h2", "reason": "gpu_reservation", "t_mono": 5.0, "t_utc": "b", "stream": "h:1:1"},
        {"kind": "hold_end", "id": "h1", "t_mono": 10.0, "t_utc": "c", "stream": "h:1:1"},
        {"kind": "hold_end", "id": "h2", "t_mono": 12.0, "t_utc": "d", "stream": "h:1:1"},
        {"kind": "hold_start", "id": "h3", "reason": "manual", "t_mono": 20.0, "t_utc": "e", "stream": "h:1:1"},   # still open
    ]
    h = events.holds(recs, end_mono_default=25.0)
    assert h["waiting_union_s"] == pytest.approx(12.0 + 5.0)
    assert h["by_reason_summed_s"] == {"release": 10.0, "gpu_reservation": 7.0}
    assert [x["open"] for x in h["holds"]] == [False, False, True]


def test_emitter_and_cli_event_round_trip(tmp_path):
    em = events.Emitter(tmp_path, proc="t")
    em.queue("submitted")
    hid = em.hold_start("release", detail="RELEASING marker")
    em.hold_end(hid)
    em.queue("admitted")
    em.queue("started")
    sid = em.span_start("match.load")
    em.span_end("match.load", sid, outcome="ok", rc=0, refs=["match/program.json"])
    em.hold_start("not-a-reason")            # unknown reason -> other + reason_detail
    rep = events.read_all(tmp_path)
    kinds = [r["kind"] for r in rep.records]
    assert kinds == ["submitted", "hold_start", "hold_end", "admitted", "started", "span_start", "span_end", "hold_start"]
    assert rep.records[-1]["reason"] == "other" and rep.records[-1]["reason_detail"] == "not-a-reason"
    q = events.queue_timeline(rep.records)
    assert q["submission_to_admission_s"] is not None and q["admission_to_start_s"] is not None
    # shell surface
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=str(PKG_ROOT))
    out = subprocess.run([py, "-m", "research.telemetry", "event", "--attempt-dir", str(tmp_path), "span-start", "--stage", "commit.c1", "--stages-txt", str(tmp_path / "stages.txt")],
                         capture_output=True, text=True, env=env, check=True).stdout.strip()
    subprocess.run([py, "-m", "research.telemetry", "event", "--attempt-dir", str(tmp_path), "span-end", "--stage", "commit.c1", "--id", out, "--rc", "0", "k=1", "--stages-txt", str(tmp_path / "stages.txt")],
                   check=True, env=env)
    acc = events.spans(events.read_all(tmp_path).records)
    assert acc["by_stage"]["commit.c1"]["calls"] == 1 and acc["by_stage"]["commit.c1"]["open"] == 0
    lines = (tmp_path / "stages.txt").read_text().splitlines()
    assert [l.split()[1] for l in lines] == ["START", "END"] and all(l.startswith("commit.c1 ") and l.endswith(f"id={out}") for l in lines)


# ---- (a) job.json --------------------------------------------------------------------------------------------------------------------------------

def test_argv_redaction_keeps_shape_and_hides_secrets():
    argv = ["bash", "row_pod.sh", "row", "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz0123", "--api-key", "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345", "--password=pw", "PLAIN=1",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"]
    red, names = job.redact_argv(argv)
    assert red[:3] == argv[:3] and red[3] == "HF_TOKEN=<redacted>" and red[4:6] == ["--api-key", "<redacted>"] and red[6] == "--password=<redacted>"
    assert red[7] == "PLAIN=1" and red[8] == "<redacted>"
    assert "HF_TOKEN" in names and "--api-key" in names and "--password" in names
    assert not any("hf_abc" in x or "sk-ABC" in x or "eyJ" in x for x in red)


def test_job_json_has_every_required_category_and_integer_bytes(tmp_path):
    doc = job.build_job(project="p", task="t", job="j", attempt=2, lane="mprof", generation=3, tree_sha="abc", release_tag=None, workload_digest=None, stage="match",
                        inputs=[job.input_ref(f"prog={tmp_path}/missing.json")], profile={"MATCH_IMPL": "fast"}, command=["python", "-m", "x", "TOKEN=abcdefghijklmnopqrstu"],
                        declared_outputs=["a"], timeout_s=10.0, reservation={"gpu": None, "host_memory_bytes": None}, policy_version="r17-rules-1",
                        policy_sha256="00", instruments=["cgroup", "process", "pss", "gpu", "storage"], cadence_s=1.0, pss_every_s=5.0)
    for k in ("ownership", "computation", "environment", "execution", "telemetry"):
        assert k in doc
    assert doc["computation"]["release_tag"] is None and doc["computation"]["workload_digest_reason"] == "not supplied"
    assert doc["computation"]["inputs"][0]["sha256"] is None and doc["computation"]["inputs"][0]["sha256_reason"] == "absent at launch"
    assert doc["execution"]["command"][-1] == "TOKEN=<redacted>"
    cg = doc["environment"]["cgroup"]
    assert cg["memory_max_kind"] in ("finite", "unlimited", "unknown")
    if cg["memory_max_kind"] == "finite":
        assert isinstance(cg["memory_max_bytes"], int)
    else:
        assert cg["memory_max_bytes"] is None
    if not doc["environment"]["gpu"]["available"]:
        assert doc["environment"]["gpu"]["reason"] and doc["telemetry"]["instruments_unavailable"]["gpu"] and "gpu" not in doc["telemetry"]["instruments"]
    hm = doc["environment"]["host_memory_total_bytes"]
    assert hm is None or isinstance(hm, int)
    json.dumps(doc)   # serializable


def test_status_state_machine(tmp_path):
    job.set_status(tmp_path, "submitted")
    job.set_status(tmp_path, "running", pid=1, pgid=1)
    st = job.set_status(tmp_path, "done", rc=0)
    assert [t["state"] for t in st["transitions"]] == ["submitted", "running", "done"] and st["pid"] == 1 and st["rc"] == 0
    with pytest.raises(ValueError):
        job.set_status(tmp_path, "bogus")


# ---- (g) policy -> RULES.md ---------------------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(SEED is None or not SEED.exists(), reason="set RESEARCH_SEED_RULES=<path to r17 RULES.md> to compare against the campaign seed")
def test_render_reproduces_the_seed_except_the_declared_corrections():
    seed = SEED.read_text().splitlines()
    out = policy.render(policy.HERE / "policy" / "r17-rules-1.toml").splitlines()
    assert len(out) == len(seed)
    diffs = [(a, b) for a, b in zip(seed, out) if a != b]
    assert len(diffs) == 1, diffs
    assert "MATCH_PIPELINE=shared+fast" in diffs[0][0] and "MATCH_PIPELINE=shared` beside `MATCH_IMPL=fast`" in diffs[0][1]


def test_check_rules_detects_a_hand_edit(tmp_path):
    pol = policy.HERE / "policy" / "r17-rules-1.toml"
    good = tmp_path / "RULES.md"
    good.write_text(policy.render(pol))
    assert policy.main(["check-rules", str(pol), str(good)]) == 0
    bad = tmp_path / "RULES2.md"
    bad.write_text(good.read_text().replace("cap $600", "cap $900"))
    assert policy.main(["check-rules", str(pol), str(bad)]) == 1
    assert policy.policy_values(pol)["defaults"] == {"MATCH_IMPL": "fast", "MATCH_PIPELINE": "shared", "GPU_GATE": 1, "disk_reserve_gb": 20, "need_var": "NEED_CG_GB"}


# ---- (f) classifier on synthetic attempt dirs ---------------------------------------------------------------------------------------------------------

def _attempt(tmp_path, *, rc, sig, samples, evs=(), pid=100, timeout_s=None, timed_out=False, start=100.0, end=110.0, stream="h:1:1"):
    d = tmp_path
    st = {"schema": "telemetry/status/v0.1", "state": "failed" if rc else "done", "rc": rc, "signal": sig, "pid": pid, "pgid": pid, "timeout_s": timeout_s,
          "timed_out": timed_out, "started_mono": start, "ended_mono": end, "updated_stream": stream, "rc_is_shell_status": True,
          "transitions": [{"state": "running", "t_mono": start}, {"state": "failed" if rc else "done", "t_mono": end}]}
    (d / "status.json").write_text(json.dumps(st))
    with open(d / "resources.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps({"schema": "telemetry/resources/v0.1", "stream": stream, **s}) + "\n")
    with open(d / "events.jsonl", "w") as f:
        for e in evs:
            f.write(json.dumps({"schema": "telemetry/events/v0.1", "stream": stream, **e}) + "\n")
    return d


def _cg(t, oom_kill, scope="shared_pod_cgroup", cur=1000):
    return {"kind": "cgroup", "scope": scope, "t_mono": t, "t_utc": f"T{t}", "cgroup": {"memory_events": {"oom_kill": oom_kill, "max": 0}, "memory_current_bytes": cur, "memory_peak_bytes": 5000}}


def _pr(t, procs, outsiders=(), gone=(), out_gone=(), scores=None):
    """procs: (pid, anon); outsiders: (pid, comm, anon); scores: {pid: oom_score} (None -> oom_score not sampled, pre-0.1.2 records)."""
    sc = scores or {}
    return {"kind": "process", "scope": "process", "t_mono": t, "t_utc": f"T{t}",
            "processes": [{"pid": p, "comm": "hog", "rss_anon_bytes": a, "rss_shmem_bytes": 0, **({"oom_score": sc[p]} if p in sc else {})} for p, a in procs],
            "outsiders": [{"pid": p, "comm": c, "anon_shmem_bytes": a, **({"oom_score": sc[p]} if p in sc else {})} for p, c, a in outsiders],
            "gone": list(gone), "outsiders_gone": [{"pid": p, "comm": "x"} for p in out_gone]}


def test_rc_137_alone_is_unknown_signal_never_oom(tmp_path):
    d = _attempt(tmp_path, rc=137, sig=None, samples=[_cg(100, 0), _cg(110, 0)])
    doc = classify(d)
    assert doc["class"] == "UNKNOWN_SIGNAL" and doc["termination"]["signal"] == 9 and doc["incidents"] == []
    assert "inferred from shell status" in doc["termination"]["note"]


def test_shared_scope_oom_with_bystander_job_is_not_attributed(tmp_path):
    # this job (pid 100, sleep) exits 0 while another process (pid 900, hog) got OOM-killed in the shared pod cgroup
    samples = [_cg(100, 0), _pr(100, [(100, 1000)], outsiders=[(900, "hog", 3_000_000_000)]),
               _cg(105, 1), _pr(105, [(100, 1000)], outsiders=[], out_gone=[900]),
               _cg(110, 1), _pr(110, [(100, 1000)])]
    doc = classify(_attempt(tmp_path, rc=0, sig=None, samples=samples))
    assert doc["class"] == "SUCCESS"
    assert doc["incidents"] and doc["incidents"][0]["kind"] == "cgroup_oom" and doc["incidents"][0]["scope"] == "shared_pod_cgroup"


def test_shared_scope_oom_killed_job_unresolved_when_an_outsider_also_vanished(tmp_path):
    samples = [_cg(100, 0), _pr(100, [(100, 2_000_000_000)], outsiders=[(900, "hog", 3_000_000_000)]),
               _cg(105, 1), _pr(105, [], gone=[100], out_gone=[900]),
               _cg(110, 1), _pr(110, [])]
    doc = classify(_attempt(tmp_path, rc=137, sig=9, samples=samples, end=106.0))
    assert doc["class"] == "OOM_UNRESOLVED_ORIGIN" and doc["certainty"] == "ambiguous"
    assert any("victim identity unresolved" in l for l in doc["limitations"])


def test_shared_scope_oom_attributed_when_only_the_largest_tree_process_vanished(tmp_path):
    samples = [_cg(100, 0), _pr(100, [(100, 3_500_000_000)], outsiders=[(900, "sshd", 5_000_000)]),
               _cg(105, 1), _pr(105, [], gone=[100], outsiders=[(900, "sshd", 5_000_000)]),
               _cg(110, 1), _pr(110, [], outsiders=[(900, "sshd", 5_000_000)])]
    doc = classify(_attempt(tmp_path, rc=137, sig=9, samples=samples, end=106.0))
    assert doc["class"] == "CGROUP_OOM_ATTRIBUTED" and doc["certainty"] == "inferred"
    assert any("shared cgroup" in l for l in doc["limitations"])
    assert doc["evidence"][-1]["kind"] == "oom.attribution" and doc["evidence"][-1]["largest_before"]["membership"] == "tree"


def test_job_cgroup_scope_oom_is_attributed_directly(tmp_path):
    doc = classify(_attempt(tmp_path, rc=137, sig=9, samples=[_cg(100, 0, "job_cgroup"), _cg(110, 1, "job_cgroup")]))
    assert doc["class"] == "CGROUP_OOM_ATTRIBUTED" and doc["certainty"] == "established"


def test_cancel_markers_classify_by_actor_and_need_delivery_or_matching_signal(tmp_path):
    for actor, cls in (("timeout", "STAGE_TIMEOUT"), ("guardian", "CANCELLED_GUARDIAN"), ("deadman", "CANCELLED_DEADMAN"), ("manual", "CANCELLED_MANUAL")):
        d = tmp_path / actor
        d.mkdir()
        evs = [{"kind": "cancel_intent", "actor": actor, "reason": "r", "signal": 9, "target": {"pid": 100}, "t_mono": 105.0, "t_utc": "T105"},
               {"kind": "cancel_result", "actor": actor, "signal": 9, "target": {"pid": 100}, "delivered": True, "t_mono": 105.1, "t_utc": "T105.1"}]
        # an oom_kill delta ALSO exists; the marker is delivered but NOT identity-verified -> rev c1: both retained, certainty ambiguous
        doc = classify(_attempt(d, rc=137, sig=9, samples=[_cg(100, 0), _cg(102, 0), _cg(110, 1)], evs=evs))
        assert doc["class"] == cls and doc["certainty"] == "ambiguous", actor
        assert [c["class"] for c in doc["competing"]] == [cls, "OOM_UNRESOLVED_ORIGIN"]
        assert doc["incidents"] and doc["incidents"][0]["kind"] == "cgroup_oom", "coexisting incident retained"
        # the same marker delivered AND identity-verified: the kill is observed -> established, incident retained, nothing competing
        d2 = tmp_path / (actor + "_verified")
        d2.mkdir()
        evs_v = [dict(evs[0], verified=True), evs[1]]
        doc = classify(_attempt(d2, rc=137, sig=9, samples=[_cg(100, 0), _cg(102, 0), _cg(110, 1)], evs=evs_v))
        assert doc["class"] == cls and doc["certainty"] == "established" and doc["competing"] == [], actor
        assert doc["incidents"][0]["kind"] == "cgroup_oom"
    # intent for ANOTHER pid does not classify this job
    d = tmp_path / "other"
    d.mkdir()
    evs = [{"kind": "cancel_intent", "actor": "guardian", "reason": "r", "signal": 9, "target": {"pid": 555}, "t_mono": 105.0, "t_utc": "T"}]
    doc = classify(_attempt(d, rc=137, sig=9, samples=[_cg(100, 0), _cg(110, 0)], evs=evs))
    assert doc["class"] == "UNKNOWN_SIGNAL"
    # intent without delivery record but the exit signal matches -> inferred
    d = tmp_path / "inferred"
    d.mkdir()
    evs = [{"kind": "cancel_intent", "actor": "manual", "reason": "r", "signal": 15, "target": {"pid": 100}, "t_mono": 105.0, "t_utc": "T"}]
    doc = classify(_attempt(d, rc=143, sig=15, samples=[], evs=evs))
    assert doc["class"] == "CANCELLED_MANUAL" and doc["certainty"] == "inferred"


def test_logs_cuda_enospc_and_checker_record(tmp_path):
    d = _attempt(tmp_path, rc=1, sig=None, samples=[])
    (d / "stderr.log").write_text("blah\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 GiB\n")
    assert classify(d)["class"] == "CUDA_ALLOC_FAIL"
    (d / "stderr.log").write_text("OSError: [Errno 28] No space left on device\n")
    assert classify(d)["class"] == "STORAGE_EXHAUSTED"
    (d / "stderr.log").write_text("Traceback (most recent call last):\n  File x\nValueError: nope\n")
    doc = classify(d)
    assert doc["class"] == "UNKNOWN_EXIT" and any(e["kind"] == "log.traceback" for e in doc["evidence"])
    (d / "gm.json").write_text(json.dumps({"verdict": "FAIL", "comparator": "global_match_fast"}))
    doc = classify(d, rc=3, check_record=str(d / "gm.json"))
    assert doc["class"] == "WORKLOAD_CHECK_FAILED" and doc["evidence"][-1]["verdict"] == "FAIL"
    (d / "gm.json").write_text(json.dumps({"verdict": "PASS"}))
    assert classify(d, rc=0, check_record=str(d / "gm.json"))["class"] == "SUCCESS"


def test_partial_telemetry_is_flagged_and_evidence_survives(tmp_path):
    d = _attempt(tmp_path, rc=0, sig=None, samples=[_cg(100, 0)], evs=[{"kind": "sampler_start", "t_mono": 99.0, "t_utc": "T"}])
    with open(d / "resources.jsonl", "ab") as f:
        f.write(b'{"kind": "cgroup", "sco')
    doc = classify(d)
    assert doc["class"] == "SUCCESS" and doc["telemetry_partial"] is True
    assert any("truncated last line" in l for l in doc["limitations"]) and any("never wrote sampler_end" in l for l in doc["limitations"])
    assert doc["classifier_version"] == CLASSIFIER_VERSION


def test_shared_scope_attribution_tolerates_the_round_offset_between_cgroup_and_process_records(tmp_path):
    # real sampler layout: cgroup record at t, process record of the SAME round a few ms later; the hog vanishes in round j
    samples = [_cg(100.000, 0), _pr(100.004, [(100, 3_500_000_000)], outsiders=[(900, "sshd", 5_000_000)]),
               _cg(101.000, 1), _pr(101.004, [], gone=[100], outsiders=[(900, "sshd", 5_000_000)]),
               _cg(102.000, 1), _pr(102.004, [], outsiders=[(900, "sshd", 5_000_000)])]
    doc = classify(_attempt(tmp_path, rc=137, sig=9, samples=samples, end=101.5))
    assert doc["class"] == "CGROUP_OOM_ATTRIBUTED", doc["evidence"][-1]
    att = doc["evidence"][-1]
    assert att["tree_gone"] == [100] and att["outsiders_gone"] == [] and att["largest_before"]["pid"] == 100


# ---- rev M-1748 d1-d8 ---------------------------------------------------------------------------------------------------------------------

def test_d1_checker_fail_with_rc_0_is_workload_check_failed_not_success(tmp_path):
    d = _attempt(tmp_path, rc=0, sig=None, samples=[])
    (d / "gm.json").write_text(json.dumps({"verdict": "FAIL", "comparator": "global_match_fast"}))
    assert classify(d, rc=0, check_record=str(d / "gm.json"))["class"] == "WORKLOAD_CHECK_FAILED"
    (d / "gm.json").write_text(json.dumps({"verdict": "TIMEOUT"}))
    assert classify(d, rc=0, check_record=str(d / "gm.json"))["class"] == "WORKLOAD_CHECK_FAILED"


def test_c2_primary_evidence_is_disappearance_and_ranking_is_corroboration_never_a_veto(tmp_path):
    # rev P3 / d2 discriminator: small job process at oom_score_adj 1000 killed; a LARGER bystander at adj 0 survives
    scores = {100: 1005, 900: 310}
    samples = [_cg(100, 0), _pr(100.01, [(100, 200)], [(900, "big", 3000)], scores=scores),
               _cg(101, 0), _pr(101.01, [(100, 250)], [(900, "big", 3000)], scores=scores),
               _cg(102, 1), _pr(102.01, [], [(900, "big", 3000)], gone=[100], scores=scores)]
    doc = classify(_attempt(tmp_path, rc=137, sig=9, samples=samples, end=102.0))
    assert doc["class"] == "CGROUP_OOM_ATTRIBUTED" and doc["certainty"] == "inferred", doc["why"]
    att = next(e for e in doc["evidence"] if e["kind"] == "oom.attribution")
    assert att["corroboration"]["highest_oom_score_is_the_vanished_tree_process"] is True
    assert att["corroboration"]["largest_anon_shmem_is_the_vanished_tree_process"] is False
    assert att["highest_oom_score_before"]["pid"] == 100 and att["largest_before"]["pid"] == 900
    assert att["attribution_sample_age_s"] == pytest.approx(0.99, abs=0.01)
    assert any("attribution_sample_age_s" in l for l in doc["limitations"]) and any("largest anon+shmem" in l for l in doc["limitations"])
    # ranking disagreement is NOT a veto: even with the outsider holding the higher score, a lone tree disappearance attributes (with the disagreement stated)
    d2 = tmp_path / "disagree"
    d2.mkdir()
    scores = {100: 300, 900: 1005}
    samples = [_cg(100, 0), _pr(100.01, [(100, 200)], [(900, "big", 3000)], scores=scores), _cg(102, 1), _pr(102.01, [], [(900, "big", 3000)], gone=[100], scores=scores)]
    doc = classify(_attempt(d2, rc=137, sig=9, samples=samples, end=102.0))
    assert doc["class"] == "CGROUP_OOM_ATTRIBUTED" and any("corroboration disagrees" in l for l in doc["limitations"])
    # an outsider vanished too -> unresolved, whatever the ranking says
    d3 = tmp_path / "outsider_gone"
    d3.mkdir()
    samples = [_cg(100, 0), _pr(100.01, [(100, 3000)], [(900, "big", 200)], scores={100: 1005, 900: 10}), _cg(102, 1), _pr(102.01, [], [], gone=[100], out_gone=[900])]
    assert classify(_attempt(d3, rc=137, sig=9, samples=samples, end=102.0))["class"] == "OOM_UNRESOLVED_ORIGIN"
    # no tree process vanished -> unresolved
    d4 = tmp_path / "none_gone"
    d4.mkdir()
    samples = [_cg(100, 0), _pr(100.01, [(100, 3000)], [(900, "big", 200)]), _cg(102, 1), _pr(102.01, [(100, 3000)], [(900, "big", 200)])]
    assert classify(_attempt(d4, rc=137, sig=9, samples=samples, end=102.0))["class"] == "OOM_UNRESOLVED_ORIGIN"


def test_c1_cuda_string_or_low_mount_beside_a_sigkill_oom_is_ambiguous_with_both_retained(tmp_path):
    samples = [_cg(100, 0), _pr(100.01, [(100, 3000)]), _cg(102, 1), _pr(102.01, [], gone=[100])]
    d = _attempt(tmp_path, rc=137, sig=9, samples=samples, end=102.0)
    (d / "stderr.log").write_text("W0921 stale warning: CUDA out of memory (retrying)\n")
    doc = classify(d)
    assert doc["class"] == "CUDA_ALLOC_FAIL" and doc["certainty"] == "ambiguous"
    assert [c["class"] for c in doc["competing"]] == ["CUDA_ALLOC_FAIL", "CGROUP_OOM_ATTRIBUTED"]
    assert doc["incidents"][0]["kind"] == "cgroup_oom"
    # rc 1 with the CUDA string and no oom delta: plain established
    d2 = tmp_path / "plain"
    d2.mkdir()
    _attempt(d2, rc=1, sig=None, samples=[])
    (d2 / "stderr.log").write_text("torch.OutOfMemoryError: CUDA out of memory\n")
    assert classify(d2)["certainty"] == "established"


def test_c1_low_mount_counts_only_when_it_holds_the_attempt_or_a_declared_output(tmp_path):
    def storage(t, path, free):
        return {"kind": "storage", "scope": "host", "t_mono": t, "t_utc": f"T{t}", "mounts": [{"path": path, "free_bytes": free}]}
    d = _attempt(tmp_path, rc=1, sig=None, samples=[storage(105, "/somewhere/else", 1024)], end=110.0)
    doc = classify(d)
    assert doc["class"] == "UNKNOWN_EXIT" and doc["incidents"][0]["kind"] == "storage_low" and doc["incidents"][0]["holds_attempt_or_declared_output"] is False
    (d / "job.json").write_text(json.dumps({"execution": {"declared_outputs": ["/somewhere/else/out.json"]}}))
    doc = classify(d)
    assert doc["class"] == "STORAGE_EXHAUSTED" and doc["incidents"][0]["holds_attempt_or_declared_output"] is True
    (d / "job.json").unlink()
    d2 = tmp_path / "own"
    d2.mkdir()
    _attempt(d2, rc=1, sig=None, samples=[storage(105, str(tmp_path), 1024)], end=110.0)
    assert classify(d2)["class"] == "STORAGE_EXHAUSTED"


def test_p6_unknown_signal_without_any_samples_is_ambiguous(tmp_path):
    doc = classify(_attempt(tmp_path, rc=137, sig=9, samples=[]))
    assert doc["class"] == "UNKNOWN_SIGNAL" and doc["certainty"] == "ambiguous" and any("no resource samples" in l for l in doc["limitations"])


def test_d4_child_oom_killed_while_parent_exits_nonzero_keeps_the_incident_detail(tmp_path):
    scores = {100: 50, 101: 900}
    samples = [_cg(100, 0), _pr(100.01, [(100, 100), (101, 4000)], scores=scores), _cg(102, 1), _pr(102.01, [(100, 100)], gone=[101], scores=scores)]
    doc = classify(_attempt(tmp_path, rc=1, sig=None, samples=samples, end=105.0))
    assert doc["class"] == "UNKNOWN_EXIT"
    inc = doc["incidents"][0]
    assert inc["kind"] == "cgroup_oom"
    w = inc["windows"][0]
    assert w["tree_gone"] == [101] and w["gone_detail"][0] == {"pid": 101, "membership": "tree", "comm": "hog", "last_oom_score": 900, "last_anon_shmem_bytes": 4000}
    assert w["to"] - w["from"] == 1 and w["counter"].endswith("oom_kill")
    assert w["attribution_sample_age_s"] is not None


def test_d5_children_attach_spans_under_the_exported_parent(tmp_path, monkeypatch):
    d = tmp_path
    em = events.Emitter(d, proc="runner")
    root = em.span_start("row")
    monkeypatch.setenv("TELE_PARENT_SPAN", root)
    monkeypatch.setenv("TELE_ATTEMPT_DIR", str(d))
    child = events.Emitter(d, proc="worker").span_start("match.fold")          # no parent passed: inherited from the environment
    out = subprocess.run([sys.executable, "-m", "research.telemetry", "event", "span-start", "--stage", "match.gm"], capture_output=True, text=True, check=True).stdout.strip()
    sp = events.spans(events.read_all(d, "events").records)
    by = {s["id"]: s for s in sp["spans"]}
    assert by[child]["parent"] == root and by[out]["parent"] == root and set(by[root]["children"]) == {child, out}


def test_d6_deferred_hash_records_identity_at_launch_and_completes_at_preserve(tmp_path, monkeypatch):
    big = tmp_path / "fold.json"
    big.write_bytes(b"x" * 4096)
    monkeypatch.setattr(job, "sha256_file", lambda p, max_bytes=256 << 20: (None, "not hashed at launch (4096 bytes > 1024)"))   # launch: too big
    ref = job.input_ref(f"fold={big}")
    assert ref["sha256"] is None and ref["sha256_deferred_to"] == "preserve.verify"
    assert ref["stat"]["size"] == 4096 and ref["stat"]["ino"] == big.stat().st_ino and "mtime_ns" in ref["stat"]
    monkeypatch.undo()
    (tmp_path / "job.json").write_text(json.dumps({"computation": {"inputs": [ref]}}))
    done = job.hash_inputs(tmp_path)
    assert len(done) == 1 and done[0]["sha256"] == __import__("hashlib").sha256(b"x" * 4096).hexdigest() and done[0]["stat_changed_since_launch"] is False
    doc = json.loads((tmp_path / "job.json").read_text())
    assert doc["computation"]["inputs"][0]["sha256_hashed_at"] == "preserve.verify" and "sha256_deferred_to" not in doc["computation"]["inputs"][0]
    # a file replaced between launch and preserve is flagged
    big.write_bytes(b"y" * 4096)
    doc["computation"]["inputs"][0]["sha256"] = None
    (tmp_path / "job.json").write_text(json.dumps(doc))
    assert job.hash_inputs(tmp_path)[0]["stat_changed_since_launch"] is True


# ---- R18 infra regressions (ported from veritor r18-int-3) -------------------------------------------------------------------------------------

def test_event_accepts_kv_positionals_after_options_on_every_python(tmp_path):
    """[r18-int-2, arch M-0025] run_row.sh writes `tele event ... submitted --stage S row=.. role=..`; 3.11 parse_args rejected the k=v
    tail ('unrecognized arguments') and the event was silently lost to telemetry_errors.log; parse_intermixed_args takes it everywhere."""
    env = dict(os.environ, PYTHONPATH=str(PKG_ROOT))
    r = subprocess.run([sys.executable, "-m", "research.telemetry", "event", "--attempt-dir", str(tmp_path), "--proc", "run_row", "submitted",
                        "--stage", "build,match", "row=r1", "role=B0"], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    recs = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines() if l.strip()]
    ev = [x for x in recs if x.get("kind") == "queue" or x.get("what") == "submitted" or "submitted" in json.dumps(x)]
    assert ev, recs
    assert json.dumps(ev[-1]).count("r1") >= 1 and json.dumps(ev[-1]).count("B0") >= 1


def test_tele_run_launched_through_the_script_starts_a_sampler_that_finds_the_package(tmp_path):
    """[r18-int-2, F-r18-infra-1] r18-int-1 canary: `zk/campaign/tele run ...` -> sampler.log '/usr/bin/python3: No module named telemetry',
    no resources.jsonl.  The launcher below is that script's shape: the import root goes on sys.path IN-PROCESS only, the environment
    carries no PYTHONPATH and the cwd is elsewhere.  The sidecar (`python -m research.telemetry sample`, a new process) must still
    resolve the same package -- run._tele_env() puts `src/` (parents[2] of research/telemetry/run.py) on the child's PYTHONPATH.

    The package under test is a COPY of research/ in tmp_path whose __main__ records its own __file__: the interpreter running these
    tests may have `research` installed (editable) in site-packages, which would let a sidecar that lost the PYTHONPATH import the
    installed copy and hide the bug (parents[1] passes the naive form of this test for exactly that reason)."""
    import shutil
    root = tmp_path / "root"
    shutil.copytree(PKG_ROOT / "research", root / "research", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    main_py = root / "research" / "telemetry" / "__main__.py"
    mark = tmp_path / "sidecar_module_file.txt"
    main_py.write_text(f"import pathlib\npathlib.Path({str(mark)!r}).write_text(__file__)\n" + main_py.read_text())
    launcher = tmp_path / "tele"
    launcher.write_text("import sys\n"
                        f"sys.path.insert(0, {str(root)!r})\n"
                        "from research.telemetry.cli import main\n"
                        "sys.exit(main())\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, str(launcher), "run", "--attempt-dir", str(tmp_path / "a"), "--stage", "t", "--cadence", "0.2", "--linger", "0.5",
                        "--", "sleep", "1.5"], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    log = (tmp_path / "a" / "sampler.log").read_text() if (tmp_path / "a" / "sampler.log").exists() else ""
    assert "No module named" not in log, log
    assert (tmp_path / "a" / "resources.jsonl").exists(), sorted(p.name for p in (tmp_path / "a").iterdir())
    assert (tmp_path / "a" / "resources.jsonl").read_text().strip(), "no sample written"
    assert mark.exists(), "the sidecar did not run the launcher's package (an installed `research` elsewhere may have been imported instead)"
    assert Path(mark.read_text()).resolve() == main_py.resolve()


def test_tele_run_reads_a_timeout_change_from_the_control_file_and_records_it(tmp_path):
    """[r19-int-8, F-r19-int-8] Seven R19 rows were launched with --timeout sized on Build alone; tele run held the value in memory and
    would have SIGTERMed healthy Match/Commit stages.  `tele set-timeout` writes control/timeout_s; the run loop re-reads it (mtime/size
    change), emits a `timeout_changed` mark carrying old/new/actor/reason and updates status.json.  Here: a 1 s launch timeout is raised
    to 60 s at t~0.4 s, the 2.5 s workload finishes with rc 0 and no timeout marker; then the reverse -- a 60 s launch timeout lowered to
    0.5 s ends the workload with the ordinary timeout cancel-intent naming the NEW value."""
    import threading
    from research.telemetry import run as trun

    def go(d, timeout, cmd):
        return subprocess.run([sys.executable, "-m", "research.telemetry", "run", "--attempt-dir", str(d), "--stage", "t", "--timeout", str(timeout),
                               "--no-sampler", "--no-classify", "--grace", "1", "--", *cmd], capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(PKG_ROOT)}, cwd=str(tmp_path))

    d = tmp_path / "raise"
    d.mkdir()
    threading.Timer(0.4, trun.set_timeout, args=(d, 60.0, "manual", "sized on Build alone")).start()
    r = go(d, 1.0, ["sleep", "2.5"])
    assert r.returncode == 0, r.stderr
    recs = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
    marks = [x for x in recs if x.get("kind") == "mark" and x.get("name") == "timeout_changed"]
    assert len(marks) == 1 and marks[0]["old_timeout_s"] == 1.0 and marks[0]["new_timeout_s"] == 60.0
    assert marks[0]["actor"] == "manual" and marks[0]["reason"] == "sized on Build alone"
    assert not [x for x in recs if x.get("kind") == "cancel_intent"]
    st = json.loads((d / "status.json").read_text())
    assert st["state"] == "done" and st["timeout_s"] == 60.0 and st.get("timed_out") is False

    d2 = tmp_path / "lower"
    d2.mkdir()
    threading.Timer(0.4, trun.set_timeout, args=(d2, 0.5, "manual", "abort")).start()
    r = go(d2, 60.0, ["sleep", "20"])
    assert r.returncode != 0
    recs = [json.loads(l) for l in (d2 / "events.jsonl").read_text().splitlines() if l.strip()]
    ci = [x for x in recs if x.get("kind") == "cancel_intent"]
    assert ci and ci[0]["actor"] == "timeout" and "timeout_s=0.5" in ci[0]["reason"]
    st = json.loads((d2 / "status.json").read_text())
    assert st["state"] == "cancelled" and st["timed_out"] is True

    # a bare number and an unparseable file: the first is accepted, the second is ignored (never the reason a workload stops)
    d3 = tmp_path / "bare"
    d3.mkdir()
    (d3 / "control").mkdir()
    (d3 / "control" / "timeout_s").write_text("not json")
    threading.Timer(0.4, lambda: (d3 / "control" / "timeout_s").write_text("30")).start()
    r = go(d3, 1.0, ["sleep", "2.0"])
    assert r.returncode == 0, r.stderr
    assert "ignoring unparseable" in r.stderr
    recs = [json.loads(l) for l in (d3 / "events.jsonl").read_text().splitlines() if l.strip()]
    assert [x["new_timeout_s"] for x in recs if x.get("kind") == "mark" and x.get("name") == "timeout_changed"] == [30.0]


def test_set_timeout_cli_writes_the_control_file():
    from research.telemetry.cli import main
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        assert main(["set-timeout", "--attempt-dir", td, "--timeout-s", "14400", "--reason", "x"]) == 0
        doc = json.loads((Path(td) / "control" / "timeout_s").read_text())
        assert doc["timeout_s"] == 14400.0 and doc["actor"] == "manual" and doc["reason"] == "x" and doc["set_utc"]
        assert main(["set-timeout", "--attempt-dir", td, "--timeout-s", "none", "--reason", "clear"]) == 0
        assert json.loads((Path(td) / "control" / "timeout_s").read_text())["timeout_s"] is None


# ---- (g) 0.1.3 rule 7b: cgroup-v1 OOM kill of a descendant behind a wrapper's rc != 0 (r20 issue 3) ------------------------------------------------

def _fake_v1_tree(root: Path, *, oom_kill: int | None, failcnt: int, under_oom: int = 0, usage: int = 4 << 30) -> None:
    """oom_kill=None writes a pre-4.13 memory.oom_control (no oom_kill line)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "memory.limit_in_bytes").write_text(str(8 << 30))
    (root / "memory.usage_in_bytes").write_text(str(usage))
    (root / "memory.max_usage_in_bytes").write_text(str(8 << 30))
    (root / "memory.failcnt").write_text(str(failcnt))
    (root / "memory.oom_control").write_text(f"oom_kill_disable 0\nunder_oom {under_oom}\n" + (f"oom_kill {oom_kill}\n" if oom_kill is not None else ""))
    (root / "memory.stat").write_text("total_rss 100\ntotal_cache 10\ntotal_dirty 0\ntotal_writeback 0\ntotal_shmem 0\n")


def _v1_samples(tmp_path, monkeypatch, *, scope="shared_pod_cgroup", oom_kill=(0, 1), failcnt=(3, 7), under_oom=(0, 0)):
    """Two sampler rounds read through research.telemetry.cgroup off a fake cgroup-v1 memory hierarchy (default: oom_kill 0 -> 1, failcnt 3 -> 7)."""
    from research.telemetry import cgroup as cgm

    sysfs = tmp_path / "sysfs" / "memory"
    monkeypatch.setattr(cgm, "mounts", lambda: {"v2": None, "v1_memory": str(sysfs), "v2_ro": None})
    monkeypatch.setattr(cgm, "self_cgroup_path", lambda pid="self": (None, "/kubepods/pod-abc", None))
    leaf = sysfs / "kubepods" / "pod-abc"
    _fake_v1_tree(leaf, oom_kill=oom_kill[0], failcnt=failcnt[0], under_oom=under_oom[0])
    cg = cgm.Cgroup("self")
    assert cg.version == 1 and cg.dir == leaf
    s0 = cg.sample()
    assert s0["memory_events"] == {"oom_kill": oom_kill[0], "under_oom": under_oom[0]} and s0["memory_failcnt"] == failcnt[0]
    _fake_v1_tree(leaf, oom_kill=oom_kill[1], failcnt=failcnt[1], under_oom=under_oom[1], usage=1 << 30)
    s1 = cg.sample()
    assert s1["memory_events"]["oom_kill"] == oom_kill[1]
    return [{"kind": "cgroup", "scope": scope, "t_mono": 100.0, "t_utc": "T100", "cgroup": s0},
            {"kind": "cgroup", "scope": scope, "t_mono": 104.0, "t_utc": "T104", "cgroup": s1}]


KILLED_LINE = "bash: line 1:  4242 Killed                  python -m verity_capture.experimental.cb_a.derive_step --row 3\n"


def test_v1_oom_control_increment_with_a_killed_line_behind_rc_1_is_cgroup_oom(tmp_path, monkeypatch):
    samples = _v1_samples(tmp_path, monkeypatch)
    samples.insert(1, _pr(100.01, [(100, 100), (4242, 4000)], scores={100: 50, 4242: 950}))
    samples.append(_pr(104.01, [(100, 100)], gone=[4242], scores={100: 50, 4242: 950}))
    d = _attempt(tmp_path, rc=1, sig=None, samples=samples, end=105.0)
    (d / "stdout.log").write_text("step 2 done\n" + KILLED_LINE + "wrapper: derive_step failed rc=137\n")
    doc = classify(d)
    assert doc["class"] == "CGROUP_OOM" and doc["certainty"] == "inferred", (doc["class"], doc["certainty"], doc["why"])
    assert doc["termination"]["rc"] == 1 and doc["termination"]["signal"] is None
    assert "4242" in doc["why"]
    kinds = [e["kind"] for e in doc["evidence"]]
    assert "log.killed" in kinds and "oom.attribution" in kinds and "cgroup.memory.events" in kinds
    att = next(e for e in doc["evidence"] if e["kind"] == "oom.attribution")
    assert att["tree_gone"] == [4242] and att["outsiders_gone"] == [] and att["v1_failcnt_delta"] == 4
    assert att["windows"][0]["window"]["counter"] == "memory_events.oom_kill"
    inc = doc["incidents"][0]
    assert inc["kind"] == "cgroup_oom" and inc["counters_delta"] == {"memory_events.oom_kill": 1} and inc["v1_failcnt_delta"] == 4
    assert any("wrapper" in l for l in doc["limitations"])


def test_rule_7b_is_established_in_the_jobs_own_cgroup_and_ambiguous_when_an_outsider_vanished_too(tmp_path, monkeypatch):
    own = tmp_path / "own"
    own.mkdir()
    d = _attempt(own, rc=2, sig=None, samples=_v1_samples(own, monkeypatch, scope="job_cgroup"), end=105.0)
    (d / "stderr.log").write_text(KILLED_LINE)
    doc = classify(d)
    assert doc["class"] == "CGROUP_OOM" and doc["certainty"] == "established"
    shared = tmp_path / "shared"
    shared.mkdir()
    samples = _v1_samples(shared, monkeypatch)
    samples.insert(1, _pr(100.01, [(100, 100), (4242, 4000)], outsiders=[(900, "hog", 9000)]))
    samples.append(_pr(104.01, [(100, 100)], gone=[4242], out_gone=[900]))
    d = _attempt(shared, rc=1, sig=None, samples=samples, end=105.0)
    (d / "stdout.log").write_text(KILLED_LINE)
    doc = classify(d)
    assert doc["class"] == "CGROUP_OOM" and doc["certainty"] == "ambiguous"
    assert any("outsider" in l for l in doc["limitations"])


def test_rule_7b_needs_both_the_counter_and_the_killed_line(tmp_path, monkeypatch):
    # counter without the line: UNKNOWN_EXIT with the incident (rev d4) and a limitation naming it
    a = tmp_path / "counter_only"
    a.mkdir()
    doc = classify(_attempt(a, rc=1, sig=None, samples=_v1_samples(a, monkeypatch), end=105.0))
    assert doc["class"] == "UNKNOWN_EXIT" and doc["incidents"][0]["kind"] == "cgroup_oom"
    assert any("without a Killed line" in l for l in doc["limitations"])
    # the line beside a kill counter that did not move: a string is evidence, not a class
    b = tmp_path / "line_only"
    b.mkdir()
    d = _attempt(b, rc=1, sig=None, samples=[_cg(100, 0), _cg(104, 0)], end=105.0)
    (d / "stdout.log").write_text("Killed by a colleague's typo, not the kernel\n" + KILLED_LINE)
    doc = classify(d)
    assert doc["class"] == "UNKNOWN_EXIT" and doc["incidents"] == []
    killed = [e for e in doc["evidence"] if e["kind"] == "log.killed"]
    assert len(killed) == 1 and killed[0]["match"] == KILLED_LINE.rstrip("\n")      # the prose line is not the match
    # rc 137 (shell status for SIGKILL) keeps its own path: rule 6 attribution, never 7b
    c = tmp_path / "rc137"
    c.mkdir()
    d = _attempt(c, rc=137, sig=None, samples=_v1_samples(c, monkeypatch, scope="job_cgroup"), end=105.0)
    (d / "stdout.log").write_text(KILLED_LINE)
    assert classify(d)["class"] == "CGROUP_OOM_ATTRIBUTED"


def test_killed_re_matches_shell_kernel_and_subprocess_reports_not_prose():
    from research.telemetry.classify import KILLED_RE

    for line in (KILLED_LINE, "./run.sh: line 12: 777 Killed python x.py", "Killed", "sh: 1: python: Killed", "[ 12.3] Memory cgroup out of memory: Killed process 9",
                 "subprocess.CalledProcessError: Command '['python']' died with <Signals.SIGKILL: 9>.", "Command 'x' returned non-zero exit status -9."):
        assert KILLED_RE.search(line), line
    for line in ("Killed by a colleague's typo", "processes Killed: none", "skilled worker", "KilledProcessCount=0"):
        assert not KILLED_RE.search(line), line


def test_v1_without_an_oom_kill_counter_falls_back_to_failcnt_or_under_oom(tmp_path, monkeypatch):
    # pre-4.13 memory.oom_control: the sampler records oom_kill=None (no counter) and says why, never 0 (no kill)
    own = tmp_path / "own"
    own.mkdir()
    samples = _v1_samples(own, monkeypatch, scope="job_cgroup", oom_kill=(None, None), failcnt=(3, 9))
    assert samples[0]["cgroup"]["memory_events"]["oom_kill"] is None and "4.13" in samples[0]["cgroup"]["memory_events_reason"]
    d = _attempt(own, rc=1, sig=None, samples=samples, end=105.0)
    (d / "stderr.log").write_text(KILLED_LINE)
    doc = classify(d)
    assert doc["class"] == "CGROUP_OOM" and doc["certainty"] == "inferred", (doc["class"], doc["certainty"], doc["why"])
    assert "failcnt grew by 6" in doc["why"] and doc["incidents"] == []
    att = next(e for e in doc["evidence"] if e["kind"] == "oom.attribution")
    assert att["kill_counter"] is False and att["v1_failcnt_delta"] == 6 and att["windows"] == []
    assert any("kernel < 4.13" in l for l in doc["limitations"])
    # shared scope, under_oom raised without failcnt growth: one step lower again
    shared = tmp_path / "shared"
    shared.mkdir()
    d = _attempt(shared, rc=1, sig=None, samples=_v1_samples(shared, monkeypatch, oom_kill=(None, None), failcnt=(5, 5), under_oom=(0, 1)), end=105.0)
    (d / "stdout.log").write_text(KILLED_LINE)
    doc = classify(d)
    assert doc["class"] == "CGROUP_OOM" and doc["certainty"] == "ambiguous" and "under_oom was raised" in doc["why"]
    # no Killed line: the limit was hit but nothing says a process died of it
    quiet = tmp_path / "quiet"
    quiet.mkdir()
    doc = classify(_attempt(quiet, rc=1, sig=None, samples=_v1_samples(quiet, monkeypatch, oom_kill=(None, None), failcnt=(3, 9)), end=105.0))
    assert doc["class"] == "UNKNOWN_EXIT"


def test_v1_kill_counter_that_did_not_move_vetoes_the_failcnt_fallback(tmp_path, monkeypatch):
    d = _attempt(tmp_path, rc=1, sig=None, samples=_v1_samples(tmp_path, monkeypatch, oom_kill=(2, 2), failcnt=(3, 90)), end=105.0)
    (d / "stdout.log").write_text(KILLED_LINE)
    doc = classify(d)
    assert doc["class"] == "UNKNOWN_EXIT" and doc["incidents"] == []
    assert "log.killed" in [e["kind"] for e in doc["evidence"]]
