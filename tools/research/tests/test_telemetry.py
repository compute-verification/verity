"""Pure-Python tests of the telemetry package (run anywhere with Python 3.11+; the Linux-only fixtures live in fixtures/run_fixtures.sh).

    uv run pytest tools/research/tests -q      (from the verity root)
"""
from __future__ import annotations

import argparse
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
from waits import Gate  # noqa: E402

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
    # realistic-looking secrets, each split into adjacent literals so that no secret scanner reads one in this file
    argv = ["bash", "row_pod.sh", "row", "HF_TOKEN=hf_" "abcdefghijklmnopqrstuvwxyz0123", "--api-key", "sk-" "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
            "--password=pw", "PLAIN=1",
            "eyJhbGciOiJIUzI1NiJ9." "eyJzdWIiOiIxMjM0NTY3ODkwIn0." "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"]
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


def test_a_cancel_of_a_large_group_keeps_its_pgid_and_classifies(tmp_path, monkeypatch):
    from research.telemetry import cancel
    members = [{"pid": 1000 + i, "comm": "python3", "starttime_ticks": 45_000_000 + i} for i in range(120)]
    monkeypatch.setattr(cancel, "pgid_members", lambda pgid: members)
    monkeypatch.setattr(cancel, "run_dir_of", lambda pid: str(tmp_path.resolve()))
    d = _attempt(tmp_path, rc=143, sig=None, samples=[], pid=100)
    a = argparse.Namespace(actor="manual", attempt_dir=str(d), fallback=None, target_pid=None, target_pgid=100, target_pod=None,
                           expect_starttime=None, expect_comm=None, expect_run_dir=str(d), expect_nonempty=False, reason="r", signal=15,
                           project=None, job=None, attempt=None, controller=None, deliver=False)
    assert cancel.cancel(a) == 0
    intent = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()][-1]
    assert "truncated_fields" not in intent
    assert intent["target"]["pgid"] == 100 and intent["target"]["members_total"] == 120 and intent["verified"] is True
    intent["t_mono"] = 105.0
    (d / "events.jsonl").write_text(json.dumps(intent) + "\n" + json.dumps(dict(intent, kind="cancel_result", delivered=True, t_mono=105.1)) + "\n")
    assert classify(d)["class"] == "CANCELLED_MANUAL"


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
    st = big.stat()                                        # later, as a replacement is: within one mtime tick the stat cannot tell
    os.utime(big, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
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
    gate = Gate(tmp_path / "sampled")            # the workload runs until the sidecar's first round is written, so the sidecar is up
    main_py.write_text(f"import os, pathlib\npathlib.Path({str(mark)!r}).write_text(__file__)\n"
                       "from research.telemetry import sample as _sample\n"
                       "_round = _sample.Sampler.round\n"
                       f"def _release_after_first(self, final=False, gate={str(gate.path)!r}):\n"
                       "    alive = _round(self, final)\n"
                       "    if self.samples == 1:\n"
                       "        fd = os.open(gate, os.O_WRONLY); os.write(fd, b'x'); os.close(fd)\n"
                       "    return alive\n"
                       "_sample.Sampler.round = _release_after_first\n" + main_py.read_text())
    launcher = tmp_path / "tele"
    launcher.write_text("import sys\n"
                        f"sys.path.insert(0, {str(root)!r})\n"
                        "from research.telemetry.cli import main\n"
                        "sys.exit(main())\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    try:
        r = subprocess.run([sys.executable, str(launcher), "run", "--attempt-dir", str(tmp_path / "a"), "--stage", "t", "--cadence", "0.2", "--linger", "0.5",
                            "--", sys.executable, "-c", Gate.WAIT, str(gate.path)], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    finally:
        gate.close()
    assert r.returncode == 0, r.stderr
    log = (tmp_path / "a" / "sampler.log").read_text() if (tmp_path / "a" / "sampler.log").exists() else ""
    assert "No module named" not in log, log
    assert (tmp_path / "a" / "resources.jsonl").exists(), sorted(p.name for p in (tmp_path / "a").iterdir())
    assert (tmp_path / "a" / "resources.jsonl").read_text().strip(), "no sample written"
    assert mark.exists(), "the sidecar did not run the launcher's package (an installed `research` elsewhere may have been imported instead)"
    assert Path(mark.read_text()).resolve() == main_py.resolve()


#: The operator's `tele set-timeout` from inside the workload, mid-run; then it waits for the gate argv[1] (never released: killed).
SETS_TIMEOUT = ("import sys; from research.telemetry.run import set_timeout; set_timeout(sys.argv[2], float(sys.argv[3]), 'manual', sys.argv[4]); "
                "open(sys.argv[1], 'rb').read(1)")


def test_tele_run_reads_a_timeout_change_from_the_control_file_and_records_it(tmp_path, monkeypatch):
    """[r19-int-8, F-r19-int-8] Seven R19 rows were launched with --timeout sized on Build alone; tele run held the value in memory and
    would have SIGTERMed healthy Match/Commit stages.  `tele set-timeout` writes control/timeout_s; the run loop re-reads it (mtime/size
    change), emits a `timeout_changed` mark carrying old/new/actor/reason and updates status.json.  The stage clock stands at 0 until
    status.json carries the new value, then reads 30 s: past a 1 s launch timeout raised to 60 s (the workload is released and finishes
    with rc 0, no timeout marker), and past a 60 s launch timeout lowered to 0.5 s (the ordinary timeout cancel-intent names the NEW
    value)."""
    from research.telemetry import run as trun
    monkeypatch.setenv("PYTHONPATH", str(PKG_ROOT))

    def go(d, launch, new, reason, gate, *, finish):
        read = []

        def clock():
            st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
            if st.get("timeout_s") != new:
                return 0.0
            if finish and not read:
                gate.release()
            read.append(True)
            return 30.0
        monkeypatch.setattr(trun, "clock", clock)
        return trun.main(["--attempt-dir", str(d), "--stage", "t", "--timeout", str(launch), "--no-sampler", "--no-classify", "--grace", "1",
                          "--", sys.executable, "-c", SETS_TIMEOUT, str(gate.path), str(d), str(new), reason])

    d, gate = tmp_path / "raise", Gate(tmp_path / "raise.gate")
    d.mkdir()
    try:
        assert go(d, 1.0, 60.0, "sized on Build alone", gate, finish=True) == 0
    finally:
        gate.close()
    recs = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
    marks = [x for x in recs if x.get("kind") == "mark" and x.get("name") == "timeout_changed"]
    assert len(marks) == 1 and marks[0]["old_timeout_s"] == 1.0 and marks[0]["new_timeout_s"] == 60.0
    assert marks[0]["actor"] == "manual" and marks[0]["reason"] == "sized on Build alone"
    assert not [x for x in recs if x.get("kind") == "cancel_intent"]
    st = json.loads((d / "status.json").read_text())
    assert st["state"] == "done" and st["timeout_s"] == 60.0 and st.get("timed_out") is False

    d2, gate = tmp_path / "lower", Gate(tmp_path / "lower.gate")
    d2.mkdir()
    try:
        assert go(d2, 60.0, 0.5, "abort", gate, finish=False) != 0
    finally:
        gate.close()
    recs = [json.loads(l) for l in (d2 / "events.jsonl").read_text().splitlines() if l.strip()]
    ci = [x for x in recs if x.get("kind") == "cancel_intent"]
    assert ci and ci[0]["actor"] == "timeout" and "timeout_s=0.5 (abort)" in ci[0]["reason"], "it names the limit that fired"
    st = json.loads((d2 / "status.json").read_text())
    assert st["state"] == "cancelled" and st["timed_out"] is True


LEAVES_A_WORKER = """
import subprocess, sys
w = subprocess.Popen([sys.executable, "-c", "import threading; threading.Event().wait()"])
open(sys.argv[1], "w").write(str(w.pid))
sys.exit(1)
"""


def test_tele_run_kills_what_is_left_of_the_workloads_group_once_its_leader_exits_and_marks_it(tmp_path):
    import select

    from research.telemetry import run as trun
    d = tmp_path / "a"
    d.mkdir()
    rc = trun.main(["--attempt-dir", str(d), "--stage", "t", "--no-sampler", "--no-classify", "--",
                    sys.executable, "-c", LEAVES_A_WORKER, str(tmp_path / "worker.pid")])
    assert rc == 1
    pid = int((tmp_path / "worker.pid").read_text())
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        fd = None
    if fd is not None:
        select.select([fd], [], [])                     # readable once the worker has exited
        os.close(fd)
    recs = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
    (m,) = [x for x in recs if x.get("kind") == "mark" and x.get("name") == "group_outlived_leader"]
    assert m["n"] == 1 and m["pids"][0].startswith(f"{pid} ")
    assert json.loads((d / "status.json").read_text())["state"] == "failed"


def test_a_run_on_a_machine_that_caps_sessions_needs_a_cap_of_its_own(tmp_path):
    """On node 1 (8 Oct 01:59Z) and cpu-1 (06:00Z) user-1001.slice sat at its MemoryMax, logind failed to create 266 login session
    scopes, and those logins' runs went on in ssh.service, which has no per-run cap. The machine's session-.scope.d MemoryMax is the
    cap a run expects; a run with no finite memory.max, or in a service's cgroup, has none of its own. A machine without the drop-in
    asks nothing."""
    from research.telemetry.cgroup import session_cap, uncapped

    drop = tmp_path / "session-.scope.d"
    assert session_cap(drop) is None
    drop.mkdir()
    (drop / "50-vy-job-memory.conf").write_text("# a comment\n[Scope]\nMemoryMax=900G\nMemorySwapMax=0\n")
    assert session_cap(drop) == "900G"
    (drop / "60-off.conf").write_text("[Scope]\nMemoryMax=infinity\n")
    assert session_cap(drop) is None, "the last one systemd reads wins"
    (drop / "60-off.conf").unlink()

    session = {"path": "/user.slice/user-1001.slice/session-9.scope", "memory_max_kind": "finite", "memory_max_bytes": 900 << 30}
    queued = {"path": "/user.slice/user-1001.slice/user@1001.service/app.slice/cluster-r1.scope", "memory_max_kind": "finite"}
    leaked = {"path": "/system.slice/ssh.service", "memory_max_kind": "finite", "memory_max_bytes": 64 << 30}
    bare = {"path": "/user.slice/user-1001.slice/user@1001.service/app.slice/x.scope", "memory_max_kind": "unlimited"}
    assert uncapped(session, "900G") is None and uncapped(queued, "900G") is None
    assert "ssh.service" in uncapped(leaked, "900G") and "no session scope" in uncapped(leaked, "900G")
    assert "unlimited" in uncapped(bare, "900G") and "--mem-gb" in uncapped(bare, "900G")
    assert uncapped(leaked, None) is None and uncapped(bare, None) is None


def test_tele_run_never_starts_a_workload_without_its_memory_cap(tmp_path, monkeypatch):
    """The guard runs before the spawn: status failed with the reason, a no_memory_cap mark, exit 75, and the workload never ran."""
    from research.telemetry import run as trun

    monkeypatch.setattr(trun, "session_cap", lambda: "900G")
    monkeypatch.setattr(trun.Cgroup, "limits", lambda self: {"path": "/system.slice/ssh.service", "memory_max_kind": "finite"})
    ran = tmp_path / "ran"
    d = tmp_path / "a"
    rc = trun.main(["--attempt-dir", str(d), "--stage", "t", "--no-sampler", "--no-classify", "--", sys.executable, "-c",
                    f"open({str(ran)!r}, 'w')"])
    assert rc == trun.EX_NO_MEMORY_CAP == 75 and not ran.exists()
    st = json.loads((d / "status.json").read_text())
    assert st["state"] == "failed" and st["rc"] == 75 and "ssh.service" in st["error"]
    recs = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
    assert [x["name"] for x in recs if x.get("kind") == "mark"] == ["no_memory_cap"]


def test_the_timeout_control_file_accepts_a_bare_number_and_ignores_what_it_cannot_parse(tmp_path, capsys):
    """An unparseable control file is reported once and otherwise ignored (never the reason a workload stops); a bare number is a
    timeout_s.  The run loop acts on whatever this returns (test above)."""
    from research.telemetry import run as trun

    ctl = trun._TimeoutControl(tmp_path)
    assert ctl.poll() is None
    (tmp_path / "control").mkdir()
    (tmp_path / "control" / "timeout_s").write_text("not json")
    assert ctl.poll() is None
    os.utime(tmp_path / "control" / "timeout_s", (1_600_000_000, 1_600_000_000))   # changed again, still unparseable: not said twice
    assert ctl.poll() is None and capsys.readouterr().err.count("ignoring unparseable") == 1
    (tmp_path / "control" / "timeout_s").write_text("30")
    assert ctl.poll() == {"timeout_s": 30.0}
    assert ctl.poll() is None, "unchanged: nothing new"


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


KILLED_LINE = "bash: line 1:  4242 Killed                  python -m verity_vllm.pipeline.build --row 3\n"


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


def _queue_attempt(tmp_path, run_id, *, sig, oom, unit=None):
    """A `research run --queue` attempt: its cgroup is the scope `cluster-<run id>.scope` capped at 8 GiB (job.json), which the
    sampler calls the pod's because the runner and the sampler share it with the workload."""
    d = tmp_path / run_id
    d.mkdir()
    path = f"/user.slice/user-1001.slice/user@1001.service/app.slice/{unit or f'cluster-{run_id}.scope'}"
    cg = [{**_cg(t, k), "cgroup": {**_cg(t, k)["cgroup"], "path": path}} for t, k in ((100, 0), (104, oom))]
    _attempt(d, rc=128 + sig, sig=sig, samples=cg, end=105.0)
    (d / "job.json").write_text(json.dumps({"ownership": {"job": run_id}, "environment": {"cgroup": {"path": path, "memory_max_bytes": 8 << 30}}}))
    return d


def test_a_queue_scope_systemd_stopped_after_an_oom_kill_is_an_oom_that_names_the_cap(tmp_path):
    doc = classify(_queue_attempt(tmp_path, "r20261005-064854-bd63", sig=15, oom=1))
    assert doc["class"] == "CGROUP_OOM_ATTRIBUTED" and doc["certainty"] == "established", doc["why"]
    assert "MemoryMax=8G" in doc["why"] and "--mem-gb" in doc["why"] and "systemd" in doc["why"]
    assert any(e["kind"] == "cgroup.queue_scope" and e["memory_max"] == "MemoryMax=8G" for e in doc["evidence"])
    killed = classify(_queue_attempt(tmp_path, "r2", sig=9, oom=1))
    assert killed["class"] == "CGROUP_OOM_ATTRIBUTED" and killed["certainty"] == "established" and "MemoryMax=8G" in killed["why"]


def test_a_sigterm_with_no_kill_or_outside_the_runs_own_scope_stays_unknown_signal(tmp_path):
    assert classify(_queue_attempt(tmp_path, "r1", sig=15, oom=0))["class"] == "UNKNOWN_SIGNAL"
    other = classify(_queue_attempt(tmp_path, "r3", sig=15, oom=1, unit="cluster-r9.scope"))
    assert other["class"] == "UNKNOWN_SIGNAL" and "MemoryMax" not in other["why"]


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


# ---- the sampler's PSS rounds (network accounting's friction note, 3 Oct: a 100 GB process's smaps_rollup read slowed its engine) ------------

GB = 10 ** 9


def _proc(pid, rss):
    return {"pid": pid, "starttime_ticks": 7, "comm": "w", "rss_bytes": rss, "rss_anon_bytes": rss - rss // 4, "rss_file_bytes": rss // 4,
            "rss_shmem_bytes": 0}


class _Clock:
    """The sampler's clock: the test sets it, and a fake read that takes time moves it."""

    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def _sampler(tmp_path, monkeypatch, table, read, *flags):
    """A Sampler over the process table `table` (re-read every round, so a test may change it), whose smaps_rollup reads are `read`."""
    from research.telemetry import sample

    monkeypatch.delenv("TELEMETRY_STREAM_SUFFIX", raising=False)
    monkeypatch.setattr(sample, "snapshot_tree", lambda root_pid, pgid, max_pids: (list(table), False, len(table), []))
    monkeypatch.setattr(sample, "pss", read)
    monkeypatch.setattr(sample, "gpu_sample", lambda timeout: {"available": False, "reason": "no GPU in this test"})
    ap = argparse.ArgumentParser()
    sample.add_args(ap)
    return sample.Sampler(ap.parse_args(["--attempt-dir", str(tmp_path), *flags]))


def _records(d, kind):
    return [r for r in map(json.loads, (d / "resources.jsonl").read_text().splitlines()) if r["kind"] == kind]


def test_pss_every_0_turns_pss_off(tmp_path, monkeypatch):
    """`--pss-every 0` used to mean a PSS round every round; it is off: no read, no pss record, and pss is not an instrument."""
    from research import cli as rcli

    opened = []
    s = _sampler(tmp_path, monkeypatch, [_proc(10, GB)], lambda pid: opened.append(pid) or {"pid": pid, "pss_bytes": GB}, "--pss-every", "0")
    for _ in range(3):
        s.round()
    s.round(final=True)
    assert opened == [] and _records(tmp_path, "pss") == [] and len(_records(tmp_path, "process")) == 4
    for every, has in (("0", False), ("5", True)):
        a = rcli._harness_parser("research run").parse_args(["--attempt-dir", str(tmp_path), "--stage", "s", "--pss-every", every])
        instruments = rcli.build_job_from(a, ["true"])["telemetry"]["instruments"]
        assert ("pss" in instruments) is has and "process" in instruments, (every, instruments)


def test_a_process_over_pss_max_rss_gb_is_never_read_in_any_round(tmp_path, monkeypatch):
    """smaps_rollup of a 91-120 GB vLLM engine took 1.6-2.7 s under its mmap lock, and the engine's steps behind it had a 2-4.5x longer
    p99. A process over --pss-max-rss-gb (default 8) is recorded from its status RSS instead, in the cadence, topology-change and
    final rounds alike, and counted `skipped`, not `denied` (which stays what the reader refused)."""
    from research.telemetry import sample

    clk = _Clock(1000.0)
    monkeypatch.setattr(sample, "clock", clk)
    opened = []

    def read(pid):
        opened.append(pid)
        return {"pid": pid, "pss_bytes": None, "reason": "permission denied"} if pid == 13 else {"pid": pid, "pss_bytes": GB // 2}

    table = [_proc(10, 97_300_000_000), _proc(12, GB), _proc(13, GB)]
    s = _sampler(tmp_path, monkeypatch, table, read, "--pss-every", "60")
    s.round()                                     # the first round
    clk.t = 1006.0
    table.append(_proc(14, GB))
    s.round()                                     # a child appeared: a topology-change round, 54 s before the cadence's
    clk.t = 1007.0
    s.round()                                     # nothing due
    clk.t = 1008.0
    s.round(final=True)
    recs = _records(tmp_path, "pss")
    assert len(recs) == 3 and 10 not in opened and opened.count(12) == 3 and opened.count(14) == 2
    for r in recs:
        assert r["processes"][0] == {
            "pid": 10, "pss_bytes": None, "skipped": True, "rss_bytes": 97_300_000_000, "rss_anon_bytes": 72_975_000_000,
            "rss_file_bytes": 24_325_000_000, "rss_shmem_bytes": 0,
            "reason": "skipped: rss 97.30 GB over --pss-max-rss-gb 8 (smaps_rollup holds the process's mmap lock for seconds)"}
        assert r["skipped"] == 1 and r["denied"] == 1 and r["pss_sum_bytes"] == GB // 2 * (len(r["processes"]) - 2)
    assert [len(r["processes"]) for r in recs] == [3, 4, 4]


def test_a_process_whose_status_has_no_rss_is_judged_and_recorded_by_its_stat_rss(tmp_path, monkeypatch):
    """/proc/<pid>/status without VmRSS (a process exiting between its stat and status reads): the threshold reads the RSS of
    /proc/<pid>/stat, and the skipped record carries that figure, not only the status fields it lacks."""
    opened = []
    row = {"pid": 10, "starttime_ticks": 7, "comm": "w", "rss_bytes_stat": 8_040_000_000, "status_reason": "gone"}
    s = _sampler(tmp_path, monkeypatch, [row, _proc(12, GB)], lambda pid: opened.append(pid) or {"pid": pid, "pss_bytes": GB // 2})
    s.round()
    [rec] = _records(tmp_path, "pss")
    assert opened == [12] and rec["skipped"] == 1 and rec["denied"] == 0
    p = rec["processes"][0]
    assert p["skipped"] is True and p["rss_bytes"] is None and p["rss_bytes_stat"] == 8_040_000_000
    assert p["reason"].startswith("skipped: rss 8.04 GB over --pss-max-rss-gb 8 ")
    assert "rss_bytes_stat" not in rec["processes"][1]


def test_research_run_passes_the_pss_flags_through_tele_run_to_the_sampler(tmp_path):
    from research import cli as rcli
    from research.telemetry import run as trun, sample

    ap = argparse.ArgumentParser()
    sample.add_args(ap)
    for flags, want in (([], (5.0, 8.0)), (["--pss-every", "0", "--pss-max-rss-gb", "2.5"], (0.0, 2.5))):
        a = rcli._harness_parser("research run").parse_args(["--attempt-dir", str(tmp_path), "--stage", "s", *flags])
        argv = trun._sampler_argv(a, tmp_path, 4242)
        assert argv[:4] == [*trun._tele_argv(), "sample"]
        got = ap.parse_args(argv[4:])
        assert (got.pss_every, got.pss_max_rss_gb, got.root_pid) == (*want, 4242)


def test_a_slow_read_holds_that_process_off_for_200_times_its_length(tmp_path, monkeypatch):
    """Under --pss-max-rss-gb a read can still take long. A process is read again no sooner than 200x its last read took, so the reads
    hold its mmap lock at most 0.5% of its time (a p99 shows a perturbation over 1%), while the other processes keep the cadence."""
    from research.telemetry import sample

    clk = _Clock(1000.0)
    monkeypatch.setattr(sample, "clock", clk)
    opened = []

    def read(pid):
        opened.append((pid, clk.t))
        clk.t += 0.5 if pid == 11 else 0.001
        return {"pid": pid, "pss_bytes": GB // 2}

    s = _sampler(tmp_path, monkeypatch, [_proc(11, 2 * GB), _proc(12, GB)], read, "--pss-every", "5")
    for i in range(22):
        clk.t = 1000.0 + 5 * i
        s.round()
    recs = _records(tmp_path, "pss")
    assert len(recs) == 22 and [t for p, t in opened if p == 11] == [1000.0, 1100.0] and sum(p == 12 for p, _ in opened) == 22
    assert [r["skipped"] for r in recs] == [0] + [1] * 19 + [0, 1] and all(r["denied"] == 0 for r in recs)
    assert recs[0]["processes"][0]["read_s"] == 0.5
    held = recs[1]["processes"][0]
    assert held["skipped"] is True and held["pss_bytes"] is None and held["rss_bytes"] == 2 * GB and held["rss_file_bytes"] == GB // 2
    assert held["reason"] == ("skipped: its last smaps_rollup read took 0.500 s, so the next waits 200x that (95.0 s more): reads hold "
                              "its mmap lock at most 1/200 of its time")


def test_a_failed_process_snapshot_keeps_the_holds(tmp_path, monkeypatch):
    """A round whose process snapshot failed knows no process. Pruning the holds to it released every held process, so the next
    round read them again at once; holds are pruned only to a snapshot that succeeded, which still drops the processes that exited."""
    from research.telemetry import sample

    clk = _Clock(1000.0)
    monkeypatch.setattr(sample, "clock", clk)
    opened = []

    def read(pid):
        opened.append(pid)
        clk.t += 0.5 if pid == 11 else 0.001
        return {"pid": pid, "pss_bytes": GB // 2}

    def no_proc(root_pid, pgid, max_pids):
        raise OSError("no /proc")

    table = [_proc(11, 2 * GB), _proc(12, GB)]
    s = _sampler(tmp_path, monkeypatch, table, read, "--pss-every", "5")
    snapshot = sample.snapshot_tree
    s.round()
    monkeypatch.setattr(sample, "snapshot_tree", no_proc)
    clk.t = 1005.0
    s.round()
    monkeypatch.setattr(sample, "snapshot_tree", snapshot)
    table.pop()
    clk.t = 1010.0
    s.round()
    assert s.errors["process"] == 1 and opened == [11, 12] and set(s.pss_hold) == {(11, 7)}
    held = _records(tmp_path, "pss")[-1]["processes"]
    assert [p["pid"] for p in held] == [11] and held[0]["skipped"] is True and "(90.0 s more)" in held[0]["reason"]


# ---- source identity: a namespace package spans every portion ----------------------------------------------------------------------------------

def _portion(tree: Path, root: str, name: str, module: str) -> str:
    (tree / root / name / module).mkdir(parents=True)
    (tree / root / name / module / "__init__.py").write_text("")
    (tree / "EXPORT.json").write_text(json.dumps({"commit": tree.name * 8}))
    return str(tree / root)


def test_a_namespace_with_a_portion_from_another_tree_is_a_mismatch(tmp_path):
    """Recording only the first portion passed a namespace whose second portion came from another checkout."""
    from research.telemetry import source_identity as sid

    ours, other = tmp_path / "ours", tmp_path / "other"
    src, lib, foreign = _portion(ours, "src", "nsprobe", "ir"), _portion(ours, "lib", "nsprobe", "ml"), _portion(other, "src", "nsprobe", "ml")
    both = sid.resolve_module("nsprobe", path=[src, lib])
    assert both["repo_root"] == str(ours.resolve()) and [p["repo_root"] for p in both["portions"]] == [str(ours.resolve())] * 2
    assert sid.check({"nsprobe": both}, str(ours)) == []
    mixed = sid.resolve_module("nsprobe", path=[src, foreign])
    assert mixed["resolved_path"] == str((ours / "src" / "nsprobe").resolve()) and mixed["repo_root"] is None
    [m] = sid.check({"nsprobe": mixed}, str(ours))
    assert "namespace portion(s) outside the requested tree" in m["why"] and str((other / "src" / "nsprobe").resolve()) in m["why"]


def test_a_cancel_also_signals_the_runs_workers_that_left_its_group_but_never_the_runner(tmp_path, monkeypatch):
    """INF-48/INF-53: circuit_check's pool (17 Oct 11:21Z, cpu-2) and vLLM's TP workers (vy-glm53-eu2) setsid out of the workload's
    group and outlived `research cancel`; they still carry the run's RESEARCH_RUN_DIR."""
    from research.telemetry import cancel
    d = _attempt(tmp_path, rc=None, sig=None, samples=[], pid=100)
    want = str(d.resolve())
    procs = {100: {"pid": 100, "comm": "bash", "ppid": 50, "pgrp": 100, "starttime_ticks": 1, "state": "S"},
             50: {"pid": 50, "comm": "python3", "ppid": 1, "pgrp": 50, "starttime_ticks": 1, "state": "S"},     # the runner
             300: {"pid": 300, "comm": "python3", "ppid": 100, "pgrp": 300, "starttime_ticks": 2, "state": "S"},  # setsid worker
             400: {"pid": 400, "comm": "python3", "ppid": 1, "pgrp": 400, "starttime_ticks": 3, "state": "S"}}   # another run's
    env = {100: want, 50: want, 300: want, 400: "/elsewhere"}
    monkeypatch.setattr(cancel, "stat_fields", lambda pid: procs.get(pid))
    monkeypatch.setattr(cancel, "run_dir_of", lambda pid: env.get(pid))
    monkeypatch.setattr(cancel, "pgid_members", lambda pgid: [{"pid": 100, "comm": "bash", "starttime_ticks": 1}])
    monkeypatch.setattr(cancel.os, "listdir", lambda p: [str(k) for k in procs])
    killed = []
    monkeypatch.setattr(cancel.os, "killpg", lambda g, s: killed.append(("pg", g)))
    monkeypatch.setattr(cancel.os, "kill", lambda p, s: killed.append(("pid", p)))
    a = argparse.Namespace(actor="manual", attempt_dir=str(d), fallback=None, target_pid=None, target_pgid=100, target_pod=None,
                           expect_starttime=None, expect_comm=None, expect_run_dir=str(d), expect_nonempty=False, reason="r", signal=15,
                           project=None, job=None, attempt=None, controller=None, deliver=True, confirm_exit_s=0.0)
    assert cancel.cancel(a) == 0
    assert killed == [("pg", 100), ("pid", 300)], "the group, then its setsid worker; never the runner (50) or another run (400)"
    result = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()][-1]
    assert result["strays_total"] == 1 and result["strays"][0]["pid"] == 300 and result["strays_signalled"] == 1


# ---- the run's cgroup scope: what research cancel and a stage timeout end (INF-93) -------------------------------------------------------

B = 5_000_000                       # past any pid_max, so no fake pid is this test's own
RID = "r20261019-002224-7f15"
APP = "/user.slice/user-1001.slice/user@1001.service/app.slice"
QUEUE, LEASE, SESSION = f"{APP}/cluster-{RID}.scope", f"{APP}/gpu-lease-{B + 100}.scope", "/user.slice/user-1001.slice/session-5.scope"


class _World:
    """A fake /proc and cgroup v2 tree for scope.Sweep: pid -> (ppid, pgrp, session, cgroup, RESEARCH_RUN_DIR, argv).  `kill`
    records each signal; a pid in `ignores_term` survives SIGTERM and one in `unkillable` survives everything (a D-state task)."""

    def __init__(self, root: Path, procs: dict, *, ignores_term=(), unkillable=()):
        self.root, self.procs, self.sent = root, dict(procs), []
        self.ignores_term, self.unkillable = set(ignores_term), set(unkillable)
        self.start = {p: 7 for p in procs}
        self._write()

    def _write(self):
        for f in self.root.rglob("cgroup.procs"):
            f.write_text("")
        for p, (_, _, _, cg, _, _) in self.procs.items():
            d = self.root / cg.lstrip("/")
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "cgroup.procs", "a") as f:
                f.write(f"{p}\n")

    def stat(self, p):
        if p not in self.procs:
            return None
        ppid, pgrp, sess, _, _, argv = self.procs[p]
        return {"pid": p, "comm": Path(argv[0]).name, "state": "S", "ppid": ppid, "pgrp": pgrp, "session": sess,
                "starttime_ticks": self.start[p]}

    def pids(self):
        return sorted(self.procs)

    def cgroup(self, p):
        return self.procs[p][3] if p in self.procs else None

    def env_of(self, p):
        return self.procs[p][4] if p in self.procs else None

    def cmdline(self, p):
        return self.procs[p][5] if p in self.procs else []

    def kill(self, p, sig):
        import signal
        if p not in self.procs:
            raise ProcessLookupError(p)
        self.sent.append((p, signal.Signals(sig).name))
        if p not in self.unkillable and (sig == signal.SIGKILL or p not in self.ignores_term):
            del self.procs[p]
            for q, (ppid, *rest) in list(self.procs.items()):
                if ppid == p:
                    self.procs[q] = (B + 2, *rest)          # reparented to the subreaper, systemd --user
            self._write()

    def sweep(self, leader, run_dir, **kw):
        from research.telemetry import scope
        return scope.Sweep(leader, run_dir, stat=self.stat, pids=self.pids, cgroup=self.cgroup, env_of=self.env_of, cmdline=self.cmdline,
                           root=self.root, kill=self.kill, **kw)


@pytest.fixture
def fake_time(monkeypatch):
    """scope's clock, advanced only by its own sleeps; `on_sleep` runs at each one."""
    from research.telemetry import scope
    t, hooks = [0.0], []
    monkeypatch.setattr(scope, "clock", lambda: t[0])

    def sleep(dt):
        t[0] += dt
        for h in hooks:
            h()
    monkeypatch.setattr(scope, "sleep", sleep)
    return hooks


def _running(d: Path, *, runner: int, pgid: int, start: int = 7) -> None:
    """status.json as the runner `runner` (start time `start`) writes it when it starts the workload, the leader of group `pgid`."""
    (d / "status.json").write_text(json.dumps({"state": "running", "pid": pgid, "pgid": pgid, "transitions": [
        {"state": "running", "stream": f"h:{runner}:{start}", "pid": pgid, "pgid": pgid}]}))


def _vllm_world(tmp_path, **kw):
    """vy-glm53-eu, 19 Oct 01:20Z: a queued run's runner and sampler in its queue scope, its workload under gpu-lease (the leader) whose
    scope holds vLLM's engine and eight TP rank workers that setsid and were reparented to systemd --user, their environment unreadable;
    beside it another run's process and a system daemon."""
    rd = str(tmp_path / RID)
    py = ["python3", "-c", "..."]
    procs = {B + 2: (1, B + 2, B + 2, f"/user.slice/user-1001.slice/user@1001.service/init.scope", None, ["systemd", "--user"]),
             B + 50: (B + 2, B + 50, B + 50, QUEUE, rd, ["python3", "-m", "research", "run", "--runs-dir", "x", "--id", RID]),
             B + 51: (B + 50, B + 51, B + 51, QUEUE, rd, ["python3", "-m", "research.telemetry", "sample"]),
             B + 100: (B + 50, B + 100, B + 100, QUEUE, rd, ["bash", "/usr/local/bin/gpu-lease", "8", "--wait", "--", "vllm"]),
             B + 101: (B + 100, B + 100, B + 100, LEASE, rd, ["python3", "-m", "vllm.entrypoints.openai.api_server"]),
             **{B + 300 + k: (B + 2, B + 300 + k, B + 300 + k, LEASE, None, py) for k in range(8)},
             B + 400: (B + 2, B + 400, B + 400, SESSION, "/workspace/research/runs/r20261019-010101-aaaa", py),
             B + 450: (1, B + 450, B + 450, "/system.slice/nvidia-persistenced.service", None, ["nvidia-persistenced"])}
    return _World(tmp_path / "cgroup", procs, **kw), rd


def test_a_sweep_ends_every_process_of_the_runs_own_scopes_and_never_the_runner(tmp_path, fake_time):
    """The run's queue scope and its gpu-lease scope are its own: the rank workers there are ended though they left the group, the
    session and the process tree and their environment can't be read; four ignore SIGTERM and get SIGKILL after the grace.  The
    runner, its sampler and systemd --user (where the workers were reparented) are spared, and so is everything outside the run."""
    from research.telemetry import scope
    w, rd = _vllm_world(tmp_path, ignores_term=[B + 300, B + 301, B + 302, B + 303])
    rec = w.sweep(B + 100, rd).end(scope.GRACE_S)
    term = {p for p, s in w.sent if s == "SIGTERM"}
    assert term == {B + 100, B + 101, *range(B + 300, B + 308)}
    assert [p for p, s in w.sent if s == "SIGKILL"] == [B + 300, B + 301, B + 302, B + 303]
    assert set(w.procs) == {B + 2, B + 50, B + 51, B + 400, B + 450}
    assert set(rec["scopes"]) == {QUEUE, LEASE} and rec["fallback"] is None
    assert rec["SIGTERM"]["n"] == 10 and rec["SIGKILL"]["n"] == 4 and rec["empty"] is True and rec["left"] == []
    assert (tmp_path / "cgroup" / LEASE.lstrip("/") / "cgroup.procs").read_text() == ""


def test_a_run_in_no_scope_of_its_own_is_ended_by_its_group_session_and_run_dir_and_says_so(tmp_path, fake_time):
    """A direct run in a person's ssh login: the session scope holds a tmux too, so only the run's processes are ended (the group,
    a setsid worker naming its run dir) and the record says why."""
    rd = str(tmp_path / RID)
    py = ["python3", "-c", "..."]
    w = _World(tmp_path / "cgroup", {
        B + 50: (B + 600, B + 50, B + 600, SESSION, None, ["python3", "-m", "research", "run", "--id", RID]),
        B + 100: (B + 50, B + 100, B + 100, SESSION, rd, py),
        B + 101: (B + 100, B + 100, B + 100, SESSION, rd, py),
        B + 300: (1, B + 300, B + 300, SESSION, rd, py),
        B + 600: (1, B + 600, B + 600, SESSION, None, ["tmux"]),
        B + 601: (B + 600, B + 601, B + 601, SESSION, None, ["bash"])})
    rec = w.sweep(B + 100, rd).end(20.0)
    assert {p for p, _ in w.sent} == {B + 100, B + 101, B + 300} and {s for _, s in w.sent} == {"SIGTERM"}
    assert set(w.procs) == {B + 50, B + 600, B + 601}
    assert rec["scopes"] == {} and "no scope of the run's own" in rec["fallback"] and "session-5.scope" in rec["fallback"]
    assert "SIGKILL" not in rec and rec["empty"] is True


def test_a_sweep_never_kills_a_reused_pid_and_names_what_survived(tmp_path, fake_time):
    """A worker that ignored SIGTERM exits during the grace and its pid goes to another user's process, which SIGKILL must miss; a
    worker in uninterruptible sleep survives SIGKILL and is named as left, the scope not empty."""
    w, rd = _vllm_world(tmp_path, ignores_term=[B + 300, B + 301], unkillable=[B + 301])

    def reuse():
        if w.procs.get(B + 300, (None,))[0] == B + 2 and w.start[B + 300] == 7:
            w.procs[B + 300] = (1, B + 300, B + 300, "/system.slice/cron.service", None, ["cron"])
            w.start[B + 300] = 99
            w._write()
    fake_time.append(reuse)
    rec = w.sweep(B + 100, rd).end(20.0)
    assert (B + 300, "SIGKILL") not in w.sent and B + 300 in w.procs
    assert (B + 301, "SIGKILL") in w.sent and rec["empty"] is False and rec["left"] == [f"{B + 301} python3"]


def test_a_cancel_with_a_grace_terms_the_runs_scope_and_asks_the_runner_to_end_the_rest(tmp_path, monkeypatch):
    """`research cancel` (cancel-intent --grace-s 20): the group's SIGTERM, then the same to the run's scope less the group it already
    reached; it SIGKILLs nothing itself, and control/cancel tells the runner the grace and the scopes."""
    from research.telemetry import cancel, scope
    w, rd = _vllm_world(tmp_path)
    Path(rd).mkdir()
    _running(Path(rd), runner=B + 50, pgid=B + 100)
    for name, fn in (("stat_fields", w.stat), ("list_pids", w.pids), ("cgroup_of", w.cgroup), ("run_dir_of", w.env_of),
                     ("cmdline_of", w.cmdline)):
        monkeypatch.setattr(cancel, name, fn)
    monkeypatch.setattr(cancel, "pgid_members", lambda g: [{"pid": p, "comm": "x", "starttime_ticks": 7} for p in w.pids()
                                                           if w.stat(p)["pgrp"] == g])
    monkeypatch.setattr(scope, "mounts", lambda: {"v2": str(w.root)})
    group = []
    monkeypatch.setattr(cancel.os, "killpg", lambda g, s: group.append((g, s)))
    monkeypatch.setattr(cancel.os, "kill", w.kill)
    a = argparse.Namespace(actor="manual", attempt_dir=rd, fallback=None, target_pid=None, target_pgid=B + 100, target_pod=None,
                           expect_starttime=None, expect_comm=None, expect_run_dir=rd, expect_nonempty=False, reason="r", signal=15,
                           project=None, job=None, attempt=None, controller=None, deliver=True, confirm_exit_s=0.0, grace_s=20.0)
    assert cancel.cancel(a) == 0
    assert group == [(B + 100, 15)] and {p for p, _ in w.sent} == set(range(B + 300, B + 308))
    assert {s for _, s in w.sent} == {"SIGTERM"}
    ctl = json.loads((Path(rd) / "control" / "cancel").read_text())
    assert ctl["grace_s"] == 20.0 and ctl["actor"] == "manual" and ctl["scopes"] == sorted([QUEUE, LEASE])
    result = [json.loads(l) for l in (Path(rd) / "events.jsonl").read_text().splitlines()][-1]
    assert result["kind"] == "cancel_result" and result["scope"]["runner_kills_after_s"] == 20.0
    assert result["scope"]["SIGTERM"]["n"] == 8 and "left" not in result["scope"]


#: The workload: a setsid worker that ignores SIGTERM (as vLLM's rank workers and xdist's did), its pid published once it does; then
#: argv[2], if given, is written as research cancel's request; the workload waits on its worker.
SETSID_WORKER = """
import os, subprocess, sys
pidf = sys.argv[1]
w = subprocess.Popen([sys.executable, "-c", "import os, signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                      "print(os.getpid(), flush=True); signal.pause()"], stdout=subprocess.PIPE, start_new_session=True)
open(pidf + ".tmp", "w").write(w.stdout.readline().decode().strip())
os.replace(pidf + ".tmp", pidf)
if len(sys.argv) > 2:
    d = os.environ["RESEARCH_RUN_DIR"]
    os.makedirs(os.path.join(d, "control"), exist_ok=True)
    open(os.path.join(d, "control", "cancel"), "w").write(sys.argv[2])
w.wait()
"""


def _gone(pid: int) -> None:
    import select
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    select.select([fd], [], [])
    os.close(fd)


def test_a_stage_timeout_ends_the_runs_workers_outside_its_group_and_records_them(tmp_path, monkeypatch):
    """The stage clock reads past the timeout once the worker is up, then past the grace: the group's SIGTERM ends the workload, the
    worker that left its group ignores it and is SIGKILLed, and status.json `scope_end` and a `scope_ended` mark say so."""
    from research.telemetry import run as trun
    d, pidf = tmp_path / "a", tmp_path / "worker.pid"
    d.mkdir()
    ticks = iter(range(1000))
    monkeypatch.setattr(trun, "clock", lambda: 0.0 if not pidf.exists() else 100.0 + 10 * next(ticks))
    rc = trun.main(["--attempt-dir", str(d), "--stage", "t", "--timeout", "50", "--grace", "5", "--no-sampler", "--no-classify", "--",
                    sys.executable, "-c", SETSID_WORKER, str(pidf)])
    pid = int(pidf.read_text())
    _gone(pid)
    assert rc != 0
    st = json.loads((d / "status.json").read_text())
    assert st["state"] == "cancelled" and st["timed_out"] is True
    se = st["scope_end"]
    assert se["empty"] is True and any(x.startswith(f"{pid} ") for x in se["SIGKILL"]["pids"])
    assert any(x.startswith(f"{pid} ") for x in se["SIGTERM"]["pids"]), "the worker got the timeout's SIGTERM before its SIGKILL"
    recs = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
    assert [x["name"] for x in recs if x.get("kind") == "mark" and x.get("name") == "scope_ended"] == ["scope_ended"]
    assert [x["actor"] for x in recs if x.get("kind") == "cancel_intent"][0] == "timeout"


def test_the_runner_ends_what_a_cancel_left_once_its_grace_is_over_and_records_it(tmp_path, monkeypatch):
    """research cancel's request with a grace of 0: the runner marks it, SIGKILLs the workload (recorded as a manual cancel) and then
    the run's worker outside the group, and records both in status.json."""
    from research.telemetry import run as trun
    d, pidf = tmp_path / "a", tmp_path / "worker.pid"
    d.mkdir()
    req = json.dumps({"grace_s": 0, "actor": "manual", "reason": "superseded", "scopes": []})
    rc = trun.main(["--attempt-dir", str(d), "--stage", "t", "--no-sampler", "--no-classify", "--",
                    sys.executable, "-c", SETSID_WORKER, str(pidf), req])
    pid = int(pidf.read_text())
    _gone(pid)
    assert rc == 128 + 9
    recs = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
    (seen,) = [x for x in recs if x.get("kind") == "mark" and x.get("name") == "cancel_seen"]
    assert seen["actor"] == "manual" and seen["grace_s"] == 0 and seen["reason"] == "superseded"
    kills = [x for x in recs if x.get("kind") == "cancel_intent"]
    assert [(x["actor"], x["signal"]) for x in kills] == [("manual", 9)]
    se = json.loads((d / "status.json").read_text())["scope_end"]
    assert se["empty"] is True and any(x.startswith(f"{pid} ") for x in se["SIGKILL"]["pids"])


# ---- who ends a cancelled run: control/cancel, claimed once by the cancel (its request) or by the runner (closed) -----------------------

#: The workload: a child that calls setsid itself and ignores SIGTERM, as the vllm.build stages under GNU `timeout` did (each in a
#: process group of its own; vy-glm53-cpu-2, 20 Oct 06:12Z), writes its pid to the gate argv[1]; the workload waits on it.
SETSID_CHILD = """
import subprocess, sys
subprocess.Popen([sys.executable, "-c", "import os, signal, sys; os.setsid(); signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                  "open(sys.argv[1], 'w').write(f'{os.getpid()}\\\\n'); signal.pause()", sys.argv[1]]).wait()
"""


def _cancel_argv(d: Path, pgid: int) -> list[str]:
    return ["--actor", "manual", "--reason", "superseded", "--target-pgid", str(pgid), "--expect-run-dir", str(d), "--attempt-dir", str(d),
            "--deliver", "--grace-s", "20"]


def test_a_cancel_whose_runner_is_gone_sigkills_the_runs_setsid_child_itself_and_records_it(tmp_path, monkeypatch, fake_time):
    """No live runner to end the run: the cancel sends the SIGTERM, waits the grace on scope's clock, SIGKILLs the setsid child that
    ignored it and records that in its cancel_result and in status.json `scope_end` (`"by": "cancel"`)."""
    import signal

    from research.telemetry import cancel
    from waits import wait_exit
    d = tmp_path / "a"
    d.mkdir()
    gate = Gate(tmp_path / "child.pid")
    workload = subprocess.Popen([sys.executable, "-c", SETSID_CHILD, str(gate.path)], start_new_session=True,
                                env={**os.environ, "RESEARCH_RUN_DIR": str(d.resolve())})
    child = int(gate.readline())
    child_fd = os.pidfd_open(child)
    _running(d, runner=B + 50, pgid=workload.pid)
    killed, real_kill = [], os.kill

    def kill(pid, sig):
        real_kill(pid, sig)
        if sig == signal.SIGKILL:
            killed.append(pid)
    monkeypatch.setattr(cancel.os, "kill", kill)
    fake_time.append(lambda: [wait_exit(p) for p in killed])
    try:
        assert cancel.main(_cancel_argv(d, workload.pid)) == 0
        assert workload.wait() == -signal.SIGTERM
        assert killed == [child], "the cancel SIGKILLs the child that ignored its SIGTERM"
        wait_exit(child)
    finally:
        try:
            signal.pidfd_send_signal(child_fd, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.close(child_fd)
        gate.close()
    result = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()][-1]
    sc = result["scope"]
    assert sc["runner_kills_after_s"] is None and sc["cancel_killed_after_s"] == 20 and sc["why"] == "the run's runner is gone"
    assert len(sc["SIGKILL"]["pids"]) == 1 and sc["SIGKILL"]["pids"][0].startswith(f"{child} ")
    assert sc["empty"] is True and sc["status_json"] == "written"
    se = json.loads((d / "status.json").read_text())["scope_end"]
    assert se["by"] == "cancel" and se["empty"] is True and se["SIGKILL"]["n"] == 1


def test_a_cancel_after_the_runner_closed_control_cancel_ends_the_run_itself(tmp_path, monkeypatch, fake_time):
    """The runner, finding no request when it looked, claimed control/cancel as closed and ends nothing more: the cancel SIGKILLs
    what ignores its SIGTERM itself, and leaves status.json to the runner, which is still live."""
    from research.telemetry import cancel, scope
    from research.telemetry import run as trun
    w, rd = _vllm_world(tmp_path, ignores_term=[B + 300, B + 301])
    Path(rd).mkdir()
    _running(Path(rd), runner=B + 50, pgid=B + 100)
    assert trun._close_cancel(Path(rd)) is None and trun._close_cancel(Path(rd)) is None, "closed once, and stays closed"
    for name, fn in (("stat_fields", w.stat), ("list_pids", w.pids), ("cgroup_of", w.cgroup), ("run_dir_of", w.env_of),
                     ("cmdline_of", w.cmdline)):
        monkeypatch.setattr(cancel, name, fn)
    monkeypatch.setattr(cancel, "pgid_members", lambda g: [{"pid": p, "comm": "x", "starttime_ticks": 7} for p in w.pids()
                                                           if w.stat(p)["pgrp"] == g])
    monkeypatch.setattr(scope, "mounts", lambda: {"v2": str(w.root)})
    monkeypatch.setattr(cancel.os, "killpg", lambda g, s: None)
    monkeypatch.setattr(cancel.os, "kill", w.kill)
    assert cancel.main(_cancel_argv(Path(rd), B + 100)) == 0
    assert [p for p, s in w.sent if s == "SIGKILL"] == [B + 100, B + 101, B + 300, B + 301]
    assert set(w.procs) == {B + 2, B + 50, B + 51, B + 400, B + 450}, "never the runner, its sampler or another run"
    assert "scopes" not in json.loads((Path(rd) / "control" / "cancel").read_text()), "the runner's closed claim stays"
    sc = [json.loads(l) for l in (Path(rd) / "events.jsonl").read_text().splitlines()][-1]["scope"]
    assert sc["why"] == "the runner had recorded the workload's exit before the cancel" and sc["empty"] is True
    assert sc["status_json"] == "not written: its runner is live and writes status.json"
    assert "scope_end" not in json.loads((Path(rd) / "status.json").read_text())
