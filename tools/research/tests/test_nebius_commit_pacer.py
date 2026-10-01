"""Node 1's Commit pacer (commit_pacer.py) keeps circuits' rule for releasing Commits in `deployments-gpu` (release.py's, 4:33 PM
PDT 30 Sep): one test per clause of the rule, what gate mode marks Ready and what it holds, which Workloads are Commits, which
failed Commits' bundles cleanup may delete, and the unit and manifests it ships with."""
from __future__ import annotations

import configparser
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from research.pods import registry

yaml = pytest.importorskip("yaml")
NEBIUS = Path(registry.__file__).resolve().parent / "nebius"
LLAMA = "llama32-1b__bf16__rtxpro6000__tp1__b{b}__i256__o32__mixed__greedy__bi-eager"
QWEN7 = "qwen25-7b__bf16__rtxpro6000__tp1__b{b}__i1024__o128__mixed__greedy__bi-eager"
PHI3 = "phi3-mini__bf16__rtxpro6000__tp1__b{b}__i1024__o128__mixed__greedy__bi-eager"
L1 = LLAMA.format(b=1)
T0 = 1_790_812_800.0   # 2026-10-01T00:00:00Z


def _load(path: Path, name: str, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def pacer(tmp_path, monkeypatch):
    monkeypatch.setenv("PACER_STATE", str(tmp_path / "state"))
    for var in ("PACER_SCOPE", "PACER_MODE", "PACER_MAX_INFLIGHT", "PACER_MAX_BIG", "PACER_DISK_PAUSE_PCT", "PACER_CAP_GB",
                "PACER_CAP_LOW_GB", "PACER_CAP_DROP_PCT"):
        monkeypatch.delenv(var, raising=False)
    return _load(NEBIUS / "commit_pacer.py", "commit_pacer", monkeypatch)


def _wl(name: str, cov: str = "cov-g055", row: str = L1, *, ws: str = "n2-build", stage: bool = True,
        active: bool = True, reserved: bool = False, admitted: bool = False, check: str | None = None, queue="deployments-gpu",
        sweep: str | None = None) -> dict:
    labels = {"verity.dev/workstream": ws, **({"verity.dev/stage": "commit"} if stage else {})}
    conds = ([{"type": "QuotaReserved", "status": "True"}] if reserved or admitted else []) + \
            ([{"type": "Admitted", "status": "True"}] if admitted else [])
    env = [{"name": "ROW", "value": row}, {"name": "SWEEP_DIR", "value": sweep or f"/workspace/jobs/cov/{cov}"}]
    return {"metadata": {"name": name, "ownerReferences": [{"kind": "Job", "name": f"job-{name}"}]},
            "spec": {"queueName": queue, "active": active,
                     "podSets": [{"name": "main", "template": {"metadata": {"labels": labels},
                                                               "spec": {"containers": [{"name": "main", "env": env}]}}}]},
            "status": {"conditions": conds, **({"admission": {"clusterQueue": queue}} if reserved or admitted else {}),
                       **({"admissionChecks": [{"name": "commit-pacer", "state": check}]} if check else {})}}


def _job(name: str, key: str, template: str = "config-run", task: str = "1") -> dict:
    return {"metadata": {"name": f"job-{name}", "labels": {"verity.dev/task": task},
                         "annotations": {"verity.dev/item": key, "verity.dev/template": f"{template}@abc123"}}}


def _view(pacer, name, cov="cov-g055", row=L1, ws="n2-build", **kw):
    return pacer.view(_wl(name, cov, row, ws=ws, **kw), _job(name, f"{ws}/{cov}"))


def _released(plan) -> list[str]:
    return [w["name"] for verb, w, _ in plan["acts"] if verb == "activate"]


# ---- the rule, clause by clause -------------------------------------------------------------------------------------------------

def test_disk_at_or_over_80_percent_releases_nothing(pacer):
    ws = [_view(pacer, "a", active=False)]
    assert _released(pacer.decide(ws, set(), {"a"}, 0.0, 80.0)) == []
    assert "80.0%" in pacer.decide(ws, set(), {"a"}, 0.0, 80.0)["wait"]["a"]
    assert _released(pacer.decide(ws, set(), {"a"}, 0.0, 79.9)) == ["a"]


def test_a_failed_disk_or_bundle_reading_releases_nothing(pacer):
    ws = [_view(pacer, "a", active=False)]
    assert _released(pacer.decide(ws, set(), {"a"}, None, 50.0)) == []
    assert _released(pacer.decide(ws, set(), {"a"}, 0.0, None)) == []
    assert "reading failed" in pacer.decide(ws, set(), {"a"}, None, 50.0)["wait"]["a"]


def test_at_most_max_inflight_commits_are_in_flight(pacer):
    names = [f"w{i}" for i in range(pacer.MAX_INFLIGHT + 1)]
    ws = [_view(pacer, "run", cov="cov-n082", admitted=True)] + [_view(pacer, n, active=False) for n in names]
    plan = pacer.decide(ws, set(), set(names), 0.0, 50.0)
    assert len(_released(plan)) == pacer.MAX_INFLIGHT - 1
    assert len(plan["wait"]) == 2 and all(f"{pacer.MAX_INFLIGHT} Commits in flight" in why for why in plan["wait"].values())


def test_at_most_max_big_at_batch_8_or_more_while_a_smaller_one_still_goes(pacer):
    ws = [_view(pacer, f"run{i}", row=LLAMA.format(b=8), admitted=True) for i in range(pacer.MAX_BIG - 1)] + [
          _view(pacer, "big1", cov="cov-n082", row=LLAMA.format(b=8), active=False),
          _view(pacer, "big2", cov="cov-n083", row=LLAMA.format(b=8), active=False),
          _view(pacer, "small", cov="cov-n084", row=LLAMA.format(b=1), active=False)]
    assert pacer.MAX_BIG + 1 <= pacer.MAX_INFLIGHT
    plan = pacer.decide(ws, set(), {"big1", "big2", "small"}, 0.0, 50.0)
    assert _released(plan) == ["big1", "small"]
    assert "batch 8+" in plan["wait"]["big2"]


def test_the_limits_are_release_pys_latest_unless_the_environment_overrides_them(pacer, monkeypatch):
    assert (pacer.MAX_INFLIGHT, pacer.MAX_BIG, pacer.DISK_PAUSE_PCT) == (6, 4, 80)
    assert (pacer.CAP_GB, pacer.CAP_LOW_GB, pacer.CAP_DROP_PCT) == (300, 150, 78)
    monkeypatch.setenv("PACER_MAX_INFLIGHT", "3")
    monkeypatch.setenv("PACER_MAX_BIG", "2")
    monkeypatch.setenv("PACER_CAP_GB", "150")
    again = _load(NEBIUS / "commit_pacer.py", "commit_pacer", monkeypatch)
    assert (again.MAX_INFLIGHT, again.MAX_BIG, again.CAP_GB) == (3, 2, 150)


def test_the_estimate_is_infras_scaled_by_model_size_and_at_least_10_gb(pacer):
    assert pacer.estimate_gb(PHI3.format(b=8)) == 120.0
    assert pacer.estimate_gb(QWEN7.format(b=8)) == pytest.approx(120 * 3584 * 28 / (3072 * 32))
    assert pacer.estimate_gb(PHI3.format(b=8).replace("phi3-mini", "unknown-3b")) == 120.0
    assert pacer.estimate_gb(LLAMA.format(b=1)) == 10.0


def test_the_bundle_cap_is_300_gb_over_bundles_on_disk_commits_in_flight_and_the_candidate(pacer):
    flying = _view(pacer, "flying", cov="cov-n082", admitted=True)                     # 10 GB
    cand = _view(pacer, "cand", row=PHI3.format(b=8), active=False)                     # 120 GB
    plan = pacer.decide([flying, cand], set(), {"cand"}, 170.0, 50.0)
    assert _released(plan) == ["cand"] and not plan["latch"]                            # 300 exactly: within
    plan = pacer.decide([flying, cand], set(), {"cand"}, 170.5, 50.0)
    assert _released(plan) == [] and "> 300 GB" in plan["wait"]["cand"]


def test_the_cap_drops_to_150_gb_once_the_disk_reaches_78_percent_and_stays_dropped(pacer):
    flying = _view(pacer, "flying", cov="cov-n082", admitted=True)
    cand = _view(pacer, "cand", row=PHI3.format(b=8), active=False)
    plan = pacer.decide([flying, cand], set(), {"cand"}, 20.5, 78.0)
    assert _released(plan) == [] and "> 150 GB" in plan["wait"]["cand"] and plan["latch"]
    assert _released(pacer.decide([flying, cand], set(), {"cand"}, 20.0, 78.0)) == ["cand"]
    plan = pacer.decide([flying, cand], set(), {"cand"}, 20.5, 50.0, latched=True)
    assert _released(plan) == [] and plan["latch"]


def test_a_commit_in_flight_counts_only_the_bundle_it_has_yet_to_write(pacer):
    flying = _view(pacer, "flying", cov="cov-n082", row=PHI3.format(b=8), admitted=True)   # 120 GB, 100 of it on disk
    cand = _view(pacer, "cand", row=PHI3.format(b=8), active=False)
    assert _released(pacer.decide([flying, cand], set(), {"cand"}, 160.0, 50.0, "hold", {flying["dir"]: 100.0})) == ["cand"]
    assert _released(pacer.decide([flying, cand], set(), {"cand"}, 160.0, 50.0, "hold", {})) == []
    assert _released(pacer.decide([flying, cand], set(), {"cand"}, 180.0, 50.0, "hold", {flying["dir"]: 500.0})) == ["cand"]


def test_kept_rows_never_run(pacer):
    ws = [_view(pacer, "listed", cov="cov-n133", active=False),
          _view(pacer, "b64", cov="cov-n082", row=LLAMA.format(b=64), active=False),
          _view(pacer, "b16-7b", cov="cov-n083", row=QWEN7.format(b=16), active=False),
          _view(pacer, "b16-1b", cov="cov-n084", row=LLAMA.format(b=16), active=False)]
    plan = pacer.decide(ws, set(), {w["name"] for w in ws}, 0.0, 10.0)
    assert _released(plan) == ["b16-1b"]
    assert {n for n, why in plan["wait"].items() if why.startswith("kept")} == {"listed", "b64", "b16-7b"}


def test_release_order_is_circuits_waves_then_batch(pacer):
    ws = [_view(pacer, "late", cov="cov-g056", active=False), _view(pacer, "unlisted", cov="cov-zzz", active=False),
          _view(pacer, "first-b2", cov="cov-n082", row=LLAMA.format(b=2), active=False),
          _view(pacer, "first-b1", cov="cov-n082", row=LLAMA.format(b=1), active=False)]
    assert _released(pacer.decide(ws, set(), {w["name"] for w in ws}, 0.0, 10.0)) == ["first-b1", "first-b2", "late", "unlisted"]


def test_nothing_admitted_is_held_and_it_counts_in_flight(pacer):
    ws = [_view(pacer, f"a{i}", cov="cov-n082", admitted=True) for i in range(pacer.MAX_INFLIGHT)] + [_view(pacer, "d", active=False)]
    plan = pacer.decide(ws, set(), {"d"}, 0.0, 10.0)
    assert plan["acts"] == [] and f"{pacer.MAX_INFLIGHT} Commits in flight" in plan["wait"]["d"]


# ---- hold mode (release.py's) and gate mode ------------------------------------------------------------------------------------

def test_hold_mode_holds_an_arrival_and_counts_a_hand_reactivation_as_released(pacer):
    ws = [_view(pacer, "new", cov="cov-n133"), _view(pacer, "hand", cov="cov-n133")]
    plan = pacer.decide(ws, set(), {"hand"}, 0.0, 10.0, "hold")
    assert [(v, w["name"]) for v, w, _ in plan["acts"]] == [("hold", "new")]
    assert plan["released"] == {"hand"} and "new" in plan["seen"]


def test_gate_mode_passes_everything_but_a_paced_commit_at_once(pacer):
    other = pacer.view(_wl("dev", stage=False, reserved=True, check="Pending"), _job("dev", "vllm-tp2-gpuless-build/x", "prover-dev", "0"))
    golden = pacer.view(_wl("golden", ws="mps-golden", reserved=True, check="Pending"), _job("golden", "mps-golden/g1"))
    running = _view(pacer, "running", admitted=True, check="Pending")    # admitted before the check was attached
    plan = pacer.decide([other, golden, running], set(), set(), 0.0, 10.0, "gate")
    assert sorted((v, w["name"]) for v, w, _ in plan["acts"]) == [("ready", "dev"), ("ready", "golden"), ("ready", "running")]


def test_gate_mode_readies_a_release_only_once_kueue_reserves_its_quota(pacer):
    queued = _view(pacer, "a")
    assert [v for v, *_ in pacer.decide([queued], {"a"}, {"a"}, 0.0, 10.0, "gate")["acts"]] == []
    reserved = _view(pacer, "a", reserved=True, check="Pending")
    plan = pacer.decide([reserved], {"a"}, {"a"}, 0.0, 10.0, "gate")
    assert [(v, why) for v, _, why in plan["acts"]] == [("ready", "released")]


def test_gate_mode_holds_a_hand_reactivation_and_never_readies_an_unreleased_commit(pacer):
    hand = _view(pacer, "hand", cov="cov-n133", reserved=True, check="Pending")      # kept, reactivated by hand
    plan = pacer.decide([hand], set(), {"hand"}, 0.0, 10.0, "gate")
    assert [(v, w["name"]) for v, w, _ in plan["acts"]] == [("hold", "hand")]
    assert "hand" not in plan["released"] and plan["wait"]["hand"].startswith("kept")


def test_gate_mode_releases_a_hand_reactivation_only_by_the_rule(pacer):
    hand = _view(pacer, "hand", reserved=True, check="Pending")
    plan = pacer.decide([hand], set(), {"hand"}, 0.0, 10.0, "gate")
    assert [v for v, *_ in plan["acts"]] == ["hold", "activate"] and plan["released"] == {"hand"}


# ---- which Workloads are Commits -----------------------------------------------------------------------------------------------

def test_a_commit_is_labelled_or_a_legacy_config_run_gpu_or_row_task(pacer):
    unlabelled = _wl("w", stage=False)
    assert pacer.is_commit(_wl("w"), None)
    assert pacer.is_commit(unlabelled, _job("w", "n2-build/c", "config-run", "1"))
    assert pacer.is_commit(unlabelled, _job("w", "vllm-epoch-run/c", "config-run-row", "0"))
    assert not pacer.is_commit(unlabelled, _job("w", "n2-build/c", "config-run", "2"))
    assert not pacer.is_commit(unlabelled, _job("w", "x/c", "commit-pack", "0"))
    assert not pacer.is_commit(unlabelled, None)


def test_a_copied_job_without_annotations_is_still_paced(pacer):
    v = pacer.view(_wl("copy", "cov-g173", ws="vllm-coverage-defs"), None)
    assert v["paced"] and v["cov"] == "cov-g173"
    assert not pacer.view(_wl("golden", ws="mps-golden"), None)["paced"]
    assert pacer.view(_wl("golden", ws="mps-golden"), None, ("*",))["paced"]


def test_config_run_and_row_render_the_commit_label_on_their_commit_tasks_only(monkeypatch, tmp_path):
    monkeypatch.setenv("VY_DISPATCH_ROOT", str(tmp_path / "dispatch"))
    monkeypatch.setenv("VY_READY", str(tmp_path / "ready"))
    dispatch = _load(NEBIUS / "dispatch.py", "dispatch", monkeypatch)
    item = {"id": "c1", "tree": str(tmp_path), "env": {"ROW": LLAMA.format(b=8), "ROLE": "LLAMA32_1B", "REPO": "r", "REVISION": "v"}}

    def stage(template: str, idx: int) -> str | None:
        labels = dispatch.render({**item, "template": template}, "n2-build/cov-x", idx, 0)["spec"]["template"]["metadata"]["labels"]
        assert labels["verity.dev/workstream"] == "n2-build"
        return labels.get("verity.dev/stage")

    assert [stage("config-run", i) for i in range(3)] == [None, "commit", None]
    assert stage("config-run-row", 0) == "commit"


# ---- cleanup -------------------------------------------------------------------------------------------------------------------

D = "/workspace/jobs/cov/cov-g058/mistral-7b__b8/commit"


def _done(key: str, state: str, t: str, task: int = 1) -> dict:
    return {"key": key, "state": state, "t": t, "task": task, "rc": 12}


def test_cleanup_waits_six_hours_after_the_failure(pacer):
    bundle = [Path(D, "replay_bundle_p0")]
    done = [_done("n2-build/cov-g058", "failed", "2026-09-30T18:00:00Z")]
    owners = {D: {"n2-build/cov-g058"}}
    assert [p for p, _ in pacer.cleanup_candidates(done, owners, set(), T0, bundle)] == bundle
    assert pacer.cleanup_candidates(done, owners, set(), T0 - 1, bundle) == []


def test_cleanup_keeps_a_bundle_a_retry_or_unfinished_job_names(pacer):
    bundle = [Path(D, "replay_bundle_p0.partial")]
    done = [_done("n2-build/cov-g058", "failed", "2026-09-30T12:00:00Z")]
    assert pacer.cleanup_candidates(done, {D: {"n2-build/cov-g058"}}, {D}, T0, bundle) == []


def test_cleanup_keeps_succeeded_running_unrecorded_and_odd_paths(pacer):
    old = "2026-09-30T12:00:00Z"
    retried = [_done("n2-build/cov-g058", "failed", old), _done("n2-build/cov-g058-r2", "succeeded", old)]
    owners = {D: {"n2-build/cov-g058", "n2-build/cov-g058-r2"}}
    assert pacer.cleanup_candidates(retried, owners, set(), T0, [Path(D, "replay_bundle_p0")]) == []
    unended = [_done("n2-build/cov-g058", "failed", old)]
    assert pacer.cleanup_candidates(unended, owners, set(), T0, [Path(D, "replay_bundle_p0")]) == []
    assert pacer.cleanup_candidates(unended, {}, set(), T0, [Path(D, "replay_bundle_p0")]) == []
    one = {D: {"n2-build/cov-g058"}}
    assert pacer.cleanup_candidates(unended, one, set(), T0, [Path(D, "replay_bundle_x")]) == []
    assert pacer.cleanup_candidates(unended, {D + "/x": {"n2-build/cov-g058"}}, set(), T0,
                                    [Path(D, "x", "replay_bundle_p0")]) == []


def test_record_commits_keeps_the_first_directory_of_each_item(pacer):
    v = _view(pacer, "a", cov="cov-g058")
    pacer.record_commits([v], dry=False)
    pacer.record_commits([{**v, "dir": "/elsewhere", "name": "b"}], dry=False)
    assert pacer.load_commits()["n2-build/cov-g058"]["dir"] == "/workspace/jobs/cov/cov-g058/" + LLAMA.format(b=1) + "/commit"


# ---- writes, the unit and the manifests ----------------------------------------------------------------------------------------

def test_ready_keeps_other_checks_and_a_wait_is_annotated_once_per_reason(pacer, monkeypatch):
    calls = []
    monkeypatch.setattr(pacer, "k", lambda *a: calls.append(a) or "")
    w = _view(pacer, "a", reserved=True, check="Pending")
    w["checks"].append({"name": "other", "state": "Pending"})
    pacer.apply({"acts": [("ready", w, "released")], "wait": {}}, dry=False)
    sent = json.loads(calls[0][-1])["status"]["admissionChecks"]
    assert [(c["name"], c["state"]) for c in sent] == [("commit-pacer", "Ready"), ("other", "Pending")]
    held = _view(pacer, "b", active=False)
    held["note"] = "held 2026-10-01T00:00:00Z: bundles 21 GB + in flight 120 GB + this 15 GB > 150 GB"
    pacer.apply({"acts": [], "wait": {"b": "bundles 22 GB + in flight 120 GB + this 15 GB > 150 GB"}, "views": {"b": held}}, False)
    pacer.apply({"acts": [], "wait": {"b": "3 Commits in flight (max 3)"}, "views": {"b": held}}, False)
    assert [c[:2] for c in calls[1:]] == [("annotate", "workload")]


def test_a_tick_that_cannot_read_the_cluster_changes_nothing(pacer, monkeypatch):
    calls = []

    def k(*args):
        calls.append(args)
        raise RuntimeError("connection refused")

    monkeypatch.setattr(pacer, "k", k)
    assert pacer.main(["once", "--mode", "gate"]) == 1
    assert all(a[0] == "get" for a in calls)


def test_the_unit_runs_the_pacer_and_restarts_on_failure():
    unit = configparser.ConfigParser(strict=False, interpolation=None)
    unit.read(NEBIUS / "commit-pacer.service")
    assert unit["Service"]["Restart"] == "on-failure"
    assert unit["Service"]["ExecStart"].endswith("commit_pacer.py run --every 10")
    timer = configparser.ConfigParser(interpolation=None)
    timer.read(NEBIUS / "commit-pacer-cleanup.timer")
    assert timer["Timer"]["OnCalendar"] == "hourly"


def test_the_admission_manifest_names_the_pacers_check_and_scope(pacer):
    docs = {d["kind"]: d for d in yaml.safe_load_all((NEBIUS / "commit-pacer-admission.yaml").read_text())}
    assert docs["AdmissionCheck"]["metadata"]["name"] == pacer.CHECK
    assert docs["AdmissionCheck"]["spec"]["controllerName"] == pacer.CONTROLLER
    expr = docs["MutatingAdmissionPolicy"]["spec"]["matchConditions"][0]["expression"]
    listed = expr.split("['verity.dev/workstream'] in [", 1)[1].split("]", 1)[0]
    assert [s.strip(" '") for s in listed.split(",")] == pacer.CIRCUITS.split(",")
    assert f"object.spec.queueName == '{pacer.QUEUE}'" in expr
    assert docs["MutatingAdmissionPolicyBinding"]["spec"]["policyName"] == docs["MutatingAdmissionPolicy"]["metadata"]["name"]
