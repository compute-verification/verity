"""The Job queue's shared cases (`research/jobs/vectors.json`) against the reference model, `research.jobs.model`. The site's
TypeScript runs the same file against its routes and Neon, so a rule changes in both places or in neither."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.jobs import model, trains

VECTORS = json.loads(Path(model.__file__).with_name("vectors.json").read_text())
TRAINS = json.loads(Path(model.__file__).with_name("train_vectors.json").read_text())


def service(vectors: dict = VECTORS) -> model.JobService:
    s = model.JobService(epoch=vectors["epoch"]) if "github" not in vectors["setup"] else trains.TrainService(epoch=vectors["epoch"])
    for t in vectors["setup"]["tokens"]:
        s.mint(t["secret"], t["name"], t["scopes"])
    for w in vectors["setup"]["workers"]:
        s.register(w["id"], w["kind"], w["token"], w["kinds"])
    if gh := vectors["setup"].get("github"):
        s.github = trains.FakeGitHub(commits=json.loads(json.dumps(gh["commits"])), refs=dict(gh["refs"]),
                                     prs={p["number"]: {**p, "labels": list(p["labels"])} for p in gh["prs"]})
    return s


def substitute(x, saved: dict):
    if isinstance(x, str) and x.startswith("$") and x[1:] in saved:
        return saved[x[1:]]
    if isinstance(x, dict):
        return {k: substitute(v, saved) for k, v in x.items()}
    if isinstance(x, list):
        return [substitute(v, saved) for v in x]
    return x


def mismatches(want, got, where: str = "body") -> list[str]:
    """`want` as a subset of `got`: objects recursively, lists element by element with the same length."""
    if isinstance(want, dict):
        if not isinstance(got, dict):
            return [f"{where}: want an object, got {got!r}"]
        return [m for k, v in want.items() for m in (mismatches(v, got[k], f"{where}.{k}") if k in got else [f"{where}.{k}: missing"])]
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            return [f"{where}: want {len(want)} elements, got {got!r}"]
        return [m for i, (w, g) in enumerate(zip(want, got)) for m in mismatches(w, g, f"{where}[{i}]")]
    return [] if want == got and type(want) is type(got) else [f"{where}: want {want!r}, got {got!r}"]


def run_case(case: dict, s: model.JobService | None = None, vectors: dict = VECTORS) -> list[tuple[str, dict]]:
    """Run one case's steps; returns each request's (path, response body) for the coverage checks."""
    s = s or service(vectors)
    saved: dict = {}
    seen = []
    for i, step in enumerate(case["steps"]):
        where = f"{case['name']!r} step {i}"
        if "revoke" in step:
            s.revoke(step["revoke"])
        elif "retire" in step:
            s.retire(step["retire"])
        elif step.get("tick"):
            assert s.tick(step["at"]) == step["expect"]["moved"], where
        elif step.get("reconcile"):
            assert s.reconcile(step["at"]) == step["expect"]["moved"], where
        elif "store" in step:
            s.store.attempts.update(json.loads(json.dumps(step["store"].get("attempts", {}))))
            s.store.manifests.update(json.loads(json.dumps(step["store"].get("manifests", {}))))
            s.store.objects.update(step["store"].get("objects", {}))
        elif "github_push" in step:
            g = step["github_push"]
            s.github.push(g["ref"], g["sha"], g["parents"], g["files"])
        elif "github_label" in step:
            s.github.prs[step["github_label"]["pr"]]["labels"].append(step["github_label"]["add"])
        elif "expect_github" in step:
            want = step["expect_github"]
            for ref, sha in want.get("refs", {}).items():
                assert s.github.refs.get(ref) == sha, f"{where}: {ref} is {s.github.refs.get(ref)}, want {sha}"
            for n, pr in want.get("prs", {}).items():
                assert not (bad := mismatches(pr, s.github.prs[int(n)], f"pr {n}")), f"{where}: {bad}"
            if "comments" in want:
                assert s.github.comments == want["comments"], f"{where}: {s.github.comments}"
            for sha, files in want.get("trees", {}).items():
                assert not (bad := mismatches(files, s.github.tree(sha), f"tree {sha}")), f"{where}: {bad}"
        else:
            method, path = step["req"]
            status, body = s.request(method, path, step.get("token"), substitute(step.get("body"), saved), step["at"])
            assert status == step["expect"]["status"], f"{where}: {method} {path} answered {status} {body}"
            assert not (bad := mismatches(step["expect"].get("body", {}), body)), f"{where}: {bad}"
            for name, fld in step.get("save", {}).items():
                saved[name] = body[fld]
            seen.append((f"{method} {path}", body))
    return seen


@pytest.mark.parametrize("case", VECTORS["cases"], ids=[c["name"] for c in VECTORS["cases"]])
def test_case(case):
    run_case(case)


@pytest.mark.parametrize("case", TRAINS["cases"], ids=[c["name"] for c in TRAINS["cases"]])
def test_train_case(case):
    run_case(case, vectors=TRAINS)


FORMS = {"req", "tick", "revoke", "retire", "reconcile", "store", "github_push", "github_label", "expect_github"}


@pytest.mark.parametrize("vectors", [VECTORS, TRAINS], ids=["jobs", "trains"])
def test_the_cases_have_unique_names_and_one_form_per_step(vectors):
    names = [c["name"] for c in vectors["cases"]]
    assert len(names) == len(set(names))
    for c in vectors["cases"]:
        for step in c["steps"]:
            assert len(FORMS & set(step)) == 1, step
            assert isinstance(step["at"], int), step


def test_the_train_cases_reach_every_train_state_and_route():
    states, routes = set(), set()
    for c in TRAINS["cases"]:
        s = service(TRAINS)
        for req, body in run_case(c, s, TRAINS):
            method, path = req.split(" ", 1)
            routes.add(f"{method} " + "/".join("{id}" if p.isdigit() else p for p in path.split("?")[0].split("/")))
            if path.startswith("/api/trains"):
                states |= {body.get("state")} | {t.get("state") for t in body.get("trains", [])}
        states |= {t.state for t in s.trains.values()}
    assert set(trains.TRAIN_STATES) <= states
    assert {"POST /api/trains", "GET /api/trains", "GET /api/trains/{id}", "POST /api/trains/{id}/cancel"} <= routes


def test_the_cases_reach_every_state_and_every_route():
    """The site's port is only as tested as these cases: every state a job can be in, and every route, appears in some answer."""
    states, routes = set(), set()
    for c in VECTORS["cases"]:
        s = service()
        for req, body in run_case(c, s):
            method, path = req.split(" ", 1)
            routes.add(f"{method} " + "/".join("{id}" if p.isdigit() else p for p in path.split("?")[0].split("/")))
            states |= {body.get("state")} | {j.get("state") for j in body.get("jobs", [])}
        states |= {j.state for j in s.jobs.values()}
    assert set(model.STATES) <= states
    assert {"POST /api/jobs", "POST /api/jobs/claim", "GET /api/jobs", "GET /api/jobs/{id}", "POST /api/jobs/{id}/renew",
            "POST /api/jobs/{id}/complete", "POST /api/jobs/{id}/cancel", "GET /api/events"} <= routes


def test_the_matcher_catches_a_wrong_answer():
    case = {"name": "wrong", "steps": [{"at": 0, "req": ["POST", "/api/jobs"], "token": "tok-coordinator",
                                        "body": {"kind": "merge-check", "payload": {}}, "expect": {"status": 201, "body": {"id": 2}}}]}
    with pytest.raises(AssertionError, match="want 2, got 1"):
        run_case(case)
    assert mismatches({"events": [{"id": 1}]}, {"events": [{"id": 1}, {"id": 2}]}) == ["body.events: want 1 elements, got [{'id': 1}, {'id': 2}]"]
    assert mismatches({"cancel": False}, {"cancel": 0}) == ["body.cancel: want False, got 0"]


def test_a_lease_token_is_never_stored():
    s = service()
    s.request("POST", "/api/jobs", "tok-coordinator", {"kind": "merge-check", "payload": {}}, 0)
    _, body = s.request("POST", "/api/jobs/claim", "tok-dispatch", {"worker": "dispatch:rc"}, 1)
    assert body["lease_token"] not in json.dumps([vars(j) for j in s.jobs.values()] + [vars(a) for a in s.attempts] + s.events)
