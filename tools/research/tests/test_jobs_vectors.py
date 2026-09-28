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


UNREAD = f"{trains.TOOL_PY} on main has no CHECK = Tool(...) this gate can read: one, at module level, of keyword arguments only"
RULES_UNREAD = f"{trains.TOOL_PY} on main has a merge_requires this gate can't read"
OK = 'CHECK = Tool(name="check", produces=["check/v1"]'
#: main's own layout: `preflight` after `merge_requires`
MAINS = ('CHECK = Tool(name="check", version="1", command=["uv", "run", "--locked", "--extra", "torch-cpu", "python", '
         '"tools/verity/check/check.py"],\n             produces=["check/v1"],\n'
         '             merge_requires={"lean-agreement": ["verity/ml/flock/", "!README.md", "!PROTOCOL.md"]},\n'
         '             preflight=["tools/verity/check/preflight.py"])\n')


def _train_on(tool_py: str | None, pr: int, kind: str = "check/v1"):
    """(the service, its one train, the PR's head): PR ``pr`` as a train on a main whose tools/verity/check/tool.py is ``tool_py``,
    checked by run r9 with a ``kind`` result whose one step, suites, passed, after the reconcile cron."""
    s = service(TRAINS)
    head = s.github.prs[pr]["head"]
    run_case({"name": "tool.py", "steps": [
        {"at": 0, "github_push": {"ref": "main", "sha": "m9", "parents": ["m0"], "files": {trains.TOOL_PY: tool_py}}},
        {"at": 0, "req": ["POST", "/api/trains"], "token": "tok-coordinator", "body": {"prs": [pr]},
         "expect": {"status": 201, "body": {"tip": f"m9+{head}"}}},
        {"at": 1, "req": ["POST", "/api/jobs/claim"], "token": "tok-dispatch",
         "body": {"worker": "dispatch:rc", "capabilities": {"lease_s": 36000}}, "expect": {"status": 200}, "save": {"L": "lease_token"}},
        {"at": 2, "req": ["POST", "/api/jobs/1/complete"], "token": "tok-dispatch",
         "body": {"lease_token": "$L", "outcome": "passed", "run_id": "r9"}, "expect": {"status": 200}},
        {"at": 3, "store": {"attempts": {"r9": {"id": "r9", "tool": {"name": "check"}, "source": {"commit": f"m9+{head}", "dirty": False},
                                                "status": {"state": "done", "rc": 0}, "outputs": {"result": "art:r9"}}},
                            "manifests": {"art:r9": {"kind": kind, "meta": {"steps": [{"step": "suites", "status": "passed"}]}}}}},
        {"at": 4, "reconcile": True, "expect": {"moved": [1]}},
    ]}, s, TRAINS)
    (tr,) = s.trains.values()
    return s, tr, head


@pytest.mark.parametrize(("tool_py", "pr", "state", "why"), [
    pytest.param(f"{OK}\n", 2, "stuck", f"{trains.TOOL_PY} on main doesn't parse (", id="unparseable"),
    pytest.param(f"{OK}, merge_requires=RULES)\n", 2, "stuck", RULES_UNREAD, id="merge_requires-a-name"),
    pytest.param(f'{OK}, merge_requires={{"lean-agreement": "verity/ml/flock/"}})\n', 2, "stuck", RULES_UNREAD,
                 id="merge_requires-of-a-string"),
    pytest.param('CHECK = Tool(name="check", produces=("check/v1",))\n', 2, "stuck",
                 f"{trains.TOOL_PY} on main has a produces this gate can't read", id="produces-a-tuple"),
    pytest.param('CHECK = Tool("check", "1", [], [], [], [], [], [], ["check/v1"])\n', 2, "stuck", UNREAD, id="positional"),
    pytest.param(f"{OK}, **RULES)\n", 6, "stuck", UNREAD, id="splat"),
    pytest.param(f'{OK}, merge_requires={{}}, merge_requires={{"x": ["b"]}})\n', 2, "stuck", UNREAD, id="repeated-keyword"),
    pytest.param(f"{OK})\n{OK})\n", 2, "stuck", UNREAD, id="two-checks"),
    pytest.param(f"if True:\n    {OK})\n", 2, "stuck", UNREAD, id="not-module-level"),
    pytest.param(f"CHECK = tool.{OK[8:]})\n", 2, "stuck", UNREAD, id="not-Tool"),
    pytest.param(None, 2, "stuck", f"main has no {trains.TOOL_PY}", id="no-tool-py"),
    pytest.param(f"{OK})\n", 6, "landed", None, id="no-merge_requires-requires-nothing"),
    pytest.param(MAINS, 6, "stuck", "the change needs lean-agreement absent to have passed in run r9", id="mains-layout"),
])
def test_the_gate_reads_only_a_check_declaration_it_can_parse(tool_py, pr, state, why):
    """Python-only beside the vectors: a syntax error's why is Python's own message; a missing `merge_requires` asks for no
    step, so a verity/ml/flock/ change lands without lean-agreement."""
    s, tr, head = _train_on(tool_py, pr)
    assert tr.state == state, tr.why
    assert tr.why is None if why is None else tr.why.startswith(why), tr.why
    assert s.github.refs["main"] == ("m9" if why else f"m9+{head}")


@pytest.mark.parametrize("above", [
    'BENCH = Tool(name="bench", version="1", produces=["bench-result/v1"])\n',
    '#: the old bench declaration had produces=["bench-result/v1"]\n',
    '"""A docstring that quotes produces=["bench-result/v1"]."""\n',
    'QUICK = Tool(name="quick", version="1", also_produces=["bench-result/v1"])\n',
], ids=["another-tool", "comment", "docstring", "suffix-keyword"])
def test_the_gate_reads_checks_own_produces_whatever_quotes_one_above_it(above):
    """A pattern for `produces=[...]` took the first one anywhere in main's tool.py: another Tool's, a comment's, a docstring's,
    or a keyword that merely ends in `produces` (red team, #990).  A bench-result/v1 never lands as a check."""
    _s, tr, _head = _train_on(above + MAINS, 2, "bench-result/v1")
    assert (tr.state, tr.why) == ("stuck", f"run r9's result is a bench-result/v1, and {trains.TOOL_PY} on main produces check/v1")
    _s, tr, _head = _train_on(above + MAINS, 2)
    assert tr.state == "landed", tr.why


def test_the_repositorys_own_check_declaration_parses_to_what_importing_it_declares(tmp_path):
    """The regression the patterns had: main's tools/verity/check/tool.py changed its layout (a keyword after merge_requires) and the
    gate silently read no merge_requires.  This reads the file itself, so a layout the gate can't parse fails here first."""
    from research.merge import declaration

    text = (Path(__file__).resolve().parents[3] / trains.TOOL_PY).read_text()
    # imported from a copy: importing the real file reads its tools/verity/check/__pycache__, which the suite's inputs don't cover
    (tmp_path / trains.TOOL_PY).parent.mkdir(parents=True)
    (tmp_path / trains.TOOL_PY).write_text(text)
    tool = declaration(tmp_path, "check")
    assert trains.declared(text) == (tool.produces, tool.merge_requires)
    assert tool.produces and tool.merge_requires


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
