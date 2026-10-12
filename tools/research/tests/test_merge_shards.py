"""`research merge` on a sharded check: a `check.py --combine` Attempt passes the gate only when the gate's own reading of its
shards (`shard_set`) finds them one passing check of the exact clean commit, and its steps are theirs; a shard alone, and a set
with a shard missing, repeated, failed or of another commit, config or plan, passes nothing.  Fake Attempts in a scratch store."""
from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from research.merge import SHARD_KIND, Refused, gate, shard_set, step_status, unchecked_pushes, unlike

COMMAND = ["uv", "run", "python", "tools/verity/check/check.py"]
DECL = ("from research.store.tool import Tool\nCHECK = Tool(name='check', version='1', "
        f"command={COMMAND!r}, produces=['check/v1'])\n")
CI = ('schema = "ci/v0"\n' + "".join(f'[[job]]\nname = "{n}"\nowner = "ci"\ngroup = {g}\n'
                                     for n, g in (("a", 1), ("lean-audit", 2), ("c", 3))))
EVERY = ["a", "lean-audit:p", "lean-audit:q", "c"]
#: shard K's parts, and its steps' (name, status, parts); the preflight runs in each
SHARDS = {1: (["a", "lean-audit:p"], [("a", "passed", ["a"]), ("lean-audit", "passed", ["lean-audit:p"])]),
          2: (["lean-audit:q", "c"], [("lean-audit", "passed", ["lean-audit:q"]), ("c", "skipped", ["c"])])}


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


@pytest.fixture
def world(tmp_path, monkeypatch):
    """(the main checkout, feat's tip, its tools/verity/check tree, the store) with tools/verity/check's declaration read from the checkout."""
    from research.store.local import LocalStore

    for name in ("tools", "tools.verity.check", "tools.verity.check.tool"):
        monkeypatch.setitem(sys.modules, name, None)
    main = tmp_path / "main"
    (main / "tools" / "verity" / "check").mkdir(parents=True)
    (main / "tools" / "verity" / "check" / "tool.py").write_text(DECL)
    (main / "tools" / "verity" / "check" / "ci.toml").write_text(CI)
    _git(main, "init", "-q", "-b", "main")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "init")
    _git(main, "checkout", "-q", "-b", "feat")
    (main / "a.txt").write_text("feat\n")
    _git(main, "add", "a.txt")
    _git(main, "commit", "-q", "-m", "feat")
    tip = _git(main, "rev-parse", "HEAD")
    _git(main, "checkout", "-q", "main")
    store = LocalStore(tmp_path / "store", config_path=tmp_path / "no-remote.toml")
    return main, tip, _git(main, "rev-parse", f"{tip}:tools/verity/check"), store


_n = iter(range(10_000))


def _attempt(store, commit: str, argv: list[str], kind: str | None, meta: dict, rc: int = 0, dirty: bool = False) -> str:
    rid = f"r20261004-000000-{next(_n):04x}"
    meta = {"schema": "research/result/v0.1", "kind": kind, "run_id": rid, "validation": {"status": "passed", "checker": "t"},
            **meta}
    art = store.put_artifact(kind, meta) if kind else None
    store.put_attempt({
        "schema": "research/attempt/v0.1", "id": rid, "tool": {"name": "check", "version": "1"}, "derivation": "drv:" + "0" * 64,
        "reuse_key": None, "argv": argv, "params": {}, "key_params": [], "inputs": {}, "conditions": {}, "key_conditions": [],
        "source": {"commit": commit, "dirty": dirty}, "closure": None, "execution": {}, "cost": None,
        "status": {"state": "done", "rc": rc}, "outputs": {"result": art} if art else {}, "telemetry": None})
    return rid


def _shard_meta(k: int, n: int, commit: str, config: str, parts=None, steps=None, every=EVERY) -> dict:
    parts = SHARDS[k][0] if parts is None else parts
    steps = SHARDS[k][1] if steps is None else steps
    return copy.deepcopy({"shard": {"k": k, "n": n, "parts": parts, "all": every, "commit": commit, "config": config},
                          "steps": [{"step": "preflight-lock", "status": "passed"},
                                    *({"step": s, "status": st, "parts": p, "seconds": 1.0} for s, st, p in steps)]})


def _shards(world, edit=None) -> list[str]:
    """Shard 1/2 and 2/2 of feat's tip, each one passing Attempt; ``edit(k, argv, meta, kw)`` changes one before it is stored."""
    _main, tip, config, store = world
    ids = []
    for k in (1, 2):
        argv, meta, kw = [*COMMAND, "--shard", f"{k}/2"], _shard_meta(k, 2, tip, config), {"commit": tip, "kind": SHARD_KIND}
        if edit:
            edit(k, argv, meta, kw)
        ids.append(_attempt(store, kw["commit"], argv, kw["kind"], meta, kw.get("rc", 0), kw.get("dirty", False)))
    return ids


def _combine(world, ids: list[str], steps=None) -> str:
    """`check.py --combine IDS` as check.py records it: a `check/v1` whose steps are `shard_set`'s (or ``steps``)."""
    main, tip, _config, store = world
    if steps is None:
        steps, why = shard_set(store, main, tip, ids, "check", tuple(COMMAND))
        steps = steps or []
    return _attempt(store, tip, [*COMMAND, "--combine", *ids], "check/v1", {"steps": steps, "shards": ids})


def _gate(world):
    main, _tip, _config, store = world
    return gate(main, "feat", "main", store, "check", log=lambda *_: None)


def test_a_combined_set_of_every_shard_passes_the_gate_as_one_check(world):
    main, tip, _config, store = world
    ids = _shards(world)
    steps, why = shard_set(store, main, tip, ids, "check", tuple(COMMAND))
    assert why is None
    assert [(s["step"], s["status"]) for s in steps] == [("preflight-lock", "passed"), ("a", "passed"), ("lean-audit:p", "passed"),
                                                         ("lean-audit:q", "passed"), ("c", "skipped")]
    assert [s.get("run_id") for s in steps[1:]] == [ids[0], ids[0], ids[1], ids[1]] and steps[0]["shards"] == ids
    rid = _combine(world, ids)
    assert _gate(world) == (tip, rid)
    assert step_status(store, rid, "c") == "skipped", "a step skipped in its shard is skipped in the check, as unsharded"
    assert not unchecked_pushes(main, [f"refs/heads/main {tip} refs/heads/main {'0' * 39}1"], store)


def test_a_shard_alone_passes_nothing(world):
    main, tip, _config, store = world
    _shards(world)
    with pytest.raises(Refused, match="shard 1/2 of a sharded check, which passes no gate alone"):
        _gate(world)
    assert unchecked_pushes(main, [f"refs/heads/main {tip} refs/heads/main {'0' * 39}1"], store) == [tip]
    assert unlike([*COMMAND, "--shard", "1/2"], COMMAND) and unlike([*COMMAND, "--combine", "r1"], COMMAND)
    assert unlike([*COMMAND, "--keep-going", "--shard", "1/2", "--cores", "8"], COMMAND).startswith("it is shard 1/2")


def _edit(k0, f):
    return lambda k, argv, meta, kw: f(argv, meta, kw) if k == k0 else None


#: (what is wrong with the set, how shard 2 or the set is made so, what the gate says)
BAD = [
    ("a shard is missing", None, lambda ids: ids[:1], "shard 2/2 is not among them"),
    ("a shard is named twice", None, lambda ids: [ids[0], ids[0]], "names each shard's attempt once"),
    ("two attempts of one shard", _edit(2, lambda a, m, kw: (a.__setitem__(-1, "1/2"), m["shard"].__setitem__("k", 1))),
     None, "shard 1/2 is named twice"),
    ("a shard's step failed", _edit(2, lambda a, m, kw: m["steps"][2].__setitem__("status", "failed")), None, "c didn't pass"),
    ("a shard's step stopped", _edit(2, lambda a, m, kw: m["steps"][1].__setitem__("status", "stopped")), None,
     "lean-audit didn't pass"),
    ("a shard's verdict failed", _edit(2, lambda a, m, kw: m.__setitem__("validation", {"status": "failed"})), None,
     "verdict is failed"),
    ("a shard exited 1", _edit(2, lambda a, m, kw: kw.__setitem__("rc", 1)), None, "its record says rc 1"),
    ("a shard of a dirty tree", _edit(2, lambda a, m, kw: kw.__setitem__("dirty", True)), None, "its record says dirty True"),
    ("a shard of another commit", _edit(2, lambda a, m, kw: kw.__setitem__("commit", "f" * 40)), None, "its record says commit"),
    ("a shard whose result names another commit", _edit(2, lambda a, m, kw: m["shard"].__setitem__("commit", "f" * 40)), None,
     "checked commit ffff"),
    ("a shard of another config", _edit(2, lambda a, m, kw: m["shard"].__setitem__("config", "e" * 40)), None,
     "with config eeee"),
    ("a shard whose result isn't its command's", _edit(2, lambda a, m, kw: m["shard"].__setitem__("k", 1)), None,
     "its result is shard 1/2, its command 2/2"),
    ("a shard of another plan", _edit(2, lambda a, m, kw: (a.__setitem__(-1, "2/3"), m["shard"].__setitem__("n", 3))), None,
     "of different plans"),
    ("shards that disagree on every part", _edit(2, lambda a, m, kw: m["shard"].__setitem__("all", [*EVERY, "d"])), None,
     "of different plans"),
    ("a part in no shard", _edit(2, lambda a, m, kw: (m["shard"].__setitem__("parts", ["c"]), m["steps"].pop(1))), None,
     "in no shard: lean-audit:q"),
    ("a part in two shards", _edit(2, lambda a, m, kw: (m["shard"]["parts"].append("a"), m["steps"].append(
        {"step": "a", "status": "passed", "parts": ["a"]}))), None, "twice: a"),
    ("a shard whose steps aren't its parts", _edit(2, lambda a, m, kw: m["steps"].pop(1)), None,
     "its steps ran c, not its shard's parts"),
    ("a shard that skipped the preflight", _edit(2, lambda a, m, kw: m["steps"].pop(0)), None,
     "different steps outside their parts"),
    ("a shard's result of the unsharded kind", _edit(2, lambda a, m, kw: kw.__setitem__("kind", "check/v1")), None,
     "its result is a check/v1, not a check-shard/v1"),
    ("a shard with no result", _edit(2, lambda a, m, kw: kw.__setitem__("kind", None)), None, "its result is missing"),
    ("a shard of another command", _edit(2, lambda a, m, kw: a.__setitem__(0, "bash")), None, "isn't the target's declared"),
    ("a run that isn't a shard", _edit(2, lambda a, m, kw: a.__delitem__(slice(-2, None))), None, "it is not a shard"),
    ("a set naming a run nobody recorded", None, lambda ids: [ids[0], "r20261004-000000-dead"], "can't be read"),
]


@pytest.mark.parametrize("what, edit, pick, why", BAD, ids=[b[0] for b in BAD])
def test_a_combine_whose_shards_arent_one_passing_check_passes_nothing(world, what, edit, pick, why):
    main, tip, _config, store = world
    ids = _shards(world, edit)
    ids = pick(ids) if pick else ids
    _steps, said = shard_set(store, main, tip, ids, "check", tuple(COMMAND))
    assert said and why in said, said
    _combine(world, ids, steps=[{"step": "preflight-lock", "status": "passed"}])
    with pytest.raises(Refused, match="its shards aren't a passing check") as e:
        _gate(world)
    assert why in str(e.value)


def test_a_job_the_commits_ci_toml_declares_that_no_shard_runs_passes_nothing(world):
    main, tip, _config, store = world
    every = EVERY[:3]

    def edit(k, argv, meta, kw):
        meta["shard"]["all"] = every
        if k == 2:
            meta["shard"]["parts"], meta["steps"] = ["lean-audit:q"], meta["steps"][:2]

    ids = _shards(world, edit)
    assert shard_set(store, main, tip, ids, "check", tuple(COMMAND))[1] == f"no shard runs c, which {tip[:12]}'s ci.toml declares"


def test_a_combine_whose_steps_arent_its_shards_passes_nothing(world):
    main, tip, _config, store = world
    ids = _shards(world)
    steps, _ = shard_set(store, main, tip, ids, "check", tuple(COMMAND))
    claimed = copy.deepcopy(steps)
    claimed[-1]["status"] = "passed"                     # c was skipped in its shard
    _combine(world, ids, steps=claimed)
    with pytest.raises(Refused, match="its steps aren't the ones its shards make"):
        _gate(world)
    _combine(world, ids, steps=steps[:-1])
    with pytest.raises(Refused, match="its steps aren't the ones its shards make"):
        _gate(world)


def test_a_combine_of_another_command_or_naming_no_shard_passes_nothing(world):
    _main, tip, _config, store = world
    ids = _shards(world)
    _attempt(store, tip, ["bash", *COMMAND[1:], "--combine", *ids], "check/v1", {"steps": []})
    _attempt(store, tip, [*COMMAND, "--combine"], "check/v1", {"steps": []})
    _attempt(store, tip, [*COMMAND, "--combine", "--keep-going", *ids], "check/v1", {"steps": []})
    with pytest.raises(Refused) as e:
        _gate(world)
    assert str(e.value).count("isn't the target's declared") == 1 and str(e.value).count("which check.py doesn't take") == 2


def test_a_combine_without_the_targets_checkout_passes_nothing(world):
    from research.merge import passing, refused

    _main, tip, _config, store = world
    rid = _combine(world, _shards(world))
    assert refused(store, rid, frozenset({"check/v1"}), "check", tip, tuple(COMMAND)) == (
        "it combines shards, which this gate reads only in the target's checkout, for a commit")
    assert passing(store, tip, kinds=frozenset({"check/v1"}), command=tuple(COMMAND)) == []

