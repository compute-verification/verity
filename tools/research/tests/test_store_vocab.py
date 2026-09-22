"""The label vocabulary (`research.store.vocab`, `research data vocab`, the `research data label` warning), `research data
refresh-fixtures` and `research data doctor` -- the three additions the store grew after a day of real use.  Every test runs against
its own store root (conftest), its own repo and its own runs directory; nothing here reads ~/.research."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.store import fixtures as F
from research.store import vocab
from research.store.cli import main
from research.store.local import LocalStore

ART = "art:" + "a" * 64
RUN_STALE, RUN_REAL, RUN_FINE = "r20260922-105905-2021", "r20260922-110000-beef", "r20260922-110100-cafe"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A tmp store root, an absent remote config, a tmp repo with an empty fixtures/ directory, a tmp runs root."""
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    monkeypatch.setenv("RESEARCH_REPO", str(tmp_path / "repo"))
    monkeypatch.setenv("RESEARCH_RUNS", str(tmp_path / "runs"))
    (tmp_path / "repo" / "fixtures").mkdir(parents=True)
    (tmp_path / "runs").mkdir()
    return tmp_path


@pytest.fixture
def fs_remote(env):
    (env / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{env / "vault"}"\n')
    return env / "vault"


# ---- vocabulary -----------------------------------------------------------------------------------------------------------------------

RESULT_KEYS = ("candidate", "proof_class", "soundness", "K", "B", "hardware", "authentication", "campaign", "relation", "mode", "zk", "overhead",
               "seconds_per_vu", "track", "scope", "label", "note", "assumptions", "omitted", "breakthrough")
VERIFICATION_KEYS = ("verified", "verifier", "verifier_seconds", "independently_verified", "verified_by")
EVIDENCE_KEYS = ("instruction", "model", "model_agrees", "model_status", "model_rejected", "evidence", "family", "instances_dataset", "instances_tier",
                 "mismatches", "target", "component", "jobs")


def test_vocabulary_pins_every_key_by_asserter_and_form():
    groups = {k.name: k.group for k in vocab.VOCAB.values()}
    assert {n for n, g in groups.items() if g == vocab.GROUP_RESULT} == set(RESULT_KEYS)
    assert {n for n, g in groups.items() if g == vocab.GROUP_VERIFICATION} == set(VERIFICATION_KEYS)
    assert {n for n, g in groups.items() if g == vocab.GROUP_EVIDENCE} == set(EVIDENCE_KEYS)
    assert vocab.KEYS == tuple(vocab.VOCAB) and len(vocab.KEYS) == len(RESULT_KEYS) + len(VERIFICATION_KEYS) + len(EVIDENCE_KEYS)
    # the constants are the strings (a consumer imports vocab.PROOF_CLASS, never spells it)
    assert (vocab.CANDIDATE, vocab.PROOF_CLASS, vocab.VERIFIED, vocab.MODEL_AGREES, vocab.INDEPENDENTLY_VERIFIED, vocab.VERIFIED_BY) == \
        ("candidate", "proof_class", "verified", "model_agrees", "independently_verified", "verified_by")
    # the closed sets asked for
    assert vocab.key("verified").values == ("accepted", "rejected", "not-transferable", "no-dumps") == vocab.VERIFIED_VALUES
    assert vocab.key("model_agrees").form == vocab.BOOL and vocab.key("model_agrees").values == ("true", "false")
    assert vocab.key("independently_verified").form == vocab.BOOL and vocab.key("independently_verified").status == vocab.DEPRECATED
    assert vocab.key("verified_by").status == vocab.IGNORED
    assert vocab.key("proof_class").values == vocab.PROOF_CLASSES and "NON_ZK_PROOF" in vocab.PROOF_CLASSES
    assert vocab.key("authentication").values == ("included", "excluded")
    assert vocab.key("mode").values == ("interactive", "fiat-shamir")
    for k in vocab.VOCAB.values():
        assert k.form in (vocab.ENUM, vocab.BOOL, vocab.INT, vocab.NUMBER, vocab.TEXT, vocab.LIST) and k.doc
        assert bool(k.values) == (k.form in (vocab.ENUM, vocab.BOOL)), k.name
        assert k.status in (vocab.CURRENT, vocab.DEPRECATED, vocab.IGNORED)


def test_check_is_enum_and_bool_only_and_reads_values_as_the_catalog_does():
    assert vocab.check("verified", "accepted") is None
    assert "not one of accepted, rejected, not-transferable, no-dumps" in vocab.check("verified", "yes")
    assert vocab.check("model_agrees", True) is None and vocab.check("model_agrees", "true") is None       # bool and its scalar_text agree
    assert vocab.check("model_agrees", False) is None and vocab.check("model_agrees", "false") is None
    assert "'True' is not one of true, false" in vocab.check("model_agrees", "True")                       # a Python repr is a different word
    assert vocab.check("model_agrees", "yes") and vocab.check("zk", 1)
    assert vocab.check("proof_class", "non-zk") and vocab.check("proof_class", "NON_ZK_PROOF") is None
    assert vocab.check("candidate", "A-GKR") and vocab.check("candidate", "A") is None
    # free forms are documentation, never checked
    for key, value in (("soundness", "2^-128 (heuristic)"), ("K", 1536), ("K", "1536"), ("overhead", 1.5), ("assumptions", ["ROM"]), ("note", "")):
        assert vocab.check(key, value) is None, (key, value)
    assert "not in the vocabulary" in vocab.check("checker", "x") and vocab.is_known("checker") is False and vocab.key("checker") is None
    assert vocab.is_known("verified")


def test_render_and_cli_vocab(env, capsys):
    text = vocab.render()
    for name in (*RESULT_KEYS, *VERIFICATION_KEYS, *EVIDENCE_KEYS):
        assert f"\n  {name:<24} " in text or text.startswith(f"  {name:<24} "), name
    assert "accepted | rejected | not-transferable | no-dumps" in text and "[deprecated]" in text and "[ignored]" in text
    assert [g for g in vocab.by_group()] == list(vocab.GROUPS) and sum(len(v) for v in vocab.by_group().values()) == len(vocab.KEYS)
    assert main(["vocab"]) == 0
    assert capsys.readouterr().out.strip() == text
    assert main(["vocab", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert set(doc) == set(vocab.KEYS)
    assert doc["verified"] == {"group": "verification", "form": "enum", "values": list(vocab.VERIFIED_VALUES), "status": "current",
                               "doc": vocab.key("verified").doc}
    assert doc["model_agrees"]["values"] == ["true", "false"] and doc["independently_verified"]["status"] == "deprecated"


def test_label_warns_on_stderr_and_writes_anyway(env, capsys):
    st = LocalStore()
    assert main(["label", ART, "checker", "v1", "--by", "lane-x"]) == 0
    out, err = capsys.readouterr()
    assert Path(out.strip()).is_file() and err.count("\n") == 1
    assert "warning: label key 'checker' is not in the vocabulary" in err and "stored anyway" in err
    assert main(["label", ART, "model_agrees", "True", "--by", "lane-x"]) == 0
    out, err = capsys.readouterr()
    assert Path(out.strip()).is_file() and "warning: model_agrees='True' is not one of true, false; stored anyway" in err
    assert main(["label", ART, "model_agrees", "true", "--by", "lane-x", "--value-json"]) == 0
    out, err = capsys.readouterr()
    assert Path(out.strip()).is_file() and err == ""
    assert main(["label", ART, "verified", "accepted", "--by", "verifier-1", "--ref", "run:x"]) == 0
    out, err = capsys.readouterr()
    assert Path(out.strip()).is_file() and err == ""
    # every one of the four is in the store, the warned ones included, exactly as given
    labels = {(l.key, l.value) for l in st.labels(ART)}
    assert labels == {("checker", "v1"), ("model_agrees", "True"), ("model_agrees", True), ("verified", "accepted")}


# ---- refresh-fixtures -----------------------------------------------------------------------------------------------------------------


def _register(st: LocalStore, repo: Path) -> tuple[str, str]:
    """Two repo replicas the way the map has always been written: a file and a tree, bare fixture/v1, meta {} refs {}."""
    f = repo / "fixtures" / "probe" / "a.json"
    d = repo / "fixtures" / "probe" / "battery"
    d.mkdir(parents=True)
    f.write_text('{"x": 1}\n')
    (d / "one.txt").write_text("one\n")
    (d / "sub").mkdir()
    (d / "sub" / "two.txt").write_text("two\n")
    a = st.put_artifact("fixture/v1", {}, None, file=f)
    b = st.put_artifact("fixture/v1", {}, None, tree=d)
    assert F.fixture_id(f) == a and F.fixture_id(d) == b          # the module recomputes exactly what `put` registered
    (repo / F.REPLICA_MAP).write_text(json.dumps({a: "fixtures/probe/a.json", b: "fixtures/probe/battery"}, indent=2) + "\n")
    return a, b


def test_refresh_fixtures_is_idempotent_and_records_history(env, capsys):
    st = LocalStore()
    repo = env / "repo"
    a, b = _register(st, repo)
    before = (repo / F.REPLICA_MAP).read_text()
    assert main(["refresh-fixtures"]) == 0
    out, err = capsys.readouterr()
    assert out == "" and "2 fixture(s)" in err and "0 changed, 2 unchanged, 0 missing; map unchanged" in err
    assert (repo / F.REPLICA_MAP).read_text() == before
    # change the file, add to the tree: dry run reports both, puts nothing, writes nothing
    (repo / "fixtures" / "probe" / "a.json").write_text('{"x": 2}\n')
    (repo / "fixtures" / "probe" / "battery" / "three.txt").write_text("three\n")
    a2, b2 = F.fixture_id(repo / "fixtures" / "probe" / "a.json"), F.fixture_id(repo / "fixtures" / "probe" / "battery")
    n_manifests = len(list((st.root / "manifests").rglob("*.json")))
    assert main(["refresh-fixtures", "--dry-run"]) == 0
    out, err = capsys.readouterr()
    assert f"{a} -> {a2}  fixtures/probe/a.json" in out and f"{b} -> {b2}  fixtures/probe/battery" in out
    assert "2 changed, 0 unchanged, 0 missing; dry run, nothing written" in err
    assert (repo / F.REPLICA_MAP).read_text() == before and len(list((st.root / "manifests").rglob("*.json"))) == n_manifests
    assert not st.manifest_path(a2[4:]).exists()
    # for real: put under the new ids with the registered shape, map rewritten, old ids kept under `previous`
    assert main(["refresh-fixtures"]) == 0
    out, err = capsys.readouterr()
    assert out.splitlines() == [f"{a} -> {a2}  fixtures/probe/a.json", f"{b} -> {b2}  fixtures/probe/battery"]
    assert "2 changed, 0 unchanged, 0 missing; map rewritten" in err
    for new in (a2, b2):
        m = st.get_manifest(new)
        assert m.kind == "fixture/v1" and m.meta == {} and m.refs == {}
    assert st.get_manifest(a2).payload.type == "file" and st.get_manifest(b2).payload.type == "tree"
    doc = json.loads((repo / F.REPLICA_MAP).read_text())
    assert list(doc) == [a2, b2, "previous"]                                       # entry order kept, history last
    assert doc[a2] == "fixtures/probe/a.json" and doc[b2] == "fixtures/probe/battery"
    assert doc["previous"] == {a2: [a], b2: [b]}
    assert (repo / F.REPLICA_MAP).read_text().endswith("}\n")
    # the store reads the rewritten map as before: entries only, the `previous` block is invisible
    fresh = LocalStore()
    assert fresh.replica_map() == {a2: "fixtures/probe/a.json", b2: "fixtures/probe/battery"}
    assert fresh.where(a2)[1]["present"] is True                                    # the repo replica is verified at the new id
    # idempotent
    text = (repo / F.REPLICA_MAP).read_text()
    assert main(["refresh-fixtures"]) == 0
    out, err = capsys.readouterr()
    assert out == "" and "0 changed, 2 unchanged, 0 missing; map unchanged" in err and (repo / F.REPLICA_MAP).read_text() == text
    # a second change accumulates history, oldest first; reverting the bytes to an old id lists that id no more
    (repo / "fixtures" / "probe" / "a.json").write_text('{"x": 3}\n')
    a3 = F.fixture_id(repo / "fixtures" / "probe" / "a.json")
    assert main(["refresh-fixtures"]) == 0
    assert f"{a2} -> {a3}  fixtures/probe/a.json  (previously also {a})" in capsys.readouterr().out
    assert json.loads((repo / F.REPLICA_MAP).read_text())["previous"] == {a3: [a, a2], b2: [b]}
    (repo / "fixtures" / "probe" / "a.json").write_text('{"x": 1}\n')
    assert main(["refresh-fixtures"]) == 0
    capsys.readouterr()
    doc = json.loads((repo / F.REPLICA_MAP).read_text())
    assert doc[a] == "fixtures/probe/a.json" and doc["previous"] == {a: [a2, a3], b2: [b]}


def test_refresh_fixtures_preserve_pushes_and_reports(env, fs_remote, capsys):
    st = LocalStore()
    repo = env / "repo"
    a, _ = _register(st, repo)
    (repo / "fixtures" / "probe" / "a.json").write_text('{"x": 9}\n')
    a2 = F.fixture_id(repo / "fixtures" / "probe" / "a.json")
    assert main(["refresh-fixtures", "--preserve", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["preserve"] and rep["written"] and rep["changed"] == 1 and rep["unchanged"] == 1
    e = next(x for x in rep["entries"] if x["status"] == "changed")
    assert e == {"id": a, "path": "fixtures/probe/a.json", "status": "changed", "new_id": a2, "previous": [a], "preserved": True, "errors": []}
    assert st.verify(a2)["preserved"] is True
    assert main(["refresh-fixtures", "--preserve"]) == 0
    assert "0 changed, 2 unchanged" in capsys.readouterr().err
    # a vault that cannot be written is reported per entry and exits 1; the map is still made honest
    (repo / "fixtures" / "probe" / "a.json").write_text('{"x": 10}\n')
    a3 = F.fixture_id(repo / "fixtures" / "probe" / "a.json")
    (env / "store.toml").write_text('[remote]\ntype = "fs"\npath = "/dev/null/not-a-dir"\n')
    assert main(["refresh-fixtures", "--preserve"]) == 1
    assert f"{a2} -> {a3}  fixtures/probe/a.json  (previously also {a})  NOT preserved: " in capsys.readouterr().out
    assert json.loads((repo / F.REPLICA_MAP).read_text())[a3] == "fixtures/probe/a.json"


def test_refresh_fixtures_missing_conflict_and_no_map(env, tmp_path, capsys):
    st = LocalStore()
    repo = env / "repo"
    a, _ = _register(st, repo)
    # a path that left the working tree is reported, kept, and fails the command; nothing else changes
    (repo / "fixtures" / "probe" / "a.json").unlink()
    text = (repo / F.REPLICA_MAP).read_text()
    assert main(["refresh-fixtures"]) == 1
    out, err = capsys.readouterr()
    assert f"{a}  fixtures/probe/a.json  MISSING: not in the working tree" in out and "0 changed, 1 unchanged, 1 missing" in err
    assert (repo / F.REPLICA_MAP).read_text() == text
    # two entries whose bytes (and bare file names -- identity is content plus name) become identical would collide on one id:
    # reported as a conflict, neither is rewritten
    (repo / "fixtures" / "probe" / "a.json").write_text('{"x": 1}\n')
    (repo / "fixtures" / "other").mkdir()
    (repo / "fixtures" / "other" / "a.json").write_text('{"x": 1}\n')
    doc = json.loads(text)
    doc["art:" + "f" * 64] = "fixtures/other/a.json"
    (repo / F.REPLICA_MAP).write_text(json.dumps(doc, indent=2) + "\n")
    assert main(["refresh-fixtures", "--json"]) == 1
    rep = json.loads(capsys.readouterr().out)
    bad = [x for x in rep["entries"] if x["status"] == "conflict"]
    assert len(bad) == 1 and bad[0]["new_id"] == a and "already registered" in bad[0]["errors"][0] and not rep["written"]
    # without a map there is nothing to refresh
    assert main(["refresh-fixtures", "--repo", str(tmp_path / "elsewhere")]) == 1
    assert "no fixtures/artifacts.json found" in capsys.readouterr().err


# ---- doctor -----------------------------------------------------------------------------------------------------------------------------


def _attempt(run_id: str) -> dict:
    t = "2026-09-22T10:59:05Z"
    return {"schema": "research/attempt/v0.1", "id": run_id, "tool": {"name": "bench_vu", "version": "1"}, "derivation": "drv:" + "a" * 64,
            "reuse_key": "b" * 64, "argv": ["python", "-m", "x"], "params": {}, "key_params": [], "inputs": {}, "conditions": {}, "key_conditions": [],
            "source": {"commit": "c" * 40, "tree": "d" * 40, "dirty": False, "dirty_digest": None, "lockfiles": {}}, "closure": {},
            "execution": {"start_utc": t, "end_utc": t, "wall_s": 1.5, "host": "vy-g5", "project": "verity", "campaign": "r21"},
            "cost": {"provider": "runpod", "usd": 0.01}, "status": {"state": "exited", "rc": 0, "class": None, "failure": None, "validation": "passed"},
            "outputs": {}, "telemetry": {"events": None, "resources": None}}


def _run_dir(root: Path, name: str, *, error: str | None, state: str | None) -> Path:
    d = root / name
    d.mkdir()
    (d / "job.json").write_text("{}")
    if state:
        (d / "status.json").write_text(json.dumps({"state": state}))
    if error is not None:
        (d / "store_publish_error.json").write_text(json.dumps({"error": error, "at_utc": "2026-09-22T09:00:00Z"}))
    return d


def test_doctor_lists_stale_versus_real_and_cleans_only_the_stale(env, capsys):
    st = LocalStore()
    runs = env / "runs"
    st.put_attempt(_attempt(RUN_STALE))
    stale = _run_dir(runs, RUN_STALE, error="ConfigError: credentials env vars AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set\nmore", state="done")
    real = _run_dir(runs, RUN_REAL, error=f"RuntimeError: run {RUN_REAL} is running: the store only sees completed or failed attempts", state="running")
    fine = _run_dir(runs, RUN_FINE, error=None, state="done")
    odd = _run_dir(runs, "not-a-run-id", error="whatever", state=None)
    (runs / "loose-file.txt").write_text("not a run dir")
    # read-only by default: two rows, two counts, nothing deleted
    assert main(["doctor"]) == 0
    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert lines == [f"{RUN_STALE}  stale  attempt in the store  ConfigError: credentials env vars AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set",
                     "not-a-run-id  REAL   state=?  whatever",
                     f"{RUN_REAL}  REAL   state=running  RuntimeError: run {RUN_REAL} is running: the store only sees completed or failed attempts"]
    assert f"3 store_publish_error.json file(s) under {runs}: 1 stale (run is an attempt in {st.root}), 2 real (not in the store); read-only" in err
    assert (stale / "store_publish_error.json").is_file() and (real / "store_publish_error.json").is_file()
    # the JSON shape
    assert main(["doctor", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["cleaned"] is False and [r["run"] for r in rep["stale"]] == [RUN_STALE] and {r["run"] for r in rep["real"]} == {RUN_REAL, "not-a-run-id"}
    assert rep["stale"][0]["state"] == "done" and rep["stale"][0]["at_utc"] == "2026-09-22T09:00:00Z" and rep["stale"][0]["deleted"] is False
    # --clean deletes the stale file and nothing else
    before = sorted(p.relative_to(runs).as_posix() for p in runs.rglob("*"))
    assert main(["doctor", "--clean"]) == 0
    out, err = capsys.readouterr()
    assert f"{RUN_STALE}  stale  deleted  ConfigError" in out and "deleted 1 stale file(s), nothing else touched" in err
    after = sorted(p.relative_to(runs).as_posix() for p in runs.rglob("*"))
    assert set(before) - set(after) == {f"{RUN_STALE}/store_publish_error.json"}
    assert (stale / "job.json").is_file() and (stale / "status.json").is_file()
    assert (real / "store_publish_error.json").is_file() and (odd / "store_publish_error.json").is_file() and (fine / "job.json").is_file()
    # a second pass finds only the real ones; --runs-dir points elsewhere
    assert main(["doctor"]) == 0
    out, err = capsys.readouterr()
    assert RUN_STALE not in out and "0 stale" in err and "2 real" in err
    assert main(["doctor", "--runs-dir", str(env / "nowhere")]) == 0
    assert "0 store_publish_error.json file(s)" in capsys.readouterr().err
