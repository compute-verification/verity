"""Private files (store.private): custody stores a run's record without them and says which it refused, leaves them on the machine
and never evicts them; `research data put`, a typed output and `research run-preserve` store none.  The remote is an `fs` vault."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from research import cli
from research import runs as R
from research.store import custody as CU
from research.store import private as P
from research.store import run_preserve as RP
from research.store.cli import main as data_main
from research.store.local import LocalStore

from test_run_custody import FIXTURE_RID, _custody_dir, _declare, _fixture_run, _other_machine

OLD_MANIFEST = '{"scheme": "pearl-c", "hm96_salt_key": "00ff", "leaves": "leaves"}\n'

#: (path under the run dir, the rule that refuses it): one of each name and place RULES holds
PRIVATE = [
    ("out/dump/keys.bin", "name keys.bin"),
    ("out/serve/private/weights.key", "name weights.key"),
    ("out/a/salts", "name salts"),
    ("out/a/salts_t", "name salts_t"),
    ("out/y/7.salts", "name *.salts"),
    ("out/registration.salts/index.json", "under <R>.salts/ beside <R>.json"),
    ("out/registration.salts/3.bin", "under <R>.salts/ beside <R>.json"),
    ("out/retain/pass-00004/calls/0.tree", "under pass-[0-9][0-9][0-9][0-9][0-9]*/"),
    ("out/retain/pass-00004/manifest.json", "under pass-[0-9][0-9][0-9][0-9][0-9]*/"),
    ("out/b/manifest.json", "manifest.json beside salts or salts_t"),
    ("out/old/manifest.json", "*.json holding hm96_salt_key"),
    ("out/row/commit/replay_bundle_p0/commit/values.json", "kept apart replay_bundle_p*"),
    ("out/row/commit/replay_slim_p1/slim.json", "kept apart replay_slim_p*"),
    ("out/serve/requests.jsonl", "name requests.jsonl"),
]
#: names in stored runs today whose owners have not said they are private: uploaded as before
UNDECIDED = ["out/saltskeyed.tsv", "out/challenger/key.json", "out/lean.key"]


def _store(tmp_path: Path, monkeypatch) -> Path:
    vault = tmp_path / "vault"
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{vault}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    return vault


def _write(d: Path, rel: str, data: bytes | str = b"x") -> Path:
    p = d / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    (p.write_bytes if isinstance(data, bytes) else p.write_text)(data)
    return p


def _private_run(runs: Path) -> Path:
    d = _fixture_run(runs)
    for rel, _ in PRIVATE:
        _write(d, rel, OLD_MANIFEST if rel == "out/old/manifest.json" else b"s" * 64)
    _write(d, "out/b/salts_t", b"t" * 64)
    _write(d, "out/registration.json", '{"root": "public"}\n')
    for rel in UNDECIDED:
        _write(d, rel, '{"challenger": "public"}\n' if rel.endswith(".json") else "kept\n")
    return d


def test_every_rule_has_a_one_line_reason_and_an_owning_lanes_handle():
    doc = json.loads(P.floor_text())
    assert P.FLOOR_UNREAD is None and P.FLOOR_ROOTS[0] / P.TREE_NAMES == Path(__file__).resolve().parents[3] / P.TREE_NAMES
    assert doc["schema"] == "verity/private-names/v1" and set(doc["why"]) == {*doc["kept_apart"], *doc["private"]}, \
        "each entry has its why, and no more"
    assert len(set(doc["private"])) == len(doc["private"]), "a name is listed once"
    assert all(len(w) == 2 for w in doc["why"].values()), "why: {glob: [owner, reason]}"
    assert len({r.what for r in P.RULES}) == len(P.RULES), "two rules print alike"
    for r in P.RULES:
        assert r.reason.strip() and "\n" not in r.reason, f"{r.what}: a one-line reason"
        assert re.fullmatch(r"@[a-z][a-z0-9-]*", r.owner), f"{r.what}: owner {r.owner!r} is not a lane handle"


@pytest.mark.parametrize("rel,rule", PRIVATE, ids=[r for r, _ in PRIVATE])
def test_each_name_and_place_is_private(tmp_path, rel, rule):
    d = _private_run(tmp_path / "runs")
    got = P.rule_for(rel, d / rel)
    assert got is not None and got.what == rule


@pytest.mark.parametrize("rel", [*UNDECIDED, "out/c/manifest.json", "out/note.txt", "out/big.json", "out/tiny.salts/index.json",
                                 "out/registration.json"])
def test_no_false_refusal(tmp_path, rel):
    """A manifest with no salts beside it, the key's name in a file that is not JSON, a JSON file over the scan's 4 MiB limit
    (never read), and a salt directory with no registration beside it (a committed fixture's shape) stay uploadable, with the
    three names nobody has ruled on yet."""
    d = _private_run(tmp_path / "runs")
    _write(d, "out/tiny.salts/index.json", "{}\n")
    _write(d, "out/c/manifest.json", '{"leaves": "leaves"}\n')
    _write(d, "out/note.txt", "hm96_salt_key\n")
    _write(d, "out/big.json", '{"hm96_salt_key": "' + "0" * P.SCAN_MAX_BYTES + '"}')
    assert P.rule_for(rel, d / rel) is None


def test_custody_stores_the_rest_lists_the_refused_and_never_evicts_them(tmp_path, monkeypatch, capsys):
    vault = _store(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    d = _private_run(runs)
    _write(d, "out/dump/keys.bin", b"k" * (2 << 20))           # big enough that custody --publish would evict it if it were stored
    _write(d, "out/big.bin", b"b" * (2 << 20))
    _declare(d, ["*", "out/a/salts"])                           # a declared glob names nothing private in
    (d / "outputs.json").write_text(json.dumps({"schema": "research/outputs/v0.1",
                                                "outputs": [{"name": "retained", "kind": "x-pass/v1", "tree": "out/retain"}]}))
    before = {rel: (d / rel).read_bytes() for rel, _ in PRIVATE}
    assert data_main(["custody", FIXTURE_RID, "--runs-dir", str(runs), "--publish"]) == 0
    io = capsys.readouterr()
    refused_paths = sorted([*(r for r, _ in PRIVATE), "out/b/salts_t"])
    assert f"custody REFUSED {len(refused_paths)} private file(s), not uploaded and left on this machine" in io.err
    assert f"{FIXTURE_RID}: custody REFUSED {len(refused_paths)} private file(s)" in io.out and "(never evicted)" in io.out
    assert "outputs.json: retained not stored" in io.err

    st2 = _other_machine(tmp_path, vault, monkeypatch)
    doc = st2.get_attempt(FIXTURE_RID)
    assert doc["outputs"].get("retained") is None, "a typed output holding a private file is not stored"
    m = st2.get_manifest(doc["outputs"]["run_record"])
    paths = {f.path for f in m.payload.files}
    assert not paths & set(refused_paths), "no private file is in the record"
    assert {"stdout.log", "job.json", "out/proof.bin", "out/big.bin", CU.REFUSED_FILE, *UNDECIDED} <= paths
    assert m.meta["refused_files"] == len(refused_paths) and [r["path"] for r in m.meta["refused"]] == refused_paths
    assert not m.meta.get("trimmed"), "a private file is refused, never listed by hash"
    assert not any(b"s" * 64 in p.read_bytes() or b"hm96_salt_key\": \"00ff" in p.read_bytes() for p in vault.rglob("*") if p.is_file())

    marker = json.loads((d / CU.MARKER).read_text())
    assert marker["preserved"] and CU.has_custody(d) == (True, "on the remote (.custody)")
    assert {(r["path"], r["rule"]) for r in marker["refused"]} >= set(PRIVATE) and all(r["owner"].startswith("@") for r in marker["refused"])
    assert P.listed(d) == marker["refused"]
    assert {rel: (d / rel).read_bytes() for rel, _ in PRIVATE} == before, "every private file is left as it was"
    assert not (d / "out/big.bin").exists(), "custody --publish evicted the stored big file, and only that"
    evicted = json.loads((d / "run-files.evicted.json").read_text())
    assert [f["path"] for f in evicted["files"]] == ["out/big.bin"]

    assert R.summarize(d)["custody_refused"] == marker["refused"]
    assert data_main(["preserved", FIXTURE_RID]) == 0
    assert f"{len(refused_paths)} private file(s) refused" in capsys.readouterr().out
    assert cli.cmd_status([FIXTURE_RID, "--runs-dir", str(runs)]) == 0
    assert f"priv {len(refused_paths)} private file(s) refused by custody" in capsys.readouterr().out


def test_the_runner_says_what_it_refused_before_the_record_and_marks_custody(tmp_path, capsys):
    runs = tmp_path / "root" / "runs"
    d = _fixture_run(runs)
    _write(d, "out/y/0.salts")
    cdir = _custody_dir(tmp_path, FIXTURE_RID, tmp_path / "vault")
    rep = CU.runner_publish(d, cdir, sleep=lambda s: None, tries=1)
    err = capsys.readouterr().err
    assert rep["preserved"], rep.get("error")
    assert ("research: custody REFUSED 1 private file(s), not uploaded and left on this machine: out/y/0.salts (name *.salts, "
            "@compute-accounting)") in err
    assert json.loads((d / CU.MARKER).read_text())["refused"] == [{"path": "out/y/0.salts", "rule": "name *.salts",
                                                                   "owner": "@compute-accounting"}]


def test_a_symlink_to_a_private_file_or_directory_is_refused_and_never_followed(tmp_path):
    d = _fixture_run(tmp_path / "runs")
    elsewhere = tmp_path / "elsewhere"
    key = _write(elsewhere, "keys.bin", b"k")
    salts = _write(elsewhere, "reg.salts/0.bin", b"s").parent
    plain = _write(elsewhere, "plain.txt", b"p")
    (d / "out" / "innocent.bin").symlink_to(key)
    (d / "out" / "reg").symlink_to(salts)
    (d / "out" / "x.salts").symlink_to(plain)
    (d / "out" / "plain").symlink_to(plain)
    kept, trimmed, refused = CU.choose(d)
    assert refused == [{"path": "out/innocent.bin", "rule": "name keys.bin", "owner": "@vllm", "link": True},
                       {"path": "out/reg", "rule": "name *.salts", "owner": "@compute-accounting", "link": True},
                       {"path": "out/x.salts", "rule": "name *.salts", "owner": "@compute-accounting", "link": True}]
    assert not any(k.startswith(("out/innocent", "out/reg", "out/x.salts", "out/plain")) for k in kept), "no link is uploaded"
    assert not trimmed and key.read_bytes() == b"k"


def test_research_data_put_refuses_a_tree_or_file_holding_a_private_file_and_stores_nothing(tmp_path, capsys):
    store = tmp_path / "store"
    tree = tmp_path / "t"
    _write(tree, "ok.txt", "fine")
    _write(tree, "pass-00000/leaves", b"l")
    assert data_main(["put", "--store", str(store), "--kind", "x/v1", "--tree", str(tree)]) == 1
    err = capsys.readouterr().err
    assert "pass-00000/leaves: private (under pass-" in err and "REFUSED, nothing stored: 1 private file(s)" in err
    assert not list((store / "manifests").rglob("*.json"))
    salt = _write(tmp_path, "f/a.salts", b"s")
    assert data_main(["put", "--store", str(store), "--kind", "x/v1", "--file", str(salt)]) == 1
    linked = tmp_path / "t2"
    _write(linked, "ok.txt", "fine")
    (linked / "alias.bin").symlink_to(salt)                     # a put stores a link's target by content: refused by the target
    assert data_main(["put", "--store", str(store), "--kind", "x/v1", "--tree", str(linked)]) == 1
    capsys.readouterr()
    assert not list((store / "manifests").rglob("*.json"))
    (linked / "alias.bin").unlink()
    assert data_main(["put", "--store", str(store), "--kind", "x/v1", "--tree", str(linked)]) == 0
    with pytest.raises(P.PrivateFiles, match="1 private file"):
        LocalStore(store).put_artifact("x/v1", {}, tree=tree)


def test_run_preserve_refuses_a_private_file_before_it_reads_the_key(tmp_path):
    d = _fixture_run(tmp_path / "runs")
    p = _write(d, "out/old/manifest.json", OLD_MANIFEST)
    with pytest.raises(RP.Refusal, match=r"out/old/manifest.json is private \(\*\.json holding hm96_salt_key"):
        RP._preserve(d, tmp_path / "no-custody-dir", {"path": str(p), "kind": RP.KIND, "meta": {}})
    assert os.path.exists(p)


def _stated(d: Path, private: list[str], names: list[str]) -> None:
    (d / P.PATHS_FILE).write_text(json.dumps({"schema": "private-paths/v1", "private": private, "names": names}))


def test_a_runs_statement_refuses_its_names_and_its_private_directories(tmp_path, capsys):
    vault = tmp_path / "vault"
    d = _fixture_run(tmp_path / "root" / "runs")
    priv = tmp_path / "private"
    secret = _write(priv, "secret.bin", b"s")
    _write(d, "out/plan.jsonl", b"p")
    _write(d, "out/capture/a.bin", b"c")
    _write(d, "out/draws.3.jsonl.gz", b"i")
    _write(d, "out/note/capture", b"a file, not a directory")
    (d / "out" / "held").symlink_to(secret)
    _stated(d, [str(priv)], ["plan.jsonl", "capture/", "draws.*.jsonl*"])
    rep = CU.runner_publish(d, _custody_dir(tmp_path, FIXTURE_RID, vault), sleep=lambda s: None, tries=1)
    assert rep["preserved"], rep.get("error")
    err = capsys.readouterr().err
    assert ("custody REFUSED 4 private file(s), not uploaded and left on this machine: out/capture/a.bin (private-paths.json: name "
            "capture/), out/draws.3.jsonl.gz (private-paths.json: name draws.*.jsonl*), out/held (private-paths.json: inside "
            + str(priv) + ") and 1 more") in err
    marker = json.loads((d / CU.MARKER).read_text())["refused"]
    assert {r["path"]: r["rule"] for r in marker} == {"out/capture/a.bin": "private-paths.json: name capture/",
                                                      "out/held": f"private-paths.json: inside {priv}",
                                                      "out/draws.3.jsonl.gz": "private-paths.json: name draws.*.jsonl*",
                                                      "out/plan.jsonl": "private-paths.json: name plan.jsonl"}
    assert all("owner" not in r for r in marker), "a statement's rule has no lane owner"
    kept, _, _ = CU.choose(d)
    assert P.PATHS_FILE in kept and "out/note/capture" in kept, "a trailing / names a directory only"


@pytest.mark.parametrize("doc", ['{"schema": "research/private-paths/v1", "rules": [{"path": "out/*", "reason": "r", "owner": "@x"}]}',
                                 "not json", '{"schema": "private-paths/v1", "private": ["relative/dir"], "names": []}',
                                 '{"schema": "private-paths/v1", "private": [], "names": [""]}'])
def test_an_unreadable_statement_stores_nothing_of_the_run(tmp_path, capsys, doc):
    vault = tmp_path / "vault"
    d = _fixture_run(tmp_path / "root" / "runs")
    (d / P.PATHS_FILE).write_text(doc)
    rep = CU.runner_publish(d, _custody_dir(tmp_path, FIXTURE_RID, vault), sleep=lambda s: None, tries=1)
    assert not rep["preserved"] and "private-paths.json" in rep["error"]
    assert "custody stores nothing of this run while its own private paths aren't known" in capsys.readouterr().err
    assert not [p for p in vault.rglob("*") if p.is_file()], "nothing of the run reached the remote"
    assert data_main(["put", "--store", str(tmp_path / "s"), "--kind", "x/v1", "--tree", str(d / "out")]) == 1


def test_a_put_under_a_run_dir_obeys_its_statement(tmp_path, capsys):
    d = _fixture_run(tmp_path / "runs")
    _write(d, "out/model/plan.jsonl", b"p")
    _stated(d, [], ["plan.jsonl"])
    assert data_main(["put", "--store", str(tmp_path / "s"), "--kind", "x/v1", "--tree", str(d / "out" / "model")]) == 1
    assert "plan.jsonl: private (private-paths.json: name plan.jsonl)" in capsys.readouterr().err


def _ship(d: Path, src: Path) -> None:
    job = json.loads((d / "job.json").read_text())
    job.setdefault("run", {}).setdefault("remote", {})["source_dir"] = str(src)
    (d / "job.json").write_text(json.dumps(job))


def test_the_runs_shipped_tree_adds_its_private_names_and_its_kept_apart_directories(tmp_path):
    d = _fixture_run(tmp_path / "runs")
    src = tmp_path / "src" / ("0" * 40)
    _write(src, P.TREE_NAMES, json.dumps({"schema": "verity/private-names/v1", "kept_apart": ["held_back_p"],
                                          "private": ["plan_values.jsonl"]}))
    _write(d, "out/plan_values.jsonl", b"r")
    _write(d, "out/held_back_p3/x/values.json", b"v")
    assert CU.choose(d)[2] == [], "without the run's tree, neither is a floor name"
    _ship(d, src)
    assert P.shipped_names(d) == (src / P.TREE_NAMES, None)
    assert CU.choose(d)[2] == [{"path": "out/held_back_p3/x/values.json", "rule": "the run's tree: kept apart held_back_p*"},
                               {"path": "out/plan_values.jsonl", "rule": "the run's tree: name plan_values.jsonl"}]
    (src / P.TREE_NAMES).write_text('{"schema": "other"}')
    with pytest.raises(P.PathsUnread, match="private_names.json"):
        CU.choose(d)


def test_a_missing_shipped_tree_is_named_in_the_record_and_the_floor_still_applies(tmp_path):
    d = _fixture_run(tmp_path / "runs")
    _write(d, "out/commit/replay_bundle_p0/values.json", b"v")
    gone = tmp_path / "src" / ("1" * 40)
    _ship(d, gone)
    why = f"the run's shipped tree {gone} is missing, so only the floor's private names applied"
    assert P.shipped_names(d) == (None, why)
    st = LocalStore(tmp_path / "store")
    m = st.get_manifest(CU.put_run_record(st, d, FIXTURE_RID))
    assert m.meta["tree_names"] == why
    assert [r["path"] for r in m.meta["refused"]] == ["out/commit/replay_bundle_p0/values.json"]
    gone.mkdir(parents=True)
    assert P.shipped_names(d) == (None, f"the run's shipped tree {gone} holds no {P.TREE_NAMES}, so only the floor's private names applied")


def _registry(repo: Path, rel: str, toys: dict | None = None) -> None:
    doc = {"art:" + "0" * 64: rel, **({"toys": toys} if toys is not None else {})}
    (repo / "artifacts.json").write_text(json.dumps(doc))


def test_a_registered_toy_fixture_is_put_with_its_private_named_files_and_nothing_else_is(tmp_path, capsys):
    from research.store import fixtures as FX

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    toy = _write(repo, "fx/toy/keys.bin", b"k").parent
    _registry(repo, "fx/toy")
    assert data_main(["put", "--store", str(tmp_path / "s"), "--kind", "x/v1", "--tree", str(toy)]) == 1
    assert "artifacts.json's `toys`" in capsys.readouterr().err
    errs, private = FX.private_fixtures(repo, json.loads((repo / "artifacts.json").read_text()), {"art:" + "0" * 64: FX.fixture_manifest(toy)})
    assert private == {"art:" + "0" * 64} and "fx/toy (art:" in errs[0] and '{"fx/toy": "why it is public"}' in errs[0]
    _registry(repo, "fx/toy", {"fx/toy": "a toy circuit's prover store on made-up inputs"})
    assert data_main(["put", "--store", str(tmp_path / "s"), "--kind", "x/v1", "--tree", str(toy)]) == 0
    assert "is a registered toy fixture (artifacts.json toys: a toy circuit's prover store" in capsys.readouterr().err
    assert FX.private_fixtures(repo, json.loads((repo / "artifacts.json").read_text()), {"art:" + "0" * 64: FX.fixture_manifest(toy)}) == ([], set())
    _registry(repo, "fx/other", {"fx/toy": "a toy, but not registered"})
    assert P.toy(toy) is None, "only a registered path is a toy"


def test_only_the_repository_roots_registry_marks_a_toy(tmp_path):
    """A hand-written artifacts.json nearer the payload than the repository's root, or in a directory that is no repository's
    root, marks nothing."""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    toy = _write(repo, "fx/toy/keys.bin", b"k").parent
    _registry(repo, "fx/other")
    _registry(repo / "fx", "toy", {"toy": "hand-written beside the payload"})
    assert P.toy(toy) is None
    loose = _write(tmp_path / "loose", "toy/keys.bin", b"k").parent
    _registry(tmp_path / "loose", "toy", {"toy": "no repository here"})
    assert P.toy(loose) is None
    _registry(repo, "fx/toy", {"fx/toy": "a toy circuit's prover store on made-up inputs"})
    assert P.toy(toy) == "a toy circuit's prover store on made-up inputs"


def test_fetch_all_copies_no_private_file_and_preserves_the_rest(tmp_path, monkeypatch):
    from research import remote

    d = _private_run(tmp_path / "machine" / "runs")
    _stated(d, [], ["plan.jsonl"])
    _write(d, "out/model/plan.jsonl", b"p")
    here = tmp_path / "here" / FIXTURE_RID
    here.mkdir(parents=True)
    (here / "remote.json").write_text(json.dumps({"remote_run_dir": str(d), "host": "h", "port": 22, "run_id": FIXTURE_RID}))
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **kw: ["sh", "-c"])
    rec = remote.fetch(here, everything=True)
    refused = sorted([*(r for r, _ in PRIVATE), "out/b/salts_t", "out/model/plan.jsonl"])
    assert [r["path"] for r in rec["refused"]] == refused and rec["preserved"], rec.get("preserve_error")
    assert not any((here / r).exists() for r in refused), "no private file reached the launcher"
    assert (here / "out/proof.bin").exists() and (here / P.PATHS_FILE).exists() and (here / UNDECIDED[0]).exists()
    assert [r["path"] for r in json.loads((here / "preserved.json").read_text())["refused"]] == refused
    assert all((d / r).exists() for r in refused), "each stays on the machine"

    (d / P.PATHS_FILE).write_text("[")
    with pytest.raises(remote.Refused, match="could not list the run's private files, so nothing was copied"):
        remote.fetch(tmp_path / "here" / FIXTURE_RID, everything=True)
