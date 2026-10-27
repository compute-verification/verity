"""The provenance half of research.store: conditions(), Tool closures and params, derivation ids, reuse keys, scratch dirs,
attempt manifests, publish (against a FakeStore with store-core's signatures), the run integration, lookup and pull."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from research import env as E
from research.cli import main as research_main
from research.store import attempt as A
from research.store import canon
from research.store import cli_prov
from research.store import derivation as D
from research.store import model as M
from research.store import tools_registry
from research.store.tool import Tool, content_identity, parse_argv, tree_listing
from waits import wait_runner

FIXTURES = Path(__file__).parent / "fixtures"
RUN_FIXTURE = FIXTURES / "run_v0.2"
RUN_ID = "r20260922-120000-abcd"
REPO = Path(__file__).resolve().parents[3]
PY = sys.executable


# ---- a store double with exactly the signatures store-core exposes (README §2/§3 layout on disk, so `pull` can read it over a fake ssh)

class FakeIndex:
    def __init__(self, store: "FakeStore"):
        self.store = store

    def attempts_by_reuse_key(self, key):
        return [d for d in self.store.attempts.values() if d.get("reuse_key") == key]

    def attempts_by_derivation(self, drv):
        return [d for d in self.store.attempts.values() if d.get("derivation") == drv]


class FakeStore:
    def __init__(self, root):
        self.root = Path(root)
        self.artifacts: dict[str, dict] = {}
        self.attempts: dict[str, dict] = {}
        self.blobs: dict[str, bytes] = {}
        self.index = FakeIndex(self)
        for d in ("objects/sha256", "manifests", "attempts"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def _blob(self, data: bytes) -> str:
        h = hashlib.sha256(data).hexdigest()
        self.blobs[h] = data
        (self.root / "objects/sha256" / h).write_bytes(data)
        return h

    def put_artifact(self, kind, meta, refs=None, *, file=None, tree=None) -> str:
        payload = None
        if file is not None:
            data = Path(file).read_bytes()
            payload = {"type": "file", "sha256": self._blob(data), "bytes": len(data), "name": Path(file).name}
        elif tree is not None:
            files = tree_listing(Path(tree))
            for f in files:
                self._blob((Path(tree) / f["path"]).read_bytes())
            payload = {"type": "tree", "sha256": canon.sha256(files), "bytes": sum(f["bytes"] for f in files), "files": files}
        manifest = {"schema": "research/artifact/v0.1", "kind": kind, "meta": meta, "refs": refs or {}, "payload": payload}
        b = canon.dumps(manifest)
        art_id = "art:" + hashlib.sha256(b).hexdigest()
        self.artifacts[art_id] = manifest
        (self.root / "manifests" / f"{art_id[4:]}.json").write_bytes(b)
        return art_id

    def put_attempt(self, doc) -> None:
        b = canon.dumps(doc)
        if doc["id"] in self.attempts:
            if canon.dumps(self.attempts[doc["id"]]) != b:
                raise ValueError(f"attempt {doc['id']} exists with different content")
            return
        self.attempts[doc["id"]] = doc
        (self.root / "attempts" / f"{doc['id']}.json").write_bytes(b)

    def get_attempt(self, run_id):
        return self.attempts.get(run_id)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
                          env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "tools" / "x" / "sub").mkdir(parents=True)
    (root / "other").mkdir()
    (root / "tools/x/a.py").write_text("a = 1\n")
    (root / "tools/x/sub/b.py").write_text("b = 2\n")
    (root / "other/c.py").write_text("c = 3\n")
    (root / ".gitignore").write_text("*.pyc\n__pycache__/\n")
    (root / "uv.lock").write_text("version = 1\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init")
    return root


# ---- conditions ----------------------------------------------------------------------------------------------------------------------------

def test_conditions_every_key_present_scalar(tmp_path):
    c = E.conditions(run_dir=tmp_path / "runs" / "r1", project="p", campaign="camp", exclusive=False, source_root=tmp_path)
    for k in E.KEYS:
        assert k in c, k
    assert all(v is None or isinstance(v, (str, int, float, bool)) for v in c.values()), "scalars only"
    assert c["exec.project"] == "p" and c["exec.campaign"] == "camp" and c["exec.exclusive"] is False
    assert c["exec.runs_root"] == str(tmp_path / "runs") and c["exec.load_end"] is None
    assert c["source.commit"] is None and c["source.dirty"] is None, "a plain directory is not a guessed clean tree"
    assert c["rt.python"] and c["hw.hostname"]


@pytest.mark.parametrize("sku,count,family", [
    ("NVIDIA H100 80GB HBM3", 1, "h100"), ("NVIDIA H100 PCIe", 2, "h100"), ("NVIDIA H200", 1, "h200"),
    ("NVIDIA A100-SXM4-80GB", 1, "a100"), ("NVIDIA A100 80GB PCIe", 1, "a100"),
    ("NVIDIA L40S", 4, "l40s"), ("NVIDIA L40", 1, "l40"),
    ("NVIDIA GeForce RTX 4090", 1, "rtx4090"), ("NVIDIA GeForce RTX 5090", 1, "rtx5090"),
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 4, "rtxpro6000"),
    ("NVIDIA B200", 8, "b200"), ("NVIDIA A10G", 1, "a10"), ("NVIDIA A10", 1, "a10"), ("Tesla T4", 1, "t4"),
    (None, 0, "cpu"), ("Some Future GPU 9000", 1, "unknown"), (None, None, None), (None, 1, None),
])
def test_sku_family_map(sku, count, family):
    assert E.sku_family(sku, count) == family


def test_conditions_without_gpu_are_none_not_guessed(monkeypatch, tmp_path):
    real_which = shutil.which
    monkeypatch.setattr(E.shutil, "which", lambda name, *a, **k: None if name in ("nvidia-smi", "rustc", "nvcc") else real_which(name, *a, **k))
    monkeypatch.setattr(E, "_version", lambda dist: None)
    c = E.conditions(source_root=tmp_path)
    assert c["hw.family"] == "cpu" and c["hw.count"] == 0
    for k in ("hw.sku", "hw.vram_mb", "rt.cuda", "rt.driver", "rt.nvcc", "rt.rust", "rt.torch", "rt.triton"):
        assert c[k] is None, k


def test_conditions_source_from_git_tree_clean_dirty_and_lockfile(tmp_path):
    root = _repo(tmp_path)
    c = E.conditions(source_root=root)
    assert c["source.commit"] == _git(root, "rev-parse", "HEAD")
    assert c["source.tree"] == _git(root, "rev-parse", "HEAD^{tree}")
    assert c["source.dirty"] is False and c["source.dirty_digest"] is None
    assert c["source.lockfiles.uv.lock"] == hashlib.sha256(b"version = 1\n").hexdigest()
    assert "source.lockfiles.Cargo.lock" not in c
    (root / "tools/x/untracked.py").write_text("u = 4\n")
    d = E.conditions(source_root=root)
    assert d["source.dirty"] is True and len(d["source.dirty_digest"]) == 64
    (root / "tools/x/untracked.py").write_text("u = 5\n")
    assert E.conditions(source_root=root)["source.dirty_digest"] != d["source.dirty_digest"], "untracked content is part of the dirty digest"
    src = E.source_identity(d)
    assert src == {"commit": d["source.commit"], "tree": d["source.tree"], "dirty": True, "dirty_digest": d["source.dirty_digest"],
                   "lockfiles": {"uv.lock": c["source.lockfiles.uv.lock"]}}


def test_the_source_hint_is_the_whole_identity_without_git_and_unread_with_it(tmp_path):
    hint = {"commit": "c" * 40, "tree": "t" * 40, "dirty": False}
    c = E.conditions(source_root=tmp_path, source_hint=hint)          # a shipped git archive: no .git
    assert (c["source.commit"], c["source.tree"], c["source.dirty"]) == ("c" * 40, "t" * 40, False)
    root = _repo(tmp_path)
    stale = {"commit": "c" * 40, "tree": "t" * 40, "dirty": True, "dirty_digest": E.EMPTY_DIGEST}
    g = E.conditions(source_root=root, source_hint=stale)
    assert (g["source.commit"], g["source.dirty"], g["source.dirty_digest"]) == (_git(root, "rev-parse", "HEAD"), False, None), \
        "with git, no field of the hint fills in (r20261019-180708-c2a5: a clean checkout took the launcher's digest)"


def test_launch_names_the_commit_it_shipped_not_the_checkout_it_read(tmp_path, monkeypatch):
    """r20261019-180708-c2a5: the launcher read its checkout (HEAD 92867710, dirty), a commit landed, and `ship_source` shipped the
    new HEAD (0040ca11, clean); the run went out with the read's identity.  The identity sent is now the shipped commit's, and it
    replaces any the caller passed (the keeper refuses the flag twice)."""
    from research import remote

    shipped, tree = "b" * 40, "e" * 40
    monkeypatch.setattr(remote, "resolve_machine", lambda name, project: {"root": "/r", "python": "python3"})
    monkeypatch.setattr(remote, "pod_target", lambda m: {"host": "h", "port": 22, "user": "u", "pod_name": "n", "cost_per_hr": 0,
                                                         "rate_observed_utc": "t"})
    monkeypatch.setattr(remote, "wake", lambda m, ssh: None)
    monkeypatch.setattr(remote, "ship_tool", lambda *a: "/r/tool/x")
    monkeypatch.setattr(remote, "ship_source", lambda ssh, root, src: (shipped, f"/r/src/{shipped}", {"tree": tree, "transport": "stream"}))
    staged = []

    def stage(ssh, root, run_id, request, sends, private=None):
        staged.append(request)
        raise remote.Refused("staged")

    monkeypatch.setattr(remote, "stage_request", stage)
    stale = json.dumps({"commit": "a" * 40, "dirty": True, "dirty_digest": E.EMPTY_DIGEST})
    with pytest.raises(remote.Refused, match="staged"):
        remote.launch(run_id="r1", local_dir=tmp_path / "runs" / "r1", machine_name="n", project="verity", command=["true"],
                      harness_args=["--stage", "t", "--source-identity", stale], source=_repo(tmp_path), sends=[], cwd=None, env={},
                      exclusive=False)
    args = staged[0]["harness_args"]
    assert args.count("--source-identity") == 1 and remote.flag_value(args, "--tree-sha") == shipped
    assert json.loads(remote.flag_value(args, "--source-identity")) == {"commit": shipped, "tree": tree, "dirty": False, "dirty_digest": None}


def test_a_runner_given_a_hint_for_another_commit_than_it_ran_keeps_only_the_shipped_sha(tmp_path, monkeypatch):
    """A launcher older than `shipped_identity` still sends what it read before shipping: the machine's runner keeps the sha it was
    shipped as (--tree-sha) and drops the rest of a hint that names another commit."""
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.chdir(tmp_path)
    stale = json.dumps({"commit": "a" * 40, "tree": "t" * 40, "dirty": True, "dirty_digest": E.EMPTY_DIGEST})
    rc = research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--tree-sha", "b" * 40,
                        "--source-identity", stale, "--", "true"])
    assert rc == 0
    run_dir = next(p for p in (tmp_path / "runs").iterdir() if p.is_dir())
    c = json.loads((run_dir / "job.json").read_text())["store"]["conditions"]
    assert (c["source.commit"], c["source.tree"], c["source.dirty"], c["source.dirty_digest"]) == ("b" * 40, None, None, None)


def test_conditions_from_old_job_record_copies_only_recorded_facts():
    job = json.loads((RUN_FIXTURE / "job.json").read_text())
    c = E.conditions_from_job(job)
    assert set(E.KEYS) <= set(c)
    assert c["hw.sku"] == "NVIDIA H100 80GB HBM3" and c["hw.family"] == "h100" and c["hw.count"] == 1 and c["hw.vram_mb"] == 81559
    assert c["rt.driver"] == "580.126.09" and c["hw.hostname"] == "vy-g5" and c["source.commit"] == "0123456789abcdef0123456789abcdef01234567"
    assert c["source.tree"] is None and c["source.dirty"] is None and c["rt.torch"] is None
    v01 = json.loads((FIXTURES / "job_v0.1.json").read_text())
    assert E.conditions_from_job(v01)["hw.family"] == "cpu"


# ---- tools: closure manifest, params, registry -----------------------------------------------------------------------------------------------

def test_closure_manifest_from_git_tree_and_plain_tree(tmp_path):
    root = _repo(tmp_path)
    (root / "tools/x/untracked.py").write_text("u = 4\n")
    (root / "tools/x/junk.pyc").write_bytes(b"\x00")
    (root / "tools/x/__pycache__").mkdir()
    (root / "tools/x/__pycache__/a.cpython-311.pyc").write_bytes(b"\x00")
    t = Tool(name="t", version="1", closure=["tools/x/**"])
    m = t.closure_manifest(root)
    assert list(m) == ["tools/x/a.py", "tools/x/sub/b.py", "tools/x/untracked.py"], "tracked + untracked-not-ignored, sorted, files only"
    assert m["tools/x/a.py"] == hashlib.sha256(b"a = 1\n").hexdigest()
    plain = tmp_path / "archive"
    shutil.copytree(root, plain, ignore=shutil.ignore_patterns(".git"))
    assert Tool(name="t", version="1", closure=["tools/x/**"]).closure_manifest(plain) == m, "a git archive (no .git) yields the same manifest"
    assert Tool(name="t", version="1", closure=["tools/x/*.py"]).closure_manifest(root) == {"tools/x/a.py": m["tools/x/a.py"], "tools/x/untracked.py": m["tools/x/untracked.py"]}
    assert Tool(name="t", version="1").closure_manifest(root) == {}


def test_path_key_param_resolves_to_content_identity_never_pathname(tmp_path):
    golden = tmp_path / "somewhere" / "golden.json"
    golden.parent.mkdir()
    golden.write_text('{"records": 1}')
    t = Tool(name="probe", version="1", key_params=["golden", "seed"], param_artifacts=["golden"])
    params, key = t.resolve_params(["python", "probe.py", "--golden", str(golden), "--seed", "7", "--out", "/tmp/o"], tmp_path)
    assert params["golden"] == str(golden) and params["argv"][2] == "--golden", "the full argv and every parsed param are recorded"
    assert key == {"golden": "sha256:" + hashlib.sha256(b'{"records": 1}').hexdigest(), "seed": "7"}
    doc = D.derivation_doc(t, {}, key, {}, [])
    assert "somewhere" not in canon.dumps(doc).decode() and "golden.json" not in canon.dumps(doc).decode()
    other = tmp_path / "elsewhere.json"
    other.write_text('{"records": 1}')
    _, key2 = t.resolve_params(["python", "probe.py", "--golden", str(other), "--seed", "7"], tmp_path)
    assert key2 == key, "same content at another path is the same request"
    # fetched artifact: a .art-id marker beside the payload
    fetched = tmp_path / "fetched"
    fetched.mkdir()
    (fetched / ".art-id").write_text("art:" + "a" * 64 + "\n")
    (fetched / "tiles.npz").write_bytes(b"npz")
    assert content_identity(str(fetched), tmp_path) == "art:" + "a" * 64
    assert content_identity(str(fetched / "tiles.npz"), tmp_path) == "art:" + "a" * 64, "the single file of a fetched file artifact"
    (fetched / "other.bin").write_bytes(b"x")
    assert content_identity(str(fetched / "tiles.npz"), tmp_path) == "sha256:" + hashlib.sha256(b"npz").hexdigest()
    # a plain directory: the canonical tree listing (the §2 tree payload sha), marker excluded
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "b.txt").write_text("b")
    (plain / "a.txt").write_text("a")
    listing = [{"path": "a.txt", "sha256": hashlib.sha256(b"a").hexdigest(), "bytes": 1}, {"path": "b.txt", "sha256": hashlib.sha256(b"b").hexdigest(), "bytes": 1}]
    assert content_identity(str(plain)) == "sha256:" + canon.sha256(listing)
    assert content_identity("does/not/exist", tmp_path) is None
    assert content_identity(7, tmp_path) == 7


def test_generic_argv_parse():
    p = parse_argv(["python", "tools/x/run.py", "--mode", "fs", "--zk", "--n=3", "--neg", "-5", "pos", "--", "tail"], ["python", "tools/x/run.py"])
    assert p == {"mode": "fs", "zk": True, "n": "3", "neg": "-5", "_positional": ["pos", "tail"]}
    assert parse_argv(["true"]) == {}


def test_registry_loads_repo_declarations_and_file_fallback(tmp_path):
    tc = tools_registry.load("tc_probe", REPO)
    assert tc.name == "tc_probe" and tc.spec == "benchmarks.tc_probe.tool:TC_PROBE" and "hw.family" in tc.key_conditions
    with pytest.raises(LookupError):
        tools_registry.load("nope", REPO)
    src = tmp_path / "src"
    (src / "mytools").mkdir(parents=True)
    (src / "mytools/decl.py").write_text("from research.store.tool import Tool\nT = Tool(name='mine', version='2', key_params=['k'])\n")
    t = tools_registry.load("mytools.decl:T", src)
    assert t.name == "mine" and t.spec == "mytools.decl:T"


def test_former_modules_map_a_module_back_through_each_move():
    moves = [{"verity/ml/flock/verifier/tool.py": "tools/lean_agreement/tool.py", "old/gkr": "attic/gkr"},
             {"tools/lean_agreement/": "tools/agree/"}]
    assert tools_registry.former_modules("tools.agree.tool", moves) == ["tools.lean_agreement.tool", "verity.ml.flock.verifier.tool"]
    assert tools_registry.former_modules("attic.gkr.gpu.tool", moves) == ["old.gkr.gpu.tool"]
    assert tools_registry.former_modules("attic.gkrx.tool", moves) == []
    assert tools_registry.former_modules("tools.verity.check.tool", moves) == []
    shipped = json.loads(tools_registry.MOVES.read_text())["moves"]
    assert all(isinstance(m["paths"], dict) for m in shipped)
    assert tools_registry.former_modules("archive.gkr.gpu.tool") == ["backends.gkr.gpu.tool"]


def test_former_paths_and_resolve_find_a_file_an_older_tree_holds_under_its_old_name():
    moves = [{"tools/verity/check/": "tools/verity/check/"}, {"tools/verity/check/ci.toml": "ci/jobs.toml"}]
    assert tools_registry.former_paths("ci/jobs.toml", moves) == ["tools/verity/check/ci.toml", "tools/verity/check/ci.toml"]
    assert tools_registry.former_paths("tools/verity/check", moves) == ["tools/verity/check"]
    assert tools_registry.former_paths("tools/verity/checks/x", moves) == []
    held = {"tools/verity/check/ci.toml"}
    assert tools_registry.resolve("ci/jobs.toml", held.__contains__, moves) == "tools/verity/check/ci.toml"
    assert tools_registry.resolve("tools/verity/check/ci.toml", held.__contains__, moves) == "tools/verity/check/ci.toml"
    assert tools_registry.resolve("ci/other.toml", held.__contains__, moves) is None


def test_declared_jobs_reads_a_commit_from_before_the_move_that_renamed_ci_toml(tmp_path, monkeypatch):
    from research import merge

    moves = tmp_path / "moves.json"
    moves.write_text(json.dumps({"moves": [{"paths": {"old/check/": "tools/verity/check/"}}]}))
    monkeypatch.setattr(tools_registry, "MOVES", moves)
    repo = tmp_path / "repo"
    (repo / "old/check").mkdir(parents=True)
    (repo / "old/check/ci.toml").write_text('[[job]]\nname = "pytest"\n')
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "before the move")
    assert merge.declared_jobs(repo, "HEAD") == ["pytest"]
    moves.write_text(json.dumps({"moves": []}))
    assert "can't be read" in merge.declared_jobs(repo, "HEAD")


def test_an_unreadable_moves_file_hides_no_declaration_at_its_current_path(tmp_path, monkeypatch):
    """Red-team 1171: `locate` read `moves.json` before looking at the current path, so a corrupt one made `merge.declaration`
    none and `merge.required_steps` require nothing (a train sent no lean-agreement inputs)."""
    from research import merge

    moves = tmp_path / "moves.json"
    moves.write_text("{not json")
    monkeypatch.setattr(tools_registry, "MOVES", moves)
    monkeypatch.setitem(tools_registry.REGISTRY, "agree", "tools.agree.tool:AGREE")
    repo = tmp_path / "repo"
    (repo / "tools/agree").mkdir(parents=True)
    (repo / "tools/agree/tool.py").write_text("from research.store.tool import Tool\nAGREE = Tool(name='agree', version='1', "
                                              "command=['true'], merge_requires={'agreement': ['verity/ml/flock/']})\n")
    (repo / "verity/ml/flock").mkdir(parents=True)
    (repo / "verity/ml/flock/a.py").write_text("x = 1\n")
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "base")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "change")

    assert tools_registry.locate("agree", repo) == ("tools.agree.tool", "AGREE")
    assert merge.declaration(repo, "agree").name == "agree"
    assert merge.required_steps(repo, "HEAD~1", "HEAD", "agree") == {"agreement": ["verity/ml/flock/"]}
    (repo / "tools/agree/tool.py").unlink()
    with pytest.raises(json.JSONDecodeError):
        tools_registry.locate("agree", repo)
    assert merge.declaration(repo, "agree") is None


def test_a_tree_from_before_a_move_still_declares_its_tools(tmp_path, monkeypatch):
    """`research run --tool`, `research merge`'s reads of the target's declaration, `after_land` and a run's preflight read
    a tool's declaration from the tree they're given: one from before a layout move holds it under its old path."""
    from research import merge, remote

    moves = tmp_path / "moves.json"
    moves.write_text(json.dumps({"moves": [{"paths": {"verity/ml/flock/verifier/tool.py": "tools/agree/tool.py"}}]}))
    monkeypatch.setattr(tools_registry, "MOVES", moves)
    monkeypatch.setitem(tools_registry.REGISTRY, "agree", "tools.agree.tool:AGREE")
    decl = ("from research.store.tool import Tool\nAGREE = Tool(name='agree', version='1', command=['true'], "
            "preflight=['echo', '{tag}'], after_land=['echo', 'landed'])\n")
    for tree, rel in (("old", "verity/ml/flock/verifier/tool.py"), ("new", "tools/agree/tool.py")):
        (tmp_path / tree / rel).parent.mkdir(parents=True)
        (tmp_path / tree / rel).write_text(decl.replace("{tag}", tree))
    (tmp_path / "none").mkdir()

    old, new, none = tmp_path / "old", tmp_path / "new", tmp_path / "none"
    assert tools_registry.locate("agree", new) == ("tools.agree.tool", "AGREE")
    assert tools_registry.locate("agree", old) == ("verity.ml.flock.verifier.tool", "AGREE")
    assert tools_registry.locate("agree", none) is None
    assert tools_registry.load("agree", old).name == "agree" and tools_registry.load("agree", old).spec == "tools.agree.tool:AGREE"
    assert remote.tool_preflight(["--tool", "agree"], old) == ["echo", "old"]
    assert remote.tool_preflight(["--tool", "agree"], new) == ["echo", "new"]
    assert merge.declaration(old, "agree").name == "agree"
    assert merge.after_land(old, "agree", "c0ffee", log=lambda *a: None, spawn=lambda argv, **kw: None) == ["echo", "landed", "c0ffee"]

    with pytest.raises(ImportError, match="declares no such tool"):
        tools_registry.from_tree("agree", none)
    assert merge.declaration(none, "agree") is None
    assert remote.tool_preflight(["--tool", "agree"], none) is None
    with pytest.raises(remote.PreflightFailed, match="does not declare"):
        remote.tool_preflight([], none, asked="agree")


def test_tc_probe_declaration_parses_real_flags_and_optional_instruction():
    tc = tools_registry.load("tc_probe", REPO)
    argv = ["python", "benchmarks/tc_probe/tc_probe.py", "--arch", "sm_89", "--out", "/run/dir", "--sweep", "--n-random", "1000", "--families", "random,subnormal"]
    params, key = tc.resolve_params(argv, REPO)
    assert params["arch"] == "sm_89" and params["out"] == "/run/dir" and "out" not in key and "procs" not in key
    assert key["n_random"] == 1000 and key["sweep"] is True and key["families"] == "random,subnormal" and "instruction" not in key
    _, key2 = tc.resolve_params(argv + ["--instruction", "mma.sync.m16n8k16"], REPO)
    assert key2["instruction"] == "mma.sync.m16n8k16"
    assert D.derivation_id(tc, {}, key, {"hw.family": "rtx4090"}, tc.key_conditions) != D.derivation_id(tc, {}, key2, {"hw.family": "rtx4090"}, tc.key_conditions)
    _, key3 = tc.resolve_params(argv + ["--replay-golden", "verity/ml/circuits/ops/tests/fixtures/golden/ada_bf16_m16n8k16.json"], REPO)
    assert key3["replay_golden"].startswith("sha256:")


# ---- derivation, reuse key, scratch ----------------------------------------------------------------------------------------------------------

def test_derivation_id_stable_under_param_order_and_sensitive_to_content():
    t = {"name": "bench_vu", "version": "1"}
    a = D.derivation_id(t, {"instances": "art:" + "1" * 64}, {"mode": "fs", "batch": 4096}, {"hw.family": "h100", "hw.sku": "x"}, ["hw.family"])
    b = D.derivation_id(t, {"instances": "art:" + "1" * 64}, {"batch": 4096, "mode": "fs"}, {"hw.sku": "y", "hw.family": "h100"}, ["hw.family"])
    assert a == b and a.startswith("drv:") and len(a) == 4 + 64
    assert D.derivation_id(t, {}, {"batch": 4096, "mode": "fs"}, {"hw.family": "h100"}, ["hw.family"]) != a, "inputs matter"
    assert D.derivation_id(t, {"instances": "art:" + "1" * 64}, {"batch": 4096, "mode": "fs"}, {"hw.family": "a100"}, ["hw.family"]) != a, "key conditions matter"
    assert D.derivation_id({"name": "bench_vu", "version": "2"}, {"instances": "art:" + "1" * 64}, {"batch": 4096, "mode": "fs"}, {"hw.family": "h100"}, ["hw.family"]) != a
    assert D.derivation_id(Tool(name="bench_vu", version="1"), {"instances": "art:" + "1" * 64}, {"batch": 4096, "mode": "fs"}, {"hw.family": "h100"}, ["hw.family"]) == a
    doc = D.derivation_doc(t, {}, {"k": 1}, {"hw.family": "h100"}, ["hw.family"])
    assert set(doc) == {"tool", "inputs", "params", "conditions"} and "source" not in json.dumps(doc)


def test_reuse_key_none_when_dirty_or_unknown_and_stable_when_clean():
    drv = "drv:" + "a" * 64
    clean = {"source.tree": "t" * 40, "source.dirty": False, "source.lockfiles.uv.lock": "l" * 64, "source.commit": "c" * 40}
    k = D.reuse_key(drv, clean)
    assert k and len(k) == 64
    assert k == D.reuse_key(drv, {"tree": "t" * 40, "dirty": False, "lockfiles": {"uv.lock": "l" * 64}}), "flat or nested source: same key"
    assert k == canon.sha256({"derivation": drv, "source": {"tree": "t" * 40, "lockfiles": {"uv.lock": "l" * 64}}})
    assert D.reuse_key(drv, {**clean, "source.dirty": True}) is None
    assert D.reuse_key(drv, {**clean, "source.dirty": None}) is None, "unknown cleanliness is not clean"
    assert D.reuse_key(drv, {**clean, "source.tree": None}) is None
    assert D.reuse_key(drv, {**clean, "source.lockfiles.uv.lock": "m" * 64}) != k, "lockfiles are part of the source identity"
    assert D.reuse_key(drv, {**clean, "source.commit": "d" * 40}) == k, "the commit is not: same tree, same key"


def test_scratch_dir_layout_and_env(tmp_path):
    p = D.scratch_dir(tmp_path, "k" * 64, "r1", "triton")
    assert p == tmp_path / "scratch" / ("k" * 64) / "triton" and p.is_dir()
    q = D.scratch_dir(tmp_path, None, "r1", "nvcc")
    assert q == tmp_path / "scratch" / "run-r1" / "nvcc" and q.is_dir()
    assert D.scratch_env("triton", p) == {"TRITON_CACHE_DIR": str(p)}
    assert D.scratch_env("nvcc-out", q) == {"RESEARCH_SCRATCH_NVCC_OUT": str(q)}
    with pytest.raises(ValueError):
        D.scratch_dir(tmp_path, None, "r1", "../x")
    assert "reuse_key ON PURPOSE" in D.scratch_dir.__doc__


# ---- attempt.build / publish -----------------------------------------------------------------------------------------------------------------

def _run_copy(tmp_path: Path) -> Path:
    d = tmp_path / "runs" / RUN_ID
    shutil.copytree(RUN_FIXTURE, d)
    (d / "state.json").rename(d / "status.json")      # the fixture cannot be tracked as status.json (tests/test_repository.py's report-genre gate)
    return d


def test_attempt_build_from_fixture_run_dir(tmp_path):
    d = _run_copy(tmp_path)
    m = A.build(d)
    assert m["schema"] == A.ATTEMPT_SCHEMA and m["id"] == RUN_ID
    assert m["tool"] == {"name": "bench_vu", "version": "1"}
    assert m["derivation"] == "drv:" + "1" * 64 and m["reuse_key"] == "2" * 64, "what the runner computed before launch is what is published"
    assert m["key_params"][:3] == ["cmd", "mode", "zk"] and m["key_conditions"] == ["hw.family"]
    assert m["params"]["mode"] == "fiat-shamir" and m["params"]["argv"][3] == "bench-vu" and m["argv"][3] == "bench-vu"
    assert m["key_params_resolved" if "key_params_resolved" in m else "params"]  # params carry the resolved values via job.store
    assert m["conditions"]["hw.family"] == "h100" and set(E.KEYS) <= set(m["conditions"])
    assert m["source"] == {"commit": "0123456789abcdef0123456789abcdef01234567", "tree": "89abcdef0123456789abcdef0123456789abcdef", "dirty": False,
                           "dirty_digest": None, "lockfiles": {"uv.lock": "4" * 64}}
    assert m["closure"] == {"backends/direct/ligero/run.py": "3" * 64}
    ex = m["execution"]
    assert ex["start_utc"] == "2026-09-22T12:00:01.000Z" and ex["end_utc"] == "2026-09-22T12:00:13.300Z" and ex["wall_s"] == 12.3
    assert ex["exclusive"] is True and ex["host"] == "vy-g5" and ex["project"] == "verity" and ex["campaign"] == "r21-fixture"
    assert ex["run_dir"] == "/workspace/research/runs/" + RUN_ID, "where it executed, not where the copy sits"
    assert m["cost"] == {"provider": "runpod", "pod_id": "podfixture", "rate_usd_per_hour": 2.69, "usd": round(2.69 * 12.3 / 3600, 4), "shared": False}
    assert m["status"]["state"] == "done" and m["status"]["rc"] == 0 and m["status"]["class"] == "SUCCESS" and m["status"]["validation"] == "passed"
    assert m["outputs"] == {"result": None, "run_files": None} and m["telemetry"] == {"events": None, "resources": None}
    canon.dumps(m)                                                                    # the manifest is canonical-JSON clean


def test_a_result_too_big_for_a_manifest_publishes_its_largest_fields_as_the_payload(tmp_path):
    """INF-43: r20261010-162750-3ea6's 593 KB result.json made a 358 KB meta, over META_MAX_BYTES, so its attempt never published."""
    d = _run_copy(tmp_path)
    doc = json.loads((d / "result.json").read_text())
    doc["findings"] = ["x" * 1000] * 400
    (d / "result.json").write_text(json.dumps(doc))
    st = FakeStore(tmp_path / "store")
    out = A.publish(d, store=st)
    assert out["published"] is True, out
    res = st.artifacts[out["outputs"]["result"]]
    assert res["meta"]["payload_fields"] == ["findings"] and "findings" not in res["meta"]
    assert res["meta"]["validation"]["status"] == "passed", "the small fields readers filter on stay in the meta"
    assert len(canon.dumps(res["meta"])) <= M.META_MAX_BYTES and res["payload"] is not None


def test_publish_against_fake_store_is_complete_and_idempotent(tmp_path):
    d = _run_copy(tmp_path)
    st = FakeStore(tmp_path / "store")
    out = A.publish(d, store=st)
    assert out["published"] is True and out["id"] == RUN_ID and out["n_outputs"] == 3, out
    res = st.artifacts[out["outputs"]["result"]]
    assert res["kind"] == "bench-result/v1"
    assert "artifacts" not in res["meta"] and "measurement_files" not in res["meta"]
    assert res["meta"]["measurements"][0]["name"] == "prove_wall" and res["meta"]["validation"]["status"] == "passed"
    assert res["refs"] == {"run_files": out["outputs"]["run_files"]} and res["payload"] is None
    rf = st.artifacts[out["outputs"]["run_files"]]
    assert rf["kind"] == "run-files/v1" and rf["payload"]["type"] == "tree"
    assert [f["path"] for f in rf["payload"]["files"]] == ["out/proof.bin", "out/verify.jsonl"], "one tree, relative paths preserved, deduplicated"
    proof_sha = hashlib.sha256((RUN_FIXTURE / "out/proof.bin").read_bytes()).hexdigest()
    assert rf["payload"]["files"][0]["sha256"] == proof_sha and st.blobs[proof_sha] == (RUN_FIXTURE / "out/proof.bin").read_bytes()
    assert rf["meta"]["files"] == 2 and rf["meta"]["missing"] == []
    ev = st.artifacts[out["telemetry"]["events"]]
    assert ev["kind"] == "telemetry/v1" and ev["payload"]["type"] == "file" and ev["payload"]["name"] == "events.jsonl" and ev["meta"]["stream"] == "events"
    assert out["telemetry"]["resources"] is None
    att = st.get_attempt(RUN_ID)
    assert att["outputs"] == out["outputs"] and att["telemetry"] == out["telemetry"] and att["derivation"] == "drv:" + "1" * 64
    assert not (d / "store_publish_error.json").exists()
    n_art, n_att = len(st.artifacts), len(st.attempts)
    again = A.publish(d, store=st)
    assert again["published"] and again.get("already") is True and again["outputs"] == out["outputs"] and again["n_outputs"] == 3
    assert (len(st.artifacts), len(st.attempts)) == (n_art, n_att), "second publish is a no-op"


def test_publish_explicit_conditions_and_tool_override_the_record(tmp_path):
    d = _run_copy(tmp_path)
    st = FakeStore(tmp_path / "store")
    conds = {**json.loads((RUN_FIXTURE / "job.json").read_text())["store"]["conditions"], "exec.load_end": 0.7}
    tool = Tool(name="bench_vu", version="1", key_params=["mode"], key_conditions=["hw.family"])
    out = A.publish(d, store=st, tool=tool, conditions=conds)
    att = st.get_attempt(RUN_ID)
    assert out["published"] and att["execution"]["load_end"] == 0.7 and att["key_params"] == ["mode"]


def test_publish_never_raises_and_records_the_error(tmp_path):
    d = _run_copy(tmp_path)

    class Broken(FakeStore):
        def put_attempt(self, doc):
            raise ValueError("attempt r exists with different content")

    out = A.publish(d, store=Broken(tmp_path / "store"))
    assert out["published"] is False and "different content" in out["error"]
    err = json.loads((d / "store_publish_error.json").read_text())
    assert err["schema"] == A.PUBLISH_ERROR_SCHEMA and "ValueError" in err["error"] and err["traceback"]


def test_publish_refuses_a_live_run_unless_forced(tmp_path):
    d = _run_copy(tmp_path)
    st = FakeStore(tmp_path / "store")
    status = json.loads((d / "status.json").read_text())
    status["state"] = "running"
    (d / "status.json").write_text(json.dumps(status))
    out = A.publish(d, store=st)
    assert out["published"] is False and "running" in out["error"] and st.get_attempt(RUN_ID) is None
    assert A.publish(d, store=st, force=True)["published"] is True


def test_publish_is_local_when_the_remote_is_configured_but_unusable_here(tmp_path, monkeypatch):
    """A `[remote]` table whose credential env vars are unset is a broken vault (README §7): the store degrades to local-only.  A
    LOCAL publish must still land: the attempt and its artifacts are written, nothing is pushed, the reason is reported."""
    from research.store.local import LocalStore, StoreError

    monkeypatch.delenv("SB_TEST_R2_KEY", raising=False)
    cfg = tmp_path / "store.toml"
    cfg.write_text('[remote]\ntype = "s3"\nbucket = "b"\nendpoint = "https://example.invalid"\n'
                   'access_key_id_env = "SB_TEST_R2_KEY"\nsecret_access_key_env = "SB_TEST_R2_KEY"\n')
    st = LocalStore(tmp_path / "store", config_path=cfg)
    assert st.get_attempt(RUN_ID) is None and "SB_TEST_R2_KEY" in (st.remote_error or "")   # broken vault: reads degrade, the reason is kept
    with pytest.raises(StoreError, match="SB_TEST_R2_KEY"):
        st.push_attempt(RUN_ID)                                    # anything that needs the vault still says why it cannot
    d = _run_copy(tmp_path)
    out = A.publish(d, store=st)
    assert out["published"] is True and out.get("already") is None and out["n_outputs"] == 3, out
    assert out["preserved"] is False and "SB_TEST_R2_KEY" in out["preserve_skipped"]
    assert st.attempt_path(RUN_ID).exists() and st.has_local(out["outputs"]["run_files"])
    assert not (d / "store_publish_error.json").exists()
    again = A.publish(d, store=st)
    assert again["published"] and again.get("already") is True, "idempotent through the same broken-vault path"
    assert cli_prov.publish_counts([out, again, {"published": False, "error": "RuntimeError: run r20260922-000000-0000 is running: the store only sees completed or failed attempts (--force to override)"},
                                    {"published": False, "error": "FileNotFoundError: /x/runs/r20260922-000000-0001: no readable job.json"}]) == {
        "published": 1, "already": 1, "refused": 1, "errors": 1, "error_classes": {"FileNotFoundError: …/runs/<run>: no readable job.json": 1}}


def test_publish_old_v01_record_without_store_section(tmp_path):
    d = tmp_path / "runs" / "oom_own"
    d.mkdir(parents=True)
    shutil.copy(FIXTURES / "job_v0.1.json", d / "job.json")
    (d / "status.json").write_text(json.dumps({"schema": "telemetry/status/v0.1", "state": "failed", "rc": 137, "wall_s": 3.2, "transitions": []}))
    st = FakeStore(tmp_path / "store")
    out = A.publish(d, store=st)
    assert out["published"], out
    att = st.get_attempt("oom_own")
    assert att["tool"] == {"name": "python3", "version": None} and att["params"] == {"argv": ["python3", "hog.py", "--step-mib", "256"]}
    assert att["key_params"] == ["argv"] and att["inputs"] == {} and att["derivation"].startswith("drv:")
    assert att["reuse_key"] is None, "no declared Tool: no reuse claim"
    assert att["conditions"]["hw.family"] == "cpu" and att["conditions"]["hw.hostname"] == "ee56316268c7" and att["conditions"]["rt.torch"] is None
    assert att["cost"]["usd"] is None and att["status"]["rc"] == 137 and att["outputs"] == {"result": None, "run_files": None}
    assert att["derivation"] == D.derivation_id({"name": "python3", "version": None}, {}, {"argv": ["python3", "hog.py", "--step-mib", "256"]}, {}, []), \
        "no store section: the whole argv is the request"


# ---- research run integration ------------------------------------------------------------------------------------------------------------------

def _store_core_available() -> bool:
    try:
        from research.store.local import LocalStore  # noqa: F401
    except ImportError:
        return False
    return True


def test_run_records_store_section_scratch_and_campaign(tmp_path, monkeypatch, capsys):
    store = tmp_path / "store"
    monkeypatch.setenv("RESEARCH_STORE", str(store))
    monkeypatch.chdir(_repo(tmp_path))
    rc = research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--scratch", "triton", "--scratch", "nvcc",
                        "--campaign", "camp-1", "--",
                        PY, "-c", "import os; print(os.environ['TRITON_CACHE_DIR']); print(os.environ['RESEARCH_SCRATCH_NVCC']); print(os.environ['RESEARCH_CAMPAIGN'])"])
    assert rc == 0
    run_dir = next(p for p in (tmp_path / "runs").iterdir() if p.is_dir())
    job = json.loads((run_dir / "job.json").read_text())
    s = job["store"]
    assert s["tool"] == {"name": Path(PY).name, "version": None, "spec": None} and s["key_params"] == ["argv"] and s["key_conditions"] == []
    assert s["derivation"].startswith("drv:") and s["params"]["argv"][0] == PY and s["inputs"] == {} and s["closure"] == {}
    assert s["conditions"]["exec.campaign"] == "camp-1" and s["conditions"]["exec.project"] == "t" and s["conditions"]["source.dirty"] is False
    assert s["reuse_key"] is None, "clean temp repo, but no declared Tool: no reuse key"
    lines = (run_dir / "stdout.log").read_text().splitlines()
    assert lines[0] == str(store / "scratch" / f"run-{run_dir.name}" / "triton") == s["scratch"]["triton"]
    assert lines[1] == str(store / "scratch" / f"run-{run_dir.name}" / "nvcc") and lines[2] == "camp-1"
    err = capsys.readouterr().err
    assert "research: attempt published" in err or "research: attempt not published" in err
    if not _store_core_available():
        assert (run_dir / "store_publish_error.json").exists(), "without research.store.local the failure is recorded, and the run's own exit status stands"


def test_run_with_tool_declaration_resolves_before_launch(tmp_path, monkeypatch):
    src = _repo(tmp_path)
    (src / "mytools/decl_run.py").parent.mkdir()
    (src / "mytools/decl_run.py").write_text(textwrap.dedent("""
        from research.store.tool import Tool
        T = Tool(name="mine", version="3", command=["python", "-c"], closure=["tools/x/**"], key_params=["golden", "n"],
                 key_conditions=["hw.family"], param_artifacts=["golden"], conditions=lambda env: {"probe": "ok"})
    """))
    (src / "golden.json").write_text("{}")
    (src / "fetched").mkdir()
    (src / "fetched/.art-id").write_text("art:" + "b" * 64)
    (src / "fetched/x.bin").write_bytes(b"x")
    monkeypatch.chdir(src)
    rc = research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--tool", "mytools.decl_run:T", "--input", "gold=golden.json",
                        "--input", "inst=fetched", "--",
                        PY, "-c", "pass", "--golden", "golden.json", "--n", "3"])
    assert rc == 0
    run_dir = next(p for p in (tmp_path / "runs").iterdir() if p.is_dir())
    s = json.loads((run_dir / "job.json").read_text())["store"]
    assert s["tool"] == {"name": "mine", "version": "3", "spec": "mytools.decl_run:T"}
    assert s["key_params_resolved"] == {"golden": "sha256:" + hashlib.sha256(b"{}").hexdigest(), "n": "3", "input.gold": "sha256:" + hashlib.sha256(b"{}").hexdigest()}
    assert s["inputs"] == {"inst": "art:" + "b" * 64}, "a fetched artifact is an input; a plain --input path enters the derivation as a key param, by content"
    assert set(s["closure"]) == {"tools/x/a.py", "tools/x/sub/b.py"} and s["conditions"]["tool.mine.probe"] == "ok"
    tool = tools_registry.load("mytools.decl_run:T", src)
    assert s["derivation"] == D.derivation_id(tool, s["inputs"], s["key_params_resolved"], s["conditions"], ["hw.family"])
    assert s["reuse_key"] == D.reuse_key(s["derivation"], s["conditions"])
    assert research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--tool", "nope", "--", "true"]) == 2, "an unknown tool is refused before anything runs"


def test_declared_key_params_are_recorded_even_when_their_name_looks_secret(tmp_path, monkeypatch):
    """`--auth included-hash` is a scientific choice the Tool declares as a key param; it was stored as `<redacted>`, so hashed and
    bare runs were indistinguishable in the catalog.  Undeclared secret-looking flags and secret-shaped values stay redacted."""
    src = _repo(tmp_path)
    (src / "mytools/decl_auth.py").parent.mkdir()
    (src / "mytools/decl_auth.py").write_text(textwrap.dedent("""
        from research.store.tool import Tool
        T = Tool(name="mine", version="1", command=["python", "-c"], key_params=["auth", "tile"])
    """))
    monkeypatch.chdir(src)
    token = "hf_" + "a" * 24
    rc = research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--tool", "mytools.decl_auth:T", "--",
                        PY, "-c", "pass", "--auth", "included-hash", "--tile=64", "--auth-token", "s3cret", "--api-key=abc"])
    assert rc == 0
    s = json.loads((next(p for p in (tmp_path / "runs").iterdir() if p.is_dir()) / "job.json").read_text())["store"]
    assert s["params"]["auth"] == "included-hash" and s["key_params_resolved"]["auth"] == "included-hash"
    assert s["params"]["auth_token"] == "<redacted>" and s["params"]["api_key"] == "<redacted>"
    assert s["params"]["argv"][-6:] == ["--auth", "included-hash", "--tile=64", "--auth-token", "<redacted>", "--api-key=<redacted>"]

    from research.cli import unredact_key_params
    from research.telemetry.job import redact_argv
    argv = ["x", "--auth", token, "--auth=included"]
    assert unredact_key_params(argv, redact_argv(argv)[0], {"auth"}) == ["x", "--auth", "<redacted>", "--auth=included"], \
        "a secret-shaped value stays redacted even under a key-param flag"


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="shipped tree: no .git to archive")
def test_machine_side_runner_loads_tool_from_shipped_source(tmp_path):
    """The machine gets a `git archive` (no .git) as cwd and the research source on PYTHONPATH: `--tool tc_probe` must resolve
    there, the closure must come from the archive, and the source identity from the control side's hint."""
    from research import remote
    from research import runs as R

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    tar = subprocess.run(["git", "archive", "--format=tar", "HEAD", "benchmarks/tc_probe", "verity/ml/circuits/silicon",
                          "verity/ml/circuits/targets/target.py", ".gitignore", "uv.lock", "pyproject.toml"],
                         cwd=REPO, capture_output=True, check=True).stdout
    subprocess.run(["tar", "-xf", "-", "-C", str(src_dir)], input=tar, check=True)
    assert (src_dir / "benchmarks/tc_probe/tool.py").exists() and not (src_dir / ".git").exists()
    hint = {"commit": "c" * 40, "tree": "t" * 40, "dirty": False}
    root = tmp_path / "root"
    (root / "requests" / "r1").mkdir(parents=True)
    req = {"schema": remote.REQUEST_SCHEMA, "run_id": "r1", "runs_root": str(root / "runs"), "tool": str(Path(remote.__file__).resolve().parents[1]),
           "python": PY, "cwd": str(src_dir), "env": {"RESEARCH_STORE": ""},   # a machine has no store configured
           "harness_args": ["--stage", "t", "--no-sampler", "--tool", "tc_probe", "--campaign", "camp", "--scratch", "nvcc", "--source-identity", json.dumps(hint)],
           "command": [PY, "-c", "import os; print(os.environ['RESEARCH_SCRATCH_NVCC'])", "--arch", "sm_89", "--out", "o", "--instruction", "mma.x"],
           "exclusive": False, "cost": {}, "remote": {"machine": "m", "pod_id": "p1"}}
    (root / "requests/r1/request.json").write_text(json.dumps(req))
    assert remote.launch_request(root / "requests/r1/request.json")["started"]
    wait_runner(root / "runs/r1")
    s = R.summarize(root / "runs/r1")
    assert s["state"] == "done" and s["rc"] == 0, (root / "runs/r1/launcher.log").read_text()
    store = json.loads((root / "runs/r1/job.json").read_text())["store"]
    assert "error" not in store, store.get("error")
    assert store["tool"] == {"name": "tc_probe", "version": "1", "spec": "benchmarks.tc_probe.tool:TC_PROBE"}
    assert "benchmarks/tc_probe/tc_probe.py" in store["closure"] and "verity/ml/circuits/silicon/silicon.py" in store["closure"]
    assert store["key_params_resolved"]["arch"] == "sm_89" and store["key_params_resolved"]["instruction"] == "mma.x" and "out" not in store["key_params_resolved"]
    c = store["conditions"]
    assert (c["source.commit"], c["source.tree"], c["source.dirty"]) == ("c" * 40, "t" * 40, False), "the control side's hint fills what git cannot say in an archive"
    assert c["source.lockfiles.uv.lock"] == hashlib.sha256((src_dir / "uv.lock").read_bytes()).hexdigest(), "lockfiles are hashed from the archive itself"
    assert c["exec.campaign"] == "camp" and len(store["reuse_key"]) == 64
    assert (root / "runs/r1/stdout.log").read_text().strip() == str(root / "store" / "scratch" / store["reuse_key"] / "nvcc")
    assert store["scratch"]["nvcc"] == str(root / "store" / "scratch" / store["reuse_key"] / "nvcc"), "the machine's store is <root>/store, beside its runs"
    assert store["store_root"] == str(root / "store")


@pytest.mark.skipif(not _store_core_available(), reason="research.store.local does not import: the local store cannot be opened")
def test_run_publishes_attempt_to_local_store(tmp_path, monkeypatch):
    from research.store.local import LocalStore

    store = tmp_path / "store"
    monkeypatch.setenv("RESEARCH_STORE", str(store))
    monkeypatch.chdir(_repo(tmp_path))
    rc = research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--", "true"])
    assert rc == 0
    run_dir = next(p for p in (tmp_path / "runs").iterdir() if p.is_dir())
    assert not (run_dir / "store_publish_error.json").exists()
    st = LocalStore(store)
    att = st.get_attempt(run_dir.name)
    assert att and att["schema"] == A.ATTEMPT_SCHEMA and att["status"]["rc"] == 0 and att["tool"]["name"] == "true"
    assert att["reuse_key"] is None and st.index.attempts_by_derivation(att["derivation"])[0]["id"] == run_dir.name


# ---- cli_prov: lookup, pull, attempt publish -------------------------------------------------------------------------------------------------

def _decl(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "mytools").mkdir(parents=True)
    (src / "mytools/decl_lk.py").write_text("from research.store.tool import Tool\n"
                                        "T = Tool(name='mine', version='1', key_params=['mode', 'n'], key_conditions=['hw.family'])\n")
    return src


def test_lookup_finds_attempts_by_reuse_key_and_derivation(tmp_path, monkeypatch, capsys):
    src = _decl(tmp_path)
    st = FakeStore(tmp_path / "store")
    monkeypatch.setattr(A, "open_store", lambda store=None, root=None: st)
    tool = tools_registry.load("mytools.decl_lk:T", src)
    conds = {"hw.family": "h100"}
    drv = D.derivation_id(tool, {}, {"mode": "fs", "n": 3}, conds, ["hw.family"])
    tree = "t" * 40
    rk = D.reuse_key(drv, {"tree": tree, "dirty": False, "lockfiles": {}})
    for rid, key in (("r1", rk), ("r2", None), ("r3", "x" * 64)):
        st.put_attempt({"id": rid, "derivation": drv, "reuse_key": key, "status": {"state": "done", "rc": 0}, "execution": {"host": "h"}, "outputs": {"result": "art:" + "9" * 64}})
    st.put_attempt({"id": "r4", "derivation": "drv:" + "f" * 64, "reuse_key": rk, "status": {}, "execution": {}, "outputs": {}})
    rc = cli_prov.main(["lookup", "--tool", "mytools.decl_lk:T", "--source-root", str(src), "--param", "mode=fs", "--param", "n=3", "--condition", "hw.family=h100",
                        "--source-tree", tree, "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["derivation"] == drv and out["reuse_key"] == rk
    assert sorted(a["id"] for a in out["attempts"]) == ["r1", "r4"], "exact reuse_key match (r4 shares the key by construction of the test double)"
    rc = cli_prov.main(["lookup", "--tool", "mytools.decl_lk:T", "--source-root", str(src), "--param", "mode=fs", "--param", "n=3", "--condition", "hw.family=h100",
                        "--any-source", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and sorted(a["id"] for a in out["attempts"]) == ["r1", "r2", "r3"]
    rc = cli_prov.main(["lookup", "--tool", "mytools.decl_lk:T", "--source-root", str(src), "--param", "mode=fs", "--param", "n=3", "--condition", "hw.family=h100"])
    assert rc == 1 and "no reuse_key" in capsys.readouterr().err, "a non-git source root cannot claim a clean tree"
    rc = cli_prov.main(["lookup", "--tool", "mytools.decl_lk:T", "--source-root", str(src), "--param", "mode=fs", "--param", "n=4", "--condition", "hw.family=h100", "--any-source"])
    assert rc == 0 and "no attempts" in capsys.readouterr().out


def test_lookup_source_tree_uses_the_lockfile_of_that_tree(tmp_path, monkeypatch, capsys):
    root = _repo(tmp_path)
    (root / "mytools").mkdir()
    (root / "mytools/decl_tree.py").write_text("from research.store.tool import Tool\nT = Tool(name='mine', version='1', key_params=['n'])\n")
    st = FakeStore(tmp_path / "store")
    monkeypatch.setattr(A, "open_store", lambda store=None, root=None: st)
    conds = E.conditions(source_root=root)
    drv = D.derivation_id({"name": "mine", "version": "1"}, {}, {"n": 1}, {}, [])
    assert conds["source.dirty"] is True, "decl.py is untracked here"
    st.put_attempt({"id": "r1", "derivation": drv, "reuse_key": D.reuse_key(drv, {**conds, "source.dirty": False}), "status": {}, "execution": {}, "outputs": {}})
    rc = cli_prov.main(["lookup", "--tool", "mytools.decl_tree:T", "--source-root", str(root), "--param", "n=1", "--source-tree", conds["source.tree"], "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and [a["id"] for a in out["attempts"]] == ["r1"], "the lockfile sha is read from the named tree, so the key matches what a run there recorded"


def test_pull_copies_attempt_and_verified_outputs_over_fake_ssh(tmp_path, monkeypatch, capsys):
    machine = FakeStore(tmp_path / "machine" / "store")
    d = _run_copy(tmp_path)
    pub = A.publish(d, store=machine)
    assert pub["published"]
    local = FakeStore(tmp_path / "local")
    monkeypatch.setattr(A, "open_store", lambda store=None, root=None: local)

    def fake_ssh_run(ssh, remote_argv, *, stdin=None, timeout=120):
        if remote_argv[0] == "cat":
            p = Path(remote_argv[1])
            return subprocess.CompletedProcess(remote_argv, 0 if p.is_file() else 1, p.read_bytes() if p.is_file() else b"", b"" if p.is_file() else b"no such file")
        assert remote_argv[:4] == ["tar", "-cf", "-", "-C"] and remote_argv[-2:] == ["-T", "-"] and stdin, "the blob list travels on stdin, never in the argv"
        return subprocess.run(remote_argv, input=stdin, capture_output=True)

    monkeypatch.setattr(cli_prov.remote, "ssh_run", fake_ssh_run)
    monkeypatch.setattr(cli_prov.remote, "resolve_machine", lambda name, project, machines=None: {"name": name, "root": str(tmp_path / "machine"), "python": "python3"})
    monkeypatch.setattr(cli_prov.remote, "pod_target", lambda m: {"host": "h", "port": 22})
    monkeypatch.setattr(cli_prov.remote, "ssh_argv", lambda host, port, **k: ["ssh", host])
    rc = cli_prov.main(["pull", "--from", "m", RUN_ID, "--project", "verity", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["attempt"] == RUN_ID
    assert {p["name"]: p["id"] for p in out["pulled"]} == {"result": pub["outputs"]["result"], "run_files": pub["outputs"]["run_files"], "events": pub["telemetry"]["events"]}
    assert set(local.artifacts) == set(machine.artifacts) and local.blobs == machine.blobs, "manifests re-derived to the same ids, blobs byte-identical"
    assert local.get_attempt(RUN_ID) == machine.get_attempt(RUN_ID)
    assert cli_prov.main(["pull", "--from", "m", RUN_ID, "--project", "verity"]) == 0, "pulling again is a no-op"
    # a corrupted blob on the machine is refused, not stored
    proof_sha = machine.artifacts[pub["outputs"]["run_files"]]["payload"]["files"][0]["sha256"]
    (machine.root / "objects/sha256" / proof_sha).write_bytes(b"tampered")
    local2 = FakeStore(tmp_path / "local2")
    monkeypatch.setattr(A, "open_store", lambda store=None, root=None: local2)
    rc = cli_prov.main(["pull", "--from", "m", RUN_ID, "--project", "verity"])
    assert rc == 1 and "different sha256" in capsys.readouterr().err and proof_sha not in local2.blobs


def test_attempt_publish_cli(tmp_path, monkeypatch, capsys):
    d = _run_copy(tmp_path)
    st = FakeStore(tmp_path / "store")
    monkeypatch.setattr(A, "open_store", lambda store=None, root=None: st)
    rc = cli_prov.main(["attempt", "publish", str(d)])
    out = capsys.readouterr().out
    assert rc == 0 and "published 3 output(s)" in out and st.get_attempt(RUN_ID)
    rc = cli_prov.main(["attempt", "publish", "--all", "--runs-dir", str(tmp_path / "runs")])
    assert rc == 0 and "1 run dir(s): 0 published, 1 already present, 0 refused (not terminal), 0 error(s)" in capsys.readouterr().out
    rc = cli_prov.main(["attempt", "publish", RUN_ID[:14], "--runs-dir", str(tmp_path / "runs"), "--json"])
    assert rc == 0 and json.loads(capsys.readouterr().out)[0]["already"] is True


def test_no_module_imports_the_storage_half_at_import_time():
    """The two halves merge later; the provenance modules must import on their own (store-core's modules are lazy-imported)."""
    import re

    here = Path(A.__file__).parent
    for name in ("attempt", "cli_prov", "derivation", "tool", "tools_registry"):
        src = (here / f"{name}.py").read_text()
        for line in src.splitlines():
            if re.match(r"^(from|import)\s+", line):
                assert not re.search(r"\.(local|index|remote|remote_s3|kinds|model|ids|cli)\b", line), f"{name}.py: {line}"
