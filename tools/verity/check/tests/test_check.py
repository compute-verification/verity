"""`tools/verity/check/check.py`: its steps in order, the Lean hook with and without the verifier, and the result a recorded run publishes."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("repo_check", ROOT / "tools" / "verity" / "check" / "check.py")
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)


def test_the_steps_are_the_suites_then_circuit_check_then_the_lean_hook(tmp_path, monkeypatch):
    monkeypatch.setattr(CHECK, "FLOCK_VERIFIER", tmp_path / "verifier")
    monkeypatch.setattr(CHECK, "FLOCK_LEAN", tmp_path / "verifier-lean")
    names = [(n, argv if isinstance(argv, str) else argv[1:4]) for n, argv, _cwd in CHECK.steps(tmp_path)]
    assert names[0] == ("pytest", [str(ROOT / "tools/verity/check/suites.py"), "--json", str(tmp_path / "suites.json")])
    assert names[1] == ("flock-circuit-build", [str(ROOT / "verity" / "ml" / "flock" / "check_build.sh")])
    assert names[2] == ("circuit-check", ["-m", "circuit_check", "--all"])
    assert names[3] == ("flock-rust-tests", ["CARGO_BUILD_JOBS=1", "RUST_TEST_THREADS=1", "bash"])
    assert names[4] == ("lean", "skipped: verity/core/service/flock is not on this tree")


def test_the_build_step_checks_every_cpu_feature_set_a_pod_builds():
    """A proving build never has `seed-injection` (1053c0c9 compiled only with it); the class sweep's proving build has no
    `glue`."""
    script = (ROOT / "verity" / "ml" / "flock" / "check_build.sh").read_text()
    assert "for features in sha512 sha512,glue sha512,glue,seed-injection; do" in script
    assert "cargo check -q -p flock-live --bin flock-circuit" in script


def test_the_rust_tests_run_the_live_crates_unit_tests_and_every_test_of_the_crates_the_patches_change():
    """In release, as upstream's CI runs them, on the checkout the build step patches; in two invocations, since `--lib` on the
    upstream crates would drop their integration tests."""
    script = (ROOT / "verity" / "ml" / "flock" / "check_build.sh").read_text()
    start = script.index('if [ "${1:-}" = test ]')
    mode = script[start:script.index("\n  exit\nfi\n", start)]
    assert "for features in sha512 sha512,glue,seed-injection; do" in mode, "the proving build's set, without seed-injection"
    assert 'cargo test --release -q -p flock-live --lib --features "$features"' in mode
    patched = set()
    for p in sorted((ROOT / "verity" / "ml" / "flock").glob("flock-*-b684b12.patch")):
        patched |= {m.group(1) for line in p.read_text().splitlines() if (m := re.match(r"\+\+\+ b/crates/([a-z-]+)/", line))}
    upstream = re.search(r"cargo test --release -q((?: -p [a-z-]+)+)$", mode, re.M).group(1).split(" -p ")[1:]
    assert sorted(upstream) == sorted(patched - {"flock-live"}), "every crate a patch changes, and only those"


def test_the_rust_tests_key_moves_with_the_script_patches_live_crate_and_toolchain_only(tmp_path, monkeypatch):
    assert CHECK.rust_toolchain(), "check_build.sh pins its toolchain: an agent VM's default (1.83) can't build upstream"
    flock = tmp_path / "verity" / "ml" / "flock"
    files = {"verity/ml/flock/check_build.sh": "set -e\n", "verity/ml/flock/flock-sha512-b684b12.patch": "p",
             "verity/ml/flock/live/src/lib.rs": "fn a() {}", "verity/ml/flock/verifier/ci.py": "", "verity/ml/flock/README.md": ""}
    for f, text in files.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text)
    assert CHECK.rust_tests_inputs(list(files)) == list(files)[:3]
    home = tmp_path / "home"
    rustc = home / ".cargo" / "bin" / "rustc"
    monkeypatch.setattr(CHECK, "ROOT", tmp_path)
    monkeypatch.setattr(CHECK, "tracked_files", lambda: list(files))
    e = {"HOME": str(home), "PATH": "/nonexistent"}
    rustc.parent.mkdir(parents=True)
    rustc.write_text('#!/bin/sh\n[ "$RUSTUP_TOOLCHAIN" = 1.98.1 ] && [ "$RUSTUP_AUTO_INSTALL" = 0 ] || exit 1\necho "rustc $RUSTUP_TOOLCHAIN"\n')
    rustc.chmod(0o755)
    assert CHECK.rust_tests_key(e) is None, "a script that pins no toolchain: what it runs is the machine's to decide"
    (flock / "check_build.sh").write_text("set -e\nRUST_TOOLCHAIN=${RUST_TOOLCHAIN:-1.97.0}\n")
    assert CHECK.rust_tests_key(e) is None, "the pin isn't installed yet: the script installs it, and that run stores nothing"
    (flock / "check_build.sh").write_text("set -e\nRUST_TOOLCHAIN=${RUST_TOOLCHAIN:-1.98.1}\n")
    rustc.rename(home / "rustc")
    assert CHECK.rust_tests_key(e) is None, "no rustc yet: the script installs one, and that run stores nothing"
    (home / "rustc").rename(rustc)
    key = CHECK.rust_tests_key(e)
    assert key, "the key's rustc runs under the script's pin, and never installs it"
    (flock / "verifier" / "ci.py").write_text("changed")
    (flock / "README.md").write_text("changed")
    assert CHECK.rust_tests_key(e) == key
    (flock / "live" / "src" / "lib.rs").write_text("fn b() {}")
    assert CHECK.rust_tests_key(e) != key
    key = CHECK.rust_tests_key(e)
    rustc.write_text("#!/bin/sh\necho 'rustc 1.99.0'\n")
    assert CHECK.rust_tests_key(e) != key


#: an upstream verifier that accepts every session it replays (the last argument)
_BIN = b"#!/bin/sh\nfor a; do s=$a; done\nprintf 'UPSTREAM\\t{\"session\":\"%s\",\"verdict\":{\"accepted\":true,\"why\":\"\"}}\\n' \"$s\"\n"


def _pinned(tmp_path, monkeypatch, binary=_BIN, pinned=_BIN):
    """A verifier tree with a pin over a one-binary upstream build and an inputs tree holding one set's statement and stored
    honest session, both built here."""
    import hashlib
    import io
    import json
    import shutil
    import tarfile

    v = tmp_path / "verifier"
    (v / "lean").mkdir(parents=True)
    (v / "lean" / "lakefile.toml").write_text("")
    (v / "ci.py").write_text("import json, sys; open(sys.argv[sys.argv.index('--out') + 1] + '/argv.json', 'w').write(json.dumps(sys.argv))\n")
    shutil.copy(ROOT / "verity" / "ml" / "flock" / "verifier" / "agree.py", v / "agree.py")
    (v / "vectors.json").write_text(json.dumps({"replayable": {"sets": [{
        "pr83": "abc", "lean_statement": "verity/flock-circuit@abc", "statement": {"circuit": "art:0123/c.txt", "public": "art:0123/pub.bin"},
        "sessions": [{"dir": "art:0456/honest", "expect": {"accepted": True, "why": ""}}]}]}}))

    def tgz(path, files):
        with tarfile.open(path, "w:gz") as t:
            for name, data in files.items():
                i = tarfile.TarInfo(name)
                i.size = len(data)
                t.addfile(i, io.BytesIO(data))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    sent = tmp_path / "out" / "inputs"
    sent.mkdir(parents=True)
    up = tgz(sent / "a.tar.gz", {"upstream-abc": binary, "BUILD.json": b"{}"})
    ins = tgz(sent / "b.tar.gz", {"0123/c.txt": b"c", "0123/pub.bin": b"p", "0456/honest/session.json": b"{}"})
    pin = {"upstream": {"art": "art:u", "sha256": up, "binaries": {"abc": hashlib.sha256(pinned).hexdigest()}},
           "inputs": {"art": "art:i", "sha256": ins}, "ci": {"seed": 1, "fuzz": 2}}
    (v / "upstream.json").write_text(json.dumps(pin))
    monkeypatch.setattr(CHECK, "FLOCK_VERIFIER", v)
    monkeypatch.setattr(CHECK, "FLOCK_LEAN", v / "lean")
    monkeypatch.setattr(CHECK, "UPSTREAM_PIN", v / "upstream.json")
    monkeypatch.setattr(CHECK, "memory_gb", lambda: 64)
    return tmp_path / "out", sent


def test_the_lean_steps_once_the_verifier_is_on_the_tree(tmp_path, monkeypatch):
    """`lake build`, the unit-cut conformance and the audit of every Lean package are required; the agreement runs when the run
    has the pinned upstream build and inputs (found by hash, whatever their names), and is skipped by name otherwise."""
    out, sent = _pinned(tmp_path, monkeypatch)
    (build, lake, _), (unit, cut, _), (audit, lean_audit, _), (agree, argv, _) = CHECK.lean_steps(out)
    assert (build, unit, audit, agree) == ("lean-build", "lean-unit-cut", "lean-audit", "lean-agreement")
    assert lake[-1].endswith("lake build")
    assert cut[1].endswith("verifier/unit_cut_agree.py") and cut[cut.index("--lean") + 1].endswith(".lake/build/bin/flock-verify")
    assert lean_audit[1].endswith("tools/verity/check/check.py") and lean_audit[2] == "--lean-slot"
    assert lean_audit[4].endswith("tools/verity/lean/lean_audit.py") and {"--cache", "--deps"} <= set(lean_audit) and "--fresh" not in lean_audit
    assert "--fresh" in CHECK.lean_steps(out, cache=False)[2][1]
    assert "--regenerate" in lean_audit and "--regenerate" not in CHECK.lean_steps(out, regenerate=False)[2][1]
    assert argv[2:4] == ["--lean-agreement", str(out)] and {Path(x).name for x in argv[4:]} == {"a.tar.gz", "b.tar.gz"}
    (sent / "b.tar.gz").unlink()
    assert CHECK.lean_steps(out)[3][1] == "only upstream of the pinned upstream build and inputs was sent"
    (sent / "a.tar.gz").unlink()
    assert CHECK.lean_steps(out)[3][1].startswith("skipped: the pinned upstream build was not sent")


def test_the_agreement_fails_while_the_verifier_requires_a_package_from_outside_it(tmp_path, monkeypatch):
    """The agreement's pass is kept under the verifier's directory only (`agreement_closure`), and `research merge` asks for
    it only for changes there, so a requirement from outside it, by the manifest or the lakefile, fails the step whether or
    not the upstream build was sent; one inside the directory doesn't, and this tree's verifier has none."""
    import json

    out, sent = _pinned(tmp_path, monkeypatch)
    lean = CHECK.FLOCK_LEAN
    (lean / "lakefile.toml").write_text('name = "flock_verifier"\n[[require]]\nname = "inner"\npath = "inner"\n')
    assert not isinstance(CHECK.lean_steps(out)[3][1], str)
    (lean / "lakefile.toml").write_text('name = "flock_verifier"\n[[require]]\nname = "identity"\npath = "../identity"\n')
    assert CHECK.lean_steps(out)[3][1].startswith("the verifier's Lean package requires identity from outside its directory")
    (lean / "lakefile.toml").write_text('name = "flock_verifier"\n')
    (lean / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "git", "name": "mathlib", "rev": "abc"}]}))
    for f in sent.iterdir():
        f.unlink()
    assert CHECK.lean_steps(out)[3][1].startswith("the verifier's Lean package requires mathlib from outside")
    assert CHECK.requires_outside(ROOT / "verity" / "core" / "service" / "flock") == []


def test_the_lean_dependency_urls_leave_the_run_directory_before_any_step(tmp_path, monkeypatch):
    """The presigned URLs sent with a run reach the audit step from outside the run directory, which the run's record keeps."""
    monkeypatch.delenv(CHECK.LEAN_DEPS_URLS_ENV, raising=False)
    assert CHECK.take_lean_deps_urls(tmp_path) is None
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / CHECK.LEAN_DEPS_URLS).write_text('{"k": "https://r2/o?sig"}')
    got = CHECK.take_lean_deps_urls(tmp_path)
    try:
        assert not (tmp_path / "inputs" / CHECK.LEAN_DEPS_URLS).exists() and not got.is_relative_to(tmp_path)
        assert got.read_text() == '{"k": "https://r2/o?sig"}' and got.stat().st_mode & 0o777 == 0o600
        audit = CHECK.lean_steps(tmp_path, urls=got)[2][1]
        assert audit[audit.index("--urls") + 1] == str(got) and "--urls" not in CHECK.lean_steps(tmp_path)[2][1]
    finally:
        got.unlink()
    monkeypatch.setenv(CHECK.LEAN_DEPS_URLS_ENV, "/somewhere/urls.json")                  # a local recorded run's launcher
    assert CHECK.take_lean_deps_urls(tmp_path) == Path("/somewhere/urls.json")


def test_custody_deletes_the_lean_dependency_urls_wherever_a_run_leaves_them():
    """A run of anything else, or a check that ends before `check.py` starts, leaves them in inputs/ for custody to delete."""
    from research.store.custody import CREDENTIALS

    assert f"inputs/{CHECK.LEAN_DEPS_URLS}" in CREDENTIALS


def test_the_agreement_leaves_memory_for_the_steps_beside_it(monkeypatch):
    """vy-mig-check-21 (128 GB, 32 cores) gave the agreement 16 verifier processes at 8 GB each while lean-suites ran beside it,
    and its cgroup OOM-killed 36 of them (6 Oct): the agreement counts on half."""
    monkeypatch.setattr(CHECK, "memory_gb", lambda: 128)
    monkeypatch.setattr(CHECK, "cores", lambda: 32)
    assert CHECK.session_jobs() * 8 <= 128 // 2
    monkeypatch.setattr(CHECK, "memory_gb", lambda: 8)
    assert CHECK.session_jobs() == 1


def test_the_agreement_runs_ci_on_the_pinned_binaries_only(tmp_path, monkeypatch):
    import json

    out, sent = _pinned(tmp_path, monkeypatch)
    assert CHECK.lean_agreement(out, sent / "a.tar.gz", sent / "b.tar.gz") == 0
    argv = json.loads((out / "flock-agreement" / "argv.json").read_text())
    assert argv[argv.index("--upstream") + 1] == f"abc={out / 'flock-agreement' / 'upstream' / 'upstream-abc'}"
    assert argv[argv.index("--inputs") + 1].endswith("flock-agreement/inputs") and argv[argv.index("--fuzz") + 1] == "2"
    assert int(argv[argv.index("--session-jobs") + 1]) == CHECK.session_jobs() >= 1
    out, sent = _pinned(tmp_path / "t", monkeypatch, binary=b"tampered")
    assert CHECK.lean_agreement(out, sent / "a.tar.gz", sent / "b.tar.gz") == 1


def test_a_build_that_cannot_run_here_fails_the_preflight_in_seconds(tmp_path, monkeypatch, capsys):
    """A binary built for another CPU starts, parses its arguments, and dies of SIGILL on the first instruction this CPU
    lacks, printing nothing: only a real replay shows it. So does a binary that replays with the wrong verdict."""
    ill = b"#!/bin/sh\nkill -ILL $$\n"
    out, sent = _pinned(tmp_path, monkeypatch, binary=ill, pinned=ill)
    assert CHECK.lean_agreement(out, sent / "a.tar.gz", sent / "b.tar.gz") == 1
    err = capsys.readouterr().err
    assert "killed by SIGILL replaying its stored honest session" in err and "target-cpu=native" in err
    assert not (out / "flock-agreement" / "argv.json").exists()                       # ci.py never started
    rejects = _BIN.replace(b'"accepted":true', b'"accepted":false')
    out, sent = _pinned(tmp_path / "r", monkeypatch, binary=rejects, pinned=rejects)
    assert CHECK.lean_agreement(out, sent / "a.tar.gz", sent / "b.tar.gz") == 1 and "expected" in capsys.readouterr().err


def test_the_real_preflight_replays_a_stored_session_per_binary_it_can():
    """Sets 0-5 store honest sessions: four of the eight pinned binaries replay one; the build flags are the same for all."""
    import json

    sets = json.loads((ROOT / "verity" / "ml" / "flock" / "verifier" / "vectors.json").read_text())["replayable"]["sets"]
    assert len({s["pr83"] for s in sets if s.get("sessions") and "tables" not in s["statement"]}) >= 4


def test_the_rule_the_gate_reads(monkeypatch):
    spec2 = importlib.util.spec_from_file_location("check_tool", ROOT / "tools" / "verity" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(tool)
    assert tool.CHECK.merge_requires["lean-agreement"] == ["verity/ml/flock/", "verity/core/service/flock/", "!README.md",
                                                           "!PROTOCOL.md"]
    assert set(tool.CHECK.merge_requires) == {"lean-agreement", "lean-suites"}
    assert {"verity/core/service/flock/VStar/In.lean", "verity/core/service/vstar/"} <= set(tool.CHECK.merge_requires["lean-suites"])
    pin = __import__("json").loads((ROOT / "verity" / "ml" / "flock" / "verifier" / "upstream.json").read_text())
    assert set(pin["upstream"]["binaries"]) == {s["pr83"] for s in __import__("json").loads(
        (ROOT / "verity" / "ml" / "flock" / "verifier" / "vectors.json").read_text())["replayable"]["sets"]}


def test_the_recorded_check_takes_cpu_only_torch():
    spec2 = importlib.util.spec_from_file_location("check_tool", ROOT / "tools" / "verity" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(tool)
    assert tool.CHECK.command[:6] == ["uv", "run", "--locked", "--all-packages", "--extra", "torch-cpu"], \
        "every member's own dependencies, not only the root's dev group"
    import tomllib
    cpu = [p for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"] if p["name"] == "torch" and "+cpu" in p["version"]]
    assert cpu and all("download.pytorch.org/whl/cpu" in p["source"]["registry"] for p in cpu)
    assert not [d["name"] for p in cpu for d in p.get("dependencies", []) if d["name"].startswith(("nvidia-", "cuda-", "triton"))]


def test_a_recorded_run_writes_a_valid_result(tmp_path, monkeypatch):
    from research import result as RES

    monkeypatch.setenv("RESEARCH_RUN_ID", "r20260927-010000-abcd")
    (tmp_path / "circuit-check.json").write_text("{}")
    done = [{"step": "pytest", "status": "passed", "rc": 0, "seconds": 1.0}, {"step": "circuit-check", "status": "failed", "rc": 1, "seconds": 2.0}]
    CHECK.write_result(tmp_path, False, done)
    r = RES.check(tmp_path, "r20260927-010000-abcd")
    assert r.valid, r.errors
    assert r.validation_status == "failed" and r.doc["kind"] == "check/v1" and r.doc["artifacts"] == ["circuit-check.json"]


def test_the_steps_see_the_trees_packages_only(monkeypatch):
    """On a machine the runner's PYTHONPATH starts with its own shipped `research`; the steps must import the tree's."""
    monkeypatch.setenv("PYTHONPATH", f"/opt/research-tool:{ROOT / 'tools/research/src'}:/elsewhere")
    assert CHECK.env()["PYTHONPATH"] == str(ROOT / "tools/research/src")
    monkeypatch.setenv("PYTHONPATH", "/opt/research-tool")
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", "abc")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", "/workspace/research/cache/pycache")
    env = CHECK.env()
    assert "PYTHONPATH" not in env and "RESEARCH_SOURCE_SHA" not in env and "PYTHONPYCACHEPREFIX" not in env


def test_a_recorded_runs_steps_require_the_store_and_get_its_custody_dir(tmp_path, monkeypatch):
    """The steps lose RESEARCH_*, so the run's custody key reaches `store_io` as CHECK_STORE_CUSTODY, and the store is required."""
    run = tmp_path / "research" / "runs" / "r1"
    run.mkdir(parents=True)
    custody = tmp_path / "research" / "requests" / "r1" / "custody"
    custody.mkdir(parents=True)
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    env = CHECK.env()
    assert env[CHECK.STORE_REQUIRED_ENV] == "1" and CHECK.CUSTODY_ENV not in env, "no key staged: required, nothing to pass"
    assert env[CHECK.REGRESSION_ENV] == "1", "a recorded check blocks on the vLLM reference regression's T0 tier"
    (custody / "store.toml").write_text("[remote]\n")
    (custody / "cred.json").write_text("{}")
    env = CHECK.env()
    assert env[CHECK.CUSTODY_ENV] == str(custody.resolve()) and not any(k.startswith("RESEARCH_") for k in env)
    assert CHECK.CUSTODY_ENV not in CHECK._env_key(env), "the per-run path must not change what a pass is keyed on"
    monkeypatch.delenv("RESEARCH_RUN_ID")
    assert not {CHECK.STORE_REQUIRED_ENV, CHECK.CUSTODY_ENV, CHECK.REGRESSION_ENV} & set(CHECK.env())


def _repo(root: Path):
    """A PR branch `pr` (b.txt) off `main`, which has since gained c.txt; checked out on `pr`. Returns (git, commit)."""
    import subprocess

    def git(*args):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=root, check=True,
                       capture_output=True)

    def commit(rel, text):
        (root / rel).write_text(text)
        git("add", "-A")
        git("commit", "-q", "-m", rel)

    git("init", "-q", "-b", "main")
    commit("a.txt", "a\n")
    git("checkout", "-q", "-b", "pr")
    commit("b.txt", "b\n")
    git("checkout", "-q", "main")
    commit("c.txt", "c\n")                                  # main moves past the PR's base
    git("checkout", "-q", "pr")
    return git, commit


def test_a_commit_lands_as_is_only_when_merging_it_into_main_leaves_its_own_tree(tmp_path, monkeypatch):
    """The 1 Oct count: 15 of 16 checks of a PR's own head counted for nothing, since `research merge` takes a check only of the tree
    that lands; `--record` warns before spending a slot on one."""
    monkeypatch.setattr(CHECK, "ROOT", tmp_path)
    git, commit = _repo(tmp_path)
    assert not CHECK.landable("main")                                # main gained c.txt, which the PR lacks
    commit("c.txt", "c\n")                                           # the same change on the PR: the merge is its own tree
    assert CHECK.landable("main")
    git("merge", "-q", "--no-edit", "main")                          # main is in it
    assert CHECK.landable("main")
    commit("c.txt", "c2\n")
    git("checkout", "-q", "main")
    commit("c.txt", "c3\n")
    git("checkout", "-q", "pr")
    assert not CHECK.landable("main")                                # a conflict lands nothing


def test_record_warns_when_the_commit_cannot_land_as_is_and_checks_it_anyway(monkeypatch, capsys):
    import subprocess
    import sys

    real, launched = subprocess.run, []

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            launched.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: None)
    monkeypatch.setattr(CHECK, "gather_verdicts", lambda d, given=None: None)
    for can, said in ((False, True), (True, False)):
        monkeypatch.setattr(CHECK, "landable", lambda base, can=can: can)
        assert CHECK.record(["--base", "HEAD"]) == 0 and len(launched) == 1
        out = capsys.readouterr().out
        assert ("check: this commit can't land as is; the train's check is the merge check" in out) is said
        launched.clear()


def test_the_ratchet_gate_step_fails_a_train_naming_the_pr_whose_merge_grows_an_allowlist(tmp_path, monkeypatch):
    """With the ratchet gate out of drive's per-merge landing gates, the train's check is where an allowlist that grows is caught:
    the step judges each `Queue: #N` merge above the run's sent base against the one below, with the gate as the base has it, so
    a train holding one merge that grows an allowlist fails naming that PR, even when the PR also rewrites the gate to pass, and a
    clean train passes.  Its verdicts are kept under the check's cache, and none under --no-cache; a base that names no commit
    in the tree skips it, which no after-landing run passes."""
    import json
    import os
    import subprocess

    p01 = "integrations/vllm/tests/lint/allowlists/p01_core.json"
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True).stdout.strip()

    def commit(message: str, files: dict[str, str]) -> str:
        for path, text in files.items():
            (repo / path).parent.mkdir(parents=True, exist_ok=True)
            (repo / path).write_text(text)
            git("add", path)
        git("commit", "-q", "-m", message)
        return git("rev-parse", "HEAD")

    def allowlist(*files: str) -> str:
        return json.dumps({"rule": "p01", "entries": [{"file": f, "kind": "k", "symbol": "s", "detail": "d"} for f in files]})

    git("init", "-q", "-b", "main")
    base = commit("base", {CHECK.RATCHET_GATE: (ROOT / CHECK.RATCHET_GATE).read_text(), p01: allowlist("a.py", "b.py")})
    heads = {}
    for n, files in ((1, {"a.py": "A = 1\n"}),
                     (2, {"c.py": "C = 1\n", p01: allowlist("a.py", "b.py", "c.py"),
                          CHECK.RATCHET_GATE: "import sys\nsys.exit(0)\n"}),
                     (3, {"d.py": "D = 1\n"})):
        git("checkout", "-q", "-b", f"pr{n}", base)
        heads[n] = commit(f"pr {n}", files)

    def train(name: str, prs: tuple[int, ...]) -> dict[int, str]:
        git("checkout", "-q", "-b", name, base)
        merges = {}
        for n in prs:
            git("merge", "-q", "--no-ff", "-m", f"Queue: #{n} pr {n}\n\nPR: #{n}\nHead: {heads[n]}\nOwner: lane\n", heads[n])
            merges[n] = git("rev-parse", "HEAD")
        return merges

    def step(cache: bool = True) -> subprocess.CompletedProcess:
        argv = CHECK.ratchet_gate(out, cache)
        assert not isinstance(argv, str), argv
        return subprocess.run(argv, cwd=repo, capture_output=True, text=True,
                              env={**env, "RATCHET_GATE_CACHE": "elsewhere", "XDG_CACHE_HOME": str(tmp_path / "xdg")})

    out = tmp_path / "run"
    (out / "inputs").mkdir(parents=True)
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "cache"))
    (out / "inputs" / CHECK.BASE_FILE).write_text(f"{CHECK._rev('HEAD')}\n")
    (built,) = [s[1] for g in CHECK.groups(out, 2) for s in g if s[0] == "ratchet-gate"]
    assert built == CHECK.ratchet_gate(out, True) and CHECK._rev("HEAD") in built, built
    monkeypatch.setattr(CHECK, "ROOT", repo)
    (out / "inputs" / CHECK.BASE_FILE).write_text(f"{base}\n")
    merges = train("next", (1, 2, 3))
    got = step()
    assert got.returncode == 1, got.stdout + got.stderr
    assert re.search(rf"^  #2 \({merges[2][:12]}\): failed: grows {p01} 2 -> 3 entries", got.stdout, re.M), got.stdout
    assert [line.split()[0] for line in got.stdout.splitlines() if ": failed: " in line] == ["#2"]
    assert list((tmp_path / "cache" / "ratchet-gate").glob("*.json")) and not (repo / "elsewhere").exists()
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "cold"))
    assert step(cache=False).returncode == 1
    assert not (tmp_path / "cold").exists() and not (tmp_path / "xdg").exists(), "--no-cache: no verdict kept anywhere"
    train("clean", (1, 3))
    got = step()
    assert got.returncode == 0 and "2 change(s), 0 failed" in got.stdout, got.stdout + got.stderr
    (out / "inputs" / CHECK.BASE_FILE).write_text("0" * 40 + "\n")
    assert CHECK.ratchet_gate(out, True) == f"skipped: no base: {'0' * 40} names no commit here"
    (out / "inputs" / CHECK.BASE_FILE).unlink()
    assert CHECK.ratchet_gate(out, True) == "skipped: no base: origin/main names no commit here"


def test_record_sends_the_verdicts_it_gathers_with_a_pack_given_added_unless_no_cache(tmp_path, monkeypatch):
    """`--verdicts-in` adds to the gathered pack, never replaces it: a stale pack given once turned reuse off for every check
    it rode with, because the gathered pack it replaced held the passes."""
    import subprocess
    import sys

    real, launched, given_to = subprocess.run, [], []

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            launched.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    def gathered(d: Path, given: Path | None = None) -> Path:
        given_to.append(given)
        (d / "verdicts-gathered.tar.gz").write_bytes(b"")
        return d / "verdicts-gathered.tar.gz"

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: None)
    monkeypatch.setattr(CHECK, "landable", lambda base: True)
    monkeypatch.setattr(CHECK, "slot_wrap", lambda priority: [])
    monkeypatch.setattr(CHECK, "gather_verdicts", gathered)
    assert CHECK.record(["--base", "HEAD", "--on", "vy-x"]) == 0
    [cmd] = launched
    tail = cmd[cmd.index("--") + 1:]
    assert tail[-2] == "--verdicts-in" and tail[-1].endswith("verdicts-gathered.tar.gz") and cmd[cmd.index("--send") + 1] == tail[-1]
    assert given_to == [None]
    launched.clear()
    assert CHECK.record(["--base", "HEAD", "--on", "vy-x", "--verdicts-in", "mine.tar.gz"]) == 0
    [cmd] = launched
    tail = cmd[cmd.index("--") + 1:]
    assert given_to[-1] == Path("mine.tar.gz").resolve() and "mine.tar.gz" not in " ".join(cmd)
    assert tail[-2:] == ["--verdicts-in", cmd[cmd.index("--send") + 1]] and tail[-1].endswith("verdicts-gathered.tar.gz")
    launched.clear()
    assert CHECK.record(["--base", "HEAD", "--no-cache"]) == 0 and len(given_to) == 2
    assert not any(x.endswith("verdicts-gathered.tar.gz") for x in launched[0])
    launched.clear()
    # drive's after-landing run: circuit-check's passes come from the gathered pack (main imports only theirs)
    assert CHECK.record(["--base", "HEAD", "--on", "vy-x", "--no-cache", "--circuit-cache"]) == 0 and len(given_to) == 3
    [cmd] = launched
    tail = cmd[cmd.index("--") + 1:]
    assert tail[-4:-2] == ["--no-cache", "--circuit-cache"] and tail[-2] == "--verdicts-in", tail
    launched.clear()
    assert CHECK.record(["--base", "HEAD", "--no-cache"]) == 0 and len(given_to) == 3
    head = launched[0][:launched[0].index("--")]
    assert [head[i + 1] for i, x in enumerate(head) if x == "--declared-output"] == list(CHECK.PACK_FILES), \
        "custody uploads the packs however the run ends, never trimming them as undeclared"


def test_record_sends_the_agreement_for_a_file_renamed_out_of_its_paths(tmp_path, monkeypatch, capsys):
    """git reports a rename by its destination alone unless asked not to, which would leave out the step the file's old path
    requires."""
    import json
    import subprocess
    import sys
    import tempfile

    import research.store.local

    def git(*args):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=tmp_path,
                       check=True, capture_output=True)

    (tmp_path / "tools" / "verity" / "check").mkdir(parents=True)
    (tmp_path / "tools" / "verity" / "check" / "tool.py").write_text(
        "from research.store.tool import Tool\nCHECK = Tool(name='check', version='1', command=['true'], "
        "merge_requires={'lean-agreement': ['verity/ml/flock/', '!README.md']})\n")
    (tmp_path / "verity" / "ml" / "flock").mkdir(parents=True)
    (tmp_path / "verity" / "ml" / "flock" / "verify.rs").write_text("fn main() {}\n")
    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    git("mv", "verity/ml/flock/verify.rs", "verify.rs")
    git("commit", "-q", "-m", "move")
    (tmp_path / "upstream.json").write_text(json.dumps({"upstream": {"art": "art:u"}, "inputs": {"art": "art:i"}}))

    class Store:
        def fetch(self, art):
            (tmp_path / art[4:]).write_text("")
            return tmp_path / art[4:]

    real, launched = subprocess.run, []

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            launched.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    monkeypatch.setattr(CHECK, "ROOT", tmp_path)
    monkeypatch.setattr(CHECK, "UPSTREAM_PIN", tmp_path / "upstream.json")
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: None)
    monkeypatch.setattr(research.store.local, "LocalStore", Store)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", run)
    assert CHECK.record(["--base", "HEAD^"]) == 0
    [cmd] = launched
    assert "flock_upstream=art:u" in cmd and "sending the pinned upstream build" in capsys.readouterr().out


def test_check_refuses_a_python_that_imports_the_trees_packages_from_elsewhere(monkeypatch, capsys):
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    monkeypatch.setenv("PYTHONPATH", f"/tmp/s1350/packages/verity/src:{ROOT / 'tools/research/src'}")
    probed = []
    monkeypatch.setattr(CHECK.SUITES, "foreign", lambda pkgs, env: probed.append(env.get("PYTHONPATH")) or "verity from /tmp/s1350/x.py")
    monkeypatch.setattr(CHECK, "run", lambda *a: pytest.fail("a step ran"))
    assert CHECK.main([]) == 2
    assert "against another checkout's code (verity from /tmp/s1350/x.py)" in capsys.readouterr().err
    assert probed == [str(ROOT / "tools/research/src")]


def test_a_recorded_run_refuses_to_skip_the_store(monkeypatch):
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setenv("VERITY_SKIP_STORE", "1")
    assert CHECK.main([]) == 2

def test_circuit_check_has_the_whole_slot_only_when_neither_pytest_nor_lean_runs_beside_it(tmp_path, monkeypatch):
    """12 Oct 17:41Z: an unsharded check (r20261012-155204-6475) gave circuit-check the slot's whole 560 GiB while pytest and
    lean-audit held 160 to 350 GiB beside it, and the kernel OOM-killed a 141 GiB circuit-check worker."""
    monkeypatch.setattr(CHECK, "FLOCK_LEAN", tmp_path / "verifier-lean")
    monkeypatch.setattr(CHECK, "memory_gb", lambda: 2048)
    monkeypatch.setenv(CHECK.SLOT.SLOT_MEM_ENV, "560")

    def mem(only):
        [cc] = [s for g in CHECK.groups(tmp_path, 16, only=only) for s in g if s[0] == "circuit-check"]
        return cc[1][cc[1].index("--mem-gb") + 1]

    assert mem(None) == mem(["circuit-check", "lean-build"]) == mem(["circuit-check", "pytest", "lean-build"]) == "210"
    assert mem(["flock-circuit-build", "circuit-check", "flock-rust-tests"]) == "560"
    assert CHECK.cc_mem_gb(400, True) == 100 and CHECK.cc_mem_gb(100, True) == 25


def test_a_cached_circuit_check_keeps_the_catalog_bindings_records_and_an_uncached_one_does_not(tmp_path):
    """With the cache, a run whose task list misses builds only the catalog bindings whose record misses
    (`circuit_check.units`); the uncached run builds the whole catalog, as the nightly check that must agree with it does."""
    def cc(cache):
        [s] = [s for g in CHECK.groups(tmp_path, 16, cache=cache) for s in g if s[0] == "circuit-check"]
        return s[1]

    assert "--catalog-units" in cc(True) and "--cache" in cc(True)
    assert "--catalog-units" not in cc(False) and "--cache" not in cc(False)


def test_beside_pytest_alone_circuit_check_takes_the_share_lean_suites_would_have(tmp_path, monkeypatch):
    """A post-landing run (`--deferred circuit-check,pytest`) runs no Lean step, yet circuit-check got 210 GiB of a 560 GiB slot
    while pytest kept to its half of the rest, 175: lean-suites' 175 went unused, and the 147-167 GiB SHA-512 targets ran one at
    a time (b103, r20261019-015150-b103: 4 of them, 5,318 s end to end).  Pytest's share is unchanged."""
    monkeypatch.setattr(CHECK, "FLOCK_LEAN", tmp_path / "verifier-lean")
    monkeypatch.setattr(CHECK, "memory_gb", lambda: 2048)
    monkeypatch.setenv(CHECK.SLOT.SLOT_MEM_ENV, "560")

    def mem(only, step):
        [s] = [s for g in CHECK.groups(tmp_path, 16, only=only) for s in g if s[0] == step]
        return s[1][s[1].index("--mem-gb") + 1]

    assert mem(["circuit-check", "pytest"], "circuit-check") == "385"
    assert mem(["circuit-check", "pytest"], "pytest") == mem(None, "pytest") == "175"
    assert CHECK.cc_mem_gb(400, True, lean=False) == 250 and CHECK.cc_mem_gb(100, True, lean=False) == 63


def test_circuit_check_is_granted_the_slot_once_every_step_beside_it_has_ended(tmp_path, monkeypatch):
    """b103 (r20261019-015150-b103): pytest ended 83 minutes into circuit-check's 125, and circuit-check kept to the share it
    had beside pytest to the end.  Beside other steps it gets ``--mem-gb-file``, which `run` writes the slot's mem= to once those
    steps have ended (`grant_when_alone`), a stale grant removed first; alone in the slot it has all of it already."""
    import threading

    monkeypatch.setattr(CHECK, "FLOCK_LEAN", tmp_path / "verifier-lean")
    monkeypatch.setattr(CHECK, "memory_gb", lambda: 2048)
    monkeypatch.setenv(CHECK.SLOT.SLOT_MEM_ENV, "560")

    def argv(only):
        [cc] = [s for g in CHECK.groups(tmp_path, 16, only=only) for s in g if s[0] == "circuit-check"]
        return cc[1]

    assert argv(["circuit-check", "pytest"])[-2:] == ["--mem-gb-file", str(tmp_path / CHECK.CC_GRANT)]
    assert "--mem-gb-file" not in argv(["flock-circuit-build", "circuit-check", "flock-rust-tests"])
    done = [threading.Event(), threading.Event()]
    for e in done:
        e.set()
    CHECK.grant_when_alone(tmp_path / CHECK.CC_GRANT, done, 560)
    assert (tmp_path / CHECK.CC_GRANT).read_text() == "560\n"

    grant, ok = tmp_path / CHECK.CC_GRANT, _py("pass")
    grant.unlink()
    _fake_run(monkeypatch, [[("pytest", ok, tmp_path)], [("circuit-check", ok, tmp_path)]])
    assert CHECK.run(tmp_path, 2)[0] and grant.read_text() == "560\n"
    grant.write_text("9999\n")
    _fake_run(monkeypatch, [[("circuit-check", ok, tmp_path)]])
    assert CHECK.run(tmp_path, 1)[0] and not grant.exists()


def test_the_tests_get_what_circuit_check_leaves_of_the_slot(tmp_path, monkeypatch):
    """Pytest's parallel suites had no bound beside circuit-check's, so the two together outgrew the slot."""
    monkeypatch.setattr(CHECK, "FLOCK_LEAN", tmp_path / "verifier-lean")
    monkeypatch.setattr(CHECK, "memory_gb", lambda: 2048)
    monkeypatch.setenv(CHECK.SLOT.SLOT_MEM_ENV, "560")

    def mem(only):
        [pt] = [s for g in CHECK.groups(tmp_path, 16, only=only) for s in g if s[0] == "pytest"]
        return pt[1][pt[1].index("--mem-gb") + 1]

    assert mem(None) == mem(["circuit-check", "pytest"]) == "175", "halved: lean-suites runs beside pytest with the same share"
    assert mem(["pytest"]) == "280"
    monkeypatch.delenv(CHECK.SLOT.SLOT_MEM_ENV)
    [pt] = [s for g in CHECK.groups(tmp_path, 16) for s in g if s[0] == "pytest"]
    assert "--mem-gb" not in pt[1]


def test_the_groups_split_the_cores_and_run_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(CHECK, "FLOCK_LEAN", tmp_path / "verifier-lean")
    monkeypatch.setattr(CHECK, "memory_gb", lambda: 2048)
    monkeypatch.delenv(CHECK.SLOT.SLOT_MEM_ENV, raising=False)
    assert "--mem-gb" not in CHECK.groups(tmp_path, 16)[1][1][1]
    monkeypatch.setenv(CHECK.SLOT.SLOT_MEM_ENV, "560")
    (pt,), (build, cc, rust), (lean,), (gate,) = CHECK.groups(tmp_path, 16)
    assert cc[1][cc[1].index("--mem-gb") + 1] == "210", "a slot's mem=, less what pytest and Lean hold beside it, caps circuit-check"
    assert build[0] == "flock-circuit-build" and build[1][-1].endswith("verity/ml/flock/check_build.sh")
    assert rust[0] == "flock-rust-tests" and rust[1][-2:] == [build[1][-1], "test"] and "CARGO_BUILD_JOBS=8" in rust[1]
    assert pt[1][1].endswith("tools/verity/check/suites.py") and pt[1][pt[1].index("--jobs") + 1] == "8" and "--fresh" not in pt[1]
    assert cc[1][cc[1].index("--jobs") + 1] == "8" and "--cache" in cc[1]
    cold = CHECK.groups(tmp_path, 16, cache=False)
    assert lean[0] == "lean" and "--cache" not in cold[1][0][1] and "--fresh" in cold[0][0][1]
    assert gate[0] == "ratchet-gate" and cold[-1][0][0] == "ratchet-gate"
    assert not any(s in pt[1] for s in CHECK.AFTER_LEAN), "without the Lean steps the suites that use Lean run with the rest"


def test_memory_is_what_the_tightest_cgroup_over_this_process_leaves(tmp_path):
    """On a VM a run's limit is on its session scope, not the root cgroup (vy-nebius-cpu-1 caps each at 240 GB); in a pod it is
    the namespace's root; MemAvailable bounds both."""
    scope = tmp_path / "sys/fs/cgroup/user.slice/user-1001.slice/session-9.scope"
    scope.mkdir(parents=True)
    (tmp_path / "proc/self").mkdir(parents=True)
    (tmp_path / "proc/self/cgroup").write_text("0::/user.slice/user-1001.slice/session-9.scope\n")
    (tmp_path / "proc/meminfo").write_text(f"MemTotal: {755 << 20} kB\nMemAvailable: {600 << 20} kB\n")
    assert CHECK.memory_gb(tmp_path) == 600
    (scope / "memory.max").write_text(str(240 << 30))
    (scope / "memory.current").write_text(str(10 << 30))
    (scope.parent / "memory.max").write_text(str(560 << 30))
    (scope.parent / "memory.current").write_text(str(400 << 30))
    assert CHECK.memory_gb(tmp_path) == 160, "the slice leaves 160 of its 560 GB"
    (scope.parent / "memory.current").write_text(str(555 << 30))
    (scope.parent / "memory.stat").write_text(f"anon {255 << 30}\nactive_file {100 << 30}\ninactive_file {200 << 30}\n")
    assert CHECK.memory_gb(tmp_path) == 230, "the slice's 300 GB of page cache is reclaimed at its limit, so it is left"
    (scope.parent / "memory.max").write_text("max")
    assert CHECK.memory_gb(tmp_path) == 230
    pod = tmp_path / "pod"
    (pod / "sys/fs/cgroup").mkdir(parents=True)
    (pod / "proc/self").mkdir(parents=True)
    (pod / "proc/self/cgroup").write_text("0::/\n")
    (pod / "sys/fs/cgroup/memory.max").write_text(str(128 << 30))
    (pod / "proc/meminfo").write_text(f"MemAvailable: {2000 << 20} kB\n")
    assert CHECK.memory_gb(pod) == 128


def test_the_suites_that_use_lean_run_after_the_verifier_is_built_and_beside_the_audit(tmp_path, monkeypatch):
    """Not beside `lake build`: whether their tests run or skip, and what `lake` finds built, must not depend on how far the
    concurrent Lean group has got (verity-flock passed or failed on build order, r20260929-015410-5a20). The audit builds only
    in scratch trees of its own, so it is a group of its own and nothing in the Lean group waits on it (r20261004-024819-11bc:
    the audit's 3,238 s, then the suites' 979 s and the agreement's 515 s, one after another). The suites need only the
    verifier lean-build built, so they wait on it alone and run beside the agreement (r20261004-150518-b201: 1,254 s, then
    1,222 s)."""
    out, _ = _pinned(tmp_path, monkeypatch)
    gs = CHECK.groups(out, 8)
    (_, pytest, _), = gs[0]
    lean = gs[2]
    assert [n for n, _, _ in lean] == ["lean-build", "lean-unit-cut", "lean-agreement"]
    assert [n for n, _, _ in gs[3]] == ["lean-audit"] and gs[3][0][1][4].endswith("tools/verity/lean/lean_audit.py")
    assert [n for n, _, _ in gs[5]] == ["lean-suites"] and CHECK.AFTER["lean-suites"] == "lean-build"
    assert all(pytest[pytest.index(s) - 1] == "--skip" for s in CHECK.AFTER_LEAN)
    argv = gs[5][0][1]
    assert argv[:2] == ["bash", "-c"] and argv[2].startswith(CHECK.ELAN), "with elan's lake on the PATH, as lean-build has it"
    assert all(f" {s} " in argv[2] for s in CHECK.AFTER_LEAN) and "suites-lean.json" in argv[2] and "--jobs 4" in argv[2]
    assert "--timeouts" in pytest and " --timeouts " in argv[2], \
        "each suite bounded by its timeout_s: without it a hung test held a train slot 24 h"


def test_a_checkouts_files_are_gits_and_the_key_environment_drops_the_tree_path(tmp_path):
    """The agreement's key reads the tree as git has it: tracked and untracked-unignored files, not bytecode or builds."""
    import subprocess

    assert CHECK.tracked_files(tmp_path) is None                                   # outside a checkout nothing is keyed
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("__pycache__/\n.lake/\n")
    (tmp_path / "a.txt").write_text("a")
    for built in ("__pycache__/x.pyc", ".lake/build/bin/flock-verify"):
        (tmp_path / built).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / built).write_bytes(b"x")
    assert CHECK.tracked_files(tmp_path) == [".gitignore", "a.txt"]
    assert CHECK._env_key({"VIRTUAL_ENV": f"{ROOT}/.venv", "SSH_CLIENT": "1 2 3"}) == {"VIRTUAL_ENV": "<tree>/.venv"}   # per-commit path


def test_the_agreement_is_not_required_for_docs_alone():
    """README.md and PROTOCOL.md under verity/ml/flock/ are docs the agreement never reads: alone they do not require it."""
    from research.merge import rule_touches

    spec2 = importlib.util.spec_from_file_location("check_tool", ROOT / "tools" / "verity" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(tool)
    rule = tool.CHECK.merge_requires["lean-agreement"]
    docs = ["verity/ml/flock/verifier/PROTOCOL.md", "verity/ml/flock/verifier/README.md", "verity/ml/flock/README.md", "AGENTS.md"]
    assert rule_touches(rule, docs) == []
    assert rule_touches(rule, [*docs, "verity/ml/flock/verifier/ci.py"]) == ["verity/ml/flock/"]
    assert rule_touches(rule, ["verity/ml/flock/verifier/DESIGN.md"]) == ["verity/ml/flock/"]      # only the two names are left out
    for f in ("ci.py", "agree.py", "fuzz.py", "redigest.py", "transcript_check.py", "selftest_records.py"):   # cited, never read
        reads = [line for line in (ROOT / "verity" / "ml" / "flock" / "verifier" / f).read_text().splitlines()
                 if any(r in line for r in ("open(", "read_text", "read_bytes")) and any(d in line for d in ("README", "PROTOCOL"))]
        assert not reads, f"{f} reads a doc the agreement trigger leaves out: {reads}"


def test_the_agreement_scripts_import_only_the_standard_library_and_each_other():
    """`agreement_key` covers verity/ml/flock/verifier/ and the interpreter only: an import from elsewhere in `ci.py` or the scripts
    it runs would be an input the key misses, so one fails here until the key covers it.  The key also leaves out the Lean
    package's `lean-audit.json` (`agreement_closure`), so neither those scripts nor the package's lakefile may mention it."""
    import ast
    import re
    import sys

    here = ROOT / "verity" / "ml" / "flock" / "verifier"
    todo, seen = ["ci"] + re.findall(r'HERE / "([a-z_]+)\.py"', (here / "ci.py").read_text()), set()
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        for node in ast.walk(ast.parse((here / f"{m}.py").read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for n in names:
                top = n.split(".")[0]
                if (here / f"{top}.py").exists():
                    todo.append(top)
                else:
                    assert top in sys.stdlib_module_names or top == "__future__", f"{m}.py imports {n}"
    assert {"agree", "fuzz", "redigest", "transcript_check"} <= seen
    readers = [here / f"{m}.py" for m in sorted(seen)] + [CHECK.FLOCK_LEAN / "lakefile.toml"]
    assert not (named := [p.name for p in readers if "lean-audit" in p.read_text()]), (
        f"{', '.join(named)} mention lean-audit, which agreement_closure leaves out of the agreement's cache key")


def test_the_agreement_reads_the_verifier_package_not_the_proof_packages_nested_in_it(tmp_path, monkeypatch):
    """`flock-verify` is built from the Lean package alone: its proofs (`Security/Proofs/Flock/`) and any Lake package
    nested in it are no input, unless the package's lakefile requires something, which keeps everything in.  Nor is the
    package's `lean-audit.json`, which only the audit reads: that one path, not every file of the name."""
    files = CHECK.tracked_files()
    closure = CHECK.agreement_closure(files)
    policy, other = "verity/core/service/flock/lean-audit.json", "verity/core/service/flock/Flock/lean-audit.json"
    assert policy not in CHECK.agreement_closure([*files, policy]) and other in CHECK.agreement_closure([*files, other])
    assert "verity/ml/flock/verifier/ci.py" in closure and "verity/core/service/flock/FlockVerify.lean" in closure
    assert "verity/ml/flock/verifier/upstream.json" in closure
    assert not [f for f in closure if f.startswith("Security/")]
    assert [f for f in files if f.startswith("Security/Proofs/Flock/")]
    nested = ["verity/core/service/flock/lakefile.toml", "verity/core/service/flock/nested/lakefile.toml",
              "verity/core/service/flock/nested/N.lean"]
    (tmp_path / "verity/core/service/flock/nested").mkdir(parents=True)
    (tmp_path / nested[0]).write_text("")
    (tmp_path / nested[1]).write_text("")
    monkeypatch.setattr(CHECK, "ROOT", tmp_path)
    assert CHECK.agreement_closure(nested) == nested[:1]
    (tmp_path / nested[0]).write_text('[[require]]\nname = "x"\n')
    assert CHECK.agreement_closure(nested) == nested


def test_the_agreement_refuses_a_machine_it_cannot_run_on_in_seconds(tmp_path, monkeypatch):
    """Train E2's agreement disagreed on every session for an hour: the pinned binaries needed GLIBC_2.39 and the machine had 2.35.
    Too little memory, an older glibc than the pin names, and a binary the loader refuses each fail the step at once, by name."""
    ok = tmp_path / "ok"
    ok.write_text("#!/bin/sh\nexit 2\n")
    bad = tmp_path / "bad"
    bad.write_text("#!/bin/sh\necho \"./bad: /lib/x86_64-linux-gnu/libc.so.6: version \\`GLIBC_2.39' not found\" >&2\nexit 1\n")
    for p in (ok, bad):
        p.chmod(0o755)
    monkeypatch.setattr(CHECK, "memory_gb", lambda: 64)
    assert CHECK.agreement_preflight({"upstream": {"glibc": "2.17"}}, [ok]) is None
    assert "GLIBC_2.39' not found" in CHECK.agreement_preflight({"upstream": {}}, [ok, bad])
    assert "needs glibc 99.1" in CHECK.agreement_preflight({"upstream": {"glibc": "99.1"}}, [ok])
    monkeypatch.setattr(CHECK, "memory_gb", lambda: 16)
    assert "may use 16 GB; the agreement needs 24" in CHECK.agreement_preflight({"upstream": {}}, [ok])


def _fake_run(monkeypatch, groups, pre=()):
    """`run` over the given groups and preflight, with nothing built first, no Lean URLs, and no plan of the real tree for
    the reused pack (`Reused.plan` reads every file, outside the suite's inputs)."""
    monkeypatch.setattr(CHECK, "PREBUILD", "pass")
    monkeypatch.setattr(CHECK.Reused, "plan", lambda self: {"parts": []})
    monkeypatch.setattr(CHECK, "preflight", lambda: list(pre))
    monkeypatch.setattr(CHECK, "groups", lambda out, n, cache=True, urls=None, **kw: groups)
    monkeypatch.delenv(CHECK.LEAN_DEPS_URLS_ENV, raising=False)


def _gate(tmp_path, monkeypatch, writer: str = "slow") -> Path:
    """A FIFO that ``writer`` writes once it runs, and that the test writes itself when ``writer``'s step ends, so a step
    reading it is released even if ``writer`` died first: otherwise that read, and the run, would never end."""
    import os

    gate, fds, end = tmp_path / "gate", [], CHECK.Steps.end
    os.mkfifo(gate)

    def ended(self, rec):
        if rec["step"] == writer:
            fds.append(fd := os.open(gate, os.O_RDWR | os.O_NONBLOCK))
            os.write(fd, b"x")
        return end(self, rec)

    monkeypatch.setattr(CHECK.Steps, "end", ended)
    return gate


def _py(src: str, *args) -> list[str]:
    import sys

    return [sys.executable, "-c", src, *map(str, args)]


def test_a_failed_step_stops_the_other_groups_and_each_status_is_written_as_it_ends(tmp_path, monkeypatch):
    """`fails` exits only once `slow` has written to a FIFO, so `slow` is running when it fails: `slow` is interrupted (SIGINT),
    its group starts nothing more, and steps.jsonl has `fails` before `slow`, each line written as its step ended."""
    import json

    gate, marker = _gate(tmp_path, monkeypatch), tmp_path / "never-ran"
    slow = _py("import signal, sys\nwith open(sys.argv[1], 'w') as f:\n    f.write('x')\nsignal.pause()", gate)
    fails = _py("import sys\nopen(sys.argv[1]).read(1)\nsys.exit(3)", gate)
    _fake_run(monkeypatch, [[("slow", slow, tmp_path), ("after-slow", _py("open(__import__('sys').argv[1], 'w')", marker), tmp_path)],
                            [("fails", fails, tmp_path)]])
    ok, done = CHECK.run(tmp_path, 2)
    by = {s["step"]: s for s in done}
    assert not ok and by["fails"]["status"] == "failed" and by["fails"]["rc"] == 3
    assert by["slow"]["status"] == "stopped" and by["slow"]["detail"] == "stopped: fails failed"
    assert "after-slow" not in by and not marker.exists()
    lines = [json.loads(line) for line in (tmp_path / CHECK.STEPS_LOG).read_text().splitlines()]
    assert [(r["step"], r["status"]) for r in lines] == [("fails", "failed"), ("slow", "stopped")] and all("ended_utc" in r for r in lines)


def test_keep_going_runs_every_other_group_to_its_end(tmp_path, monkeypatch):
    """The failing step's own group still stops; the others keep their passing steps (and the verdicts those store)."""
    ok = _py("pass")
    _fake_run(monkeypatch, [[("fails", _py("raise SystemExit(3)"), tmp_path), ("after-fails", ok, tmp_path)],
                            [("a", ok, tmp_path), ("b", ok, tmp_path)]])
    passed, done = CHECK.run(tmp_path, 2, keep_going=True)
    assert not passed and {s["step"]: s["status"] for s in done} == {"fails": "failed", "a": "passed", "b": "passed"}


def test_a_step_killed_by_a_sigterm_the_check_did_not_send_is_stopped_not_failed(tmp_path, monkeypatch):
    """r20261026-062546-11fc: drive's `research cancel` signalled the run's whole process group, lean-agreement died of it before
    check's own handler marked it stopped, and it was recorded failed with an empty log, a failure the agreement never found."""
    ok = _py("pass")
    _fake_run(monkeypatch, [[("cancelled", _py("import os, signal\nos.kill(os.getpid(), signal.SIGTERM)"), tmp_path)],
                            [("a", ok, tmp_path)]])
    passed, done = CHECK.run(tmp_path, 2, keep_going=True)
    by = {s["step"]: s for s in done}
    assert not passed and by["a"]["status"] == "passed"
    assert by["cancelled"]["status"] == "stopped" and by["cancelled"]["rc"] == -15 and "did not send" in by["cancelled"]["detail"]


def _gone(pid: int) -> None:
    """Return once ``pid`` has exited (its pidfd is readable), or at once when it is gone already."""
    import os
    import select

    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    select.select([fd], [], [])
    os.close(fd)


def test_a_step_that_hangs_writing_nothing_is_stopped_and_fails_as_hung(tmp_path, monkeypatch):
    """r20261025-021824-ba9f and r20261025-023312-fa65: under --keep-going lean-suites failed, and pytest's suites.py
    sat 23-24 h on a test that never returned, writing nothing, while circuit-check waited on the grant pytest's end writes and
    the run held its slot.  Here pytest's process waits on a child that never exits; once the watch's clock (injected) has passed
    `IDLE_S` with pytest's outputs unchanged, the child and pytest are stopped, pytest fails as hung, the grant is written and the
    run ends."""
    import json
    import os

    gate = tmp_path / "child.pid"
    os.mkfifo(gate)
    hang = _py("import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', 'import os, signal, sys\\n"
               "with open(sys.argv[1], \"w\") as f:\\n    f.write(str(os.getpid()))\\nsignal.pause()', sys.argv[1]]).wait()", gate)
    clock, child = [0.0], []

    def wait(p, seconds):
        if p.args is not hang:
            return p.wait()
        if not child:
            child.append(int(gate.read_text()))
        clock[0] += seconds
        return p.poll()

    monkeypatch.setattr(CHECK, "CLOCK", lambda: clock[0])
    monkeypatch.setattr(CHECK, "WAIT", wait)
    monkeypatch.setenv(CHECK.SLOT.SLOT_MEM_ENV, "560")
    _fake_run(monkeypatch, [[("pytest", hang, tmp_path)], [("circuit-check", _py("pass"), tmp_path)],
                            [("lean-suites", _py("raise SystemExit(1)"), tmp_path)]])
    passed, done = CHECK.run(tmp_path, 3, keep_going=True)
    _gone(child[0])
    by = {s["step"]: s for s in done}
    assert not passed and by["lean-suites"]["status"] == "failed" and by["circuit-check"]["status"] == "passed"
    assert by["pytest"]["status"] == "failed" and by["pytest"]["detail"] == f"hung: no output for {CHECK.IDLE_S // 60} min"
    assert by["pytest"]["rc"] == -15 and clock[0] == CHECK.IDLE_S
    assert (tmp_path / CHECK.CC_GRANT).read_text() == "560\n"
    lines = {r["step"]: r for r in map(json.loads, (tmp_path / CHECK.STEPS_LOG).read_text().splitlines())}
    assert lines["pytest"]["detail"].startswith("hung: ")


def test_a_step_is_hung_only_once_its_outputs_have_been_quiet_for_idle_s_and_over_budget_at_step_budget_s(tmp_path, monkeypatch):
    """pytest writes its suites' logs (`WATCH`), circuit-check its progress (a file named after it): a step whose outputs grew at
    each of its first 100 looks is hung `IDLE_S` after the last of them, and one that writes at every look is stopped at
    `STEP_BUDGET_S`, whatever it writes."""
    import subprocess

    def stopped(name: str, wrote: str, writes: int) -> tuple[str | None, float]:
        clock, f = [0.0], tmp_path / wrote

        def wait(p, seconds):
            clock[0] += seconds
            if clock[0] <= writes * seconds:
                with f.open("a") as w:
                    w.write("x")
            return p.poll()

        monkeypatch.setattr(CHECK, "CLOCK", lambda: clock[0])
        monkeypatch.setattr(CHECK, "WAIT", wait)
        f.parent.mkdir(exist_ok=True)
        p = subprocess.Popen(_py("import signal\nsignal.pause()"))
        rc, why = CHECK.watch(name, p, tmp_path)
        assert rc == -15
        return why, clock[0]

    every = CHECK.WATCH_EVERY_S
    assert stopped("pytest", "suites/a_suite.log", 100) == (f"hung: no output for {CHECK.IDLE_S // 60} min", 100 * every + CHECK.IDLE_S)
    assert stopped("circuit-check", "circuit-check.json.progress.jsonl", 100)[1] == 100 * every + CHECK.IDLE_S
    assert stopped("pytest", "suites/a_suite.log", 10**6) == (f"over budget: still running after {CHECK.STEP_BUDGET_S / 3600:.1f} h",
                                                            CHECK.STEP_BUDGET_S)


def test_a_lean_module_that_does_not_compile_stops_the_run_even_under_keep_going(tmp_path, monkeypatch):
    """`lean-changed` exits only once `slow` is running (the FIFO): under --keep-going its failure still stops `slow`."""
    gate = _gate(tmp_path, monkeypatch)
    slow = _py("import signal, sys\nwith open(sys.argv[1], 'w') as f:\n    f.write('x')\nsignal.pause()", gate)
    changed = _py("import sys\nopen(sys.argv[1]).read(1)\nsys.exit(1)", gate)
    _fake_run(monkeypatch, [[("slow", slow, tmp_path)], [("lean-changed", changed, tmp_path)]])
    passed, done = CHECK.run(tmp_path, 2, keep_going=True)
    by = {s["step"]: s for s in done}
    assert not passed and by["lean-changed"]["status"] == "failed"
    assert by["slow"]["status"] == "stopped" and by["slow"]["detail"] == "stopped: lean-changed failed"


def test_a_rust_crate_that_does_not_compile_stops_the_run_even_under_keep_going(tmp_path, monkeypatch):
    """flock-circuit-build fails, with cargo's line, only once `slow` is running (the FIFO): under --keep-going `slow` stops."""
    gate = _gate(tmp_path, monkeypatch)
    slow = _py("import signal, sys\nwith open(sys.argv[1], 'w') as f:\n    f.write('x')\nsignal.pause()", gate)
    build = _py("import sys\nopen(sys.argv[1]).read(1)\nprint('error: could not compile `flock-circuit` (lib)')\nsys.exit(101)", gate)
    _fake_run(monkeypatch, [[("slow", slow, tmp_path)], [("flock-circuit-build", build, tmp_path)]])
    passed, done = CHECK.run(tmp_path, 2, keep_going=True)
    by = {s["step"]: s for s in done}
    assert not passed and by["flock-circuit-build"]["status"] == "failed" and by["flock-circuit-build"]["halts"]
    assert by["slow"]["status"] == "stopped" and by["slow"]["detail"] == "stopped: flock-circuit-build failed"


def test_a_build_that_fails_without_compiling_does_not_halt(tmp_path, monkeypatch):
    """A failed clone or toolchain download: no `halts`, so (the Steps test below) the other groups keep their verdicts."""
    build = _py("print('fatal: unable to access https://github.com/...')\nraise SystemExit(128)")
    _fake_run(monkeypatch, [[("flock-circuit-build", build, tmp_path)]])
    passed, (rec,) = CHECK.run(tmp_path, 1, keep_going=True)
    assert not passed and rec["status"] == "failed" and "halts" not in rec


def test_a_cancelled_recorded_check_still_publishes_the_passes_its_steps_stored(tmp_path, monkeypatch):
    """`research cancel`'s SIGTERM while `slow` runs (the FIFO): `slow` is stopped, `after-slow` never starts, and the run still
    writes its result, `not_run`, with the pack of the pass `passes` stored before it, and exits 143.  Before, the signal killed
    check.py, and a cancelled check's passes (6f10's 26 minutes, 12 Oct) were run again."""
    import json
    import os
    import signal
    import tarfile
    import threading

    gate, marker = _gate(tmp_path, monkeypatch), tmp_path / "never-ran"
    run_dir, cache = tmp_path / "run", tmp_path / "tests-cache"
    run_dir.mkdir()
    for k, v in (("RESEARCH_RUN_DIR", run_dir), ("RESEARCH_RUN_ID", "r20261012-000000-abcd"), ("VERITY_TEST_CACHE", cache),
                 ("CHECK_CACHE", tmp_path / "check-cache")):
        monkeypatch.setenv(k, str(v))
    monkeypatch.setattr(CHECK.SUITES, "foreign", lambda members, env: None)
    passes = _py("import pathlib, sys\np = pathlib.Path(sys.argv[1])\np.parent.mkdir(parents=True)\np.write_text('{}')",
                 cache / "pkg" / "k1.json")
    slow = _py("import signal, sys\nwith open(sys.argv[1], 'w') as f:\n    f.write('x')\nsignal.pause()", gate)
    _fake_run(monkeypatch, [[("passes", passes, tmp_path), ("slow", slow, tmp_path),
                             ("after-slow", _py("open(__import__('sys').argv[1], 'w')", marker), tmp_path)]])

    def cancel():
        with open(gate) as f:
            f.read(1)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=cancel, daemon=True).start()
    assert CHECK.main([]) == 143
    assert signal.getsignal(signal.SIGTERM) is not CHECK._terminated
    doc = json.loads((run_dir / "result.json").read_text())
    by = {s["step"]: s for s in doc["steps"]}
    assert by["passes"]["status"] == "passed" and by["slow"]["status"] == "stopped"
    assert by["slow"]["detail"] == "stopped: check got SIGTERM" and "after-slow" not in by and not marker.exists()
    assert doc["validation"]["status"] == "not_run" and doc["validation"]["detail"].startswith("stopped by SIGTERM: ")
    assert "verdicts.tar.gz" in doc["artifacts"] and doc["verdicts"]["out"]["entries"] == 1
    with tarfile.open(run_dir / "verdicts.tar.gz") as tar:
        assert tar.getnames() == ["tests/pkg/k1.json"]


def test_a_runs_pack_is_written_as_each_step_ends_and_on_sigterm_before_any_step_stops(tmp_path, monkeypatch):
    """A cancel SIGKILLs the run 20 s after its SIGTERM, before a stopped step's grace is out and `main` writes its last pack:
    drive's cancels after landing (rc 137) left packs without the passes the run had stored.  So the pack holds a step's
    passes once it ends, and `_terminated` writes what the steps stored since at once, from the pack's own thread."""
    import queue
    import signal
    import tarfile
    import weakref

    cache = tmp_path / "tests-cache"
    monkeypatch.setenv("VERITY_TEST_CACHE", str(cache))
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "check-cache"))
    monkeypatch.setattr(CHECK, "TERMINATED", [])
    monkeypatch.setattr(CHECK, "_LIVE", weakref.WeakSet())
    monkeypatch.setattr(CHECK, "_PACKS", weakref.WeakSet())
    (cache / "pkg").mkdir(parents=True)
    (cache / "pkg" / "old.json").write_text("{}")
    wrote: queue.Queue = queue.Queue()

    class Seen(CHECK.Packs):
        def write(self):
            wrote.put(n := super().write())
            return n

    pack = tmp_path / "run" / "verdicts.tar.gz"
    packs = Seen(pack, CHECK.VERDICTS.snapshot())
    steps = CHECK.Steps(tmp_path, packs=packs)
    (cache / "pkg" / "k1.json").write_text("{}")
    steps.end({"step": "passes", "status": "passed"})
    assert wrote.get() == 1
    with tarfile.open(pack) as tar:
        assert tar.getnames() == ["tests/pkg/k1.json"], "only what the run stored"
    (cache / "pkg" / "k2.json").write_text("{}")
    CHECK._terminated(signal.SIGTERM, None)
    assert wrote.get() == 2
    with tarfile.open(pack) as tar:
        assert tar.getnames() == ["tests/pkg/k1.json", "tests/pkg/k2.json"]
    stamp = pack.stat().st_mtime_ns
    assert packs.close() == 2 and pack.stat().st_mtime_ns == stamp, "a pack holding what is new already isn't written again"
    assert not packs.thread.is_alive()


def test_a_runs_reused_pack_holds_what_its_own_plan_found_here_before_its_steps_ran(tmp_path, monkeypatch):
    """A check's `verdicts.tar.gz` holds only what its steps stored, so a check reusing a pass no recent pack holds answered for
    nothing a later plan could cite (19 Oct: the newest packs held no circuit-check target and no keyed step).  `Reused` packs
    the passes its plan of its own tree found in its own caches, of those here before its steps ran; with circuit-check's
    targets unkeyed (no task list yet) it plans again once that step ends."""
    import tarfile

    cache = tmp_path / "check-cache"
    monkeypatch.setenv("CHECK_CACHE", str(cache))
    monkeypatch.setenv("VERITY_TEST_CACHE", str(tmp_path / "tests-cache"))
    for rel in ("lean-audit/k1.json", "circuit-check/v1/aa/e1.json", "circuit-check/v1/bb/l1.json"):
        (cache / rel).parent.mkdir(parents=True, exist_ok=True)
        (cache / rel).write_text("{}")
    before = CHECK.VERDICTS.snapshot()
    (cache / "lean-audit" / "k9.json").write_text("{}")                  # stored by this run: verdicts.tar.gz's
    plans = [
        {"parts": [{"part": "lean-audit:p", "state": "passed", "found": [{"source": "own", "paths": ["check/lean-audit/k1.json"]},
                                                                         {"source": "pack:r1", "paths": ["check/lean-audit/k1.json"]}]},
                   {"part": "lean-audit:q", "state": "passed", "found": [{"source": "own", "paths": ["check/lean-audit/k9.json"]}]},
                   {"part": "circuit-check", "state": "unkeyed"}]},
        {"parts": [{"part": "circuit-check:t", "state": "passed", "found": [
            {"source": "own", "paths": ["check/circuit-check/v1/aa/e1.json", "check/circuit-check/v1/bb/l1.json"]}]}]}]
    monkeypatch.setattr(CHECK.Reused, "plan", lambda self: plans.pop(0))
    run_dir = tmp_path / "run"
    reused = CHECK.Reused(run_dir, before)
    reused.start()
    reused.close()
    assert reused.circuit_unkeyed
    with tarfile.open(run_dir / CHECK.REUSED_PACK) as tar:
        assert tar.getnames() == ["check/lean-audit/k1.json"]
    reused.ended({"step": "circuit-check", "status": "passed"})
    assert reused.close() == 2 and not plans
    with tarfile.open(run_dir / CHECK.REUSED_PACK) as tar:
        assert tar.getnames() == ["check/circuit-check/v1/aa/e1.json", "check/circuit-check/v1/bb/l1.json"]


def test_under_keep_going_only_a_halting_step_stops_the_run_and_it_is_named(tmp_path):
    steps = CHECK.Steps(tmp_path, keep_going=True)
    steps.end({"step": "pytest", "status": "failed"})
    steps.end({"step": "flock-circuit-build", "status": "failed"})
    assert not steps.stopping
    steps.end({"step": "lean-changed", "status": "failed"})
    assert steps.stopping and steps.halted == "lean-changed"
    steps = CHECK.Steps(tmp_path)
    steps.end({"step": "pytest", "status": "failed"})
    assert steps.stopping and steps.halted == "pytest"


def test_the_lean_audit_step_refuses_a_fast_path_result(tmp_path):
    """A lean-audit step that exits 0 passes only on an `audit.json` whose every package had the full audit: one with the kernel
    replay. A fast-path result (`lean_changed.py --records`, of kind `FAST_KIND`, or any pass without the replay) fails it, and
    so does an `audit.json` that can't be read."""
    import json

    kind = CHECK._load("lean_changed", CHECK.LEAN_JOBS).FAST_KIND
    write = "import pathlib, sys\np = pathlib.Path(sys.argv[1])\np.parent.mkdir(exist_ok=True)\np.write_text(sys.argv[2])"
    full = {"package": "a", "failures": [], "seconds": {"build": 1.0, "audit": 1.0, "replay": 1.0}}

    def step(doc: dict | None) -> dict:
        (tmp_path / "lean-audit" / "audit.json").unlink(missing_ok=True)
        argv = _py(write, tmp_path / "lean-audit" / "audit.json", json.dumps(doc)) if doc is not None else _py("pass")
        return CHECK._step("lean-audit", argv, tmp_path, tmp_path, True, CHECK.Steps(tmp_path, keep_going=True))

    assert step({"passed": True, "failed": [], "packages": [full]})["status"] == "passed"
    for b in ({"package": "b", "failures": [], "seconds": {"build": 1.0, "audit": 1.0}},
              {**full, "package": "b", "kind": kind}):
        rec = step({"passed": True, "failed": [], "packages": [full, b]})
        assert rec["status"] == "failed" and rec["rc"] == 0 and rec["detail"].startswith(f"refused: b: a {kind} result"), rec
    rec = step({"passed": True, "failed": [], "kind": kind, "packages": []})
    assert rec["status"] == "failed" and f"a {kind} result" in rec["detail"]
    rec = step(None)
    assert rec["status"] == "failed" and rec["detail"].startswith("refused: no readable")
    for bad in ([], {"packages": {"a": full}}, {"packages": ["a"]}):     # JSON, not an audit's report: it crashed check
        rec = step(bad)
        assert rec["status"] == "failed" and "is not an audit's report" in rec["detail"], rec
    rec = step({"passed": True, "failed": [], "packages": [{**full, "seconds": ["replay"]}]})
    assert rec["status"] == "failed" and rec["detail"].startswith(f"refused: a: a {kind} result"), rec


def test_the_early_lean_build_is_a_group_of_its_own_beside_the_lean_steps(tmp_path, monkeypatch):
    out, _ = _pinned(tmp_path, monkeypatch)
    gs = CHECK.groups(out, 8)
    assert len(gs) == 7 and [n for n, _, _ in gs[4]] == ["lean-changed"] and "lean-changed" in CHECK.HALTS
    argv, audit = gs[4][0][1], gs[3][0][1]
    assert argv[1].endswith("tools/verity/lean/lean_changed.py")
    assert argv[argv.index("--cache") + 1] == audit[audit.index("--cache") + 1], "the audit's own passes"
    assert argv[argv.index("--deps") + 1] == audit[audit.index("--deps") + 1], "and its kept dependencies"
    assert CHECK.groups(out, 8, cache=False)[4][0][1].startswith("skipped: --no-cache")
    monkeypatch.setattr(CHECK, "FLOCK_LEAN", tmp_path / "no-verifier")
    assert len(CHECK.groups(out, 8)) == 4


def test_circuit_cache_keeps_circuit_checks_reuse_alone_under_no_cache(tmp_path, monkeypatch):
    """Drive's after-landing run (`--no-cache --circuit-cache`): the suites and the Lean steps run cold, and circuit-check reuses
    its per-target passes, since a cold `--all` (~4 h) outlasted every after-landing run from 24 Oct.  `--no-cache` alone (the
    nightly `audit-main`) leaves circuit-check cold too."""
    out, _ = _pinned(tmp_path, monkeypatch)

    def steps(**kw) -> dict:
        return {name: argv for g in CHECK.groups(out, 8, **kw) for name, argv, _ in g}

    cached, cold, after = steps(), steps(cache=False), steps(cache=False, circuit_cache=True)
    cc = ["--cache", str(CHECK.cache_dir() / "circuit-check"), "--catalog-units"]
    assert " ".join(cc) in " ".join(cached["circuit-check"]) and " ".join(cc) in " ".join(after["circuit-check"])
    assert "--cache" not in cold["circuit-check"]
    for s in ("pytest", "lean-suites", "lean-audit"):
        assert "--fresh" in " ".join(after[s]) and "--fresh" not in " ".join(cached[s]), s
    assert after["lean-changed"].startswith("skipped: --no-cache")


def test_the_preflight_runs_first_and_a_failure_there_starts_nothing_else(tmp_path, monkeypatch):
    marker = tmp_path / "ran"
    touch = _py("open(__import__('sys').argv[1], 'w')", marker)
    _fake_run(monkeypatch, [[("pytest", touch, tmp_path)]],
              pre=[("preflight-lock", _py("raise SystemExit(1)"), tmp_path), ("preflight-lints", touch, tmp_path)])
    ok, done = CHECK.run(tmp_path, 2)
    assert not ok and [(s["step"], s["status"]) for s in done] == [("preflight-lock", "failed"), ("preflight-lints", "stopped")]
    assert done[1]["detail"] == "not started: preflight-lock failed" and not marker.exists()
    _fake_run(monkeypatch, [[("pytest", touch, tmp_path)]], pre=[("preflight-lock", _py("pass"), tmp_path)])
    assert CHECK.run(tmp_path, 2)[0] and marker.exists()


def test_a_failed_preflight_starts_nothing_else_even_under_keep_going(tmp_path, monkeypatch):
    """Every train's check runs with --keep-going: a failed preflight must still end it in seconds, not hold its slot."""
    marker = tmp_path / "ran"
    touch = _py("open(__import__('sys').argv[1], 'w')", marker)
    for failing in ("preflight-lock", "preflight-lints"):
        monkeypatch.setenv("CHECK_CACHE", str(tmp_path / f"cache-{failing}"))     # else the first's lints pass is reused
        pre = [(n, _py("raise SystemExit(1)") if n == failing else _py("pass"), tmp_path) for n in ("preflight-lock", "preflight-lints")]
        _fake_run(monkeypatch, [[("pytest", touch, tmp_path)]], pre=pre)
        ok, done = CHECK.run(tmp_path, 2, keep_going=True)
        assert not ok and "pytest" not in {s["step"] for s in done} and not marker.exists(), failing
        assert {s["step"]: s["status"] for s in done}[failing] == "failed"


def test_the_preflight_is_the_lockfile_and_the_repository_lints(monkeypatch):
    """Without bytecode or pytest's cache; every lint it names exists, so a rename can't turn it into a failure of
    every check. Without uv, the lockfile step fails by name."""
    (lock, argv, cwd), (lints, pytest, cwd2), (fixtures, public, cwd3) = CHECK.preflight()
    assert (lock, argv[1:], cwd) == ("preflight-lock", ["lock", "--check"], ROOT)
    assert (fixtures, public[1:], cwd3) == ("preflight-fixtures", ["-B", str(ROOT / "tools" / "verity" / "check" / "check.py"),
                                                                   "--fixtures"], ROOT)
    assert (lints, pytest[1:4], cwd2) == ("preflight-lints", ["-B", "-m", "pytest"], ROOT) and "no:cacheprovider" in pytest
    assert pytest[-len(CHECK.LINTS):] == list(CHECK.LINTS)
    for t in CHECK.LINTS:
        path, _, name = t.partition("::")
        assert (ROOT / path).is_file() and (not name or f"def {name}(" in (ROOT / path).read_text())
    monkeypatch.delenv("UV", raising=False)
    monkeypatch.setattr(CHECK.shutil, "which", lambda _name: None)
    assert CHECK.preflight()[0][1].startswith("failed: no `uv` here")


def test_the_preflight_refuses_a_fetched_fixture_that_neither_this_machine_nor_the_public_export_has(tmp_path, monkeypatch, capsys):
    """The closing train's check (cb20, 10 Oct) failed `lean-suites` after its launch: #1716 registered rec-coins, preserved
    privately and never exported, and node 1 had it neither in its store nor from the public export. A fixture the tree already
    holds exactly or this machine's store holds whole needs no export (top, 10 Oct: infra's node-1 copies keep trains moving),
    though the export is asked first (one HEAD per manifest, no blob), so the ones it lacks are named. A tracked fixture
    and the `previous` history are not asked about, and with no public source to ask it refuses rather than pass."""
    import json
    import subprocess

    from research.store.fixtures import fixture_id

    good, gone, small, local = ("art:" + c * 64 for c in "abce")
    (tmp_path / "big").mkdir()
    (tmp_path / "big" / "there.bin").write_bytes(b"exact bytes")
    there = fixture_id(tmp_path / "big" / "there.bin")
    (tmp_path / "fixtures").mkdir()
    (tmp_path / "fixtures" / "artifacts.json").write_text(json.dumps(
        {good: "big/good.bin", gone: "big/gone.bin", small: "small.json", local: "big/local.bin", there: "big/there.bin",
         "previous": {good: ["art:" + "d" * 64]}}))
    (tmp_path / "small.json").write_text("{}")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "small.json", "fixtures"], check=True)
    asked = []

    def head(key):
        asked.append(key)
        return {"bytes": 1} if key == f"manifests/{'a' * 64}.json" else None

    class Store:
        def has_local(self, art):
            return art == local

    where, missing = CHECK.fetchable_fixtures(tmp_path, "https://public.example/store", head, Store())
    assert missing == [gone]
    assert where == {good: "the public export", local: "this machine's store", there: "the tree"}
    assert sorted(asked) == sorted(f"manifests/{a.removeprefix('art:')}.json" for a in (good, gone, local, there))
    with pytest.raises(RuntimeError, match="no public source"):
        CHECK.fetchable_fixtures(tmp_path, "", head, Store())

    class Broken:
        def has_local(self, art):
            raise OSError("no store")

    assert CHECK.fetchable_fixtures(tmp_path, "https://public.example/store", head, Broken())[1] == [gone, local]

    monkeypatch.setattr(CHECK, "fetchable_fixtures", lambda: ({local: "this machine's store"}, [gone]))
    assert CHECK.main(["--fixtures"]) == 1
    out = capsys.readouterr()
    assert gone in out.err and "1 from this machine's store" in out.err and f"{local} from this machine's store" in out.out
    monkeypatch.setattr(CHECK, "fetchable_fixtures", lambda: ({local: "this machine's store"}, []))
    assert CHECK.main(["--fixtures"]) == 0 and "1 from this machine's store" in capsys.readouterr().out


def test_preflight_alone_only_reports_a_step_that_judges_the_node_the_check_runs_on(tmp_path, monkeypatch, capsys):
    """A train's prep runs `--preflight` off the node its check is pinned to, so a fixture only that node's store holds (f1-gate
    on node 1, 10 Oct) mustn't refuse the train there: the check's own preflight-fixtures judges the node, and refuses on it."""
    fail = _py("raise SystemExit(1)")
    monkeypatch.setattr(CHECK, "preflight", lambda: [("preflight-fixtures", fail, tmp_path)])
    assert CHECK.main(["--preflight"]) == 0 and "not refused here" in capsys.readouterr().out
    monkeypatch.setattr(CHECK, "preflight", lambda: [("preflight-lints", fail, tmp_path), ("preflight-fixtures", fail, tmp_path)])
    assert CHECK.main(["--preflight"]) == 1


def test_preflight_alone_runs_every_step_and_fails_if_any_does(tmp_path, monkeypatch, capsys):
    """`check.py --preflight`, what `research merge --train --prepare` runs on a train before pushing it."""
    marker = tmp_path / "ran"
    touch = _py("open(__import__('sys').argv[1], 'w')", marker)
    monkeypatch.setattr(CHECK, "preflight", lambda: [("preflight-lock", "failed: no `uv` here", tmp_path), ("preflight-lints", touch, tmp_path)])
    assert CHECK.main(["--preflight"]) == 1 and marker.exists()
    assert capsys.readouterr().out.splitlines() == ["check: preflight-lock: failed: no `uv` here", "check: preflight-lints: passed"]
    monkeypatch.setattr(CHECK, "preflight", lambda: [("preflight-lints", touch, tmp_path)])
    assert CHECK.main(["--preflight"]) == 0


def test_a_contradicted_cached_pass_is_named_in_the_result(tmp_path, monkeypatch):
    import json

    from research import result as RES

    (tmp_path / "suites.json").write_text(json.dumps([{"suite": "pkg", "contradicts": ["tests/test_a.py::test_a"]}, {"suite": "b"}]))
    (tmp_path / "suites-lean.json").write_text(json.dumps([{"suite": "lean", "nondeterministic": ["tests/test_b.py::test_b"]}]))
    bad, flaky = CHECK.contradictions(tmp_path), CHECK.contradictions(tmp_path, "nondeterministic")
    assert bad == [{"suite": "pkg", "contradicts": ["tests/test_a.py::test_a"]}]
    assert flaky == [{"suite": "lean", "nondeterministic": ["tests/test_b.py::test_b"]}]
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20260929-010000-abcd")
    CHECK.write_result(tmp_path, False, [{"step": "pytest", "status": "failed", "rc": 1, "seconds": 1.0}], None, bad, flaky)
    r = RES.check(tmp_path, "r20260929-010000-abcd")
    assert r.valid, r.errors
    assert r.doc["cache_contradictions"] == bad and r.doc["nondeterministic"] == flaky


def test_record_sends_the_agreement_for_a_flock_file_whose_name_git_quotes(tmp_path, monkeypatch, capsys):
    """`git diff --name-only` quotes a name with bytes outside ASCII (`"verity/ml/flock/\\303\\251.py"`), which no prefix of the
    agreement's rule matched, so `--record` checked such a change without sending the agreement's pinned build."""
    import shutil
    import subprocess
    import sys
    import tempfile

    from research.store import local

    def git(*args: str) -> None:
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=tmp_path,
                       check=True, capture_output=True)

    (tmp_path / "tools" / "verity" / "check").mkdir(parents=True)
    shutil.copy(ROOT / "tools" / "verity" / "check" / "tool.py", tmp_path / "tools" / "verity" / "check" / "tool.py")
    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    (tmp_path / "verity" / "ml" / "flock").mkdir(parents=True)
    (tmp_path / "verity" / "ml" / "flock" / "é.py").write_text("x = 1\n")
    git("checkout", "-q", "-b", "pr")
    git("add", "-A")
    git("commit", "-q", "-m", "é")

    class Store:
        def fetch(self, art: str) -> Path:
            return tmp_path / art.replace(":", "_")

    real, launched = subprocess.run, []

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            launched.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    monkeypatch.setattr(CHECK, "ROOT", tmp_path)
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: None)
    monkeypatch.setattr(local, "LocalStore", Store)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", run)
    assert CHECK.record(["--base", "main"]) == 0
    assert "check: changes under verity/ml/flock/, verity/core/service/flock/ against main: sending the pinned upstream build and inputs" in capsys.readouterr().out
    [cmd] = launched
    assert any(x.startswith("flock_upstream=art:") for x in cmd)


def test_the_trees_lean_steps_run_rather_than_skip():
    """On this tree, wherever the layout puts the Lean verifier's package (`FLOCK_LEAN`), the Lean steps are the build, the
    unit-cut conformance, the audit and the agreement; a tree with the verifier but no package there fails rather than skips."""
    assert (CHECK.FLOCK_LEAN / "lakefile.toml").is_file()
    steps = CHECK.lean_steps(ROOT / "nowhere")
    assert [n for n, _, _ in steps] == ["lean-build", "lean-unit-cut", "lean-audit", "lean-agreement"]
    assert all(not isinstance(argv, str) for _, argv, _ in steps[:3])


def test_a_verifier_without_its_lean_package_fails_rather_than_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(CHECK, "FLOCK_LEAN", tmp_path / "moved-away")
    [(name, detail, _)] = CHECK.lean_steps(tmp_path)
    assert name == "lean" and isinstance(detail, str) and not detail.startswith("skipped:")


def test_a_change_to_the_lean_verifier_alone_needs_its_gates_and_misses_the_agreements_cache():
    """A file of the Lean verifier's own package, wherever the layout puts it (`FLOCK_LEAN`), needs the agreement to merge and
    is read by the agreement (so a change to it misses a cached pass): a rule that still named the package's old place would
    pass such a change unchecked."""
    from research.merge import rule_touches

    rel = CHECK.FLOCK_LEAN.relative_to(ROOT).as_posix()
    changed = [f"{rel}/Flock/Gate.lean"]
    spec2 = importlib.util.spec_from_file_location("check_tool_lean", ROOT / "tools" / "verity" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(tool)
    assert rule_touches(list(tool.CHECK.merge_requires["lean-agreement"]), changed)
    assert CHECK.agreement_closure(changed) == changed


def test_every_file_the_agreement_reads_needs_the_agreement_to_merge():
    """The rule `check.py --record` sends the upstream build by, and the gate refuses by, covers each tracked file the agreement
    reads (`agreement_closure`: its scripts and the Lean package `flock-verify` is built from) but the docs, so a change to any
    of them can't land on a check that skipped it (13 Oct: the rule was read as covering verity/ml/flock/ alone)."""
    from research.merge import rule_touches

    spec2 = importlib.util.spec_from_file_location("check_tool_reads", ROOT / "tools" / "verity" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(tool)
    rule = list(tool.CHECK.merge_requires["lean-agreement"])
    reads = CHECK.agreement_closure(CHECK.tracked_files(ROOT))
    lean = CHECK.FLOCK_LEAN.relative_to(ROOT).as_posix() + "/"
    assert any(f.startswith(lean) and f.endswith(".lean") for f in reads)
    # the rule leaves out the docs only, which the agreement never reads (`test_the_agreement_is_not_required_for_docs_alone`)
    assert {f.rsplit("/", 1)[-1] for f in reads if not rule_touches(rule, [f])} <= {"README.md", "PROTOCOL.md"}


def test_a_check_reports_per_step_what_it_reused_and_what_forced_each_rerun(tmp_path, monkeypatch, capsys):
    """`reuse`, at the top of result.json and printed in the log: a flag or input that turns reuse off shows as a low rate there,
    with each suite run again and the inputs that forced it."""
    import json

    (tmp_path / "suites.json").write_text(json.dumps([
        {"suite": "a", "status": "cached", "seconds": 0.0},
        {"suite": "b", "status": "passed", "seconds": 12.0, "forced_by": {"why": ["input b changed: b/x.py", "python 3.12 -> 3.14", "x", "y"]}},
        {"suite": "c", "status": "failed", "seconds": 40.0, "forced_by": None},
        {"suite": "d", "status": "skipped"}]))
    (tmp_path / "lean-audit").mkdir()
    (tmp_path / "lean-audit" / "audit.json").write_text(json.dumps({"cached": {"P": "k"}, "packages": [{"package": "P"}, {"package": "Q"}]}))
    done = [{"step": "pytest", "status": "passed"},
            {"step": "circuit-check", "status": "passed", "cache": {"hits": 7, "stored": 2, "untraced": 1, "errors": 0, "changed": 1}},
            {"step": "flock-rust-tests", "status": "passed", "cache": {"hit": "k"}},
            {"step": "lean-agreement", "status": "passed", "cache": {"stored": "k"}},
            {"step": "lean-audit", "status": "passed"}, {"step": "lean-build", "status": "passed"}]
    r = CHECK.reuse(tmp_path, done)
    assert r["steps"]["pytest"] == {"of": "suites", "reused": 1, "rerun": 2, "rate": 0.333, "reran": [
        {"suite": "c", "seconds": 40.0, "forced_by": ["not recorded"]},
        {"suite": "b", "seconds": 12.0, "forced_by": ["input b changed: b/x.py", "python 3.12 -> 3.14", "x"]}]}
    assert r["steps"]["circuit-check"]["not_stored"] == {"untraced": 1, "errors": 0, "changed": 1}
    assert r["steps"]["flock-rust-tests"]["reused"] == 1 and r["steps"]["lean-agreement"]["rerun"] == 1
    assert r["steps"]["lean-audit"] == {"of": "packages", "reused": 1, "rerun": 1, "reran": ["Q"], "rate": 0.5}
    assert "lean-build" not in r["steps"] and r["steps"]["circuit-check"]["rate"] == 0.636
    lines = CHECK.reuse_lines(r)
    assert lines[0] == ("check: reused pytest 1/3 suites, circuit-check 7/11 targets, flock-rust-tests 1/1 step, "
                        "lean-agreement 0/1 step, lean-audit 1/2 packages")
    assert "check: reuse     b (12.0 s): input b changed: b/x.py; python 3.12 -> 3.14; x" in lines

    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    CHECK.write_result(tmp_path, True, [{**s, "seconds": 1.0} for s in done])
    doc = json.loads((tmp_path / "result.json").read_text())
    assert list(doc)[:4] == ["schema", "kind", "run_id", "reuse"] and doc["reuse"] == r


def test_the_circuit_check_step_records_its_cache_as_counts(tmp_path):
    """result.json's steps are its attempt's meta, which the store caps at 256 KiB: a circuit-check cache listing every target
    it could not store (each one, once its workers forked) made a passing check's attempt unpublishable. The step keeps counts;
    the lists stay in circuit-check.json, an artifact."""
    import json

    cache = {"hits": 0, "stored": 3, "untraced": [{"target": f"T{i}", "because": ["os.fork "] * 3} for i in range(4000)],
             "errors": [{"target": "E"}], "changed": []}
    (tmp_path / "circuit-check.json").write_text(json.dumps({"cache": cache}))
    rec = CHECK._step("circuit-check", _py("pass"), tmp_path, tmp_path, False, CHECK.Steps(tmp_path, keep_going=True))
    assert rec["status"] == "passed" and rec["cache"] == {"hits": 0, "stored": 3, "untraced": 4000, "errors": 1, "changed": 0}
    assert CHECK.reuse(tmp_path, [rec])["steps"]["circuit-check"]["not_stored"] == {"untraced": 4000, "errors": 1, "changed": 0}


def test_train_sh_refuses_every_flag_but_agreement(tmp_path):
    """The lander launches checks only through train.sh, which runs the merge queue's command: a flag of its own (`--no-cache`,
    a stale `--verdicts-in`) would change what is reused or checked, so it is refused before anything is read."""
    import subprocess

    def train(*flags: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(ROOT / "tools" / "verity" / "check" / "train.sh"), "abc", "vy-x", *flags],
                              capture_output=True, text=True, cwd=tmp_path, env={"PATH": "/usr/bin:/bin"})

    for flag in ("--no-cache", "--no-regenerate", "--verdicts-in", "--cores"):
        p = train("--agreement", flag)
        assert p.returncode == 2 and f"train: {flag} refused" in p.stderr, p.stderr
    # drive's after-landing run of a review landing: every job, the upstream build, no reuse, no regenerated lock records; past
    # the flags, the store key
    p = train("--deferred", "pytest,lean-audit", "--agreement", "--no-cache", "--no-regenerate")
    assert p.returncode == 1 and "no parent store key" in p.stderr, p.stderr
    p = train("--deferred", "pytest,circuit-check", "--no-cache", "--circuit-cache", "--no-regenerate")
    assert p.returncode == 1 and "no parent store key" in p.stderr, p.stderr
    p = train("--deferred", "pytest,circuit-check", "--circuit-cache")
    assert p.returncode == 2 and "train: --circuit-cache refused without --no-cache" in p.stderr, p.stderr
    p = train("--against", "main")
    assert p.returncode == 2 and "train: --against refused" in p.stderr, p.stderr


@pytest.mark.parametrize("twins", [False, True])
def test_train_sh_prints_every_twins_run_id(tmp_path, twins):
    """With `auto` and no free slot, check.py launches one twin per machine and only one of them runs: train.sh prints every
    twin's run id, so the lander follows the one that runs rather than the first launched, which may exit 75 not started."""
    import os
    import subprocess

    def git(*a, cwd=tmp_path / "repo"):
        subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, env={**os.environ, **GIT_ENV})

    GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    git("init", "-q", "--bare", "-b", "main", str(tmp_path / "origin.git"), cwd=tmp_path)
    git("clone", "-q", str(tmp_path / "origin.git"), str(tmp_path / "repo"), cwd=tmp_path)
    git("commit", "-q", "--allow-empty", "-m", "c")
    git("push", "-q", "origin", "HEAD:main")
    bin_, done = tmp_path / "bin", tmp_path / "launched"
    bin_.mkdir()
    lines = (["check: --on: twins 20261009T050000Z-ab12 on vy-nebius-1, vy-nebius-2: the first to take a check slot runs"]
             if twins else []) + ["research: run r20261009-050001-aaaa launched on vy-nebius-1 (p) pid=1 -> /r"]
    lines += ["research: run r20261009-050002-bbbb launched on vy-nebius-2 (p) pid=2 -> /r"] if twins else []
    (bin_ / "uv").write_text("#!/bin/sh\n" + "".join(f"echo '{x}'\n" for x in lines) + f"touch {done}\n")
    (bin_ / "sleep").write_text(f"#!/bin/sh\nwhile [ ! -e {done} ]; do :; done\n")    # waits on the launch, not on a clock
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = {**os.environ, **GIT_ENV, "PATH": f"{bin_}:/usr/bin:/bin", "TMPDIR": str(tmp_path), "AWS_ACCESS_KEY_ID": "dummy",
           "R2_ENDPOINT": "dummy"}
    for k in ("VERITY_SKIP_STORE", "AWS_SESSION_TOKEN"):
        env.pop(k, None)
    p = subprocess.run(["bash", str(ROOT / "tools" / "verity" / "check" / "train.sh"), "HEAD", "auto"],
                       capture_output=True, text=True, cwd=tmp_path / "repo", env=env)
    want = "RUN r20261009-050001-aaaa r20261009-050002-bbbb" if twins else "RUN r20261009-050001-aaaa"
    assert p.returncode == 0 and p.stdout.splitlines()[-1] == want, (p.stdout, p.stderr)


#: the stubbed `uv`'s answer to train.sh's post to the merge thread: the post's body is copied to {posted}, and nothing launches
_POST_STUB = ('case "$*" in *"msg reply"*) f=; p=; for a; do [ "$p" = --file ] && f=$a; p=$a; done; cp "$f" {posted}; '
              'echo "$*" > {posted}.args; exit 0;; '
              'esac\n')


def _train_repo(tmp_path):
    """A clone of a local origin with a root commit on main, and train.sh's environment with `uv` and `sleep` stubbed: `uv` records
    its arguments in ``argv`` and prints a launch line, and `sleep` waits for that launch, not on a clock."""
    import os
    import subprocess

    git_env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}

    def git(*a, cwd=tmp_path / "repo"):
        return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True,
                              env={**os.environ, **git_env}).stdout.strip()

    git("init", "-q", "--bare", "-b", "main", str(tmp_path / "origin.git"), cwd=tmp_path)
    git("clone", "-q", str(tmp_path / "origin.git"), str(tmp_path / "repo"), cwd=tmp_path)
    (tmp_path / "repo" / "a").write_text("a\n")
    git("add", "a")
    git("commit", "-q", "-m", "root")
    git("push", "-q", "origin", "HEAD:main")
    bin_, done, argv = tmp_path / "bin", tmp_path / "launched", tmp_path / "argv"
    bin_.mkdir()
    (bin_ / "uv").write_text(f"#!/bin/sh\n{_POST_STUB.format(posted=tmp_path / 'posted')}printf '%s\\n' \"$@\" > {argv}\n"
                             "echo 'research: run r20261009-050001-aaaa launched on vy-x (p) pid=1 -> /r'\n" f"touch {done}\n")
    (bin_ / "sleep").write_text(f"#!/bin/sh\nwhile [ ! -e {done} ]; do :; done\n")
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = {**os.environ, **git_env, "PATH": f"{bin_}:/usr/bin:/bin", "TMPDIR": str(tmp_path), "AWS_ACCESS_KEY_ID": "dummy",
           "R2_ENDPOINT": "dummy"}
    for k in ("VERITY_SKIP_STORE", "AWS_SESSION_TOKEN"):
        env.pop(k, None)

    def train(*a, **extra):
        return subprocess.run(["bash", str(ROOT / "tools" / "verity" / "check" / "train.sh"), *a], capture_output=True, text=True,
                              cwd=tmp_path / "repo", env={**env, **extra})

    return git, train, argv


def test_train_sh_refuses_an_empty_commit_and_one_origin_lacks(tmp_path):
    """The hand-built launch of an empty commit (9 Oct) is what train.sh removes: it refuses, before launching anything, a commit
    whose tree is its parent's, and one no branch of origin holds (the run ships what origin holds)."""
    git, train, argv = _train_repo(tmp_path)
    git("commit", "-q", "--allow-empty", "-m", "empty")
    git("push", "-q", "origin", "HEAD:main")
    p = train("HEAD", "vy-x")
    assert p.returncode == 1 and "is empty (its tree is its base" in p.stderr, p.stderr
    posted = tmp_path / "posted"
    assert "registered no run (exit 1)" in posted.read_text() and "is empty (its tree is its base" in posted.read_text()

    (tmp_path / "repo" / "a").write_text("b\n")
    git("commit", "-q", "-am", "local")
    p = train("HEAD", "vy-x")
    assert p.returncode == 1 and "is on no branch of origin" in p.stderr, p.stderr
    assert not argv.exists()


def test_train_sh_runs_with_priority_and_posts_a_launch_that_registers_no_run(tmp_path):
    git, train, argv = _train_repo(tmp_path)
    (tmp_path / "repo" / "a").write_text("b\n")
    git("commit", "-q", "-am", "change")
    git("push", "-q", "origin", "HEAD:main")
    p = train("HEAD", "vy-x", "--agreement")
    args = argv.read_text().split()
    assert p.returncode == 0 and {"--record", "--priority", "--keep-going", "--agreement"} <= set(args), args
    p = train("HEAD", "vy-x", "--deferred", "pytest,circuit-check", "--no-cache", "--circuit-cache", "--no-regenerate")
    args = argv.read_text().split()
    assert p.returncode == 0 and args[args.index("--deferred"):] == ["--deferred", "pytest,circuit-check", "--no-cache",
                                                                     "--circuit-cache", "--no-regenerate"], args

    (tmp_path / "origin.git").rename(tmp_path / "gone.git")
    (tmp_path / "bin" / "sleep").write_text("#!/bin/sh\n")                # the fetch's backoffs: nothing launches to wait for
    p = train("HEAD", "vy-x")
    assert p.returncode == 1 and "git fetch origin failed" in (tmp_path / "posted").read_text()


def test_train_sh_retries_a_failed_first_fetch_twice_with_a_backoff(tmp_path):
    """Drive launches train.sh detached, so its fetch in the queue's clone can meet a tick's, which
    fails one of them with `cannot lock ref`.  It tries twice more, 5 s and then 15 s later, and only then registers no run."""
    git, train, argv = _train_repo(tmp_path)
    (tmp_path / "repo" / "a").write_text("b\n")
    git("commit", "-q", "-am", "change")
    git("push", "-q", "origin", "HEAD:main")
    origin, gone, sleeps = tmp_path / "origin.git", tmp_path / "gone.git", tmp_path / "sleeps"
    origin.rename(gone)
    # the first backoff gives origin back; a poll for the launch waits on it, not on a clock
    (tmp_path / "bin" / "sleep").write_text(f'#!/bin/sh\necho "$1" >> {sleeps}\nif [ -e {gone} ]; then mv {gone} {origin}; '
                                            f'else while [ ! -e {tmp_path / "launched"} ]; do :; done; fi\n')
    p = train("HEAD", "vy-x")
    assert p.returncode == 0 and p.stdout.splitlines()[-1] == "RUN r20261009-050001-aaaa", (p.stdout, p.stderr)
    assert sleeps.read_text().split()[0] == "5"
    sleeps.unlink()
    origin.rename(gone)
    (tmp_path / "bin" / "sleep").write_text(f'#!/bin/sh\necho "$1" >> {sleeps}\n')
    p = train("HEAD", "vy-x")
    assert p.returncode == 1 and "git fetch origin failed" in p.stderr and sleeps.read_text().split() == ["5", "15"], p.stderr


def test_train_sh_takes_a_restack_by_merge_head_and_refuses_one_with_mains_tree(tmp_path):
    """#1585's head 13df320192c2 merged main into a commit that already held main's change: its tree is its first parent's, yet
    it differs from main, and train.sh refused it as empty (22:23Z, 9 Oct). Empty means the tree is main's, where it leaves main."""
    git, train, argv = _train_repo(tmp_path)
    root = git("rev-parse", "HEAD")
    (tmp_path / "repo" / "m").write_text("m\n")
    git("add", "m")
    git("commit", "-q", "-m", "main")
    git("push", "-q", "origin", "HEAD:main")
    for branch, files in (("pr", ("m", "p")), ("same", ("m",))):
        git("checkout", "-q", "-b", branch, root)
        for f in files:
            (tmp_path / "repo" / f).write_text(f"{f}\n")
            git("add", f)
        git("commit", "-q", "-m", branch)
        git("merge", "-q", "--no-edit", "origin/main")
        git("push", "-q", "origin", branch)
    head = git("rev-parse", "pr")
    assert git("rev-parse", "pr^{tree}") == git("rev-parse", "pr^1^{tree}") != git("rev-parse", "origin/main^{tree}")
    p = train(head, "vy-x")
    assert p.returncode == 0 and p.stdout.splitlines()[-1] == "RUN r20261009-050001-aaaa", (p.stdout, p.stderr)
    argv.unlink()

    assert git("rev-parse", "same^{tree}") == git("rev-parse", "origin/main^{tree}")
    p = train(git("rev-parse", "same"), "vy-x")
    assert p.returncode == 1 and "is empty (its tree is its base" in p.stderr, p.stderr
    assert not argv.exists()


def test_train_sh_stops_a_launcher_that_launched_nothing_in_the_wait(tmp_path):
    """18 Oct: train.sh gave up on a check.py still gathering verdicts after its wait, and check.py launched its runs later. Drive
    had no ids for them, so nothing cancelled them, and one held a check slot 3 h on a tree no longer on `next`."""
    import os

    git, train, _ = _train_repo(tmp_path)
    bin_, pidf, stopped = tmp_path / "bin", tmp_path / "pid", tmp_path / "stopped"
    (bin_ / "uv").write_text(f"#!/bin/sh\n{_POST_STUB.format(posted=tmp_path / 'posted')}echo $$ > {pidf}.tmp\nmv {pidf}.tmp {pidf}\n"
                             f"trap 'touch {stopped}; exit 143' TERM\nwhile :; do :; done\n")
    (bin_ / "sleep").write_text(f"#!/bin/sh\nuntil [ -e {pidf} ]; do :; done\n")       # each poll waits on the launch, not on a clock
    p = train("HEAD", "vy-x")
    assert p.returncode != 0 and "stopped its launcher" in p.stderr, (p.stdout, p.stderr)
    assert stopped.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidf.read_text()), 0)


def test_train_sh_returns_the_exit_of_a_check_that_died_before_launching(tmp_path):
    """check.py killed before it launches a run (the OOM killer, while it gathers verdicts) prints no traceback: train.sh says
    how it exited and returns that, at once, rather than polling the log for ten minutes and saying only that nothing launched."""
    import os
    import subprocess

    GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}

    def git(*a, cwd=tmp_path / "repo"):
        subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, env={**os.environ, **GIT_ENV})

    git("init", "-q", "--bare", "-b", "main", str(tmp_path / "origin.git"), cwd=tmp_path)
    git("clone", "-q", str(tmp_path / "origin.git"), str(tmp_path / "repo"), cwd=tmp_path)
    git("commit", "-q", "--allow-empty", "-m", "c")
    git("push", "-q", "origin", "HEAD:main")
    bin_, pidf = tmp_path / "bin", tmp_path / "pid"
    bin_.mkdir()
    (bin_ / "uv").write_text(f"#!/bin/sh\n{_POST_STUB.format(posted=tmp_path / 'posted')}echo 'check: verdicts: gathering'\n"
                             f"echo $$ > {pidf}.tmp\nmv {pidf}.tmp {pidf}\nexit 137\n")
    # waits until check.py has exited, not on a clock
    (bin_ / "sleep").write_text(f"#!/bin/sh\nuntil [ -e {pidf} ]; do :; done\nwhile kill -0 $(cat {pidf}) 2>/dev/null; do :; done\n")
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = {**os.environ, **GIT_ENV, "PATH": f"{bin_}:/usr/bin:/bin", "TMPDIR": str(tmp_path), "AWS_ACCESS_KEY_ID": "dummy",
           "R2_ENDPOINT": "dummy"}
    for k in ("VERITY_SKIP_STORE", "AWS_SESSION_TOKEN"):
        env.pop(k, None)
    p = subprocess.run(["bash", str(ROOT / "tools" / "verity" / "check" / "train.sh"), "HEAD", "auto"],
                       capture_output=True, text=True, cwd=tmp_path / "repo", env=env)
    assert p.returncode == 137 and "train: check.py exited 137 before launching a run (killed" in p.stderr, (p.stdout, p.stderr)
    assert "RUN" not in p.stdout
    posted = (tmp_path / "posted").read_text()
    assert "registered no run (exit 137)" in posted and "exited 137 before launching a run" in posted, posted
    assert "check: verdicts: gathering" in posted, posted


def test_the_lean_audit_step_regenerates_records_only_where_the_lean_job_takes_it(tmp_path, monkeypatch):
    """A train's Lean check passes a package whose only failures are record differences, and drive commits the
    regenerated locks as it lands. A tree whose lean_audit.py predates `--regenerate` runs as it always did."""
    monkeypatch.setattr(CHECK, "LEAN_JOBS", tmp_path)
    (tmp_path / "lean_audit.py").write_text("ap.add_argument('--fresh')\n")
    assert not CHECK.lean_regenerates()
    (tmp_path / "lean_audit.py").write_text('ap.add_argument("--regenerate", action="store_true")\n')
    assert CHECK.lean_regenerates()


def test_the_lints_are_keyed_by_the_committed_tree_and_unkeyed_on_any_change(tmp_path, monkeypatch):
    """preflight-lints read every file (`LINTS`), so their pass is kept under the commit's tree, and a tree with a change git
    hasn't committed, an untracked file too, has no key: the lints run."""
    import subprocess

    def git(*a):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *a], cwd=tmp_path,
                       check=True, capture_output=True)

    (tmp_path / "a.md").write_text("a\n")
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "a")
    monkeypatch.setattr(CHECK, "ROOT", tmp_path)
    e = {"PATH": "/usr/bin"}
    one = CHECK.lints_key(e)
    assert one and CHECK.lints_key(e) == one and CHECK.lints_key({"PATH": "/bin"}) != one
    (tmp_path / "b.md").write_text("b\n")
    assert CHECK.lints_key(e) is None
    git("add", "-A")
    git("commit", "-q", "-m", "b")
    assert CHECK.lints_key(e) not in (None, one)
