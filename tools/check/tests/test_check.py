"""`tools/check/check.py`: its steps in order, the Lean hook with and without the verifier, and the result a recorded run publishes."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("repo_check", ROOT / "tools" / "check" / "check.py")
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)


def test_the_steps_are_the_suites_then_circuit_check_then_the_lean_hook(tmp_path, monkeypatch):
    monkeypatch.setattr(CHECK, "FLOCK_VERIFIER", tmp_path / "verifier")
    names = [(n, argv if isinstance(argv, str) else argv[1:4]) for n, argv, _cwd in CHECK.steps(tmp_path)]
    assert names[0] == ("pytest", [str(ROOT / "tools/check/suites.py"), "--json", str(tmp_path / "suites.json")])
    assert names[1] == ("flock-circuit-build", [str(ROOT / "backends" / "flock" / "check_build.sh")])
    assert names[2] == ("circuit-check", ["-m", "circuit_check", "--all"])
    assert names[3] == ("flock-rust-tests", ["CARGO_BUILD_JOBS=1", "RUST_TEST_THREADS=1", "bash"])
    assert names[4] == ("lean", "skipped: backends/flock/verifier/lean is not on this tree")


def test_the_build_step_checks_every_cpu_feature_set_a_pod_builds():
    """A proving build never has `seed-injection` (1053c0c9 compiled only with it); the class sweep's proving build has no
    `glue`."""
    script = (ROOT / "backends" / "flock" / "check_build.sh").read_text()
    assert "for features in sha512 sha512,glue sha512,glue,seed-injection; do" in script
    assert "cargo check -q -p flock-live --bin flock-circuit" in script


def test_the_rust_tests_run_the_live_crates_unit_tests_and_every_test_of_the_crates_the_patches_change():
    """In release, as upstream's CI runs them, on the checkout the build step patches; in two invocations, since `--lib` on the
    upstream crates would drop their integration tests."""
    script = (ROOT / "backends" / "flock" / "check_build.sh").read_text()
    start = script.index('if [ "${1:-}" = test ]')
    mode = script[start:script.index("\n  exit\nfi\n", start)]
    assert "for features in sha512 sha512,glue,seed-injection; do" in mode, "the proving build's set, without seed-injection"
    assert 'cargo test --release -q -p flock-live --lib --features "$features"' in mode
    patched = set()
    for p in sorted((ROOT / "backends" / "flock").glob("flock-*-b684b12.patch")):
        patched |= {m.group(1) for line in p.read_text().splitlines() if (m := re.match(r"\+\+\+ b/crates/([a-z-]+)/", line))}
    upstream = re.search(r"cargo test --release -q((?: -p [a-z-]+)+)$", mode, re.M).group(1).split(" -p ")[1:]
    assert sorted(upstream) == sorted(patched - {"flock-live"}), "every crate a patch changes, and only those"


def test_the_rust_tests_key_moves_with_the_script_patches_live_crate_and_toolchain_only(tmp_path, monkeypatch):
    assert CHECK.rust_toolchain(), "check_build.sh pins its toolchain: an agent VM's default (1.83) can't build upstream"
    flock = tmp_path / "backends" / "flock"
    files = {"backends/flock/check_build.sh": "set -e\n", "backends/flock/flock-sha512-b684b12.patch": "p",
             "backends/flock/live/src/lib.rs": "fn a() {}", "backends/flock/verifier/ci.py": "", "backends/flock/README.md": ""}
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
    shutil.copy(ROOT / "backends" / "flock" / "verifier" / "agree.py", v / "agree.py")
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
    assert lean_audit[1].endswith("tools/check/check.py") and lean_audit[2] == "--lean-slot"
    assert lean_audit[4].endswith("tools/check/lean_audit.py") and {"--cache", "--deps"} <= set(lean_audit) and "--fresh" not in lean_audit
    assert "--fresh" in CHECK.lean_steps(out, cache=False)[2][1]
    assert argv[2:4] == ["--lean-agreement", str(out)] and {Path(x).name for x in argv[4:]} == {"a.tar.gz", "b.tar.gz"}
    (sent / "b.tar.gz").unlink()
    assert CHECK.lean_steps(out)[3][1] == "only upstream of the pinned upstream build and inputs was sent"
    (sent / "a.tar.gz").unlink()
    assert CHECK.lean_steps(out)[3][1].startswith("skipped: the pinned upstream build was not sent")


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

    sets = json.loads((ROOT / "backends" / "flock" / "verifier" / "vectors.json").read_text())["replayable"]["sets"]
    assert len({s["pr83"] for s in sets if s.get("sessions") and "tables" not in s["statement"]}) >= 4


def test_the_rule_the_gate_reads(monkeypatch):
    spec2 = importlib.util.spec_from_file_location("check_tool", ROOT / "tools" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(tool)
    assert tool.CHECK.merge_requires == {"lean-agreement": ["backends/flock/", "!README.md", "!PROTOCOL.md"]}
    pin = __import__("json").loads((ROOT / "backends" / "flock" / "verifier" / "upstream.json").read_text())
    assert set(pin["upstream"]["binaries"]) == {s["pr83"] for s in __import__("json").loads(
        (ROOT / "backends" / "flock" / "verifier" / "vectors.json").read_text())["replayable"]["sets"]}


def test_the_recorded_check_takes_cpu_only_torch():
    spec2 = importlib.util.spec_from_file_location("check_tool", ROOT / "tools" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(tool)
    assert tool.CHECK.command[:5] == ["uv", "run", "--locked", "--extra", "torch-cpu"]
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
    for can, said in ((False, True), (True, False)):
        monkeypatch.setattr(CHECK, "landable", lambda base, can=can: can)
        assert CHECK.record(["--base", "HEAD"]) == 0 and len(launched) == 1
        out = capsys.readouterr().out
        assert ("check: this commit can't land as is; the train's check is the merge check" in out) is said
        launched.clear()


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

    (tmp_path / "tools" / "check").mkdir(parents=True)
    (tmp_path / "tools" / "check" / "tool.py").write_text(
        "from research.store.tool import Tool\nCHECK = Tool(name='check', version='1', command=['true'], "
        "merge_requires={'lean-agreement': ['backends/flock/', '!README.md']})\n")
    (tmp_path / "backends" / "flock").mkdir(parents=True)
    (tmp_path / "backends" / "flock" / "verify.rs").write_text("fn main() {}\n")
    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    git("mv", "backends/flock/verify.rs", "verify.rs")
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

def test_the_groups_split_the_cores_and_run_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(CHECK, "FLOCK_VERIFIER", tmp_path / "verifier")
    (pt,), (build, cc, rust), (lean,) = CHECK.groups(tmp_path, 16)
    assert build[0] == "flock-circuit-build" and build[1][-1].endswith("backends/flock/check_build.sh")
    assert rust[0] == "flock-rust-tests" and rust[1][-2:] == [build[1][-1], "test"] and "CARGO_BUILD_JOBS=8" in rust[1]
    assert pt[1][1].endswith("tools/check/suites.py") and pt[1][pt[1].index("--jobs") + 1] == "8" and "--fresh" not in pt[1]
    assert cc[1][cc[1].index("--jobs") + 1] == "8" and "--cache" in cc[1]
    cold = CHECK.groups(tmp_path, 16, cache=False)
    assert lean[0] == "lean" and "--cache" not in cold[1][0][1] and "--fresh" in cold[0][0][1]
    assert not any(s in pt[1] for s in CHECK.AFTER_LEAN), "without the Lean steps the suites that use Lean run with the rest"


def test_the_suites_that_use_lean_run_after_the_lean_steps_have_built_everything(tmp_path, monkeypatch):
    """Not beside `lake build`: whether their tests run or skip, and what `lake` finds built, must not depend on how far the
    concurrent Lean group has got (verity-flock passed or failed on build order, r20260929-015410-5a20)."""
    out, _ = _pinned(tmp_path, monkeypatch)
    (_, pytest, _), = CHECK.groups(out, 8)[0]
    lean = CHECK.groups(out, 8)[2]
    assert [n for n, _, _ in lean] == ["lean-build", "lean-unit-cut", "lean-audit", "lean-suites", "lean-agreement"]
    assert all(pytest[pytest.index(s) - 1] == "--skip" for s in CHECK.AFTER_LEAN)
    argv = lean[3][1]
    assert argv[:2] == ["bash", "-c"] and argv[2].startswith(CHECK.ELAN), "with elan's lake on the PATH, as lean-build has it"
    assert all(f" {s} " in argv[2] for s in CHECK.AFTER_LEAN) and "suites-lean.json" in argv[2] and "--jobs 4" in argv[2]


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
    """README.md and PROTOCOL.md under backends/flock/ are docs the agreement never reads: alone they do not require it."""
    from research.merge import rule_touches

    spec2 = importlib.util.spec_from_file_location("check_tool", ROOT / "tools" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(tool)
    rule = tool.CHECK.merge_requires["lean-agreement"]
    docs = ["backends/flock/verifier/PROTOCOL.md", "backends/flock/verifier/README.md", "backends/flock/README.md", "AGENTS.md"]
    assert rule_touches(rule, docs) == []
    assert rule_touches(rule, [*docs, "backends/flock/verifier/ci.py"]) == ["backends/flock/"]
    assert rule_touches(rule, ["backends/flock/verifier/DESIGN.md"]) == ["backends/flock/"]      # only the two names are left out
    for f in ("ci.py", "agree.py", "fuzz.py", "redigest.py", "transcript_check.py", "selftest_records.py"):   # cited, never read
        reads = [line for line in (ROOT / "backends" / "flock" / "verifier" / f).read_text().splitlines()
                 if any(r in line for r in ("open(", "read_text", "read_bytes")) and any(d in line for d in ("README", "PROTOCOL"))]
        assert not reads, f"{f} reads a doc the agreement trigger leaves out: {reads}"


def test_the_agreement_scripts_import_only_the_standard_library_and_each_other():
    """`agreement_key` covers backends/flock/verifier/ and the interpreter only: an import from elsewhere in `ci.py` or the scripts
    it runs would be an input the key misses, so one fails here until the key covers it.  The key also leaves out the Lean
    package's `lean-audit.json` (`agreement_closure`), so neither those scripts nor the package's lakefile may mention it."""
    import ast
    import re
    import sys

    here = ROOT / "backends" / "flock" / "verifier"
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
    readers = [here / f"{m}.py" for m in sorted(seen)] + [here / "lean" / "lakefile.toml"]
    assert not (named := [p.name for p in readers if "lean-audit" in p.read_text()]), (
        f"{', '.join(named)} mention lean-audit, which agreement_closure leaves out of the agreement's cache key")


def test_the_agreement_reads_the_verifier_package_not_the_proof_packages_nested_in_it(tmp_path, monkeypatch):
    """`flock-verify` is built from the Lean package alone: level3/ and soundness/ (Lake packages of their own) are no input,
    unless the package's lakefile requires something, which keeps everything in.  Nor is the package's `lean-audit.json`, which
    only the audit reads: that one path, not every file of the name."""
    files = CHECK.tracked_files()
    closure = CHECK.agreement_closure(files)
    policy, other = "backends/flock/verifier/lean/lean-audit.json", "backends/flock/verifier/lean/Flock/lean-audit.json"
    assert policy not in CHECK.agreement_closure([*files, policy]) and other in CHECK.agreement_closure([*files, other])
    assert "backends/flock/verifier/ci.py" in closure and "backends/flock/verifier/lean/Main.lean" in closure
    assert "backends/flock/verifier/upstream.json" in closure
    assert not [f for f in closure if f.startswith(("backends/flock/verifier/lean/soundness/", "backends/flock/verifier/lean/level3/"))]
    assert [f for f in files if f.startswith("backends/flock/verifier/lean/soundness/")]
    (tmp_path / "backends/flock/verifier/lean/soundness").mkdir(parents=True)
    (tmp_path / "backends/flock/verifier/lean/lakefile.toml").write_text('[[require]]\nname = "x"\n')
    (tmp_path / "backends/flock/verifier/lean/soundness/lakefile.toml").write_text("")
    monkeypatch.setattr(CHECK, "ROOT", tmp_path)
    fake = ["backends/flock/verifier/lean/lakefile.toml", "backends/flock/verifier/lean/soundness/lakefile.toml"]
    assert CHECK.agreement_closure(fake) == fake


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
    """`run` over the given groups and preflight, with nothing built first and no Lean URLs."""
    monkeypatch.setattr(CHECK, "PREBUILD", "pass")
    monkeypatch.setattr(CHECK, "preflight", lambda: list(pre))
    monkeypatch.setattr(CHECK, "groups", lambda out, n, cache=True, urls=None: groups)
    monkeypatch.delenv(CHECK.LEAN_DEPS_URLS_ENV, raising=False)


def _py(src: str, *args) -> list[str]:
    import sys

    return [sys.executable, "-c", src, *map(str, args)]


def test_a_failed_step_stops_the_other_groups_and_each_status_is_written_as_it_ends(tmp_path, monkeypatch):
    """`fails` exits only once `slow` has written to a FIFO, so `slow` is running when it fails: `slow` is interrupted (SIGINT),
    its group starts nothing more, and steps.jsonl has `fails` before `slow`, each line written as its step ended."""
    import json
    import os

    gate, marker = tmp_path / "gate", tmp_path / "never-ran"
    os.mkfifo(gate)
    slow = _py("import signal, sys\nwith open(sys.argv[1], 'w') as f:\n    f.write('x')\nsignal.pause()", gate)
    fails = _py("import sys\nopen(sys.argv[1]).read()\nsys.exit(3)", gate)
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


def test_a_lean_module_that_does_not_compile_stops_the_run_even_under_keep_going(tmp_path, monkeypatch):
    """`lean-changed` exits only once `slow` is running (the FIFO): under --keep-going its failure still stops `slow`."""
    import os

    gate = tmp_path / "gate"
    os.mkfifo(gate)
    slow = _py("import signal, sys\nwith open(sys.argv[1], 'w') as f:\n    f.write('x')\nsignal.pause()", gate)
    changed = _py("import sys\nopen(sys.argv[1]).read()\nsys.exit(1)", gate)
    _fake_run(monkeypatch, [[("slow", slow, tmp_path)], [("lean-changed", changed, tmp_path)]])
    passed, done = CHECK.run(tmp_path, 2, keep_going=True)
    by = {s["step"]: s for s in done}
    assert not passed and by["lean-changed"]["status"] == "failed"
    assert by["slow"]["status"] == "stopped" and by["slow"]["detail"] == "stopped: lean-changed failed"


def test_a_rust_crate_that_does_not_compile_stops_the_run_even_under_keep_going(tmp_path, monkeypatch):
    """flock-circuit-build fails, with cargo's line, only once `slow` is running (the FIFO): under --keep-going `slow` stops."""
    import os

    gate = tmp_path / "gate"
    os.mkfifo(gate)
    slow = _py("import signal, sys\nwith open(sys.argv[1], 'w') as f:\n    f.write('x')\nsignal.pause()", gate)
    build = _py("import sys\nopen(sys.argv[1]).read()\nprint('error: could not compile `flock-circuit` (lib)')\nsys.exit(101)", gate)
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

    kind = CHECK._load("lean_changed").FAST_KIND
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
    for bad in ([], {"packages": {"a": full}}, {"packages": ["a"]}):     # JSON, not an audit's report: it crashed check (#966)
        rec = step(bad)
        assert rec["status"] == "failed" and "is not an audit's report" in rec["detail"], rec
    rec = step({"passed": True, "failed": [], "packages": [{**full, "seconds": ["replay"]}]})
    assert rec["status"] == "failed" and rec["detail"].startswith(f"refused: a: a {kind} result"), rec


def test_the_early_lean_build_is_a_group_of_its_own_beside_the_lean_steps(tmp_path, monkeypatch):
    out, _ = _pinned(tmp_path, monkeypatch)
    gs = CHECK.groups(out, 8)
    assert len(gs) == 4 and [n for n, _, _ in gs[3]] == ["lean-changed"] and "lean-changed" in CHECK.HALTS
    argv = gs[3][0][1]
    assert argv[1].endswith("tools/check/lean_changed.py")
    assert argv[argv.index("--cache") + 1] == gs[2][2][1][gs[2][2][1].index("--cache") + 1], "the audit's own passes"
    assert argv[argv.index("--deps") + 1] == gs[2][2][1][gs[2][2][1].index("--deps") + 1], "and its kept dependencies"
    assert CHECK.groups(out, 8, cache=False)[3][0][1].startswith("skipped: --no-cache")
    monkeypatch.setattr(CHECK, "FLOCK_VERIFIER", tmp_path / "no-verifier")
    assert len(CHECK.groups(out, 8)) == 3


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
        pre = [(n, _py("raise SystemExit(1)") if n == failing else _py("pass"), tmp_path) for n in ("preflight-lock", "preflight-lints")]
        _fake_run(monkeypatch, [[("pytest", touch, tmp_path)]], pre=pre)
        ok, done = CHECK.run(tmp_path, 2, keep_going=True)
        assert not ok and "pytest" not in {s["step"] for s in done} and not marker.exists(), failing
        assert {s["step"]: s["status"] for s in done}[failing] == "failed"


def test_the_preflight_is_the_lockfile_and_the_repository_lints(monkeypatch):
    """Without bytecode or pytest's cache; every lint it names exists, so a rename can't turn it into a failure of
    every check. Without uv, the lockfile step fails by name."""
    (lock, argv, cwd), (lints, pytest, cwd2) = CHECK.preflight()
    assert (lock, argv[1:], cwd) == ("preflight-lock", ["lock", "--check"], ROOT)
    assert (lints, pytest[1:4], cwd2) == ("preflight-lints", ["-B", "-m", "pytest"], ROOT) and "no:cacheprovider" in pytest
    assert pytest[-len(CHECK.LINTS):] == list(CHECK.LINTS)
    for t in CHECK.LINTS:
        path, _, name = t.partition("::")
        assert (ROOT / path).is_file() and (not name or f"def {name}(" in (ROOT / path).read_text())
    monkeypatch.delenv("UV", raising=False)
    monkeypatch.setattr(CHECK.shutil, "which", lambda _name: None)
    assert CHECK.preflight()[0][1].startswith("failed: no `uv` here")


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
