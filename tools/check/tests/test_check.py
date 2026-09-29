"""`tools/check/check.py`: its steps in order, the Lean hook with and without the verifier, and the result a recorded run publishes."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("repo_check", ROOT / "tools" / "check" / "check.py")
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)


def test_the_steps_are_the_suites_then_circuit_check_then_the_lean_hook(tmp_path, monkeypatch):
    monkeypatch.setattr(CHECK, "FLOCK_VERIFIER", tmp_path / "verifier")
    names = [(n, argv if isinstance(argv, str) else argv[1:4]) for n, argv, _cwd in CHECK.steps(tmp_path)]
    assert names[0] == ("pytest", [str(ROOT / "tools/check/suites.py"), "--json", str(tmp_path / "suites.json")])
    assert names[1] == ("circuit-check", ["-m", "circuit_check", "--all"])
    assert names[2] == ("lean", "skipped: backends/flock/verifier/lean is not on this tree")


_BIN = b"#!/bin/sh\nexit 0\n"


def _pinned(tmp_path, monkeypatch, binary=_BIN):
    """A verifier tree with a pin over a one-binary upstream build and an inputs tree, both built here."""
    import hashlib
    import io
    import json
    import tarfile

    v = tmp_path / "verifier"
    (v / "lean").mkdir(parents=True)
    (v / "lean" / "lakefile.toml").write_text("")
    (v / "ci.py").write_text("import json, sys; open(sys.argv[sys.argv.index('--out') + 1] + '/argv.json', 'w').write(json.dumps(sys.argv))\n")
    (v / "vectors.json").write_text('{"replayable": {"sets": [{"pr83": "abc"}]}}')

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
    ins = tgz(sent / "b.tar.gz", {"0123/x": b"x"})
    pin = {"upstream": {"art": "art:u", "sha256": up, "binaries": {"abc": hashlib.sha256(_BIN).hexdigest()}},
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
    assert lean_audit[1].endswith("tools/check/lean_audit.py") and {"--cache", "--deps"} <= set(lean_audit) and "--fresh" not in lean_audit
    assert "--fresh" in CHECK.lean_steps(out, cache=False)[2][1]
    assert argv[2:4] == ["--lean-agreement", str(out)] and {Path(x).name for x in argv[4:]} == {"a.tar.gz", "b.tar.gz"}
    (sent / "b.tar.gz").unlink()
    assert CHECK.lean_steps(out)[3][1] == "only upstream of the pinned upstream build and inputs was sent"
    (sent / "a.tar.gz").unlink()
    assert CHECK.lean_steps(out)[3][1].startswith("skipped: the pinned upstream build was not sent")


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


def test_the_rule_the_gate_reads(monkeypatch):
    spec2 = importlib.util.spec_from_file_location("check_tool", ROOT / "tools" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(tool)
    assert tool.CHECK.merge_requires == {"lean-agreement": ["backends/flock/", "!README.md", "!PROTOCOL.md"]}
    pin = __import__("json").loads((ROOT / "backends" / "flock" / "verifier" / "upstream.json").read_text())
    assert set(pin["upstream"]["binaries"]) == {s["pr83"] for s in __import__("json").loads(
        (ROOT / "backends" / "flock" / "verifier" / "vectors.json").read_text())["replayable"]["sets"]}


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


def test_the_groups_split_the_cores_and_run_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(CHECK, "FLOCK_VERIFIER", tmp_path / "verifier")
    (pt,), (cc,), (lean,) = CHECK.groups(tmp_path, 16)
    assert pt[1][1].endswith("tools/check/suites.py") and pt[1][pt[1].index("--jobs") + 1] == "8" and "--fresh" not in pt[1]
    assert cc[1][cc[1].index("--jobs") + 1] == "8" and "--cache" in cc[1]
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", "abc123")                          # a shipped tree: suites.py gets the commit
    shipped = CHECK.groups(tmp_path, 16)[0][0][1]
    assert shipped[shipped.index("--commit") + 1] == "abc123"
    cold = CHECK.groups(tmp_path, 16, cache=False)
    assert lean[0] == "lean" and "--cache" not in cold[1][0][1] and "--fresh" in cold[0][0][1]


def test_a_checkouts_files_are_gits_and_the_key_environment_drops_the_tree_path(tmp_path):
    """The agreement's key reads the tree as git has it: tracked and untracked-unignored files, not bytecode or builds."""
    import subprocess

    assert CHECK.tracked_files(tmp_path) is None                                   # no git, no shipped tree: nothing is keyed
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("__pycache__/\n.lake/\n")
    (tmp_path / "a.txt").write_text("a")
    for built in ("__pycache__/x.pyc", ".lake/build/bin/flock-verify"):
        (tmp_path / built).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / built).write_bytes(b"x")
    assert CHECK.tracked_files(tmp_path) == [".gitignore", "a.txt"]
    assert CHECK._env_key({"VIRTUAL_ENV": f"{ROOT}/.venv", "SSH_CLIENT": "1 2 3"}) == {"VIRTUAL_ENV": "<tree>/.venv"}   # per-commit path


def test_a_shipped_trees_files_are_its_commits(tmp_path):
    """On a machine the tree is a `git archive` beside the bare repository it was pushed to: its files are the commit's tree,
    not what a step built inside it since."""
    import json
    import subprocess

    work, bare = tmp_path / "work", tmp_path / "root" / "git" / "verity.git"
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    (work / "a.txt").write_text("a")
    g = ["git", "-C", str(work), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*g, "add", "a.txt"], check=True)
    subprocess.run([*g, "commit", "-q", "-m", "a"], check=True)
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True)
    sha = subprocess.run([*g, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    tree = subprocess.run([*g, "rev-parse", "HEAD^{tree}"], capture_output=True, text=True, check=True).stdout.strip()
    shipped = tmp_path / "root" / "src" / sha
    shipped.mkdir(parents=True)
    (shipped / "a.txt").write_text("a")
    (shipped / "READY.json").write_text(json.dumps({"tree": tree}))
    (shipped / "built.bin").write_bytes(b"x")
    assert CHECK.tracked_files(shipped) == ["a.txt"]


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
