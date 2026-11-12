"""`tools/verity/check/check.py`: its steps in order, the Lean hook with and without the verifier, and the result a recorded run publishes."""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
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
    assert names[1] == ("flock-circuit-build", [str(ROOT / "verity" / "protocols" / "flock" / "prover" / "check_build.sh")])
    assert names[2] == ("flock-rust-tests", ["CARGO_BUILD_JOBS=1", "RUST_TEST_THREADS=1", "bash"])
    assert names[3] == ("circuit-check", ["-m", "circuit_check", "--all"])
    assert names[4] == ("lean", "skipped: verity/protocols/flock/verifier is not on this tree")


def test_the_build_step_checks_every_cpu_feature_set_a_pod_builds():
    """A proving build never has `seed-injection` (1053c0c9 compiled only with it); the class sweep's proving build has no
    `glue`."""
    script = (ROOT / "verity" / "protocols" / "flock" / "prover" / "check_build.sh").read_text()
    assert "for features in sha512 sha512,glue sha512,glue,seed-injection; do" in script
    assert "cargo check -q -p flock-live --bin flock-circuit" in script


def test_the_rust_tests_run_the_live_crates_unit_tests_and_every_test_of_the_crates_the_patches_change():
    """In release, as upstream's CI runs them, on the checkout the build step patches; in two invocations, since `--lib` on the
    upstream crates would drop their integration tests."""
    script = (ROOT / "verity" / "protocols" / "flock" / "prover" / "check_build.sh").read_text()
    start = script.index('if [ "${1:-}" = test ]')
    mode = script[start:script.index("\n  exit\nfi\n", start)]
    assert "for features in sha512 sha512,glue,seed-injection; do" in mode, "the proving build's set, without seed-injection"
    assert 'cargo test --release -q -p flock-live --lib --features "$features"' in mode
    patched = set()
    for p in sorted((ROOT / "verity" / "protocols" / "flock" / "prover").glob("flock-*-b684b12.patch")):
        patched |= {m.group(1) for line in p.read_text().splitlines() if (m := re.match(r"\+\+\+ b/crates/([a-z-]+)/", line))}
    upstream = re.search(r"cargo test --release -q((?: -p [a-z-]+)+)(?: \\)?$", mode, re.M).group(1).split(" -p ")[1:]
    assert sorted(upstream) == sorted(patched - {"flock-live"}), "every crate a patch changes, and only those"


def test_the_rust_tests_key_moves_with_the_script_patches_live_crate_and_toolchain_only(tmp_path, monkeypatch):
    assert CHECK.rust_toolchain(), "check_build.sh pins its toolchain: an agent VM's default (1.83) can't build upstream"
    flock = tmp_path / "verity" / "protocols" / "flock" / "prover"
    files = {"verity/protocols/flock/prover/check_build.sh": "set -e\n", "verity/protocols/flock/prover/flock-sha512-b684b12.patch": "p",
             "verity/protocols/flock/prover/live/src/lib.rs": "fn a() {}", "verity/protocols/flock/agreement/ci.py": "", "verity/protocols/flock/prover/README.md": ""}
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
    (flock.parent / "agreement" / "ci.py").write_text("changed")
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
    shutil.copy(ROOT / "verity" / "protocols" / "flock" / "agreement" / "agree.py", v / "agree.py")
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


def _signed(sent: Path, at: Path) -> Path:
    """`AGREEMENT_URLS` at ``at``, as `agreement_url_file` writes it, for the pinned files `_pinned` built in ``sent``, each with a
    `file:` URL of its tarball."""
    import json

    pin = json.loads(CHECK.UPSTREAM_PIN.read_text())
    at.write_text(json.dumps({k: {"art": pin[k]["art"], "sha256": pin[k]["sha256"], "bytes": (sent / f).stat().st_size,
                                  "url": (sent / f).as_uri()} for k, f in (("upstream", "a.tar.gz"), ("inputs", "b.tar.gz"))}))
    return at


def test_the_lean_steps_once_the_verifier_is_on_the_tree(tmp_path, monkeypatch):
    """`lake build`, the unit-cut conformance and the audit of every Lean package are required; the agreement runs when the run
    was sent URLs of the pinned upstream build and inputs, each listed with its pinned sha256, is skipped by name when it was
    sent none, and fails when the file it was sent can't be read or doesn't list both, rather than passing for not sent."""
    import json

    out, sent = _pinned(tmp_path, monkeypatch)
    urls = _signed(sent, tmp_path / "agreement-urls.json")
    (build, lake, _), (unit, cut, _), (audit, lean_audit, _), (agree, argv, _) = CHECK.lean_steps(out, agreement_urls=urls)
    assert (build, unit, audit, agree) == ("lean-build", "lean-unit-cut", "lean-audit", "lean-agreement")
    assert lake[-1].endswith("lake build")
    assert cut[1].endswith("verifier/unit_cut_agree.py") and cut[cut.index("--lean") + 1].endswith(".lake/build/bin/flock-verify")
    assert lean_audit[1].endswith("tools/verity/check/check.py") and lean_audit[2] == "--lean-slot"
    assert lean_audit[4].endswith("tools/verity/lean/lean_audit.py") and {"--cache", "--deps"} <= set(lean_audit) and "--fresh" not in lean_audit
    assert "--fresh" in CHECK.lean_steps(out, cache=False)[2][1]
    assert "--regenerate" in lean_audit and "--regenerate" not in CHECK.lean_steps(out, regenerate=False)[2][1]
    warn = CHECK.lean_steps(out, bookkeeping=True)[2][1]
    assert "--bookkeeping" not in lean_audit and warn[warn.index("--bookkeeping") + 1] == "warn"
    assert "--bookkeeping" not in CHECK.lean_steps(out, regenerate=False, bookkeeping=True)[2][1]
    assert argv[1].endswith("tools/verity/check/check.py") and argv[2:] == ["--lean-agreement-urls", str(out), str(urls)]
    doc = json.loads(urls.read_text())
    urls.write_text(json.dumps({**doc, "inputs": {**doc["inputs"], "sha256": "0" * 64}}))
    assert CHECK.lean_steps(out, agreement_urls=urls)[3][1] == "the URLs sent list upstream of the pinned upstream build and inputs"
    for unsigned in ({"upstream": {**doc["upstream"], "url": None}}, ["upstream"]):
        urls.write_text(json.dumps(unsigned))
        assert CHECK.lean_steps(out, agreement_urls=urls)[3][1] == "the URLs sent list neither of the pinned upstream build and inputs"
    urls.write_text("{")
    assert CHECK.lean_steps(out, agreement_urls=urls)[3][1] == ("the URLs sent for the pinned upstream build and inputs can't be "
                                                                "read (JSONDecodeError)")
    assert CHECK.lean_steps(out)[3][1].startswith("skipped: the pinned upstream build was not sent")


def test_the_agreement_fails_while_the_verifier_requires_a_package_from_outside_the_tree(tmp_path, monkeypatch):
    """The agreement's pass is kept under the verifier's package and the packages of the tree it requires by path
    (`agreement_closure`), and `research merge` asks for it for changes there, so a requirement from outside the tree, a
    fetched package or a path that leaves the repository, by the manifest or the lakefile, fails the step whether or not
    the upstream build was sent; one inside the tree doesn't, and this tree's verifier has none from outside."""
    import json

    out, sent = _pinned(tmp_path, monkeypatch)
    urls = _signed(sent, tmp_path / "agreement-urls.json")
    lean = CHECK.FLOCK_LEAN
    (lean / "lakefile.toml").write_text('name = "flock_verifier"\n[[require]]\nname = "inner"\npath = "inner"\n')
    assert not isinstance(CHECK.lean_steps(out, agreement_urls=urls)[3][1], str)
    (lean / "lakefile.toml").write_text('name = "flock_verifier"\n[[require]]\nname = "identity"\npath = "../identity"\n')
    assert CHECK.lean_steps(out, agreement_urls=urls)[3][1].startswith(
        "the verifier's Lean package requires identity from outside the tree")
    monkeypatch.setattr(CHECK, "ROOT", tmp_path)
    assert not isinstance(CHECK.lean_steps(out, agreement_urls=urls)[3][1], str), \
        "a package of the tree, required by path, is keyed with the verifier"
    monkeypatch.setattr(CHECK, "ROOT", ROOT)
    (lean / "lakefile.toml").write_text('name = "flock_verifier"\n')
    (lean / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "git", "name": "mathlib", "rev": "abc"}]}))
    assert CHECK.lean_steps(out)[3][1].startswith("the verifier's Lean package requires mathlib from outside")
    flock = ROOT / "verity" / "protocols" / "flock" / "verifier"
    assert CHECK.requires_outside(flock) == []
    assert CHECK.required_in_tree("verity/protocols/flock/verifier/") == (
        "verity/circuits/base/", "verity/circuits/lean/", "verity/commitments/lean/")


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


def test_the_agreement_urls_leave_the_run_directory_before_any_step(tmp_path, monkeypatch):
    """The pinned upstream build's URLs grant a read until they expire, as the Lean bundles' do: taken out of the run directory
    to a file only its owner reads, the step's argument, which `run` deletes when the run ends; a local recorded run reads its
    launcher's file where it is, and leaves it."""
    out, sent = _pinned(tmp_path, monkeypatch)
    monkeypatch.delenv(CHECK.AGREEMENT_URLS_ENV, raising=False)
    assert CHECK.take_agreement_urls(out) is None
    body = _signed(sent, out / "inputs" / CHECK.AGREEMENT_URLS).read_text()
    got = CHECK.take_agreement_urls(out)
    try:
        assert not (out / "inputs" / CHECK.AGREEMENT_URLS).exists() and not got.is_relative_to(out)
        assert got.read_text() == body and got.stat().st_mode & 0o777 == 0o600 and got.name.startswith(CHECK.TAKEN)
        assert CHECK.lean_steps(out, agreement_urls=got)[3][1][-1] == str(got)
    finally:
        got.unlink()
    monkeypatch.setenv(CHECK.AGREEMENT_URLS_ENV, str(local := _signed(sent, tmp_path / CHECK.AGREEMENT_URLS)))
    assert CHECK.take_agreement_urls(out) == local and not local.name.startswith(CHECK.TAKEN)


def test_the_verdict_pack_urls_leave_the_run_directory_before_any_step(tmp_path, monkeypatch, capsys):
    """Each URL grants a read of its pack until it expires, and the run's record keeps what its run directory holds.  A file that
    isn't a list of packs imports none and goes all the same; a local recorded run reads its launcher's file where it is."""
    monkeypatch.delenv(CHECK.VERDICT_URLS_ENV, raising=False)
    assert CHECK.take_verdict_urls(tmp_path) == []
    (tmp_path / "inputs").mkdir()
    pack = {"run": "r-new", "sha256": "a" * 64, "bytes": 3, "url": "https://r2/o?sig"}
    unsigned = {k: v for k, v in pack.items() if k != "url"}
    (sent := tmp_path / "inputs" / CHECK.VERDICT_URLS).write_text(json.dumps([pack, {**pack, "bytes": "3"}, unsigned, {"run": "r-x"}, "r-y"]))
    assert CHECK.take_verdict_urls(tmp_path) == [pack] and not sent.exists()
    for garbled in ("[", json.dumps(pack), "3"):
        sent.write_text(garbled)
        assert CHECK.take_verdict_urls(tmp_path) == [] and not sent.exists()
    assert capsys.readouterr().out.count("check: verdicts: the packs' URLs sent can't be read (JSONDecodeError); the run imports none") == 1
    (local := tmp_path / "urls.json").write_text(json.dumps([pack]))
    monkeypatch.setenv(CHECK.VERDICT_URLS_ENV, str(local))                              # a local recorded run's launcher
    assert CHECK.take_verdict_urls(tmp_path) == [pack] and local.exists()


def test_custody_deletes_the_presigned_urls_wherever_a_run_leaves_them(tmp_path):
    """A run of anything else, or a check that ends before `check.py` takes them, leaves them in inputs/ for custody to delete."""
    from research.store.custody import drop_credentials

    (tmp_path / "inputs").mkdir()
    for name in (CHECK.LEAN_DEPS_URLS, CHECK.VERDICT_URLS, CHECK.AGREEMENT_URLS, "sent.txt"):
        (tmp_path / "inputs" / name).write_text("x")
    drop_credentials(tmp_path)
    assert [p.name for p in (tmp_path / "inputs").iterdir()] == ["sent.txt"]


def test_the_agreement_leaves_memory_for_the_steps_beside_it(monkeypatch):
    """A 128 GB, 32-core pod that gives the agreement 16 verifier processes at 8 GB each while lean-suites runs beside it has
    its cgroup OOM-kill them: the agreement counts on half."""
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
    assert argv[argv.index("--lean") + 1] == str(CHECK.FLOCK_LEAN / ".lake" / "build" / "bin" / "flock-verify")
    out, sent = _pinned(tmp_path / "given", monkeypatch)
    assert CHECK.main(["--lean-agreement", str(out), str(sent / "a.tar.gz"), str(sent / "b.tar.gz"), str(tmp_path / "fv")]) == 0
    argv = json.loads((out / "flock-agreement" / "argv.json").read_text())
    assert argv[argv.index("--lean") + 1] == str(tmp_path / "fv")
    out, sent = _pinned(tmp_path / "t", monkeypatch, binary=b"tampered")
    assert CHECK.lean_agreement(out, sent / "a.tar.gz", sent / "b.tar.gz") == 1


def test_the_agreement_step_fetches_the_pinned_files_through_their_urls(tmp_path, monkeypatch, capsys):
    """The launch used to fetch the pinned upstream build and inputs from the store and send them, minutes of every shard's
    launch on the control pod; the step now fetches them through the URLs the launch signed, into scratch outside the run
    directory, each held to its listed size and pinned sha256, and runs the agreement on them.  With no other copy to fall
    back on, a fetch that fails is tried again after each of `AGREEMENT_FETCH_WAITS_S`, and the step fails when the last does;
    no URL is printed, since each grants a read until it expires."""
    import json
    import urllib.error

    out, sent = _pinned(tmp_path, monkeypatch)
    urls = _signed(sent, tmp_path / "agreement-urls.json")
    (scratch := tmp_path / "scratch").mkdir()
    monkeypatch.setattr(CHECK.tempfile, "tempdir", str(scratch))
    slept = []
    assert CHECK.lean_agreement_urls(out, urls, sleep=slept.append) == 0 and slept == []
    argv = json.loads((out / "flock-agreement" / "argv.json").read_text())
    assert argv[argv.index("--upstream") + 1] == f"abc={out / 'flock-agreement' / 'upstream' / 'upstream-abc'}"
    assert "check: lean-agreement: fetched the pinned upstream build and inputs (0 MiB) in " in capsys.readouterr().out
    assert not list(scratch.iterdir()), "the fetched files are scratch, deleted when the step ends"

    real, calls = CHECK.urllib.request.urlopen, []

    def flaky(url, timeout):
        calls.append(url)
        if len(calls) <= 2:
            raise urllib.error.URLError("unreachable")
        return real(url, timeout=timeout)

    monkeypatch.setattr(CHECK.urllib.request, "urlopen", flaky)
    out, sent = _pinned(tmp_path / "flaky", monkeypatch)
    assert CHECK.lean_agreement_urls(out, _signed(sent, tmp_path / "flaky.json"), sleep=slept.append) == 0
    assert slept == list(CHECK.AGREEMENT_FETCH_WAITS_S) and len(calls) == 4
    said = capsys.readouterr().out
    assert said.count("the pinned upstream build (art:u): it can't be fetched (URLError); fetching it again in ") == 2

    monkeypatch.setattr(CHECK.urllib.request, "urlopen", real)
    out, sent = _pinned(tmp_path / "gone", monkeypatch)
    doc = json.loads(_signed(sent, urls).read_text())
    urls.write_text(json.dumps({**doc, "inputs": {**doc["inputs"], "bytes": doc["inputs"]["bytes"] - 1}}))
    slept.clear()
    assert CHECK.lean_agreement_urls(out, urls, sleep=slept.append) == 1 and slept == list(CHECK.AGREEMENT_FETCH_WAITS_S)
    err = capsys.readouterr()
    n = doc["inputs"]["bytes"]
    assert (f"error: the pinned inputs (art:i): the {n} bytes fetched aren't the {n - 1} the store lists under its sha256 "
            f"({len(CHECK.AGREEMENT_FETCH_WAITS_S) + 1} tries)") in err.err
    assert "file:" not in err.out + err.err and not (out / "flock-agreement" / "argv.json").exists()
    urls.write_text(json.dumps({"upstream": doc["upstream"]}))
    assert CHECK.lean_agreement_urls(out, urls, sleep=slept.append) == 1
    assert f"error: {urls} lists upstream of the pinned upstream build and inputs" in capsys.readouterr().err
    urls.write_text("{")
    assert CHECK.lean_agreement_urls(out, urls, sleep=slept.append) == 1
    assert f"error: {urls} can't be read (JSONDecodeError)" in capsys.readouterr().err


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

    sets = json.loads((ROOT / "verity" / "protocols" / "flock" / "agreement" / "vectors.json").read_text())["replayable"]["sets"]
    assert len({s["pr83"] for s in sets if s.get("sessions") and "tables" not in s["statement"]}) >= 4


def test_the_rule_the_gate_reads(monkeypatch):
    spec2 = importlib.util.spec_from_file_location("check_tool", ROOT / "tools" / "verity" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(tool)
    assert tool.CHECK.merge_requires["lean-agreement"] == ["verity/protocols/flock/prover/", "verity/protocols/flock/pod/", "verity/protocols/flock/agreement/",
                                                           "verity/protocols/flock/tool.py", "verity/protocols/flock/verifier/",
                                                           "verity/circuits/base/", "verity/circuits/lean/",
                                                           "verity/commitments/lean/", "!README.md", "!PROTOCOL.md"]
    assert set(tool.CHECK.merge_requires) == {"lean-agreement", "lean-suites"}
    assert {"verity/protocols/flock/verifier/VStar/In.lean", "verity/protocols/recursion/verifier/vstar_cli/"} <= set(tool.CHECK.merge_requires["lean-suites"])
    pin = __import__("json").loads((ROOT / "verity" / "protocols" / "flock" / "agreement" / "upstream.json").read_text())
    assert set(pin["upstream"]["binaries"]) == {s["pr83"] for s in __import__("json").loads(
        (ROOT / "verity" / "protocols" / "flock" / "agreement" / "vectors.json").read_text())["replayable"]["sets"]}


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
    """A check of a PR's own head counts for nothing unless it is the tree that lands, since `research merge` takes a check only
    of that tree; `--record` warns before spending a slot on one."""
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
    monkeypatch.setattr(CHECK, "verdict_url_file", lambda d, **k: None)
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
    clean train passes.  Its verdicts are kept under the check's cache, and none under --no-cache."""
    import json
    import os
    import subprocess

    p01 = "adapters/vllm/tests/lint/allowlists/p01_core.json"
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
    with monkeypatch.context() as m:
        m.setattr(CHECK, "ratchet_gate", lambda out, cache: ["gate", str(out), str(cache)])
        assert [s[1] for g in CHECK.groups(out, 2) for s in g if s[0] == "ratchet-gate"] == [["gate", str(out), "True"]]
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


def test_the_ratchet_gate_step_fails_when_its_base_is_head_or_above_it(tmp_path, monkeypatch):
    """A run of `main`'s tip against origin/main, its own tip, judged no change and passed, so the step could never fail an
    after-landing run: a base at or above HEAD fails it.  A base below HEAD, or on a branch beside it (a PR behind `main`), is
    judged; one that names no commit here (the nightly's empty tree) skips it, which no after-landing run passes."""
    import os
    import subprocess

    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=tmp_path, env=env, check=True, capture_output=True, text=True).stdout.strip()

    def commit(message: str) -> str:
        git("commit", "-q", "--allow-empty", "-m", message)
        return git("rev-parse", "HEAD")

    git("init", "-q", "-b", "main")
    below = commit("verified main")
    head = commit("landing")
    above = commit("a later landing")
    git("checkout", "-q", "-b", "side", below)
    beside = commit("a PR's base")
    git("checkout", "-q", "--detach", head)
    monkeypatch.setattr(CHECK, "ROOT", tmp_path)
    out = tmp_path / "run"
    (out / "inputs").mkdir(parents=True)

    def gate(base: str) -> list[str] | str:
        (out / "inputs" / CHECK.BASE_FILE).write_text(f"{base}\n")
        return CHECK.ratchet_gate(out, True)

    for at in (head, above):
        why = gate(at)
        assert isinstance(why, str) and why.startswith(f"failed: its base {at} is HEAD or above it"), why
        rec = CHECK._step("ratchet-gate", why, tmp_path, out, True, CHECK.Steps(out))
        assert rec["status"] == "failed" and rec["detail"] == why
    for at in (below, beside):
        argv = gate(at)
        assert not isinstance(argv, str) and argv[4] == at, argv
    assert gate("0" * 40) == f"skipped: no base: {'0' * 40} names no commit here"
    (out / "inputs" / CHECK.RATCHET_BASE_FILE).write_text(f"{below}\n")    # --ratchet-base: the gate's, over --base's
    argv = CHECK.ratchet_gate(out, True)
    assert not isinstance(argv, str) and argv[4] == below, argv
    (out / "inputs" / CHECK.RATCHET_BASE_FILE).unlink()
    (out / "inputs" / CHECK.BASE_FILE).unlink()
    assert CHECK.ratchet_gate(out, True) == "skipped: no base: origin/main names no commit here"


def test_record_sends_the_verdict_packs_urls_with_a_pack_given_beside_them(tmp_path, monkeypatch):
    """A launch sends the newest recorded checks' packs as URLs the run fetches, never the packs: fetching and combining them on
    the launcher held each shard's launch some ten minutes.  `--verdicts-in` travels beside them, never in their place: a stale
    pack given once turned reuse off for every check it rode with, because the gathered pack it replaced held the passes.  An
    uncached launch sends them too, with the commit's own recorded checks' first (`commit_packs`): its fresh suites are
    compared with every pass the commit was recorded with, not only those its slot holds."""
    import subprocess
    import sys

    real, launched, listed, commits = subprocess.run, [], [], []

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            head = cmd[:cmd.index("--")]
            launched.append((cmd[cmd.index("--") + 1:], [head[i + 1] for i, x in enumerate(head) if x == "--send"], k["env"], head))
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    def url_file(d: Path, commit: str | None = None) -> Path:
        listed.append(d / CHECK.VERDICT_URLS)
        commits.append(commit)
        listed[-1].write_text("[]")
        return listed[-1]

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: None)
    monkeypatch.setattr(CHECK, "landable", lambda base: True)
    monkeypatch.setattr(CHECK, "slot_wrap", lambda priority: [])
    monkeypatch.setattr(CHECK, "verdict_url_file", url_file)
    monkeypatch.delenv(CHECK.VERDICT_URLS_ENV, raising=False)
    assert CHECK.record(["--base", "HEAD", "--on", "vy-x"]) == 0
    [(tail, sends, env, _)] = launched
    assert str(listed[-1]) in sends and "--verdicts-in" not in tail and CHECK.VERDICT_URLS_ENV not in env
    launched.clear()
    CHECK.VERDICTS._write(mine := tmp_path / "mine.tar.gz", [("tests/k.json", b"{}")])
    with monkeypatch.context() as here:                                 # the run starts elsewhere: it's given the path whole
        here.chdir(tmp_path)
        assert CHECK.record(["--base", "HEAD", "--on", "vy-x", "--verdicts-in", "mine.tar.gz"]) == 0
    [(tail, sends, _, _)] = launched
    mine = mine.resolve()
    assert tail[tail.index("--verdicts-in") + 1] == str(mine) and str(mine) in sends and str(listed[-1]) in sends
    launched.clear()
    assert CHECK.record(["--base", "HEAD"]) == 0                                       # a local recorded run
    [(tail, sends, env, _)] = launched
    assert env[CHECK.VERDICT_URLS_ENV] == str(listed[-1]) and sends == []
    launched.clear()
    assert CHECK.record(["--base", "HEAD", "--no-cache"]) == 0 and len(listed) == 4      # the nightly `audit-main`
    [(_, sends, env, head)] = launched
    assert env[CHECK.VERDICT_URLS_ENV] == str(listed[-1]) and sends == []
    assert [head[i + 1] for i, x in enumerate(head) if x == "--declared-output"] == [*CHECK.PACK_FILES, "check-live.json"], \
        "custody uploads the packs and the live record however the run ends, never trimming them as undeclared"
    launched.clear()
    # drive's after-landing run: circuit-check's passes come from the packs sent (main imports only theirs and the suites')
    assert CHECK.record(["--base", "HEAD", "--on", "vy-x", "--no-cache", "--circuit-cache"]) == 0 and len(listed) == 5
    [(tail, sends, _, _)] = launched
    assert tail[-2:] == ["--no-cache", "--circuit-cache"] and str(listed[-1]) in sends
    assert commits == [None, None, None, CHECK._rev("HEAD"), CHECK._rev("HEAD")] and commits[-1]
    (bad := tmp_path / "bad.tar.gz").write_bytes(b"not a pack")
    for given, why in ((bad, "refused"), (tmp_path / "absent.tar.gz", "no such pack")):
        with pytest.raises(SystemExit, match=why):
            CHECK.record(["--base", "HEAD", "--on", "vy-x", "--verdicts-in", str(given)])


class _Remote:
    """The store's remote as `agreement_url_file` reads it: it holds every object at ``n`` bytes (none without ``held``) and signs
    a URL of each, keeping each signature; a read of an object fails the test."""

    def __init__(self, n: int = 7 << 20, held: bool = True):
        self.n, self.held, self.signed = n, held, []

    def head(self, key: str) -> dict | None:
        return {"bytes": self.n, "etag": None} if self.held else None

    def presign(self, key: str, expires_s: int) -> str:
        self.signed.append((key, expires_s))
        return f"https://store.invalid/{key}?X-Amz-Signature=signed"

    def get(self, key: str, path: Path) -> None:
        raise AssertionError(f"the launch read {key}")


def _remote(monkeypatch, remote) -> None:
    import research.store.remote

    monkeypatch.setattr(research.store.remote, "load_remote", lambda path=None: remote)


def test_a_launch_that_needs_the_agreement_sends_its_urls_and_fetches_nothing(tmp_path, monkeypatch, capsys):
    """The launch fetched the pinned upstream build and inputs from the store to send with the run: minutes of each shard's
    launch on the control pod.  It now reads each one's size from the store's remote and signs a GET of it for
    `AGREEMENT_URL_TTL_S`, in a file only its owner reads, sent with the run or, for a local one, named by
    `AGREEMENT_URLS_ENV`; it opens no local store, reads no object and prints no URL.  A remote that can't sign, can't be read
    or doesn't hold a pinned file refuses the launch, whose agreement would otherwise fail on the machine."""
    import tempfile
    import urllib.error

    import research.store.local
    from research.store.local import object_key

    real, launched, sent, opened = subprocess.run, [], {}, []

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            launched.append(cmd)
            named = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--send" and cmd[i + 1].endswith(CHECK.AGREEMENT_URLS)]
            for p in map(Path, named + ([u] if (u := (k.get("env") or {}).get(CHECK.AGREEMENT_URLS_ENV)) else [])):
                sent[p] = (json.loads(p.read_text()), p.stat().st_mode & 0o777)
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    class Store:
        def __init__(self, *a, **k):
            opened.append(a)
            raise AssertionError("the launch opened the local store")

    (scratch := tmp_path / "scratch").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(research.store.local, "LocalStore", Store)
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: None)
    monkeypatch.setattr(CHECK, "verdict_url_file", lambda d, **k: None)
    monkeypatch.delenv(CHECK.AGREEMENT_URLS_ENV, raising=False)
    _remote(monkeypatch, remote := _Remote())
    pin = json.loads(CHECK.UPSTREAM_PIN.read_text())
    want = {k: {"art": pin[k]["art"], "sha256": pin[k]["sha256"], "bytes": 7 << 20,
                "url": f"https://store.invalid/{object_key(pin[k]['sha256'])}?X-Amz-Signature=signed"} for k in ("upstream", "inputs")}
    assert CHECK.record(["--on", "vy-x", "--base", "HEAD", "--agreement"]) == 0
    [cmd] = launched
    [(path, (doc, mode))] = sent.items()
    assert path.name == CHECK.AGREEMENT_URLS and doc == want and mode == 0o600 and str(path) in cmd
    assert [cmd[i + 1] for i, x in enumerate(cmd) if x == "--input"] == [f"flock_{k}={pin[k]['art']}" for k in ("upstream", "inputs")]
    assert remote.signed == [(object_key(pin[k]["sha256"]), CHECK.AGREEMENT_URL_TTL_S) for k in ("upstream", "inputs")]
    out = capsys.readouterr().out
    assert "sending presigned URLs for the pinned upstream build and inputs (14 MiB, which the run fetches)" in out
    assert "X-Amz-Signature" not in out and opened == [] and not list(scratch.iterdir())
    launched.clear(), sent.clear()
    assert CHECK.record(["--base", "HEAD", "--agreement"]) == 0                        # a local recorded run
    [(path, (doc, _))] = sent.items()
    assert doc == want and CHECK.AGREEMENT_URLS not in " ".join(launched[0])

    class Unreadable(_Remote):
        def head(self, key: str) -> dict | None:
            raise urllib.error.HTTPError(f"https://store.invalid/{key}", 403, "Forbidden", {}, None)

    launched.clear()
    for given, why in ((_Remote(held=False), f"the store's remote doesn't hold the pinned upstream ({pin['upstream']['art']})"),
                       (Unreadable(), "the store's remote can't be read (HTTPError 403)"), (None, "no store remote here signs URLs")):
        _remote(monkeypatch, given)
        assert CHECK.record(["--on", "vy-x", "--base", "HEAD", "--agreement"]) == 2
        assert (f"check: --record refused: the run needs lean-agreement, which fetches the pinned upstream build and inputs "
                f"through presigned URLs, and {why}; nothing launched") in capsys.readouterr().err
        assert CHECK.main(["--agreement-files"]) == 1 and f"check: --agreement-files: {why}" in capsys.readouterr().err
    assert launched == [] and opened == [] and not list(scratch.iterdir())
    _remote(monkeypatch, _Remote())
    assert CHECK.main(["--agreement-files"]) == 0
    printed = capsys.readouterr().out.split()
    assert printed[0] == "--send" and json.loads(Path(printed[1]).read_text()) == want


def test_record_sends_the_agreement_for_a_file_renamed_out_of_its_paths(tmp_path, monkeypatch, capsys):
    """git reports a rename by its destination alone unless asked not to, which would leave out the step the file's old path
    requires."""
    import json
    import subprocess
    import sys
    import tempfile

    def git(*args):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=tmp_path,
                       check=True, capture_output=True)

    (tmp_path / "tools" / "verity" / "check").mkdir(parents=True)
    (tmp_path / "tools" / "verity" / "check" / "tool.py").write_text(
        "from research.store.tool import Tool\nCHECK = Tool(name='check', version='1', command=['true'], "
        "merge_requires={'lean-agreement': ['verity/protocols/flock/prover/', '!README.md']})\n")
    (tmp_path / "verity" / "protocols" / "flock" / "prover").mkdir(parents=True)
    (tmp_path / "verity" / "protocols" / "flock" / "prover" / "verify.rs").write_text("fn main() {}\n")
    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    git("mv", "verity/protocols/flock/prover/verify.rs", "verify.rs")
    git("commit", "-q", "-m", "move")
    (tmp_path / "upstream.json").write_text(json.dumps({"upstream": {"art": "art:u", "sha256": "a" * 64},
                                                        "inputs": {"art": "art:i", "sha256": "b" * 64}}))
    real, launched = subprocess.run, []

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            launched.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    monkeypatch.setattr(CHECK, "ROOT", tmp_path)
    monkeypatch.setattr(CHECK, "UPSTREAM_PIN", tmp_path / "upstream.json")
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: None)
    monkeypatch.setattr(CHECK, "verdict_url_file", lambda d, **k: None)
    _remote(monkeypatch, _Remote())
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", run)
    assert CHECK.record(["--base", "HEAD^"]) == 0
    [cmd] = launched
    assert "flock_upstream=art:u" in cmd and "sending presigned URLs for the pinned upstream build" in capsys.readouterr().out


def test_check_refuses_a_python_that_imports_the_trees_packages_from_elsewhere(monkeypatch, capsys):
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    monkeypatch.setenv("PYTHONPATH", f"/tmp/s1350/packages/verity/src:{ROOT / 'tools/research/src'}")
    probed = []
    monkeypatch.setattr(CHECK.SUITES, "foreign", lambda pkgs, env: probed.append(env.get("PYTHONPATH")) or "verity from /tmp/s1350/x.py")
    monkeypatch.setattr(CHECK, "run", lambda *a: pytest.fail("a step ran"))
    assert CHECK.main([]) == 2
    assert "against another checkout's code (verity from /tmp/s1350/x.py)" in capsys.readouterr().err
    assert probed == [str(ROOT / "tools/research/src")]


def test_check_run_as_a_script_in_a_fresh_trees_environment_installs_the_workspace_first(monkeypatch, capsys):
    """`uv run --extra torch-cpu python tools/verity/check/check.py` installs the root, torch and the dev group, not every member."""
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    monkeypatch.setattr(CHECK.SUITES, "foreign", lambda pkgs, env: "verity_identity from nowhere (not importable)")
    monkeypatch.setattr(CHECK.SUITES, "fresh", lambda env: True)
    monkeypatch.setattr(CHECK, "run", lambda *a: pytest.fail("a step ran"))
    resynced = []
    monkeypatch.setattr(CHECK, "resync", resynced.append)
    assert CHECK.main(["--cores", "4"], script=True) == 2 and resynced == [["--cores", "4"]]
    assert f"`{' '.join(CHECK.SUITES.SYNC)} && uv run --extra torch-cpu python" in capsys.readouterr().err
    assert CHECK.main([]) == 2 and len(resynced) == 1


def test_check_resyncs_once_and_runs_again_only_where_the_sync_filled_the_environment(monkeypatch):
    synced, ran, fresh = [], [], [False]
    monkeypatch.setattr(CHECK.subprocess, "run", lambda cmd, cwd: synced.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setattr(CHECK.os, "execv", lambda exe, args: ran.append(args))
    monkeypatch.setattr(CHECK.SUITES, "fresh", lambda env: fresh[0])
    monkeypatch.setattr(sys, "argv", ["tools/verity/check/check.py", "--cores", "4"])
    CHECK.resync(["--cores", "4"])
    assert synced == [CHECK.SUITES.SYNC] and ran == [[sys.executable, "tools/verity/check/check.py", "--cores", "4"]]
    fresh[0] = True
    CHECK.resync(["--cores", "4"])
    assert len(synced) == 2 and len(ran) == 1


def test_a_recorded_run_refuses_to_skip_the_store(monkeypatch):
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setenv("VERITY_SKIP_STORE", "1")
    assert CHECK.main([]) == 2

def test_circuit_check_has_the_whole_slot_only_when_neither_pytest_nor_lean_runs_beside_it(tmp_path, monkeypatch):
    """An unsharded check that gives circuit-check the slot's whole 560 GiB while pytest and lean-audit hold 160 to 350 GiB
    beside it gets a 141 GiB circuit-check worker OOM-killed."""
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
    """An after-landing run (`--deferred circuit-check,pytest`) runs no Lean step, so circuit-check getting 210 GiB of a 560 GiB
    slot while pytest kept to its half of the rest, 175, would leave lean-suites' 175 unused and run the 147-167 GiB SHA-512
    targets one at a time.  Pytest's share is unchanged."""
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
    """pytest can end long before circuit-check does, which must not keep to the share it had beside pytest to the end.
    Beside other steps it gets ``--mem-gb-file``, which `run` writes the slot's mem= to once those
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
    assert mem(["pytest", "lean-build"]) == "280", "halved beside a Lean step"
    assert mem(["pytest"]) == "560", "a shard of pytest alone (the 5-pod plan's 4/5) halved the slot for a lean-suites it lacks"
    monkeypatch.delenv(CHECK.SLOT.SLOT_MEM_ENV)
    [pt] = [s for g in CHECK.groups(tmp_path, 16) for s in g if s[0] == "pytest"]
    assert "--mem-gb" not in pt[1]


def test_the_groups_split_the_cores_and_run_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(CHECK, "FLOCK_LEAN", tmp_path / "verifier-lean")
    monkeypatch.setattr(CHECK, "memory_gb", lambda: 2048)
    monkeypatch.delenv(CHECK.SLOT.SLOT_MEM_ENV, raising=False)
    assert "--mem-gb" not in CHECK.groups(tmp_path, 16)[1][2][1]
    monkeypatch.setenv(CHECK.SLOT.SLOT_MEM_ENV, "560")
    (pt,), (build, rust, cc), (lean,), (gate,) = CHECK.groups(tmp_path, 16)
    assert cc[1][cc[1].index("--mem-gb") + 1] == "210", "a slot's mem=, less what pytest and Lean hold beside it, caps circuit-check"
    assert build[0] == "flock-circuit-build" and build[1][-1].endswith("verity/protocols/flock/prover/check_build.sh")
    assert rust[0] == "flock-rust-tests" and rust[1][-2:] == [build[1][-1], "test"] and "CARGO_BUILD_JOBS=8" in rust[1]
    assert pt[1][1].endswith("tools/verity/check/suites.py") and pt[1][pt[1].index("--jobs") + 1] == "8" and "--fresh" not in pt[1]
    assert cc[1][cc[1].index("--jobs") + 1] == "8" and "--cache" in cc[1]
    cold = CHECK.groups(tmp_path, 16, cache=False)
    assert lean[0] == "lean" and "--cache" not in cold[1][0][1] and "--fresh" in " ".join(cold[0][0][1])
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
    concurrent Lean group has got (otherwise verity-flock passes or fails on build order). The audit builds only in scratch
    trees of its own, so it is a group of its own and nothing in the Lean group waits on it. The suites need only the verifier
    lean-build built, so they wait on it alone and run beside the agreement."""
    out, _ = _pinned(tmp_path, monkeypatch)
    gs = CHECK.groups(out, 8)
    (_, pytest, _), = gs[0]
    lean = gs[2]
    assert [n for n, _, _ in lean] == ["lean-build", "lean-unit-cut", "lean-agreement"]
    assert [n for n, _, _ in gs[3]] == ["lean-audit"] and gs[3][0][1][4].endswith("tools/verity/lean/lean_audit.py")
    assert [n for n, _, _ in gs[4]] == ["lean-suites"] and CHECK.AFTER["lean-suites"] == "lean-build"
    assert all(pytest[pytest.index(s) - 1] == "--skip" for s in CHECK.AFTER_LEAN)
    argv = gs[4][0][1]
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
    assert CHECK._env_key({"PYTHONPATH": f"{ROOT}/x", "VIRTUAL_ENV": f"{ROOT}/.venv", "SSH_CLIENT": "1 2 3"}) == {"PYTHONPATH": "<tree>/x"}


def test_the_agreement_is_not_required_for_docs_alone():
    """README.md and PROTOCOL.md under verity/protocols/flock/prover/ are docs the agreement never reads: alone they do not require it."""
    from research.merge import rule_touches

    spec2 = importlib.util.spec_from_file_location("check_tool", ROOT / "tools" / "verity" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(tool)
    rule = tool.CHECK.merge_requires["lean-agreement"]
    docs = ["verity/protocols/flock/prover/live/PROTOCOL.md", "verity/protocols/flock/agreement/README.md", "verity/protocols/flock/prover/README.md", "AGENTS.md"]
    assert rule_touches(rule, docs) == []
    assert rule_touches(rule, [*docs, "verity/protocols/flock/agreement/ci.py"]) == ["verity/protocols/flock/agreement/"]
    assert rule_touches(rule, ["verity/protocols/flock/agreement/DESIGN.md"]) == ["verity/protocols/flock/agreement/"]  # only the two names are left out
    for f in ("ci.py", "agree.py", "fuzz.py", "redigest.py", "transcript_check.py", "selftest_records.py"):   # cited, never read
        reads = [line for line in (ROOT / "verity" / "protocols" / "flock" / "agreement" / f).read_text().splitlines()
                 if any(r in line for r in ("open(", "read_text", "read_bytes")) and any(d in line for d in ("README", "PROTOCOL"))]
        assert not reads, f"{f} reads a doc the agreement trigger leaves out: {reads}"


def test_the_agreement_scripts_import_only_the_standard_library_and_each_other():
    """`agreement_key` covers verity/protocols/flock/agreement/ and the interpreter only: an import from elsewhere in `ci.py` or the scripts
    it runs would be an input the key misses, so one fails here until the key covers it.  The key also leaves out the Lean
    package's `lean-audit.json` (`agreement_closure`), so neither those scripts nor the package's lakefile may mention it."""
    import ast
    import re
    import sys

    here = ROOT / "verity" / "protocols" / "flock" / "agreement"
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
    """`flock-verify` is built from the Lean package alone: its proofs (`security/Proofs/Flock/`) and any Lake package
    nested in it are no input, unless the package's lakefile requires something, which keeps everything in.  Nor is the
    package's `lean-audit.json`, which only the audit reads: that one path, not every file of the name."""
    files = CHECK.tracked_files()
    closure = CHECK.agreement_closure(files)
    policy, other = "verity/protocols/flock/verifier/lean-audit.json", "verity/protocols/flock/verifier/Flock/lean-audit.json"
    assert policy not in CHECK.agreement_closure([*files, policy]) and other in CHECK.agreement_closure([*files, other])
    assert "verity/protocols/flock/agreement/ci.py" in closure and "verity/protocols/flock/verifier/FlockVerify.lean" in closure
    assert "verity/protocols/flock/agreement/upstream.json" in closure
    assert not [f for f in closure if f.startswith("security/")]
    assert [f for f in files if f.startswith("security/Proofs/Flock/")]
    nested = ["verity/protocols/flock/verifier/lakefile.toml", "verity/protocols/flock/verifier/nested/lakefile.toml",
              "verity/protocols/flock/verifier/nested/N.lean"]
    (tmp_path / "verity/protocols/flock/verifier/nested").mkdir(parents=True)
    (tmp_path / nested[0]).write_text("")
    (tmp_path / nested[1]).write_text("")
    monkeypatch.setattr(CHECK, "ROOT", tmp_path)
    assert CHECK.agreement_closure(nested) == nested[:1]
    (tmp_path / nested[0]).write_text('[[require]]\nname = "x"\n')
    assert CHECK.agreement_closure(nested) == nested


def test_a_doc_edit_keeps_the_agreements_key_and_a_script_or_source_edit_moves_it(tmp_path, monkeypatch):
    """Neither Lake nor a script of the agreement reads Markdown (`test_the_agreement_is_not_required_for_docs_alone` holds
    the scripts to it), so `agreement_key` leaves it out: a README edit keeps the stored pass, which a recorded check then
    reuses, while an edit to a script or to the verifier's Lean moves the key."""
    agree, lean = "verity/protocols/flock/agreement/", "verity/protocols/flock/verifier/"
    files = {agree + "README.md": "# the job\n", agree + "ci.py": "x = 1\n", lean + "lakefile.toml": "",
             lean + "README.md": "# the verifier\n", lean + "PROTOCOL.md": "# its wire\n", lean + "Flock.lean": "def a := 1\n"}
    for f, text in files.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text)
    monkeypatch.setattr(CHECK, "ROOT", tmp_path)
    monkeypatch.setattr(CHECK, "tracked_files", lambda *_: sorted(files))
    assert CHECK.agreement_closure(sorted(files)) == [agree + "ci.py", lean + "Flock.lean", lean + "lakefile.toml"]
    key = CHECK.agreement_key({})
    for doc in (agree + "README.md", lean + "README.md", lean + "PROTOCOL.md"):
        (tmp_path / doc).write_text("# rewritten\n")
        assert CHECK.agreement_key({}) == key, f"{doc}: a doc edit keeps the agreement's pass"
    for code in (agree + "ci.py", lean + "Flock.lean"):
        (tmp_path / code).write_text("changed\n")
        assert CHECK.agreement_key({}) != key, f"{code}: what the job runs moves its key"
        key = CHECK.agreement_key({})


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


def test_the_agreements_smoke_replay_expects_upstreams_own_verdict_where_a_case_records_a_divergence(tmp_path, monkeypatch):
    """A case that records a divergence (`upstream_accepted`: an honest session Lean refuses for completeness and upstream
    accepts) is held to upstream's own verdict: held to Lean's, it would fail the preflight. A case without the key expects
    its one verdict of both."""
    v = tmp_path / "verifier"
    v.mkdir()
    (v / "agree.py").write_text("def drawn(d):\n    return False\n\ndef upstream_cmd(a, public, extra):\n    return [a.upstream]\n")
    sets = [{"pr83": c, "lean_statement": "s", "statement": {"circuit": f"art:{c}/c", "public": f"art:{c}/p"},
             "sessions": [{"dir": f"art:{c}/honest", "expect": expect}]}
            for c, expect in (("aa", {"accepted": False, "why": "setup: registered reads (complete): ", "upstream_accepted": True}),
                              ("bb", {"accepted": True, "why": ""}))]
    (v / "vectors.json").write_text(json.dumps({"replayable": {"sets": sets}}))
    monkeypatch.setattr(CHECK, "FLOCK_VERIFIER", v)
    assert {c: e for c, _, e in CHECK.upstream_smoke(tmp_path)} == {"aa": {"accepted": True}, "bb": {"accepted": True}}


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
    """drive's `research cancel` signals the run's whole process group, and a step can die of it before check's own handler
    marks it stopped; it is recorded stopped, not failed with an empty log, a failure the step never found."""
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
    """Under --keep-going after lean-suites fails, pytest's suites.py can sit for a day on a test that never returns, writing
    nothing, while circuit-check waits on the grant pytest's end writes and the run holds its slot.  Here pytest's process
    waits on a child that never exits; once the watch's clock (injected) has passed `IDLE_S` with pytest's outputs unchanged,
    the child and pytest are stopped, pytest fails as hung, the grant is written and the run ends."""
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
    writes its result, `not_run`, with the pack of the pass `passes` stored before it, and exits 143, so a cancelled check's
    passes are not run again."""
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
    nothing a later plan could cite.  `Reused` packs
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
    steps.end({"step": "flock-circuit-build", "status": "failed", "halts": True})
    assert steps.stopping and steps.halted == "flock-circuit-build"
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


def test_a_check_s_passes_are_the_packages_its_audit_passed_on_the_commit_s_own_lock_by_their_inputs_keys(tmp_path, monkeypatch):
    """result.json's `passes` ({inputs key: package}): each package the lean-audit step passed with the kernel replay on the
    commit's own lock, under the key `lean_changed.py --inputs-key` prints for it at that commit, which a records run of the
    commit keeps too, so `research queue`'s records gate finds a check's pass as it finds a records run's.  None of a
    package that failed, passed only on a record it regenerated or without the replay; none at all from a refused or stopped
    step or an audit that crashed."""
    import subprocess
    import sys

    lc = CHECK._load("lean_changed", CHECK.LEAN_JOBS)
    root = tmp_path / "repo"
    files = {"a/lakefile.toml": "", "a/lean-audit.json": "{}", "a/A.lean": "def a := 1\n",
             "a/lake-manifest.json": json.dumps({"packages": [{"name": "p", "type": "path", "dir": "../p"}]}),
             "p/lakefile.toml": "", "p/lean-audit.json": "{}", "p/P.lean": "def p := 1\n",
             "c/lakefile.toml": "", "c/lean-audit.json": "{}", "d/lakefile.toml": "", "d/lean-audit.json": "{}"}
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "packages"]):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=root,
                       check=True, capture_output=True)
    for m in (CHECK, lc, lc.LA):
        monkeypatch.setattr(m, "ROOT", root)
    full = {"failures": [], "seconds": {"build": 1.0, "audit": 1.0, "replay": 1.0}}
    audit = {"passed": False, "failed": ["p"], "cached": {"a": "k"}, "regenerated": {"c": "c/lean-audit.json"},
             "packages": [{**full, "package": "a"}, {**full, "package": "p", "failures": ["audit: no"]},
                          {**full, "package": "c", "settled": True, "regenerated": "c/lean-audit.json"},
                          {**full, "package": "d", "seconds": {"build": 1.0, "audit": 1.0}}]}
    (tmp_path / "lean-audit").mkdir()
    (tmp_path / "lean-audit" / "audit.json").write_text(json.dumps(audit))
    step = {"step": "lean-audit", "status": "failed", "rc": 1, "log": "lean-audit.log", "seconds": 1.0}
    printed = subprocess.run([sys.executable, str(CHECK.LEAN_JOBS / "lean_changed.py"), "--inputs-key", "a", "--repo", str(root)],
                             check=True, capture_output=True, text=True).stdout
    key = json.loads(printed)["a"]
    assert CHECK.lean_passes(tmp_path, [step], lc) == {key: "a"}
    for bad in ({**step, "detail": "refused: a: a lean-fast/v1 result"}, {**step, "status": "stopped", "rc": -15},
                {**step, "detail": "hung: no output for 30 min"}):
        assert CHECK.lean_passes(tmp_path, [bad], lc) == {}, bad
    (tmp_path / "lean-audit" / "audit.json").write_text(json.dumps({**audit, "failed": ["p", "check.py exit 2"]}))
    assert CHECK.lean_passes(tmp_path, [step], lc) == {}, "check.py crashed: what it reported isn't every verdict"
    (tmp_path / "lean-audit" / "audit.json").write_text(json.dumps(audit))
    (root / "a" / "A.lean").write_text("def a := 2\n")
    got = CHECK.lean_passes(tmp_path, [step], lc)
    assert list(got.values()) == ["a"] and key not in got, "keyed on the files the audit read: here a change not committed"

    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    monkeypatch.setattr(CHECK, "lean_passes", lambda out, done, lc=None: {key: "a"})
    CHECK.write_result(tmp_path, False, [step])
    assert json.loads((tmp_path / "result.json").read_text())["passes"] == {key: "a"}
    CHECK.write_result(tmp_path, False, [step], stopped_by="SIGTERM")
    assert json.loads((tmp_path / "result.json").read_text())["passes"] == {}, "a stopped run has no verdict"


def test_the_audit_is_a_group_of_its_own_and_a_tree_without_the_verifier_has_no_lean_group_but_a_skip(tmp_path, monkeypatch):
    out, _ = _pinned(tmp_path, monkeypatch)
    gs = CHECK.groups(out, 8)
    assert len(gs) == 6 and [n for n, _, _ in gs[3]] == ["lean-audit"]
    assert not any(str(a).endswith("lean_changed.py") for g in gs for _, argv, _ in g if not isinstance(argv, str) for a in argv), \
        "no step runs the fast path's file: without a mode it runs nothing"
    monkeypatch.setattr(CHECK, "FLOCK_LEAN", tmp_path / "no-verifier")
    assert len(CHECK.groups(out, 8)) == 4


def test_circuit_cache_keeps_circuit_checks_reuse_alone_under_no_cache(tmp_path, monkeypatch):
    """Drive's after-landing run (`--no-cache --circuit-cache`): the suites and the Lean steps run cold, and circuit-check reuses
    its per-target passes, since a cold `--all` (~4 h) outlasts every after-landing run.  `--no-cache` alone (the
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


def test_an_uncached_check_runs_its_steps_with_lakes_artifact_cache_off(tmp_path, monkeypatch):
    """`--no-cache` compiles what it tests: its steps (the verifier's `lake build`, and the Lean steps and suites after it,
    which build and run Lean too) run with Lake's local artifact cache off, where Lake would unpack what an earlier build on
    the node left there instead of compiling it.  A cached check's steps get no Lake setting, the caller's neither."""
    monkeypatch.setenv("LAKE_ARTIFACT_CACHE", "true")
    names = ("lean-build", "lean-suites", "pytest")
    for cache, want in ((False, "false"), (True, "unset")):
        out = tmp_path / f"cache-{cache}"
        out.mkdir()
        probe = "import os, sys; open(sys.argv[1], 'w').write(os.environ.get('LAKE_ARTIFACT_CACHE', 'unset'))"
        _fake_run(monkeypatch, [[(s, _py(probe, out / f"{s}.said"), tmp_path)] for s in names])
        ok, done = CHECK.run(out, 2, cache)
        assert ok, done
        assert {s: (out / f"{s}.said").read_text() for s in names} == dict.fromkeys(names, want)


def test_a_cold_check_runs_the_opt_in_tests_after_their_steps_suites_and_no_other_check_does(tmp_path, monkeypatch):
    """The nightly (`--no-cache`, unsharded): pytest and lean-suites each run their suites, then the opt-in tests of those suites
    (`OPT_IN`) in runs of their own.  A shard's or a deferred run's parts and a cached check leave them out, so no candidate pays
    for them and no suite's key holds their variables."""
    out, _ = _pinned(tmp_path, monkeypatch)

    def steps(**kw) -> dict:
        return {name: argv for g in CHECK.groups(out, 8, **kw) for name, argv, _ in g}

    cold = steps(cache=False)
    for step, then in (("pytest", "pytest --timeouts"), ("lean-suites", "lean-suites --lean-built --timeouts")):
        bash, flag, cmd = cold[step]
        suites, opt_in, end = cmd.split(" || rc=$?; ")
        assert (bash, flag, end) == ("bash", "-c", "exit $rc") and "suites.py" in suites and "--opt-in" not in suites, step
        assert f"check.py --opt-in {out} {then} --jobs 4 --fresh" in opt_in, step
    for kw in ({}, {"cache": False, "only": ["pytest", "lean-suites"]}, {"cache": False, "circuit_cache": True, "only": ["pytest"]}):
        assert not any("--opt-in" in " ".join(a) for a in steps(**kw).values() if isinstance(a, list)), kw


def test_the_opt_in_tests_run_with_their_variable_set_and_fail_their_step_unless_each_passed(tmp_path, monkeypatch, capsys):
    """Each suite's named tests in one `suites.py` run of its own with its variables set, every one of them, and no other
    suite's.  A named test that skipped (its mark reads a variable its suite doesn't list) or that its suite lacks (pytest's
    "not found") ran nowhere, so it fails the step."""
    from collections import namedtuple

    suite = namedtuple("suite", "name path")
    monkeypatch.setattr(CHECK, "OPT_IN", {"a/one": {"VERITY_A_SLOW": ("tests/test_a.py::test_x",),
                                                    "VERITY_A_E2E": ("tests/test_a.py::test_y",)},
                                          "b/two": {"VERITY_B_SLOW": ("tests/test_b.py::test_z",)}})
    monkeypatch.setattr(CHECK.SUITES, "discover", lambda: [suite("one", "a/one"), suite("two", "b/two")])
    monkeypatch.setattr(CHECK, "AFTER_LEAN", ("two",))
    ran: list = []
    outcome = {"a/one": (0, {"passed": 2, "failed": 0, "error": 0, "skipped": 0}),
               "b/two": (0, {"passed": 1, "failed": 0, "error": 0, "skipped": 0})}

    def run(cmd, cwd, env):
        suite = cmd[2]
        ran.append((suite, [v for v in ("VERITY_A_SLOW", "VERITY_A_E2E", "VERITY_B_SLOW") if env.get(v) == "1"],
                    cmd[cmd.index("--") + 1:]))
        rc, counts = outcome[suite]
        if counts is not None:
            Path(cmd[cmd.index("--json") + 1]).write_text(json.dumps([{"suite": suite, "status": "passed", "counts": counts}]))
        return subprocess.CompletedProcess(cmd, rc)

    monkeypatch.setattr(CHECK.subprocess, "run", run)
    assert CHECK.opt_in(tmp_path, "pytest", ["--fresh"]) == 0
    assert ran == [("a/one", ["VERITY_A_SLOW", "VERITY_A_E2E"], ["tests/test_a.py::test_x", "tests/test_a.py::test_y"])], \
        "pytest's suites alone, each in one run"
    ran.clear()
    assert CHECK.opt_in(tmp_path, "lean-suites", []) == 0 and ran == [("b/two", ["VERITY_B_SLOW"], ["tests/test_b.py::test_z"])]
    capsys.readouterr()
    outcome["a/one"] = (0, {"passed": 1, "failed": 0, "error": 0, "skipped": 1})
    outcome["b/two"] = (4, None)
    assert CHECK.opt_in(tmp_path, "all", []) == 1
    said = capsys.readouterr().out
    assert "a/one: 1 of the 2 tests named with VERITY_A_SLOW=1, VERITY_A_E2E=1 passed (1 skipped)" in said
    assert "b/two: 0 of the 1 tests named with VERITY_B_SLOW=1 passed (suites.py exited 4)" in said
    assert CHECK.opt_in(tmp_path, "circuit-check", []) == 2


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


def test_a_prebuild_that_fails_refuses_the_run_with_its_output_and_exit_status_even_under_keep_going(tmp_path, monkeypatch):
    """The C++ models' prebuild ran with its output captured and its exit ignored, and fell back to building nothing where the
    twins didn't import, so a check whose models didn't build ran on, each step compiling them untraced: it refuses now."""
    import json

    assert "except" not in CHECK.PREBUILD, "a prebuild whose models don't import fails, never falls back to building none"
    marker = tmp_path / "ran"
    touch = _py("open(__import__('sys').argv[1], 'w')", marker)
    for keep_going in (False, True):
        _fake_run(monkeypatch, [[("pytest", touch, tmp_path)]], pre=[("preflight-lock", _py("pass"), tmp_path)])
        monkeypatch.setattr(CHECK, "PREBUILD", "import sys\nprint('compiling tc_model.cpp', flush=True)\nsys.exit('g++: not found')")
        ok, done = CHECK.run(tmp_path, 2, keep_going=keep_going)
        assert not ok and not marker.exists(), keep_going
        assert [(s["step"], s["status"], s.get("rc")) for s in done] == [("preflight-lock", "passed", 0), ("prebuild", "failed", 1)]
        assert done[1]["detail"].startswith("refused: the C++ models' prebuild exited 1 (g++: not found; prebuild.log)")
        assert (tmp_path / "prebuild.log").read_text().splitlines() == ["compiling tc_model.cpp", "g++: not found"]
        assert json.loads((tmp_path / "steps.jsonl").read_text().splitlines()[-1])["step"] == "prebuild"
    _fake_run(monkeypatch, [[("pytest", touch, tmp_path)]], pre=[("preflight-lock", _py("pass"), tmp_path)])
    monkeypatch.setattr(CHECK, "PREBUILD", "print('{\"tc_model\": \"libtc_model.so\"}')")
    ok, done = CHECK.run(tmp_path, 2)
    assert ok and marker.exists() and [s["step"] for s in done] == ["preflight-lock", "pytest"]
    assert (tmp_path / "prebuild.log").read_text() == '{"tc_model": "libtc_model.so"}\n'


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
    """A fixture registered, preserved privately and never exported is neither in a node's store nor in the public export, and
    a check there would fail `lean-suites` after its launch. A fixture the tree already holds exactly or this machine's store
    holds whole needs no export (a node's own copies keep trains moving), though the export is asked first (one HEAD per
    manifest, no blob), so the ones it lacks are named. A tracked fixture and the `previous` history are not asked about,
    and with no public source to ask it refuses rather than pass."""
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
    """A train's prep runs `--preflight` off the node its check is pinned to, so a fixture only that node's store holds mustn't
    refuse the train there: the check's own preflight-fixtures judges the node, and refuses on it."""
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
    """`git diff --name-only` quotes a name with bytes outside ASCII (`"verity/protocols/flock/prover/\\303\\251.py"`), which no prefix of the
    agreement's rule matched, so `--record` checked such a change without sending the agreement's pinned build."""
    import shutil
    import subprocess
    import sys
    import tempfile

    def git(*args: str) -> None:
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=tmp_path,
                       check=True, capture_output=True)

    (tmp_path / "tools" / "verity" / "check").mkdir(parents=True)
    shutil.copy(ROOT / "tools" / "verity" / "check" / "tool.py", tmp_path / "tools" / "verity" / "check" / "tool.py")
    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    (tmp_path / "verity" / "protocols" / "flock" / "prover").mkdir(parents=True)
    (tmp_path / "verity" / "protocols" / "flock" / "prover" / "é.py").write_text("x = 1\n")
    git("checkout", "-q", "-b", "pr")
    git("add", "-A")
    git("commit", "-q", "-m", "é")
    real, launched = subprocess.run, []

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            launched.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    monkeypatch.setattr(CHECK, "ROOT", tmp_path)
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: None)
    monkeypatch.setattr(CHECK, "verdict_url_file", lambda d, **k: None)
    _remote(monkeypatch, _Remote())
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", run)
    assert CHECK.record(["--base", "main"]) == 0
    assert ("check: changes under verity/protocols/flock/prover/, verity/protocols/flock/pod/, verity/protocols/flock/agreement/, verity/protocols/flock/tool.py, "
            "verity/protocols/flock/verifier/, verity/circuits/base/, verity/circuits/lean/, "
            "verity/commitments/lean/ against main: sending presigned URLs for the pinned upstream build and inputs") \
        in capsys.readouterr().out
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
    of them can't land on a check that skipped it, not only those under verity/protocols/flock/prover/."""
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
    for bad in ("3", "0/3", "a/b"):
        p = train("--shard", bad)
        assert p.returncode == 2 and "train: --shard needs K/N" in p.stderr, p.stderr
    p = train("--deferred", "pytest,lean-audit", "--shard", "2/3", "--no-cache", "--no-regenerate")
    assert p.returncode == 1 and "no parent store key" in p.stderr, p.stderr
    p = train("--base", "main")
    assert p.returncode == 2 and "train: --base refused without --deferred" in p.stderr, p.stderr
    p = train("--ratchet-base", "main")
    assert p.returncode == 2 and "train: --ratchet-base refused without --deferred" in p.stderr, p.stderr


def test_train_sh_copied_out_of_the_tree_finds_the_trees_settings(tmp_path):
    """Drive launches a copy of train.sh from its own directory (`drive.detach`) with the tree as its working directory: the
    settings check reads the tree's settings.py, so the copy reaches the store-key check rather than refusing every launch."""
    import shutil
    import subprocess

    copy = shutil.copy(ROOT / "tools" / "verity" / "check" / "train.sh", tmp_path / "train.sh")
    p = subprocess.run(["bash", str(copy), "abc", "vy-x", "--deferred", "pytest"], capture_output=True, text=True, cwd=ROOT,
                       env={"PATH": "/usr/bin:/bin"})
    assert p.returncode == 1 and "no parent store key" in p.stderr, p.stderr


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
        done.unlink(missing_ok=True)      # else `sleep` returns at once on an earlier call's launch, before this one's
        return subprocess.run(["bash", str(ROOT / "tools" / "verity" / "check" / "train.sh"), *a], capture_output=True, text=True,
                              cwd=tmp_path / "repo", env={**env, **extra})

    return git, train, argv


def test_train_sh_refuses_an_empty_commit_and_one_origin_lacks(tmp_path):
    """train.sh refuses, before launching anything, a commit
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


def test_train_sh_launches_the_records_step_alone_and_hands_a_shard_its_artifact(tmp_path):
    """Drive's sharded after-landing run: `--cc-records` launches circuit-check's records step alone, in a worktree of its own,
    and `--cc-records-from ART@SHA256` reaches a shard's check.py; either is refused where it doesn't belong."""
    git, train, argv = _train_repo(tmp_path)
    (tmp_path / "repo" / "a").write_text("b\n")
    git("commit", "-q", "-am", "landing")
    git("push", "-q", "origin", "HEAD:main")
    p = train("HEAD", "vy-x", "--cc-records")
    args = argv.read_text().split()
    assert p.returncode == 0 and args[args.index("--keep-going"):] == ["--keep-going", "--cc-records"], (p.stderr, args)
    assert f"train-{git('rev-parse', 'HEAD')[:12]}-records.log" in p.stdout, p.stdout
    art = f"art:{'a' * 64}@{'b' * 64}"
    p = train("HEAD", "vy-x", "--deferred", "pytest,circuit-check", "--shard", "1/4", "--cc-records-from", art)
    args = argv.read_text().split()
    assert p.returncode == 0 and args[args.index("--cc-records-from"):args.index("--cc-records-from") + 2] == ["--cc-records-from", art]
    argv.unlink()
    for flags, why in ((["--cc-records", "--deferred", "pytest"], "--cc-records refused with another flag"),
                       (["--cc-records", "--agreement"], "--cc-records refused with another flag"),
                       (["--deferred", "pytest", "--cc-records-from", art], "--cc-records-from refused without --shard"),
                       (["--deferred", "pytest", "--shard", "1/4", "--cc-records-from", "art:x"], "--cc-records-from needs ART@SHA256")):
        p = train("HEAD", "vy-x", *flags)
        assert p.returncode == 2 and f"train: {why}" in p.stderr and not argv.exists(), (flags, p.stderr)


def test_train_sh_hands_an_auto_shard_its_plan_and_refuses_one_without_it(tmp_path):
    """Drive's `after_landing_shards = "auto"`: each shard is `--shard auto:K/N --shard-plan-file PLAN`, the one plan drive
    made for all N, which reaches check.py --record as given; an auto shard without a plan, a plan without an auto shard, and
    a plan by a relative path (train.sh runs check.py from its own worktree) are refused before anything launches."""
    git, train, argv = _train_repo(tmp_path)
    (tmp_path / "repo" / "a").write_text("b\n")
    git("commit", "-q", "-am", "landing")
    git("push", "-q", "origin", "HEAD:main")
    plan = tmp_path / "check-shard-plan.json"
    plan.write_text("{}\n")
    p = train("HEAD", "vy-x", "--deferred", "pytest,circuit-check", "--shard", "auto:2/3", "--shard-plan-file", str(plan))
    args = argv.read_text().split()
    assert p.returncode == 0 and args[-4:] == ["--shard", "auto:2/3", "--shard-plan-file", str(plan)], (p.stderr, args)
    assert "shard auto:2/3" in p.stdout and f"train-{git('rev-parse', 'HEAD')[:12]}-auto-2of3.log" in p.stdout, p.stdout
    argv.unlink()
    for flags, why in ((["--shard", "auto:2/3"], "--shard auto:K/N takes --shard-plan-file PLAN"),
                       (["--shard", "2/3", "--shard-plan-file", str(plan)], "--shard auto:K/N takes --shard-plan-file PLAN"),
                       (["--shard", "auto:2/3", "--shard-plan-file", plan.name], "--shard-plan-file needs a plan's absolute"),
                       (["--shard", "auto:0/3", "--shard-plan-file", str(plan)], "--shard needs K/N or auto:K/N")):
        p = train("HEAD", "vy-x", "--deferred", "pytest", *flags)
        assert p.returncode == 2 and f"train: {why}" in p.stderr and not argv.exists(), (flags, p.stderr)


def test_train_sh_gives_every_check_a_timeout_so_its_run_extends_the_nodes_lease(tmp_path):
    """Only a `research run` with a --timeout extends its node's lease (`remote.lease_extend_s`): without one a node that runs
    nothing but checks stops two hours after its boot, so every check train.sh launches carries one."""
    git, train, argv = _train_repo(tmp_path)
    (tmp_path / "repo" / "a").write_text("b\n")
    git("commit", "-q", "-am", "landing")
    git("push", "-q", "origin", "HEAD:main")
    for flags in ((), ("--deferred", "lean-audit"), ("--cc-records",)):
        p = train("HEAD", "vy-x", *flags)
        args = argv.read_text().split()
        assert p.returncode == 0 and args[args.index("--timeout") + 1] == "36000", (flags, p.stderr, args)


def test_train_sh_hands_the_lock_runs_bookkeeping_flag_to_a_check_that_takes_it(tmp_path):
    """Drive's lock run (`--deferred lean-audit --bookkeeping-warn`): the flag reaches check.py --record when the commit's
    check.py takes it, and on a commit whose check.py predates it the run launches without it, saying so; with --no-regenerate,
    --shard or without --deferred it is refused before anything launches."""
    git, train, argv = _train_repo(tmp_path)
    check = tmp_path / "repo" / "tools" / "verity" / "check" / "check.py"
    check.parent.mkdir(parents=True)
    check.write_text('BOOKKEEPING = "--bookkeeping-warn"\n')
    git("add", "tools")
    git("commit", "-q", "-m", "landing")
    git("push", "-q", "origin", "HEAD:main")
    p = train("HEAD", "vy-x", "--deferred", "lean-audit", "--bookkeeping-warn")
    args = argv.read_text().split()
    assert p.returncode == 0 and args[args.index("--deferred"):] == ["--deferred", "lean-audit", "--bookkeeping-warn"], (
        p.stderr, args)
    check.write_text("\n")
    git("commit", "-q", "-am", "older")
    git("push", "-q", "origin", "HEAD:main")
    p = train("HEAD", "vy-x", "--deferred", "lean-audit", "--bookkeeping-warn")
    args = argv.read_text().split()
    assert p.returncode == 0 and "--bookkeeping-warn" not in args and "predates --bookkeeping-warn" in p.stdout, (p.stdout, args)
    argv.unlink()
    for flags in (["--bookkeeping-warn"], ["--deferred", "lean-audit", "--no-regenerate", "--bookkeeping-warn"],
                  ["--deferred", "lean-audit", "--shard", "1/3", "--bookkeeping-warn"]):
        p = train("HEAD", "vy-x", *flags)
        assert p.returncode == 2 and "train: --bookkeeping-warn refused" in p.stderr and not argv.exists(), (flags, p.stderr)


def test_train_sh_hands_the_loops_deselect_list_to_a_check_that_takes_it(tmp_path):
    """The vqueue loop's list (`--deferred ... --deselect FILE`): it reaches check.py --record when the commit's check.py takes
    it, and on a commit whose check.py predates it the run launches without it, saying so; without --deferred, by a relative
    path or with --cc-records it is refused before anything launches."""
    git, train, argv = _train_repo(tmp_path)
    check = tmp_path / "repo" / "tools" / "verity" / "check" / "check.py"
    check.parent.mkdir(parents=True)
    check.write_text('DESELECT_FILE = "check-deselect.txt"\n')
    git("add", "tools")
    git("commit", "-q", "-m", "landing")
    git("push", "-q", "origin", "HEAD:main")
    listed = tmp_path / "deselect.sent"
    listed.write_text("tests/test_a.py::test_a\n")
    p = train("HEAD", "vy-x", "--deferred", "pytest", "--shard", "1/3", "--deselect", str(listed))
    args = argv.read_text().split()
    assert p.returncode == 0 and args[args.index("--deselect"):args.index("--deselect") + 2] == ["--deselect", str(listed)], (
        p.stderr, args)
    check.write_text("\n")
    git("commit", "-q", "-am", "older")
    git("push", "-q", "origin", "HEAD:main")
    p = train("HEAD", "vy-x", "--deferred", "pytest", "--deselect", str(listed))
    args = argv.read_text().split()
    assert p.returncode == 0 and "--deselect" not in args and "predates --deselect" in p.stdout, (p.stdout, args)
    argv.unlink()
    for flags, why in ((["--deselect", str(listed)], "--deselect refused without --deferred"),
                       (["--deferred", "pytest", "--deselect", listed.name], "--deselect needs a list's absolute path"),
                       (["--cc-records", "--deselect", str(listed)], "--deselect refused without --deferred")):
        p = train("HEAD", "vy-x", *flags)
        assert p.returncode == 2 and f"train: {why}" in p.stderr and not argv.exists(), (flags, p.stderr)


def test_train_sh_replaces_a_tree_git_no_longer_lists_as_a_worktree(tmp_path):
    """On the control pod's NFS TMPDIR a train's tree can be left half removed and no longer a worktree, which `worktree remove`
    skips and `worktree add` refuses: every relaunch of that commit registered no run."""
    git, train, argv = _train_repo(tmp_path)
    (tmp_path / "repo" / "a").write_text("b\n")
    git("commit", "-q", "-am", "change")
    git("push", "-q", "origin", "HEAD:main")
    stale = tmp_path / f"train-{git('rev-parse', 'HEAD')[:12]}"
    (stale / ".venv").mkdir(parents=True)
    (stale / "README.md").write_text("left\n")
    p = train("HEAD", "vy-x")
    assert p.returncode == 0 and p.stdout.splitlines()[-1] == "RUN r20261009-050001-aaaa", (p.stdout, p.stderr)
    assert (stale / "a").read_text() == "b\n" and not (stale / "README.md").exists()


def test_train_sh_hands_check_the_base_below_the_tip_that_drive_sends(tmp_path):
    """Drive's after-landing run of `main`'s tip names the last verified `main` (`--base`), so ratchet-gate judges the landings
    above it, not the tip against itself: train.sh resolves it after its fetch and hands check.py that commit, and registers
    no run for a base that is no commit."""
    git, train, argv = _train_repo(tmp_path)
    verified = git("rev-parse", "HEAD")
    (tmp_path / "repo" / "a").write_text("b\n")
    git("commit", "-q", "-am", "landing")
    git("push", "-q", "origin", "HEAD:main")
    p = train("HEAD", "vy-x", "--deferred", "pytest,ratchet-gate", "--no-cache", "--base", "HEAD~1")
    args = argv.read_text().split()
    assert p.returncode == 0 and args[args.index("--base"):] == ["--base", verified], (p.stderr, args)
    assert args[args.index("--deferred"):args.index("--base")] == ["--deferred", "pytest,ratchet-gate", "--no-cache"], args
    argv.unlink()
    p = train("HEAD", "vy-x", "--deferred", "ratchet-gate", "--base", "nope")
    assert p.returncode == 1 and "--base nope: no such commit" in p.stderr and not argv.exists(), p.stderr
    # ratchet-gate's own base (`--ratchet-base`: the tip of drive's last run whose gate gave a verdict), beside `--base`
    p = train("HEAD", "vy-x", "--deferred", "ratchet-gate", "--base", "HEAD~1", "--ratchet-base", "HEAD~1")
    args = argv.read_text().split()
    assert p.returncode == 0 and args[args.index("--base"):] == ["--base", verified, "--ratchet-base", verified], (p.stderr, args)
    argv.unlink()
    p = train("HEAD", "vy-x", "--deferred", "ratchet-gate", "--ratchet-base", "nope")
    assert p.returncode == 1 and "--ratchet-base nope: no such commit" in p.stderr and not argv.exists(), p.stderr


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
    """A head that merges main into a commit already holding main's change has its first parent's tree, yet differs from main,
    and is not empty: empty means the tree is main's, where it leaves main."""
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
    """A check.py still gathering verdicts after train.sh's wait would launch its runs later, with no ids for drive, so nothing
    would cancel them, and one could hold a check slot for hours on a tree no longer on `next`."""
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
    assert one and CHECK.lints_key(e) == one and CHECK.lints_key({"PATH": "/bin"}) == one
    assert CHECK.lints_key({**e, "VERITY_SLOW": "1"}) != one
    (tmp_path / "b.md").write_text("b\n")
    assert CHECK.lints_key(e) is None
    git("add", "-A")
    git("commit", "-q", "-m", "b")
    assert CHECK.lints_key(e) not in (None, one)
