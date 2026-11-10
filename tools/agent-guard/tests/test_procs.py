"""`procs.py` shows a process's command line with whatever may be a credential masked: as the lister the shell hook names
instead of `ps` and `pgrep -a`, and as the filter install.sh puts in front of those two. Every value here is a dummy, most
built at run time so that no file holds one a scanner reads as real."""
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1]
PROCS = GUARD / "procs.py"
MASK = "[REDACTED]"
HEX = "0123456789abcdef" * 4
JWT = ".".join(("eyJ" + "dummy" * 20, "eyJ" + "dummy" * 50, "dummy" * 10))
OPAQUE = "Dummy0" * 7                                     # 42 characters, no separator
BASE64 = ("Dummy0+" * 6)[:41]
LONG = ("dummy=value," * 12)[:137]                        # no run of 20: only its length says it is no word
WITH_PASSWORD = "https://" + "user:dummy" + "@github.com/o/r.git"
WITH_TOKEN = "https://" + "ghp_" + "dummy0" * 6 + "@github.com/o/r.git"
real_ps = pytest.mark.skipif(not (os.path.exists("/usr/bin/ps") and os.path.exists("/usr/bin/pgrep")),
                             reason="needs procps's ps and pgrep")


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", GUARD / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def procs():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "dont_write_bytecode", True)      # no __pycache__ in tools/agent-guard, which boot.sh copies file by file
        yield load("procs")


#: a daemon's arguments of the kind this guards against, each value a dummy of its shape: three tokens after flags (two of
#: 64 hex digits, one JWT), three opaque words on their own, a word of 137 characters and one of 41 after flags that name
#: no secret
DAEMON = ["node", "/opt/daemon/index.js", "--auth-token", HEX, "--pty-auth-token", HEX[::-1], "--trace-auth-token", JWT,
          OPAQUE, OPAQUE[::-1], OPAQUE.swapcase(), "--trace-attributes", LONG, "--mcp-input-schema-json", BASE64]
DAEMON_MASKED = (f"node /opt/daemon/index.js --auth-token {MASK} --pty-auth-token {MASK} --trace-auth-token {MASK} "
                 f"{MASK} {MASK} {MASK} --trace-attributes {MASK} --mcp-input-schema-json {MASK}")

MASKED = [
    # the value of a flag whose name says it is a secret, in both spellings
    (["node", "daemon.js", "--auth-token", HEX], f"node daemon.js --auth-token {MASK}"),
    (["node", "daemon.js", f"--auth-token={HEX}"], f"node daemon.js --auth-token={MASK}"),
    (["node", "daemon.js", "--pty-auth-token", HEX], f"node daemon.js --pty-auth-token {MASK}"),
    (["node", "daemon.js", f"--pty-auth-token={HEX}"], f"node daemon.js --pty-auth-token={MASK}"),
    (["node", "daemon.js", "--trace-auth-token", JWT], f"node daemon.js --trace-auth-token {MASK}"),
    (["node", "daemon.js", f"--trace-auth-token={JWT}"], f"node daemon.js --trace-auth-token={MASK}"),
    (DAEMON, DAEMON_MASKED),
    # a value too short and too plain for any shape, which only its name gives away
    (["tool", "--password", "dummy", "serve"], f"tool --password {MASK} serve"),
    (["tool", "--api-key=dummy", "serve"], f"tool --api-key={MASK} serve"),
    (["tool", "-token", "dummy", "serve"], f"tool -token {MASK} serve"),
    (["tool", "--passwd", "dummy two words"], f"tool --passwd {MASK}"),
    (["tool", "--client-secret", "dummy", "--credential-file", "c.json"], f"tool --client-secret {MASK} --credential-file {MASK}"),
    (["tool", "--token-file", "--verbose", "serve"], "tool --token-file --verbose serve"),     # an option is no value
    (["env", "DUMMY_API_KEY=dummy", "run"], f"env DUMMY_API_KEY={MASK} run"),
    (["tool", '{"token": "dummy", "port": 8080}'], f'tool {{"token": {MASK} "port": 8080}}'),
    (["tool", "--header", "X-Api-Key: dummy"], f"tool --header X-Api-Key:{MASK}"),
    (["curl", "-H", "Authorization: Basic dummy", "http://127.0.0.1:9/"], f"curl -H Authorization:{MASK} http://127.0.0.1:9/"),
    (["curl", "-H", "X-Upstream: Bearer dummy", "http://127.0.0.1:9/"], f"curl -H X-Upstream: Bearer {MASK} http://127.0.0.1:9/"),
    # inside a script handed over as one argument: after `Authorization:` nothing shows, and a value that starts an
    # argument runs to its end
    (["bash", "-c", "curl -H 'Authorization: Bearer dummy' http://127.0.0.1:9/"], f"bash -c curl -H 'Authorization: {MASK} {MASK} {MASK}"),
    (["bash", "-c", "tool --password dummy serve"], f"bash -c tool --password {MASK} serve"),
    (["bash", "-lc", "TOKEN=dummy tool serve"], f"bash -lc TOKEN={MASK}"),
    # a URL's userinfo, a JWT, a run of hex digits, an opaque word, a word too long to be one
    (["git", "clone", WITH_PASSWORD], f"git clone https://{MASK}@github.com/o/r.git"),
    (["git", "fetch", WITH_TOKEN], f"git fetch https://{MASK}@github.com/o/r.git"),
    (["tool", JWT], f"tool {MASK}"),
    (["tool", HEX], f"tool {MASK}"),
    (["cat", f"/blobs/{HEX[:32]}/part"], f"cat /blobs/{MASK}/part"),
    (["git", "show", HEX[:40]], f"git show {MASK}"),
    (["tool", OPAQUE], f"tool {MASK}"),
    (["tool", BASE64], f"tool {MASK}"),
    (["tool", "run-" + "0a1b2c3d4e" * 2], f"tool {MASK}"),
    (["tool", LONG], f"tool {MASK}"),
    (["tool", f"--trace-attributes={LONG}"], f"tool --trace-attributes={MASK}"),
    (["tool", "a\tb\nc\x01d"], "tool a?b?c?d"),           # one line, whatever an argument holds
]

#: what a reader came for stays: the program, its subcommand, paths, short ids, a user name that is no credential
KEPT = [
    ["python3", "-m", "pytest", "tools/agent-guard/tests/test_shell_hook.py", "-k", "process"],
    ["bash", "-lc", "uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py --changed 0123abc --quick"],
    ["research", "run", "--on", "auto", "--cwd", "source", "--", "bash", "-lc", "make"],
    ["python3", "/workspace/research/src/tools/verity/check/slot.py", "--status"],
    ["git", "fetch", "https://notes@github.com/o/r.git"],
    ["git", "show", "0123abc"],
    ["tail", "-f", "/var/log/syslog"],
]


@pytest.mark.parametrize("argv, line", MASKED)
def test_a_command_line_shows_no_value_that_may_be_a_credential(procs, argv, line):
    assert procs.masked(argv) == line


@pytest.mark.parametrize("argv", KEPT)
def test_a_command_line_with_no_credential_shows_whole(procs, argv):
    assert procs.masked(argv) == " ".join(argv)


def test_a_printed_line_is_masked_by_its_words_alone_and_keeps_its_columns(procs):
    """What the filter in front of `ps` sees: a line, not arguments. The value is then the word after its flag."""
    assert procs.masked_line(" ".join(DAEMON)) == DAEMON_MASKED
    line = f"   1234 Thu Jan  1 00:00:00 1970 node daemon.js --auth-token {HEX} --pty-auth-token={HEX} serve\n"
    assert procs.masked_line(line) == f"   1234 Thu Jan  1 00:00:00 1970 node daemon.js --auth-token {MASK} --pty-auth-token={MASK} serve\n"
    assert procs.masked_line("1234 curl -H Authorization: Basic dummy http://127.0.0.1:9/\n") == f"1234 curl -H Authorization: {MASK} {MASK} {MASK}\n"
    for plain in ("    PID STAT ELAPSED COMMAND\n", "   1234 Ssl     3600 python3\n", "1234\n", "\n"):
        assert procs.masked_line(plain) == plain


def test_no_piece_of_a_masked_value_is_left(procs):
    """Not its first characters either, which is what a column cut (`cut -c1-160`) showed of a token."""
    values = [HEX, HEX[::-1], JWT, OPAQUE, OPAQUE[::-1], OPAQUE.swapcase(), LONG, BASE64]
    for text in (procs.masked(DAEMON), procs.masked_line(" ".join(DAEMON))):
        assert [v for v in values for at in range(len(v) - 7) if v[at:at + 8] in text] == []


def test_the_names_and_the_userinfo_are_the_ones_the_other_guards_hold(procs):
    """procs.py is installed alone and runs from stdin on a machine without a checkout, so it imports neither guard: it holds
    their patterns again, and this holds them equal. Its names are the env guard's and `auth`; its userinfo masks whatever
    the git guard's CRED takes off a remote, and leaves the plain user name CRED leaves."""
    assert procs.NAMES == load("env_guard").PATTERN.pattern + "|AUTH"
    cred = re.search(r"^CRED='(.*)'$", (GUARD / "git_guard.sh").read_text(), re.M)[1]
    cred = re.compile(cred.replace("[:space:]", r"\s"))
    stripped = ["http://u:pw@127.0.0.1:9/x.git", "https://x-access-token:dummy@github.com/o/r", WITH_PASSWORD, WITH_TOKEN,
                "https://" + "github_pat_" + "dummy0" * 4 + "@github.com/o/r", "https://" + "dummy0" * 6 + "@github.com/o/r"]
    kept = ["https://notes@github.com/o/r.git", "https://github.com/o/r.git", "git@github.com:o/r.git"]
    assert [u for u in stripped if not cred.search(u)] == [] and [u for u in kept if cred.search(u)] == []
    assert [u for u in stripped if procs.USERINFO.sub(f"://{MASK}@", u) != re.sub(r"://[^@]*@", f"://{MASK}@", u)] == []
    assert [u for u in kept if procs.USERINFO.search(u)] == []


@pytest.fixture
def held():
    """A process with a dummy of each kind among its arguments, and a mark no other process has: (pid, mark, the tail of its
    masked line). It waits on its input, so it ends when the test does."""
    mark = f"mark{os.getpid()}"
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()", mark, "--auth-token", HEX,
                             f"--pty-auth-token={HEX}", "--trace-auth-token", JWT, WITH_PASSWORD, OPAQUE, "serve"],
                            stdin=subprocess.PIPE)
    try:
        yield proc.pid, mark, (f" -c import sys; sys.stdin.read() {mark} --auth-token {MASK} --pty-auth-token={MASK} "
                               f"--trace-auth-token {MASK} https://{MASK}@github.com/o/r.git {MASK} serve\n")
    finally:
        proc.stdin.close()
        proc.wait()


def run(*command, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in command], capture_output=True, text=True, **kwargs)


def test_the_lister_prints_a_process_masked_and_matches_a_pattern_against_the_masked_line(held):
    pid, mark, tail = held
    by_pattern = run(sys.executable, PROCS, mark)
    assert by_pattern.returncode == 0 and by_pattern.stdout.endswith(tail) and by_pattern.stdout.count("\n") == 1
    assert re.match(rf" *{pid} +\d+ python\S* +\S*python\S*$", by_pattern.stdout[:-len(tail)])     # pid, seconds, name, the program
    assert run(sys.executable, PROCS, "-p", pid).stdout == by_pattern.stdout
    assert run(sys.executable, PROCS, mark, "-p", f"1,{pid}").stdout == by_pattern.stdout
    with PROCS.open() as source:                          # as it runs on a machine without a checkout: `ssh HOST python3 - PATTERN`
        assert run(sys.executable, "-", mark, stdin=source).stdout == by_pattern.stdout
    # a pattern that is the start of a masked value, or matches only the line as it was, finds nothing
    for pattern in (HEX[:8], JWT[:12], OPAQUE[:6], "user:dummy", f"{mark} --auth-token 0123"):
        hidden = run(sys.executable, PROCS, pattern, "-p", pid)
        assert (hidden.returncode, hidden.stdout) == (1, ""), pattern
    assert run(sys.executable, PROCS, "auth-token .REDACTED", "-p", pid).stdout == by_pattern.stdout
    assert run(sys.executable, PROCS, f"no-such-{mark}").returncode == 1
    for usage in (["-x"], ["a", "b"], ["("], ["-p", "self"]):
        refused = run(sys.executable, PROCS, *usage)
        assert refused.returncode == 2 and refused.stdout == "" and "usage:" in refused.stderr


@real_ps
def test_as_ps_and_pgrep_it_masks_what_the_real_ones_print_and_is_otherwise_them(held, tmp_path):
    pid, mark, tail = held
    for name in ("ps", "pgrep"):
        (tmp_path / name).symlink_to(PROCS)
    assert HEX in run("/usr/bin/ps", "-o", "args=", "-p", pid).stdout          # what the real one prints: the test can fail
    assert run(tmp_path / "ps", "-o", "args=", "-p", pid).stdout.endswith(tail)
    listed = run(tmp_path / "pgrep", "-af", mark)
    assert listed.returncode == 0 and listed.stdout.startswith(f"{pid} ") and listed.stdout.endswith(tail) and listed.stdout.count("\n") == 1
    # what holds no command line comes out as the real one's, exit status and errors too
    assert run(tmp_path / "ps", "-o", "pid=,comm=", "-p", pid).stdout == run("/usr/bin/ps", "-o", "pid=,comm=", "-p", pid).stdout
    assert run(tmp_path / "pgrep", "-f", mark).stdout == f"{pid}\n"            # the filter is not one more match
    for name, args in (("pgrep", ["-f", f"no-such-{mark}"]), ("pgrep", ["-af", f"no-such-{mark}"]), ("ps", ["-p", "1", "--no-such-option"])):
        ours, real = run(tmp_path / name, *args), run(f"/usr/bin/{name}", *args)
        assert (ours.returncode, ours.stdout) == (real.returncode, real.stdout) and real.returncode != 0, (name, args)
        assert ours.stderr == real.stderr


@real_ps
def test_the_self_test_install_runs_passes_on_this_file():
    out = run("bash", GUARD / "test_procs.sh")
    assert out.returncode == 0 and "FAIL" not in out.stdout and out.stdout.count("ok  ") >= 9
