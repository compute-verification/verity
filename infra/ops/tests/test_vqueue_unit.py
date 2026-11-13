"""The verified queue's loop as a unit on a node (pods/nebius/vqueue): what loop.sh runs in each mode, with a stand-in for
`python3.12 -m research queue vqueue` that records its argv and environment; how it stops; whose token git and gh get; what
root's oneshot writes; and that deploy.toml installs every file the units name on one node, as a user of its own."""
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

import pytest

from verity_ops import deploy

VQ = deploy.MANIFEST.parent / "vqueue"
LOOP = VQ / "loop.sh"
HELPER = "infra/control/gh_app_token.sh"          # the control pod's own, which token.sh runs on the node
CONFIG, SKIP = 78, 75
KEYS = ("HOME", "PATH", "PYTHONPATH", "TMPDIR", "TRAIN_TREES", "GH_TOKEN", "AWS_ACCESS_KEY_ID", "SLACK_BOT_TOKEN",
        "RESEARCH_AGENT_URL", "RESEARCH_LANE", "GIT_TERMINAL_PROMPT")
# The stand-in tick: one JSON line a call. With VQ_TEST_TERM its first call tells the loop to stop and then goes on, as a tick
# under `systemctl stop` does; a second call, which a loop that stopped never makes, ends the loop so the test can't hang.
STUB = """#!{py}
import json, os, signal, sys
out = os.environ["VQ_TEST_OUT"]
n = len(open(out).read().splitlines()) if os.path.exists(out) else 0
if os.environ.get("VQ_TEST_TERM"):
    os.kill(os.getppid(), signal.SIGKILL if n else signal.SIGTERM)
with open(out, "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd(), "env": {{k: os.environ.get(k) for k in {keys!r}}}}}) + "\\n")
sys.exit(int(os.environ.get("VQ_TEST_RC", "0")))
"""


def script(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def envfile(path: Path, values: dict) -> None:
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))


class Node:
    """A node's files under a tmp dir: the home as vy-vqueue-setup and the switch leave it, the credentials and a fresh token."""

    def __init__(self, tmp: Path):
        self.home, self.cred, self.token, self.out = tmp / "home", tmp / "cred", tmp / "run" / "token", tmp / "ticks.jsonl"
        for d in ("live", "shadow", "src/tools/research/src/research"):
            (self.home / d).mkdir(parents=True)
        self.clone = self.home / "verity"
        subprocess.run(["git", "init", "-q", str(self.clone)], check=True)
        self.cred.mkdir()
        envfile(self.cred / "r2.env", {"AWS_ACCESS_KEY_ID": "dummy-r2", "GH_TOKEN": "dummy-env"})
        envfile(self.cred / "slack.env", {"SLACK_BOT_TOKEN": "dummy-slack"})
        self.token.parent.mkdir()
        self.token.write_text("dummy-app\n")
        # ci's args as the control pod holds them: its own paths among them, and a comment
        (self.home / "live" / "args").write_text("# depth 2\n--gate check\n--on m1,m2\n--log /data/vqueue/live/decisions.jsonl\n")
        self.env = {**os.environ, "VQ_HOME": str(self.home), "VQ_CRED": str(self.cred), "VQ_TOKEN": str(self.token),
                    "VQ_PYTHON": str(script(tmp / "stub" / "python", STUB.format(py=sys.executable, keys=KEYS))),
                    "VQ_INTERVAL": "0", "VQ_TEST_OUT": str(self.out)}

    def loop(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(LOOP), *args], env={**self.env, **env}, capture_output=True, text=True)

    def ticks(self) -> list[dict]:
        return [json.loads(ln) for ln in self.out.read_text().splitlines()] if self.out.exists() else []


@pytest.fixture
def node(tmp_path):
    return Node(tmp_path)


def flag(argv: list[str], name: str) -> str:
    """The value argparse takes for ``name``: its last."""
    return argv[len(argv) - 1 - argv[::-1].index(name) + 1]


def test_with_no_mode_file_a_tick_is_a_dry_run_in_the_node_s_own_paths_as_the_app(node):
    """What a node runs before anyone switches it: --dry-run, which writes nothing, on live/'s copy of the log; the node's
    paths after ci's args, so the pod's --log in args loses; the clone as cwd and ci's pinned tree as the code; the store's
    key and the Slack token from the credential files, and GH_TOKEN the App's token even when a file sets another."""
    r = node.loop("once")
    assert r.returncode == 0, r.stderr
    [t] = node.ticks()
    argv, env, live = t["argv"], t["env"], node.home / "live"
    assert argv[:4] == ["-m", "research", "queue", "vqueue"] and argv[4:8] == ["--gate", "check", "--on", "m1,m2"]
    assert "#" not in argv and "depth" not in argv
    assert argv[-2:] == ["--once", "--dry-run"] and "--shadow" not in argv
    assert flag(argv, "--log") == str(live / "decisions.jsonl") and "/data/vqueue/live/decisions.jsonl" in argv
    assert (flag(argv, "--repo"), flag(argv, "--lock"), flag(argv, "--state"), flag(argv, "--priority-file"), flag(argv, "--frozen")) \
        == (str(node.clone), str(live / "vqueue.lock"), str(live / "state.json"), str(live / "priority"), str(live / "frozen"))
    assert t["cwd"] == str(node.clone) and env["PYTHONPATH"] == str(node.home / "src/tools/research/src")
    assert (env["HOME"], env["TMPDIR"], env["TRAIN_TREES"]) == (str(node.home), str(node.home / "tmp"), str(node.home / "trains"))
    assert env["PATH"].split(":")[0] == os.path.realpath(VQ / "bin") and (VQ / "bin" / "gh").is_file()
    assert (env["GH_TOKEN"], env["AWS_ACCESS_KEY_ID"], env["SLACK_BOT_TOKEN"]) == ("dummy-app", "dummy-r2", "dummy-slack")
    assert env["RESEARCH_AGENT_URL"] is None and env["RESEARCH_LANE"] == "ci" and env["GIT_TERMINAL_PROMPT"] == "0"
    helper = subprocess.run(["git", "-C", str(node.clone), "config", "--local", "--get", "credential.helper"],
                            capture_output=True, text=True).stdout.strip()
    assert helper == f"!{os.path.realpath(LOOP)} git"
    # flags after `once` come last, and the loop posts as whoever live/agent-url names
    (node.home / "live" / "agent-url").write_text("https://example.invalid/agent\n")
    assert node.loop("once", "--batch", "1").returncode == 0
    t = node.ticks()[1]
    assert t["argv"][-2:] == ["--batch", "1"] and t["env"]["RESEARCH_AGENT_URL"] == "https://example.invalid/agent"


def test_live_is_refused_until_the_node_holds_the_loop_s_log(node):
    """A loop with no log would rebuild every candidate in flight: `live` with live/decisions.jsonl missing or empty runs
    nothing and exits 78, which the unit doesn't restart on; with the log it is the loop, neither dry nor shadow."""
    (node.home / "mode").write_text("live\n")
    for log in (None, ""):
        if log is not None:
            (node.home / "live" / "decisions.jsonl").write_text(log)
        r = node.loop("once")
        assert r.returncode == CONFIG and not node.ticks() and "copy it from the host the loop ran on" in r.stderr
    (node.home / "live" / "decisions.jsonl").write_text('{"action": "build"}\n')
    assert node.loop("once").returncode == 0
    [t] = node.ticks()
    assert t["argv"][-1] == "--once" and not {"--dry-run", "--shadow"} & set(t["argv"])
    assert flag(t["argv"], "--log") == str(node.home / "live" / "decisions.jsonl")
    (node.home / "mode").write_text("on\n")
    r = node.loop("once")
    assert r.returncode == CONFIG and "dry-run, shadow or live" in r.stderr and len(node.ticks()) == 1


def test_a_shadow_keeps_its_log_lock_and_args_apart_from_the_live_loop_s(node):
    (node.home / "mode").write_text("shadow\n")
    r = node.loop("once")
    assert r.returncode == CONFIG and "shadow/args" in r.stderr and not node.ticks()
    (node.home / "shadow" / "args").write_text("--gate check\n--on spare\n")
    assert node.loop("once").returncode == 0
    [t] = node.ticks()
    argv, shadow = t["argv"], node.home / "shadow"
    assert argv[-2:] == ["--once", "--shadow"] and flag(argv, "--on") == "spare"
    assert (flag(argv, "--log"), flag(argv, "--lock"), flag(argv, "--state")) \
        == (str(shadow / "decisions.jsonl"), str(shadow / "vqueue.lock"), str(shadow / "state.json"))
    assert flag(argv, "--priority-file") == str(node.home / "live" / "priority")


def test_a_node_missing_the_clone_the_pin_or_the_args_says_which_and_runs_nothing(node):
    for gone, fix in ((node.home / "live" / "args", "the loop's flags"), (node.home / "src", "check ci's pin out"),
                      (node.clone / ".git", "copy the loop's clone")):
        kept = gone.with_name(gone.name + ".kept")
        gone.rename(kept)
        r = node.loop("once")
        assert r.returncode == CONFIG and fix in r.stderr and not node.ticks(), gone
        kept.rename(gone)
    assert node.loop("once").returncode == 0 and len(node.ticks()) == 1


def test_a_missing_or_old_token_skips_the_tick_and_the_loop_goes_on(node):
    """Root's timer rewrites the token every 20 min and GitHub ends one after 60: a file over 50 min old means the timer
    failed, and a tick on it would lose its pushes halfway.  The tick is skipped (75), not a reason to stop the unit."""
    os.utime(node.token, (0, 0))
    r = node.loop("once")
    assert r.returncode == SKIP and "systemctl status vy-vqueue-token.service" in r.stderr and not node.ticks()
    node.token.unlink()
    assert node.loop("once").returncode == SKIP and not node.ticks()
    node.token.write_text("dummy-app\n")
    assert node.loop("once").returncode == 0 and len(node.ticks()) == 1


def test_sigterm_ends_the_loop_after_the_tick_in_progress(node):
    """`systemctl stop` signals the loop alone (KillMode=process).  The tick it is waiting on finishes and is logged, no
    other starts, and the loop exits 0, so a stop always falls between two ticks."""
    r = node.loop("run", VQ_TEST_TERM="1")
    assert r.returncode == 0, (r.returncode, r.stderr)
    assert len(node.ticks()) == 1 and "tick rc=0 mode=dry-run" in r.stdout and "stopped after its tick" in r.stderr


def function(name: str) -> str:
    fn = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", LOOP.read_text(), re.M | re.S)
    assert fn, f"loop.sh defines {name}() {{ ... }}"
    return fn.group(0)


def passes(tmp_path: Path, stopped: bool, rc: int = 0) -> subprocess.CompletedProcess:
    """loop.sh's `run` with a tick and a sweep that only log, and a wait that tells the loop to stop: one pass, then out."""
    d = tmp_path / "dir"
    d.mkdir(exist_ok=True)
    (d / "stopped").touch() if stopped else (d / "stopped").unlink(missing_ok=True)
    stubs = (f'mode() {{ mode=live dir={d}; }}\ntick() {{ echo tick; return {rc}; }}\nsweep() {{ echo sweep; }}\n'
             'say() { echo "$*"; }\nfunction sleep { kill -TERM $$; }\n')
    return subprocess.run(["bash", "-c", f'CONFIG={CONFIG} INTERVAL=0 CODE={tmp_path}/no-pin\n{stubs}{function("run")}run'],
                          capture_output=True, text=True)


def test_while_stopped_exists_no_tick_starts_and_a_failed_tick_is_not_the_loop_s_end(tmp_path):
    r = passes(tmp_path, stopped=True)
    assert r.returncode == 0 and r.stdout.splitlines() == ["stopped after its tick"]
    r = passes(tmp_path, stopped=False, rc=1)
    assert r.returncode == 0 and r.stdout.splitlines()[:3] == ["tick", "vy-vqueue: tick rc=1 mode=live pin=", "sweep"]
    # what the node holds can't run a tick: the loop ends with 78 before it sweeps or waits, and the unit stays failed
    r = passes(tmp_path, stopped=False, rc=CONFIG)
    assert r.returncode == CONFIG and r.stdout.splitlines() == ["tick", f"vy-vqueue: tick rc={CONFIG} mode=live pin="]


NOW, LIMIT = 100_000, 40 * 60          # the clock the sweep is given here, and how long it leaves what a launch left


def left(path: Path, ago: int) -> Path:
    """What a launch left at ``path``, last changed ``ago`` seconds before NOW: a file if the name has a suffix, else a
    directory with a file in it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix:
        path.write_text("")
    else:
        path.mkdir()
        (path / "f").write_text("")
    os.utime(path, (NOW - ago, NOW - ago))
    return path


def sweep(tmp_path: Path) -> None:
    """loop.sh's `sweep` over tmp_path's trains/ and tmp/, at NOW by its clock."""
    env = {**os.environ, "TRAIN_TREES": str(tmp_path / "trains"), "TMPDIR": str(tmp_path / "tmp"), "CLONE": str(tmp_path / "no-clone")}
    subprocess.run(["bash", "-c", f'now() {{ echo {NOW}; }}\n{function("sweep")}sweep'], env=env)


def test_the_sweep_takes_what_a_launch_left_40_minutes_ago_and_nothing_else(tmp_path):
    """A launch's train tree and its log, check.py's Lean deps, and the tree of a job of the steps after a landing that was
    killed before it removed its own (`research merge`'s `_own_tree`)."""
    trains, tmp = tmp_path / "trains", tmp_path / "tmp"
    old = [trains / "train-0123456789ab-1of3", trains / "train-0123456789ab-1of3.log", tmp / "check-lean-deps-x", tmp / "train-why.abc",
           trains / "research-after-land-0123456789ab-k1ll3d"]
    new = [trains / "train-ba9876543210-2of3", trains / "train-ba9876543210-2of3.log", tmp / "check-lean-deps-y",
           trains / "research-after-land-ba9876543210-runs0n"]
    others = [tmp / "drive-launches", tmp / "research-merged-1"]
    for p in old + others:
        left(p, LIMIT + 1)
    for p in new:
        left(p, LIMIT)
    sweep(tmp_path)
    assert not [p for p in old if p.exists()] and all(p.exists() for p in new + others)


def test_a_train_tree_stays_while_its_launcher_lives_and_goes_once_the_launcher_is_gone(tmp_path):
    """train.sh starts check.py in the tree it checked out, and check.py can stay there, on the venv under it, for as long as
    its run, while the tree's own time stays at the launch.  So a tree long past the limit stays, with its log, while a
    process works in it or under it, and goes at the first sweep after the last such process ended; a tree as old that no
    process works in goes at once.  Each launcher here works in its tree until the test closes its input."""
    trains = tmp_path / "trains"
    live, under, gone = (left(trains / f"train-{c * 12}-{k}of3", 3 * LIMIT) for k, c in enumerate("abc", 1))
    logs = [left(t.with_name(t.name + ".log"), 3 * LIMIT) for t in (live, under, gone)]
    (under / ".venv" / "bin").mkdir(parents=True)
    os.utime(under, (NOW - 3 * LIMIT, NOW - 3 * LIMIT))
    launchers = [subprocess.Popen(["cat"], cwd=d, stdin=subprocess.PIPE) for d in (live, under / ".venv" / "bin")]
    try:
        sweep(tmp_path)
        kept = sorted(p.name for p in trains.iterdir())
    finally:
        for p in launchers:
            p.stdin.close()
            p.wait()
    assert kept == sorted(p.name for p in (live, under, *logs[:2])) and (under / ".venv" / "bin").is_dir() and (live / "f").exists()
    sweep(tmp_path)
    assert not list(trains.iterdir())


def test_git_and_gh_get_the_token_of_the_moment_from_root_s_file(tmp_path):
    """A tick can outlast a token: git's helper and gh read the file at each call, so a token minted mid-tick is the one used."""
    token = tmp_path / "token"
    gh = script(tmp_path / "gh", '#!/bin/sh\necho "$GH_TOKEN $*"\n')
    env = {**os.environ, "VQ_TOKEN": str(token), "VQ_GH": str(gh)}
    for value in ("dummy-one", "dummy-two"):
        token.write_text(value + "\n")
        r = subprocess.run([str(LOOP), "git", "get"], env=env, input="protocol=https\n", capture_output=True, text=True, check=True)
        assert r.stdout == f"username=x-access-token\npassword={value}\n"
        r = subprocess.run([str(VQ / "bin" / "gh"), "api", "x"], env=env, capture_output=True, text=True, check=True)
        assert r.stdout == f"{value} api x\n"
    for verb in ("store", "erase"):
        assert subprocess.run([str(LOOP), "git", verb], env=env, input="x\n", capture_output=True, text=True, check=True).stdout == ""
    token.unlink()
    assert subprocess.run([str(LOOP), "git", "get"], env=env, input="", capture_output=True).returncode == 1
    assert subprocess.run([str(VQ / "bin" / "gh"), "api"], env=env, capture_output=True).returncode == 1


def test_root_s_oneshot_mints_with_the_node_s_key_and_writes_a_token_only_the_group_reads(tmp_path):
    """token.sh beside the control pod's own gh_app_token.sh, as deploy installs them, with openssl, curl and python3.12 stood
    in: the helper signs with the key token.sh names, and its token lands whole (written beside, renamed) and 0640.  A refusal
    from GitHub fails the unit and leaves the last token in place."""
    lib, bin_, out, key = tmp_path / "lib", tmp_path / "bin", tmp_path / "run" / "token", tmp_path / "app.pem"
    lib.mkdir()
    out.parent.mkdir()
    for f in (VQ / "token.sh", deploy.SRC / HELPER):
        shutil.copy(f, lib / f.name)
    script(bin_ / "openssl", '#!/bin/sh\ncase "$*" in *-sign*) echo "$*" >> "$CALLS";; esac\ncat\n')
    script(bin_ / "curl", '#!/bin/sh\ncat "$REPLY"\n')
    script(bin_ / "python3.12", f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    reply = tmp_path / "reply"
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "GH_APP_KEY": str(key), "CALLS": str(tmp_path / "calls"), "REPLY": str(reply)}

    def run() -> subprocess.CompletedProcess:
        return subprocess.run([str(lib / "token.sh"), str(out)], env=env, capture_output=True, text=True)

    r = run()
    assert r.returncode == 1 and "root's and 0400" in r.stderr and not out.exists()
    key.write_text("not-to-be-copied\n")
    reply.write_text(json.dumps({"token": "dummy-minted"}))
    r = run()
    assert r.returncode == 0, r.stderr
    assert out.read_text() == "dummy-minted\n" and stat.S_IMODE(out.stat().st_mode) == 0o640
    assert f"-sign {key} " in (tmp_path / "calls").read_text() and os.listdir(out.parent) == ["token"]
    reply.write_text(json.dumps({"message": "Bad credentials"}))
    r = run()
    assert r.returncode != 0 and "GitHub refused" in r.stderr and out.read_text() == "dummy-minted\n"


def installed() -> tuple[deploy.Node, dict[str, deploy.Entry]]:
    """The one node deploy.toml installs the loop on, and its entries for the loop's files by repository path."""
    nodes = deploy.load()
    files = sorted(p.relative_to(deploy.SRC).as_posix() for p in VQ.rglob("*") if p.is_file())
    on = {n.name for n in nodes.values() for e in n.entries if e.src in files}
    assert files and len(on) == 1, on
    node = nodes[on.pop()]
    by_src = {e.src: e for e in node.entries if e.src in files or e.src == HELPER}
    assert not [f for f in [*files, HELPER] if f not in by_src]
    return node, by_src


def src(name: str) -> str:
    return (VQ / name).relative_to(deploy.SRC).as_posix()


def test_deploy_installs_every_file_of_the_loop_on_one_node_where_its_units_find_them():
    """One loop: every file under vqueue/ goes to the same single node, under a pattern that node manages, the scripts
    executable; each path a unit starts is one deploy.toml installs there, and gh_app_token.sh is the control pod's own."""
    node, by_src = installed()
    paths = {e.path for e in by_src.values()}
    for e in by_src.values():
        assert any(fnmatch(e.path, m) for m in node.managed), e.path
        if not e.path.endswith((".service", ".timer")):
            assert os.access(deploy.SRC / e.src, os.X_OK), e.src
    started = [s for u in VQ.glob("vy-vqueue*.service") for ln in u.read_text().splitlines() if ln.startswith("ExecStart=")
               for s in ln.split("=", 1)[1].split()[:1]]
    assert len(started) == 2 and not [s for s in started if s not in paths]
    lib = os.path.dirname(by_src[src("loop.sh")].path)
    assert by_src[HELPER].path == f"{lib}/gh_app_token.sh" and by_src[src("token.sh")].path == f"{lib}/token.sh"   # token.sh runs the one beside it
    assert by_src[src("bin/gh")].path == f"{lib}/bin/gh"                       # loop.sh's PATH starts at bin/ beside it
    # install restarts the loop for its unit and its script alone: a restart ends after the tick in progress
    assert {e.path for e in by_src.values() if e.run} == {f"{deploy.UNITS}vy-vqueue.service", f"{lib}/loop.sh"}
    assert all(deploy.unit_name(r) == "systemd:vy-vqueue.service" for e in by_src.values() for r in e.run)


def test_the_loop_is_a_user_of_its_own_that_never_holds_the_app_s_key():
    """The unit, loop.sh, setup.sh and the token oneshot agree on the account, the home and the token's file; the loop's unit
    carries no credential and leaves its launches running at a stop; root's oneshot alone names the key."""
    unit, token_unit = (VQ / "vy-vqueue.service").read_text(), (VQ / "vy-vqueue-token.service").read_text()
    loop, setup, token_sh = LOOP.read_text(), (VQ / "setup.sh").read_text(), (VQ / "token.sh").read_text()
    home = re.search(r"\bH=\$\{VQ_HOME:-(\S+?)\}", loop).group(1)
    token = re.search(r"TOKEN=\$\{VQ_TOKEN:-(\S+?)\}", loop).group(1)
    cred = re.search(r"CRED=\$\{VQ_CRED:-(\S+?)\}", loop).group(1)
    assert "User=vy-vqueue\n" in unit and "Group=vy-vqueue\n" in unit and f"WorkingDirectory={home}\n" in unit
    assert home == "/var/lib/vy-vqueue" and "StateDirectory=vy-vqueue\n" in unit
    assert "KillMode=process\n" in unit and "NoNewPrivileges=yes\n" in unit and "Restart=on-failure\n" in unit
    assert f"RestartPreventExitStatus={CONFIG}\n" in unit and f"\nCONFIG={CONFIG} " in loop and f"\nSKIP={SKIP} " in loop
    assert "Environment" not in unit and "pem" not in unit and "pem" not in loop
    assert f"U=vy-vqueue H={home} C={cred}\n" in setup and "--system" in setup and "--shell /usr/sbin/nologin" in setup
    assert '[ "$(id -nG "$U")" = "$U" ] || die' in setup and "sudoers" not in setup
    assert "User=" not in token_unit and "Group=vy-vqueue\n" in token_unit and "RuntimeDirectoryPreserve=yes\n" in token_unit
    assert f"ExecStart={installed()[1][src('token.sh')].path} {token}\n" in token_unit
    assert os.path.dirname(token) == "/run/" + re.search(r"^RuntimeDirectory=(\S+)$", token_unit, re.M).group(1)
    assert f"${{VQ_TOKEN:-{token}}}" in (VQ / "bin" / "gh").read_text()
    key = re.search(r"GH_APP_KEY:-(\S+?)\}", token_sh).group(1)
    assert key == f"{cred}/verity-drive.pem" and f'chmod 0400 "$C/{os.path.basename(key)}"' in setup
    assert "vy-vqueue-token.timer" in re.search(r"^Wants=(.*)$", unit, re.M).group(1)
    assert "WantedBy=timers.target" in (VQ / "vy-vqueue-token.timer").read_text()
