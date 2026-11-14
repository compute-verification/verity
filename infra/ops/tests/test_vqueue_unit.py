"""The verified queue's loop as a unit on a node (pods/nebius/vqueue): what loop.sh runs in each mode, with a stand-in for
`python3.12 -m research queue vqueue` that records its argv and environment; how it stops; whose token git and gh get; what
root's oneshot writes; and that deploy.toml installs every file the units name on one node, as a user of its own."""
import inspect
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

import pytest
from research import at_main, merge

from verity_ops import deploy

VQ = deploy.MANIFEST.parent / "vqueue"
LOOP = VQ / "loop.sh"
HELPER = "infra/control/gh_app_token.sh"          # the control pod's own, which token.sh runs on the node
CONFIG, SKIP = 78, 75
KEYS = ("HOME", "PATH", "PYTHONPATH", "TMPDIR", "TRAIN_TREES", "GH_TOKEN", "AWS_ACCESS_KEY_ID", "SLACK_BOT_TOKEN",
        "RESEARCH_AGENT_URL", "RESEARCH_LANE", "GIT_TERMINAL_PROMPT", merge.DRIVE_ENV, merge.LEAN_PUBLISH_ENV, "R2_BUCKET",
        "R2_ENDPOINT", at_main.OPT_OUT)
#: each list's flag, which the loop otherwise reads at the control pod's path, and the node's file under live/
LISTS = {"--deselect": "deselect", "--keep-runs": "keep-runs", "--wide-file": "wide", "--heavy-file": "heavy", "--pins-file": "pins"}
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
        envfile(self.cred / "r2.env", {"AWS_ACCESS_KEY_ID": "dummy-r2"})
        envfile(self.cred / "slack.env", {"SLACK_BOT_TOKEN": "dummy-slack"})
        self.token.parent.mkdir()
        self.token.write_text("dummy-app\n")
        # ci's args as the control pod holds them: its own paths among them, and a comment
        (self.home / "live" / "args").write_text("# depth 2\n--gate check\n--on m1,m2\n--log /data/vqueue/live/decisions.jsonl\n")
        self.env = {**{k: v for k, v in os.environ.items() if k != merge.DRIVE_ENV},
                    "VQ_HOME": str(self.home), "VQ_CRED": str(self.cred), "VQ_TOKEN": str(self.token),
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
    key and the Slack token from the credential files, and GH_TOKEN the App's token."""
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
    assert env[merge.LEAN_PUBLISH_ENV] == os.path.realpath(VQ / "lean_publish.sh") and flag(argv, "--graph2") == os.path.realpath(VQ / "graph2.sh")
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


def test_live_alone_may_move_main_whatever_the_environment_says(node, monkeypatch):
    """The loop moves `main` only with RESEARCH_DRIVE=1 (`merge.on_drive`).  live's tick has it from mode() though nothing
    around the loop sets it; dry-run's and shadow's never do, though the unit's environment and a shell variable of mode()'s
    own name both say 1 (a credential file can say neither: the next test)."""
    def on_drive(value: str | None) -> bool:
        if value is None:
            monkeypatch.delenv(merge.DRIVE_ENV, raising=False)
        else:
            monkeypatch.setenv(merge.DRIVE_ENV, value)
        return merge.on_drive()

    (node.home / "live" / "decisions.jsonl").write_text('{"action": "build"}\n')
    (node.home / "shadow" / "args").write_text("--gate check\n")
    (node.home / "mode").write_text("live\n")
    assert node.loop("once").returncode == 0
    for mode in ("dry-run", "shadow"):
        (node.home / "mode").write_text(f"{mode}\n")
        assert node.loop("once", **{merge.DRIVE_ENV: "1", "drive": "1"}).returncode == 0
    live, dry, shadow = node.ticks()
    assert "--dry-run" in dry["argv"] and "--shadow" in shadow["argv"] and not {"--dry-run", "--shadow"} & set(live["argv"])
    assert on_drive(live["env"][merge.DRIVE_ENV])
    assert not on_drive(dry["env"][merge.DRIVE_ENV]) and not on_drive(shadow["env"][merge.DRIVE_ENV])


#: names a credential file might set to change what a tick runs: the form's own variables, the paths it runs from, what moves
#: `main` and whose code runs, and the token the App mints
FORM = ("drive", "flag", "mode", "dir", "H", "CODE", "CLONE", "PY", "LIB", "LISTS", "PATH", "PYTHONPATH", "HOME", "BASH_ENV",
        merge.DRIVE_ENV, "GH_TOKEN", "RESEARCH_AT_CHECKOUT")


def test_a_credential_file_is_data_naming_only_the_keys_the_loop_reads(node):
    """root's files in /etc/vy/vqueue are read as KEY=VALUE lines, never run: a key the loop doesn't read (`drive=1`, `flag=`,
    which once turned a dry-run tick into a live one, or anything else of FORM) refuses the tick with 78, naming the file and
    the key and never the value, in dry-run and shadow alike; a value is taken as written, its quotes aside, and never run."""
    (node.home / "shadow" / "args").write_text("--gate check\n")
    zz, secret = node.cred / "zz.env", "not-to-be-printed"
    for mode in ("dry-run", "shadow"):
        (node.home / "mode").write_text(f"{mode}\n")
        for lines in ([f"{k}={secret}" for k in FORM] + ["drive=1\nflag=", "export drive=1", "flag=", f"{secret}"]):
            zz.write_text(f"# set by hand\n\n{lines}\n")
            r = node.loop("once")
            assert r.returncode == CONFIG and not node.ticks() and f"{zz}" in r.stderr and secret not in r.stderr, (mode, lines)
    ran = node.home / "ran"
    zz.write_text(f"R2_BUCKET=\"dummy-bucket\"\nexport R2_ENDPOINT='dummy $(touch {ran})'\nSLACK_BOT_TOKEN=a=b\n")
    assert node.loop("once").returncode == 0 and not ran.exists()
    [t] = node.ticks()
    assert "--shadow" in t["argv"] and t["env"]["SLACK_BOT_TOKEN"] == "a=b"
    assert (t["env"]["R2_BUCKET"], t["env"]["R2_ENDPOINT"]) == ("dummy-bucket", f"dummy $(touch {ran})")
    # a value sh would read as other than its text is refused, never taken one way where the pod's sh took it another
    for value in ("a b", "x # note", "a&b", "~/k", '"a$b"', '"a\\b"', "'a'b", '"a"b', "a\\b", "`id`", "$HOME"):
        zz.write_text(f"R2_BUCKET={value}\n")
        r = node.loop("once")
        assert r.returncode == CONFIG and "R2_BUCKET" in r.stderr and len(node.ticks()) == 1, value


def test_a_tick_runs_research_from_ci_s_pinned_checkout_not_from_its_origin_main(node, tmp_path, monkeypatch):
    """`research` re-executes from origin/main's code when the checkout it runs from is only behind it (`at_main.reexec`),
    and ci's pin is such a checkout whenever main has moved: the tick's environment keeps the pin's code, where the same
    checkout without it runs origin/main's."""
    assert node.loop("once").returncode == 0
    [t] = node.ticks()
    pin = tmp_path / "pin"
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", str(pin)]
    (pin / "tools/research").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(pin)], check=True)
    for version in ("pin", "main"):
        (pin / "tools/research/version").write_text(version)
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-qm", version], check=True)
    subprocess.run([*git, "update-ref", "refs/remotes/origin/main", "HEAD"], check=True)
    subprocess.run([*git, "checkout", "-q", "--detach", "HEAD~"], check=True)
    for value, runs_main in ((t["env"][at_main.OPT_OUT], False), (None, True)):
        if value is None:
            monkeypatch.delenv(at_main.OPT_OUT, raising=False)
        else:
            monkeypatch.setenv(at_main.OPT_OUT, value)
        ran = []
        at_main.reexec(["queue", "vqueue"], repo=pin, cache=tmp_path / "cache", execve=lambda *a: ran.append(a))
        assert bool(ran) == runs_main, value


def test_every_tick_reads_the_five_lists_at_the_node_s_paths_and_names_the_missing_in_one_line(node):
    """The loop's defaults for --deselect, --keep-runs, --wide-file, --heavy-file and --pins-file are the control pod's paths:
    on the node each is its file under live/, in every mode and after ci's args, so a line of args naming the pod's never wins.
    A missing file reads as empty there, as it does on the pod, and the tick says which in one line and runs."""
    live = node.home / "live"
    (live / "args").write_text("--gate check\n--pins-file /data/drive/pins\n--graph2 /root/g2remote.sh\n")
    (live / "heavy").write_text("")
    r = node.loop("once")
    assert r.returncode == 0, r.stderr
    said = [ln for ln in r.stderr.splitlines() if "reads each as empty" in ln]
    assert said == [f"vy-vqueue: no deselect keep-runs wide pins in {live}: the loop reads each as empty; copy ci's from the "
                    "host the loop ran on"]
    for f in LISTS.values():
        (live / f).write_text("")
    (node.home / "mode").write_text("shadow\n")
    (node.home / "shadow" / "args").write_text("--gate check\n--keep-runs /data/vqueue/live/keep-runs\n")
    r = node.loop("once")
    assert r.returncode == 0 and "reads each as empty" not in r.stderr
    for t in node.ticks():
        assert {o: flag(t["argv"], o) for o in LISTS} == {o: str(live / f) for o, f in LISTS.items()}
        assert flag(t["argv"], "--graph2") == os.path.realpath(VQ / "graph2.sh")


FAKE_MAIN = """import json, os, sys
with open(os.environ["FAKE_CALLS"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:3] == ["pods", "ssh"] and "--print" in sys.argv:
    if sys.argv[3] != os.environ.get("FAKE_KNOWN", sys.argv[3]):
        sys.exit(f"research pods ssh: no machine named {sys.argv[3]!r}")
    print(os.environ["FAKE_SSH"])
elif sys.argv[1:3] == ["pods", "ssh"]:
    print(os.environ.get("FAKE_UNIT", "activating"))
"""
FAKE_MERGE = """import json, os
def push_mirror(repo, sha, name):
    with open(os.environ["FAKE_CALLS"], "a") as f:
        f.write(json.dumps(["push_mirror", str(repo), sha, name]) + "\\n")
    print(f"research merge: pushed main ({sha[:12]}) to {name}")
    return os.environ.get("FAKE_PUSH", "1") == "1"
"""


class Research:
    """`python3.12 -m research` and `research.merge.push_mirror` as the node's wrappers call them, stood in for under tmp: each
    call one JSON line in ``calls``; `pods ssh NAME --print` prints FAKE_SSH for a name FAKE_KNOWN allows, `pods ssh NAME --
    CMD` prints FAKE_UNIT (the unit's state), and push_mirror succeeds unless FAKE_PUSH is 0."""

    def __init__(self, tmp: Path):
        pkg, self.calls, self.home = tmp / "fake" / "research", tmp / "calls.jsonl", tmp / "home"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(FAKE_MAIN)
        (pkg / "merge.py").write_text(FAKE_MERGE)
        (self.home / "live").mkdir(parents=True)
        self.env = {**{k: v for k, v in os.environ.items() if not k.startswith(("VQ_", "FAKE_"))}, "HOME": str(self.home), "PYTHONPATH": str(pkg.parent), "VQ_PYTHON": sys.executable,
                    "FAKE_CALLS": str(self.calls)}

    def made(self) -> list[list[str]]:
        return [json.loads(ln) for ln in self.calls.read_text().splitlines()] if self.calls.exists() else []


GRAPH2 = ["sudo", "-n", "python3.12", "/root/executor/infra/graph2.py", "--db", "/root/executor/graph2.db", "--events",
          "/root/executor/events"]


def test_graph2_s_wrapper_reaches_its_machine_by_name_with_each_argument_whole(tmp_path):
    """The pod's wrapper writes graph2's machine as an address; the node's asks the register for the machine's ssh line by
    name (VQ_GRAPH2_NODE, default vy-nebius-cpu-7) and runs graph2 there over one shared connection, each argument one word
    to the remote shell however it is spelled, and the register's line taken as words, never run.  A name the register lacks
    runs nothing."""
    fake = Research(tmp_path)
    ssh = script(tmp_path / "bin" / "ssh", f'#!{sys.executable}\nimport json, os, sys\n'
                 'open(os.environ["SSH_OUT"], "a").write(json.dumps(sys.argv[1:]) + "\\n")\n')
    out, odd = tmp_path / "ssh.jsonl", tmp_path / "ran"
    env = {**fake.env, "PATH": f"{ssh.parent}:{fake.env['PATH']}", "SSH_OUT": str(out), "FAKE_KNOWN": "vy-nebius-cpu-7",
           "FAKE_SSH": shlex.join(["ssh", "-i", "/k", "-p", "22", f"research@$(touch {odd})"])}

    def graph2(*args: str, **more: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(VQ / "graph2.sh"), *args], env={**env, **more}, capture_output=True, text=True)

    words = ["list", "a b;c $(id)", "it's", "two\nlines", "", "\\'\"", "--json"]
    assert graph2(*words).returncode == 0
    [argv] = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert argv[:-1] == ["-i", "/k", "-p", "22", "-o", "ControlMaster=auto", "-o", f"ControlPath={fake.home}/.runpod/ssh/cm-g2-%C",
                         "-o", "ControlPersist=600", f"research@$(touch {odd})"] and not odd.exists()
    # ssh hands the command to the remote user's shell: shlex.split parses it by POSIX sh's rules, bash's alone not among them
    assert shlex.split(argv[-1]) == [*GRAPH2, *words]
    assert fake.made() == [["pods", "ssh", "vy-nebius-cpu-7", "--print"]]
    assert graph2().returncode == 0
    assert shlex.split(json.loads(out.read_text().splitlines()[-1])[-1]) == GRAPH2
    r = graph2("list", VQ_GRAPH2_NODE="vy-nowhere")
    assert r.returncode == 1 and "no machine named 'vy-nowhere'" in r.stderr and len(out.read_text().splitlines()) == 2


def test_the_lean_publish_pushes_the_full_sha_to_the_cache_s_machine_then_starts_its_publisher(tmp_path):
    """After a landing touching Lean, the loop's after-landing step runs lean_publish.sh SHA in a checkout of the loop's
    clone (merge.lean_publish).  The machine is a registered name (VQ_LEAN_NODE, default vy-nebius-cpu-4); the push is the merge
    mirror's (push_mirror) of the full sha; the publisher's unit is started without waiting.  A sha the clone lacks, a push
    that fails and a unit that doesn't start each exit 1, starting nothing after, with the log's last lines on stdout, which
    merge.lean_publish quotes."""
    assert list(inspect.signature(merge.push_mirror).parameters)[:3] == ["repo", "sha", "name"]
    fake, clone = Research(tmp_path), tmp_path / "clone"
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", str(clone)]
    subprocess.run(["git", "init", "-q", str(clone)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    sha = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    start = ["pods", "ssh", "vy-nebius-cpu-4", "--", "sudo -n systemctl start --no-block vy-lean-cache-daily.service; "
             "systemctl is-active vy-lean-cache-daily.service"]
    log = fake.home / "live" / "lean-publish.log"

    def publish(ref: str, **more: str) -> subprocess.CompletedProcess:
        fake.calls.unlink(missing_ok=True)
        return subprocess.run([str(VQ / "lean_publish.sh"), ref], cwd=clone, env={**fake.env, **more}, capture_output=True, text=True)

    r = publish(sha[:12])
    assert r.returncode == 0, r.stdout + r.stderr
    assert fake.made() == [["push_mirror", str(clone.resolve()), sha, "vy-nebius-cpu-4"], start]
    lines = log.read_text().splitlines()
    assert re.fullmatch(rf"\S+Z publish {sha[:12]} on vy-nebius-cpu-4", lines[0]) and lines[-1] == "start: activating"
    assert r.stdout.splitlines() == lines[-3:]
    r = publish(sha, FAKE_UNIT="failed")
    assert r.returncode == 1 and fake.made()[-1] == start and r.stdout.splitlines()[-1] == "start: failed"
    r = publish(sha, FAKE_PUSH="0")
    assert r.returncode == 1 and fake.made() == [["push_mirror", str(clone.resolve()), sha, "vy-nebius-cpu-4"]]
    r = publish("0" * 40)
    assert r.returncode == 1 and fake.made() == [] and f"{'0' * 40}: no such commit in {clone.resolve()}" in r.stdout
    assert publish(sha, VQ_LEAN_NODE="vy-other").returncode == 0
    pushed, started = fake.made()
    assert pushed[3] == started[2] == "vy-other"


def test_the_lean_publish_s_push_and_ssh_end_inside_the_time_merge_gives_the_script():
    """merge.lean_publish stops waiting at LEAN_PUBLISH_TIMEOUT_S and kills the script alone, so a push or an ssh still
    running would outlive it: each runs under `timeout` (which ends its whole process group), and the two bounds together
    fit inside merge's."""
    text = (VQ / "lean_publish.sh").read_text()
    bounds = {k: int(v) for k, v in re.findall(r"\b(push_s|start_s)=(\d+)\b", text)}
    assert set(bounds) == {"push_s", "start_s"} and sum(bounds.values()) < merge.LEAN_PUBLISH_TIMEOUT_S, bounds
    assert re.search(r"""timeout "\$push_s" "\$py" -c 'import .*push_mirror""", text)
    assert re.search(r'timeout "\$start_s" "\$py" -m research pods ssh "\$node" --', text)
    assert len(re.findall(r"\btimeout\b", "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#")))) == 2


def started(unit: Path) -> set[str]:
    """What `systemctl is-active` says of ``unit`` once started and still running: a oneshot is `activating` until its command
    exits, and `active` only after it, with RemainAfterExit=yes; a unit of any other type is `active`."""
    service = unit.read_text().partition("[Service]")[2].partition("\n[")[0]
    keys = dict(ln.split("=", 1) for ln in service.splitlines() if "=" in ln and not ln.lstrip().startswith("#"))
    if keys.get("Type", "simple") != "oneshot":
        return {"active"}
    return {"activating", "active"} if keys.get("RemainAfterExit", "no") in {"yes", "true", "1", "on"} else {"activating"}


def test_the_lean_publish_holds_the_publisher_started_by_the_states_its_unit_file_gives(tmp_path):
    """The publisher's unit, as the tree installs it on the cache's machine, decides what a started publisher reports: the
    wrapper succeeds on those states and fails on every other, so a unit that ran and ended (or never ran) is no success."""
    unit = deploy.MANIFEST.parent / "vy-lean-cache-daily.service"
    fake, clone = Research(tmp_path), tmp_path / "clone"
    subprocess.run(["git", "init", "-q", str(clone)], check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", str(clone), "commit", "-q",
                    "--allow-empty", "-m", "x"], check=True)
    running = started(unit)
    for state in ("active", "activating", "inactive", "failed", "deactivating", "unknown"):
        r = subprocess.run([str(VQ / "lean_publish.sh"), "HEAD"], cwd=clone, env={**fake.env, "FAKE_UNIT": state},
                           capture_output=True, text=True)
        assert (r.returncode == 0) == (state in running), (state, running)


def test_a_node_missing_the_clone_the_pin_or_the_args_says_which_and_runs_nothing(node):
    for gone, fix in ((node.home / "live" / "args", "the loop's flags"), (node.home / "src", "check ci's pin out"),
                      (node.clone / ".git", "copy the loop's clone")):
        kept = gone.with_name(gone.name + ".kept")
        gone.rename(kept)
        r = node.loop("once")
        assert r.returncode == CONFIG and fix in r.stderr and not node.ticks(), gone
        kept.rename(gone)
    assert node.loop("once").returncode == 0 and len(node.ticks()) == 1
    # what args is told to hold is no flag the tick passes itself, which would lose to the tick's
    own = {a for a in node.ticks()[0]["argv"] if a.startswith("--")} - {"--gate", "--on"}
    (node.home / "live" / "args").unlink()
    told = set(re.findall(r"--[a-z][-a-z0-9]*", node.loop("once").stderr))
    assert told and not told & own, told & own


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
