"""The shell hook refuses every listing of processes' full command lines run on this machine, whatever it is piped into, since
a daemon's token sits in its argv and a filter for the daemon's name keeps exactly that line; it points to `ps -eo
pid,etimes,comm`, `pgrep -l` and `procs.py`, which prints each command line with every token-like run cut out. A listing
another machine runs is judged as it always was."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1]
HOOK = GUARD / "shell_hook.py"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", GUARD / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hook():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "dont_write_bytecode", True)      # no __pycache__ in tools/agent-guard, which boot.sh copies file by file
        mp.setitem(sys.modules, "settings", load("settings"))
        yield load("shell_hook")


@pytest.fixture(scope="module")
def procs():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "dont_write_bytecode", True)
        yield load("procs")


def answer(command: str) -> dict:
    out = subprocess.run([sys.executable, HOOK], input=json.dumps({"hook_event_name": "beforeShellExecution", "command": command,
                                                                     "cwd": str(GUARD)}),
                         capture_output=True, text=True, check=True,
                         env={**os.environ, "AGENT_GUARD_MIN_AVAIL": "0", "AGENT_INBOX_POLL_S": "off"}).stdout
    return json.loads(out)


LISTINGS = [
    # filtered: the pattern names the daemon, so the filter keeps exactly its token-bearing line
    "ps -eo pid,args | grep \"[p]attern\"",
    "ps -ef | grep -v grep | grep \"[p]ython\"",
    "ps aux | rg -i cursor",
    "ps -efww | grep exec-daemon",
    "ps auxww | grep daemon",
    "ps -eo pid,etimes,cmd --sort=-etimes | grep agent",
    "pgrep -af cursor",
    "pgrep -fa exec-daemon",
    "pgrep -alf agent",
    "pgrep -u ubuntu -a daemon",
    "pgrep --list-full cursor",
    "pgrep --list-f cursor",
    # every spelling of a ps format that names the command line
    "ps -o pid= -o args=",
    "ps -o pid -o cmd -p 1",
    "ps -eopid,args",
    "ps -e -O args",
    "ps axo pid,command",
    "ps ax o pid,args",
    "ps --format pid,cmd",
    "ps --format=pid,args -e",
    "ps -eo '%p %a'",
    "ps -p 1234 -o pid,args",                             # one process: its argv holds its token as well
    "ps -o args --ppid 4242",
    # full-format and BSD forms
    "ps -f", "ps -ef", "ps -eF", "ps -efww", "ps -Af --forest", "ps -fu research",
    "ps aux", "ps auxww", "ps faux", "ps ax", "ps x", "ps 1234", "ps j", "ps -e w",
    "sudo ps -ef | sort",
    "timeout 5 ps -ef | grep x",
    # variables that change what ps prints
    "PS_FORMAT=pid,args ps -e",
    "env PS_FORMAT=args ps",
    "export PS_FORMAT=pid,cmd",
    "export PS_PERSONALITY=bsd",
    "PS_PERSONALITY=bsd ps -e",
    # other listers
    "top -bc -n1", "top -b -n 1 -c", "top -bcn1 | head", "htop", "btop", "pstree -a", "pstree -ap 1", "pstree --arguments",
    # /proc's command lines
    "cat /proc/123/cmdline",
    "tr '\\0' ' ' < /proc/123/cmdline",
    "xargs -0 < /proc/123/cmdline",
    "xargs -0 echo < /proc/self/cmdline",
    "cat /proc/*/cmdline",
    "strings /proc/1/cmdline",
    "for p in /proc/[0-9]*; do cat $p/cmdline; echo; done",
    "for p in $(pgrep python); do tr '\\0' ' ' < /proc/$p/cmdline; done",
    "cd /proc/1234 && cat cmdline",
    "python3 -c \"print(open('/proc/1/cmdline').read())\"",
    "python3 -c 'import psutil; print([p.cmdline() for p in psutil.process_iter()])'",
    "python3 -c 'import psutil; print([p.info for p in psutil.process_iter([\"cmdline\"])])'",
    "echo \"$(ps -ef)\"",
    "bash -c 'ps aux | grep x'",
]


@pytest.mark.parametrize("command", LISTINGS)
def test_every_listing_of_full_command_lines_is_refused_whatever_follows_it(command):
    out = answer(command)
    assert out["permission"] == "deny", command
    assert "full command lines" in out["agent_message"] or "environment" in out["agent_message"], out["agent_message"]


def test_ps_e_prints_the_environment_and_is_refused_as_such():
    for command in ("ps axe | grep foo", "ps e", "ps -p 1 e"):
        out = answer(command)
        assert out["permission"] == "deny" and "environment variables' values" in out["agent_message"], command


def test_the_refusal_points_to_the_safe_forms_and_never_to_a_filtered_listing(hook):
    message = answer("ps aux | grep cursor")["agent_message"]
    for safe in ("ps -eo pid,etimes,comm", "pgrep -l PATTERN", "python3 tools/agent-guard/procs.py [PATTERN]",
                 "ssh HOST python3 - [PATTERN] < tools/agent-guard/procs.py"):
        assert safe in message
    source = HOOK.read_text()
    for old in ("pgrep -af", "[p]attern", "pid,args |", "pid,stat,etimes,comm"):
        assert old not in source and all(old not in r for r in hook.REASONS.values()), old


SAFE = [
    "ps -eo pid,etimes,comm",
    "ps -eo pid,etimes,comm | grep python",
    "ps -eo pid,stat,etimes,comm,rss",
    "ps -o pid,etime,ucmd --ppid 4242",
    "pgrep -l cursor",
    "pgrep -f quick_tier",                                # pids only
    "pgrep -c python",
    "ps", "ps -e", "ps -A", "ps -e --forest", "ps -u research", "ps -C python3", "ps -p 1234", "ps -l", "ps axc",
    "ps ax -o pid,comm",
    "PS_FORMAT=pid,comm ps -e",
    "top -b -n1 | head",
    "pstree -p",
    "python3 tools/agent-guard/procs.py",
    "python3 tools/agent-guard/procs.py exec-daemon",
    "python3 tools/agent-guard/procs.py 'lake|cargo' | head -20",
    "ssh n1 python3 - slot.py < tools/agent-guard/procs.py",
    "git commit -m 'refuse ps aux and /proc/PID/cmdline'",
    "rg -n 'ps -ef' tools/",
]


@pytest.mark.parametrize("command", SAFE)
def test_a_listing_without_command_lines_and_procs_py_pass(command):
    assert answer(command)["permission"] == "allow", command


@pytest.mark.parametrize("command, permission", [
    ("ssh n1 'ps -eo pid,args | grep \"[s]lot.py\"'", "allow"),
    ("research pods ssh vy-node-1 -- 'ps -eo pid,etimes,args | grep \"[r]esearch run\"'", "allow"),
    ("research pods ssh vy-node-1 -- 'pgrep -af slot.py'", "allow"),
    ("ssh n1 'ps -p 1234 -o args'", "allow"),
    ("ssh n1 'cat /proc/1234/cmdline'", "allow"),
    ("ssh -o BatchMode=yes n1 'ps -ef | grep -v grep | grep lake'", "allow"),
    ("ssh n1 'ps -ef'", "deny"),
    ("ssh n1 ps aux", "deny"),
    ("research pods ssh vy-node-1 -- 'ps -eo pid,etimes,args'", "deny"),
    ("ssh n1 'ps -ef | grep -v grep'", "deny"),
    ("ssh n1 'cat /proc/*/cmdline'", "deny"),
    ("ssh n1 'cat /proc/1/environ'", "deny"),
])
def test_a_listing_another_machine_runs_is_judged_as_it_always_was(command, permission):
    """The node tooling tells whether a tree or a run is in use from a filtered listing over ssh many times an hour: that
    machine's argv, judged by the rule that held before this machine's listings were all refused."""
    assert answer(command)["permission"] == permission, command


TOKEN = "gh" "p_" "Q3x9Lm2Vb7Tn4Rk8Wz1Yc6Hd5"
B64 = "bm90LXRvLWJlLWNvcGllZC1" "kdW1teQ/x9Q2+zZ8w=="


@pytest.mark.parametrize("argv, shown, hidden", [
    (["/exec-daemon/exec-daemon", "--token", "dummy.Not2Copy", "--port", "8080"],
     "/exec-daemon/exec-daemon --token [cut] --port 8080", "dummy.Not2Copy"),
    (["node", "/usr/lib/agent.js", "--auth=dummyNotToBeCopied", "serve"], "node /usr/lib/agent.js --auth=[cut] serve", "dummyNot"),
    (["daemon", "--api-key", "abc", "--password=hunter2", "--secret", "s3", "--private-key-file", "/etc/k.pem"],
     "daemon --api-key [cut] --password=[cut] --secret [cut] --private-key-file /etc/k.pem", "hunter2"),
    (["curl", "-H", "Authorization: Bearer dummyvalue", "https://x.invalid/a?api_key=zz9&token=t0k&page=2"],
     "curl -H Authorization: [cut] https://x.invalid/a?api_key=[cut]&token=[cut]&page=2", "dummyvalue"),
    (["sh", "-c", "curl -H 'Bearer dummyvalue' x"], "sh -c curl -H 'Bearer [cut]' x", "dummyvalue"),
    (["git", "-c", "http.https://github.com/.extraheader=AUTHORIZATION: basic " + B64, "fetch", "origin"],
     "git -c http.https://github.com/.extraheader=AUTHORIZATION: [cut] fetch origin", "bm90"),
    (["git", "push", f"https://x-access-token:{TOKEN}@github.com/o/r.git", "HEAD:refs/heads/b"],
     "git push https://x-access-token:[cut]@github.com/o/r.git HEAD:refs/heads/b", TOKEN),
    (["git", "fetch", f"https://{TOKEN}@github.com/o/r.git"], "git fetch https://[cut]@github.com/o/r.git", TOKEN),
    (["env", "NEBIUS" "_API_KEY=abc123", "HOME=/home/u", "run"], "env NEBIUS_API_KEY=[cut] HOME=/home/u run", "abc123"),
    (["app", '{"password": "hunter2", "user": "me"}'], 'app {"password": "[cut]", "user": "me"}', "hunter2"),
    (["worker", "--id", "8f4b652b17bc5055a120b56940b7c7a9"], "worker --id [cut]", "8f4b652b"),
    (["worker", B64], "worker [cut]", "x9Q2"),
    (["hf", "download", "hf_" "abcdefgh12"], "hf download [cut]", "abcdefgh12"),
    (["python3", "-m", "vllm.entrypoints.openai.api_server", "--max-num-batched-tokens", "8192", "--disable-log-requests"],
     "python3 -m vllm.entrypoints.openai.api_server --max-num-batched-tokens 8192 --disable-log-requests", None),
    (["python3", "/tmp/wt-guard3/tools/agent-guard/procs.py", "/home/ubuntu/.cursor/projects/agent-repos-verity/terminals"],
     "python3 /tmp/wt-guard3/tools/agent-guard/procs.py /home/ubuntu/.cursor/projects/agent-repos-verity/terminals", None),
    (["python3", "/root/.cache/agent-guard/618116497b9f84d5550060c41406998375605514/shell_hook.py"],
     "python3 /root/.cache/agent-guard/[cut]/shell_hook.py", "618116"),
    (["uv", "run", "research", "run", "--on", "auto", "--", "lake", "build", "Security"],
     "uv run research run --on auto -- lake build Security", None),
])
def test_procs_cuts_every_token_shape_and_keeps_the_program_its_subcommand_and_paths(procs, argv, shown, hidden):
    line = procs.cut(argv)
    assert line == shown
    assert hidden is None or hidden not in line


def test_procs_lists_matching_processes_by_their_whole_command_line_and_prints_the_cut_form(procs, capsys):
    fake = [procs.Proc(1, 900, "init", ["/sbin/init"]),
            procs.Proc(42, 5, "exec-daemon", ["/exec-daemon/exec-daemon", "--token", TOKEN]),
            procs.Proc(77, 61, "kworker/0:1", []),
            procs.Proc(os.getpid(), 0, "python3", ["python3", "procs.py", "exec-daemon"])]
    assert procs.main(["exec-daemon"], source=lambda: fake) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == [f"{'PID':>7} {'ELAPSED':>8} {'COMMAND':<15} ARGS",
                   f"{42:>7} {5:>8} {'exec-daemon':<15} /exec-daemon/exec-daemon --token [cut]"]
    assert procs.main([TOKEN[:12]], source=lambda: fake) == 0, "matching reads the whole command line, as pgrep -f does"
    assert TOKEN[:12] not in capsys.readouterr().out
    assert procs.main(["kworker"], source=lambda: fake) == 0 and "[kworker/0:1]" in capsys.readouterr().out
    assert procs.main(["nothing-runs-this"], source=lambda: fake) == 1
    assert procs.main(["("], source=lambda: fake) == 2 and "doesn't compile" in capsys.readouterr().err


def test_procs_reads_stat_and_cmdline_and_never_an_environment(procs, tmp_path, capsys):
    """A process tree stood in for /proc: each process's environ holds a dummy secret, which nothing prints."""
    ticks = os.sysconf("SC_CLK_TCK")
    (tmp_path / "uptime").write_text("1000.00 4000.00\n")
    for pid, name, argv, started in ((7, "daemon (x)", [b"/usr/bin/daemon", b"--key", b"dummy"], 400), (9, "kidle", [], 990)):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "stat").write_text(f"{pid} ({name}) S " + " ".join(["1"] * 18) + f" {started * ticks} 0 0\n")
        (d / "cmdline").write_bytes(b"\0".join(argv) + (b"\0" if argv else b""))
        (d / "environ").write_bytes(b"DUMMY" b"_SECRET=not-to-be-copied\0PATH=/usr/bin\0")
    (tmp_path / "self").mkdir()
    (tmp_path / "11").mkdir()                            # a process that ended between the listing and the read
    listed = sorted(procs.processes(str(tmp_path)), key=lambda p: p.pid)
    assert [(p.pid, p.elapsed, p.name, p.argv) for p in listed] == [
        (7, 600, "daemon (x)", ["/usr/bin/daemon", "--key", "dummy"]), (9, 10, "kidle", [])]
    assert procs.main([], source=lambda: procs.processes(str(tmp_path))) == 0
    out, err = capsys.readouterr()
    assert "11 not listed" in err
    assert "/usr/bin/daemon --key [cut]" in out and "[kidle]" in out
    assert "not-to-be-copied" not in out and "PATH" not in out and "DUMMY" not in out
    assert "environ" not in (Path(procs.__file__).read_text().split('"""', 2)[2]), "the code never names a process's environ"
