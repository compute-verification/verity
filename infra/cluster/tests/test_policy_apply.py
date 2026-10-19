"""`cluster policy apply`: each applier's plan, its write and read-back, and that a second apply is a no-op; unknown keys never
applied, keys without an applier skipped naming why, node 2's gate refusing a write while a timed window is on, and a file
changed since it was read left alone. The read commands and the write run with `sh` on a node's files under tmp_path; only
sudo, systemctl, node 2's clock and the tmux and kubectl reads are stand-ins."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable

import pytest
from test_policy import FILES, LEASE, N1, N2, N2_FILES, NEBIUS, QUIET, DISK, STATUS, T, kueue, queues

from cluster import cli, ledger, policy, policy_apply

SLOTS, LEAN, CPUSETS = policy.CHECK_SLOTS, policy.LEAN_SLOTS, f"{policy.FILL}/cpu-sets"
DCPUS, DROPIN, LOG = policy.DIRECT["direct-cpus"], policy.USER_CPUSET, policy.OVERRIDES
STAMP = "2026-10-03T19:40:30Z"


@pytest.fixture(autouse=True)
def _clock(monkeypatch):
    monkeypatch.setattr(policy, "clock", lambda: T)


class Node:
    """A node's files under `root`, its commands run there with `sh`: every absolute path the commands name moved under it."""

    def __init__(self, root: Path, files: dict[str, str | None], status: str = STATUS, kueue_json: str | None = None):
        self.root, self.status, self.kueue = root, status, kueue_json or kueue()
        self.ran: list[str] = []
        self.after: dict[int, Callable[[], None]] = {}
        self.lock = threading.Lock()
        bin_ = root / "bin"
        self.put({"/bin/sudo": '#!/bin/sh\n[ "$1" = -n ] && shift\nexec "$@"\n',
                  "/bin/systemctl": f'#!/bin/sh\necho "$*" >> {root}/systemctl.log\n'
                                    f'[ "$1" = show ] && cat {root}/allowed\nexit 0\n',
                  "/bin/date": f'#!/bin/sh\n[ "$*" = +%s ] && {{ echo {T}; exit 0; }}\nexec {shutil.which("date")} "$@"\n',
                  "/bin/pgrep": "#!/bin/sh\nexit 1\n",
                  "/allowed": "0-119\n", f"{policy.FILL}/status.txt": status + "\n"})
        for f in bin_.iterdir():
            f.chmod(0o755)
        (root / "tmp").mkdir()
        self.put({p: t for p, t in files.items() if t is not None})

    def put(self, files: dict[str, str]) -> None:
        for p, text in files.items():
            f = self.root / p.lstrip("/")
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)

    def text(self, p: str) -> str | None:
        f = self.root / p.lstrip("/")
        return f.read_text() if f.exists() else None

    def snapshot(self) -> dict[str, bytes]:
        return {str(f.relative_to(self.root)): f.read_bytes() for f in sorted(self.root.rglob("*"))
                if f.is_file() and not f.name.endswith(".log")}

    def run(self, cmd: str) -> str:
        with self.lock:
            self.ran.append(cmd)
            i = len(self.ran)
        if cmd == policy.COMMANDS["lease"]:
            return LEASE
        if cmd == policy.COMMANDS["kueue"]:
            return self.kueue
        for top in ("/workspace/", "/etc/", "/var/", "/proc/", "/run/"):
            cmd = cmd.replace(top, f"{self.root}{top}")
        env = {**os.environ, "PATH": f"{self.root}/bin:{os.environ['PATH']}", "TMPDIR": str(self.root / "tmp")}
        r = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True, env=env)
        if r.returncode:
            raise RuntimeError(f"exited {r.returncode}: {r.stderr.strip()}")
        hook = self.after.pop(i, None)
        if hook:
            hook()
        return r.stdout.replace(str(self.root), "")


def n1(tmp_path: Path, files: dict[str, str | None]) -> Node:
    return Node(tmp_path / "n1", {**FILES, **files, QUIET: None, DISK: None})


N1_DRIFT = {LEAN: "# NAME CAP\ncheck 3   # three at once\naudit 2\n",
            SLOTS: "a 32-63 only=trains\nb 64-95\nd 128-159  # since 2 Oct\n",
            DCPUS: "0-95\n", DROPIN: "# delegate cpuset\n[Service]\nDelegate=cpu memory pids\n"}


def apply(capsys, node: Node, which: str = "nebius-1", *args: str, other: Node | None = None) -> tuple[int, dict]:
    nodes = {N1: node, N2: other or node} if which != "nebius-2" else {N2: node}
    rc = policy_apply.main_apply(NEBIUS, which, *_args(args), as_json=True, runner=lambda n: nodes[n].run)
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == "cluster/policy-apply/v1" and out["t"] == T
    return rc, {n["node"]: n for n in out["nodes"]}


def _args(args: tuple[str, ...]) -> tuple:
    a = list(args)
    get = lambda flag: a[a.index(flag) + 1] if flag in a else None
    return get("--key"), "--apply" in a, get("--as"), get("--reason")


def steps(rep: dict) -> dict[str, tuple[str, str | None]]:
    return {s["key"]: (s["action"], s["path"]) for s in rep["steps"]}


def test_a_dry_run_plans_each_applier_and_writes_nothing(tmp_path, capsys):
    node = n1(tmp_path, N1_DRIFT)
    before = node.snapshot()
    rc, reps = apply(capsys, node)
    assert rc == 1 and node.snapshot() == before
    assert {k: v for k, v in steps(reps[N1]).items() if v[0] == "change"} == {
        f"{N1}.lean-slots.check": ("change", LEAN), f"{N1}.check-slots.c": ("change", SLOTS),
        f"{N1}.direct-cpus": ("change", DCPUS), f"{N1}.user-cpuset.delegate": ("change", DROPIN)}
    files = reps[N1]["files"]
    assert files[DCPUS]["diff"][-2:] == ["-0-95", "+0-95,128-159"]
    assert files[DROPIN]["diff"][-3:] == [" [Service]", "-Delegate=cpu memory pids", "+Delegate=cpu cpuset memory pids"]
    assert reps[N1]["then"] == ["sudo -n systemctl daemon-reload (user.slice keeps its AllowedCPUs)", policy_apply.REEXEC]
    assert [(r["key"], r["by"], r["was"]) for r in reps[N1]["records"]] == [
        (f"{N1}.check-slots.c", "", None), (f"{N1}.direct-cpus", "", "0-95"), (f"{N1}.lean-slots.check", "", "3"),
        (f"{N1}.user-cpuset.delegate", "", "cpu memory pids")]
    assert reps[N1]["result"] is None and not (tmp_path / "n1/systemctl.log").exists()
    policy_apply.main_apply(NEBIUS, "nebius-1", None, False, None, None, False, runner=lambda n: node.run)
    text = capsys.readouterr().out
    assert text.startswith(f"{N1}: 4 to change, 0 by hand, 1 skipped, 0 unknown\n")
    assert f"  skip     {N1}.pools.checks  declared 8-95,128-159  live 32-95,128-159  (no applier: the node's own" in text
    assert f"  change   {N1}.direct-cpus  live 0-95 -> 0-95,128-159  ({DCPUS}; owner infra)\n" in text
    assert text.endswith("(a dry run: --apply --as WHO --reason WHY writes it)\n")


def test_apply_writes_each_file_records_who_and_why_reads_it_back_and_a_second_apply_is_a_no_op(tmp_path, capsys):
    node = n1(tmp_path, N1_DRIFT)
    rc, reps = apply(capsys, node, "nebius-1", "--apply", "--as", "infra", "--reason", "back to the description")
    assert rc == 0, "pools.checks, which has no applier, follows the check slots"
    assert reps[N1]["result"] == {"ok": True, "wrote": [SLOTS, DCPUS, LEAN, DROPIN], "left": {}}
    note = f"cluster policy apply, infra {STAMP}"
    assert node.text(LEAN) == f"# NAME CAP\n# check 3   # three at once   -- until {note}\ncheck 2  # {note}\naudit 2\n"
    assert node.text(SLOTS) == f"a 32-63 only=trains\nb 64-95\nd 128-159  # since 2 Oct\nc 8-31  # {note}\n"
    assert node.text(DCPUS) == "0-95,128-159\n"
    assert node.text(DROPIN) == "# delegate cpuset\n[Service]\nDelegate=cpu cpuset memory pids\n"
    records = ledger.read(tmp_path / "n1" / LOG.lstrip("/"))
    assert ledger.verify(records) == [] and {r["kind"] for r in records} == {"apply"}
    assert [(r["key"], r["value"], r["was"], r["by"], r["reason"], r["path"], r["t"]) for r in records] == [
        (f"{N1}.check-slots.c", "8-31", None, "infra", "back to the description", SLOTS, T),
        (f"{N1}.direct-cpus", "0-95,128-159", "0-95", "infra", "back to the description", DCPUS, T),
        (f"{N1}.lean-slots.check", 2, "3", "infra", "back to the description", LEAN, T),
        (f"{N1}.user-cpuset.delegate", "cpu cpuset memory pids", "cpu memory pids", "infra", "back to the description", DROPIN, T)]
    assert (records[0]["before"], records[0]["after"]) == (policy_apply.sha(N1_DRIFT[SLOTS].encode()),
                                                           policy_apply.sha(node.text(SLOTS).encode()))
    assert (tmp_path / "n1/systemctl.log").read_text().splitlines() == [
        "show -p AllowedCPUs --value user.slice", "daemon-reload", "show -p AllowedCPUs --value user.slice", "--user daemon-reexec"]
    after = node.snapshot()
    rc, reps = apply(capsys, node, "nebius-1", "--apply", "--as", "infra", "--reason", "again")
    assert rc == 0 and reps[N1]["steps"] == [] and reps[N1]["result"] is None and node.snapshot() == after
    assert policy.main_diff(NEBIUS, "nebius-1", True, runner=lambda n: node.run) == 0


def test_node_2_writes_its_slot_and_cpu_set_lines_behind_its_gate_and_leaves_the_cpuset_drop_in_by_hand(tmp_path, capsys):
    files = {**N2_FILES, SLOTS: N2_FILES[SLOTS].replace("64-119", "64-127"), CPUSETS: N2_FILES[CPUSETS].replace("0-47 4 40", "0-47 2 40"),
             DROPIN: "[Service]\nDelegate=cpu memory pids\n"}
    node = Node(tmp_path / "n2", files)
    rc, reps = apply(capsys, node, "nebius-2", "--apply", "--as", "infra", "--reason", "slot b off the device cores")
    assert rc == 1, "the drop-in is left by hand"
    assert steps(reps[N2]) == {f"{N2}.check-slots.b": ("change", SLOTS), f"{N2}.fill.cpu-sets.bc-e6a46970-b6ef-5738-af64-182b143075d4":
                               ("change", CPUSETS), f"{N2}.user-cpuset.delegate": ("by-hand", DROPIN)}
    assert reps[N2]["result"] == {"ok": True, "wrote": [SLOTS, CPUSETS], "left": {f"{N2}.user-cpuset.delegate": "drift"}}
    assert DROPIN not in reps[N2]["files"]
    assert reps[N2]["by_hand"][DROPIN]["diff"][-2:] == ["-Delegate=cpu memory pids", "+Delegate=cpu cpuset memory pids"]
    assert node.text(DROPIN) == files[DROPIN] and not (tmp_path / "n2/systemctl.log").exists()
    assert "64-119 windows=/workspace/pouw/fill/windows lead=90 mem=560  # cluster policy apply" in node.text(SLOTS)
    assert node.text(CPUSETS).splitlines()[-1].startswith("bc-e6a46970-b6ef-5738-af64-182b143075d4 0-47 4 40  # ")
    assert [c.startswith(policy.N2_GATE) for c in node.ran] == [True, True, False, True], "reads, the write, the read back"
    assert node.ran[2].startswith(policy.N2_WRITE_GATE), "the write starts with the gate that needs a fresh quiet status"
    assert [r["key"].split(".")[-1] for r in ledger.read(tmp_path / "n2" / LOG.lstrip("/"))] == [
        "b", "bc-e6a46970-b6ef-5738-af64-182b143075d4"]
    after = node.snapshot()
    rc, reps = apply(capsys, node, "nebius-2", "--apply", "--as", "infra", "--reason", "again")
    assert rc == 1 and [s["action"] for s in reps[N2]["steps"]] == ["by-hand"] and node.snapshot() == after


def test_unknown_keys_are_never_applied(tmp_path, capsys):
    node = n1(tmp_path, {**N1_DRIFT, DCPUS: None})
    (tmp_path / "n1" / DCPUS.lstrip("/")).mkdir()   # `cat` of it fails, as an unreadable file's would
    before = node.snapshot()
    rc, reps = apply(capsys, node, "nebius-1", "--apply", "--as", "infra", "--reason", "r")
    assert rc == 3 and reps[N1]["files"] == {} and reps[N1]["result"] is None
    assert {s["action"] for s in reps[N1]["steps"]} == {"unknown"}
    assert all("never applied while unread: the policy files couldn't be read" in s["why"] for s in reps[N1]["steps"])
    assert node.snapshot() == before and len(node.ran) == len(policy.READS[N1])


@pytest.mark.parametrize("status, why", [
    (STATUS.replace("timed False", "timed True"), "a timed window is on or waiting"),
    (STATUS.replace("19:39:59Z", "19:30:00Z"), "status.txt is 630 s old: is the fill runner running?")])
def test_node_2_is_never_written_while_its_gate_refuses(tmp_path, capsys, status, why):
    node = Node(tmp_path / "n2", {**N2_FILES, SLOTS: "a 0-63\n"}, status=status)
    before = node.snapshot()
    rc, reps = apply(capsys, node, "nebius-2", "--apply", "--as", "infra", "--reason", "r")
    assert rc == 3 and node.snapshot() == before and len(node.ran) == 1
    [s] = [s for s in reps[N2]["steps"] if s["key"] == f"{N2}.check-slots.a"]
    assert s["action"] == "unknown" and why in s["why"]


def test_node_2_with_no_fill_runner_is_planned_as_quiet_saying_so_and_never_written(tmp_path, capsys):
    stale = STATUS.replace("19:39:59Z", "17:00:00Z")
    node = Node(tmp_path / "n2", {**N2_FILES, SLOTS: "a 0-63\n" + N2_FILES[SLOTS].split("\n", 1)[1]}, status=stale)
    before = node.snapshot()
    rc, reps = apply(capsys, node, "nebius-2")
    assert rc == 1 and steps(reps[N2])[f"{N2}.check-slots.a"] == ("change", SLOTS) and node.snapshot() == before
    assert reps[N2]["quiet"].startswith("no fill runner runs (status.txt is 9630 s old), and gpu-lease shows no timed lease")
    policy_apply.main_apply(NEBIUS, "nebius-2", None, False, None, None, False, runner=lambda n: node.run)
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == f"  read as quiet: {reps[N2]['quiet']}"
    assert lines[2] == "  (--apply writes nothing here until the fill runner's status.txt is fresh and quiet again)"
    rc, reps = apply(capsys, node, "nebius-2", "--apply", "--as", "infra", "--reason", "r")
    assert rc == policy_apply.FAILED and node.snapshot() == before
    assert reps[N2]["result"]["why"] == ("nothing written: refused status.txt is 9630 s old: is the fill runner running? "
                                         f"(a write needs its fresh quiet status): {stale}")


def test_a_window_that_starts_after_the_read_refuses_the_write(tmp_path, capsys):
    node = Node(tmp_path / "n2", {**N2_FILES, SLOTS: "a 0-63\n" + N2_FILES[SLOTS].split("\n", 1)[1]})
    node.after[2] = lambda: node.put({f"{policy.FILL}/windows": "2026-10-03T19:41Z 30 # booked meanwhile\n"})
    before = node.snapshot()
    rc, reps = apply(capsys, node, "nebius-2", "--apply", "--as", "infra", "--reason", "r")
    assert rc == policy_apply.FAILED and steps(reps[N2])[f"{N2}.check-slots.a"] == ("change", SLOTS)
    assert reps[N2]["result"] == {"ok": False, "wrote": [], "why": "nothing written: refused window 2026-10-03T19:41Z (30 min) is "
                                  "on or starts within 60 s: " + STATUS, "left": {f"{N2}.check-slots.a": "change"}}
    assert len(node.ran) == 3, "read, exact read, write: nothing read again after a refused write"
    assert {k: v for k, v in node.snapshot().items() if "windows" not in k} == {k: v for k, v in before.items() if "windows" not in k}


def test_a_file_changed_since_it_was_read_is_not_written(tmp_path, capsys):
    node = n1(tmp_path, N1_DRIFT)
    node.after[6] = lambda: node.put({LEAN: N1_DRIFT[LEAN] + "build 1\n"})
    rc, reps = apply(capsys, node, "nebius-1", "--apply", "--as", "infra", "--reason", "r")
    assert rc == policy_apply.FAILED and reps[N1]["result"]["why"] == f"nothing written: changed {LEAN}"
    assert reps[N1]["result"]["wrote"] == []
    assert node.text(SLOTS) == N1_DRIFT[SLOTS] and node.text(DCPUS) == N1_DRIFT[DCPUS] and node.text(LOG) is None


def test_user_slice_losing_its_allowed_cpus_fails_the_apply(tmp_path, capsys):
    node = n1(tmp_path, {DROPIN: N1_DRIFT[DROPIN]})
    sysctl = tmp_path / "n1/bin/systemctl"
    sysctl.write_text(sysctl.read_text().replace("exit 0", f'[ "$1" = daemon-reload ] && echo 0-191 > {tmp_path}/n1/allowed\nexit 0'))
    rc, reps = apply(capsys, node, "nebius-1", "--apply", "--as", "infra", "--reason", "r")
    assert rc == policy_apply.FAILED and reps[N1]["result"]["wrote"] == [DROPIN]
    assert "user.slice AllowedCPUs moved from 0-119 to 0-191: run sudo vy-device-cores" in reps[N1]["result"]["why"]


def test_keys_without_an_applier_and_undeclared_ones_are_skipped_naming_why(tmp_path, capsys):
    node = Node(tmp_path / "n1", {**FILES, LEAN: "check 2\naudit 2\nbuild 3\n", QUIET: None, DISK: None},
                kueue_json=kueue(queues(provers__gpu__nominal="6")))
    rc, reps = apply(capsys, node)
    why = {s["key"]: (s["action"], s["why"]) for s in reps[N1]["steps"]}
    assert rc == 1 and why == {
        f"{N1}.kueue.provers.gpu.nominal": ("skip", "no applier: " + policy_apply.NO_APPLIER["kueue"]),
        f"{N1}.lean-slots.build": ("skip", "undeclared: no value in force to converge to; declare it, or remove it by hand")}
    assert len(node.ran) == len(policy.READS[N1]), "nothing to change, so nothing read exactly"


def test_key_plans_only_that_key_and_those_under_it(tmp_path, capsys):
    node = n1(tmp_path, N1_DRIFT)
    rc, reps = apply(capsys, node, "nebius-1", "--key", f"{N1}.lean-slots")
    assert rc == 1 and list(steps(reps[N1])) == [f"{N1}.lean-slots.check"] and list(reps[N1]["files"]) == [LEAN]
    assert policy_apply.main_apply(NEBIUS, "nebius-1", f"{N1}.nothing", False, None, None, False,
                                   runner=lambda n: node.run) == 2
    assert "no key vy-nebius-1.nothing" in capsys.readouterr().out


def test_two_lines_naming_one_slot_are_left_to_a_person(tmp_path, capsys):
    node = n1(tmp_path, {LEAN: "check 2\naudit 2\naudit 3\n"})
    rc, reps = apply(capsys, node, "nebius-1", "--apply", "--as", "infra", "--reason", "r")
    [s] = reps[N1]["steps"]
    assert rc == 1 and s["action"] == "skip" and s["why"] == f"2 lines of {LEAN} name audit: fix it by hand"
    assert reps[N1]["result"] is None and node.text(LEAN) == "check 2\naudit 2\naudit 3\n"


def test_an_edit_keeps_the_rest_of_each_file():
    note = "cluster policy apply, infra 2026-10-03T19:40:30Z"
    assert policy_apply._line("a 1", f"{N1}.lean-slots.b", 2, note) == (f"a 1\nb 2  # {note}\n", None)
    assert policy_apply._line("# x\na 0-7  # why\n", f"{N1}.check-slots.a", "0-7", note) == ("# x\na 0-7  # why\n", "0-7")
    assert policy_apply._whole("# old note\n4,5\n", f"{N1}.direct-gpus", "none", note) == ("none\n", "4,5")
    assert policy_apply._delegate("", f"{N1}.user-cpuset.delegate", "cpu", note)[0].endswith("\n[Service]\nDelegate=cpu\n")
    assert policy_apply._delegate("[Unit]\n[Service]\nX=1", f"{N1}.user-cpuset.delegate", "cpu", note) == \
        ("[Unit]\n[Service]\nDelegate=cpu\nX=1\n", None)
    with pytest.raises(policy_apply.Skip, match="systemd adds them up"):
        policy_apply._delegate("Delegate=cpu\nDelegate=memory\n", f"{N1}.user-cpuset.delegate", "cpu", note)


def test_a_write_touches_only_policy_files_and_the_log_and_never_reverts_or_restarts_a_unit(tmp_path):
    node = n1(tmp_path, N1_DRIFT)
    p = policy_apply.node_plan(policy.load(NEBIUS), N1, node.run, T, None, "infra", "r")
    cmd = policy_apply.script(p)
    assert set(p.files) == {LEAN, SLOTS, DCPUS, DROPIN}
    assert not set(re.findall(r"[a-z][a-z-]*", cmd.replace('rm -f "$t"', ""))) & {
        "revert", "restart", "stop", "kill", "isolate", "set-property", "rm", "kubectl", "tmux"}
    targets = set(re.findall(r'mv -f (?:"\$t"|\S+) (\S+)', cmd)) | set(re.findall(r">> (\S+)", cmd))
    assert targets == {LEAN, SLOTS, DCPUS, DROPIN, LOG}


def test_the_cli_refuses_apply_without_who_and_why(capsys, monkeypatch, tmp_path):
    node = n1(tmp_path, N1_DRIFT)
    monkeypatch.setattr(policy_apply.status, "ssh", lambda dest, key=None: node.run)
    assert cli.main(["policy", "apply", "--on", "nebius-1", "--apply", "--as", "infra"]) == 2
    assert "--apply needs --as (the handle who decided it) and --reason (why)" in capsys.readouterr().out
    before = node.snapshot()
    assert cli.main(["policy", "apply", "--on", "nebius-1", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["applied"] is False and node.snapshot() == before
