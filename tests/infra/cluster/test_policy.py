"""`cluster policy`: the description's policy keys and their owners, and `diff` against node outputs in the shapes the live
node returns: drift, undeclared and unknown keys, the exit codes, and that the commands only read."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from cluster import cli, ledger, policy
from cluster.ledger import epoch

NEBIUS = Path(__file__).parents[3] / "infra" / "cluster" / "descriptions" / "nebius.toml"
N2 = "vy-nebius-2"
T = epoch("2026-10-03T19:40:30Z")

STATUS = ("2026-10-03T19:39:59Z free GPUs 7/8, kept free -, waiters -, timed False, window waiting False, backfill False, "
          "queued 0 (gpu 0, cpu 0), done 1033, failed 30, held for an unknown exit 0")
CPU_SETS = f"{N2}.fill.cpu-sets.bc-e6a46970-b6ef-5738-af64-182b143075d4"
N2_FILES = {f"{policy.FILL}/windows": "# start UTC, minutes\n2026-10-03T18:25Z 10 # merge checks off\n",
            f"{policy.FILL}/cpu-sets": "bc-e6a46970-b6ef-5738-af64-182b143075d4 0-47 4 40  # verifies\n",
            policy.CHECK_SLOTS: "a 0-63 windows=/workspace/pouw/fill/windows lead=90 mem=560\nb 64-119 windows=/workspace/pouw/fill/windows lead=90 mem=560\n",
            "/etc/vy/direct-gpus": None, "/etc/vy/direct-cpus": None,
            policy.USER_CPUSET: "[Service]\nDelegate=cpu cpuset memory pids\n"}


def dump(files: dict[str, str | None]) -> str:
    return "".join(f"{policy.ABSENT_MARK}{p}\n" if t is None else f"{policy.FILE_MARK}{p}\n{t}\n" for p, t in files.items())


def outputs(**over) -> dict[str, str | Exception]:
    out = {"n2": f"status {STATUS}\n" + dump(N2_FILES)}
    out.update(over)
    return out


def fake(outs: dict[str, str | Exception], ran: list | None = None):
    by_cmd = {policy.COMMANDS[name]: out for name, out in outs.items()}

    def for_node(node: str):
        def run(cmd: str) -> str:
            if ran is not None:
                ran.append((node, cmd))
            out = by_cmd[cmd]
            if isinstance(out, Exception):
                raise out
            return out
        return run
    return for_node


@pytest.fixture(autouse=True)
def _clock(monkeypatch):
    monkeypatch.setattr(policy, "clock", lambda: T)


def diff(capsys, which: str = "all", description: Path = NEBIUS, ran: list | None = None, **over) -> tuple[int, dict]:
    rc = policy.main_diff(description, which, True, runner=fake(outputs(**over), ran))
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == "cluster/policy-diff/v1" and out["t"] == policy.clock()
    return rc, {r["key"]: r for rep in out["nodes"] for r in rep["keys"]}


def described(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "d.toml"
    p.write_text(NEBIUS.read_text().split("\n[policy.")[0] + "\n" + text)
    return p


def test_the_nebius_policy_loads_with_an_owner_and_a_source_per_key():
    keys = policy.load(NEBIUS)
    assert keys[f"{N2}.check-slots.a"].value.startswith("0-63 windows=")
    assert {k.owner for k in keys.values()} == {"infra"}
    assert all(policy.source(k) for k in keys)
    assert policy.source(f"{N2}.fill.cpu-sets.x") == "/workspace/pouw/fill/cpu-sets"


def test_an_owner_covers_the_keys_below_it_unless_a_table_names_its_own(tmp_path):
    keys = policy.load(described(tmp_path, '[policy.vy-nebius-2]\nowner = "infra"\ndirect-gpus = "none"\n'
                                           'check-slots = { owner = "proofs", a = "0-63" }\n'))
    assert {k: v.owner for k, v in keys.items()} == {f"{N2}.direct-gpus": "infra", f"{N2}.check-slots.a": "proofs"}


N2_TABLE = '[policy.vy-nebius-2]\nowner = "infra"\n'


@pytest.mark.parametrize("text, why", [
    ('[policy.vy-nebius-2]\ndirect-gpus = "none"\n', f"{N2}.direct-gpus: no owner"),
    ('[policy.vy-nebius-9]\nowner = "infra"\ndirect-gpus = "none"\n', "unknown node 'vy-nebius-9'"),
    ("[policy]\nvy-nebius-2 = 3\n", "policy.vy-nebius-2 is not a table"),
    (N2_TABLE + 'lean-slots = { audit = 2 }\n', "no reader for 'lean-slots' on vy-nebius-2"),
    (N2_TABLE + 'direct-cpus = ["0-3"]\n', "a policy value is a string or a number"),
    (N2_TABLE + 'check-slots = { d = "0-47x" }\n', "bad CPU list item"),
    (N2_TABLE + 'user-cpuset = { allowed = "0-3" }\n', "the cpuset drop-in's only key is user-cpuset.delegate"),
    (N2_TABLE + '[policy.vy-nebius-2.fill.window]\n"2026-10-04T06:30:00Z" = 30\n', "write the start as 2026-10-04T06:30Z"),
    (N2_TABLE + '[policy.vy-nebius-2.fill.window]\n"tomorrow" = 30\n', "is not a UTC start"),
    (N2_TABLE + '[policy.vy-nebius-2.fill.window]\n"2026-10-04T06:30Z" = 0\n', "is not a number of minutes"),
])
def test_a_description_is_refused_naming_each_problem(tmp_path, capsys, text, why):
    p = described(tmp_path, text)
    with pytest.raises(policy.PolicyError, match=re.escape(why)):
        policy.load(p)
    assert policy.main_show(p, "all", False) == 2 and policy.main_diff(p, "all", False, runner=fake({})) == 2
    assert why in capsys.readouterr().out


def test_values_compare_in_one_spelling_per_surface():
    c = policy.canon
    assert c(f"{N2}.direct-cpus", "128-159,8-95") == c(f"{N2}.direct-cpus", "8-95,128-159") == "8-95,128-159"
    assert c(f"{N2}.check-slots.d", "0-47 lead=30 windows=/w") == c(f"{N2}.check-slots.d", " 0-15,16-47  windows=/w lead=30")
    assert c(f"{N2}.fill.window.2026-10-04T06:30Z", "190.0") == c(f"{N2}.fill.window.2026-10-04T06:30Z", 190) == "190"
    assert c(f"{N2}.fill.cpu-sets.x", "47,0-46 4 40") == "0-47 4 40"
    assert c(f"{N2}.user-cpuset.delegate", "cpu  cpuset\tmemory pids") == "cpu cpuset memory pids"
    with pytest.raises(ValueError, match="not a number of minutes"):
        c(f"{N2}.fill.window.2026-10-04T06:30Z", "0")


def test_live_as_declared_is_exit_0(capsys):
    rc, rows = diff(capsys)
    assert rc == 0 and {r["status"] for r in rows.values()} == {"ok"}
    assert set(rows) == set(policy.load(NEBIUS))


def test_drift_absent_and_undeclared_keys_are_exit_1(capsys):
    files = {**N2_FILES, policy.CHECK_SLOTS: "a 0-31 windows=/workspace/pouw/fill/windows lead=90 mem=560\nc 0-7\n",
             f"{policy.FILL}/cpu-sets": ""}
    rc, rows = diff(capsys, "nebius-2", n2=f"status {STATUS}\n" + dump(files))
    assert rc == 1
    found = {k: (r["status"], r["declared"], r["live"], r["why"]) for k, r in rows.items() if r["status"] != "ok"}
    assert found == {
        f"{N2}.check-slots.a": ("drift", "0-63 windows=/workspace/pouw/fill/windows lead=90 mem=560",
                                "0-31 windows=/workspace/pouw/fill/windows lead=90 mem=560", "live differs"),
        f"{N2}.check-slots.b": ("drift", "64-119 windows=/workspace/pouw/fill/windows lead=90 mem=560", None, "absent live"),
        f"{N2}.check-slots.c": ("undeclared", None, "0-7", "live, and not in the description"),
        CPU_SETS: ("drift", "0-47 4 40", None, "absent live"),
    }
    assert rows[f"{N2}.check-slots.c"]["owner"] == "infra"


def test_text_names_each_key_that_differs_with_its_owner_and_source(capsys):
    files = {**N2_FILES, policy.USER_CPUSET: "[Service]\nDelegate=cpu\n"}
    rc = policy.main_diff(NEBIUS, "nebius-2", False, runner=fake(outputs(n2=f"status {STATUS}\n" + dump(files))))
    out = capsys.readouterr().out.splitlines()
    assert rc == 1 and out[0] == f"{N2}: 3 as declared, 1 drift, 0 undeclared, 0 unknown"
    assert out[1].split() == ["drift", f"{N2}.user-cpuset.delegate", "declared", "cpu", "cpuset", "memory", "pids", "live", "cpu",
                              "(live", "differs;", "owner", "infra;", f"{policy.USER_CPUSET}", "Delegate=)"]


def test_a_live_value_that_doesnt_parse_is_drift(capsys):
    files = {**N2_FILES, policy.CHECK_SLOTS: "a 0-6x\nb 64-119 windows=/workspace/pouw/fill/windows lead=90 mem=560\n"}
    rc, rows = diff(capsys, "nebius-2", n2=f"status {STATUS}\n" + dump(files))
    assert rc == 1 and rows[f"{N2}.check-slots.a"]["status"] == "drift"
    assert "the live value doesn't parse" in rows[f"{N2}.check-slots.a"]["why"]


@pytest.mark.parametrize("over", [{"n2": RuntimeError("ssh: connect to host: timed out")}])
def test_an_unread_command_makes_its_keys_unknown_and_exit_3(capsys, over):
    rc, rows = diff(capsys, "nebius-2", **over)
    assert rc == 3 and rows and {r["status"] for r in rows.values()} == {"unknown"}


def test_live_output_of_an_unexpected_shape_makes_the_node_unknown_and_exit_3_never_drift(capsys):
    rc, rows = diff(capsys, "nebius-2", n2=f"status {STATUS}\n" + dump({**N2_FILES, policy.OVERRIDES: "[1]\n"}))
    assert rc == 3 and rows and {r["status"] for r in rows.values()} == {"unknown"}


def test_node_2_refusing_to_be_read_runs_one_command_and_every_key_is_unknown(capsys):
    ran: list = []
    refused = ("refused a timed window is on or waiting: 2026-10-03T19:20:01Z free GPUs 0/8, kept free -, waiters -, "
               "timed True, window waiting False\n")
    rc, rows = diff(capsys, "nebius-2", ran=ran, n2=refused)
    assert rc == 3 and ran == [(N2, policy.COMMANDS["n2"])]
    assert set(rows) == {k for k in policy.load(NEBIUS) if k.startswith(N2 + ".")}
    assert {r["status"] for r in rows.values()} == {"unknown"}
    assert "node 2 refused to be read, and nothing else there was: a timed window is on or waiting: 2026-10-03T19:20:01Z" \
        in rows[f"{N2}.user-cpuset.delegate"]["why"]


@pytest.mark.parametrize("first", [f"timed {STATUS}", STATUS, ""])
def test_node_2_output_that_is_neither_a_status_nor_a_refusal_is_unknown(capsys, first):
    rc, rows = diff(capsys, "nebius-2", n2=first + "\n" + dump(N2_FILES))
    assert rc == 3 and all(r["status"] == "unknown" and "neither a status nor a refusal" in r["why"] for r in rows.values())


def test_node_2_windows_that_ended_are_left_out_and_a_new_booking_is_undeclared(tmp_path, capsys):
    p = described(tmp_path, '[policy.vy-nebius-2]\nowner = "infra"\n[policy.vy-nebius-2.fill.window]\n'
                                       '"2026-10-03T18:25Z" = 10\n"2026-10-04T06:30Z" = 190\n')
    windows = ("2026-10-03T18:25Z 10\n2026-10-04T06:30:00Z 190.0 # circuits\n2026-10-04T10:30 30 # served\n"
               "garbage 5 # fill_runner.py skips it\n")
    rc, rows = diff(capsys, "nebius-2", description=p, n2=f"status {STATUS}\n" + dump({f"{policy.FILL}/windows": windows}))
    assert rc == 1
    assert {k: r["status"] for k, r in rows.items() if ".fill." in k} == {
        f"{N2}.fill.window.2026-10-04T06:30Z": "ok", f"{N2}.fill.window.2026-10-04T10:30Z": "undeclared"}
    shown = {r["key"]: r.get("ended", False) for r in policy.show(policy.load(p), [N2], T)}
    assert shown == {f"{N2}.fill.window.2026-10-03T18:25Z": True, f"{N2}.fill.window.2026-10-04T06:30Z": False}


def test_show_prints_each_key_with_its_owner_and_source(capsys):
    assert policy.main_show(NEBIUS, "nebius-2", True, log=None) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == "cluster/policy/v1" and {r["key"].split(".")[0] for r in out["keys"]} == {N2}
    assert out["overrides"] is None
    assert {"key": f"{N2}.user-cpuset.delegate", "value": "cpu cpuset memory pids", "owner": "infra",
            "source": f"{policy.USER_CPUSET} Delegate="} in out["keys"]
    assert policy.main_show(NEBIUS, "all", False, log=None) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["KEY", "VALUE", "OWNER", "SOURCE"] and len(lines) == 1 + len(policy.load(NEBIUS))


def test_show_key_prints_one_key_or_those_under_it_and_refuses_a_key_naming_none(capsys):
    assert policy.main_show(NEBIUS, "all", True, f"{N2}.check-slots", log=None) == 0
    rows = json.loads(capsys.readouterr().out)["keys"]
    assert {r["key"] for r in rows} == {f"{N2}.check-slots.a", f"{N2}.check-slots.b"}
    assert policy.main_show(NEBIUS, "all", True, f"{N2}.check-slot", log=None) == 2
    assert "no key vy-nebius-2.check-slot, nor any under it" in capsys.readouterr().out


def test_the_cli_reads_the_default_description_and_refuses_a_bad_one(capsys, tmp_path):
    assert cli.main(["policy", "show", "--on", "nebius-2", "--json", "--log", str(tmp_path / "none.jsonl")]) == 0
    assert json.loads(capsys.readouterr().out)["description"] == str(NEBIUS)
    bad = described(tmp_path, '[policy.vy-nebius-2]\ndirect-gpus = "none"\n')
    assert cli.main(["policy", "diff", "--description", str(bad)]) == 2
    with pytest.raises(SystemExit) as e:
        cli.main(["policy", "diff", "--on", "nebius-1"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        cli.main(["policy", "override", f"{N2}.check-slots.a", "0-31", "--until", "2026-10-04T10:30:00Z"])
    assert e.value.code == 2


SLOT_A = f"{N2}.check-slots.a"
SLOT_A_WAS = "0-63 windows=/workspace/pouw/fill/windows lead=90 mem=560"
SLOT_A_NOW = "0-31 windows=/workspace/pouw/fill/windows lead=90 mem=560"
UNTIL = "2026-10-04T10:30:00Z"


def override(capsys, log: Path, key: str = SLOT_A, value: str = SLOT_A_NOW, until: str = UNTIL, by: str = "infra",
             reason: str = "the 15:07Z shift", apply: bool = True) -> tuple[int, str]:
    rc = cli.main(["policy", "override", key, value, "--until", until, "--by", by, "--reason", reason, "--log", str(log),
                   *(["--apply"] if apply else [])])
    return rc, capsys.readouterr().out


def test_an_override_is_a_dry_run_until_applied_and_show_merges_it_with_who_why_and_until(capsys, tmp_path):
    log = tmp_path / "policy" / "overrides.jsonl"
    rc, out = override(capsys, log, apply=False)
    assert rc == 0 and not log.exists()
    assert out.startswith(f"would override {SLOT_A}: {SLOT_A_WAS} -> {SLOT_A_NOW} until {UNTIL}, by infra: the 15:07Z shift "
                          f"(in {log}; --apply")
    rc, out = override(capsys, log)
    assert rc == 0 and out.startswith(f"overrode {SLOT_A}: {SLOT_A_WAS} -> {SLOT_A_NOW} until {UNTIL}")
    [r] = ledger.read(log)
    assert ledger.verify([r]) == [] and (r["kind"], r["t"], r["key"], r["value"], r["until"], r["by"]) == \
        ("override", T, SLOT_A, SLOT_A_NOW, UNTIL, "infra")
    assert policy.main_show(NEBIUS, "nebius-2", True, SLOT_A, log) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["overrides"] == str(log) and out["keys"] == [
        {"key": SLOT_A, "value": SLOT_A_NOW, "owner": "infra", "source": policy.CHECK_SLOTS,
         "override": {"by": "infra", "reason": "the 15:07Z shift", "until": UNTIL, "was": SLOT_A_WAS}}]
    assert policy.main_show(NEBIUS, "nebius-2", False, SLOT_A, log) == 0
    assert f"(override by infra until {UNTIL}, was {SLOT_A_WAS}: the 15:07Z shift)" in capsys.readouterr().out


@pytest.mark.parametrize("args, why", [
    ({"until": "2026-10-04T10:30Z"}, "an absolute UTC time, %Y-%m-%dT%H:%M:%SZ"),
    ({"until": "2026-10-03T19:40:30Z"}, "--until 2026-10-03T19:40:30Z has passed"),
    ({"by": " "}, "--by: the handle who decided it"),
    ({"reason": ""}, "--reason: why"),
    ({"key": f"{N2}.kueue.provers.gpu.nominal"}, "no reader for 'kueue' on vy-nebius-2"),
    ({"key": "vy-nebius-1.direct-gpus", "value": "none"}, "unknown node 'vy-nebius-1'"),
    ({"key": f"{N2}.check-slots.d", "value": "128-15x"}, "bad CPU list item"),
    ({"key": f"{N2}.fill.window.2026-10-04T06:30Z", "value": "0"}, "'0' is not a number of minutes"),
])
def test_an_override_is_refused_naming_each_problem(capsys, tmp_path, args, why):
    log = tmp_path / "o.jsonl"
    rc, out = override(capsys, log, **args)
    assert rc == 2 and why in out and not log.exists()


def test_an_override_may_add_a_key_and_a_later_one_replaces_it_until_its_own_until(capsys, tmp_path):
    log = tmp_path / "o.jsonl"
    window = f"{N2}.fill.window.2026-10-04T06:30Z"
    assert override(capsys, log, key=window, value="190", until="2026-10-04T09:40:00Z", reason="circuits' 235B")[0] == 0
    assert override(capsys, log, key=window, value="150", until="2026-10-03T20:00:00Z", reason="shortened")[0] == 0
    keys = policy.load(NEBIUS)
    eff = policy.effective(keys, policy.overrides(log), T)
    assert (eff[window].value, eff[window].owner, eff[window].override["was"]) == (150, "infra", None)
    assert window not in policy.effective(keys, policy.overrides(log), epoch("2026-10-03T20:00:00Z"))
    assert policy.effective(keys, policy.overrides(log), T, "vy-nebius-1") == keys


def test_a_log_that_doesnt_verify_is_refused_never_read_as_no_override(capsys, tmp_path):
    log = tmp_path / "o.jsonl"
    assert override(capsys, log)[0] == 0
    log.write_text(log.read_text().replace("0-31", "0-30"))
    assert policy.main_show(NEBIUS, "all", False, log=log) == 3
    assert "hash does not match its content" in capsys.readouterr().out
    rc, out = override(capsys, log)
    assert rc == 3 and "the log doesn't verify" in out
    rc, rows = diff(capsys, "nebius-2", n2=f"status {STATUS}\n" + dump({**N2_FILES, policy.OVERRIDES: log.read_text()}))
    assert rc == 3 and rows[SLOT_A]["status"] == "unknown" and "its override log doesn't verify" in rows[SLOT_A]["why"]


def test_diff_compares_live_with_the_override_in_force(capsys, tmp_path, monkeypatch):
    log = tmp_path / "n2.jsonl"
    assert override(capsys, log)[0] == 0
    assert override(capsys, log, key=f"{N2}.user-cpuset.delegate", value="cpu", until="2026-10-03T19:40:31Z")[0] == 0
    assert override(capsys, log, key=f"{N2}.fill.window.2026-10-04T06:30Z", value="190", until="2026-10-04T09:40:00Z")[0] == 0
    monkeypatch.setattr(policy, "clock", lambda: T + 10)                          # user-cpuset.delegate's has ended
    files = {**N2_FILES, policy.CHECK_SLOTS: f"a {SLOT_A_NOW}\nb 64-119 windows=/workspace/pouw/fill/windows lead=90 mem=560\n",
             policy.OVERRIDES: log.read_text()}
    window = f"{N2}.fill.window.2026-10-04T06:30Z"
    rc, rows = diff(capsys, n2=f"status {STATUS}\n" + dump(files))
    assert rc == 1
    assert {k: r["status"] for k, r in rows.items() if r["status"] != "ok" or "override" in r} == {SLOT_A: "ok", window: "drift"}
    assert rows[SLOT_A]["declared"] == SLOT_A_NOW and rows[SLOT_A]["override"]["was"] == SLOT_A_WAS
    assert rows[f"{N2}.user-cpuset.delegate"]["declared"] == "cpu cpuset memory pids"
    assert rows[window]["why"] == "absent live"


@pytest.mark.parametrize("name", sorted(policy.COMMANDS))
def test_every_command_only_reads(name):
    cmd = policy.COMMANDS[name]
    assert ">" not in cmd.replace("2>/dev/null", "")
    words = set(re.findall(r"[a-z][a-z-]*", cmd))
    assert not words & {"rm", "mv", "cp", "dd", "tee", "touch", "mkdir", "chmod", "chown", "sed", "kill", "pkill", "systemctl",
                        "apply", "patch", "edit", "delete", "label", "annotate", "replace", "scale", "send-keys", "nvidia-smi",
                        "du", "find", "install"}


def _sh(cmd: str, root: Path, path: str = "", tops: tuple[str, ...] = ("workspace", "etc", "var", "proc")) -> str:
    """Run one of `COMMANDS` with its absolute paths under `root`, rewritten in one pass: `root` may itself lie under one of
    `tops` (a pod's tmp is under /workspace/), and a second pass would prefix it again."""
    cmd = re.sub(rf"/(?:{'|'.join(tops)})/", lambda m: f"{root}{m[0]}", cmd)
    env = {**os.environ, "PATH": f"{path}:{os.environ['PATH']}" if path else os.environ["PATH"]}
    return subprocess.run(["sh", "-c", cmd], capture_output=True, text=True, check=True, env=env).stdout.replace(str(root), "")


N2_TOPS = ("workspace", "etc", "var", "proc", "run")


def _put(root: Path, files: dict[str, str]) -> None:
    for p, text in files.items():
        f = root / p.lstrip("/")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)


def _n2(tmp_path: Path, status: str, windows: str | None = None, cmd: str | None = None, pgrep: int = 1) -> str:
    """Run the node 2 command with node 2's clock at T: a `date` on PATH that answers `date +%s` and passes the rest on, and a
    `pgrep` that exits `pgrep` (1: no fill runner runs). /run is under tmp_path too."""
    _put(tmp_path, {f"{policy.FILL}/status.txt": status + "\n  running x\n", f"{policy.FILL}/cpu-sets": "0 0-7\n",
                    "bin/date": f'#!/bin/sh\n[ "$*" = +%s ] && {{ echo {T}; exit 0; }}\nexec {shutil.which("date")} "$@"\n',
                    "bin/pgrep": f'#!/bin/sh\necho "$*" >> {tmp_path}/pgrep.log\nexit {pgrep}\n'})
    for f in ("date", "pgrep"):
        (tmp_path / "bin" / f).chmod(0o755)
    w = tmp_path / policy.FILL.lstrip("/") / "windows"
    if windows is None:
        w.unlink(missing_ok=True)
    else:
        w.write_text(windows)
    return _sh(cmd or policy.COMMANDS["n2"], tmp_path, str(tmp_path / "bin"), N2_TOPS)


@pytest.mark.parametrize("status, windows, why", [
    (STATUS.replace("timed False", "timed True"), None, "a timed window is on or waiting"),
    (STATUS.replace("window waiting False", "window waiting True"), "2026-10-03T19:41Z 20\n", "a timed window is on or waiting"),
    (STATUS.replace("19:39:59Z", "19:39:29Z"), None, "status.txt is 61 s old: is the fill runner running?"),
    (STATUS.replace("19:39:59Z", "19:30:00Z"), None, "status.txt is 630 s old: is the fill runner running?"),
    (STATUS.replace("19:39:59Z", "17:00:00Z"), "2026-10-03T19:30Z 30\n", "window 2026-10-03T19:30Z (30 min) is on or starts "
                                                                          "within 1800 s"),
    (STATUS.replace("19:39:59Z", "19:40:31Z"), None, "status.txt is stamped 1 s ahead of node 2's clock"),
    (STATUS.split(" ", 1)[1], None, "status.txt does not say when it was written"),
    (STATUS, "2026-10-03T19:40Z 20\n", "window 2026-10-03T19:40Z (20 min) is on or starts within 60 s"),
    (STATUS, "# start UTC, minutes\n2026-10-03T19:41Z 10 # in 30 s\n", "window 2026-10-03T19:41Z (10 min) is on"),
    (STATUS, "2026-10-03T19:15Z # 30 minutes when unsaid\n", "window 2026-10-03T19:15Z (30 min) is on"),
    (STATUS, "2026-10-03T19:00Z 40.6", "window 2026-10-03T19:00Z (40.6 min) is on"),
    (STATUS, "garbage 5\n2026-10-04T06:30Z 190\n", "windows line 'garbage 5' does not parse"),
    (STATUS, "2026-10-03T19:00Z 1e3\n", "windows line '2026-10-03T19:00Z 1e3' does not parse"),
])
def test_node_2_s_gate_refuses_before_reading_anything_else_unless_quiet_fresh_and_clear_of_windows(tmp_path, status, windows,
                                                                                                    why):
    out = _n2(tmp_path, status, windows)
    assert out.startswith(f"refused {why}") and out.endswith(f": {status}\n") and out.count("\n") == 1


@pytest.mark.parametrize("windows", [
    None, "", "# start UTC, minutes\n\n2026-10-03T18:25Z 10 # merge checks off\n2026-10-03T19:42Z 10\n",
    "2026-10-03T19:00Z 40.4", "2026-10-03T19:41:31Z 1 # starts in 61 s\n"])
def test_node_2_s_gate_reads_the_files_while_quiet_fresh_and_clear_of_windows(tmp_path, windows):
    out = _n2(tmp_path, STATUS, windows)
    files = policy._files(out)
    assert out.startswith(f"status {STATUS}\n") and files[f"{policy.FILL}/cpu-sets"] == "0 0-7\n\n"
    assert files[f"{policy.FILL}/windows"] == (None if windows is None else windows + "\n")


STALE = STATUS.replace("19:39:59Z", "17:00:00Z")   # 9630 s old: past STATUS_STALE_S
STALE_WHY = ("no fill runner runs (status.txt is 9630 s old), and gpu-lease shows no timed lease, no holder or waiter of every "
             "GPU and no booked window within 1800 s, so it is read as quiet")
GL = "/run/gpu-lease"


def _leases(tmp_path: Path, owners: dict[int, str], waits: dict[str, str] | None = None, live: tuple[int, ...] = (),
            gpus: int = 8):
    """gpu-lease's directory under tmp_path: `gpus` locks, owner lines and wait files, and /proc entries for the `live` pids."""
    _put(tmp_path, {**{f"{GL}/{i}.lock": "" for i in range(gpus)}, **{f"{GL}/{i}.owner": o + "\n" for i, o in owners.items()},
                    **{f"{GL}/{name}": w + "\n" for name, w in (waits or {}).items()}})
    for p in live:
        (tmp_path / "proc" / str(p)).mkdir(parents=True, exist_ok=True)


def test_node_2_with_no_fill_runner_is_read_as_quiet_from_gpu_lease_s_state_and_says_so(tmp_path):
    _leases(tmp_path, {1: "pid=11 run=- who=a since=x preempt=1 cmd=job", 3: "pid=13 run=- who=b since=x cmd=job",
                       5: "pid=99 run=- who=c since=x timed=1 cmd=left behind"},
            {"wait.ns.21": "pid=21 n=2 run=- who=d since=x", "wait.ns.98": "pid=98 n=8 run=- who=e since=x"}, live=(11, 13, 21))
    out = _n2(tmp_path, STALE, "2026-10-03T20:11Z 30 # 30.5 min ahead\n2026-10-03T18:00Z 30 # ended\n")
    assert out.startswith(f"stale {STALE_WHY}: {STALE}\n")
    assert policy._files(out)[f"{policy.FILL}/cpu-sets"] == "0 0-7\n\n", "the files are read after the gate"
    assert (tmp_path / "pgrep.log").read_text() == f"-f {policy.RUNNER_RE}\n"
    lv = policy.Live({}, {})
    policy._read_n2(N2, out, T, lv)
    assert lv.quiet == f"{STALE_WHY}: {STALE}" and N2 not in lv.unknown
    assert [k for k in lv.values if ".fill.window." in k] == [f"{N2}.fill.window.2026-10-03T20:11Z"]


@pytest.mark.parametrize("owners, waits, live, pgrep, windows, why", [
    ({}, {}, (), 0, None, "status.txt is 9630 s old while a fill runner runs: it may be stuck"),
    ({}, {}, (), 2, None, "pgrep exited 2: whether a fill runner runs is not known"),
    ({}, {}, (), 1, "2026-10-03T20:10Z 30\n", "window 2026-10-03T20:10Z (30 min) is on or starts within 1800 s"),
    ({}, {}, (), 1, "2026-10-03T19:00Z 45\n", "window 2026-10-03T19:00Z (45 min) is on or starts within 1800 s"),
    ({4: "pid=14 run=- who=t since=x timed=1 cmd=bench"}, {}, (14,), 1, None,
     "lease 4.owner of pid 14 is timed (gpu-lease --timed)"),
    ({i: "pid=7 run=- who=w since=x cmd=all" for i in range(8)}, {}, (7,), 1, None, "pid 7 holds all 8 GPUs"),
    ({}, {"wait.ns.31": "pid=31 n=8 run=- who=w since=x"}, (31,), 1, None, "waiter wait.ns.31 wants all 8 GPUs"),
    ({}, {"wait.ns.31": "pid=31 run=- who=w since=x"}, (31,), 1, None, "waiter wait.ns.31 does not say how many GPUs it wants"),
    ({2: "who=x since=y"}, {}, (), 1, None, f"gpu-lease owner line {GL}/2.owner names no pid"),
])
def test_node_2_with_no_fill_runner_still_refuses_a_timed_window_on_or_near(tmp_path, owners, waits, live, pgrep, windows, why):
    _leases(tmp_path, owners, waits, live)
    out = _n2(tmp_path, STALE, windows, pgrep=pgrep)
    assert out == f"refused {why}: {STALE}\n"


def test_a_write_on_node_2_needs_the_fill_runner_s_fresh_quiet_status(tmp_path):
    _leases(tmp_path, {})
    out = _n2(tmp_path, STALE, cmd=policy.N2_WRITE_GATE + "echo wrote\n")
    assert out == ("refused status.txt is 9630 s old: is the fill runner running? (a write needs its fresh quiet status): "
                   f"{STALE}\n")
    assert not (tmp_path / "pgrep.log").exists()
    assert _n2(tmp_path, STATUS, cmd=policy.N2_WRITE_GATE + "echo wrote\n") == f"status {STATUS}\nwrote\n"


@pytest.mark.parametrize("cmdline, runner", [
    ("python3 /workspace/pouw/infra/bin/fill_runner.py", True), ("/usr/bin/python3 /workspace/pouw/infra/bin/fill_runner.py", True),
    ("/usr/bin/python3 -u fill_runner.py fence x", True), ("python3.12 fill_runner.py", True),
    ("bash -c set -f\npgrep -f fill_runner.py", False), ("tail -f /workspace/pouw/fill/fill_runner.py.log", False),
    ("python3 /x/not_fill_runner.pyc", False)])
def test_the_runner_pattern_matches_the_fill_runner_and_never_the_gate_s_own_shell(cmdline, runner):
    assert bool(re.search(policy.RUNNER_RE, cmdline.replace("\n", " "))) is runner


def test_diff_says_when_node_2_was_read_as_quiet(capsys):
    n2 = f"stale {STALE_WHY}: {STALE}\n" + dump(N2_FILES)
    rc, rows = diff(capsys, "nebius-2", n2=n2)
    assert rc == 0 and {r["status"] for r in rows.values()} == {"ok"}
    policy.main_diff(NEBIUS, "nebius-2", True, runner=fake(outputs(n2=n2), None))
    assert json.loads(capsys.readouterr().out)["nodes"][0]["quiet"] == f"{STALE_WHY}: {STALE}"
    policy.main_diff(NEBIUS, "nebius-2", False, runner=fake(outputs(n2=n2), None))
    assert capsys.readouterr().out.splitlines()[:2] == [f"{N2}: 4 as declared, 0 drift, 0 undeclared, 0 unknown",
                                                        f"  read as quiet: {STALE_WHY}: {STALE}"]


def test_node_2_s_gate_fails_when_status_txt_cant_be_read(tmp_path):
    _n2(tmp_path, STATUS)
    (tmp_path / policy.FILL.lstrip("/") / "status.txt").unlink()
    with pytest.raises(subprocess.CalledProcessError) as e:
        _sh(policy.COMMANDS["n2"], tmp_path, str(tmp_path / "bin"), N2_TOPS)
    assert e.value.returncode == 1 and e.value.stdout == ""


def test_a_node_whose_address_the_environment_doesnt_hold_is_unknown_naming_the_variable(capsys, monkeypatch):
    """No file holds a node's address (`status.reach`): with its variable unset every key of the node is unknown, exit 3,
    and its why names the variable; nothing is drift and nothing is sent."""
    monkeypatch.delenv("VY_NEBIUS_2_HOST", raising=False)
    monkeypatch.setattr(policy.status.subprocess, "run", lambda *a, **kw: pytest.fail("a command was sent"))
    assert policy.main_diff(NEBIUS, "nebius-2", True) == 3
    rows = [r for rep in json.loads(capsys.readouterr().out)["nodes"] for r in rep["keys"]]
    assert rows and {r["status"] for r in rows} == {"unknown"}
    assert all("VY_NEBIUS_2_HOST is unset: vy-nebius-2's address" in r["why"] for r in rows)
