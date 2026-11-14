"""research's settings registry: every environment variable research, and the infra code built on it, reads outside the
files `tests/lint/allowlists/repo_env_reads.json` still lists, declared once in `DECLARED`, and the one reader of them.

`get`, `need` and `has` are ``os.environ.get``, ``os.environ[...]`` and ``in os.environ`` for a declared name, read at the
call: a default, an empty string and an unset variable behave exactly as they do there (`need` raises KeyError). A name
`DECLARED` lacks raises `Undeclared`, which no ``except KeyError`` catches, and `undeclared` finds such a name in a module's
source, which each suite holds its readers to. A variable whose name the caller takes from data (a config's ``*_env`` field,
a ``--profile-key``, the secrets a VM injects) can't be declared: `named` reads it.

Stdlib only: research has no dependencies, and runs on machines that have nothing else.
"""
from __future__ import annotations

import os
from typing import TypeVar

T = TypeVar("T")

DECLARED: dict[str, str] = {
    # the process and the tools around it
    "PATH": "the directories a program is looked up in",
    "PYTHONPATH": "Python's import roots",
    "USER": "the login name, where no lane is named",
    "XDG_CACHE_HOME": "the user's cache directory (default ~/.cache)",
    "XDG_RUNTIME_DIR": "the user's runtime directory, which `systemctl --user` needs",
    "XDG_STATE_HOME": "the user's state directory (default ~/.local/state)",
    "PYTEST_CURRENT_TEST": "set by pytest inside a test",
    "GIT_CONFIG_COUNT": "how many GIT_CONFIG_KEY_n/GIT_CONFIG_VALUE_n pairs git takes from the environment",
    "CUDA_VISIBLE_DEVICES": "the GPUs CUDA shows the process",
    "TRAIN_TREES": "where a lander's worktrees go instead of TMPDIR (`train_trees`)",
    # who runs it
    "RESEARCH_LANE": "the lane a command acts for",
    "RESEARCH_CAMPAIGN": "the campaign a run belongs to",
    "RESEARCH_PROJECT": "the project a machine is bound to",
    "RESEARCH_AGENT": "the agent that registers a machine (default the hostname)",
    "RESEARCH_AGENT_URL": "the URL of the agent running the command, recorded as who acted",
    "CLOUD_AGENT_INJECTED_SECRET_NAMES": "the secrets Cursor injects into this VM, by name, comma-separated",
    # credentials, read by name and never printed
    "SLACK_BOT_TOKEN": "the Slack bot's token",
    "CURSOR_API_KEY": "the Cursor API key the agent watch reads agents with",
    "RUNPOD_API_KEY": "RunPod's API key",
    "RUNPOD_SSH_KEY_B64": "the RunPod ssh key, base64",
    "NEBIUS_SA_ID": "the Nebius service account's id",
    "NEBIUS_SA_PUBLIC_KEY_ID": "the Nebius service account's public key id",
    "NEBIUS_SA_PRIVATE_KEY_B64": "the Nebius service account's private key, base64",
    "R2_ENDPOINT": "the evidence store's R2 endpoint",
    "R2_BUCKET": "the evidence store's R2 bucket",
    "AWS_SESSION_TOKEN": "the S3 session token, where a store config names no variable for it",
    "STORE_INDEXER_DATABASE_URL": "the Postgres URL the store's index is written to",
    "STORE_PRIVATE_READER_URL": "store_private_reader's URL",
    "AGENT_WATCH_PG_URL": "the Postgres URL the agent watch records its passes in",
    "VY_MONITORS_SLACK_WEBHOOK": "the Slack webhook node monitors post to",
    "JOBS_URL": "the jobs service's URL",
    # the notes
    "RESEARCH_NOTES": "the notes clone (default ~/.research/notes)",
    "RESEARCH_NOTES_SYNC": "1/true/yes/on: commands that write a note sync it; 0/false/no/off: registry writes don't",
    "RESEARCH_NOTES_TOKEN": "the token the notes are pushed with",
    "RESEARCH_NOTES_TOKEN_SOURCE": "where RESEARCH_NOTES_TOKEN came from",
    "RESEARCH_NOTES_TOKEN_EXPIRES": "when RESEARCH_NOTES_TOKEN expires",
    "RESEARCH_NOTES_CITE_CHECK": "0/off/no/false: a note's citations aren't checked",
    "RESEARCH_RELAUNCH": "where relaunch briefs are kept (default ~/.research/relaunch)",
    "RESEARCH_MACHINES_D": "the machine registry's directory, in place of the notes' machines.d",
    # runs and the machines they run on
    "RESEARCH_REPO": "the repository research acts on, in place of the working directory's",
    "RESEARCH_CHECKOUT": "the checkout `research` compares with origin/main before it runs main's copy (default .)",
    "RESEARCH_AT_CHECKOUT": "set: `research` runs the checkout's own copy, never origin/main's",
    "RESEARCH_RUNS": "the local run registry's directory",
    "RESEARCH_MACHINES": "the machines file (default ~/.research/machines.toml)",
    "RESEARCH_SHIP": "stream: a run's source ships as an archive stream instead of through git",
    "RESEARCH_BIN_DIRS": "the directories binprov looks for recorded binaries in, os.pathsep-separated",
    "RESEARCH_TOOL_SNAPSHOT": "the shipped snapshot research runs from, where it isn't a git checkout",
    "RESEARCH_DIGEST_STATE": "the digest's state file (default ~/.research/digest/LANE.json)",
    "RESEARCH_DISK_HOLD_PCT": "the disk use, in percent, a queued run is held at",
    "RESEARCH_CGROUP_DIR": "the cgroup memory admission reads, in place of the process's own",
    "RESEARCH_PROC_ROOT": "/proc, for a test's stand-in",
    "RESEARCH_RAMLOCK_DIR": "the directory of live memory reservations",
    "RESEARCH_CPU_CLAIMS_DIR": "the directory of `cpu-hold`'s core claims",
    "RESEARCH_CPU_ROOT": "the CPU topology `cpu-hold` reads, in place of /sys/devices/system/cpu",
    "NODE_SLOT_ONLY_FILE": "the node's deployed slot-only rule, in place of its installed path",
    "NODE_TOOL_RULES_FILE": "the node's deployed tool rules, in place of their installed path",
    "RESEARCH_SEED_RULES": "a seed rules file the telemetry tests load",
    "AUDITOR_FILE": "the file that marks a machine as an auditor's",
    "RUNS_VIA_SLOT_FILE": "the file that marks a machine whose runs take a check slot",
    "DIRECT_CPUS_FILE": "the file naming the CPUs a direct run is pinned to",
    "GPU_LEASE_ALLOWED_FILE": "the file saying whether runs may lease GPUs here",
    "GPU_LEASE_HOST_FILE": "the file that marks a machine that hosts gpu-lease",
    # the evidence store
    "RESEARCH_STORE": "the evidence store's local directory (default ~/.research/store)",
    "RESEARCH_STORE_CONFIG": "the store's remote config (default ~/.research/store.toml)",
    "RESEARCH_WRITE_THROUGH": "whether a store write goes to the remote too",
    "RESEARCH_PUBLIC_URL": "the store's public URL fixtures are fetched from",
    "RESEARCH_CATALOG_IGNORE_VERSION": "set: a catalog newer than this code is opened anyway",
    "VERITY_RESEARCH_POSTGRES": "set: the store tests that need Postgres run",
    # telemetry
    "TELE_ATTEMPT_DIR": "the attempt directory the runner exports to its command",
    "TELE_PARENT_SPAN": "the span a command's events hang under",
    "TELEMETRY_PROC": "the process name a command's events carry (default shell)",
    "TELEMETRY_STREAM_SUFFIX": "the suffix of an attempt's event and sample streams",
    "TELEMETRY_CANCEL_FALLBACK": "what cancel does when a run's scope can't be stopped",
    "TELEMETRY_LANE": "the lane a telemetry run records",
    "TELEMETRY_PROJECT": "the project a telemetry run records",
    "RUNPOD_POD_ID": "the RunPod pod this process runs on",
    "POD_ID": "the pod this process runs on, where RunPod doesn't say",
    # pods, guards and leases
    "RESEARCH_GUARD_HOST": "the machine the budget guard runs on",
    "RESEARCH_GUARD_STATE": "the budget guard's state file",
    "RESEARCH_GUARD_DIR": "the pod guard's directory (default ~/.research/pods)",
    "RESEARCH_GUARD_SUPERVISED": "set: the pod guard runs under its supervisor",
    "RESEARCH_DRAIN_LOG": "the drain log (default ~/.research/pods/drain.log)",
    "RESEARCH_LEASE_EXPIRES": "when this process's machine lease expires",
    "LEASE_DIR": "the machine lease's directory",
    "VY_N2": "node 2's ssh host",
    "WAKE_ENV": "the control pod's env file a wake reads the Nebius key from",
    "WAKE_GUARD_STATE": "the budget guard's state a wake reads",
    "WAKE_MACHINES_D": "the machine registry a wake reads",
    "NEBIUS_BIN": "the Nebius CLI a wake runs",
    # read by the node scripts that run alone, and by their tests
    "VY_BUNDLE_SIZES": "the pacer's learned bundle sizes",
    "VY_PROVER_CPUS": "the CPUs node 1's prover runs on",
    # infra's control pod
    "AGENT_WATCH_OFFSET": "the agent watch relay's offset file (default ~/agent-watch/offset)",
    "AGENT_WATCH_ON_POD": "set: the relay runs on the guard host itself, so it reads without ssh",
    "CONTROL_DISK_SEEN": "the control disk alert's state file (default ~/control-disk/seen.json)",
    "CONTROL_POD_LOCAL": "1: the control disk alert runs on the control pod itself",
    "VM_STOP_KEY_ENV": "the env file of the Nebius key VM stops use",
    "VM_STOP_SSH_KEY": "the ssh key VM stops read VMs with",
    "VM_STOP_MACHINES_D": "the machine registry VM stops read",
}


class Undeclared(Exception):
    """A read of a variable `DECLARED` lacks."""


def _declared(name: str) -> str:
    if name not in DECLARED:
        raise Undeclared(f"{name} is not declared in tools/research/src/research/settings.py (DECLARED): declare it there "
                         f"with what it is, or read a name taken from data with `named`")
    return name


def get(name: str, default: T = None) -> str | T:
    """``os.environ.get(name, default)`` of a declared variable: an empty string is returned as it is."""
    return os.environ.get(_declared(name), default)


def need(name: str) -> str:
    """``os.environ[name]`` of a declared variable: KeyError when it is unset."""
    return os.environ[_declared(name)]


def has(name: str) -> bool:
    """``name in os.environ`` of a declared variable: set, even to an empty string."""
    return _declared(name) in os.environ


def named(name: str, default: T = None) -> str | T:
    """``os.environ.get(name, default)`` of a variable the caller's data names, which no declaration can list."""
    return os.environ.get(name, default)


def undeclared(source: str) -> list[tuple[int, str]]:
    """(line, name) of each read in ``source`` through this module (imported as ``settings``, or by another name) whose
    name, a literal or a module-level constant, `DECLARED` lacks."""
    import ast

    tree = ast.parse(source)
    aliases = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               and (n.module == "research" or n.module is None and n.level > 0) for a in n.names if a.name == "settings"}
    consts: dict[str, str] = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            t, v = n.targets[0], n.value
            pairs = zip(t.elts, v.elts) if isinstance(t, ast.Tuple) and isinstance(v, ast.Tuple) else [(t, v)]
            consts.update({a.id: b.value for a, b in pairs if isinstance(a, ast.Name) and isinstance(b, ast.Constant)
                           and isinstance(b.value, str)})
    out = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("get", "need", "has")
                and isinstance(n.func.value, ast.Name) and n.func.value.id in aliases and n.args):
            a = n.args[0]
            name = a.value if isinstance(a, ast.Constant) else consts.get(a.id) if isinstance(a, ast.Name) else None
            if isinstance(name, str) and name not in DECLARED:
                out.append((n.lineno, name))
    return sorted(out)


def train_trees() -> str | None:
    """TRAIN_TREES: where a lander's worktrees go instead of TMPDIR (train.sh's, drive's auto plans', the after-landing
    command's); the control pod's `infra/control/drive.sh` and the Nebius loop's
    `tools/research/src/research/pods/nebius/vqueue/loop.sh` always set it, to a directory on a local disk whose old trees
    they sweep.  Unset or empty, None."""
    return get("TRAIN_TREES") or None
