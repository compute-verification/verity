"""The security review's serious findings still open, as a number: the findings of `findings.toml` beside this file that no
check this script runs closes, measured as `infra/security-findings-open`. Lower is better; the goal is met at zero.

    research run --on auto --cwd source -- uv run python benchmarks/security_findings/open_findings.py

Only a check closes a finding: the one `CHECKS` holds for its id, which the list must name, run on the checkout at the commit
it is at. A finding with no check is open whatever the list says, and the list says in one line what check would close it.
A check that can't read its thing where it runs (an agent VM's environment, another machine, a reading older than
`MAX_AGE_DAYS`) says so: the finding is `unread`, which the value counts as not closed and the output names apart from open.
The value's mark is `measured` only when every finding is decided by a check of the thing itself (`tree`), and `partial`
while any has no check, is unread, or is checked on the tree's declared state (`declared`), on machines partly through a
recorded reading (`machine`) or on a recorded reading alone (`recorded`). It reads the machine it runs on, and on a CPU
machine the others its `PEERS` lists, over ssh as the agents' login with the cluster's key that machine holds; it needs no
secret of its own, changes nothing, and writes under `--out` (`out/` outside a research run). `--probe-store`, run on an
agent VM, takes the store's reading.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shlex
import socket
import subprocess
import time
import tomllib
import urllib.error
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from research import addresses

from verity_bench import kit, settings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LIST = HERE / "findings.toml"
#: what the checks of things outside the tree read: the readings taken where the thing is, the maintained kernels, the machines
READINGS = HERE / "readings.toml"
KERNELS = HERE / "kernels.toml"
DEPLOY = ROOT / "tools/research/src/research/pods/nebius/deploy.toml"
METRIC = "infra/security-findings-open"
#: what a check reads: this repository's own files, the thing the finding is about; the tree's declared state of a VM; each
#: machine, the one it runs on and those its `PEERS` lists live and the others through a recorded reading; or a recorded
#: reading alone
READS = ("tree", "declared", "machine", "recorded")
#: how many days a recorded reading decides anything: an older one leaves its finding unread, and the check names the command
MAX_AGE_DAYS = 7
#: the account agents log in with on the CPU machines, and the control pod, which the root finding names with them
AGENTS_LOGIN = "research"
CONTROL_POD = "vy-control-verity"
#: the names an agent VM must not hold: a personal Linear key and notes token, and the Google Cloud, RunPod and Nebius keys
AGENT_KEYS = ("LINEAR_API_KEY", "RESEARCH_NOTES_TOKEN", "GCP_SA_KEY_B64", "RUNPOD_API_KEY", "NEBIUS_SA_PRIVATE_KEY_B64")
AGENT_ENV_COMMAND = "compgen -e | grep -xE '" + "|".join(AGENT_KEYS) + "'"
#: one line each: the login, its uid, what `sudo -n id -u` gives it, whether it runs with no new privileges (where sudo refuses
#: everyone), the kernel, the Ubuntu release, then apt's effective unattended-upgrades lines
MACHINE_COMMAND = ("id -un; id -u; sudo -n id -u 2>/dev/null || echo refused; grep -m1 NoNewPrivs /proc/self/status | cut -f2; "
                   "uname -r; (. /etc/os-release && echo \"$VERSION_ID\") || echo unknown; apt-config dump | grep -E "
                   "\"^(APT::Periodic::Unattended-Upgrade|Unattended-Upgrade::(Allowed-Origins|Origins-Pattern|Package-Whitelist))\"")
#: each CPU machine's list of the CPU machines, itself among them: a line a machine, its name, then ssh's options and
#: destination, by which its agents' login reaches that machine with the cluster's key (`research deploy` writes it)
PEERS = Path("/etc/vy/peers")
#: ssh's options ahead of a line's own, which they override: no prompt, a bounded connect, and only a host key already known
PEER_SSH = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes")
STORE_COMMAND = "uv run python benchmarks/security_findings/open_findings.py --probe-store"
#: where the store probe copies to, from a key under it that doesn't exist
SCRATCH = "scratch/security-findings"
clock = time.time
#: why an address the rule counts is excepted from finding 6: nobody's machine of ours
EXCEPT_KINDS = ("version", "example", "third party")
#: what a [[finding]] may hold, and nothing else
KEYS = {"id", "statement", "check", "reads", "how", "would_close", "except"}
HOOKS = ".cursor/hooks.json"
#: the one place the guard's cure runs it from, outside every checkout
GUARD_ROOT = "/opt/agent-guard/"
_GUARD_FILE = re.compile(re.escape(GUARD_ROOT) + r"[A-Za-z0-9_-][A-Za-z0-9_.-]*")
#: a word with nothing the shell expands, quotes, joins or redirects, and no path
_PLAIN = re.compile(r"[A-Za-z0-9_.,:=+%@-]+")
#: the one wrapper a hook may run the guard through instead (`/bin/sh -c GUARD_WRAPPER guard NAME`), held here word for word
#: so that a change to its text reopens the finding until this check vouches for the new one. It fixes PATH to root's
#: directories and unsets what could make a program it starts load another file (PYTHON*, GIT_*, BASH_ENV, LD_*, the git
#: guard's switches), names every shell by full path, then runs `GUARD_ROOT`'s hook.sh
#: when it and NAME are readable, and no file of the checkout's working tree otherwise: on a Cursor cloud VM (/exec-daemon)
#: without them it refuses every call but the one shell command that restores the copy, and on any other machine (a desktop
#: agent's laptop) it runs origin/main's committed hook.sh out of git, which checking out a branch doesn't change.
GUARD_WRAPPER = (
    'PATH=/usr/local/bin:/usr/bin:/bin; export PATH; '
    'v=$(awk "BEGIN{for(k in ENVIRON)if(k~/^(PYTHON|GIT_)[A-Za-z0-9_]*$/)print k}") || exit 2; '
    'unset ENV BASH_ENV CDPATH LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT VERITY_REAL_GIT VERITY_GIT_GUARD '
    'AGENT_GUARD_GIT_HOOKS AGENT_GUARD_CHAIN_HOOKS $v; '
    'g=/opt/agent-guard; [ -r $g/hook.sh ] && [ -r $g/$1 ] && exec /bin/sh $g/hook.sh "$1"; '
    '[ -d /exec-daemon ] || { h=$(git show origin/main:tools/agent-guard/hook.sh 2>/dev/null) '
    '&& exec /bin/sh -c "$h" origin/main "$1"; }; '
    'r="git -C /workspace fetch -q origin main && git -C /workspace show origin/main:tools/agent-guard/boot.sh '
    '| bash -s -- /workspace"; e=$(cat); '
    'case $1:$e in sh*:*\\"command\\"*\\"command\\"*) ;; '
    'sh*:*"\\"command\\":\\"$r\\""*|sh*:*"\\"command\\": \\"$r\\""*) '
    'echo "{\\"permission\\":\\"allow\\"}"; exit;; esac; '
    'm="Refused by .cursor/hooks.json: this machine has no readable $g/hook.sh and $g/$1, the copy of the agent guard '
    'from origin/main that boot.sh puts there at boot, and the guard never runs the copy in a checkout. Until it is back '
    'every shell command but the one below and every Task and MCP call is refused, while reading and editing files still '
    'work. Restore it with exactly this command: $r"; '
    'echo "{\\"permission\\":\\"deny\\",\\"agent_message\\":\\"$m\\",\\"user_message\\":\\"$m\\"}"')
_HOOK_NAME = re.compile(r"[a-z_]+\.py")


@dataclass(frozen=True)
class Check:
    name: str
    reads: str
    #: (closed, what it found in words) for the checkout at the root it is given and the finding's entry in the list; closed is
    #: None when the check can't read its thing where it runs
    run: Callable[[Path, Mapping], tuple[bool | None, str]]


@dataclass(frozen=True)
class Status:
    id: str
    #: not closed: open, refused or unread
    open: bool
    #: what decided it: a check's `reads`, `none` (no check), `refused` (an entry the script can't take, which counts open) or
    #: `unread` (its check can't read the thing where it ran)
    basis: str
    why: str


def excepted(finding: Mapping) -> dict[str, int]:
    """{path: how many of its public addresses are excepted} from the finding's [[finding.except]] entries, each a path, a
    count, a kind of `EXCEPT_KINDS` and a reason."""
    out: dict[str, int] = {}
    for e in finding.get("except", []):
        if not (isinstance(e.get("path"), str) and isinstance(e.get("count"), int) and e["count"] > 0
                and e.get("kind") in EXCEPT_KINDS and isinstance(e.get("reason"), str) and e["reason"].strip()):
            raise ValueError(f"an exception is a path, a count above 0, a kind of {EXCEPT_KINDS} and a reason: {e!r}")
        if e["path"] in out:
            raise ValueError(f"{e['path']} is excepted twice")
        out[e["path"]] = e["count"]
    return out


def no_public_address(root: Path, finding: Mapping) -> tuple[bool, str]:
    allowed = excepted(finding)
    found = addresses.scan(addresses.tracked(root), addresses.reader(root))
    holding = sum(sum(kind == "ipv4" for kind, _ in hits) > allowed.get(f, 0) for f, hits in found.items())
    return holding == 0, f"{holding} tracked files hold a public IPv4 address the list doesn't except"


def outside_checkout(command: str, names: set[str]) -> bool:
    """Whether a hook's shell `command` runs the guard from outside the checkout: it splits into plain words (nothing the
    shell expands, joins or redirects), the first a file directly under `GUARD_ROOT`, and no other word is a path, `.` or names
    one of the checkout's top-level `names`; or it is `/bin/sh -c GUARD_WRAPPER guard NAME` exactly, for a hook file NAME.
    Anything else, a command that won't split included, counts as inside."""
    try:
        first, *rest = shlex.split(command)
    except ValueError:
        return False
    if first == "/bin/sh" and rest[:3] == ["-c", GUARD_WRAPPER, "guard"] and len(rest) == 4:
        return bool(_HOOK_NAME.fullmatch(rest[3])) and rest[3] not in names
    if not _GUARD_FILE.fullmatch(first):
        return False
    return all(_PLAIN.fullmatch(w) and ".." not in w and not set(re.split(r"[=:,]", w)) & (names | {"."}) for w in rest)


def guard_outside_checkout(root: Path, finding: Mapping) -> tuple[bool, str]:
    shown = subprocess.run(["git", "show", f"HEAD:{HOOKS}"], cwd=root, capture_output=True, text=True)
    if shown.returncode:
        return False, f"the commit tracks no {HOOKS}, so the tree declares no hook: a file a VM writes is not the tree's"
    hooks = json.loads(shown.stdout).get("hooks", {})
    commands = [h["command"] for entries in hooks.values() for h in entries]
    names = {p.split("/", 1)[0] for p in addresses.tracked(root)}
    inside = sum(not outside_checkout(c, names) for c in commands)
    return bool(commands) and inside == 0, (f"{inside} of {len(commands)} hook commands of {HOOKS} at the commit don't run "
                                            f"the guard from {GUARD_ROOT} alone")


def _at(root: Path, path: Path) -> Path:
    return root / path.relative_to(ROOT)


def reading(root: Path, name: str, command: str) -> tuple[dict | None, str]:
    """The recorded reading `name` of readings.toml in the checkout at `root` and how old it is, in words; (None, why) when there
    is none or it is older than `MAX_AGE_DAYS`. A reading names the command it ran, which must be `command`, who ran it
    (`reader`) and when (`at`, a time with its zone), beside what it found; one that doesn't raises."""
    path = _at(root, READINGS)
    entry = tomllib.loads(path.read_text(encoding="utf-8")).get(name) if path.is_file() else None
    if entry is None:
        return None, f"no reading is recorded: run `{command}` and record what it prints as [{name}] in {READINGS.name}"
    if entry.get("command") != command:
        raise ValueError(f"the reading [{name}] ran {entry.get('command')!r}, not `{command}`")
    at, reader = entry.get("at"), entry.get("reader")
    if not (isinstance(at, datetime) and at.tzinfo is not None and isinstance(reader, str) and reader.strip()):
        raise ValueError(f"the reading [{name}] names who read it (reader) and when, with its zone (at)")
    days = (clock() - at.timestamp()) / 86400
    if days < 0:
        raise ValueError(f"the reading [{name}] is dated after now")
    if days > MAX_AGE_DAYS:
        return None, f"the reading [{name}] is {days:.0f} days old, over {MAX_AGE_DAYS}: run `{command}` again"
    return entry, f"read by {reader} {days:.1f} days ago"


def agent_env_names(root: Path, finding: Mapping) -> tuple[bool | None, str]:
    entry, when = reading(root, "agent-env", AGENT_ENV_COMMAND)
    if entry is None:
        return None, f"an agent VM's environment can't be read where this runs, only recorded; {when}"
    names = entry.get("names")
    if not (isinstance(names, list) and all(isinstance(n, str) for n in names) and set(names) <= set(AGENT_KEYS)):
        raise ValueError("the agent-env reading's names are those of AGENT_KEYS that its command printed")
    held = ", ".join(sorted(names))
    return not names, f"an agent VM {when} held {len(names)} of the {len(AGENT_KEYS)} names" + (f": {held}" if held else "")


@dataclass(frozen=True)
class Machine:
    """What `MACHINE_COMMAND` printed on one machine, a line each."""
    login: str
    uid: str
    sudo: str
    no_new_privs: str
    kernel: str
    release: str
    apt: tuple[str, ...]

    @classmethod
    def of(cls, lines: Sequence[str]) -> Machine:
        if not (isinstance(lines, Sequence) and len(lines) >= 6 and all(isinstance(x, str) for x in lines)):
            raise ValueError(f"a machine's reading is the lines MACHINE_COMMAND prints, six and apt's: {lines!r}")
        return cls(*(x.strip() for x in lines[:6]), tuple(x.strip() for x in lines[6:]))


def cpu_machines(root: Path) -> list[str]:
    """The CPU machines of the deploy map in the checkout at `root`, but those not made yet (`staged`)."""
    nodes = tomllib.loads(_at(root, DEPLOY).read_text(encoding="utf-8"))["nodes"]
    names = [n for n, node in nodes.items() if re.fullmatch(r"vy-nebius-cpu-\d+", n) and "staged" not in node]
    return sorted(names, key=lambda n: int(n.rsplit("-", 1)[1]))


def here() -> str | None:
    """The machine this runs on: the name the research run that started it recorded (its job.json's `run.remote.machine`), else
    the host's name."""
    if not settings.is_set("RESEARCH_RUN_DIR"):
        return socket.gethostname()
    try:
        return json.loads((Path(settings.RESEARCH_RUN_DIR) / "job.json").read_text(encoding="utf-8"))["run"]["remote"]["machine"]
    except (KeyError, OSError, ValueError, TypeError):
        return socket.gethostname()


def read_here() -> list[str]:
    """`MACHINE_COMMAND`'s lines on this machine. It changes nothing; a refused sudo is logged as one."""
    return subprocess.run(["sh", "-c", MACHINE_COMMAND], capture_output=True, text=True, timeout=120).stdout.splitlines()


def peers(path: Path | None = None) -> dict[str, list[str]]:
    """{machine: ssh's options and destination for it} from `PEERS`, a line a machine; {} where this machine has none."""
    path = path or PEERS
    if not path.is_file():
        return {}
    return {f[0]: f[1:] for f in (ln.split("#")[0].split() for ln in path.read_text(encoding="utf-8").splitlines()) if len(f) >= 2}


def read_peer(ssh: Sequence[str]) -> list[str] | str:
    """`MACHINE_COMMAND`'s lines on the machine its `PEERS` line `ssh` reaches, as the login that line names; else why it read
    none: ssh couldn't reach it or doesn't already know its host key, it hung, or it printed less than a reading. It changes
    nothing."""
    try:
        r = subprocess.run(["ssh", *PEER_SSH, *ssh, MACHINE_COMMAND], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "ssh hung past 120 s"
    lines = r.stdout.splitlines()
    if r.returncode == 255:
        return "ssh can't reach it or doesn't know its host key"
    return lines if len(lines) >= 6 else "ssh printed less than a reading"


Judge = Callable[[Machine], tuple[bool | None, str]]


def over_machines(root: Path, names: Sequence[str], judge: Judge, what: str) -> tuple[bool | None, str]:
    """`judge` over each machine of `names`, on the first read of it that decides: this run's own when it runs on that machine,
    then one over ssh when this machine's `PEERS` lists it, then the recorded [machines] reading. Open when any machine is
    judged open, unread when any other is undecided or unread."""
    entry, when = reading(root, "machines", MACHINE_COMMAND)
    recorded = (entry or {}).get("output", {})
    if not isinstance(recorded, Mapping):
        raise ValueError("the machines reading's output is each machine's lines, by its name")
    me, ways = here(), peers()
    reach = [n for n in names if n in ways]
    with ThreadPoolExecutor(max_workers=max(1, len(reach))) as pool:
        remote = dict(zip(reach, pool.map(lambda n: read_peer(ways[n]), reach)))
    bad: dict[str, list[str]] = {}
    unread: dict[str, list[str]] = {}
    over_ssh = 0
    for name in names:
        closed, why = None, f"no fresh reading ({when})" if entry is None else "the reading holds none"
        if name == me:
            closed, why = judge(Machine.of(read_here()))
        if closed is None and isinstance(got := remote.get(name), list):
            closed, why = judge(Machine.of(got))
            over_ssh += closed is not None
        elif closed is None and got:
            why = f"{got} ({why})"
        if closed is None and name in recorded:
            closed, why = judge(Machine.of(recorded[name]))
        if closed is not True:
            (bad if closed is False else unread).setdefault(why, []).append(name)

    def listed(groups: dict[str, list[str]]) -> str:
        return "; ".join(f"{why}: {', '.join(ns)}" for why, ns in groups.items())
    live = f"read {me} itself" if me in names else "read none of them itself (it runs on none)"
    live += f" and {over_ssh} over ssh ({PEERS})" if over_ssh else ""
    counts = (f"{sum(map(len, bad.values()))} of {len(names)} machines {what}, {sum(map(len, unread.values()))} unread; {live}, "
              f"the rest {when if entry else 'unrecorded'}")
    detail = "; ".join(x for x in (listed(bad), f"unread: {listed(unread)}" if unread else "") if x)
    return (False if bad else None if unread else True), counts + (f" ({detail})" if detail else "")


def root_refused(m: Machine) -> tuple[bool | None, str]:
    if m.uid == "0":
        return False, f"{m.login} is root"
    if m.sudo == "0":
        return False, f"sudo gives {m.login} root"
    if m.sudo != "refused":
        return None, f"sudo printed {m.sudo!r}"
    if m.no_new_privs != "0":
        return None, "read with no new privileges, where sudo refuses every login"
    if m.login != AGENTS_LOGIN:
        return None, f"read as {m.login}, not the agents' {AGENTS_LOGIN}"
    return True, "refused"


def sudo_refused(root: Path, finding: Mapping) -> tuple[bool | None, str]:
    return over_machines(root, [*cpu_machines(root), CONTROL_POD], root_refused, "give the agents root")


def kernels(root: Path) -> dict[tuple[str, str], date]:
    """{(series, Ubuntu release): the first day it is no longer maintained} from kernels.toml in the checkout at `root`."""
    table = tomllib.loads(_at(root, KERNELS).read_text(encoding="utf-8"))
    if not (isinstance(table.get("source"), str) and table["source"].startswith("https://")):
        raise ValueError(f"{KERNELS.name} names its source")
    if not all(type(k.get("until")) is date and isinstance(k.get("series"), str) and isinstance(k.get("release"), str)
               for k in table["kernel"]):
        raise ValueError(f"each kernel of {KERNELS.name} is a series, a release and a day (until)")
    return {(k["series"], k["release"]): k["until"] for k in table["kernel"]}


def _apt(lines: Sequence[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in lines:
        if m := re.fullmatch(r'(\S+) "(.*)";', line):
            out.setdefault(m[1], []).append(m[2])
    return out


def updates_judge(maintained: Mapping[tuple[str, str], date]) -> Judge:
    def judge(m: Machine) -> tuple[bool | None, str]:
        apt, today = _apt(m.apt), datetime.fromtimestamp(clock(), UTC).date()
        series = ".".join(m.kernel.split(".")[:2])
        problems = []
        if not (until := maintained.get((series, m.release))) or until <= today:
            problems.append(f"kernel {series} on {m.release} is not maintained ({KERNELS.name})")
        if apt.get("APT::Periodic::Unattended-Upgrade", ["0"])[-1] in ("", "0"):
            problems.append("no unattended upgrades")
        origins = apt.get("Unattended-Upgrade::Allowed-Origins::", []) + apt.get("Unattended-Upgrade::Origins-Pattern::", [])
        if not any(re.search(r"(\$\{distro_codename\}|\b[a-z]+)-security\b", o) and "ESM" not in o for o in origins):
            problems.append("Ubuntu's security pocket is no origin")
        # any entry limits upgrades to the list; -Strict only decides whether their dependencies come too
        if any(apt.get("Unattended-Upgrade::Package-Whitelist::", [])):
            problems.append("a package allow-list")
        return not problems, "; ".join(problems) or "maintained"
    return judge


def updates_and_kernel(root: Path, finding: Mapping) -> tuple[bool | None, str]:
    return over_machines(root, cpu_machines(root), updates_judge(kernels(root)), "fall short")


def store_write_refused(root: Path, finding: Mapping) -> tuple[bool | None, str]:
    entry, when = reading(root, "store-write", STORE_COMMAND)
    if entry is None:
        return None, f"the agents' store key is on agent VMs, not where this runs; {when}"
    status, code = entry.get("status"), entry.get("code")
    if not (isinstance(status, int) and isinstance(code, str)):
        raise ValueError("the store-write reading holds the status and the code the store answered")
    if (status, code) == (403, "AccessDenied"):
        return True, f"the store refused an agent VM's key the scratch write, {when}"
    if (status, code) == (404, "NoSuchKey") or 200 <= status < 300:
        return False, f"the store took an agent VM's key's scratch write as allowed ({status} {code}), {when}"
    return None, f"the store answered {status} {code}, which says nothing of what the key may write; {when}"


def probe_store(environ: Mapping[str, str] = os.environ, remote: object | None = None) -> tuple[int, str]:
    """The status and code the store answers the store key in `environ` for a copy into a scratch key from one that doesn't
    exist: (403, AccessDenied) when the key may not write, (404, NoSuchKey) when it may. Neither writes anything, since there is
    nothing to copy, and nothing of the key is printed."""
    if remote is None:
        names = ("R2_ENDPOINT", "R2_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
        if missing := [n for n in names if not environ.get(n)]:
            raise SystemExit(f"--probe-store reads the agents' store key, and {', '.join(missing)} is not set: run it on an agent VM")
        from research.store.remote_s3 import S3Remote
        remote = S3Remote(*(environ[n] for n in names))
    tag = secrets.token_hex(8)
    try:
        with remote._request("PUT", f"{SCRATCH}/{tag}", headers={"x-amz-copy-source": f"/{remote.bucket}/{SCRATCH}/missing-{tag}"}):
            return 200, "OK"
    except urllib.error.HTTPError as e:
        code = re.search(rb"<Code>([^<]+)</Code>", e.read())
        return e.code, code[1].decode() if code else f"HTTP {e.code}"


#: each finding's check, by its id: the list names it, and can't hand a finding another's
CHECKS: Mapping[str, Check] = {
    "agent-holds-personal-and-cloud-keys": Check("agent-env-names", "recorded", agent_env_names),
    "guard-runs-from-the-branch": Check("guard-outside-checkout", "declared", guard_outside_checkout),
    "agents-have-root": Check("sudo-refused", "machine", sudo_refused),
    "security-updates-and-kernel": Check("updates-and-kernel", "machine", updates_and_kernel),
    "store-key-writes": Check("store-write-refused", "recorded", store_write_refused),
    "machine-addresses-in-the-tree": Check("no-public-address", "tree", no_public_address),
}


def status(finding: Mapping, root: Path = ROOT, checks: Mapping[str, Check] = CHECKS) -> Status:
    """Whether `finding`, one [[finding]] of the list, is open, and what decided it."""
    fid = str(finding.get("id", "?"))
    named = finding.get("check")
    if (check := checks.get(fid)) is None:
        if named is not None:
            return Status(fid, True, "refused", f"it names the check {named!r}, and open_findings.py runs none for it")
        return Status(fid, True, "none", f"no check; what would close it: {finding.get('would_close', '')}")
    if (named, finding.get("reads")) != (check.name, check.reads):
        return Status(fid, True, "refused", f"it names the check {named!r} on {finding.get('reads')!r}, and "
                                            f"open_findings.py runs {check.name} on {check.reads!r}")
    try:
        closed, why = check.run(root, finding)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as e:
        return Status(fid, True, "refused", f"its check {check.name} couldn't run: {type(e).__name__}: {e}")
    if closed is None:
        return Status(fid, True, "unread", why)
    return Status(fid, not closed, check.reads, why)


def mark(statuses: list[Status]) -> str:
    return "measured" if statuses and all(s.basis == "tree" for s in statuses) else "partial"


def load(path: Path = LIST) -> list[dict]:
    """The list's findings, each an id, a statement, and either a check (`check`, `reads`, `how`) or `would_close`: the
    check that would close it, in one line."""
    findings = tomllib.loads(path.read_text(encoding="utf-8")).get("finding", [])
    def words(f: Mapping, k: str) -> bool:
        return isinstance(f.get(k), str) and bool(f[k].strip())

    for f in findings:
        if not (words(f, "id") and words(f, "statement")):
            raise ValueError(f"{path}: each [[finding]] has an id and a statement: {f.get('id')!r}")
        if extra := sorted(set(f) - KEYS):
            raise ValueError(f"{path}: {f['id']} holds {', '.join(extra)}, which no finding holds: only a check closes one")
        if {"check", "reads", "how"} & set(f):
            whole = all(words(f, k) for k in ("check", "reads", "how")) and "would_close" not in f
        else:
            whole = words(f, "would_close") and "\n" not in f["would_close"]
        if not whole:
            raise ValueError(f"{path}: {f['id']} has either a check (check, reads and how) or, in one line, the check that "
                             f"would close it (would_close)")
    if len({f["id"] for f in findings}) != len(findings):
        raise ValueError(f"{path}: two [[finding]] entries share an id")
    return findings


def main(argv: Sequence[str] | None = None, path: Path = LIST, root: Path = ROOT,
         checks: Mapping[str, Check] = CHECKS) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="out", help="where the result goes outside a research run")
    ap.add_argument("--probe-store", action="store_true",
                    help="on an agent VM: print what the store answers its store key for a scratch write, as [store-write] records it")
    args = ap.parse_args(argv)
    if args.probe_store:
        status_, code = probe_store()
        print(f"status = {status_}\ncode = {json.dumps(code)}")
        return
    out = args.out
    statuses = [status(f, root, checks) for f in load(path)]
    n, m = sum(s.open for s in statuses), mark(statuses)
    unread = sum(s.basis == "unread" for s in statuses)
    print(f"security findings open: {n} of {len(statuses)} (mark {m})" +
          (f", {unread} of them unread: no check could read them where this ran" if unread else ""))
    for s in statuses:
        print(f"  {s.id}: unread: {s.why}" if s.basis == "unread" else
              f"  {s.id}: {'open' if s.open else 'closed'} ({s.basis}): {s.why}")
    w = kit.writer(out)
    w.measurement(METRIC, n, "findings", target=0, mark=m, inputs=[str(path.relative_to(root))])
    refused = [s.id for s in statuses if s.basis == "refused"]
    w.validation("failed" if refused else "passed", "benchmarks/security_findings/open_findings.py",
                 f"entries refused: {', '.join(refused)}" if refused else f"read {len(statuses)} findings")


if __name__ == "__main__":
    main()
