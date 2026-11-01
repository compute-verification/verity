"""The auditor host's verifier service: PoUB's verifier
(`benchmarks/remote_session/verifier.py`, the Lean verifier deciding) runs only as the `auditor` user,
from a tree infra pinned, one run at a time, each run's parameters a lane's request held to a fixed whitelist.

    vy-verifier-gate run --peer NAME --port P --period P --pin SHA512 (--paced SPEC ... [--pace-pass-ms MS]
                         | --sessions N [--windows W] [--window-ms MS] [--samples S])
                         [--minutes M] [--jitter-ms MS] [--quantile Q] [--lifetime L]
                         [--model ROLE [--engines E] --arena-blocks A]
    vy-verifier-gate status [RUN] | fetch RUN [--transcripts] | replay RUN | stop RUN | close NODE_HEX

- **Who reaches it.** Every key in `research`'s authorized_keys on the host is forced to `vy-verifier-gate` (`command=`),
  which is this file's `gate` as root through one sudoers line, as on the challenger's host
  (`benchmarks/live_challenger/auditor.py`). A lane's own key is forced with the lane's name (`vy-verifier-gate LANE`,
  from `/etc/vy/verifier-gate-keys`, `vy-verifier-setup`), and only such a key starts, replays, stops a run or closes a
  registration: the run records that name as `by`. A key forced with no name (`-`) reads only: `status`, `fetch`. A lane
  names a run's parameters and gets a run; it never names code, a path, a host or a node's capacities. Infra's
  `vy-admin` runs the same gate (`sudo vy-verifier-gate VERB ...`, `by` vy-admin).
- **What runs.** `/opt/vy-verifier/current` -> `/opt/vy-verifier/<commit>`, a root-owned `git archive` of that commit
  (`pin.sh`), whose Lean executables pin.sh built; the commit is the directory's name, and every run's request names it
  (`pinned`): a run whose pin moved before its unit started doesn't run. pin.sh appends each pin (commit, UTC time) to
  `/opt/vy-verifier/pins.jsonl`, root's and append-only (`record_pin`, `chattr +a`), before it moves `current`; a run
  starts only from the log's last pin, and its request names that line (`pin_log`), so every session can be held to a
  pin infra made. State is `auditor`'s, in `/var/lib/vy-verifier`
  (0700): `runs/<run>/`, the verifier's registry (`registry.json`, kept across runs, `verifier.py --registry`) and the key
  that reaches the challenger's gate (`.ssh/vy-challenger-gate`, which infra registers there for the lane in
  `verifier.json`).
- **A run** (`run`) is verifier.py with the whitelisted parameters and the host's own: it listens on 127.0.0.1 at
  `--port` (one of `PORTS`), takes each session's coins from the challenger host through its gate (`--challenger gate`, the
  host's key and this host's name there, under the lane's `--period` and `--pin`), holds each session's pin to the
  challenger's pins log, which it fetches through the same gate (`pins`), and to main's first-parent history at
  `/opt/vy-verifier/main-first-parent.txt` (`--challenger-main`; infra writes `git rev-list --first-parent origin/main`
  there, root's and 0644, whenever it pins either host, and no run starts without it), writes every session's transcript
  (`--transcript`) and keeps the registry. The audited node's capacities, which its memory line is measured against
  (`--host-bytes`, `--ingress-bytes-per-s`, `--hbm-bytes`), are the host's for `--peer` (`/etc/vy/verifier.json`),
  never the request's. The prover reaches the port only through its tunnel account's `ssh -L` (`vy-verifier-tunnel`),
  which may open nothing else; `status RUN` says when the verifier listens, and the prover connects after that. The
  unit, `vy-verifier@<run>.service`, leaves `runs/<run>/out/` as `benchmarks/remote_session/run.sh`'s ROLE=verifier
  does (`verifier.json`, `challenger/`, `verifier.log`, `registry.json`), which `auditor_line.py` reads, and the
  transcripts gzipped in `out/transcripts/` with their `SHA256SUMS`; then `end.json` (exit code, who stopped it, each
  file's SHA-512). One unit runs at a time, a run or a replay, so a replay never takes a live session's cores.
- **A bound run** (`--model ROLE --arena-blocks A [--engines E]`, ROLE of [A-Z0-9_], E 1..8 and verifier.py's 8 by
  default, A at most 2^25) binds the registered store to a pinned model (`verifier.py --model`,
  `remote_bandwidth.store.Binding`), which `auditor_line.py`'s `bound` check needs. The lane names the role and the store's shape, never a path: the gate
  adds `--checkpoint <models>/<ROLE>`, the host's directory of the role's checkpoint, and `--pin` core's pin in the
  pinned tree (`verity/core/protocols/pouw/lean/check/models.json`, `MODEL_PINS`), which the gate's own `--pin` (the
  challenger's key, verifier.py's `--challenger-pin`) is not. It refuses the run, before writing or starting anything,
  when the role has no directory there or the pinned tree has no `models.json`; verifier.py then holds every file the
  role's entry pins to its SHA-512 before it listens, and refuses a role the pin lacks. `<models>` is `verifier.json`'s
  `models`, default `/var/lib/vy-verifier/models`. Infra provisions a role's directory: every file the role's entry
  pins, at the pinned revision, as a root-held regular file (0644, in a root-held 0755 directory), copied out of a
  download, never an HF cache's snapshot, whose files are symlinks into its blobs and which
  `remote_bandwidth.store.check` refuses. Qwen2.5-32B in BF16, role `QWEN25_32B_BF16`, about 66 GB:

      hf download Qwen/Qwen2.5-32B --revision 1818d35814b8319459f4bd55ed1ac8709630f003 --local-dir /tmp/q
      install -d -m 0755 -o root -g root /var/lib/vy-verifier/models/QWEN25_32B_BF16
      install -m 0644 -o root -g root /tmp/q/* /var/lib/vy-verifier/models/QWEN25_32B_BF16/

  The files are held to the pin, not to where they came from: a run binds to the role once the pinned tree's
  `models.json` pins it.
- **A replay** (`replay RUN`, `vy-verifier-replay@<run>.service`) checks each transcript against `SHA256SUMS` and
  against the sha256 and bytes `verifier.json` recorded for it, and replays them through both verifiers
  (`verity/core/protocols/remote_bandwidth/lean/tests/replay.py transcript --record verifier.json`), or the record
  alone when the run left no transcript, into `runs/<run>/replay/`; a transcript that isn't the recorded one fails it.
- **The memory challenger's parties**: `wipe_party [--reads K] [--native [--threads T]]` and `challenger_party`
  run PoUS's streamed-wipe challenger (`verity/core/protocols/pous/wipe_party.py`, port 47611) and its block challenger
  (`challenger_party.py`, port 47610) as `auditor` from the pinned tree, one session per ssh connection: the request line
  on the gate's stdin, `{"listening"}` and the signed report on its stdout, as a lane's `--challenger-command` expects
  (`benchmarks/pous/block/block.py`). The prover connects to the host's address at the port over direct TCP, which the
  security groups admit from the card only. Each party's LMS key and session log are `<root>/parties/<wipe|pous>/`, made
  on the host, never leaving it; `party_key wipe|pous` prints the public half, which the lane pins as `--trusted`, and
  `party_log wipe|pous [LOG]` lets any key read the sessions and each one's stdout as the host kept it (its signed t0 and
  t1). A lane
  never names a path, an address or an egress count: a session signs `UNCOUNTED` egress until the network certifier's
  count reaches the host. A session waits for no verifier unit to run, each party serves one session at a time, and the
  gate kills a party after its session's `--minutes M` (60 by default, at most 180), so a session no prover joined ends.
- **The store.** When a run's or a replay's unit stops, `vy-verifier-publish@<run>.service` puts `runs/<run>/` in the
  evidence store (a preserved tree of kind `verifier-run/v1`, by the pinned tree's store client) with the credential infra keeps in
  `/etc/vy/verifier-store.env`, which only that unit reads.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gzip
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
VERIFIER = TREE / "benchmarks" / "remote_session" / "verifier.py"
REPLAY = TREE / "verity" / "core" / "protocols" / "remote_bandwidth" / "lean" / "tests"
DIFFTEST = TREE / "verity" / "core" / "difftest" / "python"
RESEARCH = TREE / "tools" / "research" / "src"

PORTS = range(7951, 7959)
"""The ports a run listens on, each 127.0.0.1's, the tunnel account's `PermitOpen` (`12-vy-verifier.conf`): outside the
challenger gate's 7811..7899, which a certifier coin link binds on a serving node."""
GATE_PORTS = range(7811, 7900)
PINNED = re.compile(r"[0-9a-f]{40}")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
RUN = re.compile(r"v\d{8}T\d{6}Z-[0-9a-f]{4}")
SHA512 = re.compile(r"[0-9a-f]{128}")
NODE = re.compile(r"(?:[0-9a-f]{2}){1,128}")
PACED = re.compile(r"([1-9][0-9]{0,2}):([0-9]{1,6}(?:\.[0-9]{1,3})?):([1-9][0-9]{0,6}):([1-9][0-9]{0,2})")
COUNT = re.compile(r"[1-9][0-9]{0,6}")
DECIMAL = re.compile(r"[0-9]{1,6}(?:\.[0-9]{1,3})?")
READ_ONLY = "-"
HOST_ACCOUNTS = ("root", "vy-admin", "research", "auditor", "prover", "challenger")
QUANTILES = ("p50", "p99", "p999", "max")
MINUTES = (1, 720)
SPECS = 16
INTS = {"--sessions": (1, 16), "--windows": (1, 64), "--samples": (1, 100_000), "--lifetime": (1, 1024),
        "--minutes": MINUTES}
DECIMALS = {"--window-ms": (1.0, 600_000.0), "--jitter-ms": (0.0, 1_000.0), "--pace-pass-ms": (0.0, 1_000.0)}
ROLE = re.compile(r"[A-Z0-9_]{1,64}")
STORE = {"--engines": (1, 8), "--arena-blocks": (1, 1 << 25)}
"""A bound run's store: its arenas, one a GPU of the node's 8, and each arena's 4 KiB blocks, at most 128 GiB (an RTX
PRO 6000 has 96 GB of HBM)."""
STORE_INT = re.compile(r"[1-9][0-9]{0,7}")
MODEL_PINS = Path("verity/core/protocols/pouw/lean/check/models.json")
"""Core's pin of each model role, in the pinned tree: a bound run's `verifier.py --pin`."""
PARTIES = {"wipe_party": ("wipe", "verity.core.protocols.pous.wipe_party", 47611),
           "challenger_party": ("pous", "verity.core.protocols.pous.challenger_party", 47610)}
PARTY_INTS = {"--reads": (1, 1 << 20), "--threads": (1, 256)}
PARTY_MINUTES = (1, 180)
"""A party session's limit (`--minutes M`, 60 by default): the gate kills the party then, so one no prover joined, or whose ssh
connection dropped, never holds its port and lock."""
UNIT = "vy-verifier@{}.service"
REPLAY_UNIT = "vy-verifier-replay@{}.service"
LOG_LINES = 20


class Refused(Exception):
    """A request the gate or the runner refuses, with why."""


@dataclass(frozen=True)
class Host:
    """Where the service keeps its state, and its configuration (infra's, deployed: `verifier.json`)."""

    root: Path = Path("/var/lib/vy-verifier")
    config: Path = Path("/etc/vy/verifier.json")
    pins: Path = Path("/opt/vy-verifier/pins.jsonl")
    main: Path = Path("/opt/vy-verifier/main-first-parent.txt")

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def scratch(self) -> Path:
        return self.root / "scratch"

    @property
    def registry(self) -> Path:
        return self.root / "registry.json"

    @property
    def parties(self) -> Path:
        return self.root / "parties"

    def settings(self) -> dict:
        return json.loads(self.config.read_text())

    def models(self, settings: dict) -> Path:
        """Where infra keeps each bound role's checkpoint, `<models>/<ROLE>/`: `verifier.json`'s `models`, else
        `root/models`."""
        return Path(settings.get("models") or self.root / "models")


def utc(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pinned_commit(tree: Path = TREE) -> str:
    """The commit this tree was pinned at: its directory's name under /opt/vy-verifier."""
    if not PINNED.fullmatch(tree.name):
        raise Refused(f"{tree} is not a pinned tree (/opt/vy-verifier/<commit>, pin.sh): no run runs from it")
    return tree.name


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def digest_of(path: Path, name: str = "sha512") -> str:
    h = hashlib.new(name)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def append_only(path: Path) -> None:
    subprocess.run(["chattr", "+a", str(path)], check=True)


def record_pin(log: Path, commit: str, now: datetime, *, seal: Callable[[Path], None] = append_only) -> dict:
    """pin.sh's last step before it moves `current`, as root: the pin's line (commit, UTC time) appended to `log`, which
    is root's, 0644, and append-only (`chattr +a`, which only root can lift, as the challenger's ledger) before any line
    is written. A failure here fails the pin, so `current` never names a tree the log doesn't."""
    if not PINNED.fullmatch(commit):
        raise Refused(f"pinned {commit!r}: a commit, 40 lowercase hex digits")
    if not log.exists():
        log.touch()
        log.chmod(0o644)
    seal(log)
    line = {"commit": commit, "utc": utc(now)}
    with open(log, "a") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")
    return line


def pin_of(host: Host, pinned: str) -> dict:
    """The pin log's line a run starts under: its last, which must name this tree's commit."""
    try:
        lines = host.pins.read_text().splitlines()
    except OSError:
        lines = []
    if not lines:
        raise Refused(f"{host.pins} records no pin: pin.sh logs every pin before it moves current, so no run starts "
                      "from an unlogged tree")
    try:
        last = json.loads(lines[-1])
    except ValueError:
        raise Refused(f"{host.pins}'s last line isn't a pin: no run starts until pin.sh logs one") from None
    if last.get("commit") != pinned:
        raise Refused(f"the pin log's last pin is {last.get('commit')}, not this tree's {pinned}: a run starts only "
                      "from the tree pin.sh last pinned")
    return {"line": len(lines) - 1, "utc": last.get("utc")}


# --- the gate (root) ----------------------------------------------------------------------------------------------------------

def parse_run(argv: list[str], settings: dict) -> dict:
    """A run's request, every value held to the whitelist: what verifier.py is asked, as the gate will pass it."""
    p = argparse.ArgumentParser(prog="vy-verifier-gate run", exit_on_error=False, add_help=False, allow_abbrev=False)
    for flag in ("--peer", "--port", "--period", "--pin"):
        p.add_argument(flag, required=True)
    p.add_argument("--paced", nargs="+")
    for flag in (*INTS, *DECIMALS, "--quantile", "--model", *STORE):
        p.add_argument(flag)

    def refuse(why: object) -> None:
        raise Refused(f"run: {why}; {' '.join(p.format_usage().split())} (no path, host, key or node capacity: the "
                      "host's own, /etc/vy/verifier.json)")

    p.error = refuse
    try:
        a = p.parse_args(argv)
    except argparse.ArgumentError as e:
        refuse(e)
    peers = settings.get("peers", {})
    if a.peer not in peers:
        raise Refused(f"--peer {a.peer!r}: one of {sorted(peers)} (/etc/vy/verifier.json)")
    if not re.fullmatch(r"[0-9]{4}", a.port) or int(a.port) not in PORTS:
        raise Refused(f"--port {a.port!r}: one of {PORTS.start}..{PORTS.stop - 1}, the tunnel's")
    if not NAME.fullmatch(a.period):
        raise Refused(f"--period {a.period!r}: the lane's open period on the challenger's ledger, 1 to 64 of "
                      "[A-Za-z0-9._:-]")
    if not SHA512.fullmatch(a.pin):
        raise Refused("--pin: the SHA-512 of the challenger's public key, 128 lowercase hex digits")
    req: dict[str, Any] = {"peer": a.peer, "port": int(a.port), "period": a.period, "pin": a.pin}
    for flag, (lo, hi) in INTS.items():
        v = getattr(a, flag[2:].replace("-", "_"))
        if v is not None:
            if not COUNT.fullmatch(v) or not lo <= int(v) <= hi:
                raise Refused(f"{flag} {v!r}: an integer, {lo} to {hi}")
            req[flag[2:]] = int(v)
    for flag, (lo, hi) in DECIMALS.items():
        v = getattr(a, flag[2:].replace("-", "_"))
        if v is not None:
            if not DECIMAL.fullmatch(v) or not lo <= float(v) <= hi:
                raise Refused(f"{flag} {v!r}: a decimal, {lo:g} to {hi:g}, at most 3 places")
            req[flag[2:]] = float(v)
    if a.quantile is not None:
        if a.quantile not in QUANTILES:
            raise Refused(f"--quantile {a.quantile!r}: one of {QUANTILES}")
        req["quantile"] = a.quantile
    if a.paced is not None:
        if len(a.paced) > SPECS or not all(PACED.fullmatch(s) for s in a.paced):
            raise Refused(f"--paced: 1 to {SPECS} specs K:T_MS:DRAWS:WINDOWS (K, DRAWS, WINDOWS positive integers of "
                          "at most 3, 7, 3 digits; T_MS a decimal)")
        if any(k in req for k in ("sessions", "windows", "window-ms", "samples")):
            raise Refused("--paced replaces --sessions, --windows, --window-ms and --samples (verifier.py's)")
        req["paced"] = list(a.paced)
    elif "sessions" not in req:
        raise Refused("run: --paced SPEC ... or --sessions N")
    elif "pace-pass-ms" in req:
        raise Refused("--pace-pass-ms is a paced run's")
    if a.model is not None:
        if not ROLE.fullmatch(a.model):
            raise Refused(f"--model {a.model!r}: a role of core's pin ({MODEL_PINS}), 1 to 64 of [A-Z0-9_]")
        req["model"] = a.model
    for flag, (lo, hi) in STORE.items():
        v = getattr(a, flag[2:].replace("-", "_"))
        if v is not None:
            if a.model is None:
                raise Refused(f"{flag} shapes a bound run's store: it comes with --model ROLE")
            if not STORE_INT.fullmatch(v) or not lo <= int(v) <= hi:
                raise Refused(f"{flag} {v!r}: an integer, {lo} to {hi}")
            req[flag[2:]] = int(v)
    if a.model is not None and "arena-blocks" not in req:
        raise Refused("--model needs --arena-blocks A, each arena's blocks (verifier.py's)")
    return req


def bound_paths(host: Host, settings: dict, role: str, tree: Path = TREE) -> tuple[Path, Path]:
    """A bound run's checkpoint, the host's `<models>/<ROLE>/`, and its pin, core's `models.json` in the pinned tree;
    refused when either is missing. Neither is the request's: the role names both."""
    checkpoint, pin = host.models(settings) / role, tree / MODEL_PINS
    if not checkpoint.is_dir():
        raise Refused(f"--model {role}: this host has no {checkpoint}, the role's checkpoint (infra provisions it, "
                      "root-held regular files)")
    if not pin.is_file():
        raise Refused(f"--model {role}: the pinned tree has no {MODEL_PINS}, core's pin of each role, which the "
                      "verifier binds the store to")
    return checkpoint, pin


def verifier_argv(host: Host, run: str, req: dict, settings: dict, *, python: str = "/usr/bin/python3",
                  tree: Path = TREE) -> list[str]:
    """verifier.py's command line for the run: the request's whitelisted values, and the host's paths, challenger,
    first-parent list of main (`Host.main`, refused when infra hasn't written it) and the peer's capacities; a bound
    run's (`--model`) its role's checkpoint and core's pin (`bound_paths`), refused when either is missing."""
    ch, node = settings["challenger"], settings["peers"][req["peer"]]
    if not host.main.is_file():
        raise Refused(f"this host has no {host.main}, main's first-parent history (infra writes `git rev-list "
                      "--first-parent origin/main` there, root's, 0644): the verifier refuses every gate session whose "
                      "challenger pin it can't find on it, so no run starts without it")
    argv = [python, "-I", str(tree / "benchmarks" / "remote_session" / "verifier.py"),
            "--host", "127.0.0.1", "--port", str(req["port"]), "--out", str(host.runs / run / "out"),
            "--registry", str(host.registry), "--transcript", str(host.scratch / run / "transcripts"),
            "--challenger", "gate", "--challenger-ssh", ch["ssh"], "--challenger-peer", ch["peer"],
            "--challenger-pin", req["pin"], "--challenger-period", req["period"], "--challenger-main", str(host.main)]
    for key in (*INTS, *DECIMALS, "--quantile"):
        name = key.lstrip("-")
        if name in req:
            argv += [f"--challenger-{name}" if name == "minutes" else key, str(req[name])]
    for key in ("host_bytes", "hbm_bytes", "ingress_bytes_per_s"):
        if key in node:
            argv += ["--" + key.replace("_", "-"), str(node[key])]
    if "model" in req:
        checkpoint, pin = bound_paths(host, settings, req["model"], tree)
        argv += ["--model", req["model"], "--checkpoint", str(checkpoint), "--pin", str(pin)]
        for key in STORE:
            if key[2:] in req:
                argv += [key, str(req[key[2:]])]
    if "paced" in req:
        argv += ["--paced", *req["paced"]]
    return argv


def busy_units() -> list[str]:
    """The service's live units, a run's or a replay's, a stopping one too (it still gzips its transcripts)."""
    p = subprocess.run(["systemctl", "list-units", "--plain", "--no-legend", "--state=active,activating,deactivating",
                        "vy-verifier@*", "vy-verifier-replay@*"], capture_output=True, text=True)
    if p.returncode:
        raise Refused(f"systemctl list-units failed ({p.stderr.strip()}): no unit starts while the host can't tell "
                      "what runs")
    return sorted(ln.split()[0] for ln in p.stdout.splitlines() if ln.strip())


def start_unit(unit: str) -> None:
    """Blocks until the unit runs (Type=exec)."""
    p = subprocess.run(["systemctl", "start", unit], capture_output=True, text=True)
    if p.returncode:
        raise Refused(f"{unit} didn't start ({p.stderr.strip()})")


def stop_unit(unit: str) -> None:
    subprocess.run(["systemctl", "stop", "--no-block", unit], capture_output=True)


def own_by_auditor(path: Path) -> None:
    import pwd
    pw = pwd.getpwnam("auditor")
    os.chown(path, pw.pw_uid, pw.pw_gid)


def request_of(host: Host, run: str) -> dict:
    if not RUN.fullmatch(run):
        raise Refused(f"no run {run!r}")
    req = read_json(host.runs / run / "request.json")
    if req is None:
        raise Refused(f"no run {run}")
    return req


def gate_run(argv: list[str], host: Host, *, by: str, now: datetime, token: str, pinned: str,
             busy: Callable[[], list[str]], start: Callable[[str], None], own: Callable[[Path], None],
             tree: Path = TREE) -> dict:
    """Validate a request, write it as the run's `request.json` and start its unit; refused while a unit runs."""
    settings = host.settings()
    req = parse_run(argv, settings)
    with locked(host.root / "gate.lock"):
        if units := busy():
            raise Refused(f"{', '.join(units)} runs: one unit at a time (stop RUN ends a run)")
        logged = pin_of(host, pinned)
        run = f"v{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{token}"
        req = {"run": run, "by": by, "requested_utc": utc(now), "pinned": pinned, "pin_log": logged, **req}
        req["argv"] = verifier_argv(host, run, req, settings, tree=tree)
        d = host.runs / run
        d.mkdir(mode=0o755, parents=False)
        (d / "request.json").write_text(json.dumps(req, indent=1, sort_keys=True) + "\n")
        own(d)
        own(d / "request.json")
        start(UNIT.format(run))
    return {"run": run, "by": by, "pinned": pinned, "pin_log": logged, "peer": req["peer"],
            "listen": f"127.0.0.1:{req['port']}", "argv": req["argv"]}


def log_tail(path: Path, n: int = LOG_LINES) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def gate_status(rest: list[str], host: Host, busy: Callable[[], list[str]], pinned: str | None) -> dict:
    """The host (no RUN): its pin, the pin log's last lines, its active units and its runs. A run: its request, whether
    its verifier listens (the prover connects only after), its end, its replay's, and its log's last lines."""
    units = busy()
    if not rest:
        runs = sorted(p.name for p in host.runs.iterdir() if RUN.fullmatch(p.name)) if host.runs.is_dir() else []
        pins = log_tail(host.pins, 10 ** 6)
        first = max(len(pins) - LOG_LINES, 0)
        return {"pinned": pinned, "pins": [{"line": first + i, **json.loads(ln)} for i, ln in enumerate(pins[first:])],
                "busy": units,
                "runs": [{"run": r, "by": (read_json(host.runs / r / "request.json") or {}).get("by"),
                          "rc": (read_json(host.runs / r / "end.json") or {}).get("rc"),
                          "active": UNIT.format(r) in units} for r in runs[-50:]]}
    run = rest[0]
    req = request_of(host, run)
    d = host.runs / run
    log = log_tail(d / "out" / "verifier.log", 10 ** 6)
    replay = read_json(d / "replay" / "end.json")
    return {"run": run, "request": req, "active": UNIT.format(run) in units,
            "listening": any(ln.startswith("verifier: listening on ") for ln in log),
            "end": read_json(d / "end.json"),
            "replay": replay if replay is not None else ("running" if REPLAY_UNIT.format(run) in units else None),
            "log": log[-LOG_LINES:]}


def gate_fetch(rest: list[str], host: Host, sink: BinaryIO) -> None:
    """The run's directory as a tar on `sink`: gzipped and without the transcripts (their SHA256SUMS kept), or with
    `--transcripts` everything, uncompressed (the transcripts are gzipped already)."""
    if len(rest) not in (1, 2) or rest[1:] not in ([], ["--transcripts"]):
        raise Refused("fetch RUN [--transcripts]")
    run, whole = rest[0], rest[1:] == ["--transcripts"]
    request_of(host, run)

    def keep(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if not whole and ti.name.startswith(f"{run}/out/transcripts/") and ti.name.endswith(".json.gz"):
            return None
        return ti

    with tarfile.open(fileobj=sink, mode="w|" if whole else "w|gz") as tar:
        tar.add(host.runs / run, arcname=run, filter=keep)


def gate_replay(rest: list[str], host: Host, *, by: str, busy: Callable[[], list[str]],
                start: Callable[[str], None]) -> dict:
    """Start the run's replay unit: refused while any unit runs, or before the run has ended."""
    if len(rest) != 1:
        raise Refused("replay RUN")
    run = rest[0]
    request_of(host, run)
    with locked(host.root / "gate.lock"):
        if units := busy():
            raise Refused(f"{', '.join(units)} runs: one unit at a time")
        if read_json(host.runs / run / "end.json") is None:
            raise Refused(f"run {run} has no end: it replays once it has ended")
        start(REPLAY_UNIT.format(run))
    return {"run": run, "replay": "started", "by": by}


def gate_stop(rest: list[str], host: Host, *, by: str, admin: bool, busy: Callable[[], list[str]],
              stop: Callable[[str], None], now: datetime, own: Callable[[Path], None]) -> dict:
    """Stop the run's active unit (its own lane's, or any for vy-admin): its end records who stopped it."""
    if len(rest) != 1:
        raise Refused("stop RUN")
    run = rest[0]
    req = request_of(host, run)
    if not admin and req["by"] != by:
        raise Refused(f"run {run} is {req['by']}'s, not {by}'s")
    units = [u for u in busy() if u in (UNIT.format(run), REPLAY_UNIT.format(run))]
    if not units:
        raise Refused(f"run {run} has no active unit")
    f = host.runs / run / "stop.json"
    f.write_text(json.dumps({"by": by, "utc": utc(now)}) + "\n")
    own(f)
    for u in units:
        stop(u)
    return {"run": run, "stopping": units, "by": by}


def gate_close(rest: list[str], host: Host, *, by: str, busy: Callable[[], list[str]],
               close: Callable[[str], dict], now: datetime, own: Callable[[Path], None]) -> dict:
    """Close a node's registration left live in the host's registry by a run that ended early (`verifier.py --close`):
    only while no unit runs, when every live registration there is one."""
    if len(rest) != 1 or not NODE.fullmatch(rest[0]):
        raise Refused("close NODE_HEX (the node's id, lowercase hex)")
    with locked(host.root / "gate.lock"):
        if units := busy():
            raise Refused(f"{', '.join(units)} runs: a registration is closed only while no run is live")
        doc = close(rest[0])
        f = host.root / "closes.jsonl"
        with open(f, "a") as out:
            out.write(json.dumps({"node": rest[0], "by": by, "utc": utc(now), "closed": doc}, sort_keys=True) + "\n")
        own(f)
    return {"node": rest[0], "by": by, "closed": doc}


def _as_auditor_python(*args: str) -> list[str]:
    return ["runuser", "-u", "auditor", "--", "/usr/bin/env", "-i", "PYTHONDONTWRITEBYTECODE=1", "HOME=/var/lib/vy-verifier",
            "PATH=/usr/bin:/bin", "/usr/bin/python3", "-s", "-E", *args]


def party_args(verb: str, rest: list[str]) -> tuple[list[str], int]:
    """`rest` without `--minutes M`, and M (`PARTY_MINUTES`, 60 by default)."""
    if "--minutes" not in rest:
        return list(rest), 60
    i = rest.index("--minutes")
    v = rest[i + 1] if i + 1 < len(rest) else ""
    lo, hi = PARTY_MINUTES
    if not COUNT.fullmatch(v) or not lo <= int(v) <= hi or rest.count("--minutes") > 1:
        raise Refused(f"{verb} --minutes takes one integer in [{lo}, {hi}], got {v!r}")
    return rest[:i] + rest[i + 2:], int(v)


def party_argv(verb: str, rest: list[str], host: Host) -> list[str]:
    """The party's command line for `verb` (`PARTIES`): its module from the pinned tree as `auditor`, its key and session log in
    `<root>/parties/<name>` (made there, never leaving the host), listening on every address at its port. A lane names
    `wipe_party`'s `--reads K`, `--native` (the streamer, built on this host) and `--threads T`, and nothing of
    `challenger_party`'s; never a path, an address or an egress count (the session signs UNCOUNTED, which bounds nothing,
    until the network certifier's count reaches the host)."""
    name, module, port = PARTIES[verb]
    extra: list[str] = []
    args = list(rest)
    while args:
        flag = args.pop(0)
        if verb == "wipe_party" and flag == "--native" and "--streamer" not in extra:
            extra += ["--streamer", "build"]
        elif verb == "wipe_party" and flag in PARTY_INTS and args and flag not in extra:
            lo, hi = PARTY_INTS[flag]
            v = args.pop(0)
            if not COUNT.fullmatch(v) or not lo <= int(v) <= hi:
                raise Refused(f"{verb} {flag} takes an integer in [{lo}, {hi}], got {v!r}")
            extra += [flag, v]
        else:
            raise Refused(f"{verb} takes {'[--reads K] [--native] [--threads T]' if verb == 'wipe_party' else 'no flags'}"
                          f"; got {flag!r}")
    if "--threads" in extra and "--streamer" not in extra:
        raise Refused("wipe_party --threads sets the native streamer's threads: give --native too")
    return _as_auditor_python("-m", module, "--device", str(host.parties / name), "--listen", f"0.0.0.0:{port}", *extra)


def exec_party(argv: list[str], sink: BinaryIO, log: Path, minutes: int) -> int:
    """Run the party with the gate's stdin (the request line), its stdout forwarded line by line to `sink` and to `log`; killed
    after `minutes`, which `log` records."""
    with open(log, "ab") as out, subprocess.Popen(argv, cwd=TREE, stdout=subprocess.PIPE, start_new_session=True) as p:
        ran_out = threading.Event()

        def kill() -> None:
            ran_out.set()
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)    # runuser's child too: killing runuser alone leaves the party listening
        timer = threading.Timer(minutes * 60, kill)
        timer.start()
        try:
            for line in p.stdout:
                out.write(line)
                out.flush()
                with contextlib.suppress(OSError):
                    sink.write(line)
                    sink.flush()
        finally:
            timer.cancel()
        p.wait()
        if ran_out.is_set():
            out.write(json.dumps({"killed": f"the session's {minutes} min ran out"}).encode() + b"\n")
    return p.returncode


def gate_party(verb: str, rest: list[str], host: Host, *, by: str, now: datetime, busy: Callable[[], list[str]],
               own: Callable[[Path], None], run: Callable[[list[str], BinaryIO, Path, int], int], sink: BinaryIO) -> int:
    """`wipe_party` / `challenger_party`: one session of the party, served while the ssh connection lasts; refused while a
    verifier unit runs (its cores would time the reads) or the same party serves another session. Each session's request
    and end go to `<root>/parties/<name>/requests.jsonl`, its stdout to `log/<utc>.jsonl` beside it."""
    rest, minutes = party_args(verb, rest)
    argv = party_argv(verb, rest, host)
    d = host.parties / PARTIES[verb][0]
    for p in (host.parties, d, d / "log"):
        if not p.is_dir():
            p.mkdir(mode=0o700)
            own(p)
    with open(host.parties / f"{PARTIES[verb][0]}.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Refused(f"{verb} is serving another session; one at a time") from None
        if units := busy():
            raise Refused(f"{', '.join(units)} runs: a party session waits for no unit to run")
        stamp = f"{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}"
        log = d / "log" / f"{stamp}.jsonl"
        reqs = d / "requests.jsonl"
        with open(reqs, "a") as f:
            f.write(json.dumps({"verb": verb, "by": by, "utc": utc(now), "argv": argv, "log": log.name, "minutes": minutes},
                               sort_keys=True) + "\n")
        rc = run(argv, sink, log, minutes)
        with open(reqs, "a") as f:
            f.write(json.dumps({"verb": verb, "by": by, "log": log.name, "rc": rc}, sort_keys=True) + "\n")
        for p in (reqs, log):
            if p.exists():
                own(p)
    return rc


LOG_NAME = re.compile(r"\d{8}T\d{6}Z\.jsonl")


def gate_party_log(rest: list[str], host: Host) -> dict:
    """`party_log wipe|pous [LOG]`: a party's sessions (each request and end, `requests.jsonl`), or one session's stdout as the
    host kept it (`log/LOG`: `{"listening"}` and the signed report, whose `window` is t0 and t1 on this host's clock), so a
    reader takes them from the challenger's host and never through the card. Any key reads it."""
    names = {n for n, _, _ in PARTIES.values()}
    if not rest or rest[0] not in names or len(rest) > 2 or (len(rest) == 2 and not LOG_NAME.fullmatch(rest[1])):
        raise Refused(f"party_log {'|'.join(sorted(names))} [LOG]")
    d = host.parties / rest[0]
    if len(rest) == 1:
        reqs = d / "requests.jsonl"
        return {"party": rest[0], "sessions": [json.loads(ln) for ln in reqs.read_text().splitlines() if ln.strip()]
                if reqs.is_file() else []}
    f = d / "log" / rest[1]
    if not f.is_file():
        raise Refused(f"no session log {rest[1]} for {rest[0]}")
    lines = []
    for ln in f.read_text(errors="replace").splitlines():
        try:
            lines.append(json.loads(ln))
        except ValueError:
            lines.append(ln)
    return {"party": rest[0], "log": rest[1], "lines": lines}


KEYGEN = "import sys; from verity.core.protocols.pous import period; print(period.challenger(sys.argv[1]).public_key.hex())"


def gate_party_key(rest: list[str], host: Host, *, own: Callable[[Path], None],
                   keygen: Callable[[list[str]], str]) -> dict:
    """`party_key wipe|pous`: the party's public key, its key made on this host the first time it's asked (only the public
    half leaves it)."""
    names = {n: v for v, (n, _, _) in PARTIES.items()}
    if len(rest) != 1 or rest[0] not in names:
        raise Refused(f"party_key {'|'.join(sorted(names))}")
    d = host.parties / rest[0]
    for p in (host.parties, d):
        if not p.is_dir():
            p.mkdir(mode=0o700)
            own(p)
    key = keygen(_as_auditor_python("-c", KEYGEN, str(d)))
    if not re.fullmatch(r"[0-9a-f]{16,}", key):
        raise Refused(f"the key generation printed {key[-200:]!r}, not a public key")
    return {"party": rest[0], "verb": names[rest[0]], "public_key": key}


def run_keygen(argv: list[str]) -> str:
    p = subprocess.run(argv, cwd=TREE, capture_output=True, text=True)
    if p.returncode:
        raise Refused((p.stderr.strip() or f"key generation exited {p.returncode}").splitlines()[-1])
    return (p.stdout.strip().splitlines() or [""])[-1]


USAGE = ("verbs: run --peer NAME --port P --period P --pin SHA512 (--paced SPEC ... [--pace-pass-ms MS] | --sessions N "
         "[--windows W] [--window-ms MS] [--samples S]) [--minutes M] [--jitter-ms MS] [--quantile Q] [--lifetime L] "
         "[--model ROLE [--engines E] --arena-blocks A] | replay RUN | stop RUN | close NODE_HEX "
         "| wipe_party [--reads K] [--native [--threads T]] [--minutes M] | challenger_party [--minutes M] (the request line "
         "on stdin) "
         "(a lane's own gate key, or vy-admin) | status [RUN] | fetch RUN [--transcripts] | party_key wipe|pous "
         "| party_log wipe|pous [LOG]")


def gate(argv: list[str], host: Host = Host(), sink: BinaryIO | None = None, **hooks: Any) -> int:
    """The gate's verbs, their answer on `sink` (stdout); the exit code. A refusal is exit 3 with {"refused": why}.
    Over ssh (`--ssh WORDS --key NAME`, the forced command's) `by` is NAME, the lane the key was registered for, and a
    key registered with no lane (`-`) reads only; run by root (vy-admin's sudo) `by` is the sudoer."""
    sink = sink if sink is not None else sys.stdout.buffer

    def answer(rc: int, doc: dict) -> int:
        sink.write(json.dumps(doc, sort_keys=True).encode() + b"\n")
        return rc

    by = hooks.get("by") or os.environ.get("SUDO_USER") or "root"
    admin = argv[:1] != ["--ssh"]
    if not admin:
        if len(argv) == 4 and argv[2] == "--key":
            key = argv[3]
        elif len(argv) == 2:
            key = READ_ONLY
        else:
            return answer(3, {"refused": "the gate's forced command is --ssh WORDS --key NAME"})
        if key != READ_ONLY and (not NAME.fullmatch(key) or key in HOST_ACCOUNTS):
            return answer(3, {"refused": f"gate key name {key!r}"})
        by = None if key == READ_ONLY else key
        if len(argv[1]) > 8192:
            return answer(3, {"refused": "the request is over 8192 bytes"})
        try:
            argv = shlex.split(argv[1])
        except ValueError as e:
            return answer(3, {"refused": f"the request doesn't parse as shell words: {e}"})
    verb, rest = (argv[0], argv[1:]) if argv else ("", [])
    busy = hooks.get("busy") or busy_units
    now = hooks.get("now") or datetime.now(timezone.utc)
    own = hooks.get("own") or own_by_auditor
    pinned = hooks.get("pinned")
    try:
        if verb in ("run", "replay", "stop", "close", *PARTIES) and by is None:
            raise Refused("this key reads only: a run is started, replayed, stopped or a registration closed with a "
                          "lane's own gate key, which infra registers for it (send infra the public half)")
        if verb == "run":
            doc = gate_run(rest, host, by=by, now=now, token=hooks.get("token") or secrets.token_hex(2),
                           pinned=pinned or pinned_commit(), busy=busy, start=hooks.get("start") or start_unit, own=own,
                           tree=hooks.get("tree") or TREE)
        elif verb == "status" and len(rest) <= 1:
            try:
                pin = pinned or pinned_commit()
            except Refused:
                pin = None
            doc = gate_status(rest, host, busy, pin)
        elif verb == "fetch":
            gate_fetch(rest, host, sink)
            return 0
        elif verb == "replay":
            doc = gate_replay(rest, host, by=by, busy=busy, start=hooks.get("start") or start_unit)
        elif verb == "stop":
            doc = gate_stop(rest, host, by=by, admin=admin, busy=busy, stop=hooks.get("stop") or stop_unit, now=now,
                            own=own)
        elif verb == "close":
            doc = gate_close(rest, host, by=by, busy=busy, close=hooks.get("close") or as_auditor_close, now=now,
                             own=own)
        elif verb in PARTIES:
            return gate_party(verb, rest, host, by=by, now=now, busy=busy, own=own, run=hooks.get("party") or exec_party,
                              sink=sink)
        elif verb == "party_log":
            doc = gate_party_log(rest, host)
        elif verb == "party_key":
            doc = gate_party_key(rest, host, own=own, keygen=hooks.get("keygen") or run_keygen)
        else:
            raise Refused(USAGE)
    except Refused as e:
        return answer(3, {"refused": str(e)})
    return answer(0, doc)


def _auditor(args: list[str]) -> list[str]:
    return ["runuser", "-u", "auditor", "--", "/usr/bin/env", "-i", "PYTHONDONTWRITEBYTECODE=1",
            "HOME=/var/lib/vy-verifier", f"PATH={TREE}/.lean-bin:/usr/bin:/bin", "/usr/bin/python3", "-I",
            str(Path(__file__).resolve()), *args]


def as_auditor_close(node: str) -> dict:
    p = subprocess.run(_auditor(["close", node]), capture_output=True, text=True)
    if p.returncode:
        raise Refused((p.stderr.strip() or p.stdout.strip() or f"close exited {p.returncode}").splitlines()[-1])
    return json.loads(p.stdout)


# --- the run, the replay and the publish (auditor) ----------------------------------------------------------------------------

def gzip_transcripts(src: Path, out: Path) -> list[dict]:
    """Each transcript in `src`, gzipped into `out` with `SHA256SUMS` (sha256sum's form, `sha256sum -c` checks it);
    each one's name and bytes unzipped."""
    out.mkdir(parents=True, exist_ok=True)
    listing, sums = [], []
    for f in sorted(src.glob("*.json")) if src.is_dir() else []:
        gz = out / f"{f.name}.gz"
        with open(f, "rb") as fin, gzip.open(gz, "wb", compresslevel=1) as fout:
            shutil.copyfileobj(fin, fout, 1 << 24)
        sums.append(f"{digest_of(gz, 'sha256')}  ./{gz.name}\n")
        listing.append({"name": f.name, "bytes": f.stat().st_size})
    if listing:
        (out / "SHA256SUMS").write_text("".join(sums))
        (out / "transcripts.json").write_text(json.dumps(listing, indent=1) + "\n")
    return listing


def execute(host: Host, run: str, *, pinned: str, now: Callable[[], datetime],
            argv: list[str] | None = None) -> dict:
    """Run the run's verifier (its request's `argv`), then gzip its transcripts out of scratch and write its `end.json`.
    A SIGTERM (a `stop`, the unit's) goes to the verifier, and the run still ends: its record is what it wrote."""
    req = request_of(host, run)
    d, out, scratch = host.runs / run, host.runs / run / "out", host.scratch / run
    if (d / "end.json").exists():
        raise Refused(f"run {run} has ended")
    out.mkdir(exist_ok=True)
    rc, why = 2, None
    if req["pinned"] != pinned:
        why = f"requested at {req['pinned']}, but this tree is {pinned}: the pin moved, so the run doesn't run"
    else:
        (scratch / "transcripts").mkdir(parents=True, exist_ok=True)
        with open(out / "verifier.log", "wb") as log:
            proc = subprocess.Popen(argv or req["argv"], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    cwd=TREE)
            previous = signal.signal(signal.SIGTERM, lambda *_: proc.terminate())
            try:
                rc = proc.wait()
            finally:
                signal.signal(signal.SIGTERM, previous)
    listing = gzip_transcripts(scratch / "transcripts", out / "transcripts")
    shutil.rmtree(scratch, ignore_errors=True)
    if host.registry.is_file():
        shutil.copyfile(host.registry, out / "registry.json")
    (out / "rc.txt").write_text(f"{rc}\n")
    files = {str(p.relative_to(d)): digest_of(p) for p in sorted(out.rglob("*")) if p.is_file()}
    end = {"run": run, "rc": rc, "why": why, "stopped": read_json(d / "stop.json"), "transcripts": listing,
           "files": files, "ended_utc": utc(now())}
    (d / "end.json").write_text(json.dumps(end, indent=1, sort_keys=True) + "\n")
    return end


def replay_module():
    """`replay.py`, imported from this tree as the host's python3 (no site packages, `-I`) reaches it."""
    for p in (str(TREE), str(REPLAY), str(DIFFTEST)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import replay
    return replay


def replay_main(argv: list[str], log: Path) -> int:
    """`replay.py`'s main in this process, its printout to `log`."""
    replay = replay_module()
    with open(log, "w") as f, contextlib.redirect_stdout(f):
        return replay.main([str(a) for a in argv])


def transcript_sums(verifier_json: dict, unzipped: Path) -> list[dict]:
    """Each session's transcript in `unzipped` against the sha256 and bytes `verifier.json` recorded for it."""
    rows = []
    for k, s in enumerate(verifier_json.get("sessions", [])):
        t = s.get("transcript") or {}
        p = unzipped / Path(t.get("path", f"missing-{k}")).name
        h, n = hashlib.sha256(), 0
        if p.is_file():
            with open(p, "rb") as f:
                while chunk := f.read(1 << 24):
                    h.update(chunk)
                    n += len(chunk)
        rows.append({"session": k, "accepted": s.get("accepted"), "failure": s.get("failure"), "file": p.name,
                     "recorded": {f: t.get(f) for f in ("bytes", "sha256", "events", "error")},
                     "sha256": h.hexdigest() if p.is_file() else None, "bytes": n if p.is_file() else None,
                     "match": p.is_file() and h.hexdigest() == t.get("sha256") and n == t.get("bytes")})
    return rows


def replay_run(host: Host, run: str, *, now: Callable[[], datetime],
               main: Callable[[list[str], Path], int] = replay_main) -> dict:
    """The run's transcripts checked against `SHA256SUMS` and `verifier.json`, unzipped to scratch and replayed through
    both verifiers held to the record (`replay.py transcript --record`), or with none the record itself
    (`replay.py record`), into `replay/`; its `end.json` last, whose `rc` is nonzero too when a transcript isn't the
    recorded one (a gzip off its `SHA256SUMS`, or bytes off `verifier.json`'s), which isn't replayed."""
    request_of(host, run)
    d, out = host.runs / run, host.runs / run / "out"
    rd, scratch = d / "replay", host.scratch / f"{run}-replay"
    record = out / "verifier.json"
    if not record.is_file():
        raise Refused(f"run {run} left no verifier.json: nothing to replay")
    shutil.rmtree(rd, ignore_errors=True)
    rd.mkdir()
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    try:
        tdir = out / "transcripts"
        sums = {}
        if (tdir / "SHA256SUMS").is_file():
            for ln in (tdir / "SHA256SUMS").read_text().splitlines():
                digest, name = ln.split(maxsplit=1)
                f = tdir / Path(name.removeprefix("./")).name
                sums[f.name] = f.is_file() and digest_of(f, "sha256") == digest
        unzipped = []
        for gz in sorted(tdir.glob("*.json.gz")):
            if not sums.get(gz.name):
                continue
            t = scratch / gz.name.removesuffix(".gz")
            with gzip.open(gz, "rb") as fin, open(t, "wb") as fout:
                shutil.copyfileobj(fin, fout, 1 << 24)
            unzipped.append(t)
        rows = transcript_sums(json.loads(record.read_text()), scratch)
        (rd / "transcript-sums.json").write_text(json.dumps(rows, indent=1) + "\n")
        if unzipped:
            rc = main(["transcript", *unzipped, "--record", record, "--out", rd / "replay.json"], rd / "replay.log")
        else:
            rc = main(["record", record, "--out", rd / "replay.json"], rd / "replay.log")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    held = all(sums.values()) and all(r["match"] for r in rows)
    end = {"run": run, "rc": rc or (0 if held else 1), "replay_rc": rc, "kind": "transcript" if unzipped else "record",
           "gz_sums": sums, "sums_match": held, "ended_utc": utc(now())}
    (rd / "end.json").write_text(json.dumps(end, indent=1, sort_keys=True) + "\n")
    return end


def publish(host: Host, run: str, *, put: Callable[[list[str]], int] | None = None) -> int:
    """Put the run's directory in the evidence store as a preserved tree, through the pinned tree's store client, its
    request as the artifact's meta."""
    req = request_of(host, run)
    meta = {k: req.get(k) for k in ("run", "by", "pinned", "peer", "period", "requested_utc")}
    meta["end"] = {k: (read_json(host.runs / run / "end.json") or {}).get(k) for k in ("rc", "ended_utc")}
    meta["replay"] = (read_json(host.runs / run / "replay" / "end.json") or {}).get("rc")
    argv = ["put", "--kind", "verifier-run/v1", "--meta", json.dumps(meta, sort_keys=True), "--tree",
            str(host.runs / run), "--preserve"]
    if put is None:
        sys.path.insert(0, str(RESEARCH))
        from research.store import cli
        return cli.main(argv)
    return put(argv)


def close(host: Host, node: str, argv: list[str] | None = None) -> dict:
    """`verifier.py --close NODE_HEX` on the host's registry, under the Lean verifier as a run's: what it closed."""
    p = subprocess.run(argv or ["/usr/bin/python3", "-I", str(VERIFIER), "--registry", str(host.registry),
                                "--close", node], capture_output=True, text=True, cwd=TREE)
    last = (p.stdout.strip().splitlines() or [""])[-1]
    if p.returncode or not last.startswith("verifier: closed "):
        why = last or (p.stderr.strip().splitlines() or [f"close exited {p.returncode}"])[-1]
        raise Refused(why)
    return json.loads(last.removeprefix("verifier: closed "))


# --- main ---------------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    host = Host()
    if argv[:1] == ["gate"]:
        return gate(argv[1:], host)
    clock = lambda: datetime.now(timezone.utc)  # noqa: E731
    try:
        if argv[:1] == ["run"] and len(argv) == 2:
            end = execute(host, argv[1], pinned=pinned_commit(), now=clock)
            print(json.dumps({"run": end["run"], "rc": end["rc"], "why": end["why"]}))
            return 0 if end["rc"] == 0 else 1
        if argv[:1] == ["replay"] and len(argv) == 2:
            end = replay_run(host, argv[1], now=clock)
            print(json.dumps(end))
            return end["rc"]
        if argv[:1] == ["publish"] and len(argv) == 2:
            return publish(host, argv[1])
        if argv[:1] == ["close"] and len(argv) == 2:
            print(json.dumps(close(host, argv[1])))
            return 0
        if argv[:1] == ["pinned"] and len(argv) == 2:
            print(json.dumps(record_pin(host.pins, argv[1], clock())))
            return 0
        print("usage: host.py gate VERB ... | run RUN | replay RUN | publish RUN | close NODE_HEX | pinned COMMIT",
              file=sys.stderr)
        return 2
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
