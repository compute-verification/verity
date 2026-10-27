"""`verity run` and `verity config`: a deployment configuration's parties, run on this machine, and the file itself
(`verity.core.deploy.config`; `note:memory-accounting/20261026T2305Z-draft-deployment-api-v3`, addenda 23:46Z to 00:30Z).

    verity run CONFIG [--party ROLE ...] [--machine NAME] [--periods N] [--replay DIR] [--state DIR]
    verity config new --workload W --protocols P,... [--machine NAME] [--out FILE]
    verity config check CONFIG
    verity config hash CONFIG

`verity run` checks the configuration, then runs the parties it places on this machine (``--machine``, default
``local``), or only the ``--party`` ones, for ``--periods`` periods (default 1), in the shared state ``--state`` (default
a new directory under the system's temporary one). Step 0's parties are those of the static table `REGISTRY` names
(`verity_all_protocols.deployment`, beside the sampled-proofs parties), each a function of the configuration, the state
and whether every party is on one machine. Each runs in a process of its own, one after another in the table's order,
its output in ``STATE/logs/<party>.log``; a party that serves the others (the challenger) starts first and stops once
they are done. A party that fails stops the run, and the verifier's verdict is composed anyway, so a component that
didn't run reads ``absent`` and the product rejects.

When the verifier runs here, each component's verifier decides that component alone (it sees ``protocols`` as just
its own), and each period's verdict is the product over exactly the listed components (the table's ``compose``),
printed as a line and written to ``STATE/record/verdicts.jsonl``:

    VERDICT {"block": I, "coins": C, "components": {NAME: accept|reject|undecided|error|absent}, "product": ID,
             "reason": R, "run": RUN, "simulated": [...], "verdict": V}

``coins`` is ``simulated`` when a party that sends coins shares a machine with a developer's party (its host is then
the prover's, so `Faithful` and `Secret` are unmet), else ``development`` (step 0's keys are made for the run);
``simulated`` names each such premise, and a configured program commit that isn't the running tree's.

Step 0's transcripts: what each party wrote into the state, which is how step 0's parties send their messages, one line
a file (``STATE/transcripts/<party>.jsonl``: its path, bytes, SHA-512 and kind, ``public`` under ``record/`` and
``public/``, else ``private``); the record keeps the public lines (``record/transcripts/``). ``--replay DIR`` starts
from an earlier run's state without what the parties run now wrote in it (their transcripts' files), so a new verifier
reruns over a recorded run.

The exit code is 0 when every period's verdict accepts, 1 when one rejects, 2 otherwise (and when a party failed with
no verifier here). `verity config check` prints ``ok HASH`` or ``refused NAME: DETAIL`` (exit 2), and `verity config
new` writes a configuration that states only its choices, its program the running tree's commit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.core.deploy import config as K

ROOT = Path(__file__).resolve().parents[4]
#: step 0's static table: the directory on its import path, and its module
REGISTRY = (ROOT / "benchmarks" / "all_protocols", "verity_all_protocols.deployment")
#: what isn't a party's message in the state: the build, the logs, the transcripts, the run and the tools it names
UNSENT = ("build", "logs", "transcripts", "run.json", "tools.json", "record/transcripts", "record/verdicts.jsonl")
PUBLIC = ("record/", "public/")
VERIFIER = "verifier"
Entry = Callable[[Mapping[str, Any], Path, bool], int]


@dataclass(frozen=True)
class Registry:
    """What `verity run` reads off step 0's table: each subprotocol's declaration and parties, each workload's
    declaration and party, the order a machine's parties run in, the parties that serve the others, and the verdict."""

    components: Mapping[str, K.Component]
    workloads: Mapping[str, K.Workload]
    workers: Mapping[str, Entry]
    parties: Mapping[str, Mapping[str, Entry]]
    order: Sequence[str]
    services: Mapping[str, Any]
    compose: Callable[[Mapping[str, Any], Path, int], dict[str, Any]]

    def entries(self, cfg: Mapping[str, Any], party: str) -> list[tuple[str, Entry]]:
        """``party``'s entries under ``cfg``: the workload's party, or each listed component's that declares it, a
        component's after those it requires; refused by name where a component has none yet."""
        w = cfg["workload"]["name"]
        if party in self.workloads[w].parties:
            return [(w, self.workers[w])]
        out = []
        for c in _required_first(cfg["protocols"], self.components):
            if party in self.components[c].parties:
                if party not in self.parties.get(c, {}):
                    raise K.Refused("no-party", f"{c} has no step-0 {party} yet: verity run runs "
                                                f"{', '.join(sorted(self.parties))}")
                out.append((c, self.parties[c][party]))
        return out


def _required_first(names: Sequence[str], components: Mapping[str, K.Component]) -> list[str]:
    out: list[str] = []

    def visit(n: str) -> None:
        if n not in out:
            for r in components[n].requires:
                visit(r)
            out.append(n)
    for n in names:
        visit(n)
    return out


def registry() -> Registry:
    """Step 0's table (`REGISTRY`)."""
    path, module = REGISTRY
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    D = importlib.import_module(module)
    return Registry(D.MANIFESTS, {n: w for n, (w, _) in D.WORKLOADS.items()},
                    {n: e for n, (_, e) in D.WORKLOADS.items()}, D.PARTIES, D.ORDER, D.SERVICES, D.compose)


def load(path: Path, reg: Registry) -> dict[str, Any]:
    """The configuration at ``path``, checked (`Refused` by name)."""
    return K.check(K.parse(path.read_text()), reg.components, reg.workloads)


# --- placement ---------------------------------------------------------------------------------------------------------

def placed(cfg: Mapping[str, Any]) -> dict[str, str]:
    return {p["name"]: p["machine"] for p in cfg["party"]}


def plan(cfg: Mapping[str, Any], reg: Registry, machine: str, wanted: Sequence[str] = ()) -> list[str]:
    """The parties this run runs, in the table's order: ``wanted``, each placed on ``machine``, else every party the
    configuration places there; each with its entries (`Registry.entries`)."""
    at = placed(cfg)
    if unknown := [p for p in wanted if p not in at]:
        raise K.Refused("unknown-party", f"{', '.join(unknown)} (the configuration's: {', '.join(at)})")
    if elsewhere := [p for p in wanted if at[p] != machine]:
        raise K.Refused("machine", ", ".join(f"{p} runs on {at[p]}, not {machine} (--machine)" for p in elsewhere))
    chosen = list(dict.fromkeys(wanted)) or [p for p, m in at.items() if m == machine]
    if not chosen:
        raise K.Refused("machine", f"the configuration places no party on {machine} (its machines: "
                                   f"{', '.join(sorted(set(at.values())))})")
    for p in chosen:
        reg.entries(cfg, p)
    rank = {p: i for i, p in enumerate(reg.order)}
    return sorted(chosen, key=lambda p: (rank.get(p, len(rank)), list(at).index(p)))


def unmet(cfg: Mapping[str, Any], reg: Registry, head: str | None) -> tuple[str, list[str]]:
    """The run's coins (module docstring) and the premises its placement leaves unmet: each party that sends coins on a
    developer's party's machine, and a configured program commit other than the running tree's ``head``."""
    listed = [reg.components[c] for c in cfg["protocols"]]
    parties, messages, channels = K.declared(listed, reg.workloads[cfg["workload"]["name"]])
    at = placed(cfg)
    sources = sorted({ch.source for ch in channels.values() if any(messages[m] == "coins" for m in ch.messages)})
    out = []
    for p in sources:
        if shared := sorted(q for q, m in at.items() if m == at[p] and parties[q] == "developer"):
            out.append(f"{p} shares {at[p]} with the developer's {', '.join(shared)}: Faithful and Secret unmet")
    coins = "simulated" if out else "development"
    if head != cfg["program"]["commit"]:
        out.append(f"the program is the running tree's {head}, not the configured {cfg['program']['commit']}")
    return coins, out


# --- the state ---------------------------------------------------------------------------------------------------------

def _unsent(rel: str) -> bool:
    return any(rel == u or rel.startswith(u + "/") for u in UNSENT)


def files(state: Path, skip: Iterable[str] = ()) -> dict[str, tuple[int, int]]:
    """Every regular file in ``state`` but `UNSENT` and ``skip``'s trees: its size and modification time, by path."""
    skip = tuple(skip)
    out = {}
    for d, dirs, names in os.walk(state):
        for n in names:
            f = Path(d) / n
            rel = f.relative_to(state).as_posix()
            if f.is_symlink() or not f.is_file() or _unsent(rel) or any(rel.startswith(s + "/") for s in skip):
                continue
            st = f.stat()
            out[rel] = (st.st_size, st.st_mtime_ns)
    return out


def _sha512(path: Path) -> str:
    h = hashlib.sha512()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def transcribe(state: Path, party: str, written: Iterable[str]) -> None:
    """``party``'s transcript lines for the files it ``written`` (module docstring), appended; the public ones to the
    record's copy too."""
    lines = []
    for rel in sorted(written):
        f = state / rel
        if f.is_file():
            lines.append({"file": rel, "bytes": f.stat().st_size, "sha512": _sha512(f),
                          "kind": "public" if rel.startswith(PUBLIC) else "private"})
    for d, keep in ((state / "transcripts", lines), (state / "record" / "transcripts",
                                                     [x for x in lines if x["kind"] == "public"])):
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{party}.jsonl", "a") as f:
            f.writelines(json.dumps(x, sort_keys=True) + "\n" for x in keep)


def changed(before: Mapping[str, tuple[int, int]], after: Mapping[str, tuple[int, int]]) -> list[str]:
    return [f for f, v in after.items() if before.get(f) != v]


def replay(source: Path, state: Path, parties: Sequence[str]) -> None:
    """``state`` as the earlier run ``source`` left it, without the files ``parties`` wrote in it (their transcripts),
    their transcripts, the verdicts and the logs. The build stays in ``source``, whose tools ``tools.json`` names."""
    if not (source / "run.json").exists():
        raise K.Refused("replay", f"{source} is no run's state (no run.json)")
    if state.exists() and any(state.iterdir()):
        raise K.Refused("replay", f"--state {state} is not empty: a replay starts a state of its own")
    drop = set()
    for p in parties:
        t = source / "transcripts" / f"{p}.jsonl"
        drop |= {json.loads(x)["file"] for x in t.read_text().splitlines() if x.strip()} if t.exists() else set()
    skip = {"build", "logs", "record/verdicts.jsonl", *(f"{d}/{p}.jsonl" for d in ("transcripts", "record/transcripts")
                                                         for p in parties)}

    def ignore(d: str, names: list[str]) -> set[str]:
        rel = Path(d).relative_to(source)
        return {n for n in names if (rel / n).as_posix() in skip | drop}
    shutil.copytree(source, state, symlinks=True, ignore=ignore, dirs_exist_ok=True)


def head() -> str | None:
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def begin(cfg: Mapping[str, Any], state: Path, periods: int, machine: str, hooks: Iterable[str]) -> dict[str, Any]:
    """The run's ``run.json`` (module docstring), written before any party starts, or the one in ``state`` when it is
    this configuration's run at ``periods`` (another machine's parties, or a replay)."""
    digest = K.digest(cfg)
    f = state / "run.json"
    if f.exists():
        run = json.loads(f.read_text())
        if run["configuration"] != digest or run["periods"] != periods:
            raise K.Refused("state", f"{state} holds the run of configuration {run['configuration'][:16]}… at "
                                     f"{run['periods']} periods, not {digest[:16]}… at {periods}")
        return run
    run = {"run": os.environ.get("RESEARCH_RUN_ID") or "local", "periods": periods, "configuration": digest,
           "hooks": sorted(hooks), "parties": placed(cfg), "machine": machine}
    state.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(run, indent=1, sort_keys=True))
    (state / "config.json").write_text(json.dumps(cfg, indent=1, sort_keys=True))
    return run


# --- running the parties -----------------------------------------------------------------------------------------------

def run_party(cfg: Mapping[str, Any], reg: Registry, state: Path, party: str, local: bool) -> int:
    """``party``'s entries here, one after another. Each of the verifier's sees the configuration with ``protocols``
    its own component alone, which it decides (the verdict over all of them is `execute`'s), and runs whatever the one
    before it returned; for any other party the first that fails stops it."""
    rc = 0
    for name, entry in reg.entries(cfg, party):
        got = entry({**cfg, "protocols": [name]} if party == VERIFIER else cfg, state, local)
        rc = rc or got
        if got and party != VERIFIER:
            break
    return rc


class Processes:
    """Each party in a process of its own (``python -m verity_cli.run party CONFIG STATE PARTY``), its output in
    ``STATE/logs/<party>.log``."""

    def __init__(self, config: Path, state: Path) -> None:
        self.config, self.state = config.resolve(), state
        self.argv = [sys.executable, "-m", "verity_cli.run", "party", str(self.config), str(state)]

    def _log(self, party: str) -> Path:
        (self.state / "logs").mkdir(parents=True, exist_ok=True)
        return self.state / "logs" / f"{party}.log"

    def run(self, party: str) -> int:
        with open(self._log(party), "w") as log:
            return subprocess.run([*self.argv, party], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode

    def start(self, party: str, ready: str) -> subprocess.Popen:
        log = open(self._log(party), "w")
        p = subprocess.Popen([*self.argv, party], cwd=ROOT, stdout=subprocess.PIPE, stderr=log, text=True)
        for line in p.stdout:
            log.write(line)
            log.flush()
            if line.startswith(ready):
                print(line, end="", flush=True)
                break
        else:
            log.close()
            raise RuntimeError(f"{party} exited {p.wait()} before it was ready ({self._log(party)})")
        threading.Thread(target=lambda: (log.writelines(p.stdout), log.close()), daemon=True).start()
        return p

    def wait(self, handle: subprocess.Popen) -> int:
        return handle.wait()


class InProcess:
    """Each party in this process: for a run whose entries are functions of the test that runs it."""

    def __init__(self, cfg: Mapping[str, Any], reg: Registry, state: Path, local: bool) -> None:
        self.cfg, self.reg, self.state, self.local = cfg, reg, state, local

    def run(self, party: str) -> int:
        return run_party(self.cfg, self.reg, self.state, party, self.local)

    def start(self, party: str, ready: str) -> int:
        return self.run(party)

    def wait(self, handle: int) -> int:
        return handle


def say(what: str, body: Mapping[str, Any]) -> None:
    print(f"{what} " + json.dumps(body, sort_keys=True), flush=True)


def _failed(state: Path, party: str, e: Exception) -> None:
    (state / "logs").mkdir(parents=True, exist_ok=True)
    (state / "logs" / f"{party}.error").write_text(f"{type(e).__name__}: {e}")


def execute(cfg: Mapping[str, Any], reg: Registry, state: Path, chosen: Sequence[str], runner: Any, coins: str,
            premises: Sequence[str]) -> int:
    """``chosen`` run in ``state`` by ``runner`` (`Processes` or `InProcess`), each one's transcript written, then the
    verdicts when the verifier is among them (module docstring); the exit code. A service's own directory
    (``STATE/<party>/``) is its alone, so what it writes there while the others run is in its transcript."""
    run = json.loads((state / "run.json").read_text())
    services = [p for p in chosen if p in reg.services]
    started, ready, failed = {}, {}, []
    try:
        for p in services:
            before = files(state)
            try:
                started[p] = runner.start(p, reg.services[p].ready)
            except Exception as e:  # noqa: BLE001: a service that never got ready failed, and the run stops
                _failed(state, p, e)
                failed.append(p)
            ready[p] = files(state)
            transcribe(state, p, changed(before, ready[p]))
            say("PARTY", {"party": p, "ready": p in started})
            if p in failed:
                break
        for p in (p for p in chosen if p not in reg.services and not failed):
            before = files(state, services)
            try:
                rc = runner.run(p)
            except Exception as e:  # noqa: BLE001: a party that raises failed, and the verdict is still composed
                _failed(state, p, e)
                rc = 1
            transcribe(state, p, changed(before, files(state, services)))
            say("PARTY", {"party": p, "exit": rc})
            if rc and p != VERIFIER:
                failed.append(p)
    finally:
        for p, handle in started.items():
            reg.services[p].stop(state)
            rc = runner.wait(handle)
            transcribe(state, p, [f for f in changed(ready[p], files(state)) if f.startswith(p + "/")])
            say("PARTY", {"party": p, "exit": rc})
    if VERIFIER not in chosen:
        return 2 if failed else 0
    lines = []
    for i in range(run["periods"]):
        try:
            got = reg.compose(cfg, state, i)
        except Exception as e:  # noqa: BLE001: no product to run is the period's error, never a verdict
            got = {"components": {c: "error" for c in cfg["protocols"]}, "product": None, "verdict": "error",
                   "reason": f"the product: {type(e).__name__}: {e}"[:2000]}
        line = {"block": i, "coins": coins, **got, "run": run["run"], "simulated": list(premises)}
        lines.append(line)
        say("VERDICT", line)
    (state / "record").mkdir(parents=True, exist_ok=True)
    (state / "record" / "verdicts.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in lines))
    verdicts = {x["verdict"] for x in lines}
    return 0 if verdicts == {"accept"} and not failed else 1 if "reject" in verdicts else 2


def hooks_of(cfg: Mapping[str, Any], reg: Registry) -> set[str]:
    return set().union(*(reg.components[c].needs for c in cfg["protocols"]))


def run_main(argv: Sequence[str], reg: Registry | None = None, in_process: bool = False) -> int:
    ap = argparse.ArgumentParser(prog="verity run", description="Run a deployment configuration's parties on this "
                                 "machine (module docstring of verity_cli.run).")
    ap.add_argument("config", metavar="CONFIG")
    ap.add_argument("--party", action="append", default=[], metavar="ROLE", help="only this party (repeatable)")
    ap.add_argument("--machine", default=K.LOCAL, metavar="NAME", help="this machine's name in the configuration "
                    f"(default {K.LOCAL})")
    ap.add_argument("--periods", type=int, default=1, metavar="N", help="the run's periods (default 1)")
    ap.add_argument("--replay", metavar="DIR", help="an earlier run's state, without what these parties wrote in it")
    ap.add_argument("--state", metavar="DIR", help="the parties' shared state (default a new temporary directory)")
    a = ap.parse_args(argv)
    if a.periods < 1:
        ap.error(f"--periods is at least 1, not {a.periods}")
    reg = reg or registry()
    state = Path(a.state or tempfile.mkdtemp(prefix="verity-run-")).resolve()
    try:
        cfg = load(Path(a.config), reg)
        chosen = plan(cfg, reg, a.machine, a.party)
        if a.replay:
            replay(Path(a.replay).resolve(), state, chosen)
        begin(cfg, state, a.periods, a.machine, hooks_of(cfg, reg))
    except K.Refused as e:
        print(f"verity run: refused {e.name}: {e.detail}", file=sys.stderr)
        return 2
    coins, premises = unmet(cfg, reg, head())
    say("RUN", {"state": str(state), "parties": chosen, "configuration": K.digest(cfg), "periods": a.periods})
    runner = (InProcess(cfg, reg, state, len(set(placed(cfg).values())) == 1) if in_process else
              Processes(Path(a.config), state))
    return execute(cfg, reg, state, chosen, runner, coins, premises)


def party_main(argv: Sequence[str]) -> int:
    """One party's entries in this process (`Processes`): ``party CONFIG STATE PARTY``."""
    config, state, party = argv
    reg = registry()
    cfg = load(Path(config), reg)
    return run_party(cfg, reg, Path(state), party, len(set(placed(cfg).values())) == 1)


# --- verity config -----------------------------------------------------------------------------------------------------

HEADER = ("The deployment configuration of {protocols} on the {workload} workload (`verity config new`), every party",
          "on {machine}: `verity run` this file. Parameters it doesn't state are their declarations' defaults.")


def config_main(argv: Sequence[str], reg: Registry | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity config", description="Write, check or hash a deployment configuration.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new", help="a configuration that states only its choices")
    p.add_argument("--workload", required=True)
    p.add_argument("--protocols", required=True, help="the components, comma-separated")
    p.add_argument("--machine", default=K.LOCAL, help=f"every party's machine (default {K.LOCAL})")
    p.add_argument("--commit", help="the program's commit (default the running tree's HEAD)")
    p.add_argument("--out", help="write it here (default stdout)")
    for name in ("check", "hash"):
        sub.add_parser(name).add_argument("config", metavar="CONFIG")
    a = ap.parse_args(argv)
    reg = reg or registry()
    try:
        if a.cmd == "new":
            protocols = [x for x in a.protocols.split(",") if x]
            cfg = K.new(a.workload, protocols, reg.components, reg.workloads, a.commit or head() or "", a.machine)
            text = K.render(cfg, [h.format(protocols=" and ".join(protocols), workload=a.workload, machine=a.machine)
                                  for h in HEADER])
            if a.out:
                Path(a.out).write_text(text)
            else:
                sys.stdout.write(text)
            return 0
        cfg = load(Path(a.config), reg)
    except K.Refused as e:
        print(f"refused {e.name}: {e.detail}")
        return 2
    print(f"ok {K.digest(cfg)}" if a.cmd == "check" else K.digest(cfg))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "party":
        return party_main(args[1:])
    raise SystemExit("python -m verity_cli.run party CONFIG STATE PARTY (verity run CONFIG runs the parties)")


if __name__ == "__main__":
    raise SystemExit(main())
