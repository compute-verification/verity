"""`verity run` and `verity config`: a deployment configuration's parties, run on this machine, and the file itself
(`verity.core.deploy.config`).

    verity run CONFIG [--party ROLE ...] [--on MACHINE] [--periods N] [--state DIR]
    verity config new --workload W --protocols P,... [--on MACHINE] [--out FILE]
    verity config check CONFIG
    verity config hash CONFIG

`verity run` checks the configuration, refusing any set of subprotocols that is neither remote's nor onsite's
(``composite``, `verity.core.deploy.config.COMPOSITES`: protocol soundness covers only those two composites), then runs
the parties it places on this machine (``--on``, default ``local``), or only the ``--party`` ones, for ``--periods``
periods (default 1), in the shared state ``--state`` (default a new directory under the system's temporary one). Step 0's parties are those of the static table `REGISTRY` names.
It names none since the toy it ran was removed, so `verity run` and `verity config` are refused by name (``no-table``)
until a workload's protocols bring their table; `verity_cli`'s tests run one (`tools/verity/cli/tests/step0.py`). A
party that is an entry there, a function of the configuration, the state and
whether every party is on one machine, runs first, each in a process of its own, one after another in the table's
order, its output in ``STATE/logs/<party>.log``; one that fails stops the run. Then the table's live parties here
(sampled proofs', the core parties) run together in one live run of the deployment layer (the table's ``live``): in
this process when all of them are here, else meeting the rest over TCP at the configuration's ``address`` and
``listen``. Each prints a ``PARTY`` line as it finishes, with its exit code or the name it was refused by.

When the verifier runs here, each period's verdict is its own: the composite of the listed subprotocols (remote or
onsite, its id the line's ``product``), printed as a line and written to ``STATE/record/verdicts.jsonl``:

    VERDICT {"block": I, "coins": C, "components": {NAME: accept|reject|undecided|error|absent}, "product": ID,
             "reason": R, "registration": DIGEST, "run": RUN, "simulated": [...], "verdict": V}

``coins`` is ``simulated`` when the challenger's host is a machine a prover's party is on, else ``development`` (the
keys are made for the run). ``simulated`` names each premise the placement leaves unmet (a party that sends coins on a
developer's party's machine: `Faithful` and `Secret`), and a configured program commit that isn't the running tree's.

The transcripts: an entry's messages are the files it wrote into the state, one line a file
(``STATE/transcripts/<party>.jsonl``: its path, bytes, SHA-512 and kind, ``public`` under ``record/`` and ``public/``,
else ``private``), and the record keeps the public lines (``record/transcripts/``); the live parties' are the
deployment layer's, under ``STATE/runs/``.

The exit code is 0 when every period's verdict accepts and no party was refused, 1 when one rejects, 2 otherwise (a
party that failed or was refused, and the verifier gave no verdict). `verity config check` prints ``ok HASH`` or
``refused NAME: DETAIL`` (exit 2), and `verity config new` writes a configuration that states only its choices, its
program the running tree's commit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.core.deploy import config as K
from verity.core.wire import Refused

from . import settings

ROOT = Path(__file__).resolve().parents[4]
#: step 0's static table: the directory on its import path, and its module (module docstring)
REGISTRY: tuple[Path, str] | None = None
#: what isn't an entry's message in the state: the build, the logs, the transcripts, the run and the tools it names
UNSENT = ("build", "logs", "transcripts", "run.json", "tools.json", "record/transcripts", "record/verdicts.jsonl")
PUBLIC = ("record/", "public/")
VERIFIER = "verifier"
Entry = Callable[[Mapping[str, Any], Path, bool], int]


@dataclass(frozen=True)
class Registry:
    """What `verity run` reads off step 0's table: each subprotocol's declaration and parties, each workload's
    declaration and party (an entry, or the table's mark of a live party), the order a machine's entries run in, and
    the live run (``live(config, state, here) -> RunResult``)."""

    components: Mapping[str, K.Component]
    workloads: Mapping[str, K.Workload]
    workers: Mapping[str, Entry | str]
    parties: Mapping[str, Mapping[str, Entry | str]]
    order: Sequence[str]
    live: Callable[[Mapping[str, Any], Path, Sequence[str]], Any]

    def entries(self, cfg: Mapping[str, Any], party: str) -> list[tuple[str, Entry | str]]:
        """``party``'s entries under ``cfg``: the workload's party, or each listed subprotocol's that declares it, a
        subprotocol's after those it requires; refused by name where a subprotocol has none yet."""
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

    def functions(self, cfg: Mapping[str, Any], party: str) -> list[Entry]:
        """``party``'s entries that run in a process of their own."""
        return [e for _, e in self.entries(cfg, party) if callable(e)]

    def is_live(self, cfg: Mapping[str, Any], party: str) -> bool:
        return any(not callable(e) for _, e in self.entries(cfg, party))


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


def registry(table: tuple[Path, str] | None = None) -> Registry:
    """Step 0's table: ``table``, else `REGISTRY`; refused by name (``no-table``) when there is neither."""
    table = table or REGISTRY
    if table is None:
        raise K.Refused("no-table", "step 0 has no table: the toy it ran was removed, and no workload's protocols "
                                    "have brought theirs yet")
    path, module = table
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    D = importlib.import_module(module)
    return Registry(D.MANIFESTS, {n: w for n, (w, _) in D.WORKLOADS.items()},
                    {n: e for n, (_, e) in D.WORKLOADS.items()}, D.PARTIES, D.ORDER, D.live)


def load(path: Path, reg: Registry) -> dict[str, Any]:
    """The configuration at ``path``, checked (`Refused` by name)."""
    return K.check(K.parse(path.read_text()), reg.components, reg.workloads)


# --- placement ---------------------------------------------------------------------------------------------------------

def placed(cfg: Mapping[str, Any]) -> dict[str, str]:
    return {p["name"]: p["machine"] for p in cfg["party"]}


def plan(cfg: Mapping[str, Any], reg: Registry, machine: str, wanted: Sequence[str] = ()) -> list[str]:
    """The parties this run runs, the entries first in the table's order: ``wanted``, each placed on ``machine``, else
    every party the configuration places there; each with its entries (`Registry.entries`)."""
    at = placed(cfg)
    if unknown := [p for p in wanted if p not in at]:
        raise K.Refused("unknown-party", f"{', '.join(unknown)} (the configuration's: {', '.join(at)})")
    if elsewhere := [p for p in wanted if at[p] != machine]:
        raise K.Refused("machine", ", ".join(f"{p} runs on {at[p]}, not {machine} (--on)" for p in elsewhere))
    chosen = list(dict.fromkeys(wanted)) or [p for p, m in at.items() if m == machine]
    if not chosen:
        raise K.Refused("machine", f"the configuration places no party on {machine} (its machines: "
                                   f"{', '.join(sorted(set(at.values())))})")
    for p in chosen:
        reg.entries(cfg, p)
    rank = {p: i for i, p in enumerate(reg.order)}
    return sorted(chosen, key=lambda p: (rank.get(p, len(rank)), list(at).index(p)))


def unmet(cfg: Mapping[str, Any], reg: Registry, head: str | None) -> list[str]:
    """The premises the run's placement leaves unmet (module docstring): each party that sends coins on a developer's
    party's machine, and a configured program commit other than the running tree's ``head``."""
    listed = [reg.components[c] for c in cfg["protocols"]]
    parties, messages, channels = K.declared(listed, reg.workloads[cfg["workload"]["name"]])
    at = placed(cfg)
    sources = sorted({ch.source for ch in channels.values() if any(messages[m] == "coins" for m in ch.messages)})
    out = []
    for p in sources:
        if shared := sorted(q for q, m in at.items() if m == at[p] and parties[q] == "developer"):
            out.append(f"{p} shares {at[p]} with the developer's {', '.join(shared)}: Faithful and Secret unmet")
    if head != cfg["program"]["commit"]:
        out.append(f"the program is the running tree's {head}, not the configured {cfg['program']['commit']}")
    return out


# --- the state ---------------------------------------------------------------------------------------------------------

def _unsent(rel: str) -> bool:
    return any(rel == u or rel.startswith(u + "/") for u in UNSENT)


def files(state: Path) -> dict[str, tuple[int, int]]:
    """Every regular file in ``state`` but `UNSENT`'s: its size and modification time, by path."""
    out = {}
    for d, _, names in os.walk(state):
        for n in names:
            f = Path(d) / n
            rel = f.relative_to(state).as_posix()
            if f.is_symlink() or not f.is_file() or _unsent(rel):
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


def head() -> str | None:
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def begin(cfg: Mapping[str, Any], state: Path, periods: int, machine: str, hooks: Iterable[str]) -> dict[str, Any]:
    """The run's ``run.json`` (module docstring), written before any party starts, or the one in ``state`` when it is
    this configuration's run at ``periods`` (another machine's parties)."""
    digest = K.digest(cfg)
    f = state / "run.json"
    if f.exists():
        run = json.loads(f.read_text())
        if run["configuration"] != digest or run["periods"] != periods:
            raise K.Refused("state", f"{state} holds the run of configuration {run['configuration'][:16]}… at "
                                     f"{run['periods']} periods, not {digest[:16]}… at {periods}")
        return run
    run = {"run": settings.run_id(), "periods": periods, "configuration": digest,
           "hooks": sorted(hooks), "parties": placed(cfg), "machine": machine}
    state.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(run, indent=1, sort_keys=True))
    (state / "config.json").write_text(json.dumps(cfg, indent=1, sort_keys=True))
    return run


# --- running the parties -----------------------------------------------------------------------------------------------

def run_party(cfg: Mapping[str, Any], reg: Registry, state: Path, party: str, local: bool) -> int:
    """``party``'s entries here, one after another; the first that fails stops it."""
    for entry in reg.functions(cfg, party):
        if rc := entry(cfg, state, local):
            return rc
    return 0


class Processes:
    """Each entry in a process of its own (``python -m verity_cli.run party CONFIG STATE PARTY``), its output in
    ``STATE/logs/<party>.log``."""

    def __init__(self, config: Path, state: Path) -> None:
        self.config, self.state = config.resolve(), state
        self.argv = [sys.executable, "-m", "verity_cli.run", "party", str(self.config), str(state)]

    def run(self, party: str) -> int:
        (self.state / "logs").mkdir(parents=True, exist_ok=True)
        with open(self.state / "logs" / f"{party}.log", "w") as log:
            return subprocess.run([*self.argv, party], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode


class InProcess:
    """Each entry in this process: for a run whose entries are functions of the test that runs it."""

    def __init__(self, cfg: Mapping[str, Any], reg: Registry, state: Path, local: bool) -> None:
        self.cfg, self.reg, self.state, self.local = cfg, reg, state, local

    def run(self, party: str) -> int:
        return run_party(self.cfg, self.reg, self.state, party, self.local)


def say(what: str, body: Mapping[str, Any]) -> None:
    print(f"{what} " + json.dumps(body, sort_keys=True), flush=True)


def _failed(state: Path, party: str, e: Exception) -> None:
    (state / "logs").mkdir(parents=True, exist_ok=True)
    (state / "logs" / f"{party}.error").write_text(f"{type(e).__name__}: {e}")


def execute(cfg: Mapping[str, Any], reg: Registry, state: Path, chosen: Sequence[str], runner: Any,
            premises: Sequence[str]) -> int:
    """``chosen``'s entries run in ``state`` by ``runner`` (`Processes` or `InProcess`), each one's transcript
    written, then its live parties' run, then the verdicts when the verifier is among them (module docstring); the
    exit code."""
    run = json.loads((state / "run.json").read_text())
    for p in (p for p in chosen if reg.functions(cfg, p)):
        before = files(state)
        try:
            rc = runner.run(p)
        except Exception as e:  # noqa: BLE001  (an entry that raises failed, and the run stops)
            _failed(state, p, e)
            rc = 1
        transcribe(state, p, changed(before, files(state)))
        say("PARTY", {"party": p, "exit": rc})
        if rc:
            return 2
    here = [p for p in chosen if reg.is_live(cfg, p)]
    if not here:
        return 0
    try:
        r = reg.live(cfg, state, here)
    except (K.Refused, Refused) as e:
        print(f"verity run: refused {e.name}: {e.detail}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"verity run: refused live: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    for p, o in sorted(r.outcomes.items()):
        say("PARTY", {"party": p, "refused": o["refused"]})
        if o["refused"] is not None:
            print(f"{p}: refused {o['refused']}: {json.dumps(o['outcome'], sort_keys=True, default=str)[:2000]}",
                  file=sys.stderr)
    refused = any(o["refused"] is not None for o in r.outcomes.values())
    if VERIFIER not in here:
        return 2 if refused else 0
    lines = [{**x, "run": run["run"], "simulated": list(premises)}
             for x in (r.outcomes[VERIFIER]["outcome"] or {}).get("verdicts", [])]
    for line in lines:
        say("VERDICT", line)
    (state / "record").mkdir(parents=True, exist_ok=True)
    (state / "record" / "verdicts.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in lines))
    verdicts = {x["verdict"] for x in lines}
    if "reject" in verdicts:
        return 1
    return 0 if verdicts == {"accept"} and len(lines) == run["periods"] and not refused else 2


def hooks_of(cfg: Mapping[str, Any], reg: Registry) -> set[str]:
    return set().union(*(reg.components[c].needs for c in cfg["protocols"]))


def run_main(argv: Sequence[str], reg: Registry | None = None, in_process: bool = False) -> int:
    ap = argparse.ArgumentParser(prog="verity run", description="Run a deployment configuration's parties on this "
                                 "machine (module docstring of verity_cli.run).")
    ap.add_argument("config", metavar="CONFIG")
    ap.add_argument("--party", action="append", default=[], metavar="ROLE", help="only this party (repeatable)")
    ap.add_argument("--on", dest="machine", default=K.LOCAL, metavar="MACHINE", help="this machine's name in the configuration "
                    f"(default {K.LOCAL})")
    ap.add_argument("--periods", type=int, default=1, metavar="N", help="the run's periods (default 1)")
    ap.add_argument("--state", metavar="DIR", help="the parties' shared state (default a new temporary directory)")
    a = ap.parse_args(argv)
    if a.periods < 1:
        ap.error(f"--periods is at least 1, not {a.periods}")
    state = Path(a.state or tempfile.mkdtemp(prefix="verity-run-")).resolve()
    try:
        reg = reg or registry()
        cfg = load(Path(a.config), reg)
        chosen = plan(cfg, reg, a.machine, a.party)
        begin(cfg, state, a.periods, a.machine, hooks_of(cfg, reg))
    except K.Refused as e:
        print(f"verity run: refused {e.name}: {e.detail}", file=sys.stderr)
        return 2
    say("RUN", {"state": str(state), "parties": chosen, "configuration": K.digest(cfg), "periods": a.periods})
    runner = (InProcess(cfg, reg, state, len(set(placed(cfg).values())) == 1) if in_process else
              Processes(Path(a.config), state))
    return execute(cfg, reg, state, chosen, runner, unmet(cfg, reg, head()))


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
    p.add_argument("--protocols", required=True, help="the subprotocols, comma-separated: remote's (sampled-proofs,pouw) "
                   "or onsite's")
    p.add_argument("--on", dest="machine", default=K.LOCAL, metavar="MACHINE", help=f"every party's machine (default {K.LOCAL})")
    p.add_argument("--commit", help="the program's commit (default the running tree's HEAD)")
    p.add_argument("--out", help="write it here (default stdout)")
    for name in ("check", "hash"):
        sub.add_parser(name).add_argument("config", metavar="CONFIG")
    a = ap.parse_args(argv)
    try:
        reg = reg or registry()
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
