"""`verity run` and `verity config`: a configuration's parties, run on this host, and the file itself
(`verity.config`).

    verity run CONFIG|NAME [--party ROLE ...] [--on MACHINE ...] [--periods N] [--state DIR] [--bindings FILE] [--dry-run]
    verity config new --workload W --protocols P,... [--on MACHINE] [--out FILE]
    verity config check CONFIG
    verity config hash CONFIG

`verity run` reads CONFIG, a configuration's path, or NAME, an application's, `applications/NAME.toml` (a - in NAME is
a _; one that is neither is refused by name, ``application``, with the applications' names). It checks the
configuration, refusing any set of subprotocols that is neither a composite's nor one of onsite's smaller sets
(``composite``, `verity.config.COMPOSITES`: onsite and remote, whose work is Pearl+'s run verdict, the only two protocol
soundness covers; `verity.config.UNACCEPTED`; H1's development composite is refused), then runs
the parties it places on the machines this host stands for (``--on``, once for each, default ``local``, and for NAME each
of the application's machines), or only the ``--party`` ones, for ``--periods`` periods (default 1), in the shared state
``--state`` (default a new directory under the system's temporary one). The parties are those of the static table
`REGISTRY` names: the served window's (`tools/verity/cli/verity_cli/served_table.py`);
`verity_cli`'s tests also run the toy's (`tests/tools/verity/cli/step0.py`). With no table, `verity run` and
`verity config` are refused by name (``no-table``).

``--bindings FILE`` gives the bindings: the hosts, addresses and gate commands the table's pieces reach the
parties by, which the configuration never states, so binding another host changes no hash. It is TOML, a table per
party and in it a table per listed subprotocol that declares that party, of strings and integers
(``[memory-challenger.pous] address = "HOST:PORT"``); one naming another party or subprotocol is refused by name
(``binding``). They are kept beside the run (``STATE/bindings.json``), never in its record.

``--dry-run`` runs nothing: it prints a ``PLAN`` line for each party it would run (without ``--on``, the ``--party``
ones wherever they are placed, else every party of the configuration), its machine and each entry (a table's entry
describes itself: its subprotocol, its scripts, the variables it sets, the bindings it lacks), a ``CHANNEL`` line for
each channel, and a ``DRY`` line of the counts and every unbound binding. A listed subprotocol with no entry for a party
it declares is refused ``no-party``, an entry whose script isn't in the tree ``no-piece``, and, with ``--bindings``, a
binding an entry needs that the file lacks ``unbound``.

Before the first period, where Flock's Lean verifier isn't built (`verity verify proof`'s executable), it is:
`tools/verity/lean/setup.sh`, then `lake build` in its package. Before any party starts, each party here gets a key
made for the run, an LMS key (`RUN_KEY`): its seed in ``STATE/parties/<party>/key.json``, never sent, and its public
half and that half's SHA-512 in ``STATE/public/<party>/key.json``. When a party runs in a process of its own, the run
also gets an archive of the commit that runs, ``STATE/tree.tar`` (its files under ``<commit>/``), which `run.json` names
(``tree``) beside the commit (``program``) and each party's process reads as ``VERITY_PROGRAM`` and ``VERITY_TREE``.

A party the table starts first (its ``STARTS``, `Start`: the bindings it writes, by subprotocol, its ``serve`` and
``stop``, and the seconds `verity run` waits, ``within``) starts before every other party here, in a process of its own
(``start CONFIG STATE PARTY MACHINE...``), its output in ``STATE/logs/<party>.log``; with ``--bindings`` giving every
binding it writes it isn't started, since it runs where they say, and giving only some is refused (``binding``).
`verity run` waits for what it writes, ``STATE/public/<party>/bindings.json`` (written whole, by a rename: a table per
subprotocol of strings and integers), and adds it to the run's bindings before any party that reads them starts; one
that ends first, writes other keys or writes none within ``within`` seconds is refused by name (``bindings``), and the
run stops. What it gives the verifier and the record goes under ``STATE/public/<party>/``, its own state under
``STATE/parties/<party>/``. It lives until the run ends: then, after a failure too, `verity run` stops it (SIGTERM to
its process, whose handler calls ``stop``), and prints its ``PARTY`` line with its exit code; the files it wrote are its
transcript. ``--dry-run`` prints what each one writes (``writes``) and counts none of it unbound.

A party that is an entry of the table, a function of the
configuration, the state and whether every party is on this host, runs next, each in a process of its own, one after
another in the table's order, its output in ``STATE/logs/<party>.log``; one that fails stops the run. Then the table's live parties here
(SPOT's, the core parties) run together in one live run of the deployment layer (the table's ``live``): in
this process when all of them are here, else meeting the rest over TCP at the configuration's ``address`` and
``listen``. Each prints a ``PARTY`` line as it finishes, with its exit code or the name it was refused by.

When the verifier runs here, each period's verdict is its own: the composite of the listed subprotocols (remote or
onsite, its id the line's ``product``), or for one of onsite's smaller sets, under that set's id, ``undecided``, since no
composite takes it, unless the table's line says who combined its verdict (``combined_by``, printed beside it), printed
as a line and written to ``STATE/record/verdicts.jsonl``:

    VERDICT {"block": I, "coins": C, "coins_on": {NAME: MACHINE}, "components": {NAME: accept|reject|undecided|error|absent},
             "configuration": HASH, "product": ID, "program": COMMIT, "reason": R, "registration": DIGEST, "run": RUN,
             "simulated": [...], "verdict": V}

``configuration`` is the file's hash and ``program`` the commit that ran, the running tree's (`run.json` holds both).
``coins_on`` gives each listed subprotocol with a channel that carries coins the machine its coins were flipped on: its
sender's, or the workload's when this host stands for both.

onsite's line also has ``trusted-device``: each memory attester's premise that its device ran on a machine the
developer doesn't control (`memory_attester.trusted_device`), by component, from the outcome of its run here.

``coins`` is ``simulated`` when the challenger's host is one a prover's party is on (its machine, or two machines this
host stands for), else ``development`` (the keys are made for the run). ``simulated`` names each premise the placement
leaves unmet (a party that sends coins on a developer's party's host: `Faithful` and `Secret`), and a configured program
commit that isn't the running tree's, except for NAME, whose application runs with the running tree's commit as its
program.

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
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity import config as K
from verity.channel import Refused
from verity.commitments.crypto import lms

from . import VERIFIER as LEAN_VERIFIER, settings, verify_proof_exe, verify_proof_path

ROOT = Path(__file__).resolve().parents[4]
APPLICATIONS = ROOT / "applications"
#: the static table: the directory on its import path, and its module (module docstring)
REGISTRY: tuple[Path, str] | None = (Path(__file__).resolve().parent, "served_table")
#: the bindings `begin` keeps beside the run (module docstring)
BINDINGS = "bindings.json"
#: what isn't an entry's message in the state: the build, the logs, the transcripts, the run, its bindings and the tools
UNSENT = ("build", "logs", "transcripts", "run.json", BINDINGS, "tools.json", "record/transcripts",
          "record/verdicts.jsonl", "parties", "tree.tar")
PUBLIC = ("record/", "public/")
VERIFIER = "verifier"
RUN_KEY = lms.Params(64, 1, 4)
STOP_SECONDS = 60
Entry = Callable[[Mapping[str, Any], Path, bool], int]


@dataclass(frozen=True)
class Start:
    writes: Mapping[str, tuple[str, ...]]
    serve: Callable[[Mapping[str, Any], Path], None]
    stop: Callable[[Mapping[str, Any], Path], None]
    within: int = 600


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
    starts: Mapping[str, Start] = field(default_factory=dict)

    def entries(self, cfg: Mapping[str, Any], party: str) -> list[tuple[str, Entry | str]]:
        """``party``'s entries under ``cfg``: the workload's when it is the workload's party, then each listed
        subprotocol's that declares it, a subprotocol's after those it requires; refused by name where a subprotocol has
        none yet."""
        w = cfg["workload"]["name"]
        out: list[tuple[str, Entry | str]] = []
        if any(r.name == party for r in self.workloads[w].parties):
            out.append((w, self.workers[w]))
        for c in _required_first(cfg["protocols"], self.components):
            if any(r.name == party for r in self.components[c].parties):
                if party not in self.parties.get(c, {}):
                    raise Refused("no-party", f"{c} has no step-0 {party} yet: verity run runs "
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
        raise Refused("no-table", "step 0 has no table: the toy it ran was removed, and no workload's protocols "
                                  "have brought theirs yet")
    path, module = table
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    D = importlib.import_module(module)
    return Registry(D.MANIFESTS, {n: w for n, (w, _) in D.WORKLOADS.items()},
                    {n: e for n, (_, e) in D.WORKLOADS.items()}, D.PARTIES, D.ORDER, D.live, getattr(D, "STARTS", {}))


def load(path: Path, reg: Registry) -> dict[str, Any]:
    """The configuration at ``path``, checked (`Refused` by name)."""
    return K.check(K.parse(path.read_text()), reg.components, reg.workloads)


# --- placement ---------------------------------------------------------------------------------------------------------

def placed(cfg: Mapping[str, Any]) -> dict[str, str]:
    return {p["name"]: p["machine"] for p in cfg["party"]}


def plan(cfg: Mapping[str, Any], reg: Registry, machines: Sequence[str], wanted: Sequence[str] = ()) -> list[str]:
    """The parties this run runs, the entries first in the table's order: ``wanted``, each placed on one of
    ``machines``, else every party the configuration places on them; each with its entries (`Registry.entries`)."""
    at, here = placed(cfg), ", ".join(machines)
    if unknown := [p for p in wanted if p not in at]:
        raise Refused("unknown-party", f"{', '.join(unknown)} (the configuration's: {', '.join(at)})")
    if elsewhere := [p for p in wanted if at[p] not in machines]:
        raise Refused("machine", ", ".join(f"{p} runs on {at[p]}, not {here} (--on)" for p in elsewhere))
    chosen = list(dict.fromkeys(wanted)) or [p for p, m in at.items() if m in machines]
    if not chosen:
        raise Refused("machine", f"the configuration places no party on {here} (its machines: "
                                 f"{', '.join(sorted(set(at.values())))})")
    for p in chosen:
        reg.entries(cfg, p)
    rank = {p: i for i, p in enumerate(reg.order)}
    return sorted(chosen, key=lambda p: (rank.get(p, len(rank)), list(at).index(p)))


def bindings(path: Path | None, cfg: Mapping[str, Any], reg: Registry) -> dict[str, dict[str, dict[str, Any]]]:
    """The bindings at ``path`` (module docstring), checked against ``cfg``; refused by name (``toml``,
    ``binding``)."""
    if path is None:
        return {}
    got = K.parse(path.read_text())
    parties = {p["name"] for p in cfg["party"]}
    for party, tables in got.items():
        if party not in parties:
            raise Refused("binding", f"{path} binds {party}, which the configuration has no party of (its parties: "
                                     f"{', '.join(sorted(parties))})")
        mine = [c for c in cfg["protocols"] if any(r.name == party for r in reg.components[c].parties)]
        if not isinstance(tables, Mapping) or not all(isinstance(t, Mapping) for t in tables.values()):
            raise Refused("binding", f"{path}: [{party}] holds a table per component ([{party}.COMPONENT])")
        if extra := sorted(set(tables) - set(mine)):
            raise Refused("binding", f"{path} binds {party} for {', '.join(extra)}, not a listed component that "
                                     f"declares it ({', '.join(mine) or 'none'})")
        for c, t in tables.items():
            if bad := sorted(k for k, v in t.items() if type(v) not in (str, int)):
                raise Refused("binding", f"{path}: {party}.{c}'s {', '.join(bad)} are strings or integers")
    return {p: {c: dict(t) for c, t in tables.items()} for p, tables in got.items()}


def describe(cfg: Mapping[str, Any], component: str, entry: Entry | str, bound: Mapping[str, Any]) -> dict[str, Any]:
    """An entry as the dry run prints it: its own ``describe(cfg, bindings)`` when it has one."""
    if hasattr(entry, "describe"):
        return {"component": component, **entry.describe(cfg, bound)}
    return {"component": component, "entry": "live" if not callable(entry) else getattr(entry, "__name__", "entry")}


def dry(cfg: Mapping[str, Any], reg: Registry, chosen: Sequence[str], bound: Mapping[str, Any]) -> list[str]:
    """The dry run (module docstring): every party of ``chosen`` resolved with its entries, every channel; the
    bindings still unbound. Refused by name (``no-piece``) where an entry's script is missing."""
    at, principal = placed(cfg), {p["name"]: p["principal"] for p in cfg["party"]}
    plans = [{"party": p, "machine": at[p], "principal": principal[p],
              "entries": [describe(cfg, c, e, bound) for c, e in reg.entries(cfg, p)],
              **({"writes": {c: list(k) for c, k in reg.starts[p].writes.items()}} if p in reg.starts else {})}
             for p in chosen]
    if missing := sorted({f"{x['party']}'s {e['component']} piece: {s}" for x in plans for e in x["entries"]
                          for s in e.get("missing", ())}):
        raise Refused("no-piece", f"not in the tree: {'; '.join(missing)}")
    for x in plans:
        say("PLAN", x)
    for c in cfg["channel"]:
        say("CHANNEL", {"from": c["from"], "to": c["to"], "messages": c["messages"], "after": c["after"]})
    written = {f"{p}.{c}.{k}" for p in starting(reg, chosen, bound) for c, ks in reg.starts[p].writes.items()
               for k in ks}
    unbound = sorted({u for x in plans for e in x["entries"] for u in e.get("unbound", ())} - written)
    say("DRY", {"configuration": K.digest(cfg), "parties": len(plans), "channels": len(cfg["channel"]),
                "entries": sum(len(x["entries"]) for x in plans), "unbound": unbound})
    return unbound


def starting(reg: Registry, chosen: Sequence[str], bound: Mapping[str, Any]) -> list[str]:
    out = []
    for p in (p for p in chosen if p in reg.starts):
        keys = [(c, k) for c, ks in reg.starts[p].writes.items() for k in ks]
        given = [f"{p}.{c}.{k}" for c, k in keys if k in bound.get(p, {}).get(c, {})]
        if given and len(given) < len(keys):
            raise Refused("binding", f"--bindings gives {', '.join(given)} of what {p} writes, not all of it: give "
                                     f"every one ({p} runs where they say) or none ({p} starts here)")
        if not given:
            out.append(p)
    return out


def run_keys(state: Path, parties: Iterable[str]) -> None:
    for p in parties:
        mine, public = state / "parties" / p / "key.json", state / "public" / p / "key.json"
        if mine.is_file():
            continue
        seed = os.urandom(64)
        half = lms.public_key(lms.keygen(RUN_KEY, seed))
        params = {"params": [RUN_KEY.N, RUN_KEY.W, RUN_KEY.H]}
        for f in (mine, public):
            f.parent.mkdir(parents=True, exist_ok=True)
        mine.touch(mode=0o600)
        mine.write_text(json.dumps({**params, "seed": seed.hex()}, sort_keys=True))
        public.write_text(json.dumps({**params, "public_key": half.hex(), "sha512": hashlib.sha512(half).hexdigest()},
                                     sort_keys=True))


def archive(state: Path, commit: str, root: Path = ROOT) -> str:
    out, part = state / "tree.tar", state / "build" / "tree.tar"
    if not out.is_file():
        part.parent.mkdir(parents=True, exist_ok=True)
        with open(part, "wb") as f:
            rc = subprocess.run(["git", "archive", "--format=tar", f"--prefix={commit}/", commit], cwd=root,
                                stdout=f).returncode
        if rc:
            raise Refused("tree", f"`git archive {commit}` exited {rc} in {root}: no party gets the run's tree")
        part.replace(out)
    return out.name


def sharing(cfg: Mapping[str, Any], reg: Registry, machines: Sequence[str]) -> dict[str, list[str]]:
    listed = [reg.components[c] for c in cfg["protocols"]]
    parties, channels = K.declared(listed, reg.workloads[cfg["workload"]["name"]])
    at, out = placed(cfg), {}
    for p in sorted({ch.sender for ch in channels.values() if ch.carries_coins}):
        host = set(machines) if at[p] in machines else {at[p]}
        if shared := sorted(q for q, m in at.items() if m in host and parties[q].principal == "developer"):
            out[p] = shared
    return out


def unmet(cfg: Mapping[str, Any], reg: Registry, head: str | None, machines: Sequence[str] = (K.LOCAL,)) -> list[str]:
    """The premises the run's placement leaves unmet (module docstring): each party that sends coins on a developer's
    party's host (its machine, or ``machines``, the ones this host stands for), and a configured program commit other
    than the running tree's ``head``."""
    at, host = placed(cfg), f"this host ({', '.join(machines)})"
    out = [f"{p} shares {at[p] if {at[q] for q in shared} == {at[p]} else host} with the developer's "
           f"{', '.join(shared)}: Faithful and Secret unmet" for p, shared in sharing(cfg, reg, machines).items()]
    if head != cfg["program"]["commit"]:
        out.append(f"the program is the running tree's {head}, not the configured {cfg['program']['commit']}")
    return out


def coins_on(cfg: Mapping[str, Any], reg: Registry, machines: Sequence[str]) -> dict[str, str]:
    at = placed(cfg)
    work = at[reg.workloads[cfg["workload"]["name"]].parties[0].name]
    return {c: work if at[ch.sender] in machines and work in machines else at[ch.sender]
            for c in cfg["protocols"] for ch in reg.components[c].channels if ch.carries_coins}


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


def build_verifier(root: Path = ROOT, run: Callable[..., Any] = subprocess.run) -> None:
    exe = verify_proof_path(root)
    if exe.is_file():
        return
    print(f"verity run: {exe.relative_to(root)} is not built: building the verifier (setup.sh, then lake build)",
          file=sys.stderr, flush=True)

    def step(argv: list[str], cwd: Path) -> None:
        if rc := run(argv, cwd=cwd).returncode:
            raise Refused("verifier", f"`{' '.join(argv)}` exited {rc} in {cwd}: the verifier isn't built")

    step(["bash", "tools/verity/lean/setup.sh", LEAN_VERIFIER], root)
    step([shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake"), "build", verify_proof_exe(root)],
         root / LEAN_VERIFIER)
    if not exe.is_file():
        raise Refused("verifier", f"lake build wrote no {exe.relative_to(root)}")


def begin(cfg: Mapping[str, Any], state: Path, periods: int, machines: Sequence[str], hooks: Iterable[str],
          bound: Mapping[str, Any] | None = None, program: str | None = None, tree: bool = False) -> dict[str, Any]:
    """The run's ``run.json`` (module docstring), written before any party starts, or the one in ``state`` when it is
    this configuration's run at ``periods`` (another machine's parties); the bindings ``bound`` beside it."""
    state.mkdir(parents=True, exist_ok=True)
    if bound:
        (state / BINDINGS).write_text(json.dumps(bound, indent=1, sort_keys=True))
    digest = K.digest(cfg)
    f = state / "run.json"
    if f.exists():
        run = json.loads(f.read_text())
        if run["configuration"] != digest or run["periods"] != periods:
            raise Refused("state", f"{state} holds the run of configuration {run['configuration'][:16]}… at "
                                   f"{run['periods']} periods, not {digest[:16]}… at {periods}")
        return run
    run = {"run": settings.run_id(), "periods": periods, "configuration": digest,
           "hooks": sorted(hooks), "parties": placed(cfg), "machines": list(machines), "program": program,
           "tree": archive(state, program) if tree and program else None}
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
    """Each entry in a process of its own (``python -m verity_cli.run party CONFIG STATE PARTY MACHINE...``, the
    machines this host stands for), its output in ``STATE/logs/<party>.log``."""

    def __init__(self, config: Path, state: Path, machines: Sequence[str]) -> None:
        self.config, self.state, self.machines = config.resolve(), state, list(machines)
        self.argv = [sys.executable, "-m", "verity_cli.run"]

    def _env(self) -> dict[str, str]:
        run = json.loads((self.state / "run.json").read_text())
        return {**os.environ, "VERITY_PROGRAM": run.get("program") or "",
                "VERITY_TREE": str(self.state / run["tree"]) if run.get("tree") else ""}

    def _argv(self, verb: str, party: str) -> list[str]:
        (self.state / "logs").mkdir(parents=True, exist_ok=True)
        return [*self.argv, verb, str(self.config), str(self.state), party, *self.machines]

    def run(self, party: str) -> int:
        argv = self._argv("party", party)
        with open(self.state / "logs" / f"{party}.log", "w") as log:
            return subprocess.run(argv, cwd=ROOT, env=self._env(), stdout=log, stderr=subprocess.STDOUT).returncode

    def start(self, party: str) -> Spawned:
        argv = self._argv("start", party)
        with open(self.state / "logs" / f"{party}.log", "w") as log:
            return Spawned(subprocess.Popen(argv, cwd=ROOT, env=self._env(), stdout=log, stderr=subprocess.STDOUT))


@dataclass(frozen=True)
class Spawned:
    process: subprocess.Popen

    def alive(self) -> bool:
        return self.process.poll() is None

    def stop(self) -> int:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                return self.process.wait(STOP_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()
        return self.process.wait()


@dataclass(frozen=True)
class Served:
    start: Start
    cfg: Mapping[str, Any]
    state: Path

    def alive(self) -> bool:
        return True

    def stop(self) -> int:
        self.start.stop(self.cfg, self.state)
        return 0


class InProcess:
    """Each entry in this process: for a run whose entries are functions of the test that runs it."""

    def __init__(self, cfg: Mapping[str, Any], reg: Registry, state: Path, local: bool) -> None:
        self.cfg, self.reg, self.state, self.local = cfg, reg, state, local

    def run(self, party: str) -> int:
        return run_party(self.cfg, self.reg, self.state, party, self.local)

    def start(self, party: str) -> Served:
        self.reg.starts[party].serve(self.cfg, self.state)
        return Served(self.reg.starts[party], self.cfg, self.state)


def say(what: str, body: Mapping[str, Any]) -> None:
    print(f"{what} " + json.dumps(body, sort_keys=True), flush=True)


def _failed(state: Path, party: str, e: Exception) -> None:
    (state / "logs").mkdir(parents=True, exist_ok=True)
    (state / "logs" / f"{party}.error").write_text(f"{type(e).__name__}: {e}")


def wait_bindings(state: Path, party: str, start: Start, alive: Callable[[], bool],
                  clock: Callable[[], float] | None = None, pause: Callable[[float], Any] | None = None
                  ) -> dict[str, dict[str, Any]]:
    clock, pause = clock or time.monotonic, pause or time.sleep
    f, until = state / "public" / party / "bindings.json", clock() + start.within
    while not f.is_file():
        if not alive():
            raise Refused("bindings", f"{party} ended before it wrote {f} (its log: STATE/logs/{party}.log)")
        if clock() >= until:
            raise Refused("bindings", f"{party} wrote no {f} within {start.within} s, so no party that reads it "
                                      f"starts (its log: STATE/logs/{party}.log)")
        pause(0.1)
    try:
        got = json.loads(f.read_text())
    except ValueError as e:
        raise Refused("bindings", f"{party}'s {f} isn't JSON ({e}): write it whole, by a rename") from None
    want = {c: sorted(ks) for c, ks in start.writes.items()}
    if not isinstance(got, dict) or not all(isinstance(t, dict) for t in got.values()) \
            or {c: sorted(t) for c, t in got.items()} != want \
            or any(type(v) not in (str, int) for t in got.values() for v in t.values()):
        raise Refused("bindings", f"{party} wrote {json.dumps(got, sort_keys=True)[:400]} in {f}, not a table per "
                                  f"subprotocol of strings and integers keyed {json.dumps(want, sort_keys=True)}")
    return got


def bind(state: Path, party: str, written: Mapping[str, Mapping[str, Any]]) -> None:
    f = state / BINDINGS
    bound = json.loads(f.read_text()) if f.is_file() else {}
    for c, t in written.items():
        bound.setdefault(party, {}).setdefault(c, {}).update(t)
    f.write_text(json.dumps(bound, indent=1, sort_keys=True))


def execute(cfg: Mapping[str, Any], reg: Registry, state: Path, chosen: Sequence[str], runner: Any,
            premises: Sequence[str], machines: Sequence[str] = (K.LOCAL,), starts: Sequence[str] = ()) -> int:
    """``chosen``'s parties run in ``state`` by ``runner`` (`Processes` or `InProcess`): those it ``starts`` first,
    each waited on for its bindings, then the entries, each one's transcript written, then its live parties' run, then
    the verdicts when the verifier is among them, and last each started party stopped (module docstring); the exit
    code."""
    started: list[tuple[str, Any, dict[str, tuple[int, int]]]] = []
    rc = 2
    try:
        for p in starts:
            before = files(state)
            try:
                started.append((p, runner.start(p), before))
                written = wait_bindings(state, p, reg.starts[p], started[-1][1].alive)
            except Refused as e:
                print(f"verity run: refused {e.name}: {e.detail}", file=sys.stderr)
                return 2
            except Exception as e:  # noqa: BLE001  (a party that can't start failed, and the run stops)
                _failed(state, p, e)
                print(f"verity run: {p} didn't start: {type(e).__name__}: {e}", file=sys.stderr)
                return 2
            bind(state, p, written)
            say("STARTED", {"party": p, "writes": {c: sorted(t) for c, t in written.items()}})
        rc = _parties(cfg, reg, state, [p for p in chosen if p not in starts], runner, premises, machines)
    finally:
        for p, h, before in reversed(started):
            try:
                stopped = h.stop()
            except Exception as e:  # noqa: BLE001  (a stop that raises failed)
                _failed(state, p, e)
                stopped = 1
            transcribe(state, p, [f for f in changed(before, files(state)) if f.startswith(f"public/{p}/")])
            say("PARTY", {"party": p, "exit": stopped})
            rc = rc or (2 if stopped else 0)
    return rc


def _parties(cfg: Mapping[str, Any], reg: Registry, state: Path, chosen: Sequence[str], runner: Any,
             premises: Sequence[str], machines: Sequence[str]) -> int:
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
    except Refused as e:
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
    lines = [{**x, "run": run["run"], "configuration": run["configuration"], "program": run.get("program"),
              "simulated": list(premises)}
             for x in (r.outcomes[VERIFIER]["outcome"] or {}).get("verdicts", [])]
    on = coins_on(cfg, reg, machines)
    lines = [{**x, "coins_on": {**x.get("coins_on", {}), **on}} for x in lines]
    if sharing(cfg, reg, machines):
        lines = [{**x, "coins": "simulated"} for x in lines]
    if (cid := K.composite(cfg["protocols"], reg.components)) in K.UNACCEPTED:
        lines = [{**x, "product": cid} if x.get("combined_by") else
                 {**x, "product": cid, "verdict": "undecided", "reason": x.get("reason") or f"no composite takes {cid}"}
                 for x in lines]
    for line in lines:
        say("VERDICT", line)
        if line.get("combined_by"):
            print(f"verity run: block {line['block']}'s verdict {line['verdict']} is combined by "
                  f"{line['combined_by']}; no composite takes {line['product']}", file=sys.stderr)
    (state / "record").mkdir(parents=True, exist_ok=True)
    (state / "record" / "verdicts.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in lines))
    verdicts = {x["verdict"] for x in lines}
    if "reject" in verdicts:
        return 1
    return 0 if verdicts == {"accept"} and len(lines) == run["periods"] and not refused else 2


def hooks_of(cfg: Mapping[str, Any], reg: Registry) -> set[str]:
    return set().union(*(reg.components[c].needs for c in cfg["protocols"]))


def application(arg: str) -> Path:
    if (path := Path(arg)).is_file() or path.suffix or len(path.parts) > 1:
        return path
    if (named := APPLICATIONS / f"{arg.replace('-', '_')}.toml").is_file():
        return named
    names = sorted(p.stem for p in APPLICATIONS.glob("*.toml"))
    raise Refused("application", f"{arg} is neither a configuration's path nor an application's name: NAME is one of "
                                 f"{', '.join(names) or 'none'}")


def run_main(argv: Sequence[str], reg: Registry | None = None, in_process: bool = False) -> int:
    ap = argparse.ArgumentParser(prog="verity run", description="Run a configuration's parties on this "
                                 "host (module docstring of verity_cli.run).")
    ap.add_argument("config", metavar="CONFIG|NAME", help="a configuration's path, or an application's name "
                    "(applications/NAME.toml)")
    ap.add_argument("--party", action="append", default=[], metavar="ROLE", help="only this party (repeatable)")
    ap.add_argument("--on", dest="machines", action="append", default=[], metavar="MACHINE", help="a machine's name "
                    f"in the configuration that this host stands for (repeatable; default {K.LOCAL})")
    ap.add_argument("--periods", type=int, default=1, metavar="N", help="the run's periods (default 1)")
    ap.add_argument("--state", metavar="DIR", help="the parties' shared state (default a new temporary directory)")
    ap.add_argument("--bindings", metavar="FILE", help="the bindings: hosts, addresses and gate commands (TOML)")
    ap.add_argument("--dry-run", action="store_true", help="resolve every party, channel and entry, and run nothing")
    a = ap.parse_args(argv)
    if a.periods < 1:
        ap.error(f"--periods is at least 1, not {a.periods}")
    try:
        reg = reg or registry()
        config = application(a.config)
        cfg = load(config, reg)
        bound = bindings(Path(a.bindings) if a.bindings else None, cfg, reg)
        named = config != Path(a.config)
        machines = list(dict.fromkeys(a.machines or (placed(cfg).values() if named else [K.LOCAL])))
        if a.dry_run:
            at = placed(cfg)
            if not a.machines and (unknown := [p for p in a.party if p not in at]):
                raise Refused("unknown-party", f"{', '.join(unknown)} (the configuration's: {', '.join(at)})")
            unbound = dry(cfg, reg, plan(cfg, reg, machines, a.party) if a.machines else
                          list(dict.fromkeys(a.party)) or list(at), bound)
            if a.bindings and unbound:
                raise Refused("unbound", f"{a.bindings} doesn't bind {', '.join(unbound)}")
            return 0
        state = Path(a.state or tempfile.mkdtemp(prefix="verity-run-")).resolve()
        chosen = plan(cfg, reg, machines, a.party)
        starts = starting(reg, chosen, bound)
        ran = head()
        begin(cfg, state, a.periods, machines, hooks_of(cfg, reg), bound, ran, tree=bool(starts) or not in_process)
        run_keys(state, chosen)
        build_verifier()
    except Refused as e:
        print(f"verity run: refused {e.name}: {e.detail}", file=sys.stderr)
        return 2
    say("RUN", {"state": str(state), "parties": chosen, "configuration": K.digest(cfg), "periods": a.periods})
    runner = (InProcess(cfg, reg, state, set(placed(cfg).values()) <= set(machines)) if in_process else
              Processes(config, state, machines))
    program = {**cfg, "program": {**cfg["program"], "commit": ran}} if named and ran else cfg
    return execute(cfg, reg, state, chosen, runner, unmet(program, reg, ran, machines), machines, starts)


def party_main(argv: Sequence[str]) -> int:
    """One party's entries in this process (`Processes`): ``party CONFIG STATE PARTY MACHINE...``."""
    config, state, party, *machines = argv
    reg = registry()
    cfg = load(Path(config), reg)
    return run_party(cfg, reg, Path(state), party, set(placed(cfg).values()) <= set(machines))


# --- verity config -----------------------------------------------------------------------------------------------------

HEADER = ("The deployment configuration of {protocols} on the {workload} workload (`verity config new`), every party",
          "on {machine}: `verity run` this file. Parameters it doesn't state are their declarations' defaults.")


def config_main(argv: Sequence[str], reg: Registry | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity config", description="Write, check or hash a configuration.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new", help="a configuration that states only its choices")
    p.add_argument("--workload", required=True)
    p.add_argument("--protocols", required=True, help="the subprotocols, comma-separated: remote's "
                   "(sampled-proofs,pearl-plus,inference-only) or onsite's")
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
    except Refused as e:
        print(f"refused {e.name}: {e.detail}")
        return 2
    print(f"ok {K.digest(cfg)}" if a.cmd == "check" else K.digest(cfg))
    return 0


def start_main(argv: Sequence[str]) -> int:
    config, state, party = argv[:3]
    reg = registry()
    cfg, start = load(Path(config), reg), reg.starts[party]
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    start.serve(cfg, Path(state))
    stopping.wait()
    start.stop(cfg, Path(state))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "party":
        return party_main(args[1:])
    if args and args[0] == "start":
        return start_main(args[1:])
    raise SystemExit("python -m verity_cli.run party|start CONFIG STATE PARTY MACHINE... (verity run CONFIG runs the "
                     "parties)")


if __name__ == "__main__":
    raise SystemExit(main())
