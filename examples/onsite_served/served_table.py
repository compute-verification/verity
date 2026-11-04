"""The served window's table for `verity run` (`verity_cli.run.Registry`): onsite's components as a configuration declares
them (`MANIFESTS`), the ``served-window`` workload (`SERVED`), each party's piece (`PARTIES`), and the run of the pieces
placed on a machine (`live`).

A piece is what one party does in the run (`Piece`), in one of three ways (``runs``):

- ``served-window``: inside the served window's runner, `benchmarks/one_stage/served_verdicts.sh`, which the workload's
  party (``workers``) runs once with every listed component's hooks: the variables it reads for that piece (``POUS_CMD``,
  ``WIPE_CMD``, ``SERVE_ARGS``, ...), its parameters from the configuration and its hosts and addresses from the deploy
  bindings (`verity run --bindings`), never from the configuration;
- ``gate``: the auditor's side of a session, which the auditor's gate starts on its host when the prover's piece opens it;
  `verity run` starts nothing for it;
- ``check``: the verifier's onsite check, `served_verdicts.py record` over the served window's public record, whose
  composed verdict is the period's.

A piece names the scripts it runs (``scripts``), which must be in the tree, and the deploy bindings it reads: those of
the party it reaches (``reaches``, default its own), ``party.component.key`` each, the ``binds`` ones bound before the run
starts (``unbound``) and the ``optional`` ones used when bound. `describe` is what `verity run --dry-run` prints of it.

The phases: Registration (the site, θ and the builder B: the weights registration before the first request, and the
committer's compiler), then Execution (the served window), then Verification (each component's commitment, challenge,
proof or timed response, and validation: the pieces and the onsite check).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.core.deploy import config as K
from verity.core.deploy.config import Component, Workload
from verity.core.deploy.manifest import Channel, Role
from verity.core.protocols.pous import wipe_parties as WP
from verity.core.protocols.sampled_proofs import parties as SP
from verity.core.protocols.sampled_proofs.one_stage.draw import parse_law
from verity.core.wire import Refused

ROOT = Path(__file__).resolve().parents[2]
SERVED_WINDOW, GATE, CHECK = "served-window", "gate", "check"
RUNS = (SERVED_WINDOW, GATE, CHECK)
RUNNER = "benchmarks/one_stage/served_verdicts.sh"
VERDICTS_SCRIPT = "benchmarks/one_stage/served_verdicts.py"
POUS_PIECE = "benchmarks/pous/served_piece.sh"
WIPE_PIECE = "benchmarks/pous/streamed_wipe/served_wipe.sh"
CERTIFIER_PIECE = "benchmarks/network_traces/certifier_piece.py"
ONSITE = "verity/onsite/v0"
#: the deploy bindings `begin` keeps beside the run (`verity_cli.run.BINDINGS`)
BINDINGS = "bindings.json"
WORKERS, FIREWALL, PROVER, VERIFIER = SP.WORKERS, SP.FIREWALL, WP.PROVER, SP.VERIFIER
CHALLENGER, MEMORY_CHALLENGER = SP.CHALLENGER, WP.CHALLENGER
BANDWIDTH_VERIFIER, NETWORK_CERTIFIER = "bandwidth-verifier", "network-certifier"
#: served_verdicts.py's record reads each piece's answer by its runner name
RECORD_NAMES = {"integrity": "sampled-proofs", "work": "pouw", "inference-only": "inference-only", "space": "pous",
                "bandwidth": "bandwidth", "communication": "certifier", "wipe": "wipe"}


# --- what each component and the workload declare -------------------------------------------------------------------

def _law(law: str, params: Mapping[str, Any]) -> None:
    try:
        parse_law(law)
    except ValueError as e:
        raise Refused("law", str(e)) from None


def _pous(params: Mapping[str, Any]) -> dict[str, Any]:
    """PoUS's served piece: ``challenges``, the memory challenger's reads in the period (320: ε = 2^-20)."""
    if extra := sorted(set(params) - {"challenges"}):
        raise Refused("unknown-key", f"[params.\"{SP.IDS['pous']}\"]: {', '.join(extra)} (known: challenges)")
    k = params["challenges"]
    if type(k) is not int or k < 1:
        raise Refused("params", f"challenges is a positive integer, not {k!r}")
    return {"challenges": k}


_COMMIT = re.compile(r"[0-9a-f]{40}")
LOADS = ("varied", "fixed")


def _served(params: Mapping[str, Any]) -> dict[str, Any]:
    """The served window's public shape: the model at its revision and precision, the window's request ``slots``, its
    engine steps (``window-steps``) and its ``load`` (served_verdicts.sh's LOAD)."""
    known = {"model", "revision", "precision", "slots", "window-steps", "load"}
    if extra := sorted(set(params) - known):
        raise Refused("unknown-key", f"[workload.params]: {', '.join(extra)} (known: {', '.join(sorted(known))})")
    for k in ("model", "precision"):
        if not isinstance(params[k], str) or not params[k]:
            raise Refused("workload", f"{k} is a non-empty string, not {params[k]!r}")
    if not isinstance(params["revision"], str) or not _COMMIT.fullmatch(params["revision"]):
        raise Refused("workload", f"revision is the model's 40-hex revision, not {params['revision']!r}")
    for k in ("slots", "window-steps"):
        if type(params[k]) is not int or params[k] < 1:
            raise Refused("workload", f"{k} is a positive integer, not {params[k]!r}")
    if params["load"] not in LOADS:
        raise Refused("workload", f"load is one of {', '.join(LOADS)}, not {params['load']!r}")
    return {k: params[k] for k in sorted(known)}


_VERIFIER = Role(VERIFIER, "auditor", needs=("audit", "component", "product"))
_PROVER = Role(PROVER, "developer", per="server", needs=("store",))
_MEMORY_CHALLENGER = Role(MEMORY_CHALLENGER, "auditor", needs=("clock", "coins", "signing-key"))
MANIFESTS: dict[str, Component] = {c.name: c for c in (
    Component("sampled-proofs", SP.IDS["sampled-proofs"], parties=SP.MANIFEST.parties, channels=SP.MANIFEST.channels,
              needs=frozenset({"windows"}), law=_law, default_law="subset:1"),
    Component("pouw", SP.IDS["pouw"], parties=(Role(CHALLENGER, "auditor", needs=("host",)),
                                               Role(WORKERS, "developer", needs=("serve",)), _VERIFIER),
              channels=(Channel(CHALLENGER, WORKERS, ("H1Coins",), carries_coins=True),
                        Channel(WORKERS, VERIFIER, ("H1Record",), after=("H1Coins",))),
              needs=frozenset({"windows", "noise"}), requires=("sampled-proofs",)),
    Component("inference-only", SP.IDS["inference-only"], parties=(_VERIFIER,), requires=("sampled-proofs",)),
    Component("pous", SP.IDS["pous"], parties=(_MEMORY_CHALLENGER, _PROVER, _VERIFIER),
              channels=(Channel(PROVER, MEMORY_CHALLENGER, ("Setup", "Response")),
                        Channel(MEMORY_CHALLENGER, PROVER, ("Challenge",), carries_coins=True),
                        Channel(MEMORY_CHALLENGER, VERIFIER, ("SignedRun",), after=("Response",))),
              needs=frozenset({"store"}), params=_pous, default_params={"challenges": 320}),
    Component("bandwidth", SP.IDS["bandwidth"],
              parties=(Role(BANDWIDTH_VERIFIER, "auditor", needs=("clock", "coins")), _PROVER, _VERIFIER),
              channels=(Channel(PROVER, BANDWIDTH_VERIFIER, ("Register", "Root", "Openings")),
                        Channel(BANDWIDTH_VERIFIER, PROVER, ("Hold", "Run", "Open", "End"), carries_coins=True,
                                after=("Register",)),
                        Channel(BANDWIDTH_VERIFIER, VERIFIER, ("BandwidthAnswer",), after=("Openings",))),
              needs=frozenset({"receipts"})),
    Component("certifier", SP.IDS["certifier"],
              parties=(Role(NETWORK_CERTIFIER, "auditor", needs=("coins", "signing-key")), _VERIFIER),
              channels=(Channel(NETWORK_CERTIFIER, WORKERS, ("out-opening",)),
                        Channel(NETWORK_CERTIFIER, VERIFIER, ("out-record",))),
              needs=frozenset({"windows", "ingress-relay"}), requires=("sampled-proofs",),
              colocated=((CHALLENGER, NETWORK_CERTIFIER),)),
    Component("wipe", "verity/wipe/v0", parties=(*WP.MANIFEST.parties, _VERIFIER),
              channels=(*WP.MANIFEST.channels, Channel(MEMORY_CHALLENGER, VERIFIER, ("WipeReport",), after=("End",)))),
)}
SERVED = Workload(SERVED_WINDOW, frozenset({"windows", "noise", "store", "receipts", "ingress-relay"}),
                  (Role(WORKERS, "developer", needs=("serve",)),), _served,
                  {"model": "verity-int8/Llama-3.2-1B-INT8", "revision": "15b0976d400d796ee589e7737d03a919526bf800",
                   "precision": "verity_int8", "slots": 4, "window-steps": 16, "load": "varied"})
#: onsite's subprotocols, the seven `Core.Composites.composites` holds for it
ONSITE_COMPONENTS = ("sampled-proofs", "pouw", "inference-only", "pous", "bandwidth", "certifier", "wipe")
PROTOCOLS = ONSITE_COMPONENTS
PROVER_NODE, AUDITOR_HOST, GATE_HOST = "prover-node", "auditor-host", "gate-host"
#: each party's machine: the developer's on the prover node, the auditor's devices on machines it doesn't control
PLACEMENT = {WORKERS: PROVER_NODE, FIREWALL: PROVER_NODE, PROVER: PROVER_NODE, CHALLENGER: GATE_HOST,
             NETWORK_CERTIFIER: GATE_HOST, MEMORY_CHALLENGER: AUDITOR_HOST, BANDWIDTH_VERIFIER: AUDITOR_HOST,
             VERIFIER: AUDITOR_HOST}
HEADER = ("The onsite run of the served window: onsite's seven subprotocols, each party on its machine. Run it on each",
          "machine with `verity run THIS --on MACHINE --bindings FILE` (examples/onsite_served/README.md); the deploy",
          "bindings hold every host and address, this file none. `verity run THIS --dry-run` resolves it all.")


def configuration(commit: str) -> dict[str, Any]:
    """The configuration `examples/onsite_served/config.toml` holds, at program ``commit``: `config new`'s over
    `PROTOCOLS`, each party at its `PLACEMENT`, the workload's and PoUS's parameters stated."""
    cfg = K.new(SERVED_WINDOW, list(PROTOCOLS), MANIFESTS, {SERVED_WINDOW: SERVED}, commit)
    cfg["workload"]["params"] = dict(SERVED.default_params)
    cfg["params"] = {SP.IDS["pous"]: {"challenges": 320}}
    for p in cfg["party"]:
        p["machine"] = PLACEMENT[p["name"]]
    K.check(cfg, MANIFESTS, {SERVED_WINDOW: SERVED})
    return cfg


# --- the pieces ------------------------------------------------------------------------------------------------------

Hooks = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, str]], dict[str, str]]


def _none(cfg: Mapping[str, Any], params: Mapping[str, Any], bound: Mapping[str, str]) -> dict[str, str]:
    return {}


@dataclass(frozen=True)
class Piece:
    """One party's piece of the served window (module docstring): ``hooks(cfg, params, bound)`` gives the runner's
    variables for it from the configuration, the component's checked parameters and the bindings of the party it
    reaches (``bound``: each of ``binds``, a ``<party.component.key>`` placeholder while unbound, and each bound one of
    ``optional``)."""

    component: str
    party: str
    runs: str
    what: str
    scripts: tuple[str, ...] = ()
    reaches: str = ""
    binds: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    hooks: Hooks = _none

    def __post_init__(self) -> None:
        if self.runs not in RUNS:
            raise ValueError(f"a piece runs as one of {RUNS}, not {self.runs!r}")

    def key(self, name: str) -> str:
        return f"{self.reaches or self.party}.{self.component}.{name}"

    def _bindings(self, bindings: Mapping[str, Any]) -> Mapping[str, Any]:
        return bindings.get(self.reaches or self.party, {}).get(self.component, {})

    def unbound(self, bindings: Mapping[str, Any]) -> list[str]:
        mine = self._bindings(bindings)
        return [self.key(k) for k in self.binds if k not in mine]

    def missing(self) -> list[str]:
        return [s for s in self.scripts if not (ROOT / s).is_file()]

    def variables(self, cfg: Mapping[str, Any], bindings: Mapping[str, Any]) -> dict[str, str]:
        mine = self._bindings(bindings)
        bound = {**{k: f"<{self.key(k)}>" for k in self.binds},
                 **{k: str(v) for k, v in mine.items() if k in self.binds or k in self.optional}}
        params = cfg["params"].get(MANIFESTS[self.component].id, {}) if self.component in MANIFESTS else {}
        return self.hooks(cfg, params, bound)

    def describe(self, cfg: Mapping[str, Any], bindings: Mapping[str, Any]) -> dict[str, Any]:
        """What `verity run --dry-run` prints of it; `verity_cli.run` refuses ``no-piece`` while ``missing`` names a
        script."""
        return {"component": self.component, "runs": self.runs, "what": self.what, "scripts": list(self.scripts),
                "reaches": self.reaches or self.party, "missing": self.missing(),
                "hooks": self.variables(cfg, bindings), "unbound": self.unbound(bindings)}


def _serve(cfg, params, bound):
    w = cfg["workload"]["params"]
    return {"REPO": w["model"], "REV": w["revision"], "PRECISION": w["precision"], "SLOTS": str(w["slots"]),
            "WINDOW_STEPS": str(w["window-steps"]), "LOAD": w["load"], "V3C_LAW": cfg["service"]["law"]}


#: served_verdicts.sh's variable for each optional binding of sampled proofs' live challenger
_PP = {"gate": "PP_CHALLENGER", "peer": "PP_PEER", "key": "PP_CHALLENGER_KEY", "gate-key": "PP_GATE_KEY"}


def _sampled_proofs_challenger(cfg, params, bound):
    return {v: f"gate:{bound[k]}" if k == "gate" else bound[k] for k, v in _PP.items() if k in bound}


def _pouw(cfg, params, bound):
    return {"H1_GATE": bound["gate"], "H1_PEER": bound["peer"], "H1_KEY_SHA512": bound["key-sha512"]}


def _pous_piece(cfg, params, bound):
    return {"POUS_CMD": f'bash {POUS_PIECE} "$@"',
            "POUS_ARGS": shlex.join(["--rounds", str(params["challenges"]), "--challenger-host", bound["host"],
                                     "--challenger-address", bound["address"], "--challenger-command", bound["command"],
                                     "--trusted-sha512", bound["key-sha512"]])}


def _wipe_piece(cfg, params, bound):
    return {"WIPE_CMD": f'bash {WIPE_PIECE} "$1" "$2" "$3"', "WIPE_CHALLENGER": bound["host"],
            "WIPE_ADDRESS": bound["address"], "WIPE_COMMAND": bound["command"]}


def _poub_session(cfg, params, bound):
    return {"SERVE_ARGS": shlex.join(["--remote-verifier", bound["address"]])}


def _certifier_piece(cfg, params, bound):
    py = ".venv/bin/python"
    return {"CERTIFIER_FRONT": f'{py} {CERTIFIER_PIECE} front "$1" "$2" "$3" --gate {shlex.quote(bound["gate"])}',
            "CERTIFIER_CMD": f'{py} {CERTIFIER_PIECE} piece "$1" "$2" "$3"'}


SERVE = Piece(SERVED_WINDOW, WORKERS, SERVED_WINDOW, "Execution: serve-openai on the prover node under its GPU lease, the "
              "requests, every closed window sealed and the audited window picked; then sampled proofs' records audit "
              "(C-Flock --zk behind the Lean firewall, the Lean verifier) and each listed component's piece",
              scripts=(RUNNER, VERDICTS_SCRIPT), hooks=_serve)
#: each listed component's party -> its piece
PARTIES: dict[str, dict[str, Piece]] = {
    "sampled-proofs": {
        CHALLENGER: Piece("sampled-proofs", CHALLENGER, GATE, "the gate: the auditor's live challenger, whose coins "
                          "draw the proof blocks after the window's commitment"),
        FIREWALL: Piece("sampled-proofs", FIREWALL, SERVED_WINDOW, "the Lean firewall every outer proof goes through, "
                        "one at a time, in the records audit"),
        WORKERS: Piece("sampled-proofs", WORKERS, SERVED_WINDOW, "the window's values committed under hiding leaves "
                       "and sealed, its registration before the draw, and the drawn proof blocks proved in C-Flock --zk; "
                       "the draw on the live challenger when its gate is bound, else on the records audit's own live "
                       "coins", reaches=CHALLENGER, optional=tuple(_PP), hooks=_sampled_proofs_challenger),
        VERIFIER: Piece("sampled-proofs", VERIFIER, CHECK, "the onsite check: served_verdicts.py record over the "
                        "window's rows (the records audit's verdict the integrity row), the composed verdict and the "
                        "block record", scripts=(VERDICTS_SCRIPT,)),
    },
    "pouw": {
        CHALLENGER: Piece("pouw", CHALLENGER, GATE, "the gate: each window's H1 coins from a session of its own"),
        WORKERS: Piece("pouw", WORKERS, SERVED_WINDOW, "H1 on the serving engine (--protocols pouw:...), its coins "
                       "the gate's; served_verdicts.py pouw checks them against the gate's session under its key",
                       reaches=CHALLENGER, binds=("gate", "peer", "key-sha512"), hooks=_pouw),
        VERIFIER: Piece("pouw", VERIFIER, CHECK, "PoUW's row"),
    },
    "inference-only": {
        VERIFIER: Piece("inference-only", VERIFIER, CHECK, "the inference-only policy's row: no served window produces "
                        "the declared division yet, so the record has none and the onsite check stays undecided"),
    },
    "pous": {
        PROVER: Piece("pous", PROVER, SERVED_WINDOW, "PoUS's served piece: one period over the served weights held as "
                      "P2's C* on the card, the memory challenger's timed reads answered from it",
                      scripts=(POUS_PIECE,), reaches=MEMORY_CHALLENGER,
                      binds=("host", "address", "command", "key-sha512"), hooks=_pous_piece),
        MEMORY_CHALLENGER: Piece("pous", MEMORY_CHALLENGER, GATE, "the memory challenger's side, on its own coins and "
                                 "clock, signing the session"),
        VERIFIER: Piece("pous", VERIFIER, CHECK, "PoUS's row"),
    },
    "bandwidth": {
        PROVER: Piece("bandwidth", PROVER, SERVED_WINDOW, "PoUB's served session: the engine proves its store to the "
                      "bandwidth verifier a pass an engine step (serve-openai --remote-verifier); its record is the serve "
                      "dir's remote_session.json", reaches=BANDWIDTH_VERIFIER, binds=("address",),
                      hooks=_poub_session),
        BANDWIDTH_VERIFIER: Piece("bandwidth", BANDWIDTH_VERIFIER, GATE, "the bandwidth verifier's side, on its own "
                                  "coins and clock"),
        VERIFIER: Piece("bandwidth", VERIFIER, CHECK, "PoUB's row"),
    },
    "certifier": {
        NETWORK_CERTIFIER: Piece("certifier", NETWORK_CERTIFIER, GATE, "the network certifier behind the gate: its "
                                 "warden in front of serve-openai before the requests (CERTIFIER_FRONT, which the served "
                                 "window runs), and its piece, the chain and the period check on the kept warden run",
                                 scripts=(CERTIFIER_PIECE,), binds=("gate",), hooks=_certifier_piece),
        VERIFIER: Piece("certifier", VERIFIER, CHECK, "the network certifier's row"),
    },
    "wipe": {
        PROVER: Piece("wipe", PROVER, SERVED_WINDOW, "the streamed wipe of the card before its engine loads, under the "
                      "serve's lease", scripts=(WIPE_PIECE,), reaches=MEMORY_CHALLENGER,
                      binds=("host", "address", "command"), hooks=_wipe_piece),
        MEMORY_CHALLENGER: Piece("wipe", MEMORY_CHALLENGER, GATE, "the wipe's memory challenger, on its own coins and "
                                 "clock, signing the session"),
        VERIFIER: Piece("wipe", VERIFIER, CHECK, "the wipe's row"),
    },
}
WORKLOADS: dict[str, tuple[Workload, Piece]] = {SERVED_WINDOW: (SERVED, SERVE)}
#: no party here runs as an entry of its own: every piece is the served window's, the gate's or the check's
ORDER: tuple[str, ...] = ()


# --- the run ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Result:
    """What `verity_cli.run.execute` reads of a live run: each party's outcome and refusal."""

    outcomes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def bindings_of(state: Path) -> dict[str, Any]:
    f = state / BINDINGS
    return json.loads(f.read_text()) if f.exists() else {}


def pieces(cfg: Mapping[str, Any], party: str) -> list[Piece]:
    """``party``'s pieces under ``cfg``: the workload's, then each listed component's."""
    out = [SERVE] if party == WORKERS else []
    return out + [PARTIES[c][party] for c in cfg["protocols"] if party in PARTIES.get(c, {})]


def variables(cfg: Mapping[str, Any], bindings: Mapping[str, Any]) -> dict[str, str]:
    """The served window's runner's variables: every listed piece's hooks, whichever party's they are, since the
    prover's pieces reach the auditor's sides through their bindings."""
    out: dict[str, str] = {}
    for p in [SERVE, *(x for c in cfg["protocols"] for x in PARTIES.get(c, {}).values())]:
        out.update(p.variables(cfg, bindings))
    return out


def unbound(cfg: Mapping[str, Any], bindings: Mapping[str, Any]) -> list[str]:
    return [k for p in [SERVE, *(x for c in cfg["protocols"] for x in PARTIES.get(c, {}).values())]
            for k in p.unbound(bindings)]


def served_window(cfg: Mapping[str, Any], state: Path, bindings: Mapping[str, Any]) -> int:
    """The served window's runner with every listed piece's hooks, its public record in ``state/out``; its serve dir
    stays where the runner keeps it on the node (its ``D`` and ``SERVE_ROOT``, from the environment)."""
    env = {**os.environ, **variables(cfg, bindings), "OUT": str(state / "out")}
    (state / "logs").mkdir(parents=True, exist_ok=True)
    with open(state / "logs" / "served-window.log", "w") as log:
        return subprocess.run(["bash", str(ROOT / RUNNER)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                              check=False).returncode


VERDICTS = {"accept": "accept", "reject": "reject", "pending": "undecided"}


def onsite_check(cfg: Mapping[str, Any], state: Path) -> dict[str, Any]:
    """The verifier's onsite check: `served_verdicts.py record` over the served window's public record in ``state``
    (``out/``, all a verifier on another machine needs of it, the prover node named by its ``machine.txt``); the
    period's line in `verity_cli.run`'s verdict form, undecided while the record has no row for a listed subprotocol."""
    py, out = sys.executable, state / "out"
    serve, machine = out / "serve", out / "machine.txt"
    slots = str(cfg["workload"]["params"]["slots"])
    pick = subprocess.run([py, str(ROOT / VERDICTS_SCRIPT), "window", str(serve), "--slots", slots], cwd=ROOT,
                          capture_output=True, text=True, check=False)
    if pick.returncode:
        raise Refused("window", pick.stderr.strip()[-800:] or "served_verdicts.py window picked no window")
    k = json.loads(pick.stdout)["window"]
    lines = machine.read_text().splitlines() if machine.is_file() else []
    node = ["--node", lines[1]] if len(lines) > 1 and lines[1].strip() else []
    p = subprocess.run([py, str(ROOT / VERDICTS_SCRIPT), "record", "--serve-run", str(serve), "--window", str(k),
                        "--sampled-proofs", str(out / "sampled-proofs" / "served"), "--pieces", str(out / "pieces"),
                        *node, "--out", str(state / "record" / "onsite")], cwd=ROOT, capture_output=True, text=True,
                       check=False)
    line = next((json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("RECORD ")), None)
    if line is None:
        raise Refused("record", (p.stderr or p.stdout).strip()[-800:] or "served_verdicts.py record wrote no RECORD")
    names = {v: k for k, v in RECORD_NAMES.items()}
    rows = json.loads((state / "record" / "onsite" / "verdicts.json").read_text())["protocols"]
    components = {c: rows.get(names[c], {}).get("verdict", "absent") for c in cfg["protocols"]}
    absent = sorted(c for c, v in components.items() if v == "absent")
    verdict = VERDICTS.get(line["composed"], "undecided")
    return {"block": k, "components": components, "product": K.composite(cfg["protocols"], MANIFESTS),
            "reason": line.get("why") or (f"missing {sorted(line['missing'])}" if line.get("missing") else None)
            or (f"no row for {absent}" if absent else None),
            "registration": None, "coins": coins_of(cfg),
            "verdict": "undecided" if verdict == "accept" and absent else verdict}


def coins_of(cfg: Mapping[str, Any]) -> str:
    """``simulated`` when the gate shares a machine with a developer's party, else ``development``."""
    at = {p["name"]: p for p in cfg["party"]}
    gate = at[CHALLENGER]["machine"]
    return "simulated" if any(p["machine"] == gate and p["principal"] == "developer" for p in at.values()) \
        else "development"


def live(cfg: Mapping[str, Any], state: Path, here: Sequence[str]) -> Result:
    """The pieces of the parties ``here`` (module docstring): the served window when the workload's party is here, a
    gate's side as started by the gate, and the onsite check when the verifier is here; refused by name (``unbound``)
    before the served window starts while a listed piece's binding is missing."""
    bindings = bindings_of(state)
    if WORKERS in here and (missing := unbound(cfg, bindings)):
        raise Refused("unbound", f"the deploy bindings (verity run --bindings) don't bind {', '.join(missing)}")
    out: dict[str, dict[str, Any]] = {}
    if WORKERS in here:
        rc = served_window(cfg, state, bindings)
        for p in here:
            if any(x.runs == SERVED_WINDOW for x in pieces(cfg, p)):
                out[p] = {"outcome": {"served-window": rc}, "refused": None if rc == 0 else SERVED_WINDOW}
    for p in here:
        if p in out or p == VERIFIER:
            continue
        mine = pieces(cfg, p)
        if any(x.runs == SERVED_WINDOW for x in mine):
            out[p] = {"outcome": {"error": f"{p}'s pieces run inside the served window, whose {WORKERS} this run "
                                           "doesn't run"}, "refused": SERVED_WINDOW}
        else:
            out[p] = {"outcome": {"started": f"by the auditor's gate, for {', '.join(x.component for x in mine)}"},
                      "refused": None}
    if VERIFIER in here:
        try:
            out[VERIFIER] = {"outcome": {"verdicts": [onsite_check(cfg, state)]}, "refused": None}
        except Refused as e:
            out[VERIFIER] = {"outcome": {"error": e.detail}, "refused": e.name}
    return Result(out)


__all__ = ["MANIFESTS", "ORDER", "PARTIES", "PROTOCOLS", "SERVED", "WORKLOADS", "Piece", "Result", "live",
           "onsite_check", "pieces", "served_window", "unbound", "variables"]
