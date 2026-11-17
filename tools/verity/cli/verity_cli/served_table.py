"""The served window's table for `verity run` (`verity_cli.run.Registry`): onsite's and remote's components as a
configuration declares them (`MANIFESTS`), the ``served-window`` workload (`SERVED`), each demo's subprotocols and the
configuration its file holds (`DEMOS`, `configuration`), each party's piece (`PARTIES`), and the run of the pieces placed
on a machine (`live`). Pearl+ has no piece yet, so `verity run` refuses remote by name (``no-party``).

A piece is what one party does in the run (`Piece`), in one of four ways (``runs``):

- ``served-window``: inside the served window's runner, `benchmarks/one_stage/served_verdicts.sh`, which the workload's
  party (``workers``) runs once with every listed component's hooks: the variables it reads for that piece
  (``SERVE_ARGS``, ``H1_GATE``, ...), its parameters from the configuration and its hosts and addresses from the deploy
  bindings (`verity run --bindings`), never from the configuration;
- ``gate``: the auditor's side of a session, which the auditor's gate starts on its host when the prover's piece opens it;
  `verity run` starts nothing for it;
- ``runtime``: a party of its protocol's `ProtocolDef` (`Session`), which `verity run` starts on the machine the
  configuration places it on through the runtime (`verity.runtime.runner.run`), one run of the protocol a piece: the
  wipe's, PoUS's rounds' and PoUB's parties. The auditor's side listens at its ``address`` binding and the prover's side
  dials it; a party whose manifest needs a clock gets the clock its header names (`RealClock(read, name)`). Its
  protocol starts it (`ProtocolDef.starts`: from the configuration, the piece's bindings and the party's state
  directory, its request and its capabilities, the auditor's signer refused first unless the configuration pins it), and
  this machine hands it its party key (`held`). What doesn't reach the table yet (the served store, the card, setup's
  request, the party key) raises `NotImplementedError` (`NOT_YET`) before anything starts;
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

import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

from verity import config as K
from verity.config import Component, Workload
from verity.protocol import Channel, Manifest, ProtocolDef, Role, Starter
from verity.protocols.resource_accounting.network.verifier import parties as NC
from verity.protocols.workload_classification.training_detection.inference_only.verifier import parties as IO
from verity.protocols.resource_accounting.memory.poub.verifier import p2_parties as P2
from verity.protocols.resource_accounting.memory.pous.attester import challenger_session as CS
from verity.protocols.resource_accounting.memory.pous.attester import onsite
from verity.protocols.registration.verifier import registration as RL
from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.resource_accounting.memory.pous.verifier import streamed_wipe as sw
from verity.protocols.resource_accounting.memory.pous.verifier import wipe_parties as WP
from verity.protocols.proofs.spot.verifier import parties as SP
from verity.protocols.proofs.spot.verifier.one_stage.draw import parse_rate, parse_strata_rate
from verity.channel import Refused
from verity.runtime import runner
from verity.runtime.sources import MONOTONIC, RealClock

ROOT = Path(__file__).resolve().parents[4]
SERVED_WINDOW, GATE, RUNTIME, CHECK = "served-window", "gate", "runtime", "check"
RUNS = (SERVED_WINDOW, GATE, RUNTIME, CHECK)
#: what a runtime piece raises while what its machine hands its party doesn't reach the table (`held`)
NOT_YET = "memory-accounting/onsite-memory-verdicts"
RUNNER = "benchmarks/one_stage/served_verdicts.sh"
VERDICTS_SCRIPT = "benchmarks/one_stage/served_verdicts.py"
CERTIFIER_PIECE = "benchmarks/network_traces/certifier_piece.py"
ONSITE = "verity/onsite/v0"
#: the deploy bindings `begin` keeps beside the run (`verity_cli.run.BINDINGS`)
BINDINGS = "bindings.json"
WORKERS, ZK_SERVER, PROVER, VERIFIER = SP.WORKERS, SP.ZK_SERVER, WP.PROVER, SP.VERIFIER
CHALLENGER, MEMORY_ATTESTER = SP.CHALLENGER, WP.CHALLENGER
BANDWIDTH_VERIFIER, NETWORK_ATTESTER = "bandwidth-verifier", NC.NETWORK_CERTIFIER
#: served_verdicts.py's record reads each piece's answer by its runner name
RECORD_NAMES = {"integrity": "sampled-proofs", "work": "pouw", "inference-only": "inference-only", "space": "pous",
                "bandwidth": "bandwidth", "communication": "certifier", "wipe": "wipe"}


# --- what each component and the workload declare -------------------------------------------------------------------

def _design(sampling_design: str, params: Mapping[str, Any]) -> None:
    """The served window's rule is a rate, which its audit states to its stream (`service.SamplingDesign`): a fixed count is refused
    by name, with the rate to state in its place, and so is a work rate, for which the window's class has no work table."""
    strata = parse_strata_rate(sampling_design)
    if parse_rate(sampling_design) is None and (strata is None or strata[0]):
        raise Refused("law", f"the served window's rule is a rate (rate:NUM/DEN or stratified-rate:NUM/DEN), not {sampling_design!r}: its "
                             "audit states it to its stream, which refuses a fixed count (subset:K, stratified:K, work:K), "
                             "since K of N falls as the prover pads the N it registers. In place of a count K, state K over "
                             "the proof blocks N the window's class registers: rate:K/N, or stratified-rate:K/N")


def _pous(params: Mapping[str, Any]) -> dict[str, Any]:
    """PoUS's served piece: ``challenges``, the memory attester's reads in the period (320: ε = 1% + 2^-128,
    ``period.certify``'s, since more reads than a certificate's keep its ε)."""
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


def _named(manifest: Manifest, names: Mapping[str, str]) -> tuple[tuple[Role, ...], tuple[Channel, ...]]:
    """``manifest``'s parties and channels with each of its roles as the configuration names its party (``names``)."""
    def n(role: str) -> str:
        return names.get(role, role)

    return (tuple(replace(r, name=n(r.name)) for r in manifest.parties),
            tuple(replace(c, sender=n(c.sender), receiver=n(c.receiver)) for c in manifest.channels))


#: PoUB's P2 parties and channels, its verifier the configuration's bandwidth verifier (its ``verifier`` is onsite's)
_P2_PARTIES, _P2_CHANNELS = _named(P2.MANIFEST, {P2.VERIFIER: BANDWIDTH_VERIFIER})
MANIFESTS: dict[str, Component] = {c.name: c for c in (
    Component("sampled-proofs", SP.IDS["sampled-proofs"], parties=SP.MANIFEST.parties, channels=SP.MANIFEST.channels,
              needs=frozenset({"windows"}), sampling_design=_design, default_sampling_design="rate:1/3481648"),
    Component("pouw", SP.IDS["pouw"], parties=(Role(CHALLENGER, "auditor", needs=("host",)),
                                               Role(WORKERS, "developer", needs=("serve",)), _VERIFIER),
              channels=(Channel(CHALLENGER, WORKERS, ("H1Coins",), carries_coins=True),
                        Channel(WORKERS, VERIFIER, ("H1Record",), after=("H1Coins",))),
              needs=frozenset({"windows", "noise"}), requires=("sampled-proofs",)),
    Component("pearl-plus", SP.IDS["pearl-plus"], parties=(_VERIFIER,), needs=frozenset({"windows"})),
    Component("inference-only", SP.IDS["inference-only"], parties=IO.MANIFEST.parties, requires=("sampled-proofs",)),
    Component("pous", SP.IDS["pous"], parties=(*CS.MANIFEST.parties, _VERIFIER),
              channels=(*CS.MANIFEST.channels, Channel(MEMORY_ATTESTER, VERIFIER, ("SignedRun",), after=("End",))),
              needs=frozenset({"store"}), params=_pous, default_params={"challenges": 320}),
    Component("bandwidth", SP.IDS["bandwidth"], parties=(*_P2_PARTIES, _VERIFIER),
              channels=(*_P2_CHANNELS, Channel(BANDWIDTH_VERIFIER, VERIFIER, ("BandwidthAnswer",), after=("Openings",))),
              needs=frozenset({"receipts"})),
    Component("certifier", SP.IDS["certifier"], parties=NC.MANIFEST.parties, channels=NC.MANIFEST.channels,
              needs=frozenset({"windows", "ingress-relay"}), requires=("sampled-proofs",),
              colocated=((CHALLENGER, NETWORK_ATTESTER),)),
    Component("wipe", "verity/wipe/v0", parties=(*WP.MANIFEST.parties, _VERIFIER),
              channels=(*WP.MANIFEST.channels, Channel(MEMORY_ATTESTER, VERIFIER, ("WipeReport",), after=("End",)))),
)}
SERVED = Workload(SERVED_WINDOW, frozenset({"windows", "noise", "store", "receipts", "ingress-relay"}),
                  (Role(WORKERS, "developer", needs=("serve",)),), _served,
                  {"model": "verity-int8/Llama-3.2-1B-INT8", "revision": "15b0976d400d796ee589e7737d03a919526bf800",
                   "precision": "verity_int8", "slots": 4, "window-steps": 16, "load": "varied"})
#: onsite's subprotocols, the seven `Core.Composites.composites` holds for it
ONSITE_COMPONENTS = ("sampled-proofs", "pouw", "inference-only", "pous", "bandwidth", "certifier", "wipe")
PROTOCOLS = ONSITE_COMPONENTS
#: each demo's subprotocols, the file `demos/NAME.toml` that `verity demo NAME` runs: onsite's seven, its two smaller
#: sets (`verity.config.UNACCEPTED`) and remote's
DEMOS = {"onsite_full": ONSITE_COMPONENTS,
         "onsite_compute": ("sampled-proofs", "pouw", "inference-only", "certifier", "wipe"),
         "onsite_network": ("sampled-proofs", "certifier", "wipe"),
         "remote": ("sampled-proofs", "pearl-plus", "inference-only")}
PROVER_NODE, AUDITOR_HOST, GATE_HOST = "prover-node", "auditor-host", "gate-host"
#: remote's auditor parties' machine, which isn't at the site: remote puts no auditor's device there
AUDITOR_REMOTE = "auditor-remote"
#: each party's machine: the developer's on the prover node, the auditor's devices on machines it doesn't control
PLACEMENT = {WORKERS: PROVER_NODE, ZK_SERVER: PROVER_NODE, PROVER: PROVER_NODE, CHALLENGER: GATE_HOST,
             NETWORK_ATTESTER: GATE_HOST, MEMORY_ATTESTER: AUDITOR_HOST, BANDWIDTH_VERIFIER: AUDITOR_HOST,
             VERIFIER: AUDITOR_HOST}
#: the configuration this table's demo runs, whose `[[pin]]` entries are the only statement of the keys the auditor pinned
CONFIG = ROOT / "demos" / "onsite_full.toml"
_NOT_YET = "`verity run` stops on it with NotImplementedError until goal demos/onsite-demo brings it."
#: each demo file's header
HEADERS = {
    "onsite_full": (
        "The onsite run of the served window: onsite's seven subprotocols, each party on its machine, and the keys",
        "the auditor pinned for its devices. Run it on each machine with `verity run THIS --on MACHINE --bindings",
        "FILE` (tools/verity/cli/README.md); the deploy bindings hold every host and address, this file none.",
        "`verity run THIS --dry-run` resolves it all."),
    "onsite_compute": (
        "The served window under SPOT, PoUW, the inference-only policy, the network certifier and the memory",
        "wipe (verity/onsite-compute/v0, a deployment of its own, never onsite): onsite without PoUS and PoUB,",
        "each party on its machine, and the key the auditor pinned for its memory attester.", _NOT_YET),
    "onsite_network": (
        "The served window under the network certifier, SPOT and the memory wipe (verity/onsite-network/v0, a",
        "deployment of its own, never onsite), each party on its machine, and the key the auditor pinned for",
        "its memory attester.", _NOT_YET),
    "remote": (
        "The served window under remote's subprotocols (verity/remote/v1): SPOT, PoUW's Pearl+ run verdict and the",
        "inference-only policy. No auditor's device is at the site: the developer's parties are on the prover node",
        "and the auditor's on its own machine. `verity run` refuses it by name until each party has its piece."),
}
HEADER = HEADERS["onsite_full"]


def configuration(commit: str, demo: str = "onsite_full",
                  pins: Sequence[Mapping[str, str]] | None = None) -> dict[str, Any]:
    """The configuration `demos/<demo>.toml` holds, at program ``commit``: `config new`'s over the demo's subprotocols
    (`DEMOS`), each party at its `PLACEMENT` (remote's auditor parties at `AUDITOR_REMOTE`), the workload's and PoUS's
    parameters stated, and ``pins``, the keys the auditor pinned for its devices (by default `CONFIG`'s own `[[pin]]`
    entries for the subprotocols the demo lists: no code states a key)."""
    protocols = DEMOS[demo]
    cfg = K.new(SERVED_WINDOW, list(protocols), MANIFESTS, {SERVED_WINDOW: SERVED}, commit)
    cfg["workload"]["params"] = dict(SERVED.default_params)
    if "pous" in protocols:
        cfg["params"] = {SP.IDS["pous"]: {"challenges": 320}}
    for p in cfg["party"]:
        remote = demo == "remote" and p["principal"] == "auditor"
        p["machine"] = AUDITOR_REMOTE if remote else PLACEMENT[p["name"]]
    if pins is None:
        pins = [p for p in tomllib.loads(CONFIG.read_text()).get("pin", []) if p["protocol"] in protocols]
    cfg["pin"] = [dict(p) for p in pins]
    K.check(cfg, MANIFESTS, {SERVED_WINDOW: SERVED})
    return cfg


# --- the pieces ------------------------------------------------------------------------------------------------------

Hooks = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, str]], dict[str, str]]


def _none(cfg: Mapping[str, Any], params: Mapping[str, Any], bound: Mapping[str, str]) -> dict[str, str]:
    return {}


def _protocol_def(ref: str) -> ProtocolDef:
    """The `ProtocolDef` ``module:attribute`` names, or the one value of the mapping it names (a module's
    ``PROTOCOLS``)."""
    module, attribute = ref.split(":")
    got = getattr(importlib.import_module(module), attribute)
    if isinstance(got, Mapping):
        (got,) = got.values()
    return got


@dataclass(frozen=True)
class Session:
    """A protocol whose parties the table starts through the runtime (module docstring): its `ProtocolDef`'s halves,
    each a ``module:attribute`` (`_protocol_def`), whose manifests are one; each configuration party's role in it
    (``roles``); for each role that reads a clock, the read and the name its header gives (``clocks``); the
    configuration party that listens (``listener``, the auditor's), and whether the run ends before the served window
    starts (``before``) or runs beside it. ``reads`` gives a configuration party's start, beside its bindings, the state
    directory of another party's run on the same machine: binding name -> (component, party). ``starts`` are a role's
    starts the table makes, over its halves': a start handed what its protocol's code may not load."""

    halves: tuple[str, ...]
    roles: Mapping[str, str]
    clocks: Mapping[str, tuple[Callable[[], int], str]]
    listener: str
    before: bool = False
    reads: Mapping[str, Mapping[str, tuple[str, str]]] = field(default_factory=dict)
    starts: Mapping[str, Starter] = field(default_factory=dict)

    def protocol(self) -> ProtocolDef:
        defs = [_protocol_def(h) for h in self.halves]
        if any(d.manifest != defs[0].manifest for d in defs):
            raise ValueError(f"{', '.join(self.halves)} are halves of more than one manifest")
        return ProtocolDef(defs[0].manifest, {r: f for d in defs for r, f in d.parties.items()},
                           {**{r: f for d in defs for r, f in d.starts.items()}, **self.starts})

    def absent(self) -> list[str]:
        """The halves' modules that aren't in this tree."""
        return [h.split(":")[0] for h in self.halves if importlib.util.find_spec(h.split(":")[0]) is None]


class Held(NamedTuple):
    """What a machine hands a runtime party beside what its protocol's start gives it: its party key and the check of
    its peer's."""

    key: Any
    verify: Callable[[bytes, bytes, bytes], bool]


def held(piece: Piece, role: Role) -> Held:
    """What this machine hands ``role``, ``piece``'s party: its party key, which the channels authenticate. It doesn't
    reach the table yet."""
    raise NotImplementedError(NOT_YET)


@dataclass(frozen=True)
class Piece:
    """One party's piece of the served window (module docstring): ``hooks(cfg, params, bound)`` gives the runner's
    variables for it from the configuration, the component's checked parameters and the bindings of the party it
    reaches (``bound``: each of ``binds``, a ``<party.component.key>`` placeholder while unbound, and each bound one of
    ``optional``). A ``runtime`` piece's ``session`` is the protocol it is a party of."""

    component: str
    party: str
    runs: str
    what: str
    scripts: tuple[str, ...] = ()
    reaches: str = ""
    binds: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    hooks: Hooks = _none
    session: Session | None = None

    def __post_init__(self) -> None:
        if self.runs not in RUNS:
            raise ValueError(f"a piece runs as one of {RUNS}, not {self.runs!r}")
        if (self.runs == RUNTIME) != (self.session is not None):
            raise ValueError(f"a piece has a session exactly when it runs as {RUNTIME!r}")

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
        script. A runtime piece adds its party: its protocol, its role, its machine, the clock it reads, what its
        manifest says it needs, its channels with the configuration's names, the address it listens at or dials, and
        what it raises; a protocol whose halves aren't all in this tree is ``absent``, by module."""
        out = {"component": self.component, "runs": self.runs, "what": self.what, "scripts": list(self.scripts),
               "reaches": self.reaches or self.party, "missing": self.missing(),
               "hooks": self.variables(cfg, bindings), "unbound": self.unbound(bindings)}
        if self.session is None:
            return out
        s, machine = self.session, {p["name"]: p["machine"] for p in cfg["party"]}[self.party]
        role, address = s.roles[self.party], self._bindings(bindings).get("address", f"<{self.key('address')}>")
        party = {"role": role, "machine": machine, "clock": s.clocks[role][1] if role in s.clocks else None,
                 "listens" if self.party == s.listener else "dials": address, "raises": NOT_YET}
        if absent := s.absent():
            return {**out, **party, "protocol": None, "absent": absent}
        m, name = s.protocol().manifest, {r: c for c, r in s.roles.items()}
        return {**out, **party, "protocol": m.protocol, "needs": list(m.role(role).needs),
                "channels": [{"from": name[c.sender], "to": name[c.receiver], "messages": list(c.messages),
                              "carries_coins": c.carries_coins} for c in m.channels if role in (c.sender, c.receiver)]}


def _serve(cfg, params, bound):
    w = cfg["workload"]["params"]
    return {"REPO": w["model"], "REV": w["revision"], "PRECISION": w["precision"], "SLOTS": str(w["slots"]),
            "WINDOW_STEPS": str(w["window-steps"]), "LOAD": w["load"], "V3C_LAW": cfg["service"]["law"]}


#: served_verdicts.sh's variable for each optional binding of SPOT's live challenger
_PP = {"gate": "PP_CHALLENGER", "peer": "PP_PEER", "key": "PP_CHALLENGER_KEY", "gate-key": "PP_GATE_KEY"}


def _spot_challenger(cfg, params, bound):
    """The live challenger's bindings; with no gate bound the sample's coins are a local development session's, which
    served_verdicts.sh takes only in a development run."""
    hooks = {v: f"gate:{bound[k]}" if k == "gate" else bound[k] for k, v in _PP.items() if k in bound}
    return hooks if "gate" in bound else {**hooks, "DEVELOPMENT": "1"}


def _pouw(cfg, params, bound):
    return {"H1_GATE": bound["gate"], "H1_PEER": bound["peer"], "H1_KEY_SHA512": bound["key-sha512"]}


#: the wipe (`streamed_wipe`'s parties), before the card's engine loads: its challenger reads the raw monotonic clock its
#: header names
WIPE = Session(("verity.protocols.resource_accounting.memory.pous.attester.wipe_challenger:PROTOCOL",
                "verity.protocols.resource_accounting.memory.pous.prover.wipe_prover:PROTOCOL"),
               {MEMORY_ATTESTER: WP.CHALLENGER, PROVER: WP.PROVER}, {WP.CHALLENGER: (sw.raw_clock_ns, sw.CLOCK)},
               MEMORY_ATTESTER, before=True)
#: PoUS's timed rounds (`challenger_session`'s parties), beside the served window: the challenger's header names the
#: monotonic clock
POUS = Session(("verity.protocols.resource_accounting.memory.pous.attester.challenger_party:PROTOCOL",
                "verity.protocols.resource_accounting.memory.pous.prover.challenger_prover:PROTOCOL"),
               {MEMORY_ATTESTER: CS.CHALLENGER, PROVER: CS.PROVER}, {CS.CHALLENGER: (time.monotonic_ns, MONOTONIC)},
               MEMORY_ATTESTER)
#: onsite PoUB over PoUS's P2 store (`p2_parties`' verifier, `p2_party_prover`'s prover), beside the served window, on
#: the verifier's monotonic clock; the verifier reads the memory attester's PoUS key from that party's run beside it
#: on the auditor's device (``challenger``), checked against the configuration's pin by PoUS's device (`onsite.pin`,
#: `onsite.signer`), which the table hands its start
POUB = Session(("verity.protocols.resource_accounting.memory.poub.verifier.p2_parties:PROTOCOLS",
                "verity.protocols.resource_accounting.memory.poub.prover.p2_party_prover:PROTOCOL"),
               {BANDWIDTH_VERIFIER: P2.VERIFIER, PROVER: P2.PROVER}, {P2.VERIFIER: (time.monotonic_ns, MONOTONIC)},
               BANDWIDTH_VERIFIER, reads={BANDWIDTH_VERIFIER: {"challenger": ("pous", MEMORY_ATTESTER)}},
               starts=P2.starts(onsite.pin, onsite.signer))


#: what the network attester's front asks of serve-openai: its id-form requests answered in the id form, an id event the
#: 503-byte payload of the front's 512-byte cells; every other request is answered as before
ID_FORM = "--protocol ids --id-payload 503"
#: network attestation's scan class (`certifier_piece.V3C_SCAN_CLASS`, w63k4) as served_verdicts.sh's SLOTS,PROMPT,OUTPUTS,
#: LANES,EVERY, V3c's class, so the period check's compute audit is V3c's of the served window; CLASSES holds it beside
#: served_verdicts.sh's default class
SCAN_CLASS = "8,32,4,4,16"


def _certifier_piece(cfg, params, bound):
    py = ".venv/bin/python"
    return {"SERVE_ARGS": ID_FORM,
            "CERTIFIER_FRONT": f'{py} {CERTIFIER_PIECE} front "$1" "$2" "$3" --gate {shlex.quote(bound["gate"])}',
            "CERTIFIER_CHAIN_CMD": f'{py} {CERTIFIER_PIECE} chain "$@"',
            "CERTIFIER_PERIOD_CMD": f'{py} {CERTIFIER_PIECE} period "$@"',
            "V3C_CLASS": SCAN_CLASS, "CLASSES": f"4,32,4 {SCAN_CLASS}"}


SERVE = Piece(SERVED_WINDOW, WORKERS, SERVED_WINDOW, "Execution: serve-openai on the prover node under its GPU lease, the "
              "requests, every closed window sealed and the audited window picked; then SPOT's records audit "
              "(Flock --zk behind the Lean ZK server, the Lean verifier) and each listed component's piece",
              scripts=(RUNNER, VERDICTS_SCRIPT), hooks=_serve)
#: each listed component's party -> its piece
PARTIES: dict[str, dict[str, Piece]] = {
    "sampled-proofs": {
        CHALLENGER: Piece("sampled-proofs", CHALLENGER, GATE, "the gate: the auditor's live challenger, whose coins "
                          "sample the proof blocks after the window's commitment"),
        ZK_SERVER: Piece("sampled-proofs", ZK_SERVER, SERVED_WINDOW, "the Lean ZK server every outer proof goes through, "
                        "one at a time, in the records audit"),
        WORKERS: Piece("sampled-proofs", WORKERS, SERVED_WINDOW, "the window's values committed under hiding leaves "
                       "and sealed, its registration before the sample, and the sampled proof blocks proved in Flock --zk; "
                       "the sample on the live challenger when its gate is bound, else on a local development session's "
                       "coins in a development run", reaches=CHALLENGER, optional=tuple(_PP), hooks=_spot_challenger),
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
        PROVER: Piece("pous", PROVER, RUNTIME, "PoUS's rounds' prover: the memory attester's timed reads answered "
                      "from the served weights held as P2's C* on the card", reaches=MEMORY_ATTESTER,
                      binds=("address",), session=POUS),
        MEMORY_ATTESTER: Piece("pous", MEMORY_ATTESTER, RUNTIME, "the memory attester's side, on its own coins "
                                 "and clock, signing the session", binds=("address",), session=POUS),
        VERIFIER: Piece("pous", VERIFIER, CHECK, "PoUS's row"),
    },
    "bandwidth": {
        PROVER: Piece("bandwidth", PROVER, RUNTIME, "PoUB's prover: the served store proved to the bandwidth verifier a "
                      "pass at a time", reaches=BANDWIDTH_VERIFIER, binds=("address",), session=POUB),
        BANDWIDTH_VERIFIER: Piece("bandwidth", BANDWIDTH_VERIFIER, RUNTIME, "the bandwidth verifier and its "
                                  "challenger, on its own coins and clock", binds=("address",), session=POUB),
        VERIFIER: Piece("bandwidth", VERIFIER, CHECK, "PoUB's row"),
    },
    "certifier": {
        WORKERS: Piece("certifier", WORKERS, SERVED_WINDOW, "the certifier's developer side: its front before the "
                       "requests (CERTIFIER_FRONT, which the served window runs), and its piece split around V3c: the "
                       "chain on V3c's committed window (CERTIFIER_CHAIN_CMD), then the period check on V3c's "
                       "period.json (CERTIFIER_PERIOD_CMD)", scripts=(CERTIFIER_PIECE,), reaches=NETWORK_ATTESTER,
                       binds=("gate",), hooks=_certifier_piece),
        NETWORK_ATTESTER: Piece("certifier", NETWORK_ATTESTER, GATE, "the network attester behind the gate, "
                                 "in front of serve-openai before the requests"),
        VERIFIER: Piece("certifier", VERIFIER, CHECK, "the network attester's row"),
    },
    "wipe": {
        PROVER: Piece("wipe", PROVER, RUNTIME, "the streamed wipe of the card before its engine loads, its session "
                      "trusted only under the memory attester's pinned key", reaches=MEMORY_ATTESTER,
                      binds=("address",), session=WIPE),
        MEMORY_ATTESTER: Piece("wipe", MEMORY_ATTESTER, RUNTIME, "the wipe's memory attester, on its own coins "
                                 "and raw clock, signing the session", binds=("address",), session=WIPE),
        VERIFIER: Piece("wipe", VERIFIER, CHECK, "the wipe's row"),
    },
}
WORKLOADS: dict[str, tuple[Workload, Piece]] = {SERVED_WINDOW: (SERVED, SERVE)}
#: no party here runs as an entry of its own: every piece is the served window's, the gate's, the runtime's or the check's
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


#: the runner's variables that are serve-openai's flags, which each piece adds its own to
JOINED = ("SERVE_ARGS",)


def variables(cfg: Mapping[str, Any], bindings: Mapping[str, Any]) -> dict[str, str]:
    """The served window's runner's variables: the configuration's hash and its `[[pin]]` entries
    (``VERITY_CONFIGURATION``, ``VERITY_PINS``), which every device's `trusted-device` premise reads, and every listed
    piece's hooks, whichever party's they are, since the prover's pieces reach the auditor's sides through their
    bindings. A variable of `JOINED` is every piece's flags,
    in the pieces' order; any other is one piece's."""
    out = {"VERITY_CONFIGURATION": K.digest(cfg), "VERITY_PINS": json.dumps(cfg.get("pin", []), sort_keys=True)}
    for p in [SERVE, *(x for c in cfg["protocols"] for x in PARTIES.get(c, {}).values())]:
        for k, v in p.variables(cfg, bindings).items():
            out[k] = f"{out[k]} {v}" if k in JOINED and k in out else v
    return out


def unbound(cfg: Mapping[str, Any], bindings: Mapping[str, Any]) -> list[str]:
    """Every listed piece's unbound bindings, each once: a runtime session's two sides read one ``address``."""
    return list(dict.fromkeys(k for p in [SERVE, *(x for c in cfg["protocols"] for x in PARTIES.get(c, {}).values())]
                              for k in p.unbound(bindings)))


def deployment(cfg: Mapping[str, Any], piece: Piece, protocol: ProtocolDef, address: str,
               params: Mapping[str, Any]) -> dict[str, Any]:
    """``piece``'s protocol run as the runtime reads it: under the onsite configuration's hash, with the run's request,
    each of its parties by its role there on the configuration's machine, the listener with its address."""
    s, at = piece.session, {p["name"]: p for p in cfg["party"]}
    return {"protocol": protocol.manifest.protocol, "hash": K.digest(cfg), "params": dict(params),
            "party": [{"name": r, "principal": at[c]["principal"], "machine": at[c]["machine"],
                       **({"listen": address} if c == s.listener else {})} for c, r in sorted(s.roles.items())]}


def request_bound(protocol: ProtocolDef, params: Mapping[str, Any]) -> None:
    """That the two sides of ``protocol``'s run start on one request, ``params`` this side's: each side's start computes
    the request on its own machine, and the run's handshake compares only the configuration's hash, the onsite
    configuration's that the deployment states (`deployment`) so that a signed session names it. The two sides would
    compare SHA-512 of the request's canonical JSON at the session's start; the runtime's ``Hello``
    (`verity/runtime/handshake.py`) has no field for it, and neither side's messages carry it, so a run raises here
    (`NOT_YET`) until one does."""
    raise NotImplementedError(NOT_YET)


def prepare(cfg: Mapping[str, Any], state: Path, bindings: Mapping[str, Any], piece: Piece) -> Callable[[], Any]:
    """``piece``'s party run through the runtime (`verity.runtime.runner.run`), ready to start: its `ProtocolDef`, its
    protocol's start (`ProtocolDef.starts`: its request and capabilities, from the configuration, the piece's bindings and
    its state directory ``state/runs/<component>/<party>``), the two sides' request bound (`request_bound`), what this
    machine hands it (`held`), its clock, and the listener's address, which it listens at or dials. Refused ``no-piece`` while a half of its protocol isn't in this
    tree; a party its protocol declares no start for raises `NOT_YET`."""
    s = piece.session
    if absent := s.absent():
        raise Refused("no-piece", f"{piece.party}'s {piece.component} piece: {', '.join(absent)} is not in this tree")
    protocol = s.protocol()
    role = protocol.manifest.role(s.roles[piece.party])
    if role.name not in protocol.starts:
        raise NotImplementedError(NOT_YET)
    here = state / "runs" / piece.component / piece.party
    bound = {**piece._bindings(bindings),
             **{k: str(state / "runs" / c / p) for k, (c, p) in s.reads.get(piece.party, {}).items()}}
    begun = protocol.starts[role.name](cfg, bound, here)
    request_bound(protocol, begun.params)
    h = held(piece, role)
    address, listener = bound["address"], s.roles[s.listener]
    sources = {role.name: {"clock": RealClock(*s.clocks[role.name])}} if "clock" in role.needs else None
    return lambda: runner.run(deployment(cfg, piece, protocol, address, begun.params), [role.name], "live", here,
                              protocols={protocol.manifest.protocol: protocol}, keys={role.name: h.key},
                              verify=h.verify, sources=sources, capabilities={role.name: begun.capabilities},
                              dial_to=None if role.name == listener else {listener: address}).outcomes[role.name]


def placement(cfg: Mapping[str, Any], piece: Piece, outcome: Mapping[str, Any] | None) -> dict[str, str] | None:
    """`trusted-device`'s premise for a memory attester's ``piece``, from its outcome: the key its session is signed
    under (the outcome's ``public_key``) against the pins the configuration gives that party for that protocol, under the
    configuration's hash (`memory_attester.trusted_device`, `committed` its committed set). ``None`` for a party that
    signs no session; a session that reports no key reads rehearsal."""
    if piece.party != MEMORY_ATTESTER:
        return None
    key = (outcome or {}).get("public_key")
    machine = {p["name"]: p["machine"] for p in cfg["party"]}[piece.party]
    return {mc.PREMISE: mc.trusted_device(cfg.get("pin", []), piece.party, piece.component,
                                          None if key is None else hashlib.sha512(bytes.fromhex(key)).hexdigest(),
                                          f"the {piece.party} ran on {machine}", configuration=K.digest(cfg),
                                          committed=mc.committed())}


def composite(env: Mapping[str, str]) -> dict[str, str]:
    """The runner's ``COMPOSITE``, set empty: it composes off the served node unless told otherwise, where onsite's
    composite, which reads PoUS's and the wipe's cases that only the node's card writes, can't be made, and it refuses
    to start. A caller whose environment ``env`` names ``COMPOSITE`` or ``COMPOSE_OFF_NODE=0`` (a development composite
    on the node) keeps what it named."""
    return {} if "COMPOSITE" in env or env.get("COMPOSE_OFF_NODE") == "0" else {"COMPOSITE": ""}


def served_window(cfg: Mapping[str, Any], state: Path, bindings: Mapping[str, Any]) -> int:
    """The served window's runner with every listed piece's hooks and no composite unless the caller asks for one
    (`composite`), its public record in ``state/out``; its serve dir stays where the runner keeps it on the node (its
    ``D`` and ``SERVE_ROOT``, from the environment)."""
    env = {**os.environ, **variables(cfg, bindings), **composite(os.environ), "OUT": str(state / "out")}
    (state / "logs").mkdir(parents=True, exist_ok=True)
    with open(state / "logs" / "served-window.log", "w") as log:
        return subprocess.run(["bash", str(ROOT / RUNNER)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                              check=False).returncode


VERDICTS = {"accept": "accept", "reject": "reject", "pending": "undecided"}


def onsite_check(cfg: Mapping[str, Any], state: Path) -> dict[str, Any]:
    """The verifier's onsite check: `served_verdicts.py record` over the served window's public record in ``state``
    (``out/``, all a verifier on another machine needs of it, the prover node named by its ``machine.txt``); the
    period's line in `verity_cli.run`'s verdict form, undecided while the record has no row for a listed subprotocol.
    Where the configuration's product is onsite's and the composite's public output is there (``out/composite``,
    `served_composite.sh`'s), the record reads it (``--composite``), so its verdict is the compiled Lean onsite
    check's answer as the record takes it; an undecided or rejecting line's reason, short of a row's, is the record's
    ``lean``: where that answer was read or why none was taken."""
    py, out = sys.executable, state / "out"
    serve, machine = out / "serve", out / "machine.txt"
    product = K.composite(cfg["protocols"], MANIFESTS)
    lean = ["--composite", str(out / "composite")] if product == ONSITE and (out / "composite").is_dir() else []
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
                        *node, *lean, "--out", str(state / "record" / "onsite")], cwd=ROOT, capture_output=True,
                       text=True, check=False)
    line = next((json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("RECORD ")), None)
    if line is None:
        raise Refused("record", (p.stderr or p.stdout).strip()[-800:] or "served_verdicts.py record wrote no RECORD")
    names = {v: k for k, v in RECORD_NAMES.items()}
    rows = json.loads((state / "record" / "onsite" / "verdicts.json").read_text())["protocols"]
    components = {c: rows.get(names[c], {}).get("verdict", "absent") for c in cfg["protocols"]}
    absent = sorted(c for c, v in components.items() if v == "absent")
    verdict = VERDICTS.get(line["composed"], "undecided")
    return {"block": k, "components": components, "product": product,
            "reason": line.get("why") or (f"missing {sorted(line['missing'])}" if line.get("missing") else None)
            or (f"no row for {absent}" if absent else None) or (line.get("lean") if verdict != "accept" else None),
            "registration": None, "coins": coins_of(cfg),
            "verdict": "undecided" if verdict == "accept" and absent else verdict}


#: what a registration raises with once its configuration item is written, while no run's log holds the item: the goal
#: that brings it
REGISTRATION_GOAL = "memory-accounting/architecture-on-branch"


def registration(cfg: Mapping[str, Any], state: Path) -> str:
    """The head of the run's registration log, the VERDICT line's ``registration``, once an information entry of it
    registers the configuration's hash (the item ``{"kind": "configuration", "digest": K.digest(cfg)}``), before any
    party starts and so before any coin: `live` calls it after it has prepared the parties here and before it starts
    one. The hash is ``state``'s ``run.json``'s ``configuration``, which `verity run` writes first; a ``run.json`` under
    another hash, or none, is refused (``registration``). The item is the registration log's own
    (`verity.protocols.registration.verifier.registration.configuration`), whose writer raises
    `NotImplementedError` (``proofs/architecture-on-branch``) while the log has no coin to order it before; once it
    returns, the run's log holding it beside θ and the builder isn't written either (`REGISTRATION_GOAL`). Until then
    the line's ``registration`` is ``None``."""
    f, digest = state / "run.json", K.digest(cfg)
    run = json.loads(f.read_text()) if f.is_file() else {}
    if run.get("configuration") != digest:
        raise Refused("registration", f"{f} names configuration {str(run.get('configuration'))[:16]}…, not this "
                                      f"configuration's {digest[:16]}…: the registration names the run's own hash")
    RL.configuration(digest)
    raise NotImplementedError(REGISTRATION_GOAL)


def coins_of(cfg: Mapping[str, Any]) -> str:
    """``simulated`` when the gate shares a machine with a developer's party, else ``development``."""
    at = {p["name"]: p for p in cfg["party"]}
    gate = at[CHALLENGER]["machine"]
    return "simulated" if any(p["machine"] == gate and p["principal"] == "developer" for p in at.values()) \
        else "development"


def live(cfg: Mapping[str, Any], state: Path, here: Sequence[str]) -> Result:
    """The pieces of the parties ``here`` (module docstring): each runtime party's run, the wipe's before the served
    window and the others' beside it; the served window when the workload's party is here; a gate's side as started by
    the gate; and the onsite check when the verifier is here, its line with each memory attester's `trusted-device`
    premise as `placement` read it from the challenger's outcome here (``trusted-device``: component -> premise). Refused by name (``unbound``) while a listed piece's
    binding is missing, and raised while a runtime party's `held` does, before anything starts."""
    bindings = bindings_of(state)
    runtime = [x for p in here for x in pieces(cfg, p) if x.runs == RUNTIME]
    if missing := unbound(cfg, bindings) if WORKERS in here else list(dict.fromkeys(
            k for x in runtime for k in x.unbound(bindings))):
        raise Refused("unbound", f"the deploy bindings (verity run --bindings) don't bind {', '.join(missing)}")
    ready = [(x, prepare(cfg, state, bindings, x)) for x in runtime]
    if ready:
        registration(cfg, state)
    out: dict[str, dict[str, Any]] = {}

    def done(x: Piece, got: Mapping[str, Any]) -> None:
        mine = out.setdefault(x.party, {"outcome": {}, "refused": None})
        mine["outcome"][x.component] = got.get("outcome")
        mine["refused"] = mine["refused"] or got.get("refused")
        if (where := placement(cfg, x, got.get("outcome"))) is not None:
            mine.setdefault("placement", {})[x.component] = where

    for x, start in ready:
        if x.session.before:
            done(x, start())
    with ThreadPoolExecutor(max_workers=max(1, len(ready))) as pool:
        beside = [(x, pool.submit(start)) for x, start in ready if not x.session.before]
        if WORKERS in here:
            rc = served_window(cfg, state, bindings)
            for p in here:
                if any(x.runs == SERVED_WINDOW for x in pieces(cfg, p)):
                    out[p] = {"outcome": {"served-window": rc}, "refused": None if rc == 0 else SERVED_WINDOW}
        for x, f in beside:
            done(x, f.result())
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
        devices = {c: where[mc.PREMISE] for x in out.values() for c, where in x.get("placement", {}).items()}
        try:
            out[VERIFIER] = {"outcome": {"verdicts": [{**onsite_check(cfg, state), mc.PREMISE: devices}]},
                             "refused": None}
        except Refused as e:
            out[VERIFIER] = {"outcome": {"error": e.detail}, "refused": e.name}
    return Result(out)


__all__ = ["DEMOS", "HEADERS", "MANIFESTS", "NOT_YET", "ORDER", "PARTIES", "PROTOCOLS", "REGISTRATION_GOAL", "SERVED",
           "WORKLOADS", "Held", "Piece", "Result", "Session", "held", "live", "onsite_check", "pieces", "placement",
           "prepare", "registration", "request_bound", "served_window", "unbound", "variables"]
