"""The served window's table for `verity run` (`verity_cli.run.Registry`): onsite's and remote's components as a
configuration declares them (`MANIFESTS`), the ``served-window`` workload (`SERVED`), each party's piece (`PIECES`), the
parties `verity run` starts first (`STARTS`), and the run of the pieces placed on a machine (`live`).

A piece is what one party does in the run (`Piece`), in one of four ways (``runs``):

- ``served-window``: inside the served window's runner, `benchmarks/one_stage/served_verdicts.sh`, which the workload's
  party (``workers``) runs once with every listed component's hooks: the variables it reads for that piece
  (``SERVE_ARGS``, ``H1_GATE``, ...), its parameters from the configuration and its hosts and addresses from the deploy
  bindings (`verity run --bindings`), never from the configuration;
- ``gate``: the auditor's side of a session, which the auditor's gate starts on its host when the prover's piece opens it;
  `verity run` starts nothing for it;
- ``runtime``: a party of its protocol's `ProtocolDef` (`Session`), which `verity run` starts on the machine the
  configuration places it on through the runtime (`verity.runtime.runner.run`), one run of the protocol a piece (no
  listed piece runs so now: `POUS` and `POUB` are the sessions it would start). The auditor's side listens at its ``address`` binding and the prover's side
  dials it; a party whose manifest needs a clock gets the clock its header names (`RealClock(read, name)`). Its
  protocol starts it (`ProtocolDef.starts`: from the configuration, the piece's bindings and the party's state
  directory, its request and its capabilities, the auditor's signer refused first unless the configuration pins it), and
  this machine hands it its party key (`held`). What doesn't reach the table yet (the served store, the card, setup's
  request, the party key) raises `NotImplementedError` (`NOT_YET`), which `live` refuses (``unwritten``) before anything starts;
- ``check``: the verifier's onsite check, `served_verdicts.py record` over the served window's public record, whose
  composed verdict is the period's; under remote, the record the compose answer holds, else the
  served node's own over its composite (`remote_check`). PoUW's row
  there is Pearl+'s verdict under ``pearl-plus`` (the runner's ``POUW=pearlplus``), H1's under ``pouw``.

A piece names the scripts it runs (``scripts``), which must be in the tree, and the deploy bindings it reads: those of
the party it reaches (``reaches``, default its own), ``party.component.key`` each, the ``binds`` ones bound before the run
starts (``unbound``) and the ``optional`` ones used when bound. `describe` is what `verity run --dry-run` prints of it.

`live` runs one monitoring period of `run` (`verity.protocols.run.period`), each stage's body here (`STAGES`).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

from verity import config as K
from verity.commitments.crypto import lms
from verity.config import Component, Workload
from verity.protocol import Channel, ProtocolDef, Role, Starter
from verity.protocols.run import period, schemes
from verity.lean_process import lean_env
from verity.protocols.accounting.network.verifier import parties as NC
from verity.protocols.accounting.scheme import (Executable, Lacks, Launch, LaunchRefused, Scheme, declared,
                                                             settings)
from verity.protocols.accounting.memory.poub.verifier import p2_parties as P2
from verity.protocols.accounting.memory.pous.attester import challenger_session as CS
from verity.protocols.registration.verifier import registration as RL
from verity.protocols.accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.accounting.memory.wipe.verifier import wipe_parties as WP
from verity.protocols.proof.spot.prover.audit import local_challenger as LC
from verity.protocols.proof.spot.verifier import parties as SP
from verity.protocols.proof.spot.verifier.challenger import challenger_party as CP
from verity.commitments.randomness import parse_rate, parse_strata_rate
from verity.channel import Refused
from verity.runtime import runner
from verity.runtime.sources import MONOTONIC, RealClock
from verity_cli import settings as machine_settings
from verity_cli.run import Start

ROOT = Path(__file__).resolve().parents[4]
SERVED_WINDOW, GATE, RUNTIME, CHECK = "served-window", "gate", "runtime", "check"
RUNS = (SERVED_WINDOW, GATE, RUNTIME, CHECK)
#: what a runtime piece raises while what its machine hands its party doesn't reach the table (`held`)
NOT_YET = "memory-accounting/onsite-memory-verdicts"
RUNNER = "benchmarks/one_stage/served_verdicts.sh"
VERDICTS_SCRIPT = "benchmarks/one_stage/served_verdicts.py"
PEARLPLUS_GATE = "benchmarks/one_stage/pearlplus_local_e2e.py"
ONSITE = "verity/onsite/v1"
REMOTE = "verity/remote/v1"
#: the deploy bindings `begin` keeps beside the run (`verity_cli.run.BINDINGS`)
BINDINGS = "bindings.json"
#: the parties `verity run` starts first where ``--bindings`` gives none of what they write, each registered once by
#: its own `STARTS[PARTY] = started(PARTY, ...)`, never by assigning the table again
STARTS: dict[str, Start] = {}
WORKERS, ZK_SERVER, PROVER, VERIFIER = SP.WORKERS, SP.ZK_SERVER, WP.PROVER, SP.VERIFIER
CHALLENGER, MEMORY_ATTESTER = SP.CHALLENGER, WP.CHALLENGER
BANDWIDTH_VERIFIER, NETWORK_ATTESTER = P2.CLOSER_PARTY, NC.NETWORK_ATTESTER
#: served_verdicts.py's ``answer_from`` for a row it carries from a file the served node wrote (its ``AS_WRITTEN``)
AS_WRITTEN = "as the served node wrote it"
#: remote's line read from a record the prover composed, not a compose gate's answer: the serving node's own
#: (``out``) or a deferral's follow-on's on a prover node (its ``OUT`` copied to ``out/follow-on``), each its ``source``
#: and the sentence that leads its reason; each accept in it, a row's or the composed verdict, reads `DEVELOPMENT_VERDICT`,
#: so nothing that counts accepts counts the prover's word that it passed
SERVING_NODE, FOLLOW_ON = "serving-node", "follow-on"
FROM_SERVING_NODE = ("read from the serving node's own record: a development verdict; the auditor's device did not "
                     "compose it")
FROM_FOLLOW_ON = ("read from a deferral's follow-on record, composed on a prover node: a development verdict; the "
                  "auditor's device did not compose it")
DEVELOPMENT_VERDICT = "development"


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


def _settings(scheme: Scheme, given: Mapping[str, Any]) -> dict[str, Any]:
    """``scheme``'s `settings` over ``given``, refused as ``unknown-key`` where it has no such setting and as ``params``
    where it refuses a value, with `settings`' own text: every scheme's settings are checked here."""
    try:
        return settings(scheme, given)
    except ValueError as e:
        raise Refused("unknown-key" if set(given) - set(scheme.settings) else "params", str(e)) from None


_CLASS = re.compile(r"[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*")


def _tied(scheme: Scheme, cid: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """A tied scheme's parameters, under its id ``cid``: what its ``tie`` names and nothing else (its other settings are
    values a launch gives `attach`), its settings among them checked by `_settings` over their defaults, and a ``class``
    the served window's, SLOTS,PROMPT,OUTPUTS."""
    if extra := sorted(set(params) - set(scheme.tie)):
        raise Refused("unknown-key", f"[params.\"{cid}\"]: {', '.join(extra)} (known: {', '.join(sorted(scheme.tie))})")
    if "class" in params and not (isinstance(params["class"], str) and _CLASS.fullmatch(params["class"])):
        raise Refused("params", f"class is SLOTS,PROMPT,OUTPUTS, three positive integers, not {params['class']!r}")
    own = _settings(scheme, {k: v for k, v in params.items() if k in scheme.settings})
    return {**{k: v for k, v in params.items() if k not in scheme.settings},
            **{k: own[k] for k in scheme.tie if k in scheme.settings}}


_COMMIT = re.compile(r"[0-9a-f]{40}")
#: the served window's loads (served_verdicts.sh's LOAD) an application names by ``load``; ``front``, network-accounting's:
#: the network attester's front's group is the run's one load and its audited window
LOADS = ("varied", "front")
#: an application's load: ``load`` (one of `LOADS`), or an input ``dataset`` the adapter plays at its ``timing``
LOAD_KEYS = (("load",), ("dataset", "timing"))


def _door(params: Mapping[str, Any]) -> None:
    """The adapter's statement and its door (`verity_vllm.inputs.serves`) over the application's ``dataset`` at its ``timing``,
    read through without the engine or a tokenizer; refused by the door's name."""
    import input_datasets
    from verity_vllm.inputs import serves
    try:
        serves.served(params["model"], params["revision"], params["precision"], params["slots"], params["window-steps"])
        rule, parameters = serves.timing(params["timing"])
        ds = input_datasets.open(params["dataset"])
        serves.door(ds.kind, (b for _, b in ds.timed(rule, **parameters)))
    except (serves.Refused, input_datasets.Refused) as e:
        name, text = serves.refusal(e)
        raise Refused(name, f"[workload.params] dataset {params['dataset']!r}: {text}") from e


def _served(params: Mapping[str, Any]) -> dict[str, Any]:
    """The served window's public shape: the model at its revision and precision, the window's request ``slots``, its
    engine steps (``window-steps``) and its load: a ``load`` (served_verdicts.sh's LOAD), or an input ``dataset`` and its
    ``timing``, which the adapter's door reads through here (`_door`). Both, or neither, is refused (``load``)."""
    base = {"model", "revision", "precision", "slots", "window-steps"}
    known = base.union(*LOAD_KEYS)
    if extra := sorted(set(params) - known):
        raise Refused("unknown-key", f"[workload.params]: {', '.join(extra)} (known: {', '.join(sorted(known))})")
    said = [keys for keys in LOAD_KEYS if any(k in params for k in keys)]
    if len(said) != 1 or not all(k in params for k in said[0]):
        raise Refused("load", "[workload.params] names its load by load (one of " + ", ".join(LOADS) + "), or by "
                              "dataset and timing (an input dataset and at-once or rate:R), one of the two whole, not "
                              f"{', '.join(sorted(k for keys in LOAD_KEYS for k in keys if k in params)) or 'neither'}")
    for k in ("model", "precision"):
        if not isinstance(params[k], str) or not params[k]:
            raise Refused("workload", f"{k} is a non-empty string, not {params[k]!r}")
    if not isinstance(params["revision"], str) or not _COMMIT.fullmatch(params["revision"]):
        raise Refused("workload", f"revision is the model's 40-hex revision, not {params['revision']!r}")
    for k in ("slots", "window-steps"):
        if type(params[k]) is not int or params[k] < 1:
            raise Refused("workload", f"{k} is a positive integer, not {params[k]!r}")
    if "load" in params and params["load"] not in LOADS:
        raise Refused("load", f"load is one of {', '.join(LOADS)}, not {params['load']!r}: an input dataset is named by "
                              "dataset and timing")
    if "dataset" in params:
        _door(params)
    return {k: params[k] for k in sorted(base.union(*said))}


_VERIFIER = Role(VERIFIER, "auditor", needs=("audit", "component", "product"))
_PROVER = Role(PROVER, "developer", per="server", needs=("store",))


def _declared(name: str) -> Component:
    """The `Component` of the scheme ``name`` declares (`schemes.get`): its parties, channels, colocated pairs, needs,
    and as ``requires`` its ``depends``. Its parameters are its settings (`_settings`, over their defaults), or, for a
    tied scheme, what its ``tie`` names (`_tied`). Its id is the one `SP.IDS` gives, the key of its ``[params]``, which
    may differ from the declared ``protocol``."""
    s = schemes.get(name)
    cid = SP.IDS.get(name, s.protocol)
    params = ({"params": partial(_tied, s, cid)} if s.tie else
              {"params": partial(_settings, s), "default_params": dict(s.settings)} if s.settings else {})
    return Component(name, cid, parties=declared(s, "parties"), channels=declared(s, "channels"),
                     colocated=declared(s, "colocated"), needs=declared(s, "needs"), requires=declared(s, "depends"),
                     **params)


#: the components no scheme declares yet, each with its row in the record, which only shrinks
#: (`tests/tools/verity/cli/test_undeclared_rows.py`)
UNDECLARED: dict[str, tuple[Component, str, Mapping[str, Launch]]] = {c.name: (c, row, pieces) for c, row, pieces in (
    (Component("pouw", SP.IDS["pouw"], parties=(Role(CHALLENGER, "auditor", needs=("host",)),
                                                Role(WORKERS, "developer", needs=("serve",)), _VERIFIER),
               channels=(Channel(CHALLENGER, WORKERS, ("H1Coins",), carries_coins=True),
                         Channel(WORKERS, VERIFIER, ("H1Record",), after=("H1Coins",))),
               needs=frozenset({"windows", "noise"}), requires=("sampled-proofs",)), "work",
     {WORKERS: Launch(binds=("gate", "peer", "key-sha512"), optional=("main-file",), entry="served_table:pouw_entry",
                      executables=(Executable("verity/protocols/accounting/compute/pouw/verifier/lean/check",
                                              "pouw-verifier-difftest"),))}),
)}
#: every component the served window's record has a row for: each undeclared one's, and each declared scheme's whose
#: ``record`` names its row (a scheme with no row, an experimental one or a session's own, is no component here)
MANIFESTS: dict[str, Component] = {
    **{n: c for n, (c, _, _) in UNDECLARED.items()},
    **{n: _declared(n) for n in schemes.DECLARATIONS if not isinstance(schemes.get(n).record, Lacks)},
    **{c.name: c for c in (
        replace(_declared("sampled-proofs"), sampling_design=_design, default_sampling_design="rate:1/3481648"),)}}
#: each component's row in the record, the one its scheme declares (`record`)
ROWS: dict[str, str] = {n: row for n, (_, row, _) in UNDECLARED.items()} | {
    n: declared(schemes.get(n), "record") for n in MANIFESTS if n not in UNDECLARED}
SERVED = Workload(SERVED_WINDOW, frozenset({"windows", "noise", "store", "receipts", "ingress-relay"}),
                  (Role(WORKERS, "developer", needs=("serve",)),), _served,
                  {"model": "verity-int8/Llama-3.2-1B-INT8", "revision": "15b0976d400d796ee589e7737d03a919526bf800",
                   "precision": "verity_int8", "slots": 4, "window-steps": 16})
PROVER_NODE, AUDITOR_HOST, GATE_HOST = "prover-node", "auditor-host", "gate-host"
#: remote's auditor parties' machine, which isn't at the site: remote puts no auditor's device there
AUDITOR_REMOTE = "auditor-remote"


# --- the pieces ------------------------------------------------------------------------------------------------------

Hooks = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, str], list[str]], dict[str, str]]


def _none(cfg: Mapping[str, Any], params: Mapping[str, Any], bound: Mapping[str, str], pinned: list[str]) -> dict[str, str]:
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
    configuration party that listens (``listener``, the auditor's), whose run is beside the served window. ``starts``
    are a role's starts the table makes, over its halves': a start handed what its protocol's code may not load."""

    halves: tuple[str, ...]
    roles: Mapping[str, str]
    clocks: Mapping[str, tuple[Callable[[], int], str]]
    listener: str
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
    """One party's piece of the served window (module docstring): ``hooks(cfg, params, bound, pinned)`` gives the
    runner's variables for it from the configuration, the component's checked parameters, the bindings of the party it
    reaches (``bound``: each of ``binds``, a ``<party.component.key>`` placeholder while unbound, and each bound one of
    ``optional``) and that party's key pinned for this component (``pinned``: the configuration's one pin, or none,
    and none while the piece's ``development`` binding is unbound). A ``runtime`` piece's ``session`` is the protocol it is a party of. A piece with a ``development``
    binding left unbound starts the party it reaches on its own machine, under a key and coins made for the run, in a
    development run (`coins_on`), and takes none of its other bindings; one that ``follows`` a listed component's piece
    of the same party is unbound exactly when that piece is, since its attester signs under that one's key."""

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
    development: str = ""
    follows: str = ""
    starts: tuple[str, ...] = ()
    starts_optional: tuple[str, ...] = ()
    #: the scheme's own Lean executables the piece builds and answers with (`Launch.executables`)
    executables: tuple[Executable, ...] = ()

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
        if self.development and self.development not in mine and (given := [k for k in self.optional if k in mine]):
            raise Refused("unbound", f"{self.key(self.development)} is unbound, so {self.party}'s {self.component} "
                                     f"piece starts its {self.reaches} on its own machine under a key made for the "
                                     f"run, and the deploy bindings bind {', '.join(map(self.key, given))} beside it: "
                                     f"bind {self.key(self.development)} too, or none of them")
        if self.follows in cfg["protocols"] and (self.development in mine) != (
                (lead := PIECES[self.follows][self.party]).development in lead._bindings(bindings)):
            given, alone = (self, lead) if self.development in mine else (lead, self)
            raise Refused("unbound", f"{self.component}'s attester signs under {self.follows}'s key, and the deploy "
                                     f"bindings bind {given.key(given.development)} but not "
                                     f"{alone.key(alone.development)}: bind both, or neither for a development run "
                                     "whose attesters run on the workload's machine under a key made for the run")
        bound = {**{k: f"<{self.key(k)}>" for k in self.binds},
                 **{k: str(v) for k, v in mine.items() if k in self.binds or k in self.optional}}
        params = cfg["params"].get(MANIFESTS[self.component].id, {}) if self.component in MANIFESTS else {}
        dev = not self.development or self.development in mine
        try:
            return self.hooks(cfg, params, bound,
                              _pinned(cfg, self.reaches or self.party, self.component) if dev else [])
        except LaunchRefused as e:
            raise Refused(e.name, e.detail) from e

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


def row_id(model: str, precision: str, slots: int) -> str:
    """The served window's row name, a rule over what the window declares and nothing it was built or traced from: the
    model's and the precision's names lowercased with each run of other characters one hyphen, then the window's request
    slots, ``<model>__<precision>__s<slots>``. The same three give the same name and a change in any one changes it."""
    names = [re.sub(r"[^a-z0-9]+", "-", x.lower()).strip("-") for x in (model, precision)]
    if not all(names) or type(slots) is not int or slots < 1:
        raise Refused("workload", f"a row is named by a model and a precision with a letter or digit and a positive number "
                                  f"of slots, not {model!r}, {precision!r}, {slots!r}")
    return f"{names[0]}__{names[1]}__s{slots}"


def _serve(cfg, params, bound, pinned):
    """The served window's runner's variables; for an input dataset, ``LOAD=dataset`` with ``DATASET`` and ``TIMING``,
    and the class it commits (``V3C_CLASS``, ``CLASSES``) the adapter's statement's (`verity_vllm.inputs.serves.SERVES`)."""
    w = cfg["workload"]["params"]
    out = {"REPO": w["model"], "REV": w["revision"], "PRECISION": w["precision"], "SLOTS": str(w["slots"]),
           "ROW_ID": row_id(w["model"], w["precision"], w["slots"]),
           "WINDOW_STEPS": str(w["window-steps"]), "V3C_LAW": cfg["service"]["law"]}
    if "load" in w:
        return {**out, "LOAD": w["load"]}
    from verity_vllm.inputs import serves
    s = serves.served(w["model"], w["revision"], w["precision"], w["slots"], w["window-steps"])
    return {**out, "LOAD": "dataset", "DATASET": w["dataset"], "TIMING": w["timing"], "V3C_CLASS": s.window_class,
            "CLASSES": s.window_class}


def _pouw(cfg, params, bound, pinned):
    """H1 on the serving engine (--protocols pouw:...), its coins the gate's; served_verdicts.py pouw checks them
    against the gate's session under its key."""
    return {"H1_GATE": bound["gate"], "H1_PEER": bound["peer"], "H1_KEY_SHA512": bound["key-sha512"],
            **({"MAIN_FILE": bound["main-file"]} if "main-file" in bound else {})}


def _pinned(cfg: Mapping[str, Any], party: str, protocol: str) -> list[str]:
    return [p["key-sha512"] for p in cfg.get("pin", []) if (p["party"], p["protocol"]) == (party, protocol)][:1]


#: the challenger's hosts that `verity run` started on this machine, by the run's state, from its serve to its stop
_HOSTS: dict[Path, CP.Host] = {}


def _first_parent(state: Path) -> list[str]:
    program = json.loads((state / "run.json").read_text())["program"]
    p = subprocess.run(["git", "rev-list", "--first-parent", str(program)], cwd=ROOT, capture_output=True, text=True,
                       check=False)
    if p.returncode or not p.stdout.split():
        raise Refused("challenger", f"`git rev-list --first-parent {program}` exited {p.returncode} in {ROOT}: the "
                                    "challenger's hosts pin on the run's commit, held to its first-parent line")
    return p.stdout.split()


def _deployed_local(root: Path, main: list[str]) -> LC.Local:
    """SPOT's loopback host's session, its key at `lms.DEPLOYED`, the only parameters SPOT's v2 reads a sample's coins
    under (`OneStageZk.KEY_PARAMS`), made over this machine's cores."""
    return LC.Local(root, main, params=(lms.DEPLOYED.W, lms.DEPLOYED.H), workers=len(os.sched_getaffinity(0)))


def _serve_challenger(cfg: Mapping[str, Any], state: Path, machines: Sequence[str] = ()) -> None:
    """The challenger where `verity run` starts it, on this machine: SPOT's loopback host (`challenger_party.Host`)
    and Pearl+'s gate (`PEARLPLUS_GATE`), each under a key made for the run in ``STATE/parties/challenger/``, flipping
    its own coins, pinned on the run's commit, whose first-parent line is ``main.txt``; its bindings written whole, by
    a rename. Refused by name (``challenger``) where Pearl+'s gate isn't made or isn't pinned on that commit."""
    mine, public = state / "parties" / CHALLENGER, state / "public" / CHALLENGER
    main = _first_parent(state)
    public.mkdir(parents=True, exist_ok=True)
    (line := public / "main.txt").write_text("\n".join(main) + "\n")
    print("verity run: building the Lean verifier and Pearl+'s gate: this takes minutes, and nothing prints until it is done",
          file=sys.stderr, flush=True)
    p = subprocess.run([sys.executable, str(ROOT / PEARLPLUS_GATE), "init", "--root", str(mine / "pearl-plus")],
                       cwd=ROOT, capture_output=True, text=True, check=False)
    if p.returncode:
        raise Refused("challenger", f"{PEARLPLUS_GATE} init exited {p.returncode}: {p.stderr.strip()[-800:]}")
    gate = json.loads(p.stdout.strip().splitlines()[-1])
    if gate["pin"] != main[0]:
        raise Refused("challenger", f"Pearl+'s gate pins {gate['pin']}, not the run's commit {main[0]}")
    print("verity run: making the loopback host's key at the deployed height: this takes minutes", file=sys.stderr, flush=True)
    host = _HOSTS[state] = CP.Host(mine / "sampled-proofs", main, _deployed_local)
    spot = host.start()
    made = {"loopback": spot["address"], "main-file": str(line), "gate": gate["gate"], "peer": gate["peer"],
            "lane": gate["lane"], "key-sha512": gate["key_sha512"]}
    part = public / f".{BINDINGS}"
    part.write_text(json.dumps(_written(CHALLENGER, made), sort_keys=True))
    part.replace(public / BINDINGS)


def _stop_challenger(cfg: Mapping[str, Any], state: Path) -> None:
    """Both hosts stopped, and Pearl+'s gate's public files (its ledger, pins log, public key and sessions' public
    files) copied under ``STATE/public/challenger/pearl-plus/``, which `verity run` transcribes. Refused by name
    (``challenger``) where the gate's stop fails, so the party's exit is not 0."""
    if (host := _HOSTS.pop(state, None)) is not None:
        host.stop()
    if (root := state / "parties" / CHALLENGER / "pearl-plus").is_dir():
        p = subprocess.run([sys.executable, str(ROOT / PEARLPLUS_GATE), "stop", "--root", str(root), "--out",
                            str(state / "public" / CHALLENGER / "pearl-plus")], cwd=ROOT, capture_output=True,
                           text=True, check=False)
        if p.returncode:
            raise Refused("challenger", f"{PEARLPLUS_GATE} stop exited {p.returncode}, so Pearl+'s gate's public "
                                        f"files are not all in the run's record: {p.stderr.strip()[-800:]}")


#: PoUS's timed rounds (`challenger_session`'s parties), beside the served window: the challenger's header names the
#: monotonic clock
POUS = Session(("verity.protocols.accounting.memory.pous.attester.challenger_party:PROTOCOL",
                "verity.protocols.accounting.memory.pous.prover.challenger_prover:PROTOCOL"),
               {MEMORY_ATTESTER: CS.CHALLENGER, PROVER: CS.PROVER}, {CS.CHALLENGER: (time.monotonic_ns, MONOTONIC)},
               MEMORY_ATTESTER)
#: onsite PoUB over PoUS's P2 store (`p2_parties`' verifier, `p2_party_prover`'s prover), beside the served window, on
#: the verifier's monotonic clock; the verifier reads the memory attester's PoUS key from that party's run beside it
#: on the auditor's device, as the bandwidth audit declares (`Scheme.reads`)
POUB = Session(("verity.protocols.accounting.memory.poub.verifier.p2_parties:PROTOCOLS",
                "verity.protocols.accounting.memory.poub.prover.p2_party_prover:PROTOCOL"),
               {BANDWIDTH_VERIFIER: P2.VERIFIER, PROVER: P2.PROVER}, {P2.VERIFIER: (time.monotonic_ns, MONOTONIC)},
               BANDWIDTH_VERIFIER, starts=P2.starts())


def _local_gate(cfg: Mapping[str, Any], state: Path, machines: Sequence[str] = ()) -> None:
    """The network attester's bindings where `verity run` starts it: the front's local gate, this tree's gate on the
    workload's machine under a key made for the run, its peer that machine as the configuration names it."""
    machine = {p["name"]: p["machine"] for p in cfg["party"]}[WORKERS]
    f = state / "public" / NETWORK_ATTESTER / BINDINGS
    f.parent.mkdir(parents=True, exist_ok=True)
    part = f.with_name(f".{BINDINGS}")
    part.write_text(json.dumps(_written(NETWORK_ATTESTER, {"gate": "local", "peer": machine}), sort_keys=True))
    part.replace(f)


SERVE = Piece(SERVED_WINDOW, WORKERS, SERVED_WINDOW, "Execution's record: the window's steps on what the adapter's serve "
              "left on the prover node (its seal, the window and its advice, the window circuit's count, each listed "
              "component's piece, the inference-only policy's steps); the sampled proofs are not run, so no proof is "
              "made and no Lean verifier runs on one",
              scripts=(RUNNER, VERDICTS_SCRIPT), hooks=_serve)
class Launcher(NamedTuple):
    """What a declared piece's `Launch.entry` gives, called with its scheme's declaration: ``hooks(cfg, params, bound,
    pinned)``, the runner's variables for it, whose docstring says what the piece does, and the scripts it runs."""

    hooks: Hooks
    scripts: tuple[str, ...] = ()


def pouw_entry(declaration: Any) -> Launcher:
    return Launcher(_pouw)


def _refusing(component: str, party: str, what: str) -> Launcher:
    def lacking(cfg, params, bound, pinned):
        raise Refused(f"no-{what}", f"{component} declares a piece for {party} whose launch declares no {what} "
                                    f"(`Launch.{what}`): its scheme's declaration states it, and nothing else does")
    lacking.__doc__ = f"{component}'s {party}: its launch declares no {what} (`Launch.{what}`)"
    return Launcher(lacking)


def _launcher(component: str, party: str, said: Launch, declaration: Any) -> Launcher:
    """What ``said``'s `Launch.entry` gives, called with ``declaration``; read from the declaration alone, so no
    variable, flag or file of a node's names another. Where the launch declares no entry or no executables, a launcher
    whose hooks refuse by name (``no-entry``, ``no-executables``) when the piece is launched, so nothing launches in its
    place."""
    for lacking in ("entry", "executables"):
        if isinstance(getattr(said, lacking), Lacks):
            return _refusing(component, party, lacking)
    module, attribute = said.entry.split(":")
    # this table, by whatever name it was loaded under (`run.REGISTRY`'s, a test's, `verity_cli.served_table`)
    found = globals() if module == __name__.rpartition(".")[2] else vars(importlib.import_module(module))
    made = found[attribute](declaration)
    if not (isinstance(made, tuple) and len(made) == 2 and callable(made[0])
            and isinstance(made[1], tuple) and all(isinstance(s, str) for s in made[1])):
        raise ValueError(f"{component}'s piece for {party}: {said.entry} gives {made!r}, not its hooks and scripts")
    return Launcher(*made)


def launches(name: str) -> Mapping[str, Launch]:
    """The pieces the component ``name`` declares, by developer-side party (`Scheme.pieces`; `UNDECLARED`'s own)."""
    if name in UNDECLARED:
        return UNDECLARED[name][2]
    said = schemes.get(name).pieces
    return {} if isinstance(said, Lacks) else said


def _reaches(c: Component, party: str, said: str) -> str:
    """The auditor party other than the verifier that ``party`` shares a declared channel with; ``said`` only where
    there is none."""
    auditor = {r.name for r in c.parties if r.principal == "auditor"} - {VERIFIER}
    joined = {o for ch in c.channels for a, o in ((ch.sender, ch.receiver), (ch.receiver, ch.sender))
              if a == party and o in auditor}
    if said and joined:
        raise ValueError(f"{c.name}: {party} reaches {sorted(joined)} by its declared channels, so its piece says no "
                         f"reaches (it says {said!r})")
    if len(joined) > 1:
        raise ValueError(f"{c.name}: {party} shares declared channels with {sorted(joined)}: its piece says which it "
                         "reaches")
    return said or next(iter(joined), "")


def _pieces(c: Component) -> dict[str, Piece]:
    """Each of ``c``'s declared parties' piece: the verifier's at the check, another auditor party's at the gate, a
    developer party's in the served window; a declared piece (`launches`) with its launcher, the party it reaches, and the
    declared ``depends`` it ``follows``, the one whose piece of the same party also has a development binding."""
    mine, out = launches(c.name), {}
    for r in c.parties:
        runs = CHECK if r.name == VERIFIER else GATE if r.principal == "auditor" else SERVED_WINDOW
        if r.name not in mine:
            out[r.name] = Piece(c.name, r.name, runs, f"{c.name}'s row" if runs == CHECK else
                                f"the gate: {c.name}'s {r.name}" if runs == GATE else f"{c.name}'s {r.name}")
            continue
        said = mine[r.name]
        entry = _launcher(c.name, r.name, said, UNDECLARED[c.name][0] if c.name in UNDECLARED else schemes.get(c.name))
        follows = next((d for d in c.requires if said.development and launches(d).get(r.name, Launch()).development),
                       "")
        out[r.name] = Piece(c.name, r.name, runs, " ".join((entry.hooks.__doc__ or "").split()), scripts=entry.scripts,
                            reaches=_reaches(c, r.name, said.reaches), binds=said.binds, optional=said.optional,
                            hooks=entry.hooks, development=said.development, follows=follows, starts=said.starts,
                            starts_optional=said.starts_optional,
                            executables=() if isinstance(said.executables, Lacks) else said.executables)
    return out


#: each listed component's pieces by party, from its declaration (`_pieces`)
PIECES: dict[str, dict[str, Piece]] = {n: _pieces(c) for n, c in MANIFESTS.items()}


def _starting(party: str) -> dict[str, Piece]:
    """The pieces, by component, whose declared ``starts`` (`Launch.starts`) ``party``'s start writes."""
    return {c: x for c, ps in PIECES.items() for x in ps.values() if x.reaches == party and x.starts}


def started(party: str, **kw: Any) -> Start:
    """``party``'s `Start`: what it writes and may leave out, as each piece that reaches it declares (`_starting`)."""
    mine = _starting(party)
    return Start(writes={c: x.starts for c, x in mine.items()},
                 optional=frozenset((c, k) for c, x in mine.items() for k in x.starts_optional), **kw)


def _written(party: str, made: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """What ``party``'s start writes for each component it starts for: of the values it ``made``, each one that
    component's piece declares it writes (`STARTS`)."""
    return {c: {k: made[k] for k in ks if k in made} for c, ks in STARTS[party].writes.items()}


#: the challenger, whose coins SPOT's sample and Pearl+'s salt and sample read; its Lean coins' verifier is built when
#: missing and SPOT's key is made at the deployed parameters, so it waits long. The auditor's own gate (`run.sh
#: bindings`) gives only Pearl+'s gate, peer and key, and SPOT its gate in the loopback's place
STARTS[CHALLENGER] = started(CHALLENGER, serve=_serve_challenger, stop=_stop_challenger, within=1800)
STARTS[NETWORK_ATTESTER] = started(NETWORK_ATTESTER, serve=_local_gate, stop=lambda cfg, state: None, within=60)
WORKLOADS: dict[str, tuple[Workload, Piece]] = {SERVED_WINDOW: (SERVED, SERVE)}
#: no party here runs as an entry of its own: every piece is the served window's, the gate's, the runtime's or the check's
ORDER: tuple[str, ...] = ()


# --- the run ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Result:
    """What `verity_cli.run.execute` reads of a live run: each party's outcome and refusal, and the period's line, which
    it prints as the run's last (``PERIOD``)."""

    outcomes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    period: Mapping[str, Any] | None = None


def bindings_of(state: Path) -> dict[str, Any]:
    f = state / BINDINGS
    return json.loads(f.read_text()) if f.exists() else {}


def launched(name: str) -> Mapping[str, Piece]:
    """The pieces of the component ``name`` by party (`PIECES`), refused (``unknown-protocol``) by name where no scheme
    declares it and `UNDECLARED` doesn't hold it, or where this table has no pieces for it."""
    if name not in schemes.DECLARATIONS and name not in UNDECLARED:
        raise Refused("unknown-protocol", f"{name} is no declared scheme (verity.protocols.run.schemes) and no "
                                          "row of served_table.UNDECLARED: declare it, or list a declared one")
    if name not in PIECES:
        raise Refused("unknown-protocol", f"{name} is declared, but served_table.PIECES has no pieces for it: "
                                          "list it in MANIFESTS, or list another")
    return PIECES[name]


def pieces(cfg: Mapping[str, Any], party: str) -> list[Piece]:
    """``party``'s pieces under ``cfg``: the workload's, then each listed component's."""
    out = [SERVE] if party == WORKERS else []
    return out + [launched(c)[party] for c in cfg["protocols"] if party in launched(c)]


#: the runner's variables that are flags, serve-openai's and served_piece.py's, which each piece adds its own to
JOINED = ("SERVE_ARGS", "POUS_ARGS")
#: the runner's variables that name the class V3c commits, which two pieces may set only alike
CLASS_KEYS = ("V3C_CLASS", "CLASSES")


def pins_and_configuration(cfg: Mapping[str, Any]) -> dict[str, str]:
    """The configuration's hash and its `[[pin]]` entries as the runner hands them to every piece
    (``VERITY_CONFIGURATION``, ``VERITY_PINS``); a launcher outside the road reads them from here."""
    return {"VERITY_CONFIGURATION": K.digest(cfg), "VERITY_PINS": json.dumps(cfg.get("pin", []), sort_keys=True)}


def variables(cfg: Mapping[str, Any], bindings: Mapping[str, Any]) -> dict[str, str]:
    """The served window's runner's variables: the configuration's hash and its `[[pin]]` entries
    (``VERITY_CONFIGURATION``, ``VERITY_PINS``), which every device's `trusted-device` premise reads, and every listed
    piece's hooks, whichever party's they are, since the prover's pieces reach the auditor's sides through their
    bindings. A variable of `JOINED` is every piece's flags,
    in the pieces' order; any other is one piece's."""
    out = pins_and_configuration(cfg)
    by: dict[str, str] = {}
    for p in [SERVE, *(x for c in cfg["protocols"] for x in launched(c).values())]:
        for k, v in p.variables(cfg, bindings).items():
            if k in CLASS_KEYS and k in by and out[k] != v:
                raise Refused("params", f"{k} is {out[k]} in {by[k]}'s piece and {v} in {p.component}'s: the runner "
                                        f"commits one class, so give one of them the class or list one of them alone")
            by.setdefault(k, p.component)
            out[k] = f"{out[k]} {v}" if k in JOINED and k in out else v
    return out


def unbound(cfg: Mapping[str, Any], bindings: Mapping[str, Any]) -> list[str]:
    """Every listed piece's unbound bindings, each once: a runtime session's two sides read one ``address``."""
    return list(dict.fromkeys(k for p in [SERVE, *(x for c in cfg["protocols"] for x in launched(c).values())]
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
    its state directory ``state/runs/<component>/<party>``, beside the runs its scheme declares it reads,
    `Scheme.reads`), the two sides' request bound (`request_bound`), what this
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
             **{k: str(state / "runs" / c / p)
                for k, (c, p) in declared(schemes.get(piece.component), "reads").get(piece.party, {}).items()}}
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
    configuration's hash (`memory_attester.trusted_device`, `committed` its committed set), which never holds here.
    ``None`` for a party that signs no session; a session that reports no key reads rehearsal."""
    if piece.party != MEMORY_ATTESTER:
        return None
    key = (outcome or {}).get("public_key")
    machine = {p["name"]: p["machine"] for p in cfg["party"]}[piece.party]
    return {mc.PREMISE: mc.trusted_device(cfg.get("pin", []), piece.party, piece.component,
                                          None if key is None else hashlib.sha512(bytes.fromhex(key)).hexdigest(),
                                          f"the {piece.party} ran on {machine}", configuration=K.digest(cfg),
                                          committed=mc.committed())}


def attested(cfg: Mapping[str, Any], state: Path) -> dict[str, str]:
    """`trusted-device`'s premise for each listed component whose piece's answer places it: an
    ``out/pieces/*/answer.json`` whose ``protocol`` is the component's id, by its profile's ``placement``; a piece that
    answered without one places nothing."""
    ids = {MANIFESTS[c].id: c for c in cfg["protocols"] if c in MANIFESTS}
    out = {}
    for f in sorted((state / "out" / "pieces").glob("*/answer.json")):
        doc = json.loads(f.read_text())
        if (c := ids.get(doc.get("protocol"))) and (where := (doc.get("profile") or {}).get("placement")):
            out[c] = mc.premise(where)
    return out


def composite(env: Mapping[str, str]) -> dict[str, str]:
    """The runner's ``COMPOSITE``, set empty where it composes off the served node (``COMPOSE_OFF_NODE=1``, set by a
    caller that runs a composer there, or its default under ``VERITY_HOLD_PLACEMENT=1``), where onsite's composite, which
    reads PoUS's and the wipe's cases that only the node's card writes, can't be made, and it refuses to start. ``env``
    is what the runner gets before it; a caller that names ``COMPOSITE`` or ``COMPOSE_OFF_NODE=0`` keeps what it named."""
    off = env.get("COMPOSE_OFF_NODE", "1" if machine_settings.hold_placement(env) else "0")
    return {} if "COMPOSITE" in env or off == "0" else {"COMPOSITE": ""}


def served_window(cfg: Mapping[str, Any], state: Path, bindings: Mapping[str, Any], steps: Sequence[str],
                  log: str) -> int:
    """The served window's runner on ``steps`` in their order, its lines in ``state/logs/<log>.log``, with every listed piece's hooks, the application's list (``APPLICATION_PROTOCOLS``)
    and no composite unless the caller asks for one (`composite`), its public record in ``state/out``; its serve dir stays where the runner keeps it on the node (its
    ``D`` and ``SERVE_ROOT``, from the environment). The bindings' table for the workload (``[workers.served-window]``)
    is this machine's settings for the runner, under the pieces' hooks: a setting the configuration's pieces set is
    theirs."""
    node = {k: str(v) for k, v in bindings.get(WORKERS, {}).get(SERVED_WINDOW, {}).items()}
    env = lean_env({**os.environ, **node, **variables(cfg, bindings)})
    env |= {**composite(env), "OUT": str(state / "out"), "APPLICATION_PROTOCOLS": " ".join(cfg["protocols"])}
    (state / "logs").mkdir(parents=True, exist_ok=True)
    with open(state / "logs" / f"{log}.log", "w") as f:
        return subprocess.run(["bash", str(ROOT / RUNNER), "steps", *steps], cwd=ROOT, env=env,
                              stdout=f, stderr=subprocess.STDOUT, check=False).returncode


VERDICTS = {"accept": "accept", "reject": "reject", "pending": "undecided"}


def charges_nothing(component: str) -> str | None:
    """Why a component whose declared scheme names no resource has no figure: its `Lacks`'s why ("charges nothing:
    ..."), which a check's line reads where a figure would be (its ``charges_nothing``), never in place of the
    component's verdict; None for any other component, declared or not."""
    if component not in schemes.DECLARATIONS:
        return None
    resource = schemes.get(component).resource
    return (resource.why or "charges nothing") if isinstance(resource, schemes.S.Lacks) else None


def unchecked(components: Sequence[str], said: Mapping[str, str]) -> dict[str, str]:
    """``said``, each component's stated source, with each sentence of what its declared scheme's run has that no
    auditor-side entry checks (`scheme.sentences`) at its end, for each of ``components`` whose declaration lists any,
    whatever its row; a sentence the stated source says already (the launcher's record ends it so,
    `served_verdicts.declared_unchecked`) is not said twice."""
    return {**said, **{c: "; ".join([*filter(None, [said.get(c)]), *(w for w in why if w not in said.get(c, ""))])
                       for c in components
                       if c in schemes.DECLARATIONS and (why := schemes.S.sentences(schemes.get(c)))}}


#: what the served window publishes of the execution phase its record is of, under ``out/serve``: its seal's SHA-512
#: (``seal_sha512``), never its number or steps, which `served_verdicts.py publish` keeps on the node
PHASE = "phase.json"


def onsite_check(cfg: Mapping[str, Any], state: Path) -> dict[str, Any]:
    """The verifier's onsite line over what the application's serving step published in ``state``'s ``out/`` and
    nothing else: the record (``record/verdicts.json``) of the execution phase ``serve/phase.json`` names, the line's
    ``block`` its seal's SHA-512, in `verity_cli.run`'s verdict form: its verdicts as the record has them, each row's
    ``answer_from`` (printed, deciding nothing) saying who wrote it. Under ``VERITY_HOLD_PLACEMENT=1``
    (`verity_cli.settings.hold_placement`), since the workload's node composed that record, each of its accepts, each
    row's and the combined verdict, reads undecided, `AS_WRITTEN` in the reason; a reject or an error stands either way.
    A component that charges nothing keeps its row's verdict like any other; the line's ``charges_nothing`` says why it
    has no figure. Refused
    (``window``, ``record``) when either file isn't there."""
    out = state / "out"
    phase, report = out / "serve" / PHASE, out / "record" / "verdicts.json"
    if not phase.is_file():
        raise Refused("window", f"the served window published no execution phase ({phase})")
    if not report.is_file():
        raise Refused("record", f"the served window published no record ({report})")
    seal = json.loads(phase.read_text())["seal_sha512"]
    doc = json.loads(report.read_text())
    rows = doc["protocols"]
    free = {c: why for c in cfg["protocols"] if (why := charges_nothing(c))}
    components = {c: rows.get(ROWS[c], {}).get("verdict", "absent") for c in cfg["protocols"]}
    answer_from = unchecked(cfg["protocols"], {c: rows[ROWS[c]].get("answer_from", AS_WRITTEN)
                                               for c in cfg["protocols"] if ROWS[c] in rows})
    absent = sorted(c for c, v in components.items() if v == "absent")
    missing = sorted(c for c, v in components.items() if v == "missing")
    composed = doc.get("composed") or {}
    verdict, product = VERDICTS.get(composed.get("verdict"), "undecided"), K.composite(cfg["protocols"], MANIFESTS)
    wrong = "; ".join(f"{c} {v}" for c, v in sorted(components.items()) if v in ("reject", "error"))
    reason = (wrong or composed.get("reason")) if verdict == "reject" else \
        (f"missing {missing}" if missing else None) or (f"no row for {absent}" if absent else None) \
        or (composed.get("lean") if verdict != "accept" else None)
    if machine_settings.hold_placement():
        # The workload's node composed this record, so none of its accepts stands; one recomputed by the verifier would.
        held = (["the composed verdict"] if verdict == "accept" else []) + \
            sorted(c for c, v in components.items() if v == "accept")
        components = {c: "undecided" if v == "accept" else v for c, v in components.items()}
        if verdict == "accept":
            verdict = "undecided"
        if held and verdict == "undecided":
            reason = "; ".join(x for x in (reason, "not accepted: " + ", ".join(f"{h} {AS_WRITTEN}" for h in held)) if x)
    on = coins_on(cfg, bindings_of(state))
    developer = {p["machine"] for p in cfg["party"] if p["principal"] == "developer"}
    return {"block": seal, "components": components, "answer_from": answer_from, "product": product, "reason": reason,
            "registration": None, "coins": "simulated" if developer & set(on.values()) else coins_of(cfg),
            "coins_on": on, "verdict": verdict, **({"charges_nothing": free} if free else {})}


def _record_line(said: Path) -> dict[str, Any] | None:
    lines = said.read_text().splitlines() if said.is_file() else []
    return next((json.loads(x.split(" ", 1)[1]) for x in reversed(lines) if x.startswith("RECORD ")), None)


def _composed(d: Path) -> bool:
    return (d / "record" / "verdicts.json").is_file()


def remote_check(cfg: Mapping[str, Any], state: Path) -> dict[str, Any]:
    """Remote's verifier line over its final record: the compose gate's answer (``out/compose``, `served_verdicts.sh
    gate-compose`'s) where one was written; else a deferral's follow-on's record (its ``OUT`` copied to
    ``out/follow-on``), composed on a prover node over its composite; else the served node's own record (``out``),
    composed over its composite (``out/composite``, served_composite.sh's, under ``COMPOSE_OFF_NODE=0`` with
    ``COMPOSITE`` not set empty). A record the prover composed says so: its ``source`` (`SERVING_NODE`, `FOLLOW_ON`),
    its reason led by `FROM_SERVING_NODE` or `FROM_FOLLOW_ON`, and each of its accepts, a row's or the composed
    verdict, reads `DEVELOPMENT_VERDICT`, or under ``VERITY_HOLD_PLACEMENT=1`` undecided, `AS_WRITTEN` in the reason, as
    onsite's. Refused by name: ``record`` where the record can't be read or is another product's, ``deferred`` where
    the served node deferred its composite to a follow-on whose record isn't here, ``composite`` where the record it
    reads has no composite beside it."""
    out = state / "out"
    answer, follow = out / "compose", out / "follow-on"
    source = None if _composed(answer) else FOLLOW_ON if _composed(follow) else SERVING_NODE
    base = {None: answer, FOLLOW_ON: follow, SERVING_NODE: out}[source]
    line, f = _record_line(base / "verdicts.txt"), base / "record" / "verdicts.json"
    if line is None or not f.is_file():
        raise Refused("record", f"remote's verifier reads the record in {base}, which has "
                                + ("no RECORD line in verdicts.txt" if line is None else "no record/verdicts.json"))
    if line.get("product") != "remote":
        raise Refused("record", f"the record in {base} is composed for {line.get('product')!r}, not remote")
    if source and not any((base / "composite").glob("*/composite.json")):
        if source == SERVING_NODE and line.get("deferred"):
            raise Refused("deferred", f"the served node deferred its composite to a follow-on on a prover node "
                                      f"({line['deferred']}), and no follow-on record is here: the line reads the "
                                      f"follow-on's OUT copied to {follow}, or the compose gate's answer in {answer}")
        raise Refused("composite", f"remote's record is composed over a composite, and there is none: no compose answer "
                                   f"in {answer} and no composite in {base / 'composite'} (the record there says "
                                   f"{line.get('composed')!r}: {line.get('lean')}): compose on the served node "
                                   "(COMPOSE_OFF_NODE=0, COMPOSITE not set empty) or answer through the compose gate")
    report = json.loads(f.read_text())
    rows = report.get("protocols", {})
    free = {c: why for c in cfg["protocols"] if (why := charges_nothing(c))}
    components = {c: rows.get(ROWS.get(c, c), {}).get("verdict", "absent") for c in cfg["protocols"]}
    held = {f"{c} {v}" for c, v in components.items() if v != "accept"}
    if components.get("inference-only") == "accept" \
            and str(rows[ROWS["inference-only"]].get("answer_from", AS_WRITTEN)).startswith(AS_WRITTEN):
        held.add(f"inference-only {AS_WRITTEN}")
    verdict = VERDICTS.get(line.get("composed"), "undecided")
    reason = line.get("why") or (f"missing {sorted(line['missing'])}" if line.get("missing") else None) \
        or (line.get("lean") if source and verdict != "accept" else None)
    if source:
        word = "undecided" if machine_settings.hold_placement() else DEVELOPMENT_VERDICT
        if word == "undecided":
            held |= {f"{c} {AS_WRITTEN}" for c, v in components.items() if v == "accept"}
            held |= {f"the composed verdict {AS_WRITTEN}"} if verdict == "accept" else set()
        components = {c: word if v == "accept" else v for c, v in components.items()}
        verdict = word if verdict == "accept" else verdict
    reason = reason or (f"not accepted: {', '.join(sorted(held))}" if held else None)
    if source:
        reason = "; ".join(x for x in ({SERVING_NODE: FROM_SERVING_NODE, FOLLOW_ON: FROM_FOLLOW_ON}[source], reason) if x)
    on, workload = gate_coins_on(cfg, bindings_of(state)), {p["name"]: p["machine"] for p in cfg["party"]}[WORKERS]
    return {"block": report.get("window"), "components": components, "product": REMOTE, "reason": reason,
            "registration": None, "coins": "simulated" if workload in on.values() else coins_of(cfg), "coins_on": on,
            "verdict": "undecided" if verdict == "accept" and held else verdict,
            **({"charges_nothing": free} if free else {}),
            **({"answer_from": said} if (said := unchecked(cfg["protocols"], {})) else {}),
            **({"source": source} if source else {})}


def pending(cfg: Mapping[str, Any], state: Path, rc: int) -> str | None:
    if rc != 2 or K.composite(cfg["protocols"], MANIFESTS) != REMOTE:
        return None
    said = state / "out" / "verdicts.txt"
    handed = next((x for x in reversed(said.read_text().splitlines() if said.is_file() else [])
                   if x.startswith("HANDOFF ")), None)
    return None if handed is None or handed.startswith("HANDOFF refused") else handed


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


def coins_on(cfg: Mapping[str, Any], bindings: Mapping[str, Any]) -> dict[str, str]:
    """The machine each listed component's attester's key and coins are made on, for the pieces that have a
    ``development`` binding: the attester's where it is bound, else the piece's own."""
    at = {p["name"]: p["machine"] for p in cfg["party"]}
    return {c: at[x.reaches] if x.development in x._bindings(bindings) else at[x.party] for c in cfg["protocols"]
            for x in launched(c).values() if x.development}


def gate_coins_on(cfg: Mapping[str, Any], bindings: Mapping[str, Any]) -> dict[str, str]:
    """The machine each listed component's challenger coins are flipped on, for the workers' pieces whose gate is
    optional: the challenger's where its gate or its loopback host is bound, else the workload's."""
    at = {p["name"]: p["machine"] for p in cfg["party"]}
    return {c: at[CHALLENGER] if {"gate", "loopback"} & set(x._bindings(bindings)) else at[WORKERS]
            for c in cfg["protocols"]
            if (x := launched(c).get(WORKERS)) and x.reaches == CHALLENGER and "gate" in x.optional}


class ServedWorkload(NamedTuple):
    """The served window's workload as today's road reaches it: the configuration, the run's state, its deploy bindings
    and the parties on this machine. The adapter's runner builds and commits its compiler, advice and trace."""

    cfg: Mapping[str, Any]
    state: Path
    bindings: Mapping[str, Any]
    here: Sequence[str]


def public(cfg: Mapping[str, Any]) -> period.PublicParameters:
    """`run`'s public parameters as the application file states them: the window its class as the adapter states it
    for the file's input dataset (`verity_vllm.inputs.serves.served`), and the sampling design its ``[service] law``.
    No key states the hardware or the policy's limits, nor the class of a window an application names no dataset for.
    Refused by name (``params``) where the file states no law."""
    w = cfg["workload"]["params"]
    if (law := cfg.get("service", {}).get("law")) is None:
        raise Refused("params", "the application file states no [service] law")
    window = None
    if "dataset" in w:
        from verity_vllm.inputs import serves
        window = serves.served(w["model"], w["revision"], w["precision"], w["slots"], w["window-steps"]).window_class
    return period.PublicParameters(hardware=None, limits=None, window=window, sampling_design=law)


#: what registration says where no runtime party is listed, which is every application today
REGISTRATION_PERFORMS_NOTHING = "registration performed nothing: no runtime party is listed"


def _registration(public: period.PublicParameters, w: ServedWorkload, state: period.PeriodState) -> dict[str, Any]:
    """What today's road does before anything starts: every listed piece's bindings given, each runtime party
    prepared, and where one is, the run's registration (`registration`). Refused by name (``unbound``) while a listed
    piece's binding is missing, and ``unwritten`` (naming the goal) while a runtime party's start, `held` or the
    registration is not written. Its ``prepared`` parties are the accounting stage's to start, and with none it
    performs nothing and says so (``note``)."""
    runtime = [x for p in w.here for x in pieces(w.cfg, p) if x.runs == RUNTIME]
    if missing := unbound(w.cfg, w.bindings) if WORKERS in w.here else list(dict.fromkeys(
            k for x in runtime for k in x.unbound(w.bindings))):
        raise Refused("unbound", f"the deploy bindings (verity run --bindings) don't bind {', '.join(missing)}")
    ready = []
    for x in runtime:
        try:
            ready.append((x, prepare(w.cfg, w.state, w.bindings, x)))
        except NotImplementedError as e:
            raise Refused("unwritten", f"{x.party}'s {x.component} party start is not written ({e})") from None
    if ready:
        try:
            registration(w.cfg, w.state)
        except NotImplementedError as e:
            raise Refused("unwritten", f"the run's registration is not written ({e})") from None
    return {"prepared": ready, **({} if ready else {"note": REGISTRATION_PERFORMS_NOTHING})}


#: the period's one window today: the one the runner picks from the serve dir's record (WINDOW, else plans.jsonl's last)
PICKED = "the window the runner picks"


def _windows(public: period.PublicParameters, w: ServedWorkload, state: period.PeriodState) -> tuple[str, ...]:
    return (PICKED,)


#: the window's accounting steps in their order: compilation's record and the accounting protocols' pieces, then what
#: reads the sampled proofs, which do not run
ACCOUNTING = ("preflight", "seal", "window", "advice", "circuits", "sampled-proofs", "pieces", "wipe", "network-attester",
              "record", "composite")


def _accounting(public: period.PublicParameters, w: ServedWorkload, state: period.PeriodState,
                window: str) -> dict[str, dict[str, Any]]:
    """Each party's outcome here: the served window's runner (`served_window`) on the accounting steps (`ACCOUNTING`)
    when the workload's party is here, with each prepared runtime party's run beside it."""
    out: dict[str, dict[str, Any]] = {}

    def done(x: Piece, got: Mapping[str, Any]) -> None:
        mine = out.setdefault(x.party, {"outcome": {}, "refused": None})
        mine["outcome"][x.component] = got.get("outcome")
        mine["refused"] = mine["refused"] or got.get("refused")
        if (where := placement(w.cfg, x, got.get("outcome"))) is not None:
            mine.setdefault("placement", {})[x.component] = where

    ready = state.registration["prepared"]
    with ThreadPoolExecutor(max_workers=max(1, len(ready))) as pool:
        beside = [(x, pool.submit(start)) for x, start in ready]
        if WORKERS in w.here:
            rc = served_window(w.cfg, w.state, w.bindings, ACCOUNTING, "served-window")
            wait = pending(w.cfg, w.state, rc)
            for p in w.here:
                if any(x.runs == SERVED_WINDOW for x in pieces(w.cfg, p)):
                    out[p] = {"outcome": {"served-window": rc, **({"pending": wait} if wait else {})},
                              "refused": None if rc == 0 or wait else SERVED_WINDOW}
        for x, f in beside:
            done(x, f.result())
    return out


#: the inference-only policy's steps in their order; block-counts judges its counts against the policy's c, k and n in
#: the process that counts them, so it is the policy's whole
POLICY = ("inference-policy", "block-counts", "inference-only")
#: the runner's exits where it reached its steps' end, every step run (0) or some refused or not run (20)
ENDED = (0, 20)


def worse(a: int, b: int) -> int:
    """Of two runner exits, one that did not reach its steps' end (not in `ENDED`) over one that did, else the larger."""
    return max(a, b, key=lambda rc: (rc not in ENDED, rc))


def _policy(public: period.PublicParameters, w: ServedWorkload, state: period.PeriodState, window: str,
            outcomes: Mapping[str, Any]) -> int | None:
    """The runner on the policy's steps (`POLICY`), its exit, where the workload's party is here and the accounting
    stage's runner reached its end (`ENDED`); else None."""
    if WORKERS not in w.here or not any(o["outcome"].get("served-window") in ENDED for o in outcomes.values()):
        return None
    return served_window(w.cfg, w.state, w.bindings, POLICY, "policy")


def _result(public: period.PublicParameters, w: ServedWorkload, state: period.PeriodState) -> Result:
    """Each party's outcome: its window's (the `worse` of the accounting's and the policy's exits, and the policy's as
    ``policy`` where it is not 0, since both calls end 20 on a step refused or not run), else a gate's side
    as started by the gate; and when the verifier is here, the onsite line (`remote_check` under remote), with each memory attester's `trusted-device` premise as `placement`
    read it from the challenger's outcome here, each listed component's as `attested` reads it (``trusted-device``:
    component -> premise), and beside it what the wipe, compilation and registration said (``memory-wipe``,
    ``compilation``, and ``registration`` where it performed nothing)."""
    # a party at a time: a later window's outcome would replace an earlier one's, and the period has one window today
    out = {p: o for x in state.windows for p, o in x["accounting"]["outcomes"].items()}
    said_period = _period(w, state)
    for x in state.windows:
        for o in out.values():
            if x["policy"] and o["outcome"].get("served-window") in ENDED:
                o["outcome"]["served-window"] = worse(o["outcome"]["served-window"], x["policy"])
                o["outcome"]["policy"], o["refused"] = x["policy"], SERVED_WINDOW
    for p in w.here:
        if p in out or p == VERIFIER:
            continue
        mine = pieces(w.cfg, p)
        if any(x.runs == SERVED_WINDOW for x in mine):
            out[p] = {"outcome": {"error": f"{p}'s pieces run inside the served window, whose {WORKERS} this run "
                                           "doesn't run"}, "refused": SERVED_WINDOW}
        else:
            out[p] = {"outcome": {"started": f"by the auditor's gate, for {', '.join(x.component for x in mine)}"},
                      "refused": None}
    if VERIFIER in w.here:
        devices = {c: where[mc.PREMISE] for x in out.values() for c, where in x.get("placement", {}).items()}
        devices |= attested(w.cfg, w.state)
        said = "; ".join(sorted({x["accounting"]["compilation"]["note"] for x in state.windows}))
        try:
            line = remote_check if K.composite(w.cfg["protocols"], MANIFESTS) == REMOTE else onsite_check
            beside = {"memory-wipe": state.wipe["note"], "compilation": said,
                      **({"registration": n} if (n := state.registration.get("note")) else {})}
            out[VERIFIER] = {"outcome": {"verdicts": [{**line(w.cfg, w.state), mc.PREMISE: devices, **beside}]},
                             "refused": None}
        except Refused as e:
            out[VERIFIER] = {"outcome": {"error": e.detail}, "refused": e.name}
    return Result(out, said_period)


ROAD_END = "ROAD END (the window's record): "


def _end(state: Path, log: str) -> str | None:
    """A runner call's ROAD END in ``state/logs/<log>.log``, what ran, was refused or undecided and was not run; None
    where the call printed none."""
    f = state / "logs" / f"{log}.log"
    lines = f.read_text().splitlines() if f.exists() else []
    end = next((x for x in reversed(lines) if x.startswith(ROAD_END)), None)
    return end and end.removeprefix(ROAD_END).split("; no verdict stands", 1)[0]


def _period(w: ServedWorkload, state: period.PeriodState) -> dict[str, Any]:
    """The period's one line (``PERIOD``): each stage's runner exit and ROAD END, the outcome, the `worse` of the two
    exits (the workload's ``served-window``), and what the wipe, compilation and registration said. It decides
    nothing."""
    # the period has one window today: a later window's would replace an earlier one's
    acc = pol = None
    for x in state.windows:
        acc = next((o["outcome"]["served-window"] for o in x["accounting"]["outcomes"].values()
                    if "served-window" in o["outcome"]), None)
        pol = x["policy"]
    why = (f"not run here: {WORKERS} is not on this machine" if acc is None else
           f"not run: the accounting stage's runner did not reach its steps' end (exit {acc})")
    prepared = state.registration["prepared"]
    return {"accounting": {"exit": acc, "end": _end(w.state, "served-window") if acc is not None else why},
            "policy": {"exit": pol, "end": _end(w.state, "policy") if pol is not None else why},
            "outcome": worse(acc, pol) if acc is not None and pol is not None else acc,
            "compilation": "; ".join(sorted({x["accounting"]["compilation"]["note"] for x in state.windows})),
            "memory-wipe": state.wipe["note"],
            "registration": state.registration.get("note") or
            f"performed: the run's registration, for {', '.join(x.party for x, _ in prepared)}"}


STAGES = period.Stages(_registration, _windows, _accounting, _policy, _result)


def live(cfg: Mapping[str, Any], state: Path, here: Sequence[str]) -> Result:
    """The parties ``here`` as one monitoring period of `run` (`period.run_period`), over `STAGES`."""
    w = ServedWorkload(cfg, state, bindings_of(state), here)
    return period.run_period(public(cfg), w, STAGES).result

__all__ = ["MANIFESTS", "NOT_YET", "ORDER", "PIECES", "REGISTRATION_GOAL", "SERVED", "STARTS",
           "WORKLOADS", "Held", "Piece", "Result", "Session", "attested", "held", "launched", "live",
           "onsite_check", "pending", "pieces", "placement", "prepare", "registration", "remote_check",
           "request_bound", "served_window", "unbound", "variables"]
