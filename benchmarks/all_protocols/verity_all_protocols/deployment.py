"""Step 0 of the deployment API (`note:memory-accounting/20261026T2305Z-draft-deployment-api-v3`, addenda 23:46Z to
00:30Z): what `verity run CONFIG` runs, until each protocol's package holds its own.

`MANIFESTS` is what each of the six subprotocols declares to a configuration (`verity.core.deploy.config.Component`):
its id, the workload hooks it needs, the components it requires, its parties and their principals, its channels and
messages, the parties that must share a machine, its parameters and the service laws it runs under. `TOY` is what the
toy (`integrations/torch_loop`) declares as a workload (`Workload`), and ``workers`` is its party. `PARTIES` holds each
party of each subprotocol, ``PARTIES[subprotocol][party]``: an entry, ``run(config, state, local) -> int``, which runs
that party's part of every period of the run in a process of its own, or `LIVE`, a party of the run's one live run of
the deployment layer (`live`):

- sampled proofs: the challenger, the firewall, the workers and the verifier are core's parties
  (`verity.core.protocols.sampled_proofs.parties`), on the toy's machine (`Machine`, `parties.Machine`'s capabilities).
- the network certifier: ``network-certifier`` is an entry, run before the live run: it relays the run's periods
  through the network certifier, committing each period's output at the reading circuit's out shape and signing its
  record (`tie.relay`, the certifier-only path of `legs/certifier.py`). Its ``verifier`` is sampled proofs' verifier,
  whose ``component("certifier")`` is the period's ``network-period check`` on the case the communication leg builds
  (`certifier_check`), right after that period's sampled-proofs audit: until network-accounting's parties replace
  them. The leg's tampered relay is a benchmark's self-test, so a run relays honest traffic alone unless
  ``CERTIFIER_TAMPER`` asks for one.
- PoUW, PoUS, the bandwidth audit and the wipe: none yet, so `verity run` refuses a configuration that lists them.

The toy's ``workers`` serve each period (`Machine.serve`, `runner.loop_window`): the verifier registers the circuit from
the loop's shape, then the loop writes its window into ``STATE/blocks/w<I>/`` (its rows and salts stay there) and its
public files into ``STATE/public/w<I>/``. With the certifier listed (the ``ingress-relay`` hook), each output unit reads
the output the network certifier relayed for the period, pinned from its record, opened from the rows and salts it
handed the workers.

Each period's verdict is core's verifier's: `Core.Composition.product` over exactly the listed components (onsite's
``product IDS check``), each component's status its answer's verdict, ``absent`` when its check gave none, and a
component without a case refused by its decoders.

A run's state (``verity run --state``) is `parties`' layout, with the network certifier's session in
``blocks/certifier-session/`` (witness) and its public files in ``record/certifier-session/``; each period's record is
``record/w<I>/<component>/``, its ``answer.json`` and ``case.json``; the live run's transcripts and record are under
``runs/<local, or the parties here joined by +>/``.
"""
from __future__ import annotations

import json
import os
import shutil
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity.core.deploy.config import Channel, Component, Refused, Workload
from verity.core.deploy import runtime
from verity.core.protocols.certifier.warden import WardenParams
from verity.core.protocols.sampled_proofs import parties as SP
from verity.core.protocols.sampled_proofs.one_stage.draw import parse_law

from . import parties as P, runner as RN, settings as S, tie as T
from .table import check_answer, load

Entry = Callable[[Mapping[str, Any], Path, bool], int]
Check = Callable[[Mapping[str, Any], Path, int], tuple[dict[str, Any], dict[str, Any] | None] | None]
BLOCKS, PUBLIC, RECORD = "blocks", "public", "record"
#: a party of the live run (module docstring), not an entry of its own
LIVE = "live"


# --- what each subprotocol and the toy declare -------------------------------------------------------------------------

def _law(law: str, params: Mapping[str, Any]) -> None:
    try:
        parse_law(law)
    except ValueError as e:
        raise Refused("law", str(e)) from None


def _work_law(law: str, params: Mapping[str, Any]) -> None:
    _law(law, params)
    if law.partition(":")[0] != "work":
        raise Refused("law", f"the certifier's output tie is proved whole only beside a streamed work draw (a5's "
                             f"output-class-needs-a-streamed-work-draw): work:K, not {law}")


def _warden(params: Mapping[str, Any]) -> dict[str, Any]:
    """The network certifier's parameters: `WardenParams`' fields, and ``tie``, which is true: the workers' outputs are
    tied to the ones it relayed, without which its period check has nothing of the workers' to certify."""
    names = [f.name for f in fields(WardenParams)]
    if extra := sorted(set(params) - {*names, "tie"}):
        raise Refused("unknown-key", f"[params.\"verity/certifier/v0\"]: {', '.join(extra)} (known: WardenParams' "
                                     f"{', '.join(names)}, and tie)")
    if params.get("tie") is not True:
        raise Refused("tie", "tie = true: the network certifier ties the workers' outputs to the ones it relayed")
    try:
        WardenParams(**{k: params[k] for k in names})
    except (TypeError, ValueError) as e:
        raise Refused("params", f"verity/certifier/v0: {e}") from None
    return {**{k: params[k] for k in names}, "tie": True}


def _toy(params: Mapping[str, Any]) -> dict[str, Any]:
    """The toy's shape: each request's prompt token (``requests``), its decode ``steps``, and the units a member holds at
    most (``cap``)."""
    if extra := sorted(set(params) - {"requests", "steps", "cap"}):
        raise Refused("unknown-key", f"[workload.params]: {', '.join(extra)} (known: cap, requests, steps)")
    r, s, c = params["requests"], params["steps"], params["cap"]
    if not isinstance(r, list) or not r or not all(type(t) is int and t >= 0 for t in r):
        raise Refused("workload", f"requests is a non-empty list of token ids, not {r!r}")
    if type(s) is not int or s < 1 or type(c) is not int or c < 1:
        raise Refused("workload", f"steps and cap are positive integers, not {s!r} and {c!r}")
    return {"requests": list(r), "steps": s, "cap": c}


_VERIFIER = {"verifier": "auditor"}
MANIFESTS: dict[str, Component] = {c.name: c for c in (
    Component("sampled-proofs", P.IDS["sampled-proofs"],
              parties={"challenger": "auditor", "firewall": "developer", **_VERIFIER},
              messages={"registration": "public", "open": "public", "finish": "public", "coins": "coins",
                        "record": "public", "draw": "public", "witness": "private", "proof": "public"},
              channels=(Channel("workers", "verifier", ("registration",)),
                        Channel("verifier", "challenger", ("open", "finish"), ("registration",)),
                        Channel("challenger", "verifier", ("coins", "record"), ("open",)),
                        Channel("verifier", "workers", ("draw",), ("coins",)),
                        Channel("workers", "firewall", ("witness",), ("draw",)),
                        Channel("firewall", "verifier", ("proof",), ("draw",))),
              needs=frozenset({"windows"}), law=_law, default_law="subset:8"),
    Component("pouw", P.IDS["pouw"], parties=_VERIFIER, messages={}, needs=frozenset({"windows", "noise"})),
    Component("pous", P.IDS["pous"], parties=_VERIFIER, messages={}, needs=frozenset({"store"})),
    Component("bandwidth", P.IDS["bandwidth"], parties=_VERIFIER, messages={}, needs=frozenset({"receipts"})),
    Component("certifier", P.IDS["certifier"], parties={"network-certifier": "auditor", **_VERIFIER},
              messages={"out-opening": "private", "out-record": "public"},
              channels=(Channel("network-certifier", "workers", ("out-opening",)),
                        Channel("network-certifier", "verifier", ("out-record",))),
              needs=frozenset({"windows", "ingress-relay"}), requires=("sampled-proofs",),
              colocated=(("challenger", "network-certifier"),), params=_warden,
              default_params={**asdict(WardenParams()), "tie": True}, law=_work_law, default_law="work:64"),
    Component("wipe", "verity/wipe/v0", parties=_VERIFIER, messages={}, product=False),
)}
TOY = Workload("toy", frozenset({"windows", "ingress-relay"}), {"workers": "developer"}, _toy,
               {"requests": [3, 7], "steps": 2, "cap": RN.CAP})


# --- the parties -------------------------------------------------------------------------------------------------------

def run_of(state: Path) -> dict[str, Any]:
    """The run's ``run.json``, which `verity run` writes before any party starts: its id, periods and hooks."""
    return json.loads((state / "run.json").read_text())


def _loop(cfg: Mapping[str, Any], state: Path, tie: bool) -> SimpleNamespace:
    """The runner's arguments for the toy under ``cfg`` (`runner.loop_window`'s)."""
    w, run = cfg["workload"]["params"], run_of(state)
    return SimpleNamespace(law=cfg["service"]["law"], cap=w["cap"], requests=",".join(map(str, w["requests"])),
                           steps=w["steps"], pouw=None, weights=None, tie=tie, tie_tamper=set(), wired=False,
                           tamper=set(), windows=run["periods"], node=os.uname().nodename, delta=None,
                           tools=P.tools(state), mode={"of_record": False, "configuration": run["configuration"]})


def network_certifier(cfg: Mapping[str, Any], state: Path, local: bool) -> int:
    """The network certifier at step 0 (module docstring): the run's periods relayed through it before the workers
    serve, each period's requests in a link-window of their own, and each link-window's output committed at the
    reading circuit's out shape (``serve --circuit-only --tie``) and signed under its session's key."""
    S.certifier_defaults()
    run, a = run_of(state), _loop(cfg, state, True)
    shape = RN.serve([*RN.shape_of(0, run["run"], a), "--circuit-only"])["out_shape"]
    (state / BLOCKS).mkdir(parents=True, exist_ok=True)
    T.relay(state / BLOCKS, cfg["workload"]["params"]["requests"], a.steps, run["periods"], a.tools, False, shape,
            state / RECORD)
    return 0


def _json(path: Path) -> dict[str, Any] | None:
    """``path``'s JSON, or None when its party wrote none; one it can't read raises."""
    return json.loads(path.read_text()) if path.exists() else None


def certifier_check(cfg: Mapping[str, Any], state: Path, i: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The network certifier's period check, sampled proofs' verifier's ``component(i, "certifier")``: period ``i``'s
    communication leg (`legs/certifier.py`) on its window, after the period's sampled-proofs audit, whose answer it
    reads; its answer and case, also in ``record/w<I>/certifier/``. A leg that raises leaves its error there
    (``error.txt``) and is the period's ``error``, with no case."""
    S.prover_defaults()
    block, rec = state / BLOCKS / f"w{i}", state / RECORD / f"w{i}"
    out = rec / "certifier"
    out.mkdir(parents=True, exist_ok=True)
    try:
        integrity = rec / "sampled-proofs"
        if not (integrity / "answer.json").exists():
            raise RuntimeError(f"no sampled-proofs answer for period {i}: its audit comes first")
        (block / "answers").mkdir(exist_ok=True)
        shutil.copy(integrity / "answer.json", block / "answers" / "sampled-proofs.json")
        if not (rec / "integrity").exists():
            (rec / "integrity").symlink_to("sampled-proofs")
        ans = check_answer(load("certifier").leg(block, os.uname().nodename, out))
    except Exception as e:  # noqa: BLE001  (a leg that can't decide is the period's error, never a verdict)
        (out / "error.txt").write_text(traceback.format_exc())
        ans = {"verdict": "error", "reason": f"{type(e).__name__}: {e}"[:2000]}
        (out / "answer.json").write_text(json.dumps(ans, indent=1, sort_keys=True))
        return ans, None
    (out / "answer.json").write_text(json.dumps(ans, indent=1, sort_keys=True))
    if (out / "certifier.case.json").exists():
        shutil.copy(out / "certifier.case.json", out / "case.json")
    return ans, _json(out / "case.json")


#: each listed component's period check on sampled proofs' verifier (its ``component``), sampled proofs' own aside
CHECKS: dict[str, Check] = {"certifier": certifier_check}


class Machine(P.Machine):
    """The toy's machine for the live run: `parties.Machine`'s capabilities over ``state``, with the workers' ``serve``
    the configuration's workload (`_loop`), tied to the network certifier's relay when the run uses the
    ``ingress-relay`` hook, and the verifier's ``component`` each listed component's check (`CHECKS`)."""

    def __init__(self, config: Mapping[str, Any], state: Path, checks: Mapping[str, Check] = CHECKS) -> None:
        super().__init__(config, state)
        self.checks = checks

    def serve(self, i: int) -> bytes:
        run = run_of(self.state)
        try:
            a = _loop(self.config, self.state, "ingress-relay" in run["hooks"])
            if a.tie:
                relay = self.state / BLOCKS / load("certifier").SESSION / "relay.json"
                if not relay.exists():
                    raise SP.Unavailable(f"no relay of the network certifier ({relay}): its party runs first")
                a.relay = json.loads(relay.read_text())
            (self.state / BLOCKS).mkdir(parents=True, exist_ok=True)
            RN.loop_window(i, run["run"], a, self.state / BLOCKS, self.state / PUBLIC / f"w{i}")
            return (self.state / BLOCKS / f"w{i}" / "registration.json").read_bytes()
        except P.FAILED as e:
            raise SP.Unavailable(f"serving period {i}: {type(e).__name__}: {e}") from e

    def component(self, i: int, name: str) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
        check = self.checks.get(name)
        return super().component(i, name) if check is None else check(self.config, self.state, i)


def listening(cfg: Mapping[str, Any]) -> dict[str, str]:
    """Where each of sampled proofs' listening parties listens for the verifier: its ``address`` (default loopback) and
    its ``listen`` port, for those the configuration gives one."""
    return {p["name"]: f"{p.get('address', '127.0.0.1')}:{p['listen']}" for p in cfg["party"]
            if p["name"] in P.SPOKES and "listen" in p}


def live(cfg: Mapping[str, Any], state: Path, here: Sequence[str],
         machine: Callable[[Mapping[str, Any], Path], Machine] = Machine) -> runtime.RunResult:
    """The live run of sampled proofs' parties ``here`` (`verity.core.deploy.runtime.run`) on ``machine``: every one
    of them in this process when all four are here (their coins then ``simulated``), else each listening one at its
    configured address (`listening`) and the verifier dialing those that run elsewhere; refused by name (``listen``)
    when one the verifier needs has none. Its transcripts and record are under ``STATE/runs/<local|here joined by +>/``.
    """
    local, at = set(here) == set(P.ROLES), listening(cfg)
    view = {"protocols": list(cfg["protocols"]), "workload": {"blocks": run_of(state)["periods"]},
            "party": [p for p in cfg["party"] if p["name"] in P.ROLES]}
    dial = {} if local or SP.VERIFIER not in here else {p: at[p] for p in P.SPOKES if p not in here and p in at}
    if not local and (missing := [p for p in P.SPOKES if p not in here and SP.VERIFIER in here and p not in at]):
        raise Refused("listen", f"the verifier dials {', '.join(missing)}, which run elsewhere, and the configuration "
                                "gives them no listen port")
    if not local and (silent := [p for p in P.SPOKES if p in here and p not in at]):
        raise Refused("listen", f"{', '.join(silent)} listen for the verifier, which runs elsewhere, and the "
                                "configuration gives them no listen port")
    m = machine(cfg, state)
    return runtime.run(P.deployment(view, local, {} if local else at), list(here), "live",
                       state / "runs" / ("local" if local else "+".join(here)), protocols=SP.PROTOCOLS,
                       keys={p: P.PartyKey() for p in here}, verify=P.verify,
                       capabilities={p: m.capabilities(p) for p in here}, dial_to=dial or None)


PARTIES: dict[str, dict[str, Entry | str]] = {
    "sampled-proofs": {p: LIVE for p in ("challenger", "firewall", "verifier")},
    "certifier": {"network-certifier": network_certifier, "verifier": LIVE},
}
WORKLOADS: dict[str, tuple[Workload, Entry | str]] = {"toy": (TOY, LIVE)}
#: the order a machine's entries run in, all before the live run: a party's sends come before the parties that read them
ORDER = ("network-certifier",)


__all__ = ["CHECKS", "LIVE", "MANIFESTS", "ORDER", "PARTIES", "TOY", "WORKLOADS", "Machine", "certifier_check",
           "listening", "live", "network_certifier", "run_of"]
