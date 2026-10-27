"""Step 0 of the deployment API (`note:memory-accounting/20261026T2305Z-draft-deployment-api-v3`, addenda 23:46Z to
00:30Z): what `verity run CONFIG` runs, beside the sampled-proofs parties (`parties`, compute-accounting's), until each
protocol's package holds its own.

`MANIFESTS` is what each of the six subprotocols declares to a configuration (`verity.core.deploy.config.Component`):
its id, the workload hooks it needs, the components it requires, its parties and their principals, its channels and
messages, the parties that must share a machine, its parameters and the service laws it runs under. `TOY` is what the
toy (`integrations/torch_loop`) declares as a workload (`Workload`), and `workers` is its party. `PARTIES` holds each
party's entry, ``PARTIES[subprotocol][party] = run(config, state, local) -> int``, which runs that party's part of every
period of the run:

- sampled proofs: the challenger, the firewall and the verifier are `parties`' (`parties.PARTIES`).
- the network certifier: ``network-certifier`` relays the run's periods through the network certifier before the
  workers serve, committing each period's output at the reading circuit's out shape and signing its record
  (`tie.relay`, the certifier-only path of `legs/certifier.py`), and ``verifier`` decides each period by
  ``network-period check`` on the case the communication leg builds (`legs/certifier.py`), after sampled proofs'
  verifier: until network-accounting's parties replace them. The leg's tampered relay is a benchmark's self-test, so a
  run relays honest traffic alone unless ``CERTIFIER_TAMPER`` asks for one.
- PoUW, PoUS, the bandwidth audit and the wipe: none yet, so `verity run` refuses a configuration that lists them.

The toy's ``workers`` serve each period (`runner.loop_window`): the verifier registers the circuit from the loop's
shape, then the loop writes its window into ``STATE/blocks/w<I>/`` (its rows and salts stay there) and its public files
into ``STATE/public/w<I>/``. With the certifier listed (the ``ingress-relay`` hook), each output unit reads the output the
network certifier relayed for the period, pinned from its record, opened from the rows and salts it handed the workers.

On one machine step 0 runs the parties one after another (`ORDER`), the challenger beside them from the first until the
last (`SERVICES`). `compose` is the verifier's verdict on a period: `Core.Composition.product` over exactly the listed
components (onsite's ``product IDS check``; onsite's own protocol when all five are listed), each component's status its
answer's verdict, ``absent`` when its verifier wrote none, and a component without a case refused by its decoders.

A run's state (``verity run --state``) is `parties`' layout, with the network certifier's session in
``blocks/certifier-session/`` (witness) and its public files in ``record/certifier-session/``; each period's record is
``record/w<I>/<component>/``, its ``answer.json`` and ``case.json``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import traceback
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity.core.deploy.config import Channel, Component, Refused, Workload
from verity.core.protocols.certifier.warden import WardenParams
from verity.core.protocols.sampled_proofs.one_stage.draw import parse_law

from . import compose as C, parties as P, runner as RN, tie as T
from .table import check_answer, load

Entry = Callable[[Mapping[str, Any], Path, bool], int]
BLOCKS, PUBLIC, RECORD = "blocks", "public", "record"
ABSENT = "absent"


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


def workers(cfg: Mapping[str, Any], state: Path, local: bool) -> int:
    """The toy's workers (module docstring): each period served into ``STATE/blocks``, tied to the network certifier's
    relay when the run uses the ``ingress-relay`` hook."""
    run = run_of(state)
    a = _loop(cfg, state, "ingress-relay" in run["hooks"])
    if a.tie:
        relay = state / BLOCKS / load("certifier").SESSION / "relay.json"
        if not relay.exists():
            raise RuntimeError(f"no relay of the network certifier ({relay}): run --party network-certifier first")
        a.relay = json.loads(relay.read_text())
    (state / BLOCKS).mkdir(parents=True, exist_ok=True)
    for i in range(run["periods"]):
        RN.loop_window(i, run["run"], a, state / BLOCKS, state / PUBLIC / f"w{i}")
    return 0


def network_certifier(cfg: Mapping[str, Any], state: Path, local: bool) -> int:
    """The network certifier at step 0 (module docstring): the run's periods relayed through it before the workers
    serve, each period's requests in a link-window of their own, and each link-window's output committed at the
    reading circuit's out shape (``serve --circuit-only --tie``) and signed under its session's key."""
    os.environ.setdefault("CERTIFIER_TAMPER", "")
    run, a = run_of(state), _loop(cfg, state, True)
    shape = RN.serve([*RN.shape_of(0, run["run"], a), "--circuit-only"])["out_shape"]
    (state / BLOCKS).mkdir(parents=True, exist_ok=True)
    T.relay(state / BLOCKS, cfg["workload"]["params"]["requests"], a.steps, run["periods"], a.tools, False, shape,
            state / RECORD)
    return 0


def certifier_verifier(cfg: Mapping[str, Any], state: Path, local: bool) -> int:
    """The network certifier's verifier at step 0: each period's communication leg (`legs/certifier.py`) on the
    period's window, after sampled proofs' verifier, whose answer and audit it reads; its answer and case in
    ``record/w<I>/certifier/``. A leg that raises leaves its error there (``error.txt``) and an error answer, and the
    entry returns 1."""
    os.environ.setdefault("RAYON_NUM_THREADS", str(os.cpu_count() or 1))
    os.environ.setdefault("RUST_MIN_STACK", str(64 << 20))
    leg, failed = load("certifier").leg, 0
    for i in range(run_of(state)["periods"]):
        block, rec = state / BLOCKS / f"w{i}", state / RECORD / f"w{i}"
        out = rec / "certifier"
        out.mkdir(parents=True, exist_ok=True)
        try:
            integrity = rec / "sampled-proofs"
            if not (integrity / "answer.json").exists():
                raise RuntimeError(f"no sampled-proofs answer for period {i}: run sampled proofs' verifier first")
            (block / "answers").mkdir(exist_ok=True)
            shutil.copy(integrity / "answer.json", block / "answers" / "sampled-proofs.json")
            if not (rec / "integrity").exists():
                (rec / "integrity").symlink_to("sampled-proofs")
            ans = check_answer(leg(block, os.uname().nodename, out))
        except Exception as e:  # noqa: BLE001: a leg that can't decide is the period's error, never a verdict
            (out / "error.txt").write_text(traceback.format_exc())
            ans, failed = {"verdict": "error", "reason": f"{type(e).__name__}: {e}"[:2000]}, 1
        (out / "answer.json").write_text(json.dumps(ans, indent=1, sort_keys=True))
        if (out / "certifier.case.json").exists():
            shutil.copy(out / "certifier.case.json", out / "case.json")
    return failed


PARTIES: dict[str, dict[str, Entry]] = {
    "sampled-proofs": {p: e for p, e in P.PARTIES["sampled-proofs"].items() if p != "workers"},
    "certifier": {"network-certifier": network_certifier, "verifier": certifier_verifier},
}
WORKLOADS: dict[str, tuple[Workload, Entry]] = {"toy": (TOY, workers)}
#: the order step 0 runs a machine's parties in: a party's sends come before the parties that read them
ORDER = ("challenger", "firewall", "network-certifier", "workers", "verifier")


@dataclass(frozen=True)
class Service:
    """A party that serves the others while they run: started first, ready once it prints a line starting ``ready``,
    and stopped (``stop(state)``) once the rest are done."""

    ready: str
    stop: Callable[[Path], None]


SERVICES = {"challenger": Service("CHALLENGER ", P.stop)}


# --- the verdict -------------------------------------------------------------------------------------------------------

def _json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def compose(cfg: Mapping[str, Any], state: Path, period: int,
            manifests: Mapping[str, Component] = MANIFESTS) -> dict[str, Any]:
    """Period ``period``'s verdict (module docstring): ``components``, each listed component's status; ``product``,
    the product's id; ``verdict`` and ``reason``, the product's answer. A listed component outside the product (the
    wipe) rejects the verdict when it doesn't accept."""
    exe, rec = P.tools(state)["product"], state / RECORD / f"w{period}"
    listed = list(cfg["protocols"])
    inside = [c for c in listed if manifests[c].product]
    status, cases, digests = {}, {}, {}
    for c in listed:
        ans = _json(rec / c / "answer.json")
        status[c] = ABSENT if ans is None else str(ans.get("verdict", "error"))
        if c in inside:
            cases[manifests[c].id] = _json(rec / c / "case.json") or C.NO_CASE
            digests[manifests[c].id] = (ans or {}).get("commitment") or ""
    argv = [exe, ",".join(sorted(cases))]
    pid = subprocess.run([*argv, "id"], capture_output=True, text=True, check=True).stdout.strip()
    got = C.onsite_check(argv, cases, digests, pid)
    verdict, reason = got["verdict"], got.get("reason")
    for c in (c for c in listed if c not in inside):
        if status[c] not in ("accept", "undecided"):
            verdict, reason = "reject", f"{c} {status[c]}"
        elif status[c] == "undecided" and verdict == "accept":
            verdict, reason = "undecided", f"{c} undecided"
    return {"components": status, "product": pid, "verdict": verdict, "reason": reason}


__all__ = ["ABSENT", "MANIFESTS", "ORDER", "PARTIES", "SERVICES", "TOY", "WORKLOADS", "Service", "certifier_verifier",
           "compose", "network_certifier", "run_of", "workers"]
