"""The step-0 table `verity run`'s tests run (`verity_cli.run.Registry`, read with `verity_cli.run.registry`): the seven
subprotocols' declarations to a configuration (`MANIFESTS`), the ``toy`` workload's declaration (`TOY`), each party's
entry or `LIVE` (`PARTIES`), and the live run of sampled proofs' core parties (`live`,
`verity.core.protocols.sampled_proofs.parties`) on a machine (`Machine`).

No workload serves here: `Machine`'s ``serve``, ``pin``, ``host`` and ``audit`` are `SP.Unavailable` until a test's
stand-in gives them, and the network certifier's entry (`network_certifier`) and every other subprotocol's period check
(`CHECKS`) likewise. The toy workload provides every subprotocol's hooks, so both composites, remote and onsite, run on
it.
``component`` reads ``STATE/record/w<I>/<name>/``'s ``answer.json`` and ``case.json``, and ``product`` runs
``STATE/tools.json``'s ``product`` as ``product IDS check PARAMS COMMITMENT EVIDENCE``.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from verity.core.deploy import runtime
from verity.core.deploy.config import Component, Workload
from verity.core.deploy.manifest import Channel, Role
from verity.core.protocols.certifier.warden import WardenParams
from verity.core.protocols.sampled_proofs import parties as SP
from verity.core.protocols.sampled_proofs.one_stage.draw import parse_law
from verity.core.wire import Refused
from verity.ml.circuits.codec import canonical_json
from verity.ml.crypto import lms

Entry = Callable[[Mapping[str, Any], Path, bool], int]
Check = Callable[[Mapping[str, Any], Path, int], tuple[dict[str, Any], dict[str, Any] | None] | None]
#: a party of the live run, not an entry of its own
LIVE = "live"
ROLES = tuple(SP.PROTOCOL.parties)
SPOKES = (SP.CHALLENGER, SP.FIREWALL, SP.WORKERS)
PROVER = ("prover", "developer")
KEY = lms.Params(64, 1, 3)


# --- what each subprotocol and the workload declare --------------------------------------------------------------------

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
    """The network certifier's parameters: `WardenParams`' fields, and ``tie``, which is true."""
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
    """The workload's shape: each request's prompt token (``requests``), its decode ``steps``, and the units a member
    holds at most (``cap``)."""
    if extra := sorted(set(params) - {"requests", "steps", "cap"}):
        raise Refused("unknown-key", f"[workload.params]: {', '.join(extra)} (known: cap, requests, steps)")
    r, s, c = params["requests"], params["steps"], params["cap"]
    if not isinstance(r, list) or not r or not all(type(t) is int and t >= 0 for t in r):
        raise Refused("workload", f"requests is a non-empty list of token ids, not {r!r}")
    if type(s) is not int or s < 1 or type(c) is not int or c < 1:
        raise Refused("workload", f"steps and cap are positive integers, not {s!r} and {c!r}")
    return {"requests": list(r), "steps": s, "cap": c}


_VERIFIER = (Role("verifier", "auditor"),)
MANIFESTS: dict[str, Component] = {c.name: c for c in (
    Component("sampled-proofs", SP.IDS["sampled-proofs"],
              parties=(Role("challenger", "auditor"), Role("firewall", "developer"), *_VERIFIER),
              channels=(Channel("workers", "verifier", ("registration",)),
                        Channel("verifier", "challenger", ("open", "finish"), after=("registration",)),
                        Channel("challenger", "verifier", ("coins", "record"), carries_coins=True, after=("open",)),
                        Channel("verifier", "workers", ("draw",), after=("coins",)),
                        Channel("workers", "firewall", ("witness",), after=("draw",)),
                        Channel("firewall", "verifier", ("proof",), after=("draw",))),
              needs=frozenset({"windows"}), law=_law, default_law="subset:8"),
    Component("pouw", SP.IDS["pouw"], parties=_VERIFIER, needs=frozenset({"windows", "noise"})),
    Component("inference-only", SP.IDS["inference-only"], parties=_VERIFIER, requires=("sampled-proofs",)),
    Component("pous", SP.IDS["pous"], parties=_VERIFIER, needs=frozenset({"store"})),
    Component("bandwidth", SP.IDS["bandwidth"], parties=_VERIFIER, needs=frozenset({"receipts"})),
    Component("certifier", SP.IDS["certifier"], parties=(Role("network-certifier", "auditor"), *_VERIFIER),
              channels=(Channel("network-certifier", "workers", ("out-opening",)),
                        Channel("network-certifier", "verifier", ("out-record",))),
              needs=frozenset({"windows", "ingress-relay"}), requires=("sampled-proofs",),
              colocated=(("challenger", "network-certifier"),), params=_warden,
              default_params={**asdict(WardenParams()), "tie": True}, law=_work_law, default_law="work:64"),
    Component("wipe", "verity/wipe/v0", parties=_VERIFIER),
)}
TOY = Workload("toy", frozenset({"windows", "ingress-relay", "noise", "store", "receipts"}), (Role("workers", "developer"),),
               _toy, {"requests": [3, 7], "steps": 2, "cap": 8})


# --- the parties -------------------------------------------------------------------------------------------------------

def run_of(state: Path) -> dict[str, Any]:
    """The run's ``run.json``, which `verity run` writes before any party starts: its id, periods and hooks."""
    return json.loads((state / "run.json").read_text())


def _json_if(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def network_certifier(cfg: Mapping[str, Any], state: Path, local: bool) -> int:
    """The network certifier's entry: no relay serves here (module docstring)."""
    print("network-certifier: no relay serves at step 0 here", flush=True)
    return 2


def unserved(what: str) -> Check:
    """A period check that no machine here serves: `SP.Unavailable` for every period."""
    def check(cfg: Mapping[str, Any], state: Path, i: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
        raise SP.Unavailable(f"no {what} serves period {i} at step 0 here")
    return check


#: each listed subprotocol's period check on sampled proofs' verifier (its ``component``), sampled proofs' own aside
CHECKS: dict[str, Check] = {"certifier": unserved("network-period check"), "pouw": unserved("PoUW window check"),
                            "inference-only": unserved("inference-only policy check"),
                            "pous": unserved("PoUS run check"), "bandwidth": unserved("remote bandwidth session"),
                            "wipe": unserved("memory challenger's wipe check")}


class Machine:
    """This machine's capabilities for sampled proofs' parties over ``state`` (module docstring)."""

    def __init__(self, config: Mapping[str, Any], state: Path, checks: Mapping[str, Check] = CHECKS) -> None:
        self.config, self.state, self.checks = config, state, checks

    def serve(self, i: int) -> bytes:
        raise SP.Unavailable(f"no workload serves period {i} at step 0 here")

    def pin(self) -> dict[str, Any]:
        raise SP.Unavailable("no firewall is pinned at step 0 here")

    def host(self) -> Any:
        raise SP.Unavailable("no challenger's host runs at step 0 here")

    def audit(self, i: int, registration: bytes, challenger: Mapping[str, Any], firewall: Mapping[str, Any]
              ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        raise SP.Unavailable(f"no records audit of period {i} runs at step 0 here")

    def component(self, i: int, name: str) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
        check = self.checks.get(name)
        if check is not None:
            return check(self.config, self.state, i)
        rec = self.state / "record" / f"w{i}" / name
        ans, case = _json_if(rec / "answer.json"), _json_if(rec / "case.json")
        return None if ans is None or case is None else (ans, case)

    def product(self, ids: Sequence[str], cases: Mapping[str, Any], digests: Mapping[str, str]) -> dict[str, Any]:
        exe = json.loads((self.state / "tools.json").read_text())["product"]
        argv = [exe, ",".join(ids)]
        pid = subprocess.run([*argv, "id"], capture_output=True, text=True, check=True).stdout.strip()
        order = sorted(cases)
        files = {"params.json": canonical_json({"components": [{"id": c, "params": cases[c]["params"]} for c in order]}),
                 "commitment.bin": canonical_json({c: digests[c] for c in order}),
                 "evidence.json": canonical_json({"components": [{"commitment": cases[c]["commitment"],
                                                                  "evidence": cases[c]["evidence"], "id": c,
                                                                  "params": cases[c]["params"]} for c in order]})}
        with tempfile.TemporaryDirectory() as d:
            for name, data in files.items():
                (Path(d) / name).write_bytes(data)
            p = subprocess.run([*argv, "check", *(str(Path(d) / n) for n in files)], capture_output=True, text=True,
                               check=False)
        if p.returncode not in (0, 1, 2):
            raise SP.Unavailable(f"the product exited {p.returncode}: {p.stderr[-2000:]}")
        got = json.loads(p.stdout.strip().splitlines()[-1])
        return {"product": pid, "verdict": got["verdict"], "reason": got.get("reason")}

    def capabilities(self, party: str) -> dict[str, Any]:
        if party == SP.CHALLENGER:
            return {"host": self.host()}
        return {SP.FIREWALL: {"pin": self.pin}, SP.WORKERS: {"serve": self.serve},
                SP.VERIFIER: {"audit": self.audit, "component": self.component, "product": self.product}}[party]


class PartyKey:
    """A party key made for the run, which signs its channels' handshakes."""

    def __init__(self) -> None:
        self.key, self.q = lms.keygen(KEY, os.urandom(64)), 0
        self.public_key = lms.public_key(self.key)

    def sign(self, message: bytes) -> bytes:
        self.q += 1
        return lms.sign_at(self.key, self.q - 1, message).encode()


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    return lms.verify(KEY, public, message, lms.Signature.decode(KEY, signature))


def coins_of(config: Mapping[str, Any], local: bool) -> str:
    """``simulated`` when every party is here or the challenger shares a machine with a prover's party, else
    ``development``."""
    at = {p["name"]: p for p in config["party"]}
    shared = any(at[p]["machine"] == at[SP.CHALLENGER]["machine"] and at[p].get("principal") in PROVER
                 for p in (SP.FIREWALL, SP.WORKERS))
    return "simulated" if local or shared else "development"


def deployment(config: Mapping[str, Any], local: bool, at: Mapping[str, str]) -> dict[str, Any]:
    """The run as the deployment layer reads it: core's protocol and parameters, and each party with the address a
    listening one listens at."""
    return {"protocol": SP.ID,
            "params": SP.params_of(config["workload"]["blocks"], config["protocols"], coins_of(config, local)),
            "party": [{"name": p, **({"listen": at[p]} if p in at and p in SPOKES else {})} for p in ROLES]}


def listening(cfg: Mapping[str, Any]) -> dict[str, str]:
    """Where each of sampled proofs' listening parties listens for the verifier: its ``address`` (default loopback) and
    its ``listen`` port, for those the configuration gives one."""
    return {p["name"]: f"{p.get('address', '127.0.0.1')}:{p['listen']}" for p in cfg["party"]
            if p["name"] in SPOKES and "listen" in p}


def live(cfg: Mapping[str, Any], state: Path, here: Sequence[str],
         machine: Callable[[Mapping[str, Any], Path], Machine] = Machine) -> runtime.RunResult:
    """The live run of sampled proofs' parties ``here`` (`verity.core.deploy.runtime.run`) on ``machine``: every one
    of them in this process when all four are here, else each listening one at its configured address (`listening`)
    and the verifier dialing those that run elsewhere; refused by name (``listen``) when one the verifier needs has
    none. Its transcripts and record are under ``STATE/runs/<local|here joined by +>/``."""
    local, at = set(here) == set(ROLES), listening(cfg)
    view = {"protocols": list(cfg["protocols"]), "workload": {"blocks": run_of(state)["periods"]},
            "party": [p for p in cfg["party"] if p["name"] in ROLES]}
    dial = {} if local or SP.VERIFIER not in here else {p: at[p] for p in SPOKES if p not in here and p in at}
    if not local and (missing := [p for p in SPOKES if p not in here and SP.VERIFIER in here and p not in at]):
        raise Refused("listen", f"the verifier dials {', '.join(missing)}, which run elsewhere, and the configuration "
                                "gives them no listen port")
    if not local and (silent := [p for p in SPOKES if p in here and p not in at]):
        raise Refused("listen", f"{', '.join(silent)} listen for the verifier, which runs elsewhere, and the "
                                "configuration gives them no listen port")
    m = machine(cfg, state)
    return runtime.run(deployment(view, local, {} if local else at), list(here), "live",
                       state / "runs" / ("local" if local else "+".join(here)), protocols=SP.PROTOCOLS,
                       keys={p: PartyKey() for p in here}, verify=verify,
                       capabilities={p: m.capabilities(p) for p in here}, dial_to=dial or None)


PARTIES: dict[str, dict[str, Entry | str]] = {
    "sampled-proofs": {p: LIVE for p in ("challenger", "firewall", "verifier")},
    "certifier": {"network-certifier": network_certifier, "verifier": LIVE},
    **{c: {"verifier": LIVE} for c in ("pouw", "inference-only", "pous", "bandwidth", "wipe")},
}
WORKLOADS: dict[str, tuple[Workload, Entry | str]] = {"toy": (TOY, LIVE)}
#: the order a machine's entries run in, all before the live run: a party's sends come before the parties that read them
ORDER = ("network-certifier",)
