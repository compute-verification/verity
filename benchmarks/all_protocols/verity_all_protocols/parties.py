"""The toy's machine for sampled proofs' parties, and the command that runs them: `verity run CONFIG --party ROLE` on the toy
(`integrations/torch_loop`).

    python -m verity_all_protocols.parties CONFIG --state DIR (--local | --party ROLE ...) [--at ROLE=HOST:PORT ...]
        [--tools FILE] [--tamper I,...]

The parties, their messages and their table are core's (`verity.core.protocols.sampled_proofs.parties`); this module
gives each the capabilities this machine provides (`capabilities`), over the parties' shared state ``DIR``:

* the workers' ``serve``: the toy serves sampling block ``i`` (`runner.loop_window`: the verifier's registration of its
  circuit, then the loop) into ``STATE/blocks/w<i>/``, where the rows and salts stay, its public files in
  ``STATE/public/w<i>/``; the block's registration is its ``registration.json``.
* the firewall's ``pin``: the Lean ``flock-firewall`` every outer proof of a drawn member goes through, one at a time
  (`runner.ONE_AT_A_TIME`), by its SHA-512, also in ``STATE/firewall/ready.json``.
* the challenger's ``host``: a development challenger host under ``STATE/challenger/host``
  (`benchmarks/one_stage/challenger_party.py`), answering the audit's open and finish on loopback TCP.
* the verifier's ``audit``: the block's one-stage audit (`legs/sampled_proofs.py`: ``records_audit`` and the Lean verifier
  of record, its prover sessions behind the pinned firewall) on that host's coins, into
  ``STATE/record/w<i>/sampled-proofs/`` (``answer.json`` and ``case.json``); ``component``: another component's
  ``STATE/record/w<i>/<name>/answer.json`` and ``case.json``; ``product``: onsite's ``product IDS check``.

``--local`` runs the four parties in one run of the deployment layer (`runtime.run`, live). ``--party ROLE`` runs only
those, the others meeting them over TCP: the challenger, the firewall and the workers each listen at their ``--at``
address for the verifier, which dials them, so the verifier starts once they listen; every process of a run is given
the same ``--at``, since the addresses are the deployment's. The channels authenticate with party keys made for the
run (development keys: the configuration names none). Each run's transcripts and record are under
``STATE/runs/<here>/`` (``local``, or the parties the process runs, joined by ``+``), the deployment layer's state.

``coins`` is ``simulated`` under ``--local`` or when the configuration places the challenger on a machine a prover's
party is on (its host is then the prover's, so `trusted-nebius-host` is unmet), else ``development``. The verifier's
lines (core's), with the research run's id as ``run``, are printed as ``VERDICT {...}`` and appended to
``STATE/record/verdicts.jsonl``: 0 when every block's product accepts, 1 when one rejects, 2 otherwise; another
party's process exits 0 when its party finished unrefused, else 2.

``STATE/tools.json`` names the built executables (`build.sh`'s), built there on first use. ``STATE/main.txt`` is main's
first-parent history (``git rev-list --first-parent origin/main``), which the challenger's pin must be on. ``--tamper``:
those blocks' workers record outputs that are not the model's (``serve --tamper``), and the prover skips its
self-check, so a drawn member's proof is of a false witness: a cheating prover, for the demonstration.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity.core.deploy.runtime import RunResult, run
from verity.core.protocols.sampled_proofs import parties as SP, remote as R
from verity.core.wire import Refused
from verity.ml.crypto import lms

from . import compose as C, runner as RN
from .table import load

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "benchmarks" / "all_protocols" / "build.sh"
IDS = SP.IDS
ROLES = tuple(SP.PROTOCOL.parties)
SPOKES = (SP.CHALLENGER, SP.FIREWALL, SP.WORKERS)
#: the principals of a prover's party in a configuration
PROVER = ("prover", "developer")
#: the toy's shape when the configuration names none
WORKLOAD = {"name": "toy", "requests": [3], "steps": 1, "cap": RN.CAP, "blocks": 1, "tamper": []}
#: a party key's LMS parameters: a run's handshakes sign once a channel
KEY = lms.Params(64, 1, 3)


def config_of(path: Path) -> dict[str, Any]:
    """The configuration at ``path`` (`verity config check` is deploy-cli's; this reads only what these parties use)."""
    config = tomllib.loads(path.read_text())
    try:
        SP.params_of(1, config.get("protocols", []), "simulated")
    except ValueError as e:
        raise SystemExit(f"{path}: protocols lists {config.get('protocols')}: {e}") from None
    if workload(config)["name"] != "toy":
        raise SystemExit(f"{path}: these parties serve the toy ([workload] name = \"toy\"), not {workload(config)['name']!r}")
    if sorted(p["name"] for p in config.get("party", [])) != sorted(ROLES):
        raise SystemExit(f"{path}: its [[party]] entries are sampled proofs' {sorted(ROLES)}")
    return config


def workload(config: Mapping[str, Any]) -> dict[str, Any]:
    return {**WORKLOAD, **config.get("workload", {})}


def tools(state: Path) -> dict[str, str]:
    """``STATE/tools.json``, built there by `build.sh` when it isn't yet."""
    f = state / "tools.json"
    if not f.exists():
        d = state / "build"
        r = subprocess.run(["bash", str(BUILD), str(d)], cwd=ROOT, check=False)
        if r.returncode:
            raise SystemExit(f"building the sampled-proofs tools failed ({r.returncode}): see {d}/*.log")
        f.write_text((d / "tools.json").read_text())
    return json.loads(f.read_text())


def main_history(state: Path) -> Path:
    """``STATE/main.txt``: main's first-parent history, newest first, from this clone's ``origin/main``."""
    f = state / "main.txt"
    if not f.exists():
        r = subprocess.run(["git", "-C", str(ROOT), "rev-list", "--first-parent", "refs/remotes/origin/main"],
                           capture_output=True, text=True, check=False)
        if r.returncode or not r.stdout.strip():
            raise SystemExit("the verifier's copy of main's history needs origin/main: `git fetch origin main` "
                             f"(git: {r.stderr.strip()[:300]})")
        f.write_text(r.stdout)
    return f


def _challenger_party() -> Any:
    path = ROOT / "benchmarks" / "one_stage" / "challenger_party.py"
    spec = importlib.util.spec_from_file_location("challenger_party", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def product(exe: str, ids: Sequence[str], cases: Mapping[str, Any], digests: Mapping[str, str]) -> dict[str, Any]:
    """``Core.Composition.product`` over exactly ``ids``, on their cases and digests: its id and its answer."""
    argv = [exe, ",".join(ids)]
    pid = subprocess.run([*argv, "id"], capture_output=True, text=True, check=True).stdout.strip()
    got = C.onsite_check(argv, dict(cases), dict(digests), pid)
    return {"product": pid, "verdict": got["verdict"], "reason": got.get("reason")}


class Machine:
    """This machine's capabilities for the parties (module docstring), over ``state``."""

    def __init__(self, config: Mapping[str, Any], state: Path) -> None:
        self.config, self.state = config, state

    def serve(self, i: int) -> bytes:
        w, t = workload(self.config), tools(self.state)
        a = SimpleNamespace(law=self.config.get("service", {}).get("law", RN.LAW), cap=w["cap"],
                            requests=",".join(map(str, w["requests"])), steps=w["steps"], pouw=None, weights=None,
                            tamper=set(w["tamper"]), windows=w["blocks"], node=os.uname().nodename, delta=None, tools=t)
        (self.state / "blocks").mkdir(parents=True, exist_ok=True)
        RN.loop_window(i, os.environ.get("RESEARCH_RUN_ID") or "local", a, self.state / "blocks",
                       self.state / "public" / f"w{i}")
        return (self.state / "blocks" / f"w{i}" / "registration.json").read_bytes()

    def pin(self) -> dict[str, Any]:
        exe = Path(tools(self.state)["flock-firewall"])
        if not os.access(exe, os.X_OK):
            raise RuntimeError(f"{exe} is not an executable")
        got = {"firewall": str(exe), "sha512": hashlib.sha512(exe.read_bytes()).hexdigest(),
               "sessions": RN.ONE_AT_A_TIME}
        d = self.state / "firewall"
        d.mkdir(parents=True, exist_ok=True)
        (d / "ready.json").write_text(json.dumps(got, sort_keys=True) + "\n")
        return got

    def host(self) -> Any:
        return _challenger_party().Host(self.state / "challenger" / "host",
                                        R.main_lines(str(main_history(self.state))))

    def audit(self, i: int, registration: bytes, challenger: Mapping[str, Any], firewall: Mapping[str, Any]
              ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        t, block = tools(self.state), self.state / "blocks" / f"w{i}"
        if hashlib.sha512(Path(t["flock-firewall"]).read_bytes()).hexdigest() != firewall["sha512"]:
            raise RuntimeError("the firewall the prover's sessions run is not the one the firewall party pinned")
        if (block / "registration.json").read_bytes() != registration:
            raise RuntimeError(f"{block}/registration.json is not the registration the workers sent")
        out = self.state / "record" / f"w{i}" / "sampled-proofs"
        out.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("RAYON_NUM_THREADS", str(os.cpu_count() or 1))
        os.environ.setdefault("RUST_MIN_STACK", str(64 << 20))
        try:
            ans = load("sampled-proofs").leg(block, os.uname().nodename, out, challenger={
                "mode": "loopback:" + challenger["address"], "main": str(main_history(self.state))})
        except Exception as e:  # noqa: BLE001: a leg that can't decide is the block's error, never a verdict
            ans = {"verdict": "error", "reason": f"{type(e).__name__}: {e}"[:2000]}
        (out / "answer.json").write_text(json.dumps(ans, sort_keys=True))
        case = _json(out / "sampled-proofs.case.json")
        if case is not None:
            (out / "case.json").write_text(json.dumps(case, sort_keys=True))
        return ans, case

    def component(self, i: int, name: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        rec = self.state / "record" / f"w{i}" / name
        ans, case = _json(rec / "answer.json"), _json(rec / "case.json")
        return None if ans is None or case is None else (ans, case)

    def product(self, ids: Sequence[str], cases: Mapping[str, Any], digests: Mapping[str, str]) -> dict[str, Any]:
        return product(tools(self.state)["product"], ids, cases, digests)

    def capabilities(self, party: str) -> dict[str, Any]:
        if party == SP.CHALLENGER:
            return {"host": self.host()}
        return {SP.FIREWALL: {"pin": self.pin}, SP.WORKERS: {"serve": self.serve},
                SP.VERIFIER: {"audit": self.audit, "component": self.component, "product": self.product}}[party]


class PartyKey:
    """A party key made for the run (development: no key record), which signs its channels' handshakes."""

    def __init__(self) -> None:
        self.key, self.q = lms.keygen(KEY, os.urandom(64)), 0
        self.public_key = lms.public_key(self.key)

    def sign(self, message: bytes) -> bytes:
        self.q += 1
        return lms.sign_at(self.key, self.q - 1, message).encode()


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    return lms.verify(KEY, public, message, lms.Signature.decode(KEY, signature))


def coins_of(config: Mapping[str, Any], local: bool) -> str:
    """``simulated`` or ``development`` (module docstring)."""
    at = {p["name"]: p for p in config["party"]}
    shared = any(at[p]["machine"] == at[SP.CHALLENGER]["machine"] and at[p].get("principal") in PROVER
                 for p in (SP.FIREWALL, SP.WORKERS))
    return "simulated" if local or shared else "development"


def deployment(config: Mapping[str, Any], local: bool, at: Mapping[str, str]) -> dict[str, Any]:
    """The run as the deployment layer reads it: core's protocol and parameters, and each party with the address a
    listening one listens at."""
    w = workload(config)
    return {"protocol": SP.ID, "params": SP.params_of(w["blocks"], config["protocols"], coins_of(config, local)),
            "party": [{"name": p, **({"listen": at[p]} if p in at and p in SPOKES else {})} for p in ROLES]}


def run_parties(config: Mapping[str, Any], state: Path, here: Sequence[str], *, local: bool,
                at: Mapping[str, str] | None = None) -> tuple[RunResult | None, int]:
    """The parties ``here`` of ``config``'s run (module docstring): the run, and the process's exit code."""
    at = dict(at or {})
    dial = {} if local or SP.VERIFIER not in here else {p: at[p] for p in SPOKES if p not in here and p in at}
    if missing := [p for p in SPOKES if not local and p not in here and SP.VERIFIER in here and p not in at]:
        print(f"verifier: {', '.join(missing)} runs elsewhere and no --at says where it listens", file=sys.stderr)
        return None, 2
    m = Machine(config, state)
    try:
        r = run(deployment(config, local, {} if local else at), list(here), "live",
                state / "runs" / ("local" if local else "+".join(here)),
                protocols=SP.PROTOCOLS, keys={p: PartyKey() for p in here}, verify=verify,
                capabilities={p: m.capabilities(p) for p in here}, dial_to=dial or None)
    except (Refused, OSError) as e:
        print(f"the run refused: {e}", file=sys.stderr)
        return None, 2
    print("RUN " + json.dumps({"status": r.status, "run": r.run,
                               "refused": {p: o["refused"] for p, o in sorted(r.outcomes.items())}}, sort_keys=True),
          flush=True)
    for p, o in sorted(r.outcomes.items()):
        if o["refused"] is not None and p != SP.VERIFIER:
            print(f"{p}: refused {o['refused']}: {json.dumps(o['outcome'], sort_keys=True, default=str)[:2000]}",
                  file=sys.stderr)
    if SP.VERIFIER not in here:
        return r, 0 if all(o["refused"] is None for o in r.outcomes.values()) else 2
    lines = [{**x, "run": os.environ.get("RESEARCH_RUN_ID") or "local"}
             for x in (r.outcomes[SP.VERIFIER]["outcome"] or {}).get("verdicts", [])]
    (state / "record").mkdir(parents=True, exist_ok=True)
    for line in lines:
        print("VERDICT " + json.dumps(line, sort_keys=True), flush=True)
        with open(state / "record" / "verdicts.jsonl", "a") as f:
            f.write(json.dumps(line, sort_keys=True) + "\n")
    verdicts = {x["verdict"] for x in lines}
    if r.outcomes[SP.VERIFIER]["refused"] is not None or not lines:
        print(f"verifier: refused {r.outcomes[SP.VERIFIER]['refused']} after {len(lines)} block(s)", file=sys.stderr)
        return r, 1 if "reject" in verdicts else 2
    return r, 0 if verdicts == {"accept"} else 1 if "reject" in verdicts else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--state", required=True, help="the parties' shared state directory")
    ap.add_argument("--party", action="append", choices=ROLES, help="a party to run here (repeatable)")
    ap.add_argument("--local", action="store_true", help="every party here, its coins simulated")
    ap.add_argument("--at", action="append", default=[], metavar="ROLE=HOST:PORT",
                    help="where a listening party (challenger, firewall, workers) listens for the verifier")
    ap.add_argument("--tools", help="built executables (build.sh's tools.json), instead of building them into STATE")
    ap.add_argument("--tamper", default="", help="blocks whose workers tamper with their outputs, e.g. 0")
    a = ap.parse_args(argv)
    if bool(a.local) == bool(a.party):
        ap.error("give --local (every party here) or --party ROLE")
    at = dict(x.split("=", 1) for x in a.at)
    if bad := sorted(set(at) - set(SPOKES)):
        ap.error(f"--at names a listening party ({', '.join(SPOKES)}), not {', '.join(bad)}")
    config = config_of(Path(a.config))
    if a.tamper:
        config.setdefault("workload", {})["tamper"] = [int(x) for x in a.tamper.split(",") if x.strip()]
    state = Path(a.state).resolve()
    state.mkdir(parents=True, exist_ok=True)
    if a.tools:
        (state / "tools.json").write_text(Path(a.tools).read_text())
    here = list(ROLES) if a.local else sorted(set(a.party), key=ROLES.index)
    return run_parties(config, state, here, local=a.local, at=at)[1]


__all__ = ["IDS", "ROLES", "SPOKES", "Machine", "PartyKey", "config_of", "coins_of", "deployment", "main", "product",
           "run_parties", "verify"]

if __name__ == "__main__":
    raise SystemExit(main())
