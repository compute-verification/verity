"""The sampled-proofs subprotocol's roles, as `verity run CONFIG --party ROLE` runs them (step 0: today's code, each role
a function of the configuration, a state directory and whether every role runs on this machine).

    PARTIES[subprotocol][role] = run(config: dict, state: Path, local: bool) -> int

The roles are the proof service's, on the toy (`integrations/torch_loop`):

* ``workers`` (the prover's): the toy serves each sampling block (`runner.loop_window`: the verifier's registration of
  its circuit, then the loop) into ``STATE/blocks/w<i>/``. The rows and salts stay there: a block's public files are in
  ``STATE/public/w<i>/``.
* ``firewall`` (the prover's): the Lean ``flock-firewall`` every outer proof of a drawn member goes through, one at a
  time (`runner.ONE_AT_A_TIME`). In step 0 each prover session runs it in-line (``records_audit --firewall``); the role
  pins the binary those sessions run in ``STATE/firewall/ready.json``.
* ``challenger`` (the auditor's): its own process (``benchmarks/one_stage/challenger_party.py``), a development
  challenger host under ``STATE/challenger/host`` (its LMS key, signed ledger and pins log), answering the verifier's
  ``open`` and ``finish`` on loopback TCP, its address in ``STATE/challenger/address.json``. It runs until the verifier
  is done (``stop``).
* ``verifier`` (the auditor's): each block's one-stage audit (`legs/sampled_proofs.py`: ``records_audit`` and the Lean
  verifier of record) drawn on the challenger's coins once the block's registration is in, into
  ``STATE/record/w<i>/sampled-proofs/``; then ``Core.Composition.product`` over exactly the configuration's
  ``protocols`` (onsite's ``product IDS check``). A listed component whose party wrote no answer and case for the block
  (``STATE/record/w<i>/<component>/answer.json`` and ``case.json``) is ``absent``: its case is `compose.NO_CASE`, which
  its own decoders refuse, so the product never accepts over it. Each block's line, printed and appended to
  ``STATE/record/verdicts.jsonl``:

      VERDICT {"block": I, "coins": C, "components": {NAME: accept|reject|undecided|error|absent},
               "product": ID, "verdict": V, "reason": R, "run": RUN}

  ``coins`` is ``simulated`` under ``local`` (the challenger's host is the prover's machine, so `trusted-nebius-host`
  is unmet) and ``development`` otherwise (a key made for the session, not an auditor's key record).

Step 0's roles share ``STATE`` on one machine, and run in the configuration's order (each channel's ``after``): the
challenger listening, the firewall pinned, the blocks served, then the verifier. ``local`` runs them all here, the
challenger as a subprocess (`run_local`). The private channels (the prover's rows and salts, the challenger's session
secret) never leave this machine (`WitnessPathPrivate`): the challenger's ``finish`` hands the verifier only the signed
record, its ledger and pins log, and the verifier keeps only public files in ``STATE/record``.

``STATE/tools.json`` names the built executables (`build.sh`'s); a role builds them there on first use. ``STATE/main.txt``
is the verifier's copy of main's first-parent history (``git rev-list --first-parent origin/main``), which the
challenger's pin must be on.

    python -m verity_all_protocols.parties CONFIG --state DIR (--local | --party ROLE ...) [--tools FILE] [--tamper I,...]

``--tamper``: those blocks' workers record outputs that are not the model's (``serve --tamper``), and the prover skips
its self-check, so a drawn member's proof is of a false witness: a cheating prover, for the demonstration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import compose as C, runner as RN
from .table import load

ROOT = Path(__file__).resolve().parents[3]
CHALLENGER = ROOT / "benchmarks" / "one_stage" / "challenger_party.py"
BUILD = ROOT / "benchmarks" / "all_protocols" / "build.sh"
#: a configuration's component names and their protocol ids, the ones `Core.Onsite.components` holds
IDS = {"bandwidth": "verity/remote-bandwidth/v0", "certifier": "verity/certifier/v0", "pous": "verity/pous/p2/v4/run",
       "pouw": "verity/pouw/v1", "sampled-proofs": "verity/sampled-proofs/one-stage/v1"}
#: the toy's shape when the configuration names none
WORKLOAD = {"name": "toy", "requests": [3], "steps": 1, "cap": RN.CAP, "blocks": 1, "tamper": []}
Role = Callable[[Mapping[str, Any], Path, bool], int]


def config_of(path: Path) -> dict[str, Any]:
    """The configuration at ``path`` (`verity config check` is deploy-cli's; this reads only what these roles use)."""
    config = tomllib.loads(path.read_text())
    unknown = [c for c in config.get("protocols", []) if c not in IDS]
    if unknown or not config.get("protocols"):
        raise SystemExit(f"{path}: protocols lists {config.get('protocols')}; each is one of {sorted(IDS)}")
    if workload(config)["name"] != "toy":
        raise SystemExit(f"{path}: these roles serve the toy ([workload] name = \"toy\"), not {workload(config)['name']!r}")
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


def challenger(config: Mapping[str, Any], state: Path, local: bool) -> int:
    """The challenger party in this process's foreground, until the verifier stops it (module docstring)."""
    d = state / "challenger"
    d.mkdir(parents=True, exist_ok=True)
    return subprocess.run([sys.executable, str(CHALLENGER), "--root", str(d / "host"), "--main", str(main_history(state)),
                           "--address-file", str(d / "address.json")], cwd=ROOT, check=False).returncode


def firewall(config: Mapping[str, Any], state: Path, local: bool) -> int:
    """Pin the Lean firewall every prover session runs (module docstring)."""
    exe = Path(tools(state)["flock-firewall"])
    if not os.access(exe, os.X_OK):
        print(f"firewall: {exe} is not an executable", file=sys.stderr)
        return 1
    d = state / "firewall"
    d.mkdir(parents=True, exist_ok=True)
    (d / "ready.json").write_text(json.dumps({"firewall": str(exe), "sha512": hashlib.sha512(exe.read_bytes()).hexdigest(),
                                              "sessions": RN.ONE_AT_A_TIME}, sort_keys=True) + "\n")
    return 0


def workers(config: Mapping[str, Any], state: Path, local: bool) -> int:
    """Serve each sampling block of the toy into ``STATE/blocks`` (module docstring)."""
    w, t = workload(config), tools(state)
    run = os.environ.get("RESEARCH_RUN_ID") or "local"
    a = SimpleNamespace(law=config.get("service", {}).get("law", RN.LAW), cap=w["cap"],
                        requests=",".join(map(str, w["requests"])), steps=w["steps"], pouw=None, weights=None,
                        tamper=set(w["tamper"]), windows=w["blocks"], node=os.uname().nodename, delta=None, tools=t)
    (state / "blocks").mkdir(parents=True, exist_ok=True)
    for i in range(w["blocks"]):
        RN.loop_window(i, run, a, state / "blocks", state / "public" / f"w{i}")
    return 0


def _component(rec: Path) -> tuple[str, dict | None, dict | None]:
    """A component's status in a block's record and its answer and case (None for both when absent)."""
    try:
        ans, case = json.loads((rec / "answer.json").read_text()), json.loads((rec / "case.json").read_text())
    except (OSError, ValueError):
        return "absent", None, None
    return ans.get("verdict", "error"), ans, case


def product(exe: str, listed: Sequence[str], record: Path) -> dict[str, Any]:
    """``Core.Composition.product`` over exactly ``listed`` (component names) on the block's ``record``: each
    component's status, the product's id and its answer."""
    ids = [IDS[c] for c in listed]
    argv = [exe, ",".join(ids)]
    pid = subprocess.run([*argv, "id"], capture_output=True, text=True, check=True).stdout.strip()
    status, cases, digests = {}, {}, {}
    for c in listed:
        status[c], ans, case = _component(record / c)
        cases[IDS[c]] = case or C.NO_CASE
        digests[IDS[c]] = (ans or {}).get("commitment") or ""
    got = C.onsite_check(argv, cases, digests, pid)
    return {"components": status, "product": pid, "verdict": got["verdict"], "reason": got.get("reason")}


def verifier(config: Mapping[str, Any], state: Path, local: bool) -> int:
    """Audit each served block on the challenger's coins, then the product over the listed components (module
    docstring): 0 when every block's product accepts, 1 when one rejects, 2 otherwise."""
    t = tools(state)
    address = state / "challenger" / "address.json"
    if not address.exists():
        print(f"verifier: no challenger is listening ({address}): run --party challenger first", file=sys.stderr)
        return 2
    ch = {"mode": "loopback:" + json.loads(address.read_text())["address"], "main": str(main_history(state))}
    os.environ.setdefault("RAYON_NUM_THREADS", str(os.cpu_count() or 1))
    os.environ.setdefault("RUST_MIN_STACK", str(64 << 20))
    leg = load("sampled-proofs").leg
    run = os.environ.get("RESEARCH_RUN_ID") or "local"
    lines = []
    for block in sorted((state / "blocks").glob("w*/period.json"), key=lambda p: int(p.parent.name[1:])):
        i = int(block.parent.name[1:])
        rec = state / "record" / f"w{i}"
        out = rec / "sampled-proofs"
        out.mkdir(parents=True, exist_ok=True)
        try:
            ans = leg(block.parent, os.uname().nodename, out, challenger=ch)
        except Exception as e:  # noqa: BLE001: a leg that can't decide is the block's error, never a verdict
            ans = {"verdict": "error", "reason": f"{type(e).__name__}: {e}"[:2000]}
        (out / "answer.json").write_text(json.dumps(ans, sort_keys=True))
        if (out / "sampled-proofs.case.json").exists():
            (out / "case.json").write_text((out / "sampled-proofs.case.json").read_text())
        line = {"block": i, "coins": "simulated" if local else "development",
                **product(t["product"], config["protocols"], rec), "run": run}
        lines.append(line)
        print("VERDICT " + json.dumps(line, sort_keys=True), flush=True)
        with open(state / "record" / "verdicts.jsonl", "a") as f:
            f.write(json.dumps(line, sort_keys=True) + "\n")
    if not lines:
        print(f"verifier: no block was served under {state / 'blocks'}: run --party workers first", file=sys.stderr)
        return 2
    verdicts = {x["verdict"] for x in lines}
    return 0 if verdicts == {"accept"} else 1 if "reject" in verdicts else 2


PARTIES: dict[str, dict[str, Role]] = {
    "sampled-proofs": {"challenger": challenger, "firewall": firewall, "workers": workers, "verifier": verifier}}
#: the order step 0 runs them in on one machine (the configuration's ``after``)
ORDER = ("challenger", "firewall", "workers", "verifier")


def stop(state: Path) -> None:
    """Ask the challenger party to stop."""
    host, port = json.loads((state / "challenger" / "address.json").read_text())["address"].rsplit(":", 1)
    with socket.create_connection((host, int(port))) as s, s.makefile("rwb") as f:
        f.write(b'{"op": "stop"}\n')
        f.flush()
        f.readline()


def run_local(config_path: Path, config: Mapping[str, Any], state: Path) -> int:
    """Every role here (``local``): the challenger as a subprocess, read until it listens, then the rest in `ORDER`; the
    challenger stopped once the verifier is done. -> the verifier's code."""
    tools(state)
    main_history(state)
    p = subprocess.Popen([sys.executable, "-m", "verity_all_protocols.parties", str(config_path), "--state", str(state),
                          "--party", "challenger"], stdout=subprocess.PIPE, text=True, cwd=ROOT,
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(Path(__file__).parents[1]),
                                                                                       os.environ.get("PYTHONPATH")]))})
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
            if line.startswith("CHALLENGER "):
                break
        else:
            print(f"challenger: exited {p.wait()} before it listened", file=sys.stderr)
            return 2
        threading.Thread(target=lambda: [print(x, end="", flush=True) for x in p.stdout], daemon=True).start()
        rc = 0
        for role in ORDER[1:]:
            rc = PARTIES["sampled-proofs"][role](config, state, True)
            if rc and role != "verifier":
                print(f"{role}: exited {rc}", file=sys.stderr)
                return rc
        return rc
    finally:
        if p.poll() is None:
            stop(state)
            p.wait()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--state", required=True, help="the roles' shared state directory")
    ap.add_argument("--party", action="append", choices=ORDER, help="a role to run here (repeatable, run in order)")
    ap.add_argument("--local", action="store_true", help="every role on this machine, its coins simulated")
    ap.add_argument("--tools", help="built executables (build.sh's tools.json), instead of building them into STATE")
    ap.add_argument("--tamper", default="", help="blocks whose workers tamper with their outputs, e.g. 0")
    a = ap.parse_args(argv)
    if bool(a.local) == bool(a.party):
        ap.error("give --local (every role here) or --party ROLE")
    config = config_of(Path(a.config))
    if a.tamper:
        config.setdefault("workload", {})["tamper"] = [int(x) for x in a.tamper.split(",") if x.strip()]
    state = Path(a.state).resolve()
    state.mkdir(parents=True, exist_ok=True)
    if a.tools:
        (state / "tools.json").write_text(Path(a.tools).read_text())
    if a.local:
        return run_local(Path(a.config).resolve(), config, state)
    rc = 0
    for role in sorted(a.party, key=ORDER.index):
        rc = PARTIES["sampled-proofs"][role](config, state, False) or rc
    return rc


__all__ = ["IDS", "ORDER", "PARTIES", "challenger", "config_of", "firewall", "main", "product", "run_local", "verifier",
           "workers"]

if __name__ == "__main__":
    raise SystemExit(main())
