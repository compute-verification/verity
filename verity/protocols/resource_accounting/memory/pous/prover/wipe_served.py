"""The wipe's ``attach`` (`pous/scheme.py`'s ``WIPE``): one wipe of this node's memory (`wipe_run.subprotocol`), run whole
before the attach returns. The engine core's process attaches its schemes just before its model runner loads the model
(`scheme_plugin.attach`), so the wipe ends before any workload byte reaches the card.

Its settings (the declaration's ``settings``) are `period.json`'s ``wipe`` (`wipe_run.CONFIG`), each typed so that a configuration's
``[params."wipe"]`` can say it, with CONFIG's None as ``0``, ``-1``, ``""`` or ``{}`` (`period_wipe`), and:

- ``foreign_past_bound``: the cheat one label past the bound (CONFIG's ``foreign_bytes`` ``"fM"``);
- ``simulated``: a challenger on this node named the stand-in for one on its own host (``placement`` ``"simulated"``);
- ``host``, ``address`` and ``command``: the memory attester on a host of its own, all three or none (`period.json`'s
  ``challenger_host`` and ``challenger.wipe``). None is refused unless ``run_keys``, a challenger on this node under a
  key the run makes, trusted for the run: a rehearsal;
- ``idle_mib``: the most of a ``cuda`` store's card in use that counts as idle.

What its launcher hands it in ``ctx`` (`LAUNCHER`), beside ``settings`` and ``private``:

- ``side_of``: the runtime's `Side.of`, which no protocol imports;
- ``pins``: the application's ``[[pin]]`` entries, read-only. Its ``memory-challenger`` pin for ``wipe`` is the one
  key the verifier trusts, and all of them are the deployment's pins (`period.deployment`);
- ``configuration``: the committed configuration's hash, or absent;
- ``trusted``: a key the run pins for the wipe itself (the launcher's ``WIPE_KEY_SHA512``), trusted in place of the pins'
  key; the pins still decide the premise, so a key they don't give reads rehearsal (`wipe_party.placement`);
- ``node``: the wiped server's id (this host's name by default);
- ``judge``: the Lean ``wipe-protocol`` executable. Without it the Python's accept decides nothing (`wipe_period.NO_JUDGE`);
- ``ready``: called with the judge's path before the judge answers (`wipe_run.subprotocol`'s ``ready``); `built` by
  default, which waits for a judge its launcher still builds beside the wipe (`served_wipe.sh`).

It refuses, by name, before any session: outside the engine's process (``ctx["process"]``), with no side, with no
private directory, with no pin handed to it, with no challenger binding, and on a busy card. The case and the record
go to ``ctx["private"]`` (``wipe.case.json``, ``wipe.json``); the report returned (`Wiped`) gives the record and its
summary. The challenger's run key and session log are in a scratch directory removed before it returns."""

from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity import hooks as H
from verity.protocols.resource_accounting.memory.pous.prover import wipe_run, wipe_store
from verity.protocols.resource_accounting.memory.pous.verifier.wipe_parties import CHALLENGER

NAME = "wipe"
SMI = "nvidia-smi"
#: the settings that stand for CONFIG's None, by the value that does
UNSET = {"segments": 0, "tier": "", "threads": 0, "egress_bits": -1, "reads": 0, "disk": {}, "dram": {}}
BINDING = ("host", "address", "command")
#: what a launcher hands the wipe in ``ctx``, beside ``settings``, ``private`` and ``process``
LAUNCHER = ("side_of", "pins", "configuration", "trusted", "node", "judge", "ready")


def period_wipe(settings: Mapping[str, Any]) -> dict[str, Any]:
    """`period.json`'s ``wipe`` for ``settings`` (the declaration's), its key pinned by none (the attach adds the pin's)."""
    out = {k: None if k in UNSET and v == UNSET[k] else v for k, v in settings.items()
           if k in wipe_run.CONFIG}
    if settings["foreign_past_bound"]:
        out["foreign_bytes"] = "fM"
    out["placement"] = "simulated" if settings["simulated"] else None
    return out


def check_settings(values: Mapping[str, Any]) -> str | None:
    """`Scheme.check_settings`: what the wipe refuses of its settings beyond their types."""
    bound = [k for k in BINDING if values[k]]
    if bound and len(bound) != len(BINDING):
        return f"host, address and command bind the challenger together, not {bound} alone"
    if bound and (values["run_keys"] or values["simulated"]):
        return "run_keys and simulated run the challenger on this node, not on the bound host"
    if not all(type(a) is str for a in values["command"]):
        return f"command is the challenger's argv, strings, not {values['command']!r}"
    if values["foreign_past_bound"] and values["foreign_bytes"]:
        return "foreign_past_bound is the cheat one label past the bound, with foreign_bytes 0"
    if values["idle_mib"] < 0:
        return f"idle_mib is a count of MiB, not {values['idle_mib']}"
    try:
        wipe_run.config({NAME: period_wipe(values)})
    except ValueError as e:
        return str(e)
    return None


def pin_of(pins: Any) -> str | None:
    """The memory challenger's ``key-sha512`` for the wipe among the application's ``[[pin]]`` entries, or None."""
    return next((p["key-sha512"] for p in pins if (p.get("party"), p.get("protocol")) == (CHALLENGER, NAME)), None)


def busy(device: str, idle_mib: int) -> str | None:
    """Why card ``device`` (an index or a UUID, as `nvidia-smi -i` takes it) isn't idle: another process on it, or more
    than ``idle_mib`` MiB of its memory in use; None when it is."""
    def query(*args: str) -> list[str]:
        r = subprocess.run([SMI, *args, "--format=csv,noheader,nounits", "-i", device], capture_output=True,
                           text=True)
        if r.returncode:
            raise ValueError(f"wipe: nvidia-smi can't read card {device}: {(r.stderr or r.stdout).strip()[:300]}")
        return [line.strip() for line in r.stdout.splitlines() if line.strip()]

    others = [p for p in query("--query-compute-apps=pid") if p != str(os.getpid())]
    if others:
        return f"the card {device} isn't idle: process {', '.join(others)} on it, and the wipe kills nothing"
    used = int(query("--query-gpu=memory.used")[0])
    if used > idle_mib:
        return f"the card {device} isn't idle: {used} MiB of its memory in use, past idle_mib {idle_mib}"
    return None


def built(exe: str) -> None:
    """Returns once the judge ``exe`` is built: at once with no lock beside it (``EXE.lock``), else once the build that
    holds that lock has ended and left ``EXE.ok``. A build that failed raises its own error (``failed`` beside it),
    as `benchmarks/pous/lean_ready.sh` does for the shell."""
    lock = Path(f"{exe}.lock")
    if not lock.exists():
        return
    with lock.open() as f:
        fcntl.flock(f, fcntl.LOCK_SH)
    if not Path(f"{exe}.ok").exists():
        raise RuntimeError((lock.parent / "failed").read_text().strip())


def card_of(ordinal: int) -> str:
    """The UUID of the card the wipe fills, CUDA's ``ordinal`` of those this process sees; `nvidia-smi -i` reads
    indices unmapped, so the card is named by UUID."""
    card = wipe_store.Cuda(ordinal)
    try:
        return card.info()["uuid"]
    finally:
        card.close()


def refusals(ctx: Mapping[str, Any]) -> tuple[str, ...]:
    """Why the wipe can't run with ``ctx``, each by name; empty when it can."""
    s, out = ctx["settings"], []
    if ctx.get("process") != H.ENGINE:
        out.append(f"it runs in the engine's process, before the model loads, not in {ctx.get('process')!r}")
    if ctx.get("side_of") is None:
        out.append("no challenger's side was handed to it (ctx['side_of'], the runtime's Side.of)")
    if ctx.get("private") is None:
        out.append("no private directory was handed to it, where its case goes")
    if not s["run_keys"]:
        if ctx.get("pins") is None:
            out.append("no pin was handed to it (ctx['pins'], the application's [[pin]] entries)")
        elif pin_of(ctx["pins"]) is None and not ctx.get("trusted"):
            out.append(f"the pins handed to it give the {CHALLENGER} no key for the wipe")
        if not s["host"]:
            out.append("no challenger binding (host, address, command) and run_keys unset")
        if (t := ctx.get("trusted")) and not re.fullmatch(r"[0-9a-f]{128}", t):
            out.append(f"the trusted key is a SHA-512 in 128 lowercase hex digits, not {t:.16}")
        key = next((os.path.expanduser(k) for i, k in zip(s["command"], s["command"][1:]) if i == "-i"), None)
        if key and not os.access(key, os.R_OK):
            out.append(f"no wipe gate key {key} on {ctx.get('node') or socket.gethostname()} ({socket.gethostname()}): "
                       f"infra installs it, and its gate verb on {s['host']} for this card")
    return tuple(out)


@dataclass(frozen=True)
class Wiped:
    """The wipe's report: ``answer``, the verifier's, and ``wipe``, the record `wipe_run.subprotocol` wrote with
    ``served`` beside it: the wiped server, the card (a cuda store's UUID) and ``key_pin``, ``configuration`` where the
    trusted key is the one the pins give the wipe, else ``run``."""

    answer: Mapping[str, Any]
    wipe: Mapping[str, Any]

    def record(self) -> Mapping[str, Any]:
        return self.wipe

    def summary(self) -> dict[str, Any]:
        return wipe_run.summary(self.wipe)


def attach(exchange: Any, ctx: Mapping[str, Any]) -> Wiped:
    """One wipe of this node as ``ctx["settings"]`` say, with what its launcher handed in ``ctx`` (`LAUNCHER`)."""
    if why := refusals(ctx):
        raise ValueError(f"{NAME}: {'; '.join(why)}")
    s = ctx["settings"]
    card = card_of(s["device"]) if s["store"] == "cuda" else None
    if card and (why := busy(card, s["idle_mib"])):
        raise ValueError(f"{NAME}: {why}")
    pins = [dict(p) for p in ctx.get("pins") or ()]
    key = None if s["run_keys"] else ctx.get("trusted") or pin_of(pins)
    node = ctx.get("node") or socket.gethostname()
    doc: dict[str, Any] = {
        "tools": {} if ctx.get("judge") is None else {"wipe-protocol": str(ctx["judge"])},
        NAME: period_wipe(s) | {"trusted_sha512": key},
        "deployment": {"configuration": ctx.get("configuration"), "pins": pins}}
    if s["host"]:
        doc |= {"challenger_host": s["host"], "challenger": {NAME: {"command": list(s["command"]),
                                                                     "address": s["address"]}}}
    private = Path(ctx["private"])
    private.chmod(0o700)
    with tempfile.TemporaryDirectory(prefix="wipe-") as t:
        period = Path(t) / "periods" / "p0"
        period.mkdir(parents=True)
        (Path(t) / "periods" / "wipe-challenger").mkdir(mode=0o700)
        (period / "period.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
        (private / "period.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
        answer = wipe_run.subprotocol(period, node, private,
                                      side_of=ctx["side_of"], ready=ctx.get("ready") or built)
    (private / "wipe.case.json").chmod(0o600)
    served = {"server": node, "card": card, "key_pin": "configuration" if key and key == pin_of(pins) else "run"}
    record = json.loads((private / "wipe.json").read_text()) | {"served": served}
    (private / "wipe.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    return Wiped(answer, record)


__all__ = ["BINDING", "LAUNCHER", "NAME", "SMI", "UNSET", "Wiped", "attach", "built", "busy", "card_of", "check_settings",
           "period_wipe", "pin_of", "refusals"]
