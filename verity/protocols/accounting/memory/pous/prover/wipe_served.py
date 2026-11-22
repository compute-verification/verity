"""The wipe's ``attach`` (`pous/scheme.py`'s ``WIPE``): one wipe of this node's memory (`wipe_run.subprotocol`), run whole
before the attach returns. The engine core's process attaches its schemes just before its model runner loads the model
(`scheme_plugin.attach`), so the wipe ends before any workload byte reaches the card.

Its settings (the declaration's ``settings``, which an application states and its hash covers) are the protocol's
parameters alone, what changes what the wipe proves: `period.json`'s ``wipe`` keys ``store``, ``tier``, ``segments``,
``reserve_bytes``, ``limit_ns`` and ``reads`` (`wipe_run.CONFIG`), with CONFIG's None as ``0`` or ``""``.

What says only where and how this node runs it is a launcher's fact (`FACTS`), handed in ``ctx`` by name, each one
there or refused by name, of its default's type: ``device``, ``streamer``, ``threads``, ``links``, ``disk``, ``dram``,
``egress_bits`` (the network attester's count over the session, evidence and not a parameter), ``idle_mib`` (the most
of a ``cuda`` store's card in use that counts as idle), and:

- ``host``, ``address`` and ``command``: the memory attester on a host of its own, all three or none (`period.json`'s
  ``challenger_host`` and ``challenger.wipe``). None is refused unless ``run_keys``;
- the development facts (`DEVELOPMENT`): ``run_keys``, a challenger on this node under a key the run makes and pins
  for itself (`wipe_run.run_key`);
  ``simulated``, a challenger on this node named the stand-in for one on its own host (``placement`` ``"simulated"``);
  ``foreign_bytes`` and ``foreign_past_bound``, the prover a cheat that keeps that much prior content (``"fM"``, one
  label past the bound). A record made under any of them, or under a key the configuration doesn't pin, says why in
  ``served.development`` (`period.development_of`), its piece writes ``development.json`` and no ``answer.json``
  (`period.publish`), and its placement's
  premise never holds (`wipe_party.placement`: a key the run made, or a challenger on the developer's node, reads
  rehearsal), so it never reads as a wipe.

The scheme hands it ``party``, the memory challenger's party name (``MEMORY_ATTESTER``), and ``development_of``
(`period.development_of`, why a run is a development run), so that this prover file loads no verifier file. What else its launcher hands it in ``ctx`` (`LAUNCHER`), beside ``settings``, ``private`` and
the facts:

- ``side_of``: the runtime's `Side.of`, which no protocol imports;
- ``pins``: the application's ``[[pin]]`` entries, read-only. Its ``memory-challenger`` pin for ``wipe`` is the one
  key the verifier trusts, and all of them are the deployment's pins (`period.deployment`);
- ``configuration``: the committed configuration's hash, or absent;
- ``trusted``: a key the run pins for the wipe itself (the launcher's ``WIPE_KEY_SHA512``), trusted in place of the pins'
  key; the pins still decide the premise, so a key they don't give reads rehearsal (`wipe_party.placement`);
- ``node``: the wiped server's id (this host's name by default);
- ``fill``: the fill floor the card file gives for this node's card (its ``store.wipe_fill``, a floor on the fill and
  not the card's memory), `period.json`'s ``wipe.fill_bytes``, which the judge holds the session's fill to; a ``cuda``
  store without it is refused;
- ``judge``: the Lean ``wipe-protocol`` executable. Without it the wipe is undecided (`wipe_period.NO_JUDGE`);
- ``wipe_stream``: the native streamer (``wipe_stream.c``) its launcher built where programs run, `period.json`'s
  ``tools.wipe-stream``, which the memory attester on this node gates and runs in place of building one beside its key
  (whose scratch, under the serve dir, may be mounted noexec); refused beside a bound host, which builds its own;
- ``ready``: called with the judge's path before the judge answers (`wipe_run.subprotocol`'s ``ready``); `built` by
  default, which waits for a judge its launcher still builds beside the wipe (`served_wipe.sh`).

It refuses, by name, before any session: outside its piece's process (``ctx["process"]``), with no side, with no
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
from verity.protocols.accounting.memory.pous.prover import wipe_run, wipe_store

NAME = "wipe"
SMI = "nvidia-smi"
#: the value of each setting or fact that stands for CONFIG's None, which the declaration's and FACTS' defaults read
UNSET = {"segments": 0, "tier": "", "threads": 0, "egress_bits": -1, "reads": 0, "disk": {}, "dram": {}}
#: the launcher's facts, at the values its launcher hands by default: never in an application
FACTS = {k: wipe_run.CONFIG[k] for k in ("device", "links", "foreign_bytes")} | {
    k: UNSET[k] for k in ("threads", "egress_bits", "disk", "dram")} | {
    "streamer": "native", "idle_mib": 1024, "host": "", "address": "", "command": [], "run_keys": False,
    "simulated": False, "foreign_past_bound": False}
#: the facts that make a run a development run, named in its record's ``served.development``
DEVELOPMENT = ("run_keys", "simulated", "foreign_bytes", "foreign_past_bound")
BINDING = ("host", "address", "command")
#: what a launcher hands the wipe in ``ctx``, beside ``settings``, ``private`` and ``process``
LAUNCHER = ("side_of", "pins", "configuration", "trusted", "node", "fill", "judge", "wipe_stream", "ready")


def period_wipe(settings: Mapping[str, Any]) -> dict[str, Any]:
    """`period.json`'s ``wipe`` for ``settings`` (the declaration's), its key pinned by none (the attach adds the pin's)."""
    out = {k: None if k in UNSET and v == UNSET[k] else v for k, v in settings.items()
           if k in wipe_run.CONFIG}
    if settings["foreign_past_bound"]:
        out["foreign_bytes"] = "fM"
    out["placement"] = "simulated" if settings["simulated"] else None
    return out


def check_settings(values: Mapping[str, Any]) -> str | None:
    """`Scheme.check_settings`: what the wipe refuses of its settings beyond their types, at the default facts."""
    return check({**FACTS, **values})


def facts_of(ctx: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The launcher's facts in ``ctx`` (`FACTS`), and why they can't stand, each by name."""
    out, why = {k: ctx[k] for k in FACTS if k in ctx}, []
    if missing := [k for k in FACTS if k not in ctx]:
        why.append(f"no {', '.join(missing)} handed to it: the launcher's facts, each in ctx by name")
    for k, v in out.items():
        if type(v) is not type(FACTS[k]):
            why.append(f"{k} is a {type(FACTS[k]).__name__}, not {v!r}")
    return out, why


def check(values: Mapping[str, Any]) -> str | None:
    """What the wipe refuses of its settings and facts together, beyond their types."""
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


def pin_of(pins: Any, party: str) -> str | None:
    """The memory challenger's (``party``'s) ``key-sha512`` for the wipe among the application's ``[[pin]]`` entries, or
    None."""
    return next((p["key-sha512"] for p in pins if (p.get("party"), p.get("protocol")) == (party, NAME)), None)


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
    facts, out = facts_of(ctx)
    if out:
        return tuple(out)
    s = {**ctx["settings"], **facts}
    if why := check(s):
        out.append(why)
    if ctx.get("process") != H.PIECE:
        out.append(f"it runs in its piece's process, before any engine is on the card, not in {ctx.get('process')!r}")
    if ctx.get("side_of") is None:
        out.append("no challenger's side was handed to it (ctx['side_of'], the runtime's Side.of)")
    if ctx.get("private") is None:
        out.append("no private directory was handed to it, where its case goes")
    if "party" not in ctx:
        out.append("no party was handed to it (ctx['party'], the memory challenger's, which the scheme hands)")
    if s["store"] == "cuda" and not (type(ctx.get("fill")) is int and ctx["fill"] > 0):
        out.append("no fill floor for the card (ctx['fill'], the card file's store.wipe_fill), which the judge "
                   "holds a cuda store's fill to")
    if ctx.get("wipe_stream") and s["host"]:
        out.append("wipe_stream is the streamer of a memory attester on this node; one on its own host (host, address, "
                   "command) builds its own")
    if not s["run_keys"]:
        if ctx.get("pins") is None:
            out.append("no pin was handed to it (ctx['pins'], the application's [[pin]] entries)")
        elif "party" in ctx and pin_of(ctx["pins"], ctx["party"]) is None and not ctx.get("trusted"):
            out.append(f"the pins handed to it give the {ctx['party']} no key for the wipe, so the verifier would trust "
                       f"none: state the party's [[pin]] in the application's configuration file, with party = "
                       f"\"{ctx['party']}\", protocol = \"{NAME}\", scheme = \"lms-sha512/n64-w1-h20/v0\" and key-sha512 = the "
                       f"SHA-512 of its public key in 128 lowercase hex digits")
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
    ``served`` beside it: the wiped server, the card (a cuda store's UUID), ``key_pin``, ``configuration`` where the
    trusted key is the one the pins give the wipe, else ``run``, and ``development``, why the run is a development run
    (the facts of `DEVELOPMENT` it ran under, a key no pin gives: `period.development_of`), None where it isn't."""

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
    development_of = ctx["development_of"]
    s = {**ctx["settings"], **facts_of(ctx)[0]}
    card = card_of(s["device"]) if s["store"] == "cuda" else None
    if card and (why := busy(card, s["idle_mib"])):
        raise ValueError(f"{NAME}: {why}")
    pins = [dict(p) for p in ctx.get("pins") or ()]
    node = ctx.get("node") or socket.gethostname()
    private = Path(ctx["private"])
    private.chmod(0o700)
    with tempfile.TemporaryDirectory(prefix="wipe-") as t:
        period = Path(t) / "periods" / "p0"
        period.mkdir(parents=True)
        (device := Path(t) / "periods" / "wipe-challenger").mkdir(mode=0o700)
        key = wipe_run.run_key(device) if s["run_keys"] else ctx.get("trusted") or pin_of(pins, ctx["party"])
        doc: dict[str, Any] = {
            "tools": {tool: str(ctx[k]) for tool, k in (("wipe-protocol", "judge"), ("wipe-stream", "wipe_stream"))
                      if ctx.get(k)},
            NAME: period_wipe(s) | {"trusted_sha512": key, "fill_bytes": ctx.get("fill")},
            "deployment": {"configuration": ctx.get("configuration"), "pins": pins}}
        if s["host"]:
            doc |= {"challenger_host": s["host"], "challenger": {NAME: {"command": list(s["command"]),
                                                                         "address": s["address"]}}}
        (period / "period.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
        (private / "period.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
        answer = wipe_run.subprotocol(period, node, private,
                                      side_of=ctx["side_of"], ready=ctx.get("ready") or built)
    (private / "wipe.case.json").chmod(0o600)
    pin = pin_of(pins, ctx["party"])
    served = {"server": node, "fill": ctx.get("fill"), "card": card, "key_pin": "configuration" if key and key == pin else "run",
              "development": development_of(key, {pin} - {None}, [k for k in DEVELOPMENT if s[k]])}
    record = json.loads((private / "wipe.json").read_text()) | {"served": served}
    (private / "wipe.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    return Wiped(answer, record)


__all__ = ["BINDING", "DEVELOPMENT", "FACTS", "LAUNCHER", "NAME", "SMI", "UNSET", "Wiped", "attach", "built", "busy", "card_of", "check", "check_settings", "facts_of",
           "period_wipe", "pin_of", "refusals"]
