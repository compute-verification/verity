"""The verifier's process: the remote bandwidth audit v0's `Session` (verity/core/protocols/remote_bandwidth/audit.py)
driven over the wire against a prover on another process or machine, with a secret of its own.

One connection serves one registration and then `--sessions` sessions over it, each of `--windows` windows of
`--window-ms`:

1. registration: the prover's hello names its node, N, payload root and, if it has one, its store's layout (each
   region's kind and tier, `regions_of`, the memory line's labels), and whether it defers an appended block's
   residency to its encode (`deferred`, `audit.Params.deferred`: refused with `bye` under `--lifetime` 1, since a
   block appended in window w is resident from w or w + 1 and its re-key falls at age L − 1); the verifier draws the
   codec seed from its
   secret (`audit.register`) and sends the registration; the prover encodes its store and answers `ready` with its
   calibration (pass times, the gap, its measured read rate);
2. the round trip, measured by pings;
3. each session: `Params` with the allowance A = `timing.deadline` at a high percentile of the calibration's passes
   and gaps and of the measured round trip (`--quantile`), `--jitter-ms`, and the lifetime cap L on an appended block
   (`--lifetime`, windows, sent with the session so the prover re-keys in time); the coin commitment sent first, then
   σ_0, a boundary message every window, each append answered with its coin (`Verifier.append`), each root fed to
   `Session.receive` stamped with the time its last byte was read (with the releases its message carries first, and
   the withdrawn blocks among them, its plan as a change when it carries one, and the step its pass covers:
   `Verifier.root`), each window closed
   (`Session.close`) as soon as its roots are final and its draws sent; the prover's openings
   checked as their frames arrive, by `--check-workers` processes (`checks.py`, the driver's own check of an opening,
   verdict as `Session.verify`'s) or by the same check on a thread of the verifier's, so a window's checks run while
   the next window's passes arrive; once every window is accepted the record (`Session.record`), its γ terms
   (`audit.ledger`) and the node's HBM it shows held (`memory`: the credited share and the residual, at ε, held at
   every pass and held on average), and with `--host-bytes` the same by kind against the node's HBM and host DRAM
   (`node_line`).

A node has one live registration (`audit.Registry`), this process's own or kept across processes in `--registry`: a
hello for a node with a live registration is answered `bye` with the reason and draws nothing. The process closes its
registration after its last session, counting each session (a failed one by its failure); one left live by a
process that ended early stays live, refusing the node, until `--close NODE` closes it with its sessions as they stood,
the one in flight unended.

The verdicts are the verifier of record's, the Lean process `remote-audit-verifier` (`lean_verifier.py`), under
`--verdict lean` or `both` (the default, `VERDICT`): it holds the registry and every session, and each registration,
session, event and window goes to it and to `audit.Session` at one reading of the clock (`Pair`). `audit.Session`
keeps the driver's books (its coins, its plans, the record the ledger reads), and an event, a window's verdict or a
record on which the two differ refuses the session at that window as `disagreement`. Under `both` `audit.Session`
checks every window's openings too, beside the Lean, and a window's two verdicts are set on the two sessions at once;
under `lean` it checks none and takes the Lean's verdict. Under `python` there is no Lean process and `audit.Session`
decides. `--close` closes a registration in both registries alike.

`--transcript DIR` writes each session's difftest case to DIR as the session runs (`lean_verifier.Transcript`): every
event as the Lean took it, with the clock it read, the session's coins, and each window's openings where its verdict
was set, which `replay.py transcript` (verity/core/protocols/remote_bandwidth/lean/tests) replays through both
verifiers, rejections included. A message the driver couldn't read is in no event: the case ends before it. A run
keeps DIR in scratch and preserves what it keeps of it (the writing-runs skill): a 60 s window's openings are about
0.6 GB of JSON.

Openings are checked in the clear: zero-knowledge proof units are not in this harness (PROTOCOL.md §8).

    python verifier.py --port 7781 --out OUT [--windows 7 --window-ms 1000 --samples 2773 --sessions 1]
                       [--registry REGISTRY.json] [--verdict both|lean|python] [--lean-exe PATH]
                       [--transcript DIR]
    python verifier.py --registry REGISTRY.json --close NODE_HEX [--verdict both|lean|python]
    python verifier.py --memory-of VERIFIER_JSON [--host-bytes N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import resource
import secrets
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth import audit, timing  # noqa: E402

import lean_verifier as lv  # noqa: E402
import wire  # noqa: E402
from checks import Checks, check  # noqa: E402

VERDICT = "both"
"""Who decides the verdicts (module docstring): `lean`, `both` or `python`."""
VERDICTS = ("lean", "both", "python")
CHECK_BATCH = 1024
"""Openings a worker checks at once (`checks.Checks`), and a Lean thread (`checkWindowChunked`)."""
GPU_HBM_BYTES = 96 * 2 ** 30
"""One RTX PRO 6000's memory, 96 GiB (`benchmarks/census/hardware.json`)."""
HBM_BYTES = timing.GPUS * GPU_HBM_BYTES
KINDS = ("weights", "kv", "filler", "activations")
TIERS = ("hbm", "host")


def timing_parts(*, capacity_bytes_per_s: float, span_ns: int, passes: int, reads_bytes: int, tiers: dict,
                 pass_ns: int) -> dict:
    """The ledger's timing term, `1 − reads/(R·span)`, split per pass into parts that sum to it exactly, each a share
    of R·span.  `tiers` is the calibration's bare read of each tier (`read_tiers`: its bytes and milliseconds, the
    prover's own measurement), and `pass_ns` the median spacing of the session's root arrivals (the verifier's clock).
    With s the span a pass, G every GPU's stated read and H the host's:

    - `above_stated_tiers`: R beyond G + H (at spec R, the hardware's gap);
    - `host_dram_idle`: H·s less the host tier's bytes;
    - `hbm_below_stated`: G over the bare HBM read's time less the HBM tier's bytes (a slower read, or idle GPUs);
    - `receipts_beyond_read`: G over the median pass less the bare HBM read;
    - `boundaries_and_allowance`: G over s less the median pass (window boundaries and A);
    - `blocks_left_out`: the store's bytes less a pass's mean reads.

    A split for display, not a bound: the prover's bare read moves share only between `hbm_below_stated` and
    `receipts_beyond_read`."""
    s = span_ns / passes / 1e9
    whole = capacity_bytes_per_s * s
    gpus, host = timing.GPUS * timing.GPU_READ_BYTES_PER_S, timing.HOST_READ_BYTES_PER_S
    hbm_bytes, host_bytes = tiers["hbm"]["bytes"], tiers.get("host", {}).get("bytes", 0)
    hbm_s, pass_s = tiers["hbm"]["ms"] / 1e3, pass_ns / 1e9
    return {"above_stated_tiers": (capacity_bytes_per_s - gpus - host) / capacity_bytes_per_s,
            "host_dram_idle": (host * s - host_bytes) / whole,
            "hbm_below_stated": (gpus * hbm_s - hbm_bytes) / whole,
            "receipts_beyond_read": gpus * (pass_s - hbm_s) / whole,
            "boundaries_and_allowance": gpus * (s - pass_s) / whole,
            "blocks_left_out": (hbm_bytes + host_bytes - reads_bytes / passes) / whole}


def regions_of(layout, blocks: int) -> tuple[tuple[str, str, int, int], ...] | None:
    """The hello's layout of the registered store as regions `(kind, tier, first, blocks)` in index order, covering its
    `blocks` exactly (`verity_vllm.store.registry.Store.layout`'s), or None without one.  The labels are the prover's,
    public as protocol structure: that a plan's weights blocks hold the circuit's weights is the plan unit's to show
    (`audit.RequirementLink`), not this."""
    if layout is None:
        return None
    if not isinstance(layout, list):
        raise ValueError(f"a layout is a list of regions, not {layout!r}")
    out, at = [], 0
    for r in layout:
        kind, tier, first, n = (r.get(k) for k in ("kind", "tier", "first", "blocks")) if isinstance(r, dict) else [0] * 4
        if (kind not in KINDS or tier not in TIERS or type(first) is not int or first != at or type(n) is not int
                or n < 1):
            raise ValueError(f"a layout is regions of a kind in {KINDS} and a tier in {TIERS}, each from the end of "
                             f"the one before: not {r!r}")
        out.append((kind, tier, first, n))
        at += n
    if at != blocks:
        raise ValueError(f"the layout covers {at} blocks, and the store is {blocks}")
    return tuple(out)


WORST_PASS, AVERAGE = "held at every pass", "held on average"
"""The memory line's two forms (`memory`): what every pass of a window held, and what its passes held on average."""


def averaged(held: dict, kv: float, *, wrong_blocks: float, rebuild_blocks: float, passes: int, chip_bytes: int,
             block_bytes: int) -> dict:
    """One window's memory line held on average over its passes, G's reading: the blocks in node memory when each pass
    read them, averaged over the window's passes, not what memory held between reads.  The audit doesn't pace passes
    inside a window (a root can be made early and sent late), so a prover may read in a burst and free the memory
    until the next window's reads, and this is not memory held over time.  `held` the mean bytes of each kind
    (weights, hbm, host) a pass of the window read, `kv` the mean KV bytes a pass credits, and `wrong_blocks` =
    f_t·U_t/P_t, the receipts the pooled event leaves wrong spread over the passes.  A block missing over the window
    gives a wrong receipt in every pass that reads it, so m missing blocks make m·P_t wrong receipts ≤ f_t·U_t, and
    m ≤ f_t·N.  Each kind is credited less that, the window's rebuilds (`rebuild_blocks`, every block the node's
    squarings could re-encode in its span) and the node's on-chip storage (`chip_bytes`).  `credited_rebuild_per_pass`
    is the same with the rebuilds spread over the passes too, as G charges them (X_r·τ a pass): a correct receipt not
    read from memory or chip was read from its own rebuild (a rebuilt block kept for a later pass is in memory then),
    so the passes' shortfalls sum to at most the window's rebuilds.  Both take the side link as carrying no block
    (Identity); G charges it, L_r·τ a pass."""
    b = block_bytes
    chip = chip_bytes / b
    basis = held["hbm"] + kv

    def credit(lost: float) -> dict:
        return {"hbm": max(0.0, basis - lost * b), "weights": max(0.0, held["weights"] - lost * b),
                "kv": max(0.0, kv - lost * b)}

    lost, per_pass = wrong_blocks + rebuild_blocks + chip, wrong_blocks + rebuild_blocks / passes + chip
    return {"form": AVERAGE, "held": held, "kv": kv, "wrong_blocks": wrong_blocks, "rebuild_blocks": rebuild_blocks,
            "chip_blocks": chip, "lost_blocks": lost, "basis": basis,
            "epsilon": min(1.0, lost * b / basis) if basis > 0 else 1.0, "credited": credit(lost),
            "lost_blocks_rebuild_per_pass": per_pass, "credited_rebuild_per_pass": credit(per_pass)}


def average_line(windows: list[dict]) -> dict:
    """The session's average line: its worst window's by the HBM credited (`averaged`'s per window)."""
    worst = min(windows, key=lambda x: x["credited"]["hbm"])
    return {"form": AVERAGE, "credited_bytes": worst["credited"]["hbm"], "epsilon": worst["epsilon"],
            "window": worst["window"], "weights_bytes": min(x["credited"]["weights"] for x in windows),
            "kv_bytes": min(x["credited"]["kv"] for x in windows),
            "weights_bytes_rebuild_per_pass": min(x["credited_rebuild_per_pass"]["weights"] for x in windows)}


def memory(record: audit.Record, plans, regions, *, hbm_bytes: int, rebuild: audit.Rebuild, chip_bytes: int = 0,
           credit_post_seed: audit.PostSeedVerdict | None = None) -> dict:
    """The node's memory a session's record shows held, against its HBM, in two forms.

    Held at every pass (`WORST_PASS`, the worst pass's): each window's passes each read the distinct blocks of their
    plan (`plans`, a list of each window's, as `Session` received them) inside the window's span, so at every pass the
    node held them, but for the receipts the pooled event leaves wrong (`audit.floors`, f_t of the window's U_t, every
    window's at once except with the record's δ) and the blocks its GPUs could re-encode from their payloads in the
    span instead (`rebuild`, its squarings at `codec.encode_squarings` a block).  Those lost blocks could all fall in
    one pass, so they come off every pass, and the held bytes a window credits are its least pass's, less them all.

    Held on average (`AVERAGE`, each window's `average` and the session's `average`, `averaged`): the mean of the
    window's passes, less f_t·U_t/P_t, the rebuilds and `chip_bytes` of the node's on-chip storage.

    A registered block counts as the layout labels it (`regions_of`; None, the drivers' stores: the whole store
    filler in HBM), HBM only toward the share; an appended block
    (index N and up) is KV the session wrote, in HBM, and is credited only on the property's verdict, less the most
    its carriers hold (`credit_post_seed`, as `audit.ledger`'s μ_carrier).  Weights and KV are each credited on their
    own: the lost blocks come off each.  The session's line is its worst window's: `credited_bytes`, `share` of
    `hbm_bytes` and `residual` = 1 − share, the HBM it doesn't show used, with ε, the share of the window's held
    basis the losses took, at δ.  ε is this reading's (the floor and rebuilds against held blocks), not a composed
    guarantee's."""
    if credit_post_seed is not None and credit_post_seed.holds is not True:
        raise ValueError("a verdict that the property fails credits nothing after the seed: pass None")
    if len(plans) != len(record.windows) or any(len(p) != w.passes for p, w in zip(plans, record.windows)):
        raise ValueError("a plan for each pass of each window of the record")
    if rebuild.codec != record.codec:
        raise ValueError("the rebuild model is of another codec than the record's")
    b, labels = record.block_bytes, regions is not None
    regions = regions if labels else (("filler", "hbm", 0, record.blocks),)
    per_ns = rebuild.squarings_per_s / record.codec.encode_squarings / 1e9
    windows = []
    for w, ps, f in zip(record.windows, plans, audit.floors(record)):
        carrier = credit_post_seed.carrier_bytes(w.receipts * b) if credit_post_seed is not None else 0.0
        passes = []
        for plan, post in zip(ps, w.post_seed):
            by = {"weights": 0, "hbm": 0, "host": 0}
            for kind, tier, lo, n in regions:
                o = sum(max(0, min(f0 + c, lo + n) - max(f0, lo)) for f0, c in plan.runs)
                by[tier if tier == "host" else "hbm"] += o
                by["weights"] += o if kind == "weights" and tier == "hbm" else 0
            kv = post * b - min(carrier, post * b) if credit_post_seed is not None else 0.0
            passes.append((by, post, kv))
        lost = f * w.receipts + per_ns * w.span_ns
        held = {"weights": min(p[0]["weights"] for p in passes) * b, "kv": min(p[1] for p in passes) * b,
                "hbm": min(p[0]["hbm"] for p in passes) * b, "host": min(p[0]["host"] for p in passes) * b}
        basis = min(p[0]["hbm"] * b + p[2] for p in passes)
        mean = {k: sum(p[0][k] for p in passes) * b / w.passes for k in ("weights", "hbm", "host")}
        windows.append({
            "window": w.window, "passes": w.passes, "span_ns": w.span_ns, "floor": f, "lost_blocks": lost,
            "held": held, "read": {"weights": sum(p[0]["weights"] for p in passes) * b, "kv": sum(w.post_seed) * b},
            "basis": basis, "epsilon": min(1.0, lost * b / basis) if basis > 0 else 1.0,
            "credited": {"hbm": max(0.0, basis - lost * b), "weights": max(0.0, held["weights"] - lost * b),
                         "kv": max(0.0, min(p[2] for p in passes) - lost * b),
                         "kv_read": max(0.0, sum(p[2] for p in passes) - lost * b)},
            "average": {"window": w.window, **averaged(
                mean, sum(p[2] for p in passes) / w.passes, wrong_blocks=f * w.receipts / w.passes,
                rebuild_blocks=per_ns * w.span_ns, passes=w.passes, chip_bytes=chip_bytes, block_bytes=b)}})
    return summed(windows, hbm_bytes=hbm_bytes, block_bytes=b, delta=record.windows[0].profile.delta, labels=labels,
                  kv_credited=credit_post_seed is not None, chip_bytes=chip_bytes)


def summed(windows: list[dict], *, hbm_bytes: int, block_bytes: int, delta: float, labels: bool, kv_credited: bool,
           chip_bytes: int) -> dict:
    """`memory`'s session line from its windows: the worst window's in each form, against `hbm_bytes`."""
    worst = min(windows, key=lambda x: x["credited"]["hbm"])
    credited = worst["credited"]["hbm"]
    average = average_line([x["average"] for x in windows])
    return {"form": WORST_PASS, "hbm_bytes": hbm_bytes, "block_bytes": block_bytes, "delta": delta,
            "labels": labels, "kv_credited": kv_credited, "chip_bytes": chip_bytes,
            "credited_bytes": credited, "share": credited / hbm_bytes, "residual": 1 - credited / hbm_bytes,
            "epsilon": worst["epsilon"], "window": worst["window"],
            "weights_bytes": min(x["credited"]["weights"] for x in windows),
            "kv_bytes": min(x["credited"]["kv"] for x in windows),
            "kv_read_bytes": sum(x["credited"]["kv_read"] for x in windows),
            "read": {k: sum(x["read"][k] for x in windows) for k in ("weights", "kv")},
            "average": average | {"share": average["credited_bytes"] / hbm_bytes,
                                  "residual": 1 - average["credited_bytes"] / hbm_bytes},
            "windows": windows}


def recorded_memory(session: dict, *, chip_bytes: int = timing.CHIP_BYTES, host_bytes: int = 0) -> dict:
    """An accepted session's memory line in both forms, from its record in `verifier.json` (`Verifier.session`'s),
    for one written before the average form (`averaged`): each window's mean held per pass is its read weights over its
    passes, and its registered receipts over its passes less its least host tier for HBM, exact while every pass reads
    the same host blocks; KV, credited on no verdict, is none.  The rebuilds are the recorded lost blocks less f_t·U_t.
    With `host_bytes`, the node line too (`node_line`)."""
    mem, pub = session["ledger"]["memory"], session["record"]["public"]
    if mem["kv_credited"]:
        raise ValueError("a record that credits KV keeps each pass's KV, which a recorded window doesn't")
    b, windows = mem["block_bytes"], []
    for x, receipts, post in zip(mem["windows"], pub["receipts"], pub["post_seed"]):
        p, wrong = x["passes"], x["floor"] * receipts
        hbm = (receipts - post) * b / p - x["held"]["host"]
        mean = {"weights": x["read"]["weights"] / p, "hbm": hbm, "host": x["held"]["host"]}
        windows.append(x | {"average": {"window": x["window"], **averaged(
            mean, 0.0, wrong_blocks=wrong / p, rebuild_blocks=x["lost_blocks"] - wrong, passes=p,
            chip_bytes=chip_bytes, block_bytes=b)}})
    out = summed(windows, hbm_bytes=mem["hbm_bytes"], block_bytes=b, delta=mem["delta"], labels=mem["labels"],
                 kv_credited=False, chip_bytes=chip_bytes)
    if (error := memory_error(session["ledger"])) is not None:
        out["error"] = error
    if host_bytes:
        out["node"] = node_line(out, host_bytes=host_bytes)
    return out


def memory_error(ledgers: dict) -> float | None:
    """The memory line's failure probability from `Verifier.ledgers`' output: its floors are the pooled event's, so the
    horizon's error (δ once, and AuditRO's 2^-slack a pass), plus the sketch's (`RateAuditSketch`); `delta` alone
    understates it by the passes' slack.  None when no reading has a ledger."""
    for name in ("spec", "measured"):
        if "error" in ledgers.get(name, {}):
            return ledgers[name]["error"]["horizon"] + 2.0 ** ledgers.get("sketch_failure", {}).get("log2", -math.inf)
    return None


def node_line(mem: dict, *, host_bytes: int) -> dict:
    """`memory`'s line against the node's whole memory, its HBM and `host_bytes` of host DRAM, by kind: each window's
    held weights (HBM), the rest of its held HBM (filler) and its held host tier, each less the window's lost blocks
    (as `memory` takes them off each kind), the KV it credits, and in all its held HBM and host less the lost blocks
    once, plus that KV. The line is the worst window's by that total: each kind's bytes and share of the node, and the
    residual, the share it doesn't show used. `held` is the same before the losses, which the line shows beside.
    `average` is the same line held on average (each window's `average`, `averaged`), and `average_rebuild_per_pass`
    that line with the rebuilds spread over the passes."""
    node, b = mem["hbm_bytes"] + host_bytes, mem["block_bytes"]

    def line(windows: list[dict], lost_key: str = "lost_blocks") -> dict:
        rows = []
        for x in windows:
            h, lost = x["held"], x[lost_key] * b
            held = {"weights": h["weights"], "filler_hbm": h["hbm"] - h["weights"], "host": h["host"]}
            kinds = {k: max(0.0, v - lost) for k, v in held.items()} | {"kv": x["credited"]["kv"]}
            rows.append((max(0.0, h["hbm"] + h["host"] - lost) + kinds["kv"], kinds, held, x))
        total, kinds, held, x = min(rows, key=lambda r: r[0])
        return {"window": x["window"], "lost_bytes": x[lost_key] * b, "credited_bytes": total,
                "credited": kinds, "share": {k: v / node for k, v in kinds.items()}, "residual": 1 - total / node,
                "held": held, "held_share": {k: v / node for k, v in held.items()}}

    return {"form": WORST_PASS, "node_bytes": node, "hbm_bytes": mem["hbm_bytes"], "host_bytes": host_bytes,
            **line(mem["windows"]), "kv_read_bytes": mem["read"]["kv"],
            "average": {"form": AVERAGE, **line([x["average"] for x in mem["windows"]])},
            "average_rebuild_per_pass": {"form": AVERAGE, **line([x["average"] for x in mem["windows"]],
                                                                  "lost_blocks_rebuild_per_pass")}}


class RealClock:
    def now(self) -> int:
        return time.monotonic_ns()

    def wait(self, inbox: queue.Queue, until_ns: int | None):
        """The next message, or None once `until_ns` has passed with none."""
        if until_ns is None:
            return inbox.get()
        left = until_ns - self.now()
        if left <= 0:
            return None
        try:
            return inbox.get(timeout=left / 1e9)
        except queue.Empty:
            return None

    def idle(self) -> None:
        """The prover has said its passes are done (informational on a real clock)."""

    def busy(self) -> None:
        """A session's passes have started."""


def start_lean(source: bytes, registry: audit.Registry, exe: Path | None, done) -> lv.Lean:
    """The Lean verifier's process, with the verifier's secret and `registry` (which it must read back as
    `audit.Registry` writes it), `done` called with each window's check as it ends."""
    lean = lv.Lean(exe or lv.build(), done)
    read = lean.call({"op": "init", "source": source.hex(), "registry": registry.to_json(), "chunk": CHECK_BATCH})
    if read["registry"] != registry.to_json():
        lean.close()
        raise ValueError(f"the Lean verifier reads the registry as {read['registry']}, not as audit.Registry does: "
                         f"{registry.to_json()}")
    return lean


class Pair:
    """Session k through the verifiers, with `audit.Session`'s interface: each event goes to `s`, the driver's books,
    and, given `lean`, to the Lean process's session k, at one reading of the verifier's clock. The result is the one
    they agree on; where they differ, the session is refused at the event's window as `disagreement` (in both
    verifiers), returned as that rejection, or raised as `audit.SessionError` where `audit.Session` returns no verdict.
    Everything else is read from `s`."""

    def __init__(self, v: Verifier, s: audit.Session, k: int, tape: lv.Tape, case: lv.Transcript | None = None) -> None:
        self.v, self.s, self.k, self.lean, self.case = v, s, k, v.lean, case
        self.node, self.disagreement = s.registration.node.hex(), None
        self.frames: dict[int, list[bytes]] = {}
        if self.lean is not None:
            got = self._lean({"op": "session", "node": self.node, "params": lv.params_json(s.params),
                              "coins": [c.hex() for c in tape.coins], "append_coins": [],
                              "not_before_ns": s.not_before_ns, "index": k})
            if got != {"session": k}:
                self._disagree(0, "session", got, {"session": k})

    def __getattr__(self, name: str):
        return getattr(self.s, name)

    def _lean(self, request: dict) -> dict:
        try:
            return self.lean.call(request)
        except lv.LeanError as e:
            return {"error": str(e)}

    def _disagree(self, t: int, event: str, lean: dict, python: dict) -> audit.Verdict:
        self.disagreement = {"t": t, "event": event, "lean": lean, "python": python}
        v = self.s._fail(t, "disagreement")
        if self.lean is not None:
            self._lean({"op": "abort", "node": self.node, "session": self.k, "outcome": "disagreement"})
        self.v.log(f"verifier: session {self.k} refused at window {t}: the verifiers disagree on {event}: "
                   f"Lean {lean}, Python {python}")
        return v

    def _step(self, t: int, event, python, as_json, *, clocked: bool = False, coin=None):
        """`python()` on the books and `event()` through the Lean: `python()`'s return, or the disagreement's
        rejection (None as the second value)."""
        if self.disagreement is not None:
            raise audit.SessionError("the session ended: the verifiers disagree")
        ev = event() if self.lean is not None or self.case is not None else None
        held = self.v._stamp
        now = self.v._stamp = self.v._now()
        if clocked and ev is not None:
            ev["now"] = now
        if self.case is not None:
            self.case.event(ev)
        try:
            try:
                out = python()
                got = as_json(out)
            except audit.SessionError:
                if self.lean is None:
                    raise
                out, got = None, lv.REFUSED
        finally:
            self.v._stamp = held
        if self.lean is None:
            return out, True
        c = coin(out) if coin is not None else None
        lean = self._lean({"op": "event", "node": self.node, "session": self.k, "event": ev,
                           "coin": None if c is None else c.hex()})
        if lean != got:
            return self._disagree(t if type(t) is int and t >= 0 else self.s.filling, ev["kind"], lean, got), False
        if got == lv.REFUSED:
            raise audit.SessionError("refused by both verifiers")
        return out, True

    def start(self):
        out, agreed = self._step(0, lambda: {"kind": "start"}, self.s.start, lv.released_json, clocked=True)
        if not agreed:
            raise audit.SessionError("the verifiers disagree on the session's start")
        return out

    def boundary(self):
        t = next((t for t, w in enumerate(self.s.windows) if w.boundary_ns is None), len(self.s.windows))
        out, agreed = self._step(t, lambda: {"kind": "boundary"}, self.s.boundary, lv.released_json, clocked=True)
        if not agreed:
            raise audit.SessionError(f"the verifiers disagree on window {t}'s boundary")
        return out

    def append(self, root: bytes, count: int):
        return self._step(self.s.filling, lambda: {"kind": "append", "root": root.hex(), "count": count},
                          lambda: self.s.append(root, count), lv.appended_json,
                          coin=lambda out: out[1] if isinstance(out, tuple) else None)[0]

    def release(self, runs, withdrawn=()):
        return self._step(self.s.filling, lambda: lv.release_event(runs, withdrawn),
                          lambda: self.s.release(runs, withdrawn), lv.done_json)[0]

    def receive(self, t: int, i: int, root: bytes, runs=None, *, change=None, step=None):
        if runs is not None:
            raise ValueError("the driver sends a plan as a change")
        return self._step(t, lambda: lv.receive_event(t, i, root, change, step),
                          lambda: self.s.receive(t, i, root, change=change, step=step), lv.done_json, clocked=True)[0]

    def close(self, t: int):
        return self._step(t, lambda: {"kind": "close", "t": t}, lambda: self.s.close(t), lv.drew_json,
                          clocked=True)[0]

    def openings(self, t: int, body: bytes) -> None:
        """One frame of window t's openings to the Lean, kept there until the window is verified."""
        try:
            self.lean.call({"op": "openings", "node": self.node, "session": self.k, "t": t, "bytes": len(body)}, body)
        except lv.LeanError as e:
            self.v.log(f"verifier: the Lean refused window {t}'s openings frame: {e}")

    def keep(self, t: int, body: bytes) -> None:
        """One frame of window t's openings, kept for the transcript until the window's verdict is set."""
        if self.case is not None:
            self.frames.setdefault(t, []).append(body)

    def transcribe(self, t: int) -> None:
        """Window t's openings into the transcript, as a `verify` event where its verdict was set."""
        if self.case is not None:
            self.case.verify(t, self.frames.pop(t, []))

    def verify(self, t: int, check: int) -> bool:
        """Window t's frames checked on the Lean's threads as check `check`: False where it refuses to."""
        got = self._lean({"op": "verify", "node": self.node, "session": self.k, "t": t, "id": check})
        return got == {"spawned": True}

    def settle(self, check: int) -> dict:
        """Check `check`'s verdict applied to the Lean's session: its result, the openings it read and its ns."""
        return self._lean({"op": "settle", "id": check})

    def judge(self, t: int, lean: dict, python: dict | None) -> dict:
        """Window t's verdict, `{"accepted", "failure", "at"}`, from the Lean's settled result and the books' verdict
        (`python`, as `verdicts` holds it): under `lean` the Lean's verdict set on the books (`Checks.settle`), under
        `both` the books' own check's; where the two differ, `disagreement`."""
        mine = {k: x for k, x in lean.items() if k not in ("openings", "ns")}
        if python is None:
            try:
                n = len(self.s.windows[t].draws)
                v = Checks.settle(self.s, t, [{"failure": mine.get("failure"),
                                               "at": tuple(mine["at"]) if mine.get("at") else None}], n)
                theirs = lv.verdict_json(v)
            except audit.SessionError:
                theirs = lv.REFUSED
        elif python["failure"] is not None and python["failure"].startswith("refused"):
            theirs = lv.REFUSED
        else:
            theirs = lv.verdict_json(audit.Verdict(t, python["accepted"], python["failure"],
                                                   tuple(python["at"]) if python["at"] else None))
        if self.disagreement is None and mine != theirs:
            self._disagree(t, "verify", mine, theirs)
        if self.disagreement is not None:
            return {"accepted": False, "failure": "disagreement", "at": None}
        return {"accepted": "accepted" in mine, "failure": mine.get("failure", "refused" if mine == lv.REFUSED else None),
                "at": mine.get("at")}

    def end(self, failure: dict | None) -> None:
        """The session ended with the driver's `failure` where neither verifier gives it one (a message it couldn't
        read): set on both. And an accepted session's record held to the Lean's: one that differs is `disagreement`."""
        if self.lean is None:
            if failure is not None and self.s.outcome is None:
                self._fail_books(failure)
            return
        got = self._lean({"op": "outcome", "node": self.node, "session": self.k})
        if failure is not None:
            if self.s.outcome is None:
                self._fail_books(failure)
            if got.get("outcome") is None and self.disagreement is None:
                self._lean({"op": "abort", "node": self.node, "session": self.k, "outcome": str(failure["failure"])})
            return
        mine = {"outcome": self.s.outcome, "record": lv.summary_json(self.s.record()) if self.s.outcome == "accepted"
                else None}
        if got != mine and self.disagreement is None:
            self._disagree(len(self.s.windows) - 1, "record", got, mine)

    def _fail_books(self, failure: dict) -> None:
        t = failure["t"]
        self.s._fail(t if type(t) is int and 0 <= t < len(self.s.windows) else len(self.s.windows),
                     str(failure["failure"]))


class Verifier:
    def __init__(self, conn: wire.Conn, *, source: bytes, windows: int, window_ns: int, samples: int, delta: float,
                 jitter_ns: int, clock=None, log=print, checks=None, lifetime: int = audit.LIFETIME,
                 registry: audit.Registry | None = None, hbm_bytes: int = HBM_BYTES, host_bytes: int = 0,
                 verdict: str = VERDICT, lean_exe: Path | None = None, transcript: Path | None = None) -> None:
        if verdict not in VERDICTS:
            raise ValueError(f"the verdict is one of {VERDICTS}")
        self.conn, self.source, self.W, self.T, self.lifetime = conn, source, windows, window_ns, lifetime
        self.transcript = transcript
        self.hbm_bytes, self.host_bytes, self.regions, self.deferred = hbm_bytes, host_bytes, None, False
        self.verdict, self.checks = verdict, checks
        self.registry = audit.Registry() if registry is None else registry
        self.samples, self.delta, self.jitter_ns = samples, delta, jitter_ns
        self.clock = clock or RealClock()
        self.log = log
        self.inbox: queue.Queue = queue.Queue()
        self.jobs: queue.Queue = queue.Queue()
        self._stamp: int | None = None
        self._check = 0
        self.lean = None if verdict == "python" else start_lean(source, self.registry, lean_exe, self._lean_done)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._verify, daemon=True).start()

    def _now(self) -> int:
        return self._stamp if self._stamp is not None else self.clock.now()

    def _lean_done(self, msg: dict) -> None:
        self.inbox.put(({"type": "lean-done", "id": msg["done"], "accepted": msg["accepted"]}, b"", self.clock.now()))

    def registry_json(self) -> dict:
        """The verifier's registry, `registry/v0`: the Lean's, but under `python` or once its process has ended (when
        every session it was in is refused as `disagreement`)."""
        if self.lean is not None:
            try:
                return self.lean.call({"op": "registry"})
            except lv.LeanError as e:
                self.log(f"verifier: the Lean's registry is gone ({e}): keeping audit.Registry's")
        return self.registry.to_json()

    def close_registration(self, node: bytes) -> dict:
        """The node's live registration closed in both registries (`audit.RegistryError` where it has none), as
        the registry of record counts it; the books' count beside it where the two differ (`python`)."""
        try:
            mine = self.registry.close(node).to_json()
        except audit.RegistryError as e:
            mine, why = lv.REFUSED, e
        if self.lean is None:
            if mine == lv.REFUSED:
                raise why
            return mine
        try:
            got = self.lean.call({"op": "close", "node": node.hex()})
        except lv.LeanError as e:
            got = {"error": str(e)}
        if got == lv.REFUSED or "error" in got:
            raise audit.RegistryError(f"node {node.hex()} has no live registration to close in the Lean's registry"
                                      + ("" if mine == lv.REFUSED else f" (audit.Registry closed it: {mine})"))
        return got if got == mine else {**got, "python": mine}

    def stop(self) -> None:
        if self.lean is not None:
            self.lean.close()

    def _read(self) -> None:
        try:
            while True:
                self.inbox.put(self.conn.recv())
        except (wire.Closed, OSError) as e:
            self.inbox.put(({"type": "lost", "why": str(e)}, b"", self.clock.now()))

    def _verify(self) -> None:
        while True:
            k, snap, t, draws, bodies = self.jobs.get()
            c0 = time.thread_time_ns()
            r = check(snap, t, draws, bodies)
            self.inbox.put(({"type": "checks", "k": k, "t": t, "results": [r], "count": r["n"],
                             "cpu_ns": time.thread_time_ns() - c0}, b"", self.clock.now()))
            self.jobs.task_done()

    def expect(self, kind: str):
        while True:
            head, body, stamp = self.inbox.get()
            if head["type"] == kind:
                return head, body, stamp
            if head["type"] == "lost":
                raise wire.Closed(head["why"])
            if head["type"] not in ("root", "append", "openings", "closed", "checks", "checked", "lean-done"):
                self.log(f"verifier: skipped a {head['type']} while waiting for {kind}")

    def register(self) -> tuple[audit.Registration, dict]:
        """The hello's registration, drawn through the registry: refused, with `bye` and the reason, for a node with a
        live registration (`audit.RegistryError`, raised), with a malformed layout (`regions_of`'s ValueError), or
        deferring residency under a lifetime of one window (`audit.Params`'s ValueError). Both registries draw it
        under one nonce: one on which they differ is refused as `disagreement` (`audit.RegistryError`), and closed
        where it went live."""
        hello, _, _ = self.expect("hello")
        try:
            self.regions = regions_of(hello.get("layout"), hello["blocks"])
            self.deferred = audit.Params(1, 1, 1, lifetime=self.lifetime, deferred=hello.get("deferred", False)).deferred
        except ValueError as e:
            self.conn.send({"type": "bye", "why": f"refused: {e}"})
            raise
        node, blocks, root = bytes.fromhex(hello["node"]), hello["blocks"], bytes.fromhex(hello["payload_root"])
        c, receipt, rows = wire.codec_named(hello["codec"]), hello["receipt"], hello.get("rows")
        nonce = secrets.token_bytes(audit.NONCE_BYTES)
        try:
            self.registration = self.registry.register(self.source, node, blocks, root, c, receipt, rows=rows,
                                                       nonce=nonce)
            mine, why = {"codec_seed": self.registration.codec_seed.hex(), "digest": self.registration.digest.hex()}, None
        except (audit.RegistryError, ValueError) as e:
            mine, why = (lv.REFUSED if isinstance(e, audit.RegistryError) else {"error": True}), e
        if self.lean is not None:
            try:
                got = self.lean.call(lv.register_request(node, blocks, root, c, receipt, rows, nonce))
            except (lv.LeanError, ValueError):
                got = {"error": True}
            if got != mine:
                if "digest" in mine:
                    self.registry.close(node)
                if "digest" in got:
                    self.lean.call({"op": "close", "node": node.hex()})
                mine, why = got, audit.RegistryError(f"disagreement: the Lean verifier's registration is {got}, "
                                                     f"audit.Registry's {mine}")
        if why is not None:
            self.conn.send({"type": "bye", "why": f"refused: {why}"})
            raise why
        self.registered_ns = self.clock.now()
        self.conn.send({"type": "registration", "registration": wire.registration_json(self.registration)})
        ready, _, _ = self.expect("ready")
        self.calibration = ready["calibration"]
        return self.registration, self.calibration

    def ping(self, n: int) -> dict:
        rtts = []
        for k in range(n):
            t0 = self.clock.now()
            self.conn.send({"type": "ping", "k": k})
            _, _, stamp = self.expect("pong")
            rtts.append(stamp - t0)
        rtts.sort()
        at = lambda q: rtts[min(n - 1, int(q * n))]
        return {"n": n, "min_ns": rtts[0], "median_ns": rtts[n // 2], "max_ns": rtts[-1], "p50": at(0.5),
                "p99": at(0.99), "p999": at(0.999), "max": rtts[-1]}

    def allowance(self, rtt: dict, quantile: str = "p999") -> int:
        """`timing.deadline` at the calibration's pass, gap and round trip at `quantile` (p50, p99, p999 or max),
        plus the jitter allowed each way."""
        cal = self.calibration
        pass_ns = cal["pass_ns"][quantile] if "pass_ns" in cal else cal["pass_ns_max"]
        gap_ns = cal["gap_ns"][quantile] if "gap_ns" in cal else cal["gap_ns_max"]
        r = rtt[quantile]
        return timing.deadline(int(pass_ns), gap_ns=int(gap_ns), down_ns=r // 2 + 1, up_ns=r - r // 2 + 1,
                               jitter_ns=self.jitter_ns)

    def session(self, k: int, allowance_ns: int, not_before_ns: int | None) -> dict:
        W, reg = self.W, self.registration
        params = audit.Params(reg.blocks, allowance_ns, W, self.samples, self.delta, window_ns=self.T, codec=reg.codec,
                              receipt=reg.receipt, rows=reg.rows, lifetime=self.lifetime, deferred=self.deferred)
        not_before_ns = self.registered_ns if not_before_ns is None else not_before_ns
        tape = lv.Tape(params.coins, os.urandom)
        books = self.registry.session(params, reg, None, clock=self._now, not_before_ns=not_before_ns, index=k,
                                      entropy=tape)
        case = None if self.transcript is None else lv.Transcript(
            self.transcript / f"{reg.digest.hex()[:16]}-session-{k}.json", reg, params, tape.coins, not_before_ns)
        session = Pair(self, books, k, tape, case)
        self.conn.send({"type": "session", "k": k, "windows": W, "lifetime": params.lifetime,
                        "commitment": session.commitment.to_json()})
        self.expect("armed")
        while self.clock.now() < not_before_ns:
            item = self.clock.wait(self.inbox, not_before_ns)
            if item is not None:
                self.log(f"verifier: skipped a {item[0]['type']} before the session's first coin")
        sent = drawn = 0
        arrivals: list[list[int]] = [[] for _ in range(W)]
        verdicts: dict[int, dict] = {}
        checked: dict[int, dict] = {}
        lean_ready: dict[int, tuple[int | dict, int]] = {}
        lean_checks: dict[int, int] = {}
        frames: dict[int, int] = {}
        draws_at: dict[int, int] = {}
        parts: dict[int, list] = {}
        checking: dict[int, dict] = {}
        failure = None
        cpu0 = time.process_time_ns()

        def failed(t: int) -> dict:
            v = session.failed
            if v is None:
                return {"t": t, "failure": "refused", "at": None}
            return {"t": v.window, "failure": v.failure, "at": v.at}

        def decide(t: int) -> None:
            """Window t's verdict once each verifier it waits on has checked the window: the Python's result
            (`checked`) set on the books and the Lean's (`lean_ready`) settled on its session at one moment, so
            neither session moves on a verdict the other hasn't had (`Pair.judge`). Nothing once the session has a
            failure."""
            nonlocal failure
            py, ln = checked.get(t), lean_ready.get(t)
            if (failure is not None or t in verdicts or (self.verdict != "lean" and py is None)
                    or (self.lean is not None and ln is None)):
                return
            mine = None
            if py is not None:
                try:
                    v = Checks.settle(books, t, py["results"], py["count"])
                except audit.SessionError as e:
                    v = audit.Verdict(t, False, f"refused: {e}")
                mine = {"accepted": v.accepted, "failure": v.failure, "at": v.at}
            if self.lean is None:
                verdicts[t] = {"t": t, **mine, "cpu_ns": py["cpu_ns"], "openings": py["count"],
                               "checked_ns": py["checked_ns"]}
            else:
                check_id, latency = lean_ready.pop(t)
                res = session.settle(check_id) if type(check_id) is int else check_id
                verdicts[t] = {"t": t, **session.judge(t, res, mine), "cpu_ns": py["cpu_ns"] if py else 0,
                               "openings": py["count"] if py else res.get("openings"),
                               "checked_ns": max(latency, py["checked_ns"]) if py else latency,
                               "python_checked_ns": py["checked_ns"] if py else None,
                               "lean_ns": res.get("ns"), "lean_checked_ns": latency}
            session.transcribe(t)
            if not verdicts[t]["accepted"]:
                failure = {"t": t, "failure": verdicts[t]["failure"], "at": verdicts[t]["at"]}

        def lean_done(head: dict, stamp: int) -> None:
            """A Lean check that ended, ready to settle (`decide`); a process that ended leaves every check running
            an error."""
            ids = [head["id"]] if head["id"] in lean_checks else list(lean_checks) if head["id"] is None else []
            for check_id in ids:
                t = lean_checks.pop(check_id)
                lean_ready[t] = (check_id if head["id"] is not None else {"error": "the process ended"},
                                 stamp - draws_at[t])
                decide(t)

        def close(t: int) -> None:
            nonlocal failure
            try:
                out = session.close(t)
            except audit.SessionError:
                failure = failed(t)
                return
            if isinstance(out, audit.Verdict):
                failure = {"t": t, "failure": out.failure, "at": None}
                return
            draws_at[t] = self.clock.now()
            self.conn.send({"type": "draws", "t": t, "draws": [list(d) for d in out]})

        try:
            self.conn.send({"type": "coin", "coin": wire.coin_json(session.start())})
            self.clock.busy()
        except audit.SessionError:
            failure = failed(0)
        while failure is None and len(verdicts) < W:
            if sent < W:
                until = session.windows[sent].released_ns + self.T
            elif drawn < W:
                until = session.deadline(W - 1) + 1
            else:
                until = None
            item = self.clock.wait(self.inbox, until)
            if item is None:
                if sent < W:
                    try:
                        coin = session.boundary()
                    except audit.SessionError:
                        failure = failed(sent)
                        break
                    sent += 1
                    self.conn.send({"type": "coin", "coin": wire.coin_json(coin)} if coin else {"type": "close"})
                else:
                    close(drawn)
                    drawn += 1
                continue
            head, body, stamp = item
            kind = head["type"]
            if kind == "root":
                t, i = head["t"], head["i"]
                self._stamp = stamp
                try:
                    v = self.root(session, head)
                except audit.SessionError:
                    failure = failed(t)
                    break
                except (TypeError, ValueError, KeyError):
                    failure = {"t": t, "failure": "malformed", "at": None}
                    break
                finally:
                    self._stamp = None
                if v is not None:
                    failure = {"t": v.window, "failure": v.failure, "at": v.at}
                    break
                arrivals[t].append(stamp)
                while drawn < t and failure is None:
                    close(drawn)
                    drawn += 1
            elif kind == "append":
                try:
                    got = self.append(session, head)
                except audit.SessionError:
                    failure = failed(session.filling)
                    break
                except (TypeError, ValueError, KeyError):
                    failure = {"t": session.filling, "failure": "malformed", "at": None}
                    break
                if isinstance(got, audit.Verdict):
                    failure = {"t": got.window, "failure": got.failure, "at": got.at}
                    break
                self.conn.send(got)
            elif kind == "openings":
                t = head["t"]
                if not (type(t) is int and 0 <= t < len(session.windows)) or session.windows[t].draws is None:
                    failure = {"t": t, "failure": "malformed", "at": None}
                    break
                session.keep(t, body)
                if self.lean is not None:
                    session.openings(t, body)
                    frames[t] = frames.get(t, 0) + 1
                    if frames[t] == head["parts"]:
                        self._check += 1
                        if session.verify(t, self._check):
                            lean_checks[self._check] = t
                        else:
                            lean_ready[t] = (lv.REFUSED, stamp - draws_at[t])
                            decide(t)
                if self.verdict == "lean":
                    continue
                if self.checks is None:
                    have = parts.setdefault(t, [])
                    have.append(body)
                    if len(have) == head["parts"]:
                        self.jobs.put((k, Checks.snapshot(books, t), t, list(session.windows[t].draws), have))
                    continue
                c = checking.get(t)
                if c is None:
                    c = checking[t] = {"snap": self.checks.snapshot(books, t), "at": 0, "bodies": [], "n": 0,
                                       "parts": 0, "futures": []}
                c["bodies"].append(body)
                c["n"] += max(0, int(head.get("n", 0)))
                c["parts"] += 1
                if c["n"] >= CHECK_BATCH or c["parts"] == head["parts"]:
                    draws = list(session.windows[t].draws[c["at"]:c["at"] + c["n"]])
                    f = self.checks.submit(c["snap"], t, draws, c["bodies"])
                    f.add_done_callback(lambda _, t=t: self.inbox.put(({"type": "checked", "k": k, "t": t}, b"",
                                                                       self.clock.now())))
                    c["futures"].append(f)
                    c["at"], c["bodies"], c["n"] = c["at"] + c["n"], [], 0
                c["whole"] = c["parts"] == head["parts"]
            elif kind == "checked" and head["k"] == k:
                c = checking[head["t"]]
                if c.get("whole") and all(f.done() for f in c["futures"]) and head["t"] not in checked:
                    results = [f.result() for f in c["futures"]]
                    checked[head["t"]] = {"results": results, "count": sum(r["n"] for r in results),
                                          "cpu_ns": sum(r["cpu_ns"] for r in results),
                                          "checked_ns": stamp - draws_at[head["t"]]}
                    decide(head["t"])
            elif kind == "checks" and head["k"] == k:
                checked[head["t"]] = {"results": head["results"], "count": head["count"], "cpu_ns": head["cpu_ns"],
                                      "checked_ns": stamp - draws_at[head["t"]]}
                decide(head["t"])
            elif kind == "lean-done":
                lean_done(head, stamp)
            elif kind == "closed":
                self.clock.idle()
            elif kind == "lost":
                raise wire.Closed(head["why"])
        self.jobs.join()
        lost = None
        while lean_checks:
            item = self.inbox.get()
            head, _, stamp = item
            if head["type"] == "lean-done":
                lean_done(head, stamp)
            elif head["type"] == "lost":
                lost = item
        if lost is not None:
            self.inbox.put(lost)
        session.end(failure)
        for check_id, _ in lean_ready.values():
            if type(check_id) is int:
                session.settle(check_id)
        if failure is None and session.disagreement is not None:
            failure = {"t": session.disagreement["t"], "failure": "disagreement", "at": None}
        accepted = failure is None and len(verdicts) == W and all(v["accepted"] for v in verdicts.values())
        self.conn.send({"type": "end", "k": k, "accepted": accepted, "failure": failure})
        while True:
            head, _, _ = self.expect("ended")
            if head["k"] == k:
                break
        out = {"k": k, "accepted": accepted, "failure": failure, "allowance_ns": allowance_ns,
               "samples": self.samples, "delta": self.delta, "window_ns": self.T, "windows": W,
               "registration": reg.digest.hex(), "coins": session.commitment.root.hex(),
               "appended": session.extent - reg.blocks, "released": session.released,
               "verdict": self.verdict, "disagreement": session.disagreement,
               "transcript": None if case is None else case.close(tape.appends),
               "verify_cpu_ns": {t: v["cpu_ns"] for t, v in sorted(verdicts.items())},
               "verify_latency_ns": {t: v.get("checked_ns") for t, v in sorted(verdicts.items())},
               "lean_check_ns": {t: v.get("lean_ns") for t, v in sorted(verdicts.items())},
               "lean_latency_ns": {t: v.get("lean_checked_ns") for t, v in sorted(verdicts.items())},
               "python_latency_ns": {t: v.get("python_checked_ns", v.get("checked_ns"))
                                     for t, v in sorted(verdicts.items())},
               "check_workers": self.checks.workers if self.checks else 0,
               "session_process_cpu_ns": time.process_time_ns() - cpu0,
               "draws": {t: [list(d) for d in w.draws] for t, w in enumerate(session.windows) if w.draws},
               "windows_seen": [{"t": t, "released_ns": w.released_ns, "boundary_ns": w.boundary_ns,
                                 "deadline_ns": session.deadline(t) if w.boundary_ns is not None else None,
                                 "passes": len(w.roots), "arrivals_ns": arrivals[t],
                                 "slack_ns": (session.deadline(t) - arrivals[t][-1])
                                 if arrivals[t] and w.boundary_ns is not None else None,
                                 "draws_sent_ns": draws_at.get(t)} for t, w in enumerate(session.windows)]}
        if accepted:
            record = session.record()
            out["record"] = {"public": record.public(), "profile": record.profile(), "end_ns": record.end_ns}
            out["end_ns"] = record.end_ns
            out["ledger"] = self.ledgers(record, arrivals, [w.plans for w in session.windows])
        return out

    @staticmethod
    def append(session: audit.Session, head: dict) -> dict | audit.Verdict:
        """An append's message to the session (`Session.append`), `{"type": "append", "root": hex, "count": n}`: the
        reply `{"type": "appended", "first": j, "coin": hex}`, its first index and its coin `c_a`, drawn now, or the
        rejection."""
        got = session.append(bytes.fromhex(head["root"]), head["count"])
        if isinstance(got, audit.Verdict):
            return got
        first, coin = got
        return {"type": "appended", "first": first, "coin": coin.hex()}

    @staticmethod
    def root(session: audit.Session, head: dict) -> audit.Verdict | None:
        """A root's message to the session in the order the prover sends it: the blocks it releases, with those among
        them it withdrew (`Session.release`), then the root with its plan, a change from the last plan if it carries
        one and the registered store otherwise, and its step (`Session.receive`).  The first rejection, if any.
        Appends come as messages of their own (`append`), each answered with its coin: a root's message carrying any
        is malformed."""
        if "appends" in head:
            raise ValueError("an append is a message of its own, answered with its coin")
        withdrawn = [tuple(r) for r in head.get("withdrawn", ())]
        if ((head.get("released") or withdrawn)
                and (v := session.release([tuple(r) for r in head.get("released", ())], withdrawn)) is not None):
            return v
        change = head.get("change")
        if change is not None:
            added, removed = change
            change = ([tuple(r) for r in added], [tuple(r) for r in removed])
        return session.receive(head["t"], head["i"], bytes.fromhex(head["root"]), change=change, step=head.get("step"))

    def ledgers(self, record: audit.Record, arrivals: list[list[int]] = (), plans=None) -> dict:
        """γ against the node's R at spec and measured (`timing.SPEC_BYTES_PER_S`, `timing.READ_BYTES_PER_S`: every
        GPU's HBM and the host's DRAM), with the node's S_chip and squaring ceiling, whatever share of the node the
        store is on.  Both R are the verifier's, which the prover's calibration can only raise: a prover reporting a
        slower read would shrink every term but the timing (MEM-24).  R cancels in the certified bytes,
        `(1 − γ)·R·Σ`, the same in both readings: they are what a session over part of the node establishes, whose γ
        is the node's only once its store spans the node.  Each reading carries its timing term's parts
        (`timing_parts`) when the calibration has a bare read of each tier covering the registered store and a window
        had two roots.  With the session's plans, `memory` is the node's HBM it shows held (`memory`), KV credited
        on no verdict yet.  Each window's bound on wrong receipts is computed once for both readings, exactly, in the
        check workers when there are some (`checks.Checks.bounds`)."""
        cal = self.calibration
        reported = float(cal["read_bytes_per_s"])
        measured = max(timing.READ_BYTES_PER_S, reported)
        spec = max(timing.SPEC_BYTES_PER_S, measured)
        rebuild = audit.Rebuild(record.codec, timing.SQUARINGS_PER_S[record.codec.field.name])
        node = {"gpus": timing.GPUS, "gpu_spec_bytes_per_s": timing.GPU_SPEC_BYTES_PER_S,
                "gpu_read_bytes_per_s": timing.GPU_READ_BYTES_PER_S, "host_spec_bytes_per_s": timing.HOST_SPEC_BYTES_PER_S,
                "host_read_bytes_per_s": timing.HOST_READ_BYTES_PER_S, "gpu_chip_bytes": timing.GPU_CHIP_BYTES,
                "host_chip_bytes": timing.HOST_CHIP_BYTES}
        out = {"inputs": {"spec_bytes_per_s": spec, "measured_read_bytes_per_s": measured,
                          "reported_read_bytes_per_s": reported, "node": node,
                          "chip_bytes": timing.CHIP_BYTES, "encode_bytes_per_s": cal["encode_bytes_per_s"],
                          "squarings_per_s": rebuild.squarings_per_s, "rows": record.rows,
                          "span_ns": record.span_ns, "receipts": record.receipts, "passes": record.passes,
                          "reads_bytes": record.receipts * record.block_bytes,
                          "post_seed_receipts": sum(sum(w.post_seed) for w in record.windows),
                          "post_seed_credited": False}}
        tiers = cal.get("read_tiers")
        spacings = sorted(b - a for arr in arrivals for a, b in zip(arr, arr[1:]))
        parts = None
        if (tiers and "hbm" in tiers and spacings
                and sum(t["bytes"] for t in tiers.values()) == self.registration.blocks * record.block_bytes):
            parts = {"span_ns": record.span_ns, "passes": record.passes,
                     "reads_bytes": record.receipts * record.block_bytes,
                     "tiers": {k: {"bytes": t["bytes"], "ms": t["ms"]} for k, t in tiers.items()},
                     "pass_ns": spacings[len(spacings) // 2]}
            out["inputs"]["timing_parts"] = {"pass_ns": parts["pass_ns"], "tiers": parts["tiers"]}
        profiles = [w.profile for w in record.windows]
        bounds = self.checks.bounds(profiles) if self.checks else [p.bound() for p in profiles]
        by = {id(p): x for p, x in zip(profiles, bounds)}
        for name, r in (("spec", spec), ("measured", measured)):
            try:
                lg = audit.ledger(record, capacity_bytes_per_s=r, chip_bytes=timing.CHIP_BYTES, rebuild=rebuild,
                                  wrong=lambda p: by[id(p)])
                whole = r * record.span_ns / 1e9
                out[name] = {"capacity_bytes_per_s": r, **lg.to_json(), "certified_bytes": (1 - lg.gamma) * whole,
                             "certified_bytes_horizon": (1 - lg.gamma_horizon) * whole}
                if parts:
                    out[name]["timing_parts"] = timing_parts(capacity_bytes_per_s=r, **parts)
            except ValueError as e:
                out[name] = {"capacity_bytes_per_s": r, "refused": str(e)}
        # RateAuditSketch's term: every compression the node's GPUs could make over the span a guess or a key.
        compressions = math.ceil(timing.SHA512_PER_S * record.span_ns / 1e9)
        out["sketch_failure"] = {"compressions": compressions,
                                 "log2": timing.log2(timing.sketch_failure(compressions, record.passes, record.rows))}
        if plans is not None:
            out["memory"] = memory(record, plans, self.regions, hbm_bytes=self.hbm_bytes, rebuild=rebuild,
                                   chip_bytes=timing.CHIP_BYTES)
            if (error := memory_error(out)) is not None:
                out["memory"]["error"] = error
            if self.host_bytes:
                out["memory"]["node"] = node_line(out["memory"], host_bytes=self.host_bytes)
        return out


def load_registry(path: Path | None) -> audit.Registry:
    """The registrations kept at `path`, none if it is None or holds none yet."""
    if path is None or not path.exists():
        return audit.Registry()
    return audit.Registry.from_json(json.loads(path.read_text()))


def save_registry(path: Path | None, registry: audit.Registry | dict) -> None:
    """`registry`, or its `registry/v0` JSON (`Verifier.registry_json`), kept at `path`."""
    if path is not None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(registry if isinstance(registry, dict) else registry.to_json(), indent=1))
        tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7781)
    ap.add_argument("--out", type=Path, help="required unless --close")
    ap.add_argument("--windows", type=int, default=7)
    ap.add_argument("--window-ms", type=float, default=1000.0)
    ap.add_argument("--samples", type=int, default=2773)
    ap.add_argument("--delta-log2", type=float, default=-40.0)
    ap.add_argument("--jitter-ms", type=float, default=2.0)
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--lifetime", type=int, default=audit.LIFETIME,
                    help="L: windows an appended block may stay in plans after the window it joined")
    ap.add_argument("--pings", type=int, default=1000)
    ap.add_argument("--check-workers", type=int, default=len(os.sched_getaffinity(0)) - 1,
                    help="processes checking openings (0: one thread of the verifier's own)")
    ap.add_argument("--quantile", default="p999", choices=("p50", "p99", "p999", "max"),
                    help="the calibration's pass and gap and the round trip the allowance takes")
    ap.add_argument("--registry", type=Path,
                    help="the verifier's registrations, kept across its processes (`audit.Registry`): a node with a "
                         "live registration there is refused (default: this process's own)")
    ap.add_argument("--hbm-bytes", type=int, default=HBM_BYTES,
                    help="the node's HBM, the memory line's capacity (default: its GPUs' at 96 GiB each)")
    ap.add_argument("--host-bytes", type=int, default=0,
                    help="the node's host DRAM: with it the memory line is also by kind against HBM and DRAM "
                         "(`node_line`)")
    ap.add_argument("--close", metavar="NODE_HEX",
                    help="close the node's live registration in --registry, its sessions counted as they stood, "
                         "and exit")
    ap.add_argument("--verdict", default=VERDICT, choices=VERDICTS,
                    help="who decides: the Lean verifier (lean), the Lean with audit.Session checking beside it and "
                         "a mismatch refused as `disagreement` (both), or audit.Session alone (python)")
    ap.add_argument("--lean-exe", type=Path,
                    help="a built remote-audit-verifier (default: `lake build` it in its package)")
    ap.add_argument("--transcript", type=Path, metavar="DIR",
                    help="write each session's difftest case to DIR (`lean_verifier.Transcript`: about 0.6 GB a "
                         "window at 33,273 openings), its path, bytes and sha256 in verifier.json")
    ap.add_argument("--memory-of", type=Path, metavar="VERIFIER_JSON",
                    help="print each accepted session's memory line in both forms from a verifier.json written "
                         "before the average form (`recorded_memory`, with --host-bytes the node line), and exit")
    a = ap.parse_args()
    if a.memory_of is not None:
        v = json.loads(a.memory_of.read_text())
        print(json.dumps([{"k": s["k"], "memory": recorded_memory(s, host_bytes=a.host_bytes)}
                          for s in v["sessions"] if s["accepted"]], indent=1))
        return 0
    registry = load_registry(a.registry)
    if a.close is not None:
        if a.registry is None:
            ap.error("--close needs --registry")
        node = bytes.fromhex(a.close)
        lean = None if a.verdict == "python" else start_lean(os.urandom(32), registry, a.lean_exe, lambda m: None)
        closed = registry.close(node).to_json()
        kept = registry.to_json()
        if lean is not None:
            got, kept = lean.call({"op": "close", "node": node.hex()}), lean.call({"op": "registry"})
            lean.close()
            if got != closed:
                print(f"verifier: the verifiers disagree on closing {node.hex()}: Lean {json.dumps(got)}, "
                      f"audit.Registry {json.dumps(closed)}; {a.registry} is unchanged", flush=True)
                return 3
        save_registry(a.registry, kept)
        print(f"verifier: closed {json.dumps(closed)}", flush=True)
        return 0
    if a.out is None:
        ap.error("--out is required")
    a.out.mkdir(parents=True, exist_ok=True)
    save_registry(a.registry, registry)
    sys.setswitchinterval(2e-4)
    w0 = time.monotonic_ns()
    checks = Checks(a.check_workers) if a.check_workers > 0 else None
    server = wire.listen(a.host, a.port)
    print(f"verifier: listening on {a.host}:{a.port}", flush=True)
    sock, peer = server.accept()
    # Read again: another verifier process may have closed a registration while this one listened, and a refusal
    # saves this process's registry over the file.
    registry = load_registry(a.registry)
    conn = wire.Conn(sock)
    v = Verifier(conn, source=os.urandom(32), windows=a.windows, window_ns=round(a.window_ms * 1e6), samples=a.samples,
                 delta=2.0 ** a.delta_log2, jitter_ns=round(a.jitter_ms * 1e6),
                 log=lambda m: print(m, flush=True), checks=checks, lifetime=a.lifetime, registry=registry,
                 hbm_bytes=a.hbm_bytes, host_bytes=a.host_bytes, verdict=a.verdict, lean_exe=a.lean_exe,
                 transcript=a.transcript)
    c0 = time.process_time_ns()
    try:
        reg, cal = v.register()
    except (audit.RegistryError, ValueError) as e:
        print(f"verifier: refused the registration: {e}", flush=True)
        kept = v.registry_json()
        save_registry(a.registry, kept)
        (a.out / "verifier.json").write_text(json.dumps(
            {"schema": "verity/remote-session/verifier/v0", "peer": list(peer), "verdict": a.verdict,
             "refused": str(e), "registry": kept}, indent=1))
        if checks:
            checks.close(wait=True)
        v.stop()
        return 3
    save_registry(a.registry, v.registry_json())
    print(f"verifier: registered {reg.blocks} blocks, digest {reg.digest.hex()[:16]}, calibration {json.dumps(cal)}",
          flush=True)
    rtt = v.ping(a.pings)
    allowance = v.allowance(rtt, a.quantile)
    print(f"verifier: rtt {json.dumps(rtt)}, allowance A = {allowance / 1e6:.3f} ms", flush=True)
    results, not_before = [], None

    def record() -> None:
        kept = v.registry_json()
        (a.out / "verifier.json").write_text(json.dumps(
            {"schema": "verity/remote-session/verifier/v0", "peer": list(peer), "registration": wire.registration_json(reg),
             "verdict": a.verdict, "calibration": cal, "rtt": rtt, "allowance_ns": allowance, "sessions": results,
             "process_cpu_ns": time.process_time_ns() - c0, "wire": conn.totals(), "registry": kept}, indent=1))
        save_registry(a.registry, kept)

    record()
    try:
        for k in range(a.sessions):
            r = v.session(k, allowance, not_before)
            results.append(r)
            not_before = r.get("end_ns", v.clock.now())
            lg, pub = r.get("ledger", {}).get("spec", {}), r.get("record", {}).get("public", {})
            mem = r.get("ledger", {}).get("memory", {})
            print(f"verifier: session {k}: accepted={r['accepted']} failure={r['failure']} "
                  f"passes={[w['passes'] for w in r['windows_seen']]} gamma={lg.get('gamma')} "
                  f"appended={r['appended']} released={r['released']} post_seed={pub.get('post_seed')} "
                  f"memory share={mem.get('share')} residual={mem.get('residual')} epsilon={mem.get('epsilon')} "
                  f"average share={mem.get('average', {}).get('share')} "
                  f"node={json.dumps(mem['node']['share']) if 'node' in mem else None} "
                  f"node average={json.dumps(mem['node']['average']['share']) if 'node' in mem else None}",
                  flush=True)
            record()
    finally:
        save_registry(a.registry, v.registry_json())
    closed = v.close_registration(reg.node)
    print(f"verifier: closed the registration: {json.dumps(closed)}", flush=True)
    record()
    conn.send({"type": "bye"})
    if checks:
        checks.close(wait=True)
    v.stop()
    me, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    out = json.loads((a.out / "verifier.json").read_text()) if (a.out / "verifier.json").exists() else {}
    out["cpu"] = {"self_s": me.ru_utime + me.ru_stime, "workers_s": kids.ru_utime + kids.ru_stime,
                  "wall_s": (time.monotonic_ns() - w0) / 1e9, "cores": len(os.sched_getaffinity(0))}
    (a.out / "verifier.json").write_text(json.dumps(out, indent=1))
    print(f"verifier: cpu {json.dumps(out['cpu'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
