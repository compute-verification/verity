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
   (`--lifetime`, windows, by default those of `--window-ms` in 30 s, `audit.lifetime_windows`, sent with the
   session so the prover re-keys in time); σ_0, then a boundary message every window, each coin sent as its value and
   the root it was opened after (`coins.Released`), each append answered with its coin (`Verifier.append`), each root
   fed to `Session.receive` stamped with the time its last byte was read (with the releases its message carries first, and
   the withdrawn blocks among them, its plan as a change when it carries one, and the step its pass covers:
   `Verifier.root`), each window closed
   (`Session.close`) as soon as its roots are final and its draws sent; the prover's openings
   checked by the Lean verifier on its threads as their frames arrive (`Pair.verify`), so a window's checks run while
   the next window's passes arrive; once every window is accepted the record (`Session.record`), its declared outputs
   (`ledger.declared`'s: γ at both R and the memory line, snapshot and, paced, time-held and link-charged), and as
   diagnostics the node's HBM it shows held by kind (`memory`: the credited share and the residual, at ε, held at
   every pass and held on average), and with `--host-bytes` the same by kind against the node's HBM and host DRAM
   (`node_line`); last, once the prover has answered the session's `end` with `ended`, the coin commitment and every
   released coin's opening (`coins`, `audit.Session.disclose`), which the prover checks each coin it was sent against.

With `--paced K:T_MS:DRAWS:WINDOWS[:F]` each session is paced (`pace_of`, `audit.Pace(K, T, F)`): a coin every K
passes, each boundary message at `Session.next_boundary_ns`, WINDOWS stretches of as many groups as T_MS holds at the
`--pace-pass-ms` pass and a closing stretch, and γ declared per fixed T, so the record has a time-held memory line
(`ledger.memory`); `paced_line`'s per-stretch line stays beside it as a diagnostic. A group is drawn at its k-th
root, its roots final then, so its openings coin is asked for after that root first: the challenger's signed log
fixes each group's end there, and `auditor_line.py` reads the line on the challenger's clock from it. A boundary due by
then leaves right behind the close, so its coin, after the same root, is asked for in the same round trip, second
(`coins.SessionCoins.ahead`): at k = 1, every root's two coins are one round trip to the challenger.  At F = 1 a group's
boundary is due at its k-th root, so its coin is asked for first, after that root, and the close second, and the
prover waits a round trip on each coin.

A node has one live registration (`audit.Registry`), this process's own or kept across processes in `--registry`: a
hello for a node with a live registration is answered `bye` with the reason and draws nothing. The process closes its
registration after its last session, counting each session (a failed one by its failure); one left live by a
process that ended early stays live, refusing the node, until `--close NODE` closes it with its sessions as they stood,
the one in flight unended.

The payload's salts are the verifier's: its first message (`payload-key`) carries a fresh key, and the prover salts its
payload tree's leaves under it (`remote_bandwidth.store.store_salt`). With `--model ROLE --checkpoint DIR --pin FILE
--arena-blocks A [--engines E]` the registered store is bound to a pinned model (`remote_bandwidth.store.Binding`, the
remote audit's own code, so the circuit a period commits to names it): at startup
every file the pin names is held to its SHA-512 and the arena the checkpoint lays out is digested; a hello's store is
then E arenas of A blocks, weights and zero filler each, and the rest host filler, and its payload root must be that
store's under the key sent, else the hello is refused with the reason before the registry draws anything. The
Lean verifier decides it: it takes the pin's bytes and the weights blocks' digests at
startup (`bind_lean`, `RemoteAudit.Store.bind`) and holds the hello to them (`hold_store`, `Store.hold`, its root's
subtrees on `--layout-workers` threads), `Binding.hold` beside it as the reference (its root in as many processes),
the two computing their roots at once on the same cores and the hello decided once both have answered, and a hello on
which the two differ is refused as `disagreement`. The regions
are then the binding's, and verifier.json keeps the binding (`binding`: the pin's name and checkpoint_sha512, the
store's shape, both roots, the Lean's answer and `bound`), which `auditor_line.py` credits on. A process takes one
hello, so its sessions share the one root it held; another process's payload key is fresh, so its root is another.

The verdicts are the Lean verifier's, the process `remote-audit-verifier` (`lean_verifier.py`): it holds
the registry and every session, and each registration, session, event and window goes to it and to `audit.Session`
at one reading of the clock (`Pair`). An accepted session's record is the Lean's (`Pair.record`), and its ledger
reads that record. `audit.Session` keeps the driver's books (its coins, its plans, when the next boundary leaves) and
is the reference: an event, a window's verdict or a record field on which the two differ refuses the session at that
window as `disagreement`. It checks no openings: it takes the Lean's window verdicts (`Checks.settle`), and the
Python check of a window's openings beside the Lean's is the transcript's replay (`--transcript`). `--close` closes
a registration in both registries alike. While a session runs the driver doesn't wait on
the Lean: an event both take goes out on the books' result and the Lean's answer is held to it as it comes, and
openings frames and window checks go to the Lean the same way, so a slow Lean answer or check delays the session's
verdict, never a coin, a root's intake, a boundary or a draw. A session is accepted only once every one of its
Lean answers has come and agreed.

`--transcript DIR` writes each session's difftest case to DIR as the session runs (`lean_verifier.DifftestCase`): every
event as the Lean took it, with the clock it read, the session's coins, and each window's openings where its verdict
was set, which `replay.py transcript` (verity/core/protocols/remote_bandwidth/lean/tests) replays through both
verifiers, rejections included. A message the driver couldn't read is in no event: the case ends before it. A run
keeps DIR in scratch and preserves what it keeps of it (the writing-runs skill): a 60 s window's openings are about
0.6 GB of JSON.

A session's coins are an epoch of the proof system's one challenger (`coins.SessionCoins`): from a challenger in this process
(`--challenger local`, the default), from `benchmarks/live_challenger/serve.py`, started for the session on this
machine and reached through `sampled_proofs.remote`'s client (`--challenger serve`, `challengers.Serve`), or from the
auditor host's challenger through its gate, a session reserved in the audit's period and opened at the session's
window root (`--challenger gate`, `challengers.ChallengerGate`). Each coin is asked for after the root it follows, the
Lean takes it with the event that released it, and once the session ends what the challenger signed holds the session's
coins (`Session.released_reason`, and under the gate the auditor's ledger, `Session.trusted_reason`, which also holds the
session's pin to the host's pins log and to `--challenger-main`, the auditor's first-parent list of main): a session
whose coins fail it is refused as `coins`, the reason in its `challenger`. Each session reserves W challenges, ⌈W/G⌉
openings coins (`--groups-per-coin` G, 1 by default: each draw's G windows drawn together at its last window's close,
at two windows in flight, as the Lean's `session` refuses one in flight at G > 1) and `--appends` append coins,
`coins.APPENDS` by default, or none, and no append family, when the prover's hello says its sessions append nothing
(a plan of honest and skip sessions, `Verifier.appends_of`), so the challenger's close of such a session that ran to
its end is complete. An append past them refuses the session as `disagreement`: the books refuse it as `coin`, and
the Lean's session, which holds only the append coins released, otherwise.

Openings are checked in the clear: zero-knowledge proof blocks are not in this harness (PROTOCOL.md §8).

    python verifier.py --port 7781 --out OUT [--windows 7 --window-ms 1000 --samples 2773 --sessions 1]
                       [--registry REGISTRY.json] [--lean-exe PATH]
                       [--paced K:T_MS:DRAWS:WINDOWS[:F] ...] [--transcript DIR]
                       [--challenger local|serve --challenger-key DIR [--challenger-params W,H]] [--appends A]
                       [--groups-per-coin G]
                       [--challenger gate --challenger-ssh WORDS --challenger-peer NAME --challenger-pin SHA512
                        --challenger-period P --challenger-main FILE [--challenger-minutes M]]
                       [--model ROLE --checkpoint DIR --pin FILE --arena-blocks A [--engines E]]
    python verifier.py --registry REGISTRY.json --close NODE_HEX
    python verifier.py --memory-of VERIFIER_JSON [--host-bytes N]
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
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

from verity.core.protocols.remote_bandwidth import audit, coins, ledger, line, timing  # noqa: E402
from verity.core.protocols.remote_bandwidth import store as rstore  # noqa: E402

import challengers  # noqa: E402
import lean_verifier as lv  # noqa: E402
import wire  # noqa: E402
from checks import Checks  # noqa: E402

CHECK_BATCH = 1024
"""Openings a Lean thread checks at once (`checkWindowChunked`)."""
GPU_HBM_BYTES = 96 * 2 ** 30
"""One RTX PRO 6000's memory, 96 GiB (`benchmarks/census/hardware.json`)."""
HBM_BYTES = timing.GPUS * GPU_HBM_BYTES
DISK_BYTES_PER_S = 4.4e9
"""The node's disks' read ceiling, each disk's highest second summed (4.311e9) and rounded up:
how much of a paced line's ingress charge is its disks (`ingress_bursts`)."""
KINDS = ("weights", "kv", "filler", "activations")
TIERS = ("hbm", "host")
UNRECORDED_SLACK_LOG2 = 41
"""The AuditRO slack a ledger whose `inputs` carry no `slack_log2` was charged at, bits a pass: every record written
before they carried it, gate runs 8 and 9's among them."""


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
    `blocks` exactly (`verity_vllm.protocols.remote_bandwidth.registry.NodeMemory.layout`'s), or None without one.  The labels are the prover's,
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
           credit_post_seed: audit.PostSeedVerdict | None = None, schedule: audit.Scheduled | None = None) -> dict:
    """The node's memory a session's record shows held, against its HBM, in two forms.

    Held at every pass (`WORST_PASS`, the worst pass's): each window's passes each read the distinct blocks of their
    plan (`plans`, a list of each window's, as `Session` received them) inside the window's span, so at every pass the
    node held them, but for the receipts the pooled event leaves wrong (`audit.floors`, f_t of the window's U_t, every
    window's at once except with the record's δ), the blocks its GPUs could re-encode from their payloads in the
    span instead (`rebuild`, its squarings at `codec.encode_squarings` a block), and the blocks its on-chip storage
    holds at the pass instead of HBM (`chip_bytes`, `chip_blocks`).  Those lost blocks could all fall in one pass, so
    they come off every pass, and the held bytes a window credits are its least pass's, less them all.

    Held on average (`AVERAGE`, each window's `average` and the session's `average`, `averaged`): the mean of the
    window's passes, less f_t·U_t/P_t, the rebuilds and `chip_bytes` of the node's on-chip storage.

    A registered block counts as the layout labels it (`regions_of`; None, the drivers' stores: the whole store
    filler in HBM), HBM only toward the share; an appended block
    (index N and up) is KV the session wrote, in HBM, and is credited only on the property's verdict and the session's
    schedule check (`schedule`, `audit.Session.scheduled`, as `audit.ledger` requires them), less the most its carriers
    hold (`credit_post_seed`, as `audit.ledger`'s μ_carrier).  Weights and KV are each credited on their
    own: the lost blocks come off each.  The session's line is its worst window's: `credited_bytes`, `share` of
    `hbm_bytes` and `residual` = 1 − share, the HBM it doesn't show used, with ε, the share of the window's held
    basis the losses took, at δ.  ε is this reading's (the floor and rebuilds against held blocks), not a composed
    guarantee's."""
    if credit_post_seed is not None and credit_post_seed.holds is not True:
        raise ValueError("a verdict that the property fails credits nothing after the seed: pass None")
    if credit_post_seed is not None and (type(schedule) is not audit.Scheduled or not schedule.of(record)):
        raise ValueError("KV is credited only on this session's schedule check (`audit.Session.scheduled`)")
    if len(plans) != len(record.windows) or any(len(p) != w.passes for p, w in zip(plans, record.windows)):
        raise ValueError("a plan for each pass of each window of the record")
    if rebuild.codec != record.codec:
        raise ValueError("the rebuild model is of another codec than the record's")
    b, labels = record.block_bytes, regions is not None
    regions = regions if labels else (("filler", "hbm", 0, record.blocks),)
    per_ns, chip = rebuild.squarings_per_s / record.codec.encode_squarings / 1e9, chip_bytes / b
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
        lost = f * w.receipts + per_ns * w.span_ns + chip
        held = {"weights": min(p[0]["weights"] for p in passes) * b, "kv": min(p[1] for p in passes) * b,
                "hbm": min(p[0]["hbm"] for p in passes) * b, "host": min(p[0]["host"] for p in passes) * b}
        basis = min(p[0]["hbm"] * b + p[2] for p in passes)
        mean = {k: sum(p[0][k] for p in passes) * b / w.passes for k in ("weights", "hbm", "host")}
        windows.append({
            "window": w.window, "passes": w.passes, "span_ns": w.span_ns, "floor": f, "chip_blocks": chip,
            "lost_blocks": lost, "held": held, "read": {"weights": sum(p[0]["weights"] for p in passes) * b, "kv": sum(w.post_seed) * b},
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
    the same host blocks; KV, credited on no verdict, is none.  The rebuilds are the recorded lost blocks less f_t·U_t
    and the recorded chip blocks (none in a record written before the worst-pass form charged the chip), and both
    forms charge `chip_bytes`.  With `host_bytes`, the node line too (`node_line`)."""
    mem, pub = session["ledger"]["memory"], session["record"]["public"]
    if mem["kv_credited"]:
        raise ValueError("a record that credits KV keeps each pass's KV, which a recorded window doesn't")
    b, windows = mem["block_bytes"], []
    for x, receipts, post in zip(mem["windows"], pub["receipts"], pub["post_seed"]):
        p, wrong = x["passes"], x["floor"] * receipts
        rebuilt = x["lost_blocks"] - wrong - x.get("chip_blocks", 0.0)
        lost = wrong + rebuilt + chip_bytes / b
        hbm = (receipts - post) * b / p - x["held"]["host"]
        mean = {"weights": x["read"]["weights"] / p, "hbm": hbm, "host": x["held"]["host"]}
        windows.append(x | {
            "chip_blocks": chip_bytes / b, "lost_blocks": lost,
            "epsilon": min(1.0, lost * b / x["basis"]) if x["basis"] > 0 else 1.0,
            "credited": {"hbm": max(0.0, x["basis"] - lost * b), "weights": max(0.0, x["held"]["weights"] - lost * b),
                         "kv": 0.0, "kv_read": 0.0},
            "average": {"window": x["window"], **averaged(
                mean, 0.0, wrong_blocks=wrong / p, rebuild_blocks=rebuilt, passes=p, chip_bytes=chip_bytes,
                block_bytes=b)}})
    out = summed(windows, hbm_bytes=mem["hbm_bytes"], block_bytes=b, delta=mem["delta"], labels=mem["labels"],
                 kv_credited=False, chip_bytes=chip_bytes)
    if (error := memory_error(session["ledger"])) is not None:
        out["error"] = error
    if host_bytes:
        out["node"] = node_line(out, host_bytes=host_bytes)
    return out


def memory_error(ledgers: dict) -> float | None:
    """The memory line's failure probability from `Verifier.ledgers`' output: its floors are the pooled event's, so the
    horizon's error (δ once, and AuditRO's 2^-slack a pass), plus the sketch's (`RateAuditSketch`), plus the coin
    commitment's hiding terms, the receipts' binding term and the draw's rejection (`ledger.hiding`, `ledger.binding`,
    `ledger.rejection`, the ledger's `delta` parts); `delta` alone understates it by the passes' slack.  None when no
    reading has a ledger."""
    for name in ("spec", "measured"):
        if "error" in ledgers.get(name, {}):
            d = ledgers.get("delta", {})
            return (ledgers[name]["error"]["horizon"] + 2.0 ** ledgers.get("sketch_failure", {}).get("log2", -math.inf)
                    + d.get("hiding", 0.0) + d.get("binding", 0.0) + d.get("rejection", 0.0))
    return None


def sketch_at(session: dict, registration: dict, outside: int | None = None) -> dict | None:
    """A session's SketchBound δs from its record (`Verifier.ledgers`' `sketch_failure`) against Q = F·C queries, C the
    compressions the verifier charged the node over the span and F the outside-compute factor: the record's (1 for one
    written before F was recorded, which charged the node's alone), or `outside` to read it at another; with the memory
    line's error there, the horizon's plus δs (`memory_error` at F).  None for a session whose record has no sketch
    term."""
    lg = session.get("ledger") or {}
    sf, inputs = lg.get("sketch_failure"), lg.get("inputs") or {}
    if not sf or not inputs.get("rows"):
        return None
    f = sf.get("outside", 1) if outside is None else outside
    c = sf["compressions"]
    d = timing.sketch_failure(c, inputs["passes"], inputs["rows"], blocks=registration["blocks"], outside=f)
    horizon = next((lg[n]["error"]["horizon"] for n in ("spec", "measured") if "error" in lg.get(n, {})), None)
    return {"outside": f, "compressions": c, "queries": f * c, "log2": timing.log2(d),
            "error": None if horizon is None else horizon + float(d)}


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


TIME_HELD = "held over time"
"""The paced session's third form (`paced_line`): the blocks in node memory at each instant, averaged over time."""


def pace_of(spec: str, pass_ms: float) -> dict:
    """`K:T_MS:DRAWS:WINDOWS[:F]` as a session's pacing: groups of K passes, each a `Session` window of its own whose
    coin is a live coin, WINDOWS credited stretches of T_MS and a closing one after them, `groups` groups a stretch (at
    `pass_ms` a pass) and `draws` = DRAWS/groups openings a group, so a stretch draws as an unpaced T_MS window would;
    F the windows in flight (`audit.Pace.in_flight`, 2 when absent, named only at 1)."""
    parts = spec.split(":")
    if len(parts) not in (4, 5):
        raise ValueError(f"a pace is K:T_MS:DRAWS:WINDOWS[:F], not {spec!r}")
    k, t_ms, draws_t, windows = (float(x) if i == 1 else int(x) for i, x in enumerate(parts[:4]))
    f = int(parts[4]) if len(parts) == 5 else 2
    if f not in (1, 2):
        raise ValueError(f"a pace's windows in flight are 1 or 2, not {f}")
    groups = max(1, round(t_ms / (k * pass_ms)))
    return {"k": k, "t_ms": t_ms, "draws_t": draws_t, "windows": windows, "groups": groups,
            "draws": math.ceil(draws_t / groups), "pass_ms": pass_ms} | ({"in_flight": 1} if f == 1 else {})


def paced_groups(mem: dict, seen: list[dict], *, block_bytes: int) -> list[line.Group]:
    """A paced session's groups as its line reads them (`line.Group`), from its memory windows (`memory`'s, one a
    group) and the windows it saw: each one's coin's release, its last root's arrival (its coin's, with no root),
    its passes, the registered blocks its least pass read (`held`), and each kind's blocks its passes read in all,
    which their mean a pass (`average`'s `held`) gives back exactly or the group is refused."""
    b = block_bytes
    if len(mem["windows"]) != len(seen):
        raise ValueError(f"a memory window for each group: {len(mem['windows'])} for {len(seen)} groups")
    out = []
    for g, (x, w) in enumerate(zip(mem["windows"], seen)):
        n, least, mean = w["passes"], x["held"]["hbm"] + x["held"]["host"], x["average"]["held"]
        read = {k: round(mean[k] * n / b) for k in ("weights", "hbm", "host")}
        if least % b or any(read[k] * b / n != mean[k] for k in read):
            raise ValueError(f"group {g}'s held bytes are no whole number of blocks: its least pass's {least} and "
                             f"its mean a pass {mean} over {n} passes of {b}-byte blocks")
        released = w["released_ns"]
        out.append(line.Group(released, w["arrivals_ns"][-1] if w["arrivals_ns"] else released, n, least // b,
                              read["weights"], read["hbm"], read["host"]))
    return out


def ingress_bursts(ingress_bytes_per_s: float, disk_bytes_per_s: float = DISK_BYTES_PER_S) -> float:
    """B, the bytes the routes back into node memory bring at once beyond their rates at an ingress charge of
    `ingress_bytes_per_s`, `ledger.Node.stated`'s bursts: one rebuild batch in flight (S_chip), one second at the disks'
    ceiling (the charge up to `disk_bytes_per_s`), and one link window (`timing.LINK_WINDOW_BYTES`) when the charge is
    above that, the boundary links charged too."""
    disk = min(ingress_bytes_per_s, disk_bytes_per_s)
    n = ledger.Node.stated(disk_bytes_per_s=disk, link_bytes_per_s=ingress_bytes_per_s - disk)
    return n.rebuild_burst_bytes + n.disk_burst_bytes + n.link_burst_bytes


def paced_line(mem: dict, seen: list[dict], *, groups: int, wrong: float, blocks: int, rebuild_per_ns: float,
               chip_bytes: int, block_bytes: int, node_bytes: int, burst_bytes: float,
               ingress_per_ns: float = 0.0) -> dict:
    """A paced session's memory line over each stretch of `groups` groups (all but the closing stretch), on the
    verifier's clock, in two forms: `line.stretches`' readings held over time (`TIME_HELD`) and at the passes
    (`AVERAGE`), of its groups (`paced_groups`) with `wrong` the session's pooled wrong receipts (`audit.pooled`), in
    floats for display.

    A block also comes back into node memory, or onto the chip, without a rebuild from the node's own disks, which
    Identity leaves in (they are inside the node), and over its boundary links, which Identity takes as carrying no
    block.  `ingress_per_ns`, those routes' read ceiling in blocks a ns, is charged with the rebuilds at
    `rebuild_per_ns` in both forms, and `burst_bytes` (B, `ingress_bursts`) beyond them: `SquaringCeiling`, `Disks`
    and `Links` hold as rate·length + burst, and a burst of 0 states the rate down to one nanosecond.  This is the
    verifier's diagnostic; the line of record (`recorded_line`, `auditor_line.py`) charges each form its own rates and
    bursts.

    Both forms need every pass to read every registered block, so a session any of whose passes leaves one out (a
    group's least pass, `memory`'s `held`, short of the store) is refused (`line.Refused`)."""
    b, ingress, burst = block_bytes, line.number(ingress_per_ns), line.number(burst_bytes) / block_bytes
    xs = line.stretches(paced_groups(mem, seen, block_bytes=b), per=groups, blocks=blocks, block_bytes=b,
                        chip_bytes=chip_bytes, wrong=wrong, node_bytes=node_bytes,
                        charge=line.Charge(line.number(rebuild_per_ns) + ingress, burst))
    readings = {AVERAGE: line.SNAPSHOT, TIME_HELD: line.TIME}
    stretches = [{"stretch": x.stretch, "groups": [x.lo, x.hi], "span_ns": x.span_ns, "passes": x.passes,
                  "l_max_ns": x.l_max_ns, "l_mean_ns": x.l_sum_ns / (x.hi - x.lo),
                  "rebuild_blocks": float(x.rebuild_blocks - burst),
                  "ingress_blocks": float(ingress * (x.span_ns + 2 * x.l_max_ns)), "burst_blocks": float(burst),
                  "rebuild_absent_blocks": float(2 * x.l_max_ns * (x.rebuild_blocks - burst) / x.span_ns),
                  "burst_absent_blocks": float(2 * x.l_max_ns * burst / x.span_ns),
                  "wrong_absent_blocks": float(x.fill / x.span_ns),
                  "held": {k: float(v * b) for k, v in zip(("weights", "hbm", "host"), x.held)},
                  **{form: {"lost_blocks": float(r.lost_blocks), "credited": {
                      "weights": float(r.weights), "hbm": float(r.hbm), "filler": float(r.filler)}}
                     for form, r in ((form, x.reading(k)) for form, k in readings.items())}} for x in xs]

    def worst(form: str) -> dict:
        x = line.worst(xs, readings[form])
        r = x.reading(readings[form])
        return {"form": form, "stretch": x.stretch, "lost_blocks": float(r.lost_blocks),
                "weights_bytes": float(r.weights), "share": float(r.weights / node_bytes),
                "hbm_bytes": float(r.hbm), "hbm_share": float(r.hbm / node_bytes), "filler_bytes": float(r.filler),
                "filler_share": float(r.filler / node_bytes)}

    return {"groups": groups, "wrong": wrong, "chip_blocks": -(-chip_bytes // b), "burst_blocks": float(burst),
            "node_bytes": node_bytes, "stretches": stretches, "snapshot": worst(AVERAGE), "time": worst(TIME_HELD)}


def recorded_chip_bytes(mem: dict) -> int:
    """S_chip as a recorded session's line deducts it: the record's, or the auditor's `timing.CHIP_BYTES` where that is
    larger. A record keeps the S_chip its verifier ran at, which can be short of the node's: gate runs 7 to 9's
    3,045,746,688 B took the host's term from an estimate of its caches, 59,906,048 B under `timing.HOST_CHIP_BYTES`."""
    return max(mem["chip_bytes"], timing.CHIP_BYTES)


def recorded_paced(session: dict, registration: dict, *, host_bytes: int, ingress_bytes_per_s: float = 0.0) -> dict:
    """An accepted paced session's line (`paced_line`) from its record in `verifier.json`, against its HBM and
    `host_bytes`, its S_chip at least the auditor's (`recorded_chip_bytes`), with `ingress_bytes_per_s` of other routes
    into node memory charged, and their bursts (`ingress_bursts`); the pooled wrong receipts are the recorded line's."""
    lg, mem = session["ledger"], session["ledger"]["memory"]
    b, codec = mem["block_bytes"], wire.codec_named(registration["codec"])
    return paced_line(mem, session["windows_seen"], groups=session["pace"]["groups"], wrong=lg["paced"]["wrong"],
                      blocks=registration["blocks"],
                      rebuild_per_ns=line.number(lg["inputs"]["squarings_per_s"]) / codec.encode_squarings / line.NS,
                      chip_bytes=recorded_chip_bytes(mem), block_bytes=b, node_bytes=mem["hbm_bytes"] + host_bytes,
                      burst_bytes=ingress_bursts(ingress_bytes_per_s),
                      ingress_per_ns=line.number(ingress_bytes_per_s) / b / line.NS)


def recorded_rates(session: dict, registration: dict, **node: float) -> line.Rates:
    """The node's ways back into memory as the line of record reads them (`line.Rates`): the record's squaring ceiling
    and its codec's encode, and `node`'s disks' and links' ceilings and bursts (`line.RATES`' others)."""
    return line.Rates(session["ledger"]["inputs"]["squarings_per_s"],
                      wire.codec_named(registration["codec"]).encode_squarings, **node)


def lean_record(session: dict) -> dict:
    """The Lean verifier's record of an accepted session in `verifier.json` (its `outcome` request's `record`,
    `lean_verifier.summary_json`'s form), as the session keeps it (`lean_record`), the Lean's where its verdict is
    the Lean's (lean or both). A run from before sessions kept it wrote the same record read through
    `lean_verifier.record_of` (`Pair.record`) as its `record`'s `public` and `profile`, which give back every field
    but each pass's root's arrival, and its `windows_seen` those stamps: it is read back from them, refused where they
    don't fit."""
    if "lean_record" in session:
        return session["lean_record"]
    pub, ws, seen = session["record"]["public"], session["record"]["profile"]["windows"], session["windows_seen"]
    if len(seen) != len(ws) or any(len(x["arrivals_ns"]) != len(w["lengths"]) for w, x in zip(ws, seen)):
        raise ValueError(f"the session's windows seen are not its record's: {len(seen)} for {len(ws)} windows, or "
                         "a window's arrivals not one a pass")
    return {"appended": pub["appended"], "appends": pub["appends"], "blocks": pub["blocks"],
            "lifetime": pub["lifetime"], "pace": pub["pace"], "registration": pub["registration"],
            "released": pub["released"], **{k: pub[k] for k in ("groups_per_coin",) if k in pub},
            "windows": [{"arrived_ns": list(x["arrivals_ns"]), "deadline_ns": w["deadline_ns"],
                         "drawn": w["profile"]["drawn"]["receipt"], "heads": w["heads"], "lengths": w["lengths"],
                         "opened_ns": w["opened_ns"], "post_seed": w["post_seed"], "released_ns": w["released_ns"],
                         "stale": w["stale"], "steps": w["steps"]} for w, x in zip(ws, seen)]}


def recorded_windows(session: dict) -> tuple[list[tuple[int, int]], float]:
    """The pooled bound's inputs (`pooled.bound`) from an accepted session's record in `verifier.json`: each draw's
    receipts and draws (`audit.pooled_windows`: a window's own at `groups_per_coin` g = 1, its g windows' together
    above), and δ, the ones the line reads off the Lean verifier's record (`lean_record`) and the session."""
    g, ws = session["record"]["public"].get("groups_per_coin", 1), session["record"]["profile"]["windows"]
    return ([(sum(sum(w["lengths"]) for w in ws[a:a + g]), sum(w["profile"]["drawn"]["receipt"] for w in ws[a:a + g]))
             for a in range(0, len(ws), g)], ws[0]["profile"]["delta"])


def record_seen(mem: dict, seen: list[dict], record: dict, *, block_bytes: int) -> list[line.Seen]:
    """What the auditor saw of each window beside the Lean verifier's `record` of it (`line.Seen`): its group's times
    and reads as `paced_groups` reads them, its passes the record's or refused."""
    gs = paced_groups(mem, seen, block_bytes=block_bytes)
    if [g.passes for g in gs] != [len(w["lengths"]) for w in record["windows"]]:
        raise ValueError("the windows seen are not the record's: a group for each of its windows, a pass for each of "
                         "its passes")
    return [line.Seen(g.released_ns, g.last_ns, g.least, g.weights, g.hbm, g.host) for g in gs]


def recorded_inputs(session: dict, registration: dict, *, host_bytes: int, rates: line.Rates, clock: str = "record",
                    transcript: dict | None = None) -> dict:
    """An accepted paced session's line's inputs in the line's record form (`line.record_inputs_json`, the Lean
    verifier's `line_record` request reads them): the Lean verifier's record of it (`lean_record`), whose groups'
    passes, windows and registered blocks the line reads, at the session's `samples` and δ; and what the auditor saw of
    each window beside it (`record_seen`, on the clock its `windows_seen` read: `clock` record, the verifier's, which
    the record's times must be, or another, `line.CLOCKS`), against its HBM and `host_bytes`, at `rates`
    (`recorded_rates`), with its S_chip at least the auditor's (`recorded_chip_bytes`); with `transcript`, the
    session's (`lean_verifier.DifftestCase`'s file), from which the Lean makes the record itself. Refused unless the
    record is of `registration` (`wire.registration_json`'s form), which the inputs name as the Lean reads one, so
    that the line refuses a record not of it too (`line.unregistered`)."""
    mem, record = session["ledger"]["memory"], lean_record(session)
    if record["registration"] != registration["digest"]:
        raise ValueError(f"the Lean's record is of registration {record['registration'][:16]}…, not "
                         f"{registration['digest'][:16]}…")
    return line.record_inputs_json(
        record, registration=lv.registration_json(wire.registration_of(registration)), samples=session["samples"],
        seen=record_seen(mem, session["windows_seen"], record, block_bytes=mem["block_bytes"]),
        per=session["pace"]["groups"], block_bytes=mem["block_bytes"], chip_bytes=recorded_chip_bytes(mem),
        node_bytes=mem["hbm_bytes"] + host_bytes, delta=session["delta"], rates=rates, clock=clock, session=transcript)


def recorded_line(session: dict, registration: dict, *, host_bytes: int, rates: line.Rates) -> line.Line:
    """An accepted paced session's line in both forms (`line.line_of_record`, exact) of its `recorded_inputs`."""
    return line.line_of_record(**line.of_record_json(recorded_inputs(session, registration, host_bytes=host_bytes,
                                                                     rates=rates)))


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
    try:
        read = lean.call({"op": "init", "source": source.hex(), "registry": registry.to_json(), "chunk": CHECK_BATCH})
    except BaseException:
        lean.close()
        raise
    if read["registry"] != registry.to_json():
        lean.close()
        raise ValueError(f"the Lean verifier reads the registry as {read['registry']}, not as audit.Registry does: "
                         f"{registry.to_json()}")
    return lean


def bind_lean(lean: lv.Lean, b: rstore.Binding) -> dict:
    """The store binding started in the Lean verifier (`bind`, `RemoteAudit.Store.bind`): the pin from the bytes
    `b` read of its file, the store's shape, and the weights blocks' digests `b` computed, which the Lean's `hold`
    salts and climbs on `b.workers` threads. Its answer, `{"pin", "weights_blocks"}` or `{"refused": why}`, or
    `{"error": why}`."""
    digests = b.digests.read_bytes()
    try:
        return lean.call({"op": "bind", "bytes": len(digests), "pin": b.pin_data.hex(), "path": str(b.pin_path),
                          "role": b.role, "engines": b.engines, "arena_blocks": b.arena_blocks, "workers": b.workers},
                         digests)
    except lv.LeanError as e:
        return {"error": str(e)}


def hold_store(lean: lv.Lean, bound: dict | None, b: rstore.Binding, node: bytes, blocks, root: bytes,
               key: bytes, layout) -> tuple[tuple[str, str, int, int], ...]:
    """The hello's store held to the pinned model's under the payload key `key`: the binding's regions, else
    `remote_bandwidth.store.Refused` with why. The Lean verifier decides (`hold`, on the binding
    `bind_lean` started and answered `bound`), `Binding.hold` beside it as the reference: the Lean's request goes out
    first and its root is computed on its threads while `Binding.hold` computes store.py's in its processes, and the
    hello is decided once both have answered, whichever is first. Where the two answer otherwise, the regions, the
    reason or the expected root, the hello is refused as `disagreement`, and `bound` is the Lean's only where they
    agree. The binding's record keeps the Lean's answer (`lean`) and its time from request to answer (`lean_s`)."""
    answer: queue.SimpleQueue = queue.SimpleQueue()
    t0 = time.monotonic()
    if bound != {"pin": b.pin, "weights_blocks": b.arena.weights_blocks}:
        answer.put(({"bind": bound}, t0))
    else:
        lean.send({"op": "hold", "node": node.hex(), "blocks": blocks, "payload_root": root.hex(), "key": key.hex(),
                   "layout": None if layout is None else [list(r) for r in layout]},
                  answered=lambda got: answer.put((got, time.monotonic())))
    regions = None
    try:
        regions = b.hold(node, blocks, root, key, layout)
        mine = {"regions": [list(r) for r in regions], "expected_root": b.state["expected_root"]}
    except ValueError as e:
        mine, why = {"refused": str(e), "expected_root": b.state.get("expected_root")}, e
    got, t1 = answer.get()
    b.state.update(lean=got, lean_s=t1 - t0)
    if got != mine:
        b.state["bound"] = False
        raise rstore.Refused(f"disagreement: the Lean verifier's binding is {json.dumps(got)}, store.py's "
                             f"{json.dumps(mine)}")
    if regions is None:
        raise why
    return regions


class Pair:
    """Session k through the verifiers, with `audit.Session`'s interface: each event goes to `s`, the driver's books,
    and to the Lean process's session k, at one reading of the verifier's clock. The result is the one
    they agree on; where they differ, the session is refused at the event's window as `disagreement` (in both
    verifiers), returned as that rejection, or raised as `audit.SessionError` where `audit.Session` returns no verdict.
    Everything else is read from `s`. The Lean's session takes its coins live (`"coins": null`): each the challenger
    releases goes with the event that released it, σ_t with its start or boundary, window t's openings coin with its
    close and an append's coin with the append.

    The driver never waits on the Lean for an event the books take: it acts on the books' result and the Lean's
    answer is held to it as it comes (`_answered`), a different one refusing the session at that event's window
    (`caught_up`) before any later verdict, rejection or record is taken. The Lean's session is a function of the
    events in order with their clock readings, so its answers are the ones it would give were each waited on. An
    event the books reject or refuse is waited on, behind every one before it."""

    def __init__(self, v: Verifier, s: audit.Session, k: int, case: lv.DifftestCase | None = None) -> None:
        self.v, self.s, self.k, self.lean, self.case = v, s, k, v.lean, case
        self.node, self.disagreement, self.outcome_json = s.registration.node.hex(), None, None
        self.late: tuple | None = None
        self.frames: dict[int, list[bytes]] = {}
        got = self._lean({"op": "session", "node": self.node, "params": lv.params_json(s.params),
                          "coins": None, "append_coins": [], "not_before_ns": s.not_before_ns, "index": k})
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
        self._lean({"op": "abort", "node": self.node, "session": self.k, "outcome": "disagreement"})
        self.v.log(f"verifier: session {self.k} refused at window {t}: the verifiers disagree on {event}: "
                   f"Lean {lean}, Python {python}")
        return v

    def _answered(self, t: int, event: str, lean: dict, python: dict) -> None:
        """The Lean's answer to an event the driver acted on, on the Lean's reader thread: the first that differs from
        the books' is kept for `caught_up`, and the session's loop woken to apply it."""
        if lean != python and self.late is None:
            self.late = (t, event, lean, python)
            self.v.inbox.put(({"type": "lean-disagree", "k": self.k}, b"", self.v.clock.now()))

    def caught_up(self) -> audit.Verdict | None:
        """The first event the Lean answered otherwise than the books, applied as `disagreement` (`_disagree`): its
        rejection, or None. The Lean answers in order and each answer is held to the books' before the next is read,
        so once the driver holds any answer, every event before it that disagreed is here."""
        if self.late is None or self.disagreement is not None:
            return None
        return self._disagree(*self.late)

    def _step(self, t: int, event, python, as_json, *, clocked: bool = False, coin=None):
        """`python()` on the books and `event()` through the Lean: `python()`'s return, or the disagreement's
        rejection (None as the second value)."""
        self.caught_up()
        if self.disagreement is not None:
            raise audit.SessionError("the session ended: the verifiers disagree")
        ev = event()
        held = self.v._stamp
        now = self.v._stamp = self.v._now()
        if clocked:
            ev["now"] = now
        if self.case is not None:
            self.case.event(ev)
        try:
            try:
                out = python()
                got = as_json(out)
            except audit.SessionError:
                out, got = None, lv.REFUSED
        finally:
            self.v._stamp = held
        c = coin(out) if coin is not None else None
        request = {"op": "event", "node": self.node, "session": self.k, "event": ev,
                   "coin": None if c is None else c.hex()}
        where = t if type(t) is int and t >= 0 else self.s.filling
        if got != lv.REFUSED and not isinstance(out, audit.Verdict):
            self.lean.send(request, answered=lambda lean: self._answered(where, ev["kind"], lean, got))
            return out, True
        lean = self._lean(request)
        if (v := self.caught_up()) is not None:
            return v, False
        if lean != got:
            return self._disagree(where, ev["kind"], lean, got), False
        if got == lv.REFUSED:
            raise audit.SessionError("refused by both verifiers")
        return out, True

    def _released(self, label: str, index: int) -> bytes | None:
        """Coin `index` of family `label`, once the challenger has released it."""
        opened = self.s.coins.opened(label)
        return coins.coin(opened[index]) if type(index) is int and 0 <= index < len(opened) else None

    def start(self):
        out, agreed = self._step(0, lambda: {"kind": "start"}, self.s.start, lv.released_json, clocked=True,
                                 coin=lambda out: self._released(coins.CHALLENGE, 0))
        if not agreed:
            raise audit.SessionError("the verifiers disagree on the session's start")
        return out

    def boundary(self):
        t = next((t for t, w in enumerate(self.s.windows) if w.boundary_ns is None), len(self.s.windows))
        out, agreed = self._step(t, lambda: {"kind": "boundary"}, self.s.boundary, lv.released_json, clocked=True,
                                 coin=lambda out: None if out is None else self._released(coins.CHALLENGE, out.index))
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
        """Window t's close: at `groups_per_coin` g, its draw's (t // g), whose openings coin goes with it."""
        return self._step(t, lambda: {"kind": "close", "t": t}, lambda: self.s.close(t), lv.drew_json, clocked=True,
                          coin=lambda out: self._released(coins.OPENINGS, t // self.s.params.groups_per_coin))[0]

    def openings(self, t: int, body: bytes) -> None:
        """One frame of window t's openings to the Lean, kept there until the window is verified."""
        def answered(got: dict) -> None:
            if "error" in got:
                self.v.log(f"verifier: the Lean refused window {t}'s openings frame: {got['error']}")

        self.lean.send({"op": "openings", "node": self.node, "session": self.k, "t": t, "bytes": len(body)}, body,
                       answered)

    def keep(self, t: int, body: bytes) -> None:
        """One frame of window t's openings, kept for the transcript until the window's verdict is set."""
        if self.case is not None:
            self.frames.setdefault(t, []).append(body)

    def transcribe(self, t: int) -> None:
        """Window t's openings into the transcript, as a `verify` event where its verdict was set."""
        if self.case is not None:
            self.case.verify(t, self.frames.pop(t, []))

    def passes_ended(self) -> None:
        """The prover's passes are over: the transcript may write the openings it held back."""
        if self.case is not None:
            self.case.passes_ended()

    def verify(self, t: int, check: int, refused) -> None:
        """Window t's frames checked on the Lean's threads as check `check`, whose end comes as a `lean-done`;
        `refused(answer)` called on the Lean's reader thread where it refuses to start it."""
        self.lean.send({"op": "verify", "node": self.node, "session": self.k, "t": t, "id": check},
                       answered=lambda got: None if got == {"spawned": True} else refused(got))

    def settle(self, check: int, answered=None) -> dict | None:
        """Check `check`'s verdict applied to the Lean's session, behind every event before it: its result, the
        openings it read and its ns, returned, or given `answered`, passed to it on the Lean's reader thread."""
        if answered is None:
            return self._lean({"op": "settle", "id": check})
        self.lean.send({"op": "settle", "id": check}, answered=answered)
        return None

    def judge(self, t: int, lean: dict) -> dict:
        """Window t's verdict, `{"accepted", "failure", "at"}`: the Lean's settled result, set on the books
        (`Checks.settle`); where the books refuse it, `disagreement`."""
        mine = {k: x for k, x in lean.items() if k not in ("openings", "ns")}
        try:
            n = len(self.s.windows[t].draws)
            v = Checks.settle(self.s, t, [{"failure": mine.get("failure"),
                                           "at": tuple(mine["at"]) if mine.get("at") else None}], n)
            theirs = lv.verdict_json(v)
        except audit.SessionError:
            theirs = lv.REFUSED
        if self.disagreement is None and mine != theirs:
            self._disagree(t, "verify", mine, theirs)
        if self.disagreement is not None:
            return {"accepted": False, "failure": "disagreement", "at": None}
        return {"accepted": "accepted" in mine, "failure": mine.get("failure", "refused" if mine == lv.REFUSED else None),
                "at": mine.get("at")}

    def end(self, failure: dict | None) -> None:
        """The session ended with the driver's `failure` where neither verifier gives it one (a message it couldn't
        read): set on both. And an accepted session's record, the Lean's (`record`), held to the books': one that
        differs in any field is `disagreement`."""
        got = self.outcome_json = self._lean({"op": "outcome", "node": self.node, "session": self.k})
        self.caught_up()
        if self.disagreement is not None:
            return
        if failure is not None:
            if self.s.outcome is None:
                self._fail_books(failure)
            if got.get("outcome") is None:
                self._lean({"op": "abort", "node": self.node, "session": self.k, "outcome": str(failure["failure"])})
            return
        mine = {"outcome": self.s.outcome, "record": lv.summary_json(self.s.record()) if self.s.outcome == "accepted"
                else None}
        if got != mine:
            self._disagree(len(self.s.windows) - 1, "record", got, mine)

    def record(self) -> audit.Record:
        """The accepted session's record: the Lean's (its `outcome` at `end`, as `lv.record_of` reads it, with the
        session's bounds, `audit.Session.bounded`)."""
        return self.s.bounded(lv.record_of(self.outcome_json["record"], self.s.registration, self.s.commitment.root,
                                           self.s.params.delta, self.s.params.samples))

    def _fail_books(self, failure: dict) -> None:
        t = failure["t"]
        self.s._fail(t if type(t) is int and 0 <= t < len(self.s.windows) else len(self.s.windows),
                     str(failure["failure"]))


class Verifier:
    def __init__(self, conn: wire.Conn, *, source: bytes, windows: int, window_ns: int, samples: int, delta: float,
                 jitter_ns: int, clock=None, log=print, checks=None, lifetime: int | None = None,
                 registry: audit.Registry | None = None, hbm_bytes: int = HBM_BYTES, host_bytes: int = 0,
                 ingress_bytes_per_s: float = 0.0, lean_exe: Path | None = None,
                 transcript: Path | None = None, disk_bytes_per_s: float = 0.0, link_bytes_per_s: float = 0.0,
                 disk_burst_bytes: float | None = None, link_burst_bytes: float | None = None,
                 challenger: challengers.Local | challengers.Serve | challengers.ChallengerGate | None = None,
                 appends: int | None = None, payload_key: bytes | None = None,
                 binding: rstore.Binding | None = None, groups_per_coin: int = 1) -> None:
        coins.openings_coins(1, groups_per_coin)
        self.groups_per_coin = groups_per_coin
        self.conn, self.source, self.W, self.T, self.lifetime = conn, source, windows, window_ns, lifetime
        self.payload_key = secrets.token_bytes(rstore.KEY_BYTES) if payload_key is None else payload_key
        self.binding = binding
        self.transcript = transcript
        self.challenger, self.appends = challengers.Local() if challenger is None else challenger, appends
        self.hbm_bytes, self.host_bytes, self.regions, self.deferred = hbm_bytes, host_bytes, None, False
        self.appending = True
        self.ingress_bytes_per_s = ingress_bytes_per_s
        self.disk_bytes_per_s, self.link_bytes_per_s = disk_bytes_per_s, link_bytes_per_s
        self.disk_burst_bytes, self.link_burst_bytes = disk_burst_bytes, link_burst_bytes
        self.checks = checks
        self.registry = audit.Registry() if registry is None else registry
        self.samples, self.delta, self.jitter_ns = samples, delta, jitter_ns
        self.clock = clock or RealClock()
        self.log = log
        self.inbox: queue.Queue = queue.Queue()
        self._stamp: int | None = None
        self._check = 0
        self.lean = start_lean(source, self.registry, lean_exe, self._lean_done)
        self.bound = None if binding is None else bind_lean(self.lean, binding)
        threading.Thread(target=self._read, daemon=True).start()

    def _now(self) -> int:
        return self._stamp if self._stamp is not None else self.clock.now()

    def _lean_done(self, msg: dict) -> None:
        self.inbox.put(({"type": "lean-done", "id": msg["done"], "accepted": msg["accepted"]}, b"", self.clock.now()))

    def registry_json(self) -> dict:
        """The verifier's registry, `registry/v0`: the Lean's, but once its process has ended (when every session it
        was in is refused as `disagreement`) audit.Registry's."""
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
        except audit.RegistryError:
            mine = lv.REFUSED
        try:
            got = self.lean.call({"op": "close", "node": node.hex()})
        except lv.LeanError as e:
            got = {"error": str(e)}
        if got == lv.REFUSED or "error" in got:
            raise audit.RegistryError(f"node {node.hex()} has no live registration to close in the Lean's registry"
                                      + ("" if mine == lv.REFUSED else f" (audit.Registry closed it: {mine})"))
        return got if got == mine else {**got, "python": mine}

    def stop(self) -> None:
        self.lean.close()

    def _read(self) -> None:
        try:
            while True:
                self.inbox.put(self.conn.recv())
        except (wire.Closed, OSError) as e:
            self.inbox.put(({"type": "lost", "why": str(e)}, b"", self.clock.now()))

    def expect(self, kind: str):
        while True:
            head, body, stamp = self.inbox.get()
            if head["type"] == kind:
                return head, body, stamp
            if head["type"] == "lost":
                raise wire.Closed(head["why"])
            if head["type"] not in ("root", "append", "openings", "closed", "lean-done", "lean-settled",
                                    "lean-disagree"):
                self.log(f"verifier: skipped a {head['type']} while waiting for {kind}")

    def register(self) -> tuple[audit.Registration, dict]:
        """The payload salt key sent (`payload-key`, the verifier's own), then the hello's registration, drawn through
        the registry: refused, with `bye` and the reason, for a node with a live registration (`audit.RegistryError`,
        raised), with a malformed layout (`regions_of`'s ValueError), deferring residency under a lifetime of one
        window (`audit.Params`'s ValueError), asking for receipts that bind PoUW's tile outputs (`bind`, which the Lean
        covers none of: ValueError), or, with a `binding`, a store that isn't the pinned model's under that key
        (`hold_store`'s Refused), all before the registry draws anything. Both registries draw it
        under one nonce: one on which they differ is refused as `disagreement` (`audit.RegistryError`), and closed
        where it went live."""
        self.conn.send({"type": "payload-key", "key": self.payload_key.hex()})
        hello, _, _ = self.expect("hello")
        try:
            self.regions = regions_of(hello.get("layout"), hello["blocks"])
            self.deferred = audit.Params(1, 1, 1, lifetime=self.lifetime, deferred=hello.get("deferred", False)).deferred
            if type(hello.get("appends", True)) is not bool:
                raise ValueError("the hello's appends says whether the prover's sessions append, true or false")
            self.appending = hello.get("appends", True)
            if self.binding is not None:
                self.regions = hold_store(self.lean, self.bound, self.binding, bytes.fromhex(hello["node"]),
                                          hello["blocks"], bytes.fromhex(hello["payload_root"]), self.payload_key,
                                          self.regions)
        except ValueError as e:
            self.conn.send({"type": "bye", "why": f"refused: {e}"})
            raise
        node, blocks, root = bytes.fromhex(hello["node"]), hello["blocks"], bytes.fromhex(hello["payload_root"])
        c, receipt, rows = wire.codec_named(hello["codec"]), hello["receipt"], hello.get("rows")
        bind = hello.get("bind", False)
        nonce = secrets.token_bytes(audit.NONCE_BYTES)
        try:
            self.registration = self.registry.register(self.source, audit.site_log(node, blocks, root), audit.STORE, c,
                                                       receipt, bind=bind, rows=rows, nonce=nonce)
            mine, why = {"codec_seed": self.registration.codec_seed.hex(), "digest": self.registration.digest.hex()}, None
        except (audit.RegistryError, ValueError) as e:
            mine, why = (lv.REFUSED if isinstance(e, audit.RegistryError) else {"error": True}), e
        try:
            got = self.lean.call(lv.register_request(audit.site_chain(node, blocks, root), c, receipt, rows, nonce, bind))
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

    def appends_of(self) -> int:
        """A session's append coins (`coins.families`): `appends` if given, else `coins.APPENDS`, or none, and no append
        family, when the hello said the prover's sessions append nothing (`prover.BANDWIDTH`).  That says only what
        the prover will forgo: an append it sends then has no coin and refuses the session."""
        if self.appends is not None:
            return self.appends
        return coins.APPENDS if self.appending else 0

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

    def session(self, k: int, allowance_ns: int, not_before_ns: int | None, pace: dict | None = None) -> dict:
        """One session.  With `pace` (`pace_of`), the protocol paces it (`audit.Pace(k, T)`, T its stretch, over which
        γ is declared): its windows are groups of `pace["k"]` passes, and each boundary leaves at
        `Session.next_boundary_ns`: once its group has k − 1 roots (the pass in flight is its last), or for k = 1 once
        the group before has its root (the first group's at once), so the honest prover never waits on a coin; and at
        the latest at max(its coin + (k − 1)·A, the group before's deadline), so its roots are due within about k
        allowances of its coin.  The lifetime cap, unless `lifetime` states it, is the windows of T in 30 s, or paced,
        the groups of their nominal length, the stretch over its groups (`audit.lifetime_windows`)."""
        reg, K = self.registration, None if pace is None else pace["k"]
        W = self.W if pace is None else pace["groups"] * (pace["windows"] + 1)
        samples, window_ns = (self.samples, self.T) if pace is None else (pace["draws"], 1)
        life = self.lifetime if self.lifetime is not None else audit.lifetime_windows(
            self.T if pace is None else round(pace["t_ms"] * 1e6) // pace["groups"], self.deferred)
        params = audit.Params(reg.blocks, allowance_ns, W, samples, self.delta, window_ns=window_ns, codec=reg.codec,
                              receipt=reg.receipt, rows=reg.rows, lifetime=life, deferred=self.deferred,
                              pace=None if pace is None else audit.Pace(K, round(pace["t_ms"] * 1e6)),
                              groups_per_coin=self.groups_per_coin)
        not_before_ns = self.registered_ns if not_before_ns is None else not_before_ns
        one = {} if pace is None or pace.get("in_flight", 2) == 2 else {"in_flight": pace["in_flight"]}
        if one:
            params = dataclasses.replace(params, pace=dataclasses.replace(params.pace, **one))
        g = params.groups_per_coin
        shared, per = ({}, {}) if g == 1 else ({"groups_per_coin": g}, {"per_coin": g})
        appends = self.appends_of()
        books = self.registry.session(params, reg, None, clock=self._now, not_before_ns=not_before_ns, index=k,
                                      challenger=lambda e: self.challenger.open(reg.digest, e, W, appends, **per))
        case = None if self.transcript is None else lv.DifftestCase(
            self.transcript / f"{reg.digest.hex()[:16]}-session-{k}.json", reg, params, not_before_ns)
        session = Pair(self, books, k, case)
        self.conn.send({"type": "session", "k": k, "windows": W, "lifetime": params.lifetime, "pace": K} | one | shared)
        self.expect("armed")
        while self.clock.now() < not_before_ns:
            item = self.clock.wait(self.inbox, not_before_ns)
            if item is not None:
                self.log(f"verifier: skipped a {item[0]['type']} before the session's first coin")
        sent = drawn = 0
        arrivals: list[list[int]] = [[] for _ in range(W)]
        verdicts: dict[int, dict] = {}
        lean_ready: dict[int, tuple[int | dict, int, bool]] = {}
        lean_checks: dict[int, int] = {}
        settling: set[int] = set()
        frames: dict[int, int] = {}
        draws_at: dict[int, int] = {}
        failure = None
        cpu0 = time.process_time_ns()

        def failed(t: int) -> dict:
            v = session.failed
            if v is None:
                return {"t": t, "failure": "refused", "at": None}
            return {"t": v.window, "failure": v.failure, "at": v.at}

        def decide(t: int) -> None:
            """Window t's verdict once the Lean has checked the window (`settled`). The Lean settles
            a check that accepted behind the events sent so far, the books taking its verdict when its answer comes
            (`lean-settled`): an accepted window is nothing any event reads. One that rejected is waited on, so
            neither session moves on a rejection the other hasn't had. Nothing once the session has a failure."""
            if failure is not None or t in verdicts or t in settling or t not in lean_ready:
                return
            check_id, latency, accepted = lean_ready.pop(t)
            if type(check_id) is not int:
                settled(t, check_id, latency)
            elif accepted:
                settling.add(t)
                session.settle(check_id, lambda res, t=t, latency=latency: self.inbox.put(
                    ({"type": "lean-settled", "k": k, "t": t, "res": res, "latency": latency}, b"", self.clock.now())))
            else:
                settled(t, session.settle(check_id), latency)

        def settled(t: int, res: dict, latency: int) -> None:
            """Window t's verdict: the Lean's settled result (`res`) set on the books (`Pair.judge`); an event before it
            that the two answered differently refuses the session first."""
            nonlocal failure
            if (v := session.caught_up()) is not None:
                failure = {"t": v.window, "failure": v.failure, "at": v.at}
                return
            verdicts[t] = {"t": t, **session.judge(t, res), "openings": res.get("openings"), "checked_ns": latency,
                           "lean_ns": res.get("ns")}
            session.transcribe(t)
            if not verdicts[t]["accepted"]:
                failure = {"t": t, "failure": verdicts[t]["failure"], "at": verdicts[t]["at"]}

        def lean_done(head: dict, stamp: int) -> None:
            """A Lean check that ended, ready to settle (`decide`); a process that ended leaves every check running
            an error."""
            ids = [head["id"]] if head["id"] in lean_checks else list(lean_checks) if head["id"] is None else []
            for check_id in ids:
                t = lean_checks.pop(check_id)
                ready = (lv.REFUSED if "refused" in head else check_id) if head["id"] is not None else \
                    {"error": "the process ended"}
                lean_ready[t] = (ready, stamp - draws_at[t], head["accepted"])
                decide(t)

        def close(t: int) -> None:
            """Window t's draw, at its draw's last window (`Params.draw_of`; every window at g = 1): each window's
            share goes to the prover, and one whose share is none is accepted as the close accepted it."""
            nonlocal failure
            group = params.draw_of(t)
            if t != group[-1]:
                return
            try:
                out = session.close(t)
            except audit.SessionError:
                failure = failed(t)
                return
            if isinstance(out, audit.Verdict):
                failure = {"t": out.window, "failure": out.failure, "at": None}
                return
            now = self.clock.now()
            for u in group:
                share = session.windows[u].draws
                draws_at[u] = now
                self.conn.send({"type": "draws", "t": u, "draws": [list(d) for d in share]})
                if not share:
                    verdicts[u] = {"t": u, "accepted": True, "failure": None, "at": None, "openings": 0,
                                   "checked_ns": 0}

        def bound() -> None:
            """The next window's coin, or `close` after the last; where the session refuses it, `failure`."""
            nonlocal sent, failure
            try:
                coin = session.boundary()
            except audit.SessionError:
                failure = failed(sent)
                return
            sent += 1
            self.conn.send({"type": "coin", "coin": wire.coin_json(coin)} if coin else {"type": "close"})

        try:
            self.conn.send({"type": "coin", "coin": wire.coin_json(session.start())})
            self.clock.busy()
        except audit.SessionError:
            failure = failed(0)
        while failure is None and len(verdicts) < W:
            if sent < W:
                until = session.next_boundary_ns()
            elif drawn < W:
                until = session.deadline(W - 1) + 1
            else:
                until = None
            item = self.clock.wait(self.inbox, until)
            if item is None:
                if sent < W:
                    bound()
                    while failure is None and K is not None and drawn < sent and len(session.windows[drawn].roots) >= K:
                        close(drawn)
                        drawn += 1
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
                while failure is None and (drawn < t or (drawn == t and K is not None and i + 1 >= K and sent > t)):
                    due = session.next_boundary_ns() if drawn == t and sent + 1 < W else None
                    with (session.coins.ahead(coins.CHALLENGE, sent + 1, bytes.fromhex(head["root"]))
                          if due is not None and due <= self.clock.now() else contextlib.nullcontext()):
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
                session.openings(t, body)
                frames[t] = frames.get(t, 0) + 1
                if frames[t] == head["parts"]:
                    self._check += 1
                    lean_checks[self._check] = t
                    session.verify(t, self._check, lambda got, c=self._check: self.inbox.put(
                        ({"type": "lean-done", "id": c, "accepted": False, "refused": got}, b"", self.clock.now())))
            elif kind == "lean-done":
                lean_done(head, stamp)
            elif kind == "lean-settled" and head["k"] == k:
                settling.discard(head["t"])
                if failure is None:
                    settled(head["t"], head["res"], head["latency"])
            elif kind == "lean-disagree" and head["k"] == k:
                if (v := session.caught_up()) is not None:
                    failure = {"t": v.window, "failure": v.failure, "at": v.at}
            elif kind == "closed":
                self.clock.idle()
                session.passes_ended()
            elif kind == "lost":
                raise wire.Closed(head["why"])
        lost = None
        while lean_checks or settling:
            item = self.inbox.get()
            head, _, stamp = item
            if head["type"] == "lean-done":
                lean_done(head, stamp)
            elif head["type"] == "lean-settled" and head["k"] == k:
                settling.discard(head["t"])
            elif head["type"] == "lost":
                lost = item
        if lost is not None:
            self.inbox.put(lost)
        coin_check = self.challenger.finish(session.s)
        if coin_check["reason"] is not None:
            self.log(f"verifier: session {k}'s coins are not the challenger's as it signed them: {coin_check['reason']}")
            if failure is None:
                failure = {"t": max(0, len(session.windows) - 1), "failure": "coins", "at": None}
        session.end(failure)
        for check_id, *_ in lean_ready.values():
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
        commitment, opened = session.disclose()
        self.conn.send({"type": "coins", "k": k, "commitment": commitment.to_json()}, wire.openings_json(opened))
        out = shared | {"k": k, "accepted": accepted, "failure": failure, "allowance_ns": allowance_ns,
               "samples": samples, "delta": self.delta, "window_ns": window_ns, "windows": W, "pace": pace,
               "lifetime": params.lifetime,
               "registration": reg.digest.hex(), "coins": session.commitment.root.hex(),
               "appended": session.extent - reg.blocks, "released": session.released, "append_coins": appends,
               "verdict": "lean", "disagreement": session.disagreement, "challenger": coin_check,
               "transcript": None if case is None else case.close(lv.released_coins(session.coins)),
               "verify_latency_ns": {t: v.get("checked_ns") for t, v in sorted(verdicts.items())},
               "lean_check_ns": {t: v.get("lean_ns") for t, v in sorted(verdicts.items())},
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
            if session.outcome_json is not None:
                out["lean_record"] = session.outcome_json["record"]
            out["end_ns"] = record.end_ns
            out["ledger"] = self.ledgers(record, [list(w.arrived_ns) for w in record.windows],
                                         [w.plans for w in session.windows])
            if pace is not None:
                rebuild = audit.Rebuild(record.codec, timing.SQUARINGS_PER_S[record.codec.field.name])
                try:
                    out["ledger"]["paced"] = paced_line(
                        out["ledger"]["memory"], out["windows_seen"], groups=pace["groups"],
                        wrong=audit.pooled(record)[0], blocks=reg.blocks,
                        rebuild_per_ns=line.number(rebuild.squarings_per_s) / record.codec.encode_squarings / line.NS,
                        chip_bytes=timing.CHIP_BYTES, block_bytes=record.block_bytes,
                        node_bytes=self.hbm_bytes + self.host_bytes,
                        burst_bytes=ingress_bursts(self.ingress_bytes_per_s),
                        ingress_per_ns=line.number(self.ingress_bytes_per_s) / record.block_bytes / line.NS)
                except ValueError as e:
                    out["ledger"]["paced"] = {"refused": str(e)}
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
        """The session's declared outputs (`declared`), the protocol's (`ledger`: γ against the node's R at spec and
        measured, per window as declared, and the memory line's forms), on the verifier's node (`ledger.Node.stated`:
        every GPU's HBM and the host's DRAM, the node's S_chip and squaring ceiling, `--disk-bytes-per-s` and
        `--link-bytes-per-s`, and their bursts), whatever share of the node the store is on.  Both R are the verifier's, which the
        prover's calibration can only raise: a prover reporting a slower read would shrink every term but the timing.
        Each reading (`spec`, `measured`) also carries the session's terms (`audit.Ledger`) and its
        certified bytes, `(1 − γ)·R·Σ`, the same in both readings: they are what a session over part of the node
        establishes, whose γ is the node's only once its store spans the node; and its timing term's parts
        (`timing_parts`) when the calibration has a bare read of each tier covering the registered store and a window
        had two roots.  A reading the ledger refuses carries why, and the session then declares nothing.  With the
        session's plans, `memory` is the node's HBM it shows held by kind (`memory`), KV credited on no verdict yet.
        `delta` is the memory line's δ in its six parts (`ledger.delta_parts`), whose sum `declared`'s forms carry.
        Each window's bound on wrong receipts is computed once for both readings, exactly, in the check workers when
        there are some (`checks.Checks.bounds`).  `inputs` carries the AuditRO slack γ and δ were charged at
        (`slack_log2`); a record without it was charged `UNRECORDED_SLACK_LOG2`."""
        cal = self.calibration
        reported = float(cal["read_bytes_per_s"])
        stated = ledger.Node.stated(reported_read_bytes_per_s=reported, disk_bytes_per_s=self.disk_bytes_per_s,
                                    link_bytes_per_s=self.link_bytes_per_s, disk_burst_bytes=self.disk_burst_bytes,
                                    link_burst_bytes=self.link_burst_bytes)
        spec, measured = stated.spec_bytes_per_s, stated.read_bytes_per_s
        rebuild = stated.rebuild(record.codec)
        node = {"gpus": timing.GPUS, "gpu_spec_bytes_per_s": timing.GPU_SPEC_BYTES_PER_S,
                "gpu_read_bytes_per_s": timing.GPU_READ_BYTES_PER_S, "host_spec_bytes_per_s": timing.HOST_SPEC_BYTES_PER_S,
                "host_read_bytes_per_s": timing.HOST_READ_BYTES_PER_S, "gpu_chip_bytes": timing.GPU_CHIP_BYTES,
                "host_chip_bytes": timing.HOST_CHIP_BYTES}
        out = {"inputs": {"spec_bytes_per_s": spec, "measured_read_bytes_per_s": measured,
                          "reported_read_bytes_per_s": reported, "node": node,
                          "disk_bytes_per_s": stated.disk_bytes_per_s, "link_bytes_per_s": stated.link_bytes_per_s,
                          "rebuild_burst_bytes": stated.rebuild_burst_bytes,
                          "disk_burst_bytes": stated.disk_burst_bytes, "link_burst_bytes": stated.link_burst_bytes,
                          "chip_bytes": stated.chip_bytes, "encode_bytes_per_s": cal["encode_bytes_per_s"],
                          "squarings_per_s": rebuild.squarings_per_s, "rows": record.rows,
                          "span_ns": record.span_ns, "receipts": record.receipts, "passes": record.passes,
                          "slack_log2": audit.SLACK_LOG2, "reads_bytes": record.receipts * record.block_bytes,
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
        readings: dict[str, ledger.Gamma] = {}
        for name, r in (("spec", spec), ("measured", measured)):
            try:
                g = readings[name] = ledger.gamma(record, stated, r, wrong=lambda p: by[id(p)])
            except ValueError as e:
                out[name] = {"capacity_bytes_per_s": r, "refused": str(e)}
                continue
            whole = r * record.span_ns / 1e9
            out[name] = {"capacity_bytes_per_s": r, **g.ledger.to_json(), "certified_bytes": (1 - g.gamma) * whole,
                         "certified_bytes_horizon": (1 - g.gamma_horizon) * whole}
            if parts:
                out[name]["timing_parts"] = timing_parts(capacity_bytes_per_s=r, **parts)
        out["delta"] = ledger.delta_parts(record, stated)
        out["declared"] = (ledger.Declared(len(record.windows), readings["spec"], readings["measured"],
                                           ledger.memory(record, stated)).to_json()
                           if len(readings) == 2 else {"refused": next(out[n]["refused"] for n in ("spec", "measured")
                                                                       if n not in readings)})
        # SketchBound's δ: every compression the prover could make over the span a guess or a key, Q = F·C.
        compressions, outside = ledger.compressions(record, stated), stated.sketch_outside
        out["sketch_failure"] = {"compressions": compressions, "outside": outside,
                                 "log2": timing.log2(timing.sketch_failure(compressions, record.passes, record.rows,
                                                                           blocks=record.blocks, outside=outside))}
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
    ap.add_argument("--lifetime", type=int,
                    help="L: windows (a paced session's groups) an appended block may stay in plans after the window "
                         "it joined (default: those of the window, or of the nominal group, in 30 s, "
                         "`audit.lifetime_windows`)")
    ap.add_argument("--pings", type=int, default=1000)
    ap.add_argument("--check-workers", type=int, default=len(os.sched_getaffinity(0)) - 1,
                    help="processes computing each window's exact bound on wrong receipts (`checks.Checks.bounds`; 0: "
                         "this process)")
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
    ap.add_argument("--disk-bytes-per-s", type=float, default=0.0,
                    help="the node's disks' read ceiling, charged in the memory line (`ledger.Node`)")
    ap.add_argument("--link-bytes-per-s", type=float, default=0.0,
                    help="the node's boundary links' read ceiling, charged in the link-charged memory line")
    ap.add_argument("--disk-burst-bytes", type=float,
                    help="the most the disks bring back at once beyond their ceiling (default: a second of it)")
    ap.add_argument("--link-burst-bytes", type=float,
                    help="one link window, charged with the links (default: `timing.LINK_WINDOW_BYTES`)")
    ap.add_argument("--close", metavar="NODE_HEX",
                    help="close the node's live registration in --registry, its sessions counted as they stood, "
                         "and exit")
    ap.add_argument("--lean-exe", type=Path,
                    help="a built remote-audit-verifier (default: `lake build` it in its package)")
    ap.add_argument("--transcript", type=Path, metavar="DIR",
                    help="write each session's difftest case to DIR (`lean_verifier.Transcript`: about 0.6 GB a "
                         "window at 33,273 openings), its path, bytes and sha256 in verifier.json")
    ap.add_argument("--paced", nargs="+", metavar="K:T_MS:DRAWS:WINDOWS[:F]",
                    help="pace each session's passes, one spec a session (`pace_of`; replaces --sessions, --windows, "
                         "--window-ms and --samples): groups of K passes under a live coin each (`audit.Pace`), "
                         "γ per fixed T_MS, and F windows in flight (2, the default, or 1: each coin after the root "
                         "before it)")
    ap.add_argument("--pace-pass-ms", type=float, default=17.0,
                    help="the pass the paced sessions' groups a stretch are counted at (0: the calibration's median "
                         "pass interval)")
    ap.add_argument("--ingress-bytes-per-s", type=float, default=0.0,
                    help="the read ceiling of every route into node memory but a rebuild, which the paced line "
                         "charges (`paced_line`): the node's disks, and its boundary links to drop Identity")
    ap.add_argument("--challenger", default="local", choices=challengers.KINDS,
                    help="where each session's coins come from (`challengers`): a challenger in this process (local, "
                         "the default), serve.py started for the session on this machine (serve), or the auditor "
                         "host's challenger through its gate (gate), its out/ under OUT/challenger and its record "
                         "and release log checked when the session ends")
    ap.add_argument("--challenger-key", type=Path, metavar="DIR",
                    help="serve's LMS key directory, outside OUT (made there on first use)")
    ap.add_argument("--challenger-params", metavar="W,H",
                    help="serve's LMS width and height for a key it makes (default lms.DEPLOYED's)")
    ap.add_argument("--challenger-ssh", metavar="WORDS",
                    help="gate: ssh's options and destination for the auditor host, the lane's gate key among them "
                         "(-i KEY research@HOST)")
    ap.add_argument("--challenger-peer", metavar="NAME", help="gate: this machine's name in the host's /etc/vy/peers")
    ap.add_argument("--challenger-pin", metavar="SHA512", help="gate: the SHA-512 of the challenger's public key")
    ap.add_argument("--challenger-period", metavar="P",
                    help="gate: the audit's period, open on the ledger (period-open) before the first session")
    ap.add_argument("--challenger-main", type=Path, metavar="FILE",
                    help="gate: the auditor's own first-parent list of main, a commit a line (`git rev-list "
                         "--first-parent origin/main`): a session whose host ran a tree its pins log doesn't record on "
                         "it is refused as coins")
    ap.add_argument("--challenger-minutes", type=int, default=120, metavar="M",
                    help="gate: each session's reserve, the most it serves")
    ap.add_argument("--appends", type=int, metavar="A",
                    help=f"each session's append coins, the most appends it takes (`coins.families`): by default "
                         f"{coins.APPENDS}, or none when the prover's hello says its sessions append nothing")
    ap.add_argument("--groups-per-coin", type=int, default=1, metavar="G",
                    help="each session's windows drawn on one openings coin (`audit.Params.groups_per_coin`): its "
                         "draw's last window's close draws them together, ⌈W/G⌉ openings coins in all, and the line "
                         "of record's pooled bound reads its draws (`line.draws_of`); none is at one window in "
                         "flight (a `--paced` F = 1), which no property covers (`line.SHARED`)")
    ap.add_argument("--memory-of", type=Path, metavar="VERIFIER_JSON",
                    help="print each accepted session's memory line in both forms from a verifier.json written "
                         "before the average form (`recorded_memory`, with --host-bytes the node line), and a paced "
                         "session's paced line with --ingress-bytes-per-s (`recorded_paced`), and exit")
    ap.add_argument("--model", metavar="ROLE",
                    help="bind the registered store to this pinned model (`remote_bandwidth.store.Binding`): a hello whose "
                         "payload root isn't the root of the store the pinned checkpoint lays out, under the payload "
                         "key this verifier sends, is refused before the registry draws anything")
    ap.add_argument("--checkpoint", type=Path, metavar="DIR",
                    help="--model's checkpoint on this machine, every file the pin names held to its SHA-512 first")
    ap.add_argument("--pin", type=Path, metavar="FILE",
                    help="--model's pin: a local JSON in the auditor's model registry's shape ({name, checkpoint: "
                         "{file: SHA-512 hex}, checkpoint_sha512}), the registry's models.jsonl and its lines "
                         "of ROLE, or core's models.json ({ROLE: {checkpoint_sha512, files, ...}}) and its entry "
                         "of ROLE")
    ap.add_argument("--engines", type=int, default=timing.GPUS,
                    help="--model's store: its arenas, an engine's at one GPU each, in GPU order")
    ap.add_argument("--arena-blocks", type=int, metavar="A",
                    help="--model's store: each arena's blocks, its weights then zero filler; the store's blocks "
                         "past the arenas are host filler")
    ap.add_argument("--layout-workers", type=int, default=len(os.sched_getaffinity(0)),
                    help="processes hashing the pinned files and recomputing the expected payload root, and the Lean "
                         "verifier's threads recomputing it at the same time")
    a = ap.parse_args()
    if a.memory_of is not None:
        v = json.loads(a.memory_of.read_text())
        print(json.dumps([{"k": s["k"], "memory": recorded_memory(s, host_bytes=a.host_bytes)} | (
            {"paced": recorded_paced(s, v["registration"], host_bytes=a.host_bytes,
                                     ingress_bytes_per_s=a.ingress_bytes_per_s)} if s.get("pace") else {})
                          for s in v["sessions"] if s["accepted"]], indent=1))
        return 0
    registry = load_registry(a.registry)
    if a.close is not None:
        if a.registry is None:
            ap.error("--close needs --registry")
        node = bytes.fromhex(a.close)
        lean = start_lean(os.urandom(32), registry, a.lean_exe, lambda m: None)
        closed = registry.close(node).to_json()
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
    if a.appends is not None and a.appends < 0:
        ap.error("--appends is a count, 0 or more")
    if a.groups_per_coin > 1 and any(pace_of(x, 1.0).get("in_flight") == 1 for x in a.paced or ()):
        ap.error("--groups-per-coin above 1 takes no --paced session at one window in flight (F = 1): the Lean "
                 "verifier refuses one, as no property covers it; drop :1 or --groups-per-coin")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.challenger == "gate" and a.port in challengers.GATE_PORTS:
        ap.error(f"--port {a.port} is one of the gate's session ports ({challengers.GATE_PORTS.start}.."
                 f"{challengers.GATE_PORTS.stop - 1}), which this machine's certifier coin link binds")
    try:
        challenger = challengers.of(a.challenger, a.out / "challenger", a.challenger_key, a.challenger_params,
                                    ssh=a.challenger_ssh, peer=a.challenger_peer, pin=a.challenger_pin,
                                    period=a.challenger_period, main=a.challenger_main, minutes=a.challenger_minutes)
    except ValueError as e:
        ap.error(str(e))
    binding = None
    if a.model is not None:
        if a.checkpoint is None or a.pin is None or a.arena_blocks is None:
            ap.error("--model needs --checkpoint DIR, --pin FILE and --arena-blocks A")
        try:
            binding = rstore.Binding(a.model, a.checkpoint, a.pin, a.engines, a.arena_blocks,
                                     workers=a.layout_workers, log=lambda m: print(m, flush=True))
        except rstore.Refused as e:
            print(f"verifier: refused the pinned model: {e}", flush=True)
            (a.out / "verifier.json").write_text(json.dumps(
                {"schema": "verity/remote-session/verifier/v0", "verdict": "lean", "refused": str(e),
                 "binding": {"name": a.model, "pin": str(a.pin), "checkpoint": str(a.checkpoint), "bound": False}},
                indent=1))
            return 3
    elif a.checkpoint is not None or a.pin is not None or a.arena_blocks is not None:
        ap.error("--checkpoint, --pin and --arena-blocks bind the store to --model ROLE")
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
                 hbm_bytes=a.hbm_bytes, host_bytes=a.host_bytes, ingress_bytes_per_s=a.ingress_bytes_per_s,
                 lean_exe=a.lean_exe, transcript=a.transcript, disk_bytes_per_s=a.disk_bytes_per_s,
                 link_bytes_per_s=a.link_bytes_per_s, disk_burst_bytes=a.disk_burst_bytes,
                 link_burst_bytes=a.link_burst_bytes,
                 challenger=challenger, appends=a.appends, binding=binding, groups_per_coin=a.groups_per_coin)
    c0 = time.process_time_ns()
    try:
        reg, cal = v.register()
    except (audit.RegistryError, ValueError) as e:
        print(f"verifier: refused the registration: {e}", flush=True)
        kept = v.registry_json()
        save_registry(a.registry, kept)
        (a.out / "verifier.json").write_text(json.dumps(
            {"schema": "verity/remote-session/verifier/v0", "peer": list(peer), "verdict": "lean",
             "refused": str(e), "registry": kept} | ({"binding": binding.record()} if binding else {}), indent=1))
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
             "verdict": "lean", "calibration": cal, "rtt": rtt, "allowance_ns": allowance, "sessions": results,
             "process_cpu_ns": time.process_time_ns() - c0, "wire": conn.totals(), "registry": kept}
            | ({"binding": binding.record()} if binding else {}), indent=1))
        save_registry(a.registry, kept)

    record()
    pass_ms = a.pace_pass_ms or cal["pass_ns"]["p50"] / 1e6
    paces = [pace_of(x, pass_ms) for x in a.paced] if a.paced else None
    try:
        for k in range(len(paces) if paces else a.sessions):
            r = v.session(k, allowance, not_before, paces[k] if paces else None)
            results.append(r)
            not_before = r.get("end_ns", v.clock.now())
            lg, pub = r.get("ledger", {}).get("spec", {}), r.get("record", {}).get("public", {})
            mem = r.get("ledger", {}).get("memory", {})
            lines = r.get("ledger", {}).get("declared", {}).get("memory", {})
            if (pl := r.get("ledger", {}).get("paced")) is not None:
                line = f"refused: {pl['refused']}" if "refused" in pl else \
                    f"snapshot {pl['snapshot']['share']:.4%} time {pl['time']['share']:.4%}"
                print(f"verifier: session {k} paced {json.dumps(r['pace'])}: {line} "
                      f"gamma horizon {lg.get('horizon', {}).get('gamma')}", flush=True)
            passes = [w["passes"] for w in r["windows_seen"]]
            print(f"verifier: session {k}: accepted={r['accepted']} failure={r['failure']} "
                  f"passes={passes if r.get('pace') is None else sum(passes)} gamma={lg.get('gamma')} "
                  f"appended={r['appended']} released={r['released']} post_seed={pub.get('post_seed')} "
                  f"memory share={mem.get('share')} residual={mem.get('residual')} epsilon={mem.get('epsilon')} "
                  f"average share={mem.get('average', {}).get('share')} "
                  f"node={json.dumps(mem['node']['share']) if 'node' in mem else None} "
                  f"node average={json.dumps(mem['node']['average']['share']) if 'node' in mem else None} "
                  f"declared={json.dumps({f: x.get('epsilon', x.get('refused')) for f, x in lines.items()})}",
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
