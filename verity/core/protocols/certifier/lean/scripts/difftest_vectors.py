"""The difftest's data: random instances, and the Python audit's results on each, for `NetTimingDifftest.lean`.

    python scripts/difftest_vectors.py <verity checkout> --out DIR [--count N]

Writes `DIR/difftest/instances.jsonl`, one instance per line as `NetTiming.Difftest.rawOfJson` reads it, and
`DIR/difftest/python.jsonl`, the same line's results from the checkout's `verity.core.protocols.certifier`:
`schedule.expected_grids` of each egress link (`rows`), `Audit.check_link`'s verdicts (`verdicts`) and the first bucket
of each recorded window whose row differs from the expected one (`diff`), `Audit.check_ingress_link`'s verdict for
every boundary link-window (`status`), and whether the audit accepts every link-window of the first `m` windows
(`accepted`) and every status there (`complete`), for `m = 0 … W`. The Lean audit runs `NetTimingDifftest.lean` on
both files (`runs` in `lean-audit.json`, which pins their SHA-256), and `verity/core/protocols/certifier/tests/
test_network_certifier_lean_difftest.py` checks that this script still writes them.

An instance has one to three egress links and up to two ingress links, varied `T`, `W`, allocation, `Σ_sync` (δ, and ρ
fractional or not), anchors (some past the horizon) and indices, sessions out of anchor order, bursts beyond the
allocation, late frames, a sender on another clock, a malformed frame, grids forged slot by slot, statuses, a sync
outside `Σ_sync` or other than the unit's, and runs whose records stop at a window boundary. It has no negative index or
anchor (Lean's are ℕ), no row of another length than the allocation, and exactly one record per link-window recorded
(Lean's observation has one status per link and step): the audit's own tests cover duplicates and missing records.
"""
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path

import verity.core.protocols.certifier
from verity.core.protocols.certifier.audit import Audit
from verity.core.protocols.certifier.grid import GridParams, Reason, BlockGrid, Status, Certifier
from verity.core.protocols.certifier.schedule import (ClockSync, LogicalFrame, SyncSet, expected_grids, handing_buckets,
                                            run_shaper)

COUNT = 10_000
RHOS = [Fraction(1), Fraction(2), Fraction(3), Fraction(1, 2), Fraction(1, 3), Fraction(2, 3), Fraction(3, 2),
        Fraction(5, 2), Fraction(3, 4)]
MODES = ["honest", "late", "clock", "forged", "malformed"]


def _sync_outside(rng: random.Random, ss: SyncSet) -> ClockSync:
    if rng.random() < 0.5:
        return ClockSync(max(ss.deltas) + 1 + rng.randrange(3), rng.choice(ss.rhos))
    return ClockSync(rng.choice(ss.deltas), rng.choice([r for r in RHOS + [Fraction(7, 5)] if r not in ss.rhos]))


def _frames(rng: random.Random, horizon: int, buckets: int, allocation: int, size: int) -> list[LogicalFrame]:
    """Sessions of frames, each at one anchor (some past the horizon), with indices 0, 1, … or drawn, some bursts beyond
    the allocation, and in anchor order or not."""
    sessions = []
    for _ in range(rng.randrange(0, 2 + horizon)):
        anchor = rng.randrange(horizon) if rng.random() < 0.85 else rng.randrange(horizon, horizon + buckets + 1)
        n = rng.randint(1, 3 * allocation if rng.random() < 0.2 else 3)
        indices = list(range(n)) if rng.random() < 0.8 else [rng.randrange(8) for _ in range(n)]
        sessions.append([(anchor, i) for i in indices])
    if rng.random() < 0.7:
        sessions.sort(key=lambda s: s[0][0])
    return [LogicalFrame(rng.randbytes(size), a, i) for s in sessions for a, i in s]


def _forge(rng: random.Random, record: BlockGrid, payloads: list[bytes], allocation: int) -> BlockGrid:
    grid = [list(row) for row in record.grid]
    b, s = rng.randrange(len(grid)), rng.randrange(allocation)
    kind = rng.randrange(4)
    if kind == 0:  # another frame's content in a slot
        grid[b][s] = rng.choice(payloads) if payloads else b"\x00"
    elif kind == 1:  # one frame more than the certifier released: the first padding slot filled, or a full row's slot
        free = [i for i, x in enumerate(grid[b]) if x is None]
        grid[b][free[0] if free else s] = rng.choice(payloads) if payloads else b"\x00"
    elif kind == 2:  # a frame withheld
        grid[b][s] = None
    elif len(grid) > 1:  # two buckets swapped
        c = rng.randrange(len(grid))
        grid[b], grid[c] = grid[c], grid[b]
    return BlockGrid(record.link, record.window, tuple(tuple(r) for r in grid), record.status)


def instance(seed: int) -> dict:
    """Instance `seed`: its parameters, each egress link's frames and declared syncs (link 0's are the unit's), and
    every record of the run, from `run_shaper` and then late frames, another clock, a malformed frame, a forged grid, a
    status or a truncation."""
    rng = random.Random(seed)
    buckets, windows, allocation, size = rng.randint(1, 5), rng.randint(1, 4), rng.randint(1, 3), rng.randint(1, 2)
    horizon = buckets * windows
    params = GridParams(buckets=buckets, allocation=allocation, frame_bytes=size,
                        queue_capacity=rng.choice([1 << 16] * 9 + [2]))
    ss = SyncSet(tuple(rng.sample(range(4), rng.randint(1, 3))), tuple(rng.sample(RHOS, rng.randint(1, 3))))
    unit = [rng.choice(list(ss)) if rng.random() < 0.9 else _sync_outside(rng, ss) for _ in range(windows)]
    egress = [f"e{i}" for i in range(rng.randint(1, 3))]
    ingress = [f"i{i}" for i in range(rng.randint(0, 2))]
    frames, declared, records = {}, {}, []
    for link in egress:
        fs = _frames(rng, horizon, buckets, allocation, size)
        syncs = list(unit)
        if link != egress[0] and rng.random() < 0.15:
            syncs[rng.randrange(windows)] = rng.choice(list(ss)) if rng.random() < 0.6 else _sync_outside(rng, ss)
        frames[link], declared[link] = fs, syncs
        mode = rng.choices(MODES, [5, 2, 1, 2, 1])[0]
        handed = handing_buckets(fs, syncs, buckets)
        ready = [h + (rng.randint(1, 3) if mode == "late" and rng.random() < 0.3 else 0) for h in handed]
        clock = [rng.choice(list(ss)) if mode == "clock" and rng.random() < 0.5 else s for s in syncs]
        sent = fs
        if mode == "malformed" and fs:
            k = rng.randrange(len(fs))
            sent = fs[:k] + [LogicalFrame(fs[k].payload + b"\x00", fs[k].anchor, fs[k].index)] + fs[k + 1:]
        rs = run_shaper(sent, ready, clock, Certifier(link, params))
        if mode == "forged":
            w = rng.randrange(windows)
            rs[w] = _forge(rng, rs[w], [f.payload for f in fs], allocation)
        records += rs
    for link in ingress:
        fs = _frames(rng, horizon, buckets, allocation, size)
        records += run_shaper(fs, [0] * len(fs), unit, Certifier(link, params))
    recorded = windows if rng.random() < 0.85 else rng.randrange(windows)
    statuses = [Status(rng.choice(list(Reason))) if rng.random() < 0.04 else None for _ in records]
    records = [BlockGrid(r.link, r.window, r.grid, s or r.status) for r, s in zip(records, statuses, strict=True)
               if r.window < recorded]
    return {"params": params, "sync_set": ss, "windows": windows, "egress": egress, "ingress": ingress,
            "frames": frames, "declared": declared, "records": records, "recorded": recorded}


def _hex(slot: bytes | None) -> str | None:
    return None if slot is None else slot.hex()


def lean_input(inst: dict) -> dict:
    """The instance as `NetTiming.Difftest.rawOfJson` reads it: step `t` is row `t mod T` of each egress link's record
    for window `t / T`, with every boundary link's status for that window."""
    p, links = inst["params"], inst["egress"] + inst["ingress"]
    by = {(r.link, r.window): r for r in inst["records"]}

    def sync(s: ClockSync) -> list[int]:
        return [s.delta, s.rho.numerator, s.rho.denominator]

    return {
        "T": p.buckets, "W": inst["windows"], "S": p.allocation, "K": len(links),
        "deltas": list(inst["sync_set"].deltas), "rhos": [[r.numerator, r.denominator] for r in inst["sync_set"].rhos],
        "frames": [[[f.payload.hex(), f.anchor, f.index] for f in inst["frames"][link]] for link in inst["egress"]],
        "syncs": [[sync(s) for s in inst["declared"][link]] for link in inst["egress"]],
        "obs": [{"rows": [[_hex(x) for x in by[(link, t // p.buckets)].grid[t % p.buckets]] for link in inst["egress"]],
                 "status": [(s := by[(link, t // p.buckets)].status).reason and s.reason.value for link in links]}
                for t in range(inst["recorded"] * p.buckets)]}


def python_results(inst: dict) -> dict:
    """The Python audit's results on the instance, in the shape `NetTiming.Difftest.evaluate` gives them."""
    p, windows, records = inst["params"], inst["windows"], inst["records"]
    audit = Audit()
    rows, verdicts, diffs = [], [], []
    for link in inst["egress"]:
        frames, syncs = inst["frames"][link], inst["declared"][link]
        expected = expected_grids(frames, syncs, p)
        rows.append([[_hex(x) for x in row] for grid in expected for row in grid])
        verdicts.append([v.violation for v in audit.check_link(link, records, frames, syncs, inst["sync_set"], p)])
        by = {r.window: r for r in records if r.link == link}
        diffs.append([next((w * p.buckets + b for b, (a, e) in enumerate(zip(by[w].grid, expected[w], strict=True))
                            if a != e), None) if w in by else None for w in range(windows)])
    status = []
    for link in inst["egress"] + inst["ingress"]:
        checked = (audit if link in inst["ingress"] else Audit()).check_ingress_link(link, records, windows)
        status.append([v.violation for v in checked])
    return {"rows": rows, "verdicts": verdicts, "diff": diffs, "status": status,
            "accepted": [all(v.accepted for v in audit.verdicts if v.window < m) for m in range(windows + 1)],
            "complete": [all(v is None for vs in status for v in vs[:m]) for m in range(windows + 1)]}


def line(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


def write(out: Path, count: int = COUNT) -> dict[str, Path]:
    """Both files for instances 0 … count − 1 under ``out/difftest``."""
    d = out / "difftest"
    d.mkdir(parents=True, exist_ok=True)
    paths = {"difftest/instances.jsonl": d / "instances.jsonl", "difftest/python.jsonl": d / "python.jsonl"}
    with open(paths["difftest/instances.jsonl"], "w") as fi, open(paths["difftest/python.jsonl"], "w") as fp:
        for seed in range(count):
            inst = instance(seed)
            fi.write(line(lean_input(inst)))
            fp.write(line(python_results(inst)))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.core.protocols.certifier is audited")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=COUNT)
    a = ap.parse_args()
    here = Path(verity.core.protocols.certifier.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.core.protocols.certifier comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
