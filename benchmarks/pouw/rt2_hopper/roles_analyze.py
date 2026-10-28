"""A test on the H200: are the JP*90 rows' wg_spec points block-timed artifacts?

Reads `roles.jsonl` (sm90_clock's `roles`: wg_spec with each warp's loop end and the MMA iterations done when the first
lane warp ended) and `joint.jsonl` (the probe's wg_s8_n256_a1, the tensor alone, and the wgspec/wgU points as timed).
For each roles point and block: which role ended first, the faster role's rate over its own time against its fixed work
over the block time, and the joint phase's exact point (tensor fraction, lane ops a SMSP-cycle) while both ran. Then each
JP*90 row's load at every point, in SMSP-cycles a cycle (the census's coefficients, rows.csv): a load over 1 is a point the
chip ran that the row calls impossible.

Run: `python roles_analyze.py OUT` (OUT holds roles.jsonl and joint.jsonl).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROWS = {  # IGMMA_S8_N256 and the lane op's SMSP-cycles a warp instruction (rows.csv, sm_90)
    "lop3": {"JPA90_0": (62.774, 1.991), "JPA90_1": (121.407, 0.158)},
    "imad": {"JPI90_0": (63.785, 1.960), "JPI90_1": (121.244, 0.165)},
}
MMA_WARPS, LANE_WARPS, IGMMA_ITER = 8, 4, 4   # a block; IGMMA a warp an iteration


def lines(p: Path) -> list[dict]:
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def tensor_alone(joint: list[dict]) -> float:
    """The wgmma alone's cycles an iteration (median block, best rep)."""
    reps = [statistics.median(r["cyc"]) / r["iters"] for r in joint if r.get("op") == "time"
            and r.get("probe") == "wg_s8_n256_a1"]
    return min(reps)


def point(r: dict, it_alone: float) -> dict:
    lane = r["name"].split("_")[1]
    R, iters, nb = r["lane_ops_iter"], r["iters"], r["blocks"]
    first_lane, mfrac, lrate, llo, lane_faster, fast_own, fast_blk = [], [], [], [], 0, [], []
    for b in range(nb):
        e = r["ends"][b * 12:(b + 1) * 12]
        mma, ln = e[:MMA_WARPS], e[MMA_WARPS:]
        at = r["mma_iters_at_lane_end"][b * MMA_WARPS:(b + 1) * MMA_WARPS]
        blk = r["block_cyc"][b]
        t_lane0 = min(ln)
        if all(a >= 0 for a in at):
            lane_faster += 1
            # the joint phase [0, t_lane0]: the tensor's iterations then over the alone's, the lanes' own rate
            mfrac.append(statistics.mean(at) * it_alone / t_lane0)
            lrate.append(R * iters / max(ln))
            llo.append(R * iters / max(ln))
            fast_own.append(R * iters / max(ln))
            fast_blk.append(R * iters / blk)
        else:
            mfrac.append(iters * it_alone / max(mma))
            # lanes still running when the tensor ended: their rate over the whole run bounds the joint phase's from
            # above (the tail alone runs at least as fast), so the loads below are the worst case; from below, the tail
            # at the lane's alone rate (at most 0.5 a SMSP-cycle)
            lrate.append(R * iters / max(ln))
            llo.append((R * iters - 0.5 * (max(ln) - max(mma))) / max(mma))
            fast_own.append(iters * it_alone / max(mma))
            fast_blk.append(iters * it_alone / blk)
        first_lane.append(t_lane0 < min(mma))
    med = statistics.median
    p = {"name": r["name"], "rep": r["rep"], "lane": lane, "R": R, "blocks": nb,
         "lane_faster_blocks": lane_faster, "lane_first_blocks": sum(first_lane),
         "tensor_fraction": med(mfrac), "lane_ops_per_smsp_cycle": med(lrate), "lane_ops_low": med(llo),
         "faster_rate_own": med(fast_own), "faster_rate_block": med(fast_blk),
         "block_cyc_med": med(r["block_cyc"])}
    t = p["tensor_fraction"] / (it_alone / (MMA_WARPS // 4 * IGMMA_ITER))
    p["loads"] = {row: round(ct * t + cl * p["lane_ops_per_smsp_cycle"], 4) for row, (ct, cl) in ROWS[lane].items()}
    return p


def main() -> int:
    out = Path(sys.argv[1])
    joint = lines(out / "joint.jsonl")
    it_alone = tensor_alone(joint)
    print(f"tensor alone: {it_alone:.3f} cycles an iteration (wg_s8_n256_a1)")
    for r in lines(out / "roles.jsonl"):
        if r.get("probe") != "roles":
            continue
        p = point(r, it_alone)
        who = "lane" if p["lane_faster_blocks"] > p["blocks"] / 2 else "tensor"
        print(f"{p['name']} rep {p['rep']}: faster role {who} ({p['lane_faster_blocks']}/{p['blocks']} blocks lane), "
              f"block {p['block_cyc_med']:.0f}; faster role's rate own/block {p['faster_rate_own']:.4f}/"
              f"{p['faster_rate_block']:.4f} ({p['faster_rate_own'] / p['faster_rate_block'] - 1:+.2%}); joint "
              f"phase tensor {p['tensor_fraction']:.4f}, lane {p['lane_ops_low']:.4f}-{p['lane_ops_per_smsp_cycle']:.4f}"
              f" a SMSP-cycle; "
              f"loads {p['loads']}")
    for r in joint:
        if r.get("op") != "time" or r.get("rep") != 0 or not r["probe"].startswith(("wgspec_", "wgU_")):
            continue
        lane = r["probe"].split("_")[1]
        blk = statistics.median(r["cyc"])
        R = int(r["probe"].rsplit("_r", 1)[1]) if "_r" in r["probe"] else 64
        if "trips" in r:
            stop = statistics.median(r["stop"])
            l, tf = statistics.median(r["trips"]) * R / LANE_WARPS / stop, r["iters"] * it_alone / blk
        else:
            l, tf = R * r["iters"] / blk, r["iters"] * it_alone / blk
        t = tf / (it_alone / (MMA_WARPS // 4 * IGMMA_ITER))
        loads = {row: round(ct * t + cl * l, 4) for row, (ct, cl) in ROWS[lane].items()}
        print(f"{r['probe']} (as timed): tensor {tf:.4f}, lane {l:.4f} a SMSP-cycle; loads {loads}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
