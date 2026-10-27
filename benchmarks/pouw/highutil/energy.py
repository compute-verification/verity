"""The unaccounted share from energy (CMP-231's S8, measure D1, not yet ruled): under a power premise, the hidden useful
MACs that a period's measured energy could still pay for after the floors of the work it declared, over θ · wall:

    U = min(θ · wall − certified MACs, (E − F_declared − F_peels − F_hash − F_forming − P_idle · wall) / e_hidden) / (θ · wall)

- E is the period's energy from a power trace over its pinned window (terms.py's `windows`, trimmed a second at each
  end; the trapezoid mean over the samples times the period's wall). The trace is nvidia-smi's power.draw until an
  auditor-side meter exists, and the report says which (`--source`): the GPU's own sensor is a stand-in, not a premise.
- The floors are per clock and per operand class (`FLOORS`, measured on the H200; CMP-247 replaces them by class as it
  lands): the declared MACs at the declared class (D2's third Prop: the served kernel's own energy), the peels at theirs,
  the compressions the blocks hash (`compressions_per_step`: value roots and coins; the operand roots are built in prep,
  untimed) at SHA-512's own, forming at a measured figure or 0, and idle power over the whole wall, which takes the host
  gaps and tensor-idle forming. A floor set too low only raises U.
- e_hidden is the cheapest useful MAC (2-bit operands, `HIDDEN`), and the cap is tensor time: hidden MACs fit only in
  the θ · wall the certified MACs (declared plus peels) leave.

Every share is reported with its gap to 3% (points over or under), at each reading of the clock (terms.py's), beside the
certified MACs over θ · wall and the declared ones (the time measure's accounted share), which no power premise lifts
past the card's power-bound rate.

    energy.py SUMMARY PERIODS TRACE [--source LABEL] [--declared lt] [--peels rand] [--hidden u2] [--hash-m M]
              [--forming-j-per-step J]      prints one JSON object
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import terms

BAR = 0.03
SENSOR = "GPU sensor (nvidia-smi power.draw), not a premise"
MEASURE = ("U: the hidden useful MACs the measured energy could still pay for after the floors of the declared work, "
           "the peels, the hashing, forming and idle, at the cheapest useful MAC's energy, capped by the tensor time the "
           "certified MACs leave, over θ · wall (CMP-231 D1, not ruled). The certified and declared MACs over θ · wall are "
           "beside it.")

_T1830, _T1980 = 4096 * 132 * 1830e6, 4096 * 132 * 1980e6
_DFA1 = "r20261027-220336-dfa1 (art:8703db86), measured"
#: the H200's floors over idle: watts, joules a MAC by operand class, joules a SHA-512 compression
FLOORS = {"h200-sxm-141gb": {
    "idle_w": (78.0, f"the card idle ({_DFA1})"),
    "mac_j": {
        "zero": ((295 - 78) / _T1830, f"zero operands, the pipe alone at 1,830 MHz: 295 W ({_DFA1}, zero r0 and r1)"),
        "u2": ((494 - 78) / _T1830, f"2-bit operands, the pipe alone at 1,830 MHz: 494 W ({_DFA1}, u2 r0 489 W, r1 498 W)"),
        "rand": ((689 - 78) / 928e12, f"uniform int8 operands, the pipe alone at the cap: 689 W at 928 T MAC/s ({_DFA1}, rand r1)"),
        "lt": ((694 - 78) / (0.564 * _T1980), "the served GEMM, cuBLASLt on 8B's linears at the cap: 694 W at 0.564 of "
               "θ_1980 (r20261027-204829-e27a, plain lt periods hu-01 and hu-04, measured)")},
    "compression_j": ((424 - 78) / 4.751e9, f"SHA-512 alone: 424 W at 4.751 G compressions/s ({_DFA1}, hash alone)")}}
#: the cheapest useful MAC, e_hidden's class
HIDDEN = "u2"
TILE = 128


def vs3(share: float) -> dict:
    """An unaccounted share with its gap to 3%."""
    gap = (share - BAR) * 100
    return {"share": share, "gap_points": gap, "vs_3pct": f"{abs(gap):.2f} {'over' if gap > 0 else 'under'}"}


def compressions_per_step(workload: dict, row: int) -> int:
    """SHA-512 compressions a step hashes inside its blocks (the shape table's count, art:9f8d21ae shapes.py `call`): each
    call's value root (hm96 rows of `row` bytes, ceil((row + 17) / 128) + 9 a leaf and 2 a tree node, a cell's 2 W bytes
    of T words and, served, 4 of O, W = S - 1 served else S) and its coins (the dither bits, a compression a 512 bits of
    each row of A and B; with W > 0 the noise, 16 entries a compression, and the permutation). The operand roots are built
    in prep, outside the periods' windows."""
    m, served = workload["m"], workload.get("served", False)
    total = 0
    for name, n, k in workload["layers"]:
        lp, L = workload["lp"][name], workload["L"]
        S = -(-k // L)
        W = S - 1 if served else S
        lpt = W * (2 * TILE * TILE // row) + (4 * TILE * TILE // row if served else 0)
        leaves = (m // TILE) * (n // TILE) * lpt
        commit = leaves * (math.ceil((row + 17) / 128) + 9) + 2 * (leaves - 1)
        coins = (m + n) * math.ceil(k / 512) + (((m + n) * lp // 16 + k // 16) if W else 0)
        total += commit + coins
    return total


def mean_power(samples: list, lo: datetime, hi: datetime) -> tuple[float, int] | None:
    """The trapezoid mean of the trace's watts over the samples in [lo, hi], and their count; a lone sample is its own."""
    got = [(t, v["w"]) for t, v in samples if lo <= t <= hi and "w" in v]
    if not got:
        return None
    if len(got) == 1:
        return got[0][1], 1
    area = sum((t1 - t0).total_seconds() * (w0 + w1) / 2 for (t0, w0), (t1, w1) in zip(got, got[1:]))
    return area / (got[-1][0] - got[0][0]).total_seconds(), len(got)


def share(energy_j: float, wall_s: float, steps: int, macs: dict, comps: int, floors: dict, theta: float) -> dict:
    """U and its terms for one period (or the run's sum) at one θ."""
    e_hidden = floors["mac_j"][floors["hidden"]][0]
    f = {"idle": floors["idle_w"][0] * wall_s,
         "declared": steps * macs["declared"] * floors["mac_j"][floors["declared"]][0],
         "peels": steps * macs["peels"] * floors["mac_j"][floors["peels"]][0],
         "hash": steps * comps * floors["compression_j"][0],
         "forming": steps * floors["forming_j_per_step"]}
    cap = theta * wall_s
    certified, declared = steps * macs["certified"], steps * macs["declared"]
    left = max(0.0, cap - certified)

    def u_at(declared_j: float) -> tuple[float, float]:
        rest = energy_j - sum(f.values()) + f["declared"] - declared_j
        hidden = max(0.0, rest) / e_hidden
        return min(hidden, left) / cap, hidden / cap

    u, u_energy = u_at(f["declared"])
    pts_w = 100 / (e_hidden * theta)
    return {
        "U": vs3(u), "U_energy_uncapped": vs3(u_energy), "tensor_time_cap": vs3(left / cap),
        "certified_share": certified / cap, "certified_unaccounted": vs3(1 - certified / cap),
        "declared_share": declared / cap, "declared_unaccounted": vs3(1 - declared / cap),
        "watts": {"metered": energy_j / wall_s, **{k: v / wall_s for k, v in f.items()},
                  "unexplained": (energy_j - sum(f.values())) / wall_s},
        "U_points": {k: v / (e_hidden * cap) * 100 for k, v in f.items()},
        "hidden_macs": u_energy * cap,
        "points_a_watt_of_meter_error": pts_w,
        "points_a_percent_of_declared_floor": f["declared"] * 0.01 / (e_hidden * cap) * 100,
        "U_by_declared_class": {c: vs3(u_at(declared * e)[0]) for c, (e, _) in floors["mac_j"].items()}}


def energy(summary: dict, periods: list[dict], samples: list, source: str = SENSOR, declared: str = "lt",
           peels: str = "rand", hidden: str = HIDDEN, hash_m: float | None = None,
           forming_j_per_step: float = 0.0) -> dict:
    card = summary["card"]
    if not card:
        raise SystemExit("energy.py: summary.json has no card (period.py run --card)")
    if card["sku"] not in FLOORS:
        raise SystemExit(f"energy.py: no energy floors for {card['sku']} (FLOORS has {sorted(FLOORS)})")
    floors = {**FLOORS[card["sku"]], "declared": declared, "peels": peels, "hidden": hidden,
              "forming_j_per_step": forming_j_per_step}
    for c in (declared, peels, hidden):
        if c not in floors["mac_j"]:
            raise SystemExit(f"energy.py: no MAC class {c!r} (FLOORS has {sorted(floors['mac_j'])})")
    w = summary["workload"]
    macs = terms.macs_per_step(w)
    honest = [(r, win) for r, win in zip(periods, terms.windows(periods)) if r["arm"] == "proto" and r.get("cheat") == "none"]
    if not honest:
        raise SystemExit("energy.py: no honest protocol period")
    row = honest[0][0]["registration"]["row"]
    if hash_m is None:
        comps, comps_source = compressions_per_step(w, row), f"counted from the workload at row {row} (value roots and coins)"
    else:
        comps, comps_source = round(hash_m * 1e-6 * macs["declared"]), f"--hash-m {hash_m}"
    readings = terms.readings(card)
    out = {"card": card["sku"], "measure": MEASURE, "energy_source": source, "bar": BAR,
           "floors": {"idle_w": {"value": floors["idle_w"][0], "source": floors["idle_w"][1]},
                      "compression_j": {"value": floors["compression_j"][0], "source": floors["compression_j"][1]},
                      "forming_j_per_step": {"value": forming_j_per_step,
                                             "source": "--forming-j-per-step" if forming_j_per_step else
                                             "not measured (CMP-247): 0, the side that raises U"},
                      "mac_j": {c: {"value": e, "source": s} for c, (e, s) in floors["mac_j"].items()},
                      "classes": {"declared": declared, "peels": peels, "hidden": hidden}},
           "macs_per_step": macs,
           "hash": {"compressions_per_step": comps, "M_per_TMAC": comps / macs["declared"] * 1e6, "source": comps_source},
           "periods": [], "run": None}
    tot = {"energy_j": 0.0, "wall_s": 0.0, "steps": 0}
    for r, win in honest:
        row_out = {"period": r["period"], "wall_s": r["wall_s"], "steps": r["steps"], "accepted": r.get("accepted"),
                   "samples": 0, "mean_w": None, "energy_j": None, "readings": None}
        mp = mean_power(samples, win[0] + timedelta(seconds=1), win[1] - timedelta(seconds=1)) if win else None
        if mp:
            e = mp[0] * r["wall_s"]
            row_out.update(samples=mp[1], mean_w=mp[0], energy_j=e,
                           readings={n: share(e, r["wall_s"], r["steps"], macs, comps, floors,
                                              card["tensor"] * card["sms"] * mhz * 1e6) for n, mhz in readings.items()})
            tot["energy_j"] += e
            tot["wall_s"] += r["wall_s"]
            tot["steps"] += r["steps"]
        out["periods"].append(row_out)
    if tot["steps"]:
        out["run"] = {**tot, "mean_w": tot["energy_j"] / tot["wall_s"],
                      "readings": {n: {"clock_mhz": mhz, **share(tot["energy_j"], tot["wall_s"], tot["steps"], macs, comps,
                                                                 floors, card["tensor"] * card["sms"] * mhz * 1e6)}
                                   for n, mhz in readings.items()}}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("summary")
    ap.add_argument("periods", help="periods.jsonl")
    ap.add_argument("trace", help="the power trace: run.sh's smi.csv, or a meter's in its columns (timestamp, power.draw)")
    ap.add_argument("--source", default=SENSOR, help="what the trace is (default: the GPU's own sensor, not a premise)")
    ap.add_argument("--declared", default="lt", help="the declared MACs' floor class (default lt, the served kernel's own)")
    ap.add_argument("--peels", default="rand", help="the peels' floor class (default rand)")
    ap.add_argument("--hidden", default=HIDDEN, help="e_hidden's class (default u2, the cheapest useful MAC)")
    ap.add_argument("--hash-m", type=float, help="M compressions a declared T MAC hashed in the blocks (default: counted)")
    ap.add_argument("--forming-j-per-step", type=float, default=0.0, help="forming's floor a step over idle (default 0)")
    a = ap.parse_args(argv)
    s = json.loads(Path(a.summary).read_text())
    periods = [json.loads(x) for x in Path(a.periods).read_text().splitlines() if x.strip()]
    samples = terms.smi_samples(Path(a.trace).read_text())
    print(json.dumps(energy(s, periods, samples, a.source, a.declared, a.peels, a.hidden, a.hash_m, a.forming_j_per_step),
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
