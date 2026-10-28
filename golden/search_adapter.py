"""GD-MODEL's J(design) behind GD-SEARCH's model interface (benchmarks/golden_search/search.py `load_model`):

    python search.py run ... --model /path/to/checkout:golden.search_adapter

`INPUTS` is every input of `golden.model` (name -> Input(value, low, high, source, status, uncertain)): a measured
input is certain; an estimate carries the range its source allows. `evaluate(knobs, inputs, point)` maps GD-SEARCH's
knobs (benchmarks/golden_search/space.py) onto a `golden.model` design at the point (t8192: the throughput preset;
t512: the demo's window, a prefill of 64 real tokens in T 512 and 3 decode steps), evaluates J under those input
values, and returns GD-SEARCH's result keys. A knob GD-SEARCH leaves out keeps the model's default; the extra knobs
`scheme`, `form`, `open`, `steps`, `ptile` (PoUW's tile, (R, C)), `kmax` (the block cap's k_log, 27 to 30), `mmax`
(C-Flock's m cap, 35 to 40), `reps` (1 | 2), `device` ("h200" | "sm120"), `warm` (the prover kept across sessions,
default True), `period` (windows certified together), `retain` ("device" | "host"), `akeys` (keys an attention
block), `census` (None | "b4" | "b4-buckets", the t512 point only, default "b4"; "b4-buckets" is GD-PLANNER's
attention templates merged), `udraw` (JR2's U, "period" | "call"), `b1` ("card" 1/8 | "wgmma" 3/16) and `pricing`
("tensor" | "in-model") reach pouw.scheme, pouw.form, commit.open, point.steps_per_window, pouw.tile, prover.k_max,
prover.m_max, prover.reps, prover.device, prover.warm, sampling.period, sampling.retain, partition.attn_keys,
partition.census, pouw.u_draw, pouw.b1 and pouw.pricing. `noise: segment` is pi re-drawn each segment. `bits` is the
T_s word's checked low bits: 8 or fewer cap the floor at 7/8 (GD-BITS). `kernels: bar` is the row kernel GD-HASH
measured (0.92 of the bar), not the bar.

A design is feasible when every credited linear's floor (Dither's forced work as a share of its G, at the best attack
found) is at least 1 - eps ("floor"), every proof block is within the block cap, 2^27 unless kmax raises it
("k_log"), its prover memory fits the H200 ("prover_mem"), its largest block lowers within lowering.ram_gib and 15
minutes ("lowering"), the window's retained bytes fit the device's or the host's bound ("retain"), and an aggregate
word covers at least 8 cells ("agg").
"""
from __future__ import annotations

from dataclasses import dataclass

from golden import model as M


@dataclass(frozen=True)
class Input:
    value: float
    low: float
    high: float
    source: str
    status: str
    uncertain: bool = True


#: (low, high) of each estimate: the source's range where it gives one, else a judgement, said in the note
RANGES = {
    "launch_us": (2.0, 8.0),
    "guess_p16": (1e-5, 1e-4),
    "h1_rest_floor_ms_8k": (10.0, 30.0),
    "noise_fused_keep": (0.05, 0.30),
    "permuted_limbs": (2.0, 3.0),                        # every limb dense on 2:4 to none
    "jr2_noise_g": (1.0, 1.81),                          # the noised product alone to GD-NOISE's per-call row
    "jr2_wprg_g_m": (873.0, 3492.0),                     # one PRG byte an entry (GD-NOISE finding 5) to four
    "host_retain_GB": (100.0, 1600.0),                   # half an eighth of a node to the whole node's host memory
    # costfit's per-AND ranges, per row like the values (M.ZK_ROWS)
    "o_per_and": (0.5e-8 / M.ZK_ROWS, 2.2e-8 / M.ZK_ROWS),
    "staged_per_and": (0.5e-10 / M.ZK_ROWS, 2e-10 / M.ZK_ROWS),
    "stmt_fixed": (1.5, 6.0),
    "stmt_per_and": (0.0, 7e-6 / M.ZK_ROWS),             # 0: a template's preprocessing kept across windows
    "stage_fixed": (0.8, 3.2),
    "stage_per_and": (3e-7 / M.ZK_ROWS, 1.2e-6 / M.ZK_ROWS),
    "stage_per_and_staged": (1.4e-9 / M.ZK_ROWS, 5.7e-9 / M.ZK_ROWS),
    "verify_fixed": (6.0, 25.0),
    "verify_per_and": (3.4e-6 / M.ZK_ROWS, 1.34e-5 / M.ZK_ROWS),
    "sm120_warm_a": (1.38, 2.72),                        # GD-PROVER's refit, +-1 sigma
    "sm120_warm_b": (1.05e-6 / M.ZK_ROWS, 1.23e-6 / M.ZK_ROWS),
    "sm120_warm_c": (1.09e-8 / M.ZK_ROWS, 1.31e-8 / M.ZK_ROWS),
    "sm120_gpu_price": (0.3, 1.0),
    "rep_share": (0.15, 0.5),
    "one_rep_query_cost": (0.0, 0.1),
    "ew_ands_per_elem": (1000.0, 10000.0),
    "attn_ands_per_mac": (300.0, 1200.0),
    # TOP-138's target (low) against today's measured lowering (r20261028-184657-1d80) a gate (high)
    "lower_bytes_per_gate": (M.I("lower_bytes_per_gate"), M.I("lower_bytes_per_and_today") / M.I("gates_per_and")),
    "lower_s_per_gate": (M.I("lower_s_per_gate"), M.I("lower_s_per_and_today") / M.I("gates_per_and")),
    "gates_per_and": (1.0, 2.1),
    "k_pad": (0, 3),                                     # 0: the inbox's 2^27 ANDs; class_size's units 1 to 3
    "h1_ands_per_mac": (250.0, 400.0),
    "retain_GB": (50.0, 1000.0),                         # device retention's margin to a host cgroup's pinned memory
    "prover_bytes_per_row": (237.0, 948.0),              # the circuit alone to twice the estimate
    "cap_cost_per_bit": (0.0, 0.2),
}

INPUTS: dict[str, Input] = {
    k: Input(v, *RANGES.get(k, (v, v)), src, s, s == "E" and RANGES.get(k, (v, v))[0] != RANGES.get(k, (v, v))[1])
    for k, (v, s, src) in M.INPUTS.items()
}

POINTS = {"t8192": "throughput", "t512": "demo"}
LOWER_MIN = 15.0

_FUSE = {"all": "every_op", "dedupe": "dedup", "fuse": "fused_mlp", "e1": "e1", "e2": "e2"}
_NOISE = {"perm": "permuted", "periodic": "periodic", "fresh": "fresh", "segment": "segment"}
_LAW = {"work": "achievability", "uniform": "uniform", "stratified": "stratified"}
_OVERLAP = {"none": False, "corun": True, "corun4": "corun4"}


def to_design(k: dict, point: str = "t8192") -> dict:
    """GD-SEARCH's knobs as a `golden.model` design at the point."""
    s: dict[str, dict] = {"partition": {}, "commit": {}, "pouw": {}, "sampling": {}, "prover": {}, "point": {}}
    if "tile" in k:
        s["partition"]["tile"] = tuple(k["tile"])
    if "kchunk" in k:
        s["partition"]["kchunk"] = k["kchunk"]
    if "fuse" in k:
        s["commit"]["values"] = _FUSE[k["fuse"]]
    if "row" in k:
        s["commit"]["row_bytes"] = k["row"]
    for knob, key in (("seal", "seal"), ("host", "host"), ("open", "open")):
        if knob in k:
            s["commit"][key] = k[knob]
    if "kernels" in k:
        s["commit"].update({"hash_eff": M.I("hash_eff_rows"), "gather": False} if k["kernels"] == "bar" else {})
    if "overlap" in k:
        s["commit"]["corun"] = _OVERLAP[k["overlap"]]
    if "noise" in k:
        s["pouw"]["noise"] = _NOISE[k["noise"]]
    if "lp" in k or "t" in k:
        lp = k.get("lp", 128)
        s["pouw"]["L_prime"] = lp
        s["pouw"]["L"] = k.get("t", 8) * lp
    for knob, key in (("drow", "value_row_bytes"), ("bits", "word_bits"), ("agg", "agg"), ("credit", "credit"),
                      ("scheme", "scheme"), ("form", "form"), ("b1", "b1"), ("pricing", "pricing")):
        if knob in k:
            s["pouw"][key] = k[knob]
    if "joint" in k:
        s["pouw"]["joint"] = k["joint"] == "joint"
    if "ptile" in k:
        s["pouw"]["tile"] = tuple(k["ptile"])
    if "akeys" in k:
        s["partition"]["attn_keys"] = k["akeys"]
    if "census" in k:
        s["partition"]["census"] = k["census"]
    if "udraw" in k:
        s["pouw"]["u_draw"] = k["udraw"]
    for knob, key in (("kmax", "k_max"), ("mmax", "m_max"), ("reps", "reps"), ("device", "device"), ("warm", "warm")):
        if knob in k:
            s["prover"][key] = k[knob]
    for knob, key in (("period", "period"), ("retain", "retain")):
        if knob in k:
            s["sampling"][key] = k[knob]
    if "sampling" in k:
        s["sampling"]["law"] = _LAW[k["sampling"]]
    if "stmt" in k:
        s["prover"]["stmt"] = k["stmt"]
    if "steps" in k:
        s["point"]["steps_per_window"] = k["steps"]
    return M.design(POINTS[point], **{sec: v for sec, v in s.items() if v})


def floors(d: dict) -> dict:
    """Each credited linear's floor: Dither's forced share of its G at the best attack found, 0 where one segment
    covers its K (O alone, un-noised). J10-R credits nothing (un-noised at serve time) and JR2 its K's linears."""
    w = d["pouw"]
    if w["scheme"] == "dither":
        f = M.pouw_forced_frac(d)
        return {name: (f if w["L"] < K else 0.0) for name, K, _ in M.dither_linears(d)}
    if w["scheme"] == "jr2":
        return {name: M.pouw_forced_frac(d) for name, K, _ in M.LINEARS if K == w["K"]}
    return {}


def evaluate(knobs: dict, v: dict | None = None, point: str = "t8192") -> dict:
    d = to_design(knobs, point)
    with M.inputs({k: x for k, x in (v or {}).items() if k in M.INPUTS}):
        r = M.J(d)
        fl = floors(d)
        pr = r["proving"]
        blocks = [b for pop in ("values", "pouw") for c in pr[pop]["classes"] for b in c["blocks"]]
        max_ands = max((b["ands"] for b in blocks), default=0.0)
        gpa = d["lowering"]["gates_per_and"] or M.I("gates_per_and")
        spg = d["lowering"]["s_per_gate"] or M.I("lower_s_per_gate")
        bpg = d["lowering"]["bytes_per_gate"] or M.I("lower_bytes_per_gate")
        gates = max_ands * gpa
        lower_gib, lower_min = gates * bpg / 2 ** 30, gates * spg / 60
        lower_gib_today = max_ands * M.I("lower_bytes_per_and_today") / 2 ** 30
    eps = d["sampling"]["eps"]
    floor_min = min(fl.values()) if fl else 0.0
    reasons = []
    if floor_min < 1 - eps:
        reasons.append("floor")
    if any("k_log" in m for m in r["infeasible"]):
        reasons.append("k_log")
    if any("prover memory" in m for m in r["infeasible"]):
        reasons.append("prover_mem")
    if lower_gib > d["lowering"]["ram_gib"] or lower_min > LOWER_MIN:
        reasons.append("lowering")
    if any(m.startswith("a window retains") for m in r["infeasible"]):
        reasons.append("retain")
    if any(m.startswith("pouw.agg") for m in r["infeasible"]):
        reasons.append("agg")
    return {
        "J": r["J"], "serve": r["serve_term"], "prove_ratio": r["prove_term"],
        "prove_gpu_s": pr["total_gpu_s"], "verify_cpu_s": r["verification"]["cpu_s"],
        "t_remote_ms": r["t_remote_ms"], "t_stock_ms": r["t_stock_ms"], "t_window_ms": r["t_remote_window_ms"],
        "parts_ms": r["serve_overhead_ms"],
        "prove_parts": {"values": pr["values"]["gpu_s"], "pouw": pr["pouw"]["gpu_s"], "cpu_in_gpu_s": pr["cpu_in_gpu_s"]},
        "commit_gb": r["commit"].get("step_bytes", 0.0) / 1e9, "words_gb": r["pouw"].get("value_bytes", 0.0) / 1e9,
        "retained_gb": r["retained_GB_per_window"], "floor_min": floor_min, "floors": fl, "credited": sorted(fl),
        "credited_gpu_s": r["credited_gpu_s"], "forced_gpu_s": r["forced_gpu_s"],
        "max_ands": max_ands, "gates": gates, "lower_gib": lower_gib, "lower_gib_today": lower_gib_today,
        "lower_min": lower_min, "draws_values": r["draws"]["values"],
        "draws_compute": pr["pouw"]["draws"] + pr["values"].get("draws_compute", 0.0),
        "feasible": not reasons, "reasons": reasons, "infeasible": r["infeasible"],
    }
