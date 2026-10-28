"""GD-MODEL's J(design) behind GD-SEARCH's model interface (benchmarks/golden_search/search.py `load_model`):

    python search.py run ... --model /path/to/checkout:golden.search_adapter

`INPUTS` is every input of `golden.model` (name -> Input(value, low, high, source, status, uncertain)): a measured
input is certain; an estimate carries the range its source allows. `evaluate(knobs, inputs, point)` maps GD-SEARCH's
knobs (benchmarks/golden_search/space.py) onto a `golden.model` design at the point (t8192: the throughput preset;
t512: the demo's window, a prefill of 64 real tokens in T 512 and 3 decode steps), evaluates J under those input
values, and returns GD-SEARCH's result keys. A knob GD-SEARCH leaves out keeps the model's default; the extra knobs
`scheme`, `form`, `open` and `steps` reach pouw.scheme, pouw.form, commit.open and point.steps_per_window.

A design is feasible when every credited linear's floor (Dither's forced work as a share of its G, at the best attack
found) is at least 1 - eps ("floor"), every proof block is within compose's 2^27 ("k_log"), its largest block lowers
within lowering.ram_gib and 15 minutes ("lowering"), and the window's retained bytes fit the retain_GB input
("retain").
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
    "corun_share": (0.10, 0.25),
    "corun4_share": (0.30, 0.50),
    "corun4_gemm_cost": (0.10, 0.25),
    "guess_p16": (1e-5, 1e-4),
    "h1_rest_floor_ms_8k": (10.0, 30.0),
    "b1_rate_vs_s8": (4.0, 16.0),
    "wrap_g_frac": (0.10, 0.50),
    "noise_fused_keep": (0.05, 0.30),
    "o_per_and": (0.5e-8, 2.2e-8),
    "staged_per_and": (0.5e-10, 2e-10),
    "stmt_fixed": (1.5, 6.0),
    "stmt_per_and": (0.0, 7e-6),                         # 0: a template's preprocessing kept across windows
    "stage_fixed": (0.8, 3.2),
    "stage_per_and": (3e-7, 1.2e-6),
    "stage_per_and_staged": (1.4e-9, 5.7e-9),
    "verify_fixed": (6.0, 25.0),
    "verify_per_and": (3.4e-6, 1.34e-5),
    "h200_vs_sm120": (0.7, 2.0),
    "ew_ands_per_elem": (1000.0, 10000.0),
    "attn_ands_per_mac": (300.0, 1200.0),
    # TOP-138's target (low) against today's measured lowering (r20261028-184657-1d80) a gate (high)
    "lower_bytes_per_gate": (M.I("lower_bytes_per_gate"), M.I("lower_bytes_per_and_today") / M.I("gates_per_and")),
    "lower_s_per_gate": (M.I("lower_s_per_gate"), M.I("lower_s_per_and_today") / M.I("gates_per_and")),
    "gates_per_and": (1.0, 2.1),
    "k_pad": (0, 3),                                     # 0: the inbox's 2^27 ANDs; class_size's units 1 to 3
    "h1_ands_per_mac": (250.0, 500.0),
    "retain_GB": (50.0, 1000.0),                         # device retention's margin to a host cgroup's pinned memory
}

INPUTS: dict[str, Input] = {
    k: Input(v, *RANGES.get(k, (v, v)), src, s, s == "E" and RANGES.get(k, (v, v))[0] != RANGES.get(k, (v, v))[1])
    for k, (v, s, src) in M.INPUTS.items()
}

POINTS = {"t8192": "throughput", "t512": "demo"}
LOWER_MIN = 15.0

_FUSE = {"all": "every_op", "dedupe": "dedup", "fuse": "fused_mlp"}
_NOISE = {"perm": "permuted", "periodic": "periodic", "fresh": "fresh"}
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
        s["commit"].update({"hash_eff": 1.0, "gather": False} if k["kernels"] == "bar" else {})
    if "overlap" in k:
        s["commit"]["corun"] = _OVERLAP[k["overlap"]]
    if "noise" in k:
        s["pouw"]["noise"] = _NOISE[k["noise"]]
    if "lp" in k or "t" in k:
        lp = k.get("lp", 128)
        s["pouw"]["L_prime"] = lp
        s["pouw"]["L"] = k.get("t", 8) * lp
    for knob, key in (("drow", "value_row_bytes"), ("bits", "word_bits"), ("agg", "agg"), ("credit", "credit"),
                      ("scheme", "scheme"), ("form", "form")):
        if knob in k:
            s["pouw"][key] = k[knob]
    if "joint" in k:
        s["pouw"]["joint"] = k["joint"] == "joint"
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
    if lower_gib > d["lowering"]["ram_gib"] or lower_min > LOWER_MIN:
        reasons.append("lowering")
    if any(m.startswith("a window retains") for m in r["infeasible"]):
        reasons.append("retain")
    return {
        "J": r["J"], "serve": r["serve_term"], "prove_ratio": r["prove_term"],
        "prove_gpu_s": pr["total_gpu_s"], "verify_cpu_s": r["verification"]["cpu_s"],
        "t_remote_ms": r["t_remote_ms"], "t_stock_ms": r["t_stock_ms"], "t_window_ms": r["t_remote_window_ms"],
        "parts_ms": r["serve_overhead_ms"],
        "prove_parts": {"values": pr["values"]["gpu_s"], "pouw": pr["pouw"]["gpu_s"], "cpu_in_gpu_s": pr["cpu_in_gpu_s"]},
        "commit_gb": r["commit"].get("step_bytes", 0.0) / 1e9, "words_gb": r["pouw"].get("value_bytes", 0.0) / 1e9,
        "retained_gb": r["retained_GB_per_window"], "floor_min": floor_min, "floors": fl, "credited": sorted(fl),
        "max_ands": max_ands, "gates": gates, "lower_gib": lower_gib, "lower_gib_today": lower_gib_today,
        "lower_min": lower_min, "draws_values": r["draws"]["values"],
        "draws_compute": pr["pouw"]["draws"] + pr["values"].get("draws_compute", 0.0),
        "feasible": not reasons, "reasons": reasons, "infeasible": r["infeasible"],
    }
