"""The reference datacenter: how much the remote and onsite protocols slow training at datacenter scale, per training
class, with the adversary's operating points. Daniel's ruling: claims come from this model, not the demo; results are per
monitoring period (a week, a month, a year) over all compute, pooled, at delta = 1% and 1e-6 where probabilistic terms
enter, with the per-window figure beside them. Every parameter and its source is in ``reference_dc.json``; hardware
values are datasheet values until compute-research's replace them there.

The committed computation is ``requests_per_year`` requests of ``input_tokens`` tokens served by DeepSeek-V4-Pro in FP8,
2 x 49e9 FLOP a token (prefill only unless ``decode_tokens``). A fleet is the whole world's compute in one shape (eight-GPU
H200 servers, or NVL72 racks), sized so the committed work uses it fully. Dark compute has two parts, reported apart and
together:

* (A) dark islands, ``dark_shares`` of all compute, run outside the protocols. An island is one location; it may hold
  anything from before the one registration, so it trains at full speed (rho = 1), limited by its compute, its memory and
  coordination over side channels (``beta_side`` per location). What it learns leaves through side channels only.
* (B) uncertified compute U on monitored servers: PoUW's deficit eps_W, certificate slack eps' = p_window_fail / delta,
  and escapes (the verdict lag, once per identity; the faulty blocks sampled proofs accept at delta; and the training
  failing blocks' chosen outputs carry into inference blocks), the residual of ``tiers.residual_share`` with
  ``tiers.evaluate``'s skipped work and injected bits. Each training class bounds it per server-window (a server or a
  rack, ``window_s`` between wipes) by

      Train_T <= sum_{s,w} min(q_T(C_sw, K_sw), n_sw q_T(c, k) + U_sw / kappa_F)

  in forward products. Cut (a), onsite only, is the whole server-window as one block: C_sw = sigma Phi window_s and
  K_sw = R + I_cert + beta_side window_s. Cut (b) is the window's inference blocks, each at the policy point k with
  c = k phi_F, plus U at face value. Remote has no K, so only cut (b). The pooled cap is free to use remotely; onsite
  a run that pools servers sends every sync out through each one's side channels.

Training classes and their per-block allowance (``allowance``), in counted products, with ``c`` products counted per
forward product:

* ``fan`` (T-fan): e K / beta_x, linear in K, on the products not token-reachable (``fan_x_lead``, and at a larger block
  the parameters whose update at ``fan_b_upd`` fits in K). Conditional on the joint activation premise beta_x; not yet
  connected to the verifier.
* ``thread`` (T-thread): K Lambda / b_a with Lambda = max(e, 2 K / (4 d b_w)) (``tiers.region_cap``), after
  ``slowdown.rates``' caps at the block's size: when a block holds a token's whole active state the threading collapses
  to (1, a token's counted multiplications). Connected to the verifier through #4304's DescThrottles* and
  ThreadedThrottled. It counts forward and backward-to-activation multiplications (c = 2).
* ``cover`` (T-cover): u (K^2 / b + K), connected through #4314's CheckedCoverTraining, counted at the declaration's gate
  granularity, a Dot16 step; ``cover_unit_products`` is the flagged normalization u to products.
* ``small`` (N1): unthrottled whenever the complete persistent state and a step's inputs fit in the block's budget; it
  reports the state threshold, not a slowdown.

rho = C / (kappa_F products) per block kind. A rho below 1, or a premise that fails at the block's size, means no
useful penalty is proved there, not that dark training is efficient. Every quantity is in the FLOP model; converting to
gates is a later pass.

    python benchmarks/nci/reference_dc.py               # every table, Markdown
    python benchmarks/nci/reference_dc.py --json        # the same, as JSON
    python benchmarks/nci/reference_dc.py --out DIR     # tables.md, results.json and plots, for the evidence store
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import slowdown as S  # noqa: E402
import tiers as T  # noqa: E402

PARAMS = HERE / "reference_dc.json"
YEAR = T.YEAR
CLASSES = ("fan", "thread", "cover")
MODES = ("remote", "onsite")
#: the exfiltration references and their parameters
REFERENCES = {"rollouts": "exfil_rollout_bits", "weights": "exfil_weights_bits", "one sync": "exfil_sync_bits",
              "1,000 syncs": "exfil_sparseloco_bits"}
#: when the dark share's or the monitored fleet's side channels may be pooled for a reference: only when it already
#: sits split across the locations
LEGITIMATE = {
    "rollouts": "yes: rollouts are generated at every location",
    "weights": "only for a replica every location holds (after a sync); one island's weights leave at its own rate",
    "one sync": "only when every location holds the aggregate; one island's pseudo-gradient leaves at its own rate",
    "1,000 syncs": "only when every location holds the aggregates",
}
BETA_SWEEP = [10 ** (e / 2) for e in range(-2, 15)]
DARK_SWEEP = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def load(path: Path = PARAMS) -> tuple[dict, dict]:
    """The parameters' values and the fleets' values."""
    raw = json.loads(path.read_text())
    p = {k: v["value"] for k, v in raw["parameters"].items()}
    fleets = {f: {k: v["value"] for k, v in spec.items() if isinstance(v, dict)} for f, spec in raw["fleets"].items()}
    return p, fleets


def sources(path: Path = PARAMS) -> list[dict]:
    raw = json.loads(path.read_text())
    rows = [{"name": k, **v} for k, v in raw["parameters"].items()]
    rows += [{"name": f"{f}.{k}", **v} for f, spec in raw["fleets"].items() for k, v in spec.items()
             if isinstance(v, dict)]
    return rows


# --- the committed computation and the references -------------------------------------------------------------------

def committed_flop(p: dict, H: float = YEAR) -> float:
    """The committed work over ``H`` seconds: 2 N_active FLOP per served token."""
    return 2 * p["n_active"] * p["requests_per_year"] * (p["input_tokens"] + p["decode_tokens"]) * H / YEAR


def h100_hour(p: dict) -> float:
    return p["h100_flops"] * 3600


def phi_F(p: dict) -> float:
    """FLOP an inference block does per novel bit it reads: 2 N_active (in + out) / (b_tok in) for this model's tokens,
    or the traffic calibration."""
    if p["phi_F_mode"] == "traffic":
        return p["phi_F_traffic"]
    tokens = p["input_tokens"] + p["decode_tokens"]
    return 2 * p["n_active"] * tokens / (p["b_tok"] * p["input_tokens"])


def work_step(p: dict) -> float:
    return p["kappa_F"] * p["n_active"] * p["batch_tokens"]


def work_run(p: dict) -> float:
    return p["kappa_F"] * p["n_active"] * p["pretrain_tokens"]


def steps_per_run(p: dict) -> float:
    return p["pretrain_tokens"] / p["batch_tokens"]


def state_bytes(p: dict) -> float:
    return p["n_total"] * p["state_bytes_per_param"]


def whole_update_bits(p: dict) -> float:
    """The whole update at C1's floor: a block that reads this much can hold every updated weight at the premises' rate,
    so no premise charges it."""
    return p["n_total"] * p["thread_b_w"]


# --- fleets and blocks ------------------------------------------------------------------------------------------------

def servers(p: dict, f: dict) -> float:
    """Monitored locations: the committed work at full use of the fleet."""
    return committed_flop(p) / (p["sigma"] * f["flops"] * YEAR)


def block(p: dict, f: dict) -> dict:
    """One server-window and one inference block: compute, reads and their ratios."""
    W = p["window_s"]
    C = p["sigma"] * f["flops"] * W
    R = p["R_device_bytes"] * 8 * (f["devices"] if p["R_scope"] == "device" else 1)
    requests = p["requests_per_year"] * W / YEAR / servers(p, f)
    I_cert = requests * p["input_tokens"] * p["b_tok"] * p["I_cert_multiplier"]
    side = p["beta_side"] * W
    K = R + I_cert + side
    c_F = p["k_inference"] * phi_F(p)
    return {"C": C, "R": R, "I_cert": I_cert, "side": side, "K": K, "requests": requests, "k": p["k_inference"],
            "c": c_F, "K_over_k": K / p["k_inference"], "C_over_c": C / c_F,
            "beta_takeover": (R + I_cert) / W}


# --- U, the uncertified compute on monitored servers ------------------------------------------------------------------

def u_parts(p: dict, f: dict, H: float, delta: float, cls: str | None = None) -> dict:
    """U's shares of the fleet's compute over ``H``: ``steady`` in every window (r-hat, eps_W on the certified work and
    the certificate slack, ``tiers.residual_share``); ``lag``, the verdict lag once per identity, all declared work in
    one stretch; and ``spread``, what sampled proofs let through at ``delta`` wherever the adversary puts it:
    ``escape``, the faulty blocks they accept (``tiers.evaluate``'s skipped work), and with ``cls`` ``inject``, the
    training the chosen outputs of failing blocks carry into inference blocks (``tiers.evaluate``'s cheat)."""
    slack = p["p_window_fail"] / delta
    steady = T.residual_share(T.DEFAULTS | {"residual": p["r_hat"], "host_share": 0.0, "clock_share": 0.0,
                                            "eps_w": p["eps_w"], "eps_agg": slack})
    lag = min(1.0, p["identities"] * p["lag_s"] / H)
    if p["escape_share"] is None:
        accepted = p["luck"] * p["identities"] * math.log(1 / delta) / sampled_q(p)
        escape = accepted * p["sampled_step_flop"] / committed_flop(p, H)
    else:
        escape = p["escape_share"]
    inject = injected(p, cls, delta)["flop"] / committed_flop(p, H) if cls else 0.0
    spread = escape + inject
    return {"steady": steady, "slack": slack, "lag": lag, "escape": escape, "inject": inject, "spread": spread,
            "once": min(1.0, lag + spread), "total": min(1.0, steady + lag + spread)}


def sampled_q(p: dict) -> float:
    """Sampled proofs' draw rate per step: their compute share over the proving and replay cost."""
    return p["sampled_share"] / (p["sampled_overhead"] + p["sampled_replay"])


def injected(p: dict, cls: str, delta: float) -> dict:
    """Failing blocks' chosen outputs, as in ``tiers.evaluate``: identities ln(1/delta) (1/q + floor_steps) of them
    escape at ``delta``, ``sampled_step_bits`` each, into inference blocks. Each bit is worth the steepest secant of
    the class's allowance from k, out to the bits or to where a block's allowance reaches its compute (the allowance
    is convex, so the secant peaks there); in reference-training FLOP."""
    L = math.log(1 / delta)
    bits = p["identities"] * L * (1 / sampled_q(p) + p["sampled_floor_steps"]) * p["sampled_step_bits"]
    k = p["k_inference"]
    cap = k * phi_F(p) / p["kappa_F"]
    base = allowance(p, cls, k)["products"]

    def gain(b: float) -> float:
        a = allowance(p, cls, k + b)["products"]
        return cap - base if a is None else min(a, cap) - base

    end = bits
    if gain(bits) >= cap - base:
        lo, hi = 1.0, bits
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            lo, hi = (lo, mid) if gain(mid) >= cap - base else (mid, hi)
        end = hi
    per_bit = gain(end) / end
    return {"bits": bits, "per_bit": per_bit, "flop": p["kappa_F"] * bits * per_bit}


# --- the training classes ---------------------------------------------------------------------------------------------

def token_reachable(p: dict, K: float) -> float:
    """T-fan's excluded share: the given leading-layer share, or at a larger block the parameters whose update at
    ``fan_b_upd`` fits in K."""
    return max(p["fan_x_lead"], min(1.0, K / (p["n_total"] * p["fan_b_upd"])))


def thread_inputs(p: dict) -> dict:
    """``slowdown.rates``' inputs for full-parameter training of this model."""
    c = p["thread_c_reuse"]
    return {"b_w": p["thread_b_w"], "U_bits": whole_update_bits(p), "n_updated": p["n_total"], "b_a": p["thread_b_a"],
            "w_cut": p["d_model"] * p["thread_b_a"], "d": p["thread_d"], "e": p["thread_e"],
            "U_token_bits": p["n_active"] * p["thread_b_w"], "kappa": 2 * c / p["kappa_F"],
            "flops_per_token": p["kappa_F"] * p["n_active"]}


def allowance(p: dict, cls: str, K: float) -> dict:
    """The most counted products one block reading ``K`` bits holds under ``cls``, and in forward products. ``premise``
    names what fails at this block's size (then ``products`` is None: no bound)."""
    premise = None
    if K >= whole_update_bits(p):
        premise = "fails: the block holds the whole update at C1's floor (small-state)"
    if cls == "fan":
        x = token_reachable(p, K)
        if x >= 1 and premise is None:
            premise = "fails: every product is token-reachable"
        q = math.inf if premise else p["fan_e"] * K / p["fan_beta_x"] / (1 - x)
        out = {"q": q, "c": 1, "regime": "linear", "x_lead": x}
    elif cls == "thread":
        b_w, b_a, d, e = S.rates(thread_inputs(p), K, p["b_tok"])
        t = {"e": e, "d": d, "b_w": b_w, "b_a": b_a}
        lam = T.lam(t, K)
        collapsed = (d, e) != (p["thread_d"], p["thread_e"])
        if collapsed and premise is None:
            premise = "collapses: the block holds a token's whole active state, so (d, e) = (1, a token's products)"
        out = {"q": T.region_cap(t, K), "c": p["thread_c_reuse"], "regime": "linear" if lam == e else "quadratic",
               "d": d, "e": e, "Lambda": lam, "knee_bits": 4 * d * b_w * e / p["thread_c_reuse"]}
    elif cls == "cover":
        out = {"q": p["cover_unit_products"] * (K * K / p["cover_b"] + K), "c": 1, "regime": "quadratic"}
    else:
        raise ValueError(f"unknown class {cls}")
    valid = premise is None or premise.startswith("collapses")
    out["products"] = out["q"] / out["c"] if valid else None
    out["premise"] = premise
    return out


def rho(p: dict, C: float, a: dict) -> float | None:
    """Reference training FLOP per FLOP the block may spend: C / (kappa_F products); None where no bound holds."""
    return None if a["products"] is None else C / (p["kappa_F"] * a["products"])


def small_state(p: dict, K: float) -> dict:
    """N1's threshold: the persistent state that fits beside one step's inputs, in bits and in parameters."""
    room = max(0.0, K - p["batch_tokens"] * p["b_tok"])
    return {"K": K, "state_bits": room, "state_bits_no_data": K,
            "params_adam": room / (8 * p["state_bytes_per_param"]), "params_floor": room / p["thread_b_w"],
            "model_fits_adam": state_bytes(p) * 8 <= room, "model_fits_floor": whole_update_bits(p) <= room}


# --- (B): monitored compute per server-window and per period ----------------------------------------------------------

def window(p: dict, f: dict, cls: str, delta: float, H: float = YEAR) -> dict:
    """One server-window's training cap in forward products, both cuts, with U's steady share."""
    b = block(p, f)
    u = u_parts(p, f, H, delta)["steady"]
    full = b["C"] / p["kappa_F"]
    inf_a, sd_a = allowance(p, cls, b["k"]), allowance(p, cls, b["K"])
    per_block = min(inf_a["products"], b["c"] / p["kappa_F"]) if inf_a["products"] is not None else b["c"] / p["kappa_F"]
    n = (1 - u) * b["C"] / b["c"]
    cut_b = min(full, n * per_block + u * full)
    cut_a = min(full, sd_a["products"]) if sd_a["products"] is not None else full
    return {"block": b, "u": u, "n_blocks": n, "full": full, "cut_a": cut_a, "cut_b": cut_b,
            "rho_inference": rho(p, b["c"], inf_a), "rho_server_window": rho(p, b["C"], sd_a),
            "inference": inf_a, "server_window": sd_a,
            "train": {"remote": cut_b, "onsite": min(cut_a, cut_b)},
            "binding_onsite": "a" if cut_a < cut_b else "b"}


def period(p: dict, f: dict, cls: str, H: float, delta: float, mode: str) -> dict:
    """The pooled training cap over ``H`` seconds of the whole fleet, in reference-training FLOP: every window at U's
    steady share, plus the lag (all declared work, in the windows ``lag_s`` spans) and the spread escapes (wherever the
    adversary puts them), at face value remotely and capped by each server-window's cut (a) onsite. Pooling the
    monitored servers' training is free remotely; onsite every sync of a cooperating run leaves each server through its
    side channels (``coordinated_t_run``)."""
    w = window(p, f, cls, delta, H)
    N = servers(p, f)
    n_win = H / p["window_s"]
    parts = u_parts(p, f, H, delta, cls)
    per = committed_flop(p, H) / N / p["kappa_F"]
    lag, spread = parts["lag"] * per, parts["spread"] * per
    hit = min(n_win, max(1.0, math.ceil(p["lag_s"] / p["window_s"] - 1e-9)))
    if mode == "remote":
        per_server = min(n_win * w["full"], n_win * w["cut_b"] + lag + spread)
    else:
        base, top = w["train"]["onsite"], min(w["cut_a"], w["full"])
        lag_gain = hit * max(0.0, min(top, w["cut_b"] + lag / hit) - base)
        per_server = n_win * base + lag_gain + min(spread, n_win * (top - base) - lag_gain)
    flop = p["kappa_F"] * N * per_server
    C = committed_flop(p, H)
    rate = flop / H
    t_run = work_run(p) / rate
    sync_bound = p["coop_syncs"] * p["exfil_sync_bits"] / p["beta_side"]
    alone = work_run(p) / (rate / N)
    coordinated = t_run if mode == "remote" else min(alone, max(t_run, sync_bound))
    return {"fleet_flop": C, "train_flop": flop, "D": C / flop, "steps": flop / work_step(p), "runs": flop / work_run(p),
            "t_step": work_step(p) / rate, "t_run": t_run, "t_run_one_server": alone, "coordinated_t_run": coordinated,
            "u": parts, "window_train_flop": p["kappa_F"] * w["train"][mode], "window_D": w["full"] / w["train"][mode],
            "binding": "b" if mode == "remote" else w["binding_onsite"]}


def one_block(p: dict, f: dict) -> dict:
    """The one-block regime: the first steps after the one registration whose inputs fit in a block's K and whose replay
    fits in its compute. The block computes their updated weights from theta and its inputs, so the incompressibility
    premises fail there and those steps run at rho = 1."""
    b = block(p, f)
    data = p["batch_tokens"] * p["b_tok"]

    def steps(K: float, C: float) -> dict:
        by_inputs, by_replay = K / data, C / work_step(p)
        s = math.floor(min(by_inputs, by_replay))
        return {"by_inputs": by_inputs, "by_replay": by_replay, "steps": s, "binding": "inputs" if by_inputs < by_replay
                else "replay", "run_share": s / steps_per_run(p), "flop": s * work_step(p)}

    return {"server_window": steps(b["K"], b["C"]), "inference": steps(b["k"], b["c"]),
            "committed_data_server_window": steps(math.inf, b["C"])}


# --- (A): dark islands ------------------------------------------------------------------------------------------------

def dark_locations(p: dict, f: dict, share: float) -> float:
    """Dark locations when ``share`` of all compute is dark."""
    return share / (1 - share) * servers(p, f)


def island(p: dict, f: dict) -> dict:
    """One island training the reference model alone: step and run times at its compute, its memory, and the offload's
    floor (the offloaded optimizer state read and written once a step, overlapped with compute; streaming weights per
    micro-batch would add to it)."""
    phi = p["sigma"] * f["flops"]
    need = state_bytes(p)
    off = max(0.0, need - f["hbm_bytes"])
    host = min(off, f["host_bytes"])
    nvme = max(0.0, off - f["host_bytes"])
    feasible = nvme <= f["nvme_bytes"]
    t_off = 2 * host / f["host_Bps"] + (2 * nvme / f["nvme_Bps"] if nvme else 0.0)
    compute = work_step(p) / phi
    step = max(compute, t_off) if feasible else math.inf
    where = ("HBM" if off == 0 else "HBM + host" if nvme == 0 else "HBM + host + NVMe" if feasible else "nowhere")
    return {"flops": phi, "state_bytes": need, "hbm_bytes": f["hbm_bytes"], "fits": where, "feasible": feasible,
            "offload_bytes": off, "t_offload": t_off, "t_compute": compute, "t_step": step,
            "t_step_serial": compute + t_off if feasible else math.inf, "t_run": steps_per_run(p) * step,
            "offload_binds": t_off > compute}


def dark(p: dict, f: dict, share: float, H: float) -> dict:
    """The dark share's operating points: one island, the share cooperating (a replica per island, ``coop_syncs`` syncs
    a run across side channels; ``coop_t_step`` is its time per reference step), and its raw capacity (ignoring
    coordination)."""
    one = island(p, f)
    N = dark_locations(p, f, share)
    t_sync = p["exfil_sync_bits"] / p["beta_side"]
    coop_run = max(one["t_run"] / N, p["coop_syncs"] * t_sync)
    coop_step = coop_run / steps_per_run(p)
    capacity = N * one["flops"] * H
    return {"share": share, "locations": N, "island": one, "t_sync": t_sync, "coop_t_run": coop_run,
            "coop_t_step": coop_step, "coop_beats_island_above_bps": p["coop_syncs"] * p["exfil_sync_bits"] / one["t_run"],
            "island_runs_in_H": H / one["t_run"], "coop_runs_in_H": H / coop_run,
            "capacity_flop": capacity, "capacity_runs": capacity / work_run(p),
            "capacity_t_run": work_run(p) / (N * one["flops"])}


def exfil(p: dict, f: dict, share: float) -> dict:
    """Seconds to move each reference out: through one location's side channels, the dark share's pooled, the monitored
    fleet's pooled (onsite), and one monitored server's line rate (remote: the protocols bound nothing). And the
    chance sampled proofs catch it leaving as failing blocks' chosen outputs in the committed responses,
    ``sampled_step_bits`` a step: 1 - exp(-q n) over its n steps, at any rate."""
    N_d, N_m, beta = dark_locations(p, f, share), servers(p, f), p["beta_side"]
    out = {}
    for name, key in REFERENCES.items():
        bits = p[key]
        out[name] = {"bits": bits, "per_location": bits / beta, "dark_pooled": bits / (N_d * beta),
                     "monitored_pooled": bits / (N_m * beta), "remote_line": bits / f["nic_bps"],
                     "faulty_outputs_p_detect": -math.expm1(-sampled_q(p) * bits / p["sampled_step_bits"]),
                     "legitimate": LEGITIMATE[name]}
    return out


def together(p: dict, f: dict, cls: str, H: float, delta: float, mode: str, share: float) -> dict:
    """(A) and (B) together: the run's time at the dark share's capacity plus the monitored cap (ignoring coordination,
    a floor on the time), and coordination-limited, the best of one island, the dark share cooperating and the
    monitored fleet's coordinated run."""
    m = period(p, f, cls, H, delta, mode)
    d = dark(p, f, share, H)
    rate = d["capacity_flop"] / H + m["train_flop"] / H
    return {"capacity_t_run": work_run(p) / rate, "capacity_runs_in_H": rate * H / work_run(p),
            "coordinated_t_run": min(d["island"]["t_run"], d["coop_t_run"], m["coordinated_t_run"])}


def scenarios(p: dict, f: dict, H: float = YEAR, delta: float = 1e-6) -> list[dict]:
    """Cut (a) under R's two scopes and two window lengths: K, each class's server-window rho and premise, and D
    onsite."""
    rows = []
    for scope in ("device", "location"):
        for W in (86400.0, 3600.0):
            q = p | {"R_scope": scope, "window_s": W}
            b = block(q, f)
            row = {"R_scope": scope, "window_s": W, "K": b["K"], "C": b["C"]}
            for cls in CLASSES:
                w = window(q, f, cls, delta, H)
                row[cls] = {"rho": w["rho_server_window"], "premise": w["server_window"]["premise"],
                            "D_onsite": period(q, f, cls, H, delta, "onsite")["D"]}
            rows.append(row)
    return rows


# --- sweeps and sensitivity -------------------------------------------------------------------------------------------

def headline_D(p: dict, f: dict, cls: str = "fan", mode: str = "onsite", H: float = YEAR, delta: float = 1e-6) -> float:
    return period(p, f, cls, H, delta, mode)["D"]


def beta_sweep(p: dict, f: dict, betas: list[float] = BETA_SWEEP) -> list[dict]:
    rows = []
    for beta in betas:
        q = p | {"beta_side": beta}
        b = block(q, f)
        row = {"beta_side": beta, "K": b["K"], "side_share_of_K": b["side"] / b["K"],
               "island_coop_t_run": dark(q, f, 0.01, YEAR)["coop_t_run"],
               "weights_per_location_s": p["exfil_weights_bits"] / beta}
        for cls in CLASSES:
            w = window(q, f, cls, 1e-6)
            row[f"rho_sw_{cls}"] = w["rho_server_window"]
            row[f"D_onsite_{cls}"] = headline_D(q, f, cls)
        rows.append(row)
    return rows


def dark_sweep(p: dict, f: dict, shares: list[float] = DARK_SWEEP) -> list[dict]:
    rows = []
    for s in shares:
        d = dark(p, f, s, YEAR)
        e = exfil(p, f, s)
        rows.append({"share": s, "locations": d["locations"], "capacity_t_run": d["capacity_t_run"],
                     "coop_t_run": d["coop_t_run"], "weights_pooled_s": e["weights"]["dark_pooled"],
                     "syncs_pooled_s": e["1,000 syncs"]["dark_pooled"]})
    return rows


def ranged(path: Path = PARAMS) -> dict[str, tuple[float, float, str]]:
    raw = json.loads(path.read_text())["parameters"]
    return {k: (v["range"][0], v["range"][1], v["range_source"]) for k, v in raw.items() if "range" in v}


def tornado(p: dict, f: dict, metric, path: Path = PARAMS) -> list[dict]:
    """The metric at each ranged parameter's ends, the others at their values, ranked by log swing; elasticity
    d ln metric / d ln p by a central difference at the value (None at a value of zero or unset)."""
    rows = []
    base = metric(p, f)
    for key, (lo, hi, why) in ranged(path).items():
        at = {end: metric(p | {key: v}, f) for end, v in (("low", lo), ("high", hi))}
        v = p[key]
        el = None
        if isinstance(v, (int, float)) and v:
            up, down = (metric(p | {key: v * math.exp(s * 0.05)}, f) for s in (1, -1))
            el = (math.log(up) - math.log(down)) / 0.1
        rows.append({"key": key, "value": v, "low": lo, "high": hi, "source": why, "at_low": at["low"],
                     "at_high": at["high"], "base": base, "elasticity": el,
                     "swing": abs(math.log(max(at.values()) / min(at.values())))})
    return sorted(rows, key=lambda r: -r["swing"])


# --- the whole model --------------------------------------------------------------------------------------------------

def sanity(p: dict, fleets: dict) -> dict:
    """Top's sanity numbers."""
    F = committed_flop(p)
    hours = F / h100_hour(p)
    return {"committed_flop_year": F, "h100_hours": hours, "h100_equivalents": hours / (YEAR / 3600),
            "h200_servers": servers(p, fleets["h200-server"]), "nvl72_racks": servers(p, fleets["nvl72"]),
            "sync_years_at_1kbps": p["exfil_sync_bits"] / 1e3 / YEAR, "run_flop": work_run(p),
            "step_h100_hours": work_step(p) / h100_hour(p),
            "run_years_h200_island": work_run(p) / (fleets["h200-server"]["flops"] * p["sigma"]) / YEAR,
            "run_days_nvl72_island": work_run(p) / (fleets["nvl72"]["flops"] * p["sigma"]) / 86400}


def evaluate(p: dict, fleets: dict, path: Path = PARAMS) -> dict:
    out = {"sanity": sanity(p, fleets), "fleets": {}}
    for name, f in fleets.items():
        b = block(p, f)
        fo = {"servers": servers(p, f), "block": b, "one_block": one_block(p, f),
              "small_state": {"inference": small_state(p, b["k"]), "server_window": small_state(p, b["K"])},
              "classes": {}, "monitored": [], "dark": [], "exfil": {}, "together": []}
        for cls in CLASSES:
            w = window(p, f, cls, p["deltas"][0])
            fo["classes"][cls] = {"rho_inference": w["rho_inference"], "rho_server_window": w["rho_server_window"],
                                  "inference": w["inference"], "server_window": w["server_window"]}
        for hname, H in p["horizons"].items():
            for delta in p["deltas"]:
                for cls in CLASSES:
                    for mode in MODES:
                        fo["monitored"].append({"horizon": hname, "delta": delta, "class": cls, "mode": mode,
                                                **period(p, f, cls, H, delta, mode)})
            for share in p["dark_shares"]:
                fo["dark"].append({"horizon": hname, **dark(p, f, share, H)})
                for cls in CLASSES:
                    for mode in MODES:
                        fo["together"].append({"horizon": hname, "share": share, "class": cls, "mode": mode,
                                               "delta": p["deltas"][-1],
                                               **together(p, f, cls, H, p["deltas"][-1], mode, share)})
        for share in p["dark_shares"]:
            fo["exfil"][share] = exfil(p, f, share)
        fo["scenarios"] = scenarios(p, f)
        fo["beta_sweep"] = beta_sweep(p, f)
        fo["dark_sweep"] = dark_sweep(p, f)
        fo["tornado"] = {f"D_{cls}_{mode}": tornado(p, f, lambda q, g, c=cls, m=mode: headline_D(q, g, c, m), path)
                         for cls in CLASSES for mode in MODES}
        out["fleets"][name] = fo
    return out


# --- rendering --------------------------------------------------------------------------------------------------------

def g(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "no bound"
    if x == math.inf:
        return "inf"
    return f"{x:.{digits}g}"


def dur(s: float | None) -> str:
    if s is None or s == math.inf:
        return "never"
    for unit, size in (("y", YEAR), ("d", 86400.0), ("h", 3600.0), ("min", 60.0)):
        if s >= size:
            return f"{s / size:.3g} {unit}"
    return f"{s:.3g} s"


def val(v) -> str:
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (int, float)):
        return f"{v:.6g}"
    if isinstance(v, list):
        return ", ".join(val(x) for x in v)
    return ", ".join(f"{k} {val(x)}" for k, x in v.items())


def elast(x: float | None) -> str:
    return "n/a (value 0 or unset)" if x is None else g(x)


def table(head: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(" --- " for _ in head) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def headline(p: dict, r: dict, delta: float) -> str:
    """Per fleet x horizon x dark share x class: rho, Train_T and D remote and onsite (pooled, with the per-window D
    beside it), U's share, and the islands' and the monitored fleet's operating points."""
    rows = []
    for name, fo in r["fleets"].items():
        mon = {(m["horizon"], m["class"], m["mode"]): m for m in fo["monitored"] if m["delta"] == delta}
        dk = {(d["horizon"], d["share"]): d for d in fo["dark"]}
        for hname in p["horizons"]:
            for share in p["dark_shares"]:
                d = dk[(hname, share)]
                for cls in CLASSES:
                    c, rm, on = fo["classes"][cls], mon[(hname, cls, "remote")], mon[(hname, cls, "onsite")]
                    rows.append([name, hname, g(share), cls, g(c["rho_inference"]), g(c["rho_server_window"]),
                                 g(rm["train_flop"]), f"{g(rm['D'])} ({g(rm['window_D'])})", g(on["train_flop"]),
                                 f"{g(on['D'])} ({g(on['window_D'])})", g(rm["u"]["total"]),
                                 dur(d["island"]["t_run"]), g(d["capacity_flop"]), dur(d["coop_t_run"]),
                                 dur(rm["coordinated_t_run"]), dur(on["coordinated_t_run"])])
    return table(["fleet", "horizon", "dark share", "class", "rho inference block", "rho server-window",
                  "Train_T remote (FLOP)", "D remote (per window)", "Train_T onsite (FLOP)", "D onsite (per window)",
                  "U share", "island t(run)", "dark capacity (FLOP)", "dark cooperating t(run)",
                  "monitored t(run), remote", "monitored t(run), onsite"], rows)


def caveats(p: dict, fleets: dict, r: dict) -> list[str]:
    """Every caveat, with its numbers."""
    out = []
    for name, fo in r["fleets"].items():
        f, b = fleets[name], fo["block"]
        for cls, c in fo["classes"].items():
            sw, inf = c["server_window"], c["inference"]
            out.append(f"{name}, {cls}: at the inference block the premise {inf['premise'] or 'holds'} "
                       f"({inf['regime']}, rho {g(c['rho_inference'])}); at the server-window (K {g(b['K'])} bit) "
                       f"it {sw['premise'] or 'holds'} ({sw['regime']}, rho {g(c['rho_server_window'])}"
                       + (", vacuous" if c["rho_server_window"] is not None and c["rho_server_window"] < 1 else "")
                       + ").")
        i = fo["dark"][0]["island"]
        out.append(f"{name}, memory: {g(i['state_bytes'])} B of training state against {g(f['hbm_bytes'])} B of HBM; "
                   f"it fits in {i['fits']}, offloading {g(i['offload_bytes'])} B, whose read and write take at least "
                   f"{dur(i['t_offload'])} a step against {dur(i['t_compute'])} of compute"
                   + (" (the offload binds)" if i["offload_binds"] else " (overlapped, it doesn't bind)") + ".")
        ob = fo["one_block"]["server_window"]
        out.append(f"{name}, one-block regime: a server-window holds the first {ob['steps']} steps after the one "
                   f"registration ({ob['binding']}-bound; inputs fit for {g(ob['by_inputs'])}, replay for "
                   f"{g(ob['by_replay'])}), {g(ob['run_share'])} of the run, at rho = 1; an inference block holds "
                   f"none.")
        inj = injected(p, "fan", p["deltas"][-1])["bits"]
        out.append(f"{name}, faulty outputs: at delta {g(p['deltas'][-1])} sampled proofs let {g(inj)} chosen output "
                   f"bits through fleet-wide before a detection, {g(inj / fo['servers'] / YEAR)} b/s a location over "
                   f"a year and {g(inj / fo['servers'] / p['horizons']['week'])} over a week (beta_side "
                   f"{g(p['beta_side'])}); they're counted as training (U's 'injected'), not in the exfiltration "
                   f"times: the weights leave that way with P(caught) "
                   f"{g(fo['exfil'][p['dark_shares'][0]]['weights']['faulty_outputs_p_detect'])}.")
        out.append(f"{name}, host CPUs: r-hat is 0; counting them at datasheet peak would add "
                   f"{g(f['host_share_peak'])} to U (against eps_W {g(p['eps_w'])}).")
    mon = r["fleets"][next(iter(r["fleets"]))]["monitored"]
    lo, hi = ({m["delta"]: m["u"]["slack"] for m in mon}[d] for d in p["deltas"])
    top = max(mon, key=lambda m: m["u"]["spread"])
    out.append(f"delta barely moves U: the certificate slack is {g(lo)} at delta {g(p['deltas'][0])} and {g(hi)} at "
               f"{g(p['deltas'][-1])}, against eps_W {g(p['eps_w'])}. The spread terms (accepted faulty blocks and "
               f"failing blocks' injected training) reach at most {g(top['u']['spread'])} of the compute ({top['class']}, "
               f"a {top['horizon']}, delta {g(top['delta'])}); the verdict lag is {g(p['lag_s'])} s once, "
               f"{g(p['lag_s'] / p['horizons']['week'])} of a week.")
    out.append("T-fan is conditional (the joint activation premise beta_x) and not connected to the verifier; its "
               "token-reachable share above the inference block is our extrapolation (fan_b_upd). T-cover's Dot16 "
               "normalization (cover_unit_products) is flagged. The premise check 'the block holds the whole update "
               "at C1's floor' is ours: no class charges a block that reads every updated weight at b_w.")
    out.append("Pooling: the monitored fleet pools freely remotely; onsite, a cooperating run's syncs leave every "
               "server through its side channels. The dark share's pooled exfiltration is legitimate only where the "
               "reference already sits split across its locations (the table's last column).")
    out.append("Units: FLOP throughout; the gate conversion is a later pass. Hardware values are datasheets until "
               "compute-research's Phi, memory, sigma and inventory replace them in reference_dc.json.")
    return out


def render(p: dict, fleets: dict, r: dict, path: Path = PARAMS) -> str:
    out = []
    s = r["sanity"]
    out.append("# The reference datacenter\n")
    out.append("Units: the FLOP model throughout (reference training FLOP at kappa_F = "
               f"{p['kappa_F']:g} per forward product); converting to gates is a later pass. A rho below 1, or "
               "'no bound', means no useful penalty is proved there, not that dark training is efficient. "
               f"phi_F = {g(phi_F(p))} FLOP per novel bit ({p['phi_F_mode']}"
               + (": 2 N_active / b_tok" if p["phi_F_mode"] == "model" else ": the traffic calibration") + "). "
               "D is the fleet's compute over the reference training it may do in the period, pooled over every "
               "location; the per-window D is beside it. T-fan is conditional on the joint activation premise and "
               "not connected to the verifier; T-thread and T-cover are (#4304, #4314).\n")
    out.append(f"## Headline (delta {g(p['deltas'][-1])}; every delta in the per-fleet tables)\n")
    out.append("Monitored t(run) is the reference run's time at the fleet's pooled cap: free to pool remotely; onsite "
               "each of its syncs leaves every server through side channels, or one server trains alone.\n")
    out.append(headline(p, r, p["deltas"][-1]))
    out.append("\n## Caveats\n")
    out.extend(f"* {c}" for c in caveats(p, fleets, r))
    out.append("\n## Sanity numbers\n")
    out.append(table(["quantity", "value"], [
        ["committed work (FLOP/yr, prefill only)" if not p["decode_tokens"] else "committed work (FLOP/yr)",
         g(s["committed_flop_year"])],
        ["H100-hours/yr", g(s["h100_hours"])], ["H100-equivalents at full use", g(s["h100_equivalents"])],
        ["H200 servers", g(s["h200_servers"])], ["NVL72 racks", g(s["nvl72_racks"])],
        ["one 17 GB sync at 1 kb/s (years)", g(s["sync_years_at_1kbps"])],
        ["pretraining run (FLOP)", g(s["run_flop"])], ["one optimizer step (H100-hours)", g(s["step_h100_hours"])],
        ["the run on one H200 server (years)", g(s["run_years_h200_island"])],
        ["the run on one NVL72 (days)", g(s["run_days_nvl72_island"])]]))
    out.append("\n## Parameters\n")
    out.append(table(["parameter", "value", "unit", "source"],
                     [[x["name"], val(x["value"]), x["unit"], x["source"]] for x in sources(path)]))
    for name, fo in r["fleets"].items():
        b = fo["block"]
        out.append(f"\n## Fleet: {name}\n")
        out.append(table(["quantity", "value"], [
            ["locations (monitored)", g(fo["servers"])], ["C per server-window (FLOP)", g(b["C"])],
            ["R, wipe residual (bit)", g(b["R"])], ["I_cert, certified ingress per window (bit)", g(b["I_cert"])],
            ["side channels per window (bit)", g(b["side"])], ["K per server-window (bit)", g(b["K"])],
            ["inference block k (bit) / c (FLOP)", f"{g(b['k'])} / {g(b['c'])}"],
            ["server-window over inference block, K / C", f"{g(b['K_over_k'])} / {g(b['C_over_c'])}"],
            ["side channels take over K above (bit/s)", g(b["beta_takeover"])],
            ["whole update at C1's floor (bit)", g(whole_update_bits(p))]]))
        out.append("\n### rho per class and block kind (delta 1%; U's share doesn't enter rho)\n")
        rows = []
        for cls, c in fo["classes"].items():
            for kind in ("inference", "server_window"):
                a = c[kind]
                extra = (f"x_lead {a['x_lead']:.3g}" if cls == "fan" else
                         f"(d, e) = ({g(a['d'])}, {g(a['e'])})" if cls == "thread" else
                         f"u = {p['cover_unit_products']} products a counted unit (flagged)")
                rv = c[f"rho_{kind}"]
                rows.append([cls, kind.replace("_", "-"), g(rv), "vacuous (< 1)" if rv is not None and rv < 1 else "",
                             a["regime"], extra, a["premise"] or "holds at this size"])
        out.append(table(["class", "block", "rho", "", "regime", "detail", "premise at this block's size"], rows))
        sm = fo["small_state"]
        out.append("\n### Small-state (N1): unthrottled below these thresholds\n")
        out.append(table(["block", "state + one step's inputs fit below (bit)", "params at 16 B (Adam)",
                          "params at C1's floor", "V4-Pro fits (Adam / floor)"],
                         [[k, g(v["state_bits"]), g(v["params_adam"]), g(v["params_floor"]),
                           f"{v['model_fits_adam']} / {v['model_fits_floor']}"] for k, v in sm.items()]))
        ob = fo["one_block"]
        out.append("\n### The one-block regime (premises fail for the first steps after the one registration)\n")
        out.append(table(["block", "steps whose inputs fit", "steps whose replay fits", "free steps", "binding",
                          "share of the run"],
                         [[k, g(v["by_inputs"]), g(v["by_replay"]), str(v["steps"]), v["binding"], g(v["run_share"])]
                          for k, v in ob.items()]))
        out.append("\n### (B) Monitored compute: Train_T per period, pooled, with the per-window figure beside it\n")
        out.append("D is the fleet's compute over the reference training it may do (the effective slowdown); "
                   "binding: a = the server-window, b = inference blocks plus U.\n")
        rows = []
        for m in fo["monitored"]:
            u = m["u"]
            rows.append([m["horizon"], g(m["delta"]), m["class"], m["mode"], g(m["train_flop"]), g(m["D"]),
                         g(m["runs"]), dur(m["t_step"]), dur(m["t_run"]), dur(m["coordinated_t_run"]),
                         f"{g(u['total'])} = {g(u['steady'])} + {g(u['lag'])} + {g(u['escape'])} + {g(u['inject'])}",
                         m["binding"], g(m["window_train_flop"]), g(m["window_D"])])
        out.append(table(["horizon", "delta", "class", "mode", "Train_T (FLOP)", "D", "runs", "t(step), pooled",
                          "t(run), pooled", "t(run), coordinated", "U share = steady + lag + escapes + injected",
                          "binds", "per window (FLOP)", "D per window"], rows))
        out.append("\n### Cut (a) by R's scope and the window's length (year, delta 1e-6)\n")
        rows = []
        for x in fo["scenarios"]:
            rows.append([x["R_scope"], dur(x["window_s"]), g(x["K"]), g(x["K"] / whole_update_bits(p))]
                        + [f"{g(x[c]['rho'])}; D {g(x[c]['D_onsite'])}" + (f"; {x[c]['premise']}" if x[c]["premise"]
                                                                            else "") for c in CLASSES])
        out.append(table(["R per", "window", "K (bit)", "K / whole update"]
                         + [f"{c}: rho server-window; D onsite" for c in CLASSES], rows))
        out.append("\n### (A) Dark islands, per dark share and horizon\n")
        rows = []
        for d in fo["dark"]:
            i = d["island"]
            rows.append([d["horizon"], g(d["share"]), g(d["locations"]), dur(i["t_step"]), dur(i["t_run"]),
                         g(d["island_runs_in_H"]), i["fits"], dur(i["t_offload"]), dur(d["coop_t_step"]),
                         dur(d["coop_t_run"]), g(d["coop_beats_island_above_bps"]), dur(d["capacity_t_run"]),
                         g(d["capacity_runs"])])
        out.append(table(["horizon", "dark share", "islands", "island t(step)", "island t(run)", "runs in H (island)",
                          "state fits", "offload floor a step", "cooperating t(step)", "cooperating t(run)",
                          "coop beats one island above (b/s)", "capacity t(run), no coordination",
                          "capacity runs in H"], rows))
        out.append("\n### Exfiltration (side channels at beta_side per location)\n")
        rows = []
        for share, refs in fo["exfil"].items():
            for ref, e in refs.items():
                rows.append([g(share), ref, g(e["bits"]), dur(e["per_location"]), dur(e["dark_pooled"]),
                             dur(e["monitored_pooled"]), dur(e["remote_line"]), g(e["faulty_outputs_p_detect"]),
                             e["legitimate"]])
        out.append(table(["dark share", "reference", "bits", "one location (island or onsite server)",
                          "dark share pooled", "monitored fleet pooled (onsite)", "remote: one server's line rate",
                          "as failing blocks' outputs: P(caught)", "pooling legitimate"], rows))
        out.append(f"\n### (A) and (B) together (delta {g(p['deltas'][-1])})\n")
        rows = [[t["horizon"], g(t["share"]), t["class"], t["mode"], dur(t["capacity_t_run"]),
                 g(t["capacity_runs_in_H"]), dur(t["coordinated_t_run"])] for t in fo["together"]]
        out.append(table(["horizon", "dark share", "class", "mode", "t(run) at capacity, no coordination",
                          "runs in H at capacity", "t(run), coordination-limited"], rows))
        out.append("\n### Side-channel sweep (dark share 1% for the cooperating run; year, delta 1e-6)\n")
        rows = [[g(x["beta_side"]), g(x["K"]), g(x["side_share_of_K"]), g(x["rho_sw_fan"]), g(x["D_onsite_fan"]),
                 g(x["D_onsite_thread"]), g(x["D_onsite_cover"]), dur(x["island_coop_t_run"]),
                 dur(x["weights_per_location_s"])] for x in fo["beta_sweep"]]
        out.append(table(["beta_side (b/s)", "K (bit)", "side share of K", "rho server-window (fan)", "D onsite (fan)",
                          "D onsite (thread)", "D onsite (cover)", "cooperating t(run)", "weights, one location"],
                         rows))
        out.append("\n### Dark-share sweep (year)\n")
        rows = [[g(x["share"]), g(x["locations"]), dur(x["capacity_t_run"]), dur(x["coop_t_run"]),
                 dur(x["weights_pooled_s"]), dur(x["syncs_pooled_s"])] for x in fo["dark_sweep"]]
        out.append(table(["dark share", "islands", "capacity t(run)", "cooperating t(run)", "weights pooled",
                          "1,000 syncs pooled"], rows))
        for metric, rows_t in fo["tornado"].items():
            out.append(f"\n### Tornado: {metric} (year, delta 1e-6)\n")
            out.append(table(["parameter", "value", "low", "high", "metric at low", "metric at high", "elasticity",
                              "range source"],
                             [[x["key"], val(x["value"]), g(x["low"]), g(x["high"]), g(x["at_low"]),
                               g(x["at_high"]), elast(x["elasticity"]), x["source"]] for x in rows_t]))
    return "\n".join(out) + "\n"


def plots(r: dict, outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    colors = dict(zip(CLASSES, ("#0072B2", "#D55E00", "#009E73"), strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, (name, fo) in zip(axes, r["fleets"].items(), strict=True):
        xs = [x["beta_side"] for x in fo["beta_sweep"]]
        remote = {m["class"]: m["D"] for m in fo["monitored"]
                  if m["horizon"] == "year" and m["delta"] == 1e-6 and m["mode"] == "remote"}
        for cls in CLASSES:
            ax.plot(xs, [x[f"D_onsite_{cls}"] for x in fo["beta_sweep"]], color=colors[cls], lw=2,
                    label=f"{cls}, onsite")
            ax.axhline(remote[cls], ls=":", lw=1.5, color=colors[cls], label=f"{cls}, remote")
        ax.axvline(fo["block"]["beta_takeover"], ls="--", color="k", lw=1,
                   label=f"side channels = R + I_cert ({g(fo['block']['beta_takeover'])} b/s)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{name}: D over a year (delta 1e-6)")
        ax.set_xlabel("side channels per location (bit/s)")
        ax.legend(fontsize=8, loc="center left")
    axes[0].set_ylabel("D, fleet compute / reference training")
    fig.tight_layout()
    fig.savefig(outdir / "beta_side.png")
    names.append("beta_side.png")
    plt.close(fig)
    for name, fo in r["fleets"].items():
        for metric, rows in fo["tornado"].items():
            fig, ax = plt.subplots(figsize=(12, 7))
            rows = rows[:12][::-1]
            base = rows[0]["base"] if rows else 1
            ends = [v for x in rows for v in (x["at_low"], x["at_high"])] + [base]
            for i, x in enumerate(rows):
                lo, hi = sorted((x["at_low"], x["at_high"]))
                ax.barh(i, hi - lo, left=lo, color="#0072B2")
                ax.text(hi * 1.05, i, f"{g(x['low'])} -> {g(x['at_low'])}, {g(x['high'])} -> {g(x['at_high'])}",
                        va="center", fontsize=8)
            ax.axvline(base, color="k", lw=1)
            ax.set_yticks(range(len(rows)), [x["key"] for x in rows])
            ax.set_xscale("log")
            ax.set_xlim(min(ends) / 1.5, max(ends) * 8)
            ax.set_title(f"{name}: {metric} (year, delta 1e-6), tornado: parameter end -> metric")
            fig.tight_layout()
            fn = f"tornado_{name}_{metric}.png"
            fig.savefig(outdir / fn)
            names.append(fn)
            plt.close(fig)
    return names


def _json(x):
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    if isinstance(x, dict):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json(v) for v in x]
    return x


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="every result as JSON")
    ap.add_argument("--out", type=Path, help="write tables.md, results.json and plots here")
    ap.add_argument("--params", type=Path, default=PARAMS, help="a parameter file (default reference_dc.json)")
    args = ap.parse_args(argv)
    p, fleets = load(args.params)
    r = evaluate(p, fleets, args.params)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "tables.md").write_text(render(p, fleets, r, args.params))
        (args.out / "results.json").write_text(json.dumps(_json(r), indent=1))
        (args.out / "parameters.json").write_text(args.params.read_text())
        print("\n".join(["tables.md", "results.json", "parameters.json", *plots(r, args.out)]))
        return 0
    print(json.dumps(_json(r), indent=1) if args.json else render(p, fleets, r, args.params))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
