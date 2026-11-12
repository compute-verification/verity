"""The slowdown calculator (`verity/protocols/inference_only/rho`) as this package reads it: the one module here that
names it, and the benchmark's one entry point to a slowdown.

`slowdown(policy, step, run=None)` is that entry point: from a policy's block and a step's record to the lower bound
on the slowdown and what binds. The grade (`formalism.bound`, which `grade.slowdown_of` reads) and the library's table
(`library.bound`) both call it, and nothing else of this benchmark computes a slowdown from a step
(`tests/test_one_calculator.py`). It reads the work the record states (`training_slowdown.work`: M, the products whose
two operands both depend on the update), chooses the subset of M the bound is largest on, and asks the calculator's
`min_cut.bound`, the statement its `min_cut` names (`theorem`) solved for the fewest blocks.

What else the calculator states is read from it through its public functions, never restated here:

- a fixed policy's numbers (`policy`, `policies`): the `[policy.*]` tables of its registry, Γ in FLOP and K in bits;
- the compute a figure is presented over (`fleets`): its `periods.FLEETS`, one datacenter and the world, each a fleet
  of its registry, in AND gates a second;
- the monitoring periods (`periods`): its `PERIODS`, core's, with the window of the census's reference deployment
  before them;
- (H)'s rows of β(k) (`beta_rows`) and the statement's Lean name (`theorem`), which published words state.

No other module or test of this benchmark imports the calculator or names its directory
(`tests/test_one_calculator.py`).

The calculator counts in FLOP and in its own gates (`units.gates_per_flop`); training's slowdown code counts in the
gates of the circuits' Definitions (`training_slowdown.gates`). A figure crosses in FLOP, two a MAC, never in gates.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import tomllib
from functools import cache
from pathlib import Path

from . import work as W
from .gates import per_and, per_mac

#: the calculator's directory, whose name is its package's too
CALCULATOR = Path(__file__).resolve().parents[3] / "verity" / "protocols" / "inference_only" / "rho"
DEPLOYMENTS = Path(__file__).resolve().parents[2] / "census" / "deployments.json"
#: the precision the registry's locations state their peak rate at: the census's dense FP8 E4M3
FLEET_PRECISION = "e4m3"
#: the theorem's inputs on the work `slowdown` chooses, as its `figure` names them and its bound's `work` carries them
#: beside |C| (`C_gates`): N in MACs, O in wires, the bits a word of each operand holds, an output's bits, and F
WORK = ("N", "O", "beta_w", "beta_x", "b_y", "F")
#: what a row calls the β an activation's word holds at k, the calculator's rows (`beta_rows`). They are a derived
#: activation's; a source activation (a norm's output, a layer's output gradient) holds at least as much, so every
#: activation takes them
BETA_K_LABEL = "β(k) per the red-team of (H)"
#: the subsets of M `slowdown` takes the best of: all of M, M less its short contractions, and the weight gradients
#: alone
HE_SUBSETS = ("M", "M without its short contractions", "the weight gradients alone")
#: what `slowdown` says binds: the theorem's count on M (`products`), nothing (`none`: the slowdown is the floor's 1),
#: or `unmeasured` where the record gives no slowdown
BINDS = ("products", "none", "unmeasured")
#: why there is no slowdown on a trace the capture refused
REFUSED = "refused: the capture refused the step ({reason}, {op}: {message})"
#: why there is no slowdown where neither the run nor the record's circuit gives |C|
NO_GATES = "unmeasured: neither the run nor the step's record states the circuit's gates, which the slowdown is over"


@cache
def _calculator():
    """The calculator's `periods` (its fleets), its package's `envelopes`, `hardware` and `params`, and its registry.
    Its directory leads the import path, as its own runner puts it: `periods` imports its sibling `run` by that name."""
    if sys.path[0] != str(CALCULATOR):
        sys.path.insert(0, str(CALCULATOR))
    import periods
    from rho import envelopes, hardware, params

    return periods, envelopes, hardware, params, params.Registry.load()


def policies() -> list[str]:
    """The names of the registry's fixed policies: the `[policy.*]` tables of its data files, read without importing
    the calculator, since a job checks its config's policy by name and runs none of the calculator's code."""
    return sorted({name for f in (CALCULATOR / "data").glob("*.toml")
                   for name in tomllib.loads(f.read_text()).get("policy", {})})


def policy(name: str) -> dict | None:
    """The registry's policy `name`: Γ in FLOP (`gamma_flop`) and K in bits (`k_bits`); None where it has none."""
    if name not in policies():
        return None
    _, envelopes, _, _, registry = _calculator()
    p = envelopes.policy(registry, name)
    return {"gamma_flop": p.gamma, "k_bits": p.k}


def window() -> tuple[str, float]:
    """The census's reference deployment and its window in seconds."""
    d = next(d for d in json.loads(DEPLOYMENTS.read_text()) if d["role"] == "reference")
    return d["id"], float(d["window_s"])


def periods() -> dict[str, float]:
    """The periods a figure is presented over, in seconds: the reference deployment's window, then the calculator's
    monitoring periods."""
    return {"window": window()[1]} | {name: float(s) for name, s in _calculator()[3].PERIODS.items()}


def fleets() -> list[dict]:
    """The calculator's fleets (`periods.FLEETS`), each with its label, its locations and its compute in AND gates a
    second: its registry's peak FLOP a second, two a MAC, each MAC a GEMM's AND gates at `FLEET_PRECISION`."""
    mod, _, hardware, _, registry = _calculator()
    units, kind = hardware.units(registry), f"gemm/{FLEET_PRECISION}"
    out = []
    for name, label in mod.FLEETS:
        f = hardware.fleet(registry, name)
        flops = units.flops(f.count * hardware.location(registry, f.location).phi)
        out.append({"fleet": name, "over": label, "locations": f.count, "location": f.location, "flop_per_s": flops,
                    "and_per_s": flops / 2 * per_mac(kind) / per_and(kind)})
    return out


@cache
def _min_cut():
    """The calculator's module for the bound, its `min_cut`, loaded by its path: it is standard library only and
    imports nothing of its package, so reading it loads no registry and leaves the import path as it was."""
    path = CALCULATOR / CALCULATOR.name / "min_cut.py"
    spec = importlib.util.spec_from_file_location("training_slowdown_min_cut", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def beta_rows() -> tuple[tuple[float, float], ...]:
    """The rows (k, β) of (H)'s β for an activation at a block of k bits, k rising (the calculator's `min_cut.BETA`):
    between two of them the next one's, none past the last."""
    return _min_cut().BETA


def theorem() -> str:
    """The statement the bound of record instantiates, by the Lean name the calculator's `min_cut` gives it (its
    `HE`; `HEShared` where the products share arrays)."""
    return _min_cut().HE


@cache
def he_past() -> str:
    """Why there is no slowdown past the last k the calculator has a β(k) at."""
    return (f"unmeasured: no β an activation holds is known past k = {beta_rows()[-1][0]:.0e} "
            f"({BETA_K_LABEL} stops there)").replace("e+0", "e")


def _bound(c_gates: float, k_bits: float, work: dict | None) -> dict:
    """The bound of record at the policy (c, k) on a circuit's `work` (`WORK` and its `C_gates`): what every partition
    of that circuit into the policy's blocks pays, the `slowdown` of the calculator's `min_cut.bound` (the statement's
    count solved for the fewest blocks, times c/|C|, and at least 1), with its least `blocks`, its `statement` and
    what it `rests_on`. A policy here states no output bits a block, so the function's n is None. Not `available`,
    with what is missing, where the circuit has no `work`."""
    out = {"available": False, "theorem": theorem(), "c_gates": c_gates, "k_bits": k_bits}
    if work is None:
        return out | {"why": "not available: the bound reads a traced step's N and O on M, and the config's search "
                             "record gives none"}
    b = _min_cut().bound(c_gates, k_bits, None, N=work["N"], O=work["O"], size=work["C_gates"],
                         beta_w=work["beta_w"], beta_x=work["beta_x"], b=work["b_y"], F=work["F"])
    return out | {"available": True, "theorem": b.statement, "slowdown": b.slowdown, "blocks": b.blocks,
                  "rests_on": b.rests_on, "work": work}


def _figure(he: dict, c: float, C: float, tau: float | None = None, beta_code: float | None = None) -> dict:
    """The choice of the work on M (`work.he_of_rows`) at (c, he's k) and |C|: the best of three valid choices of M
    (`HE_SUBSETS`), each at its best threshold, with its term (`rho`: the bound of record's least blocks times c/|C|,
    before the floor at 1, and 0 where no choice is positive). Each operand is charged its β, a
    weight's at most `beta_code` (the run record's code bits a trained weight), an activation's at most β(k)
    (the calculator's `min_cut.beta`); from `work.HEAD_K` the head's products leave M; and where M is over a run's
    steps (he's `steps`, of `step_tokens` tokens each), their N and O lose the ⌊k/(τ·T)⌋ steps after the first update
    (a block of k bits holds the inputs they read), τ the bits a token's input costs. On a subset, for each product P
    of a group's two β, M_P its groups with product at least P, charged β_w·β_x = P (the group's weight's β and its
    activation's, or its wider and narrower where its operands are of one class), the least pair among M_P's products,
    which F needs where a layer's arrays differ in β (Motzkin-Straus sums u_r·u_s/(β_r·β_s) over the pairs M_P
    multiplies), b_y M_P's least output width and F = 2/√3 where M_P holds a backward product (the calculator's
    `min_cut.share`: a layer's forward, input-gradient and weight-gradient products share arrays; Motzkin-Straus bounds
    π by 4/3 where no four arrays are pairwise multiplied), else 1, and 1 on the weight gradients alone; the subset's
    term is the largest bound of record over P (`_bound`'s `blocks` times c/|C|). The second subset leaves out of M's
    best the groups whose contraction K is below K_thr = (F·k/2)²·b_y/(β_w·β_x·(k + α)): on a group that short the
    count's outputs' term is at least its main one at the m = N/(k + α) its MACs alone would need. What it read: the
    binding subset, each subset's term, β_w, β_x, β (their geometric mean), b_y, F, α (the calculator's
    `min_cut.alpha` times F), N, O, K_thr, β(k), `beta_code` where it bounds a weight in M (`code_bound`), the first
    steps left out, M_P's share of the candidates' MACs, whether an operand's own β (its rate or β_eff, before either
    bound) is below its width (`below_width`) and the step's r_step."""
    k, mod = he["k"], _min_cut()
    bk = mod.beta(k)
    if bk is None:
        raise ValueError(f"the work at k = {k:g}: {he_past()}")
    first, scale = None, 1.0
    if he.get("steps") and he.get("step_tokens") and tau:
        first = math.floor(k / (he["step_tokens"] * tau))
        scale = max(0.0, he["steps"] - first) / he["steps"]

    def charged(g: dict) -> list[float]:
        return [(b if beta_code is None else min(b, beta_code)) if w else min(b, bk)
                for b, w in zip(g["beta"], g.get("weight", (False, False)))]

    groups = [(math.prod(charged(g)), charged(g), g) for g in he["groups"] if not (k >= W.HEAD_K and g.get("head"))]

    def w_x(cb: list[float], g: dict) -> tuple[float, float]:
        w = g.get("weight", (False, False))
        if sum(w) == 1:
            return (cb[0], cb[1]) if w[0] else (cb[1], cb[0])
        return tuple(sorted(cb, reverse=True))

    def best(gs: list, one: bool = False) -> dict | None:
        out = None
        for P in sorted({p for p, _, _ in gs if p > 0}, reverse=True):
            M = [x for x in gs if x[0] >= P]
            bw, bx = next(w_x(cb, g) for p, cb, g in M if p == P)
            N, O = scale * sum(g["macs"] for _, _, g in M), scale * sum(g["wires"] for _, _, g in M)
            b_y = min(g["b_y"] for _, _, g in M)
            F = 1.0 if one or all(g["forward"] for _, _, g in M) else mod.share(3)
            work = {"N": N, "O": O, "beta_w": bw, "beta_x": bx, "b_y": b_y, "F": F, "C_gates": C}
            raw = _bound(c, k, work)["blocks"] * c / C
            if out is None or raw > out["raw"]:
                out = {"raw": raw, "beta_w": bw, "beta_x": bx, "beta": math.sqrt(P), "b_y": b_y, "F": F,
                       "alpha": F * mod.alpha(k, P, b_y), "N": N, "O": O,
                       "code_bound": beta_code is not None and any(
                           w and beta_code < b for _, _, g in M for b, w in zip(g["beta"], g.get("weight", ()))),
                       "below_width": any(b < w for _, _, g in M for b, w in zip(g["beta"], g["width"])),
                       "macs": sum(g["macs"] for _, _, g in M)}
        return out

    whole = best(groups)
    K_thr = None
    subsets = [whole]
    if whole is not None:
        K_thr = (whole["F"] * k / 2) ** 2 * whole["b_y"] / (whole["beta_w"] * whole["beta_x"] * (k + whole["alpha"]))
        subsets.append(best([x for x in groups if x[2].get("k_dim", math.inf) >= K_thr]))
    else:
        subsets.append(None)
    subsets.append(best([x for x in groups if x[2].get("dw")], one=True))
    total = he["candidates_macs"]
    common = {"K_thr": K_thr, "beta_k": bk, "beta_code": beta_code, "first_steps": first, "r_step": he["r_step"],
              "subsets": {name: None if s is None else max(0.0, s["raw"]) for name, s in zip(HE_SUBSETS, subsets)}}
    on = [(s["raw"], -i, name, s) for i, (name, s) in enumerate(zip(HE_SUBSETS, subsets)) if s is not None]
    if not on:
        return {"rho": 0.0, "subset": None, "beta_w": None, "beta_x": None, "beta": None, "b_y": None, "F": None,
                "alpha": None, "N": 0.0, "O": 0.0, "macs_share": 0.0, "below_width": False,
                "code_bound": False} | common
    _, _, name, top = max(on)
    top = dict(top)
    raw, macs = top.pop("raw"), top.pop("macs")
    return {"rho": max(0.0, raw), "subset": name} | top | {"macs_share": macs / total if total else 0.0} | common


def slowdown(policy: dict, step: dict | None, run: dict | None = None) -> dict:
    """The benchmark's one slowdown: a lower bound on what every partition of a step's circuit into the policy's
    blocks pays, the calculator's bound of record (`min_cut.bound`) on the work the step's record states.

    `policy` is the block: its `c_gates` and `k_bits` (`training_slowdown.policy.named`'s). `step` is a step's record:
    a job's trace (`run.trace_step`'s, its raw product rows priced here, `work.priced`), or the products stated from
    one over a run's steps (`work.products_of_trace`'s, which `formalism.measure` keeps as `Measured.products`); None
    is a record with neither. `run` is what the run the step stands for adds, each where it has it: its circuit's
    `gates` (|C|; without it a trace's own circuit's), the bits a token's input costs (`tau_bits`) and the bits of its
    update code (`code_bits`).

    Returns
    - `slowdown`: the bound's, at least 1; None where the record gives none, with the reason in `why`;
    - `binds` (`BINDS`): `products` where the theorem's count on M exceeds 1; `none` where it doesn't, or M is empty,
      so the slowdown is the floor's 1; `unmeasured` where there is no slowdown;
    - `term`: the count's figure before the floor, the least blocks times c/|C|, never below 0 and 0 where M is
      empty; None where there is no slowdown;
    - `bound`: the calculator's bound on the work chosen: `available`, its `theorem` by its Lean name, `c_gates`,
      `k_bits`, its `slowdown`, its least `blocks`, what it `rests_on` and the `work` it read (`WORK` and `C_gates`);
      not `available`, with `why`, where the record gives no work or M is empty (no operand holds a bit, so there
      is no β to hand over);
    - `figure`: the choice of the work (`_figure`'s: the subset of M that binds, each subset's term and what the
      choice read); None where there is no slowdown;
    - `read`: what the record says of itself: the k its rows were classified at (`k_classified`), its `capture` and,
      where it has them, its `capture_code`, `stand_in` and `other_program`, and its `p_match` where that is why it
      has no slowdown.

    The work is M at the policy's k (the record's own, or a trace's at each k of the console's dial,
    `work.classified_at`), the best of three choices of it (`HE_SUBSETS`). A weight's β is at most the run's code
    bits over M's trained weights where M counts them, and τ is the run's. The bound needs no summation order and no
    premise that blocks hold whole multiply cells (a block's split MACs are charged k a block), so a step that writes
    products with elementwise ops has one (they are left out of M, and stay in |C|), and so does a circuit that
    declares a tree.

    There is no slowdown on a trace the capture refused (`REFUSED`), on a record with no product rows
    (`work.PRODUCTS_UNMEASURED`), on a trace of another program (`work.OTHER_PROGRAM`), at a k the record's rows
    weren't classified at, on rows that predate `work.SYSTEM_FROM` (`work.HE_RETRACE`), on products without M
    (`work.HE_NO_ROWS`), on a step whose elementwise ops fan out into an operand (`work.p_match_reason`), past the
    calculator's last β(k) (`he_past`), from `work.HEAD_K` on products that don't say which are the head's
    (`work.HE_HEAD`), and where neither the run nor the record's circuit gives |C| (`NO_GATES`)."""
    c, k = policy["c_gates"], policy["k_bits"]
    out = {"slowdown": None, "binds": "unmeasured", "why": None, "term": None, "bound": _bound(c, k, None),
           "figure": None, "read": {}}
    gates = (run or {}).get("gates")
    if step is not None and "refused" in step:
        r = step["refused"]
        return out | {"why": REFUSED.format(reason=r.get("reason"), op=r.get("op"), message=r.get("message"))}
    if step is not None and "rows" in step:
        if gates is None:
            gates = (step.get("circuit") or {}).get("gates")
        step = W.products_of_trace(W.priced(step))
    if step is None:
        return out | {"why": W.PRODUCTS_UNMEASURED}
    p = W.classified_at(step, k) or step
    read = {"k_classified": p["k"]} | {x: p[x] for x in ("capture", "capture_code", "stand_in", "other_program")
                                       if x in p}
    out |= {"read": read}
    if "other_program" in p:
        o = p["other_program"]
        largest = ", ".join(f"{m}×{K} by {K}×{n}" for m, K, n in o["largest"])
        return out | {"why": W.OTHER_PROGRAM.format(unmatched=o["unmatched"], shapes=o["shapes"], ran=o["ran"],
                                                    largest=largest)}
    if not math.isclose(p["k"], k, rel_tol=1e-12):
        return out | {"why": (f"unmeasured: the products were classified at k = {p['k']:g}, not at this k, and keep "
                              f"none at it (a trace from {W.THE_CODE} {W.SYSTEM_FROM} on keeps them at each k of the "
                              f"console's dial); take them again at it from the run's captured step or its trace")}
    he = p.get("he")
    if he is None:
        return out | {"why": W.HE_RETRACE if "he" in p else W.HE_NO_ROWS}
    if p.get("p_match"):
        return out | {"why": W.p_match_reason(p["p_match"]), "read": read | {"p_match": p["p_match"]}}
    if _min_cut().beta(k) is None:
        return out | {"why": he_past()}
    if k >= W.HEAD_K and any(g.get("head") is None for g in he["groups"]):
        return out | {"why": W.HE_HEAD}
    if gates is None:
        return out | {"why": NO_GATES}
    bits = (run or {}).get("code_bits")
    code = bits / he["trained_weights"] if bits and he.get("trained_weights") else None
    figure = _figure(he, c, gates, (run or {}).get("tau_bits"), code)
    if figure["subset"] is None:
        return out | {"slowdown": 1.0, "binds": "none", "term": figure["rho"], "figure": figure}
    b = _bound(c, k, {x: figure[x] for x in WORK} | {"C_gates": gates})
    return out | {"slowdown": b["slowdown"], "binds": "products" if b["slowdown"] > 1 else "none",
                  "term": figure["rho"], "bound": b, "figure": figure}
