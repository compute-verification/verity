"""Front F6's per-unit-type cost inventory: for every served workload in the census and every unit type in its template mix
(``vu_family`` + parameters), what one unit costs as a Boolean circuit, how many units an audit draws, and the unit's public
work weight.

    python -m verity_numerical.bench.unit_costs --out DIR [--exact-nnz 4e8] [--jobs 8] [--workload SUBSTRING] [--reuse JSON]

**Registered lowerings (the primary count).** Where the template has a registered C-Flock lowering (``bench.lowerings``,
:func:`registered`), a subcircuit is that lowering's ``units.n`` identical units, each one ``flock-ir-unit/v2`` layout: the
matrix a session evaluates, with the intermediates the lowering commits as rows and the native tail outside it. A unit's
committed rows are ``units x rows`` and its nonzeros ``units x (nnz_a + nnz_b + rows)`` (C = I over rows). Attention's
units are its tensor-core steps, :func:`attention_steps` (checked against the lowering at :data:`ATTENTION_REG_T`),
evaluated at the stratum's mean T and mean ceil(T / 16). Bindings the lowering refuses (its ``supports`` reason, or Hopper
GEMM) and templates without one (token-select, MoE, FP8 GEMM, element-wise scalars) keep the generic count below, marked
``source: generic``; every row also carries the generic count as ``nnz_generic``. token-select is priced at
``TokenSelect_v2``'s argmax tree (:func:`argmax_tree`, :func:`retree`, ``source: tree``), which supersedes its generic scan
count (kept as ``nnz_superseded``) and its ``nnz_step_model``, a cut scan at (V − 1) middle ArgmaxStep units
(:func:`argmax_step`); gumbel's rows carry their select's F32 tree as ``select_tree``. ``--retree JSON`` applies only that to
an earlier inventory. :func:`reconcile` sets circuits' F3 unit figures beside these.

**Gates and nonzeros (generic).** A unit type's subcircuit is its census template bound at the census parameters
(``bench.templates.subcircuit``), and its circuit is that Definition lowered whole, piece by piece, by C-Flock's generic
lowering (``verity_flock.ir_lower.lower``) into ``verity.ml.circuits.boolean.forms.Circuit`` with CSE, the builder C-Flock's
units use. ``counts()`` gives the live cone's committed input bits, ANDs, output copies and ``nnz_AB`` (the nonzeros of A and
B); C = I adds one nonzero per AND row, so a unit's C-Flock nonzeros are ``nnz_AB + ands``. Table reads (``lookups``) are not
ANDs; their tables and widths are listed.

Every number says how it was obtained (``method``):

- ``exact``: the circuit built at the census parameters.
- ``affine``: built at four small values of one parameter (:data:`AFFINE`: GEMM-like K, an element-wise width), every count
  checked to be exactly affine in it (fitted on the first two points, equal on the others), and evaluated at the census
  value. When the census value is within ``--exact-nnz`` the circuit is also built there and must agree. Attention: the
  head at the census D, BN and semantics built at key counts T within the first key block (:data:`ATTENTION_T`), checked
  affine in T, and built at T = BN + 16 and 2 BN + 16 (:data:`BLOCK_CHECKS`), whose excess over that fit must be one and
  two times one block's; a stratum of NB blocks is then the fit plus NB − 1 blocks' excess, evaluated at the stratum's
  mean T. The builds are at 16-aligned T: PV runs in 16-key steps, so a T between pads its last step.
- ``scan law`` (token-select): not affine in V (:func:`argmax_scan_law`); the law is read off builds at every V in
  :data:`SCAN_POINTS`, checked by an exact build at :data:`SCAN_CHECK`, and evaluated at the census V.
- ``model`` (attention whose excess past the first key block is not affine in the block count, :func:`block_law`): that
  excess is fitted with a curvature through the three points, unchecked; or, when those builds are over ``--exact-nnz``,
  the first block's fit alone. Not exact.

``--reuse unit-costs.json`` re-derives the inventory from an earlier run's subcircuit results instead of building.
- ``missing``: no piece for a primitive of the Definition (the reason names it), or ``excluded`` (embedding-row is a
  commitment opening, not proved by a proof backend), or ``not built`` (a structured reduction whose census size is over
  ``--exact-nnz``).

A unit = one VU is ``subcircuits_per_vu`` subcircuits: its counts are that multiple of the subcircuit's, without the
sharing a VU-level circuit could make between its subcircuits (an operand they share decoded once), so an upper bound on
ANDs and nonzeros.

**Sampled instances.** The draw is uniform over all units, because a hidden partition cannot be stratified by type: K =
``record_k(ε, δ)`` units (``verity.core.protocols.pouw.circuit.plan``, 27,713 at the default ε = 0.1%, δ = 2⁻⁴⁰) without
replacement from the workload's N evaluable units, at each granularity. A type with n of the N units gets a hypergeometric
count: its mean ``K n / N`` and its upper ``1 − δ`` quantile, the least k with P(X > k) ≤ δ, summed exactly over the tail
in log space.

**Work weight.** The template's census ``work_model``: ``ops_per_input`` and ``bytes_per_input`` at the bound parameters
(``BITS`` the datatype's width; attention's T as above), times ``subcircuits_per_vu`` for a VU.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import multiprocessing
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import census

METRICS = ("inputs", "ands", "output_copies", "committed", "nnz_AB", "lookups")
EPSILON = Fraction(1, 1000)
DELTA = Fraction(1, 2 ** 40)

#: template -> (the parameter the circuit is affine in, its small build points given the other parameters)
AFFINE: dict[str, tuple[str, Callable[[Mapping[str, Any]], tuple[int, ...]]]] = {
    "gemm-coordinate": ("K", lambda p: (32, 48, 64, 80)),
    "moe-expert-coordinate": ("K", lambda p: (32, 48, 64, 80)),
    "moe-expert-coordinate-weighted": ("K", lambda p: (32, 48, 64, 80)),
    "scaled-mm-fp8-block-coordinate": ("K", lambda p: tuple(int(p["G"]) * m for m in (2, 3, 4, 5))),
    "rope-head": ("D", lambda p: (16, 32, 48, 64)),
    "silu-mul": ("I", lambda p: (16, 32, 48, 64)),
    "bias-add": ("N", lambda p: (16, 32, 48, 64)),
    "bf16-mul-scalar": ("N", lambda p: (16, 32, 48, 64)),
    "moe-sum": ("H", lambda p: (16, 32, 48, 64)),
    "gumbel-top-p-token-select": ("V", lambda p: (64, 128, 192, 256)),
}
#: token-select's V − 1 ArgmaxStep scan: the law (:func:`argmax_scan_law`) is read off builds at every V in SCAN_POINTS and
#: checked against an exact build at SCAN_CHECK
SCAN_POINTS = tuple(range(2, 140))
SCAN_CHECK = 1030
#: templates whose Definition has a structured reduction (chunks, thread blocks): built only at the census parameters
EXACT_ONLY = ("rmsnorm-fused-cuda", "rmsnorm-triton", "moe-router-topk")
#: a small instance of each exact-only template, to estimate the census size's nonzeros (linear in the width) before building it
PROBE = {"rmsnorm-fused-cuda": "N", "rmsnorm-triton": "N", "moe-router-topk": None}
EXCLUDED = {"embedding-row": "commitment opening (census `commitment_opening`): opened against the commitment, not proved"}
#: attention's key counts T, within the first key block, at which the head is built
ATTENTION_T = (16, 32, 48, 64)
#: j for the checks at T = j BN + 16, past the first key block: their excess over the first block's fit must be j times one
#: block's
BLOCK_CHECKS = (1, 2)


# --- building --------------------------------------------------------------------------------------------------------------

def build(template: str, params: Mapping[str, Any], statics: Mapping[str, int] | None = None) -> dict[str, Any]:
    """The subcircuit's Definition lowered whole into a CSE'd forms Circuit: ``counts()`` plus the live table reads by table."""
    from verity.ml.circuits.evaluation.reference import resolve, signature_of
    from verity_flock import ir_lower
    from verity_flock.gf2 import Circuit

    from . import templates as TM

    d = resolve(TM.subcircuit(template, **params).definition(**dict(statics or {})))
    sig, _ = signature_of(d)
    C = Circuit(cse=True)
    ins = [C.input(f"in{j}", w) for j, w in enumerate(w for p in sig for w in p.leaf_widths())]
    for j, o in enumerate(ir_lower.lower(C, d, ins)):
        C.output(f"out{j}", o)
    out = dict(C.counts())
    live = C.live()
    groups = {C.data[v][0] for v in live if C.kind[v] == "lookup"}
    tables: dict[str, dict[str, int]] = {}
    for g in sorted(groups):
        name, idx, width = C.lookups[g]
        t = tables.setdefault(f"{name}/{len(idx)}->{width}", {"reads": 0, "index_bits": len(idx), "width": width})
        t["reads"] += 1
    out["tables"] = tables
    out["definition"] = d.id
    return out


def _build_job(job: tuple) -> tuple[tuple, dict[str, Any]]:
    template, params, statics = job
    try:
        return job, build(template, dict(params), dict(statics))
    except Exception as e:  # a missing piece (NoPiece) or an unsupported parameter: reported, never guessed around
        return job, {"error": f"{type(e).__name__}: {e}"}


def _key(template: str, params: Mapping[str, Any], statics: Mapping[str, int] | None = None) -> tuple:
    return (template, tuple(sorted(params.items())), tuple(sorted((statics or {}).items())))


def run_builds(jobs: Iterable[tuple], n_jobs: int, fn: Callable[[tuple], tuple] = _build_job) -> dict[tuple, dict[str, Any]]:
    """Each build in its own spawned process (a build's forms are only freed with its process, and two registered lowerings
    register kernels that collide); a worker that dies (the OOM killer) raises BrokenProcessPool rather than hanging."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = list(dict.fromkeys(jobs))
    if n_jobs <= 1:
        return dict(fn(j) for j in jobs)
    with ProcessPoolExecutor(n_jobs, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as ex:
        return dict(ex.map(fn, jobs))


# --- registered lowerings ----------------------------------------------------------------------------------------------------

#: the C-Flock template module's function giving the lowering a session evaluates (default ``lowering``)
REGISTERED_FN = {"gemm-coordinate": "circuit_lowering"}
SCHEME = "frame-v3/blake3-keyed"
#: attention's key counts at which its registered head is lowered, to check its tensor-core step count (:func:`attention_steps`)
ATTENTION_REG_T = (1, 15, 16, 17, 100, 128, 129, 1000)


def registered(template: str, params: Mapping[str, Any], T: int | None = None) -> dict[str, Any]:
    """The template's registered C-Flock lowering (``bench.lowerings``): its units per subcircuit and the unit's
    ``flock-ir-unit/v2`` layout counts, the matrix a session evaluates per unit with the intermediates the lowering commits.
    The native tail (``tail_gates``) is evaluated by the verifier and has no rows. ``{"none": why}`` without one."""
    from . import lowerings
    from . import templates as TM

    mod = lowerings.registry("C-Flock").module(template)
    if mod is None:
        return {"none": "no registered C-Flock lowering"}
    sub = TM.get(template).bind(**params)
    if template not in REGISTERED_FN and (why := mod.supports(sub, SCHEME)):
        return {"none": why}
    fn = getattr(mod, REGISTERED_FN.get(template, "lowering"))
    low = fn(sub, T) if T is not None else fn(sub)
    c = low.layout.counts
    return {"lowering": low.name, "units": int(low.units.n), "unit": {k: int(c[k]) for k in ("rows", "ands", "nnz_a", "nnz_b", "reads")},
            "tail_gates": len(getattr(low.units, "tail", None) or [])}


def _reg_job(job: tuple) -> tuple[tuple, dict[str, Any]]:
    template, params, statics = job
    try:
        return job, registered(template, dict(params), dict(statics).get("T"))
    except Exception as e:
        return job, {"none": f"{type(e).__name__}: {e}"}


def attention_steps(D: int, T: int) -> int:
    """The registered FA2 head's tensor-core steps: D/16 per key for QK, D per 16 visible keys for PV (a key block's PV pads
    to 16 keys; BN is a multiple of 16, so only the first-computed block pads)."""
    return D // 16 * T + D * -(-T // 16)


def registered_costs(types: Sequence[UnitType], n_jobs: int, log: Callable[[str], None] = print) -> dict[tuple, dict[str, Any]]:
    """Per distinct (template, census parameters): :func:`registered`. Attention per (D, BN, semantics): one unit layout for
    every T, the step count checked equal to :func:`attention_steps` at :data:`ATTENTION_REG_T`."""
    distinct = {_key(t.template, t.params): (t.template, t.params) for t in types if t.template not in EXCLUDED}
    jobs = []
    for template, params in distinct.values():
        Ts = ATTENTION_REG_T if template == "attention-head" else (None,)
        jobs += [_key(template, params, {"T": T} if T else None) for T in Ts]
    log(f"registered lowerings: {len(jobs)}")
    got = run_builds(jobs, n_jobs, _reg_job)
    out = {}
    for key, (template, params) in distinct.items():
        if template != "attention-head":
            out[key] = got[_key(template, params)]
            continue
        res = {T: got[_key(template, params, {"T": T})] for T in ATTENTION_REG_T}
        bad = next((r for r in res.values() if "none" in r), None)
        D = int(params["D"])
        if bad:
            out[key] = bad
        elif len({json.dumps(r["unit"], sort_keys=True) for r in res.values()}) != 1:
            out[key] = {"none": f"the head's unit layout differs between key counts: {res}"}
        elif any(r["units"] != attention_steps(D, T) for T, r in res.items()):
            out[key] = {"none": f"tensor-core steps differ from attention_steps: {[(T, r['units']) for T, r in res.items()]}"}
        else:
            r0 = res[ATTENTION_REG_T[-1]]
            out[key] = {"lowering": r0["lowering"].rsplit("/", 1)[0] + "/t{T}", "units": "attention_steps(D, T)", "unit": r0["unit"],
                        "checked_T": list(ATTENTION_REG_T), "tail_gates": {T: r["tail_gates"] for T, r in res.items()}}
    return out


def argmax_step() -> dict[str, Any]:
    """token-select has no registered lowering. Its scan's middle ArgmaxStep as one unit, the carry (best, best_i, i) as cut
    words like gumbel's registered lane: the layout a cut lowering would give per step (the first and last steps differ)."""
    from verity.ml.circuits.evaluation.reference import resolve
    from verity_flock import ir_lower
    from verity_flock.gf2 import Circuit

    from . import templates as TM

    _p, call = ir_lower.standalone(resolve(TM.subcircuit("token-select-greedy", V=8).definition()))
    gates = ir_lower.live_gates(call)
    gt = sorted((g for g in gates if g.prim.id == "Bf16GtStrict_v1"), key=lambda g: g.index)[3]
    sb = next(g for g in gates if g.prim.id == "SelectBf16_v1" and g.operands[0] == gt.index)
    si = next(g for g in gates if g.prim.id == "SelectI32_v1" and g.operands[0] == gt.index)
    ad = next(g for g in gates if g.prim.id == "I32Add_v1" and g.operands[0] == si.operands[1])
    step = [next(g for g in gates if g.index == ad.operands[1]), gt, sb, si, ad]
    inside = {g.index for g in step}
    ext = sorted({o for g in step for o in g.operands if o not in inside})
    width = {gt.operands[0]: 16, gt.operands[1]: 16, si.operands[1]: 32, si.operands[2]: 32}
    C = Circuit(cse=True)
    wires = {o: C.input(f"in{j}", width[o]) for j, o in enumerate(ext)}
    ir_lower.lower_gates(C, step, wires)
    for j, o in enumerate((sb.index, si.index, ad.index)):
        C.output(f"out{j}", wires[o])
    c = ir_lower.layout(C, [f"in{j}" for j in range(len(ext))], [f"out{j}" for j in range(3)]).counts
    return {k: int(c[k]) for k in ("rows", "ands", "nnz_a", "nnz_b")}


def _layout(d: Any) -> dict[str, int]:
    """A Boolean Definition laid out whole as one unit (CSE'd forms Circuit): rows, ANDs, nnz_a, nnz_b."""
    from verity.ml.circuits.evaluation.reference import resolve, signature_of
    from verity_flock import ir_lower
    from verity_flock.gf2 import Circuit

    d = resolve(d)
    sig, _ = signature_of(d)
    C = Circuit(cse=True)
    ins = [C.input(f"in{j}", w) for j, w in enumerate(w for p in sig for w in p.leaf_widths())]
    outs = ir_lower.lower(C, d, ins)
    for j, o in enumerate(outs):
        C.output(f"out{j}", o)
    c = ir_lower.layout(C, [f"in{j}" for j in range(len(ins))], [f"out{j}" for j in range(len(outs))]).counts
    return {k: int(c[k]) for k in ("rows", "ands", "nnz_a", "nnz_b")}


#: the argmax trees: token-select-greedy's (`TokenSelect_v2{V}`, bf16 keys) and gumbel's select (f32 keys)
TREES = {"token-select-greedy": "TokenSelectStage", "gumbel-top-p-token-select": "F32ArgmaxStage"}


def argmax_tree(V: int, stage: str = "TokenSelectStage") -> dict[str, Any]:
    """The argmax tree over V logits (``sampling_boolean.stages``): each stage is separable
    (``verity.ml.circuits.cut.separable``), so it costs its members' own layouts times their multiplicities, each member
    laid out as one unit (nnz = nnz_a + nnz_b + rows, C = I over rows)."""
    from verity.ml.circuits.cut import separable
    from verity.ml.circuits.evaluation.reference import resolve
    from verity.ml.definitions.registry import sampling_boolean as SB

    members: list[dict[str, Any]] = []
    memo: dict[str, dict[str, int]] = {}
    for st in SB.stages(V, getattr(SB, stage)):
        sep = separable(resolve(st))
        if sep is None:
            raise ValueError(f"{stage}{{V={V}}} has a stage that is not separable")
        for callee, m in sep:
            u = memo.get(callee.id) or memo.setdefault(callee.id, _layout(callee))
            members.append({"member": callee.id, "n": int(m), **u, "nnz": u["nnz_a"] + u["nnz_b"] + u["rows"]})
    tot = {k: sum(x["n"] * x[k] for x in members) for k in ("rows", "ands", "nnz_a", "nnz_b", "nnz")}
    return {"stage": stage, "V": V, "members": members, "units": sum(x["n"] for x in members), **tot,
            "nnz_per_logit": tot["nnz"] / V}


def retree(inv: dict[str, Any], trees: Mapping[tuple[str, int], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Prices token-select-greedy's rows at its argmax tree (``TokenSelect_v2``, :func:`argmax_tree`) in place: the tree's
    members become the row's units (``source: tree``), and the generic count it replaces is kept as ``nnz_superseded``.
    gumbel's rows keep their registered lane and carry their select's F32 tree as ``select_tree``. Idempotent."""
    trees = dict(trees or {})
    for r in inv["rows"]:
        if r["template"] not in TREES:
            continue
        V = int(r["subcircuit_parameters"]["V"])
        key = (r["template"], V)
        if key not in trees:
            trees[key] = argmax_tree(V, TREES[r["template"]])
        t = trees[key]
        summary = {k: t[k] for k in ("stage", "V", "units", "rows", "ands", "nnz", "nnz_per_logit")}
        if r["template"] == "gumbel-top-p-token-select":
            r["select_tree"] = summary
            continue
        if r.get("source") == "tree":
            continue
        r["source"], r["tree"] = "tree", {**summary, "members": t["members"]}
        for gran in ("vu", "subcircuit"):
            g = r[gran]
            mult = r["subcircuits_per_vu"] if gran == "vu" else 1
            old = g.get("per_unit") or {}
            c = {"units": t["units"] * mult, "committed": t["rows"] * mult, "ands": t["ands"] * mult, "lookups": 0,
                 "nnz_AB": (t["nnz_a"] + t["nnz_b"]) * mult, "nnz_C": t["rows"] * mult, "nnz": t["nnz"] * mult,
                 "nnz_generic": old.get("nnz_generic"), "nnz_superseded": old.get("nnz")}
            if "generic" in old:
                c["generic"] = old["generic"]
            g.update(per_unit=c, audit_nnz_mean=g["sampled_mean"] * c["nnz"], audit_nnz_q=g["sampled_q"] * c["nnz"])
            g.pop("nnz_range", None)
    inv["token_select"] = ("TokenSelect_v2's argmax tree: each stage separable, its members' layouts times their "
                           "multiplicities; the generic scan count is superseded (nnz_superseded)")
    return inv


#: circuits' F3 unit nonzeros (``benchmarks/private_circuit/nonzero_cost.py`` on cursor/f3-nonzero-cost-f253 @76e405893,
#: art:b56410cabae792600c86d960160b0c8da7a3ca408b867357103ce4719e07dd47): nnz_a + nnz_b of one registered unit (the RMSNorm
#: figure is the fused lowering's unit, one warp)
CIRCUITS_F3 = {"rope-pair": 156_755, "tc-step-k2048": 254_460, "silu-element": 378_759, "f32mul-ftz": 94_333,
               "rmsnorm-fused-warp-n2048": 8_045_790}


def _f32mul() -> dict[str, Any]:
    from verity.ml.circuits.boolean.forms import Circuit
    from verity.ml.circuits.ops.boolean import attention as AT
    from verity_flock import ir_lower

    out: dict[str, Any] = {}
    for cse in (False, True):
        C = Circuit(cse=cse)
        a, b = C.input("in0", 32), C.input("in1", 32)
        C.output("out0", AT.F32MulFtz.build(C, a, b))
        c = ir_lower.layout(C, ["in0", "in1"], ["out0"]).counts
        out["cse" if cse else "plain"] = {k: int(c[k]) for k in ("rows", "ands", "nnz_a", "nnz_b")}
        if cse:
            out["generic_nnz_AB"] = int(C.counts()["nnz_AB"])
    return out


def reconcile(regs: Mapping[tuple, Mapping[str, Any]], subs: Mapping[tuple, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """circuits' five F3 units against this inventory: the registered unit's nnz_AB, the units per subcircuit, and the
    subcircuit's registered and generic nnz_AB."""
    cases = [("rope-pair", "rope-head", {"D": 64}, "pair; a head is D/2 pairs"),
             ("tc-step-k2048", "gemm-coordinate", {"K": 2048, "datatype": "bf16", "semantics": "sm80.mma.m16n8k16.bf16"},
              "tensor-core k-step; a coordinate is K/16 steps"),
             ("silu-element", "silu-mul", {"I": 8192}, "element; a row is I elements"),
             ("rmsnorm-fused-warp-n2048", "rmsnorm-fused-cuda", {"N": 2048, "EPS": 1e-05},
              "CUDA warp; a row is min(N, 1024)/32 warps, its sum/rsqrt tail native")]
    out = []
    for name, template, params, what in cases:
        reg, sub = regs.get(_key(template, params), {}), subs.get(_key(template, params), {})
        row: dict[str, Any] = {"unit": name, "circuits": CIRCUITS_F3[name], "what": what, "template": template, "params": params}
        if "unit" in reg:
            u = reg["unit"]
            row.update(unit_nnz_AB=u["nnz_a"] + u["nnz_b"], units=reg["units"], subcircuit_nnz_AB=reg["units"] * (u["nnz_a"] + u["nnz_b"]))
        if sub.get("counts"):
            row["generic_subcircuit_nnz_AB"] = sub["counts"]["nnz_AB"]
        out.append(row)
    f = _f32mul()
    plain, cse = (f[k]["nnz_a"] + f[k]["nnz_b"] for k in ("plain", "cse"))
    out.append({"unit": "f32mul-ftz", "circuits": CIRCUITS_F3["f32mul-ftz"], "what": "F32MulFtz_v3 alone, laid out from a "
                f"Circuit() without CSE as circuits builds it ({cse:,} with CSE); an attention-tail and RMSNorm piece",
                "unit_nnz_AB": plain, "unit_nnz_AB_cse": cse, "units": 1, "subcircuit_nnz_AB": plain,
                "generic_subcircuit_nnz_AB": f["generic_nnz_AB"]})
    return out


# --- exact affine fits -----------------------------------------------------------------------------------------------------

def affine(points: Sequence[tuple[int, Mapping[str, int]]], metrics: Sequence[str] = METRICS) -> dict[str, tuple[Fraction, Fraction]]:
    """Per metric ``(slope, intercept)`` through the first two points, checked equal on every other point (ValueError naming
    the metric and point otherwise). Needs at least three points."""
    if len(points) < 3:
        raise ValueError(f"an affine fit is checked on at least three points, not {len(points)}")
    (x0, c0), (x1, c1) = points[0], points[1]
    fit = {}
    for m in metrics:
        a = Fraction(c1[m] - c0[m], x1 - x0)
        b = c0[m] - a * x0
        for x, c in points[2:]:
            if a * x + b != c[m]:
                raise ValueError(f"{m} is not affine: {a} * {x} + {b} = {a * x + b}, the build gives {c[m]}")
        fit[m] = (a, b)
    return fit


def at(fit: Mapping[str, tuple[Fraction, Fraction]], x: Fraction | int) -> dict[str, Fraction]:
    return {m: a * x + b for m, (a, b) in fit.items()}


def _pow2(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0


def argmax_scan_law(builds: Mapping[int, Mapping[str, int]]) -> Callable[[int], dict[str, int]]:
    """token-select's counts at any V, from builds at consecutive V. Its step counter i is a constant, so a step's select of
    i costs one AND per bit of i that is not constant zero, and the running (best, best_i) are linear forms that grow by one
    term a step, never re-committed: with b = bit_length(V − 2), going from V − 1 to V adds ``alpha + b`` ANDs, the nonzeros'
    increment grows by ``beta + b`` (one more when V − 2 is a power of two), output copies move by +1 when V − 2 and −1
    when V − 1 is a power of two, inputs by a constant, and committed = inputs + ANDs + output copies. Checked at every
    consecutive V given (ValueError naming the first that breaks it); nonzeros are quadratic in V."""
    vs = sorted(builds)
    if len(vs) < 8 or vs != list(range(vs[0], vs[-1] + 1)) or vs[0] < 2:
        raise ValueError(f"the scan law is read off at least 8 consecutive V >= 2, not {vs[:3]}..")
    c = {v: builds[v] for v in vs}
    d = lambda v, m: c[v][m] - c[v - 1][m]  # noqa: E731
    v0 = vs[0] + 2
    alpha, beta = d(v0, "ands") - (v0 - 2).bit_length(), d(v0, "nnz_AB") - d(v0 - 1, "nnz_AB") - (v0 - 2).bit_length() - _pow2(v0 - 2)
    di = d(v0, "inputs")
    for v in vs[2:]:
        b = (v - 2).bit_length()
        want = {"ands": alpha + b, "d2nnz": beta + b + _pow2(v - 2), "output_copies": _pow2(v - 2) - _pow2(v - 1), "inputs": di,
                "lookups": 0}
        got = {"ands": d(v, "ands"), "d2nnz": d(v, "nnz_AB") - d(v - 1, "nnz_AB"), "output_copies": d(v, "output_copies"),
               "inputs": d(v, "inputs"), "lookups": d(v, "lookups")}
        if got != want or c[v]["committed"] != c[v]["inputs"] + c[v]["ands"] + c[v]["output_copies"]:
            raise ValueError(f"the argmax scan law breaks at V = {v}: {got} != {want}")
    top = vs[-1]

    def law(V: int) -> dict[str, int]:
        if V < vs[0]:
            raise ValueError(f"V = {V} is below the builds")
        if V <= top:
            return {m: c[V][m] for m in METRICS}
        r, dn = {m: c[top][m] for m in METRICS}, d(top, "nnz_AB")
        for v in range(top + 1, V + 1):
            b = (v - 2).bit_length()
            r["ands"] += alpha + b
            dn += beta + b + _pow2(v - 2)
            r["nnz_AB"] += dn
            r["output_copies"] += _pow2(v - 2) - _pow2(v - 1)
            r["inputs"] += di
        r["committed"] = r["inputs"] + r["ands"] + r["output_copies"]
        return {m: r[m] for m in METRICS}
    law.constants = {"alpha": alpha, "beta": beta, "inputs_per_V": di}  # type: ignore[attr-defined]
    return law


# --- the draw --------------------------------------------------------------------------------------------------------------

def sample_size(epsilon: Fraction = EPSILON, delta: Fraction = DELTA) -> int:
    from verity.core.protocols.pouw.circuit.plan import record_k

    return record_k(epsilon, delta)


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def draw_count(N: int, n: int, K: int, tail: float = float(DELTA)) -> tuple[float, int]:
    """A type's units in a uniform K-subset of N units, n of them the type's: ``(mean, q)``, q the least k with P(X > k) <= tail
    (hypergeometric, the upper tail summed term by term from the largest possible count down)."""
    if not 0 <= n <= N or not 0 <= K <= N:
        raise ValueError(f"N = {N}, n = {n}, K = {K}")
    lo, hi = max(0, K - (N - n)), min(n, K)
    base = _log_comb(N, K)
    acc = 0.0
    for k in range(hi, lo - 1, -1):
        p = math.exp(_log_comb(n, k) + _log_comb(N - n, K - k) - base)
        if acc + p > tail:
            return K * n / N, k
        acc += p
    return K * n / N, lo


# --- work weights ----------------------------------------------------------------------------------------------------------

def work(template: str, params: Mapping[str, Any], T: float | None = None) -> dict[str, float | None]:
    """``ops_per_input`` and ``bytes_per_input`` of the census work model at ``params`` (and T)."""
    wm = census.SUBCIRCUIT_TEMPLATES[template].work_model
    env = {k: v for k, v in params.items() if isinstance(v, int)}
    dt = params.get("datatype") or ("e4m3" if "e4m3" in str(params.get("semantics", "")) else "bf16")
    env["BITS"] = census.DATATYPES[dt].bits
    out: dict[str, float | None] = {}
    for name in ("ops_per_input", "bytes_per_input"):
        expr = getattr(wm, name, None)
        if expr is None:
            out[name] = None
        elif "T" in {n.id for n in ast.walk(ast.parse(expr, mode="eval")) if isinstance(n, ast.Name)}:
            # affine in T for every template that reads it (attention): exact at an integer T, the mean at the mean T
            lo = census.evaluate(expr, {**env, "T": 0})
            out[name] = lo + (census.evaluate(expr, {**env, "T": 1}) - lo) * (T if T is not None else float("nan"))
        else:
            out[name] = census.evaluate(expr, env)
    return out


# --- unit types ------------------------------------------------------------------------------------------------------------

@dataclass
class UnitType:
    workload: str
    family: str
    parameters: dict[str, Any]
    vus: int
    template: str
    params: dict[str, Any]
    per_vu: int
    T: dict[str, float] | None = None       # attention: min, max, mean of the stratum's key count

    @property
    def label(self) -> str:
        skip = {"DOT", "INV", "EPS", "KVH"}
        return self.family + "{" + ",".join(f"{k}={v}" for k, v in self.parameters.items() if k not in skip) + "}"


def _ceil16_sum(n: int) -> int:
    """sum of ceil(T / 16) over T = 1..n"""
    q, r = divmod(n, 16)
    return 8 * q * (q + 1) + r * (q + 1)


def _t_stats(entry: Mapping[str, Any], BN: int) -> dict[str, float]:
    """The stratum's key count T: min, max, mean, and the mean of ceil(T / 16) (the registered head's PV steps per D)."""
    NB, last = int(entry["parameters"]["NB"]), entry["parameters"]["last"]
    lo, hi = ((NB * BN, NB * BN) if last == "full" else ((NB - 1) * BN + 1, NB * BN - 1))
    runs = entry.get("variable_counts", {}).get("T", {}).get("runs")
    if runs:
        n = sum((r["max"] - r["min"] + 1) * r["vus_per_value"] for r in runs)
        s = sum((r["min"] + r["max"]) * (r["max"] - r["min"] + 1) / 2 * r["vus_per_value"] for r in runs)
        c = sum((_ceil16_sum(r["max"]) - _ceil16_sum(r["min"] - 1)) * r["vus_per_value"] for r in runs)
        return {"min": min(r["min"] for r in runs), "max": max(r["max"] for r in runs), "mean": s / n, "ceil16_mean": c / n,
                "basis": "variable_counts"}
    return {"min": lo, "max": hi, "mean": (lo + hi) / 2, "ceil16_mean": (_ceil16_sum(hi) - _ceil16_sum(lo - 1)) / (hi - lo + 1),
            "basis": "stratum range, uniform"}


def unit_types(workloads: Sequence[Mapping[str, Any]]) -> list[UnitType]:
    out = []
    for w in workloads:
        for e in w["template_mix"]:
            p = dict(e["subcircuit_parameters"])
            T = _t_stats(e, int(p["BN"])) if e["subcircuit_template"] == "attention-head" else None
            out.append(UnitType(w["id"], e["vu_family"], dict(e["parameters"]), int(e["vus"]), e["subcircuit_template"], p,
                                int(e["subcircuits_per_vu"]), T))
    return out


# --- the inventory ---------------------------------------------------------------------------------------------------------

def _plain(c: Mapping[str, Any]) -> dict[str, int]:
    return {m: int(c[m]) for m in METRICS}


def _num(v: Fraction) -> int | float:
    return int(v) if v.denominator == 1 else float(v)


def subcircuit_costs(types: Sequence[UnitType], exact_nnz: float, n_jobs: int, attention_t: Sequence[int] = ATTENTION_T,
                     log: Callable[[str], None] = print) -> dict[tuple, dict[str, Any]]:
    """Per distinct (template, census parameters): the per-subcircuit counts and how they were obtained; attention per
    (D, BN, semantics) carries an affine model in T instead."""
    distinct: dict[tuple, tuple[str, dict[str, Any]]] = {}
    for t in types:
        distinct.setdefault(_key(t.template, t.params), (t.template, t.params))
    jobs: list[tuple] = []
    for template, params in distinct.values():
        if template in AFFINE:
            var, pts = AFFINE[template]
            jobs += [_key(template, {**params, var: x}) for x in pts(params)]
        elif template == "token-select-greedy":
            jobs += [_key(template, {**params, "V": v}) for v in (*SCAN_POINTS, SCAN_CHECK)]
        elif template in EXACT_ONLY and PROBE[template]:
            jobs.append(_key(template, {**params, PROBE[template]: 128}))
        elif template == "attention-head":
            jobs += [_key(template, params, {"T": T}) for T in attention_t if T <= int(params["BN"])]
        elif template not in EXCLUDED:
            jobs.append(_key(template, params))
    log(f"stage 1: {len(set(jobs))} small builds")
    built = run_builds(jobs, n_jobs)

    out: dict[tuple, dict[str, Any]] = {}
    exact_jobs: dict[tuple, tuple] = {}
    block_jobs: list[tuple] = []
    for key, (template, params) in distinct.items():
        r: dict[str, Any] = {"template": template, "params": params}
        out[key] = r
        if template in EXCLUDED:
            r.update(method="excluded", reason=EXCLUDED[template])
            continue
        if template in AFFINE:
            var, pts = AFFINE[template]
            res = [(x, built[_key(template, {**params, var: x})]) for x in pts(params)]
            err = next((c["error"] for _, c in res if "error" in c), None)
            if err:
                r.update(method="missing", reason=err)
                continue
            try:
                fit = affine(res)
            except ValueError as e:
                r.update(method="missing", reason=f"not affine in {var} on {[x for x, _ in res]}: {e}")
                continue
            x = int(params[var])
            r.update(method="affine", variable=var, points=[x for x, _ in res], counts={m: _num(v) for m, v in at(fit, x).items()},
                     fit={m: [str(a), str(b)] for m, (a, b) in fit.items()}, tables=res[-1][1]["tables"],
                     definition=res[-1][1]["definition"])
            if x in r["points"]:
                r["method"] = "exact"
            elif r["counts"]["nnz_AB"] <= exact_nnz:
                exact_jobs[key] = _key(template, params)
        elif template == "token-select-greedy":
            res = {v: built[_key(template, {**params, "V": v})] for v in (*SCAN_POINTS, SCAN_CHECK)}
            err = next((c["error"] for c in res.values() if "error" in c), None)
            if err:
                r.update(method="missing", reason=err)
                continue
            try:
                law = argmax_scan_law({v: res[v] for v in SCAN_POINTS})
            except ValueError as e:
                r.update(method="missing", reason=str(e))
                continue
            check = law(SCAN_CHECK)
            agree = check == _plain(res[SCAN_CHECK])
            V = int(params["V"])
            r.update(method="exact" if V in res else "scan law", variable="V", points=[SCAN_POINTS[0], SCAN_POINTS[-1]],
                     counts=law(V), law=law.constants, tables=res[SCAN_POINTS[-1]]["tables"],
                     definition=res[SCAN_POINTS[-1]]["definition"],
                     exact_check={"V": SCAN_CHECK, "result": "agrees" if agree else {m: [check[m], res[SCAN_CHECK][m]] for m in METRICS}})
            if not agree:
                r.update(method="missing", reason=f"the scan law disagrees with the exact build at V = {SCAN_CHECK}")
        elif template in EXACT_ONLY:
            probe = PROBE[template]
            if probe:
                small = built[_key(template, {**params, probe: 128})]
                if "error" in small:
                    r.update(method="missing", reason=small["error"])
                    continue
                est = small["nnz_AB"] * int(params[probe]) / 128
                r["estimated_nnz_AB"] = est
                if int(params[probe]) == 128:
                    r.update(method="exact", counts=_plain(small), tables=small["tables"], definition=small["definition"])
                elif est <= exact_nnz:
                    exact_jobs[key] = _key(template, params)
                    r["method"] = "pending"
                else:
                    r.update(method="not built", reason=f"a structured reduction, built only at the census size; about {est:.2e} "
                                                         f"nonzeros is over --exact-nnz {exact_nnz:.2e}")
            else:
                c = built[_key(template, params)]
                if "error" in c:
                    r.update(method="missing", reason=c["error"])
                else:
                    r.update(method="exact", counts=_plain(c), tables=c["tables"], definition=c["definition"])
        elif template == "attention-head":
            res = [(T, built[_key(template, params, {"T": T})]) for T in attention_t if T <= int(params["BN"])]
            err = next((c["error"] for _, c in res if "error" in c), None)
            if err:
                r.update(method="missing", reason=err)
                continue
            try:
                fit = affine(res)
            except ValueError as e:
                r.update(method="missing", reason=f"not affine in T on {[T for T, _ in res]}: {e}")
                continue
            r.update(method="model", variable="T", points=[T for T, _ in res], fit={m: [str(a), str(b)] for m, (a, b) in fit.items()},
                     built={T: _plain(c) for T, c in res}, tables=res[-1][1]["tables"], definition=res[-1][1]["definition"])
            r["_fit"] = fit
            BN = int(params["BN"])
            for j in BLOCK_CHECKS:
                T = j * BN + 16
                if at(fit, T)["nnz_AB"] <= exact_nnz:
                    block_jobs.append((key, j, T, _key(template, params, {"T": T})))
        else:
            c = built[_key(template, params)]
            if "error" in c:
                r.update(method="missing", reason=c["error"])
            else:
                r.update(method="exact", counts=_plain(c), tables=c["tables"], definition=c["definition"])

    log(f"stage 2: {len(exact_jobs)} builds at the census parameters and {len(block_jobs)} attention heads past the first key "
        f"block (nonzeros <= {exact_nnz:.2e})")
    big = run_builds([*exact_jobs.values(), *(job for *_, job in block_jobs)], n_jobs)
    checks: dict[tuple, dict[int, tuple[int, dict[str, Any]]]] = {}
    for key, j, T, job in block_jobs:
        checks.setdefault(key, {})[j] = (T, big[job])
    for key, by_j in checks.items():
        r = out[key]
        if any("error" in c for _, c in by_j.values()):
            r.setdefault("notes", []).append(f"a build past the first key block failed: {[c.get('error') for _, c in by_j.values()]}")
            continue
        r["block_checks"] = {T: {"built": _plain(c)} for T, c in by_j.values()}
        block_law(r)
    log("stage 2: done")
    for key, job in exact_jobs.items():
        r, c = out[key], big[job]
        if "error" in c:
            r.setdefault("notes", []).append(f"the build at the census parameters failed: {c['error']}")
            if r.get("method") == "pending":
                r.update(method="missing", reason=c["error"])
            continue
        if r["method"] == "affine":
            agree = all(r["counts"][m] == c[m] for m in METRICS)
            r["exact_check"] = "agrees" if agree else {m: [r["counts"][m], c[m]] for m in METRICS}
            if not agree:
                r.setdefault("notes", []).append("the affine extrapolation disagrees with the exact build; the exact build is used")
        r.update(method="exact", counts=_plain(c), tables=c["tables"], definition=c["definition"])
    return out


def block_law(r: dict[str, Any]) -> None:
    """An attention head's cost past its first key block, from ``r["block_checks"]`` (builds at T = j BN + 16, j in
    :data:`BLOCK_CHECKS`): each metric's excess over the first block's fit is ``lin j + curv j (j − 1) / 2``. A metric with
    ``curv`` 0 is affine in the block count, checked on j = 0, 1, 2; when every metric is, the head is ``affine``. Otherwise
    it stays a ``model``: the curvature is fitted through three points with none left to check it."""
    BN, fit = int(r["params"]["BN"]), r["_fit"]
    by_j = {(int(T) - 16) // BN: c["built"] for T, c in r["block_checks"].items()}
    if sorted(by_j) != list(BLOCK_CHECKS):
        return
    resid = {j: {m: c[m] - at(fit, j * BN + 16)[m] for m in METRICS} for j, c in by_j.items()}
    for T, c in r["block_checks"].items():
        c["over_first_block_fit"] = {m: _num(v) for m, v in resid[(int(T) - 16) // BN].items()}
    lin = resid[1]
    curv = {m: resid[2][m] - 2 * lin[m] for m in METRICS}
    r["_block"] = (lin, curv)
    r.update(per_extra_block={m: _num(v) for m, v in lin.items()})
    if any(curv.values()):
        r["block_curvature"] = {m: _num(v) for m, v in curv.items() if v}
        r.setdefault("notes", []).append(f"not affine in the block count for {sorted(m for m, v in curv.items() if v)}: the "
                                         "excess over the first block is fitted with a curvature through j = 0, 1, 2, unchecked")
    else:
        r.update(method="affine", variable="T, NB")


def reuse(path: str) -> dict[tuple, dict[str, Any]]:
    """The subcircuit results of an earlier ``unit-costs.json`` (its builds), with the fits and block laws re-derived."""
    subs = {}
    for s in json.loads(open(path).read())["subcircuits"]:
        s = {k: v for k, v in s.items() if k not in ("per_extra_block", "block_curvature", "notes")}
        if s["template"] == "attention-head" and "fit" in s:
            s["method"] = "model"
            s["_fit"] = {m: (Fraction(a), Fraction(b)) for m, (a, b) in s["fit"].items()}
            if s.get("block_checks"):
                block_law(s)
        subs[_key(s["template"], s["params"])] = s
    return subs


def _generic(c: Mapping[str, Any]) -> dict[str, Any]:
    return {**c, "nnz_C": c["ands"], "nnz": c["nnz_AB"] + c["ands"]}


def _registered_sub(reg: Mapping[str, Any], t: UnitType, T: float | None = None, ceil16: float | None = None) -> dict[str, float]:
    """A subcircuit's counts from its registered lowering: units x the unit's layout (attention: the steps at T, or at the
    stratum's mean T and mean ceil(T / 16)). C = I over every row, so nnz = nnz_a + nnz_b + rows."""
    u = reg["unit"]
    if t.template == "attention-head":
        D = int(t.params["D"])
        n: float = attention_steps(D, int(T)) if ceil16 is None else D // 16 * T + D * ceil16
    else:
        n = reg["units"]
    nnz_AB = n * (u["nnz_a"] + u["nnz_b"])
    return {"units": n, "committed": n * u["rows"], "ands": n * u["ands"], "lookups": n * u["reads"], "nnz_AB": nnz_AB,
            "nnz_C": n * u["rows"], "nnz": nnz_AB + n * u["rows"]}


def inventory(workloads: Sequence[Mapping[str, Any]], exact_nnz: float = 0, n_jobs: int = 1,
              attention_t: Sequence[int] = ATTENTION_T, log: Callable[[str], None] = print,
              subs: dict[tuple, dict[str, Any]] | None = None, regs: dict[tuple, dict[str, Any]] | None = None,
              step: Mapping[str, int] | None = None) -> dict[str, Any]:
    """``regs`` (default :func:`registered_costs`) gives each binding's registered lowering; a unit's primary counts come from
    it where one exists (``source`` registered) and from the generic whole lowering otherwise, whose nonzeros every row
    also carries as ``nnz_generic``. ``step`` (default :func:`argmax_step`) prices token-select's scan per step."""
    K = sample_size()
    types = unit_types(workloads)
    if subs is None:
        subs = subcircuit_costs(types, exact_nnz, n_jobs, attention_t, log)
    if regs is None:
        regs = registered_costs(types, n_jobs, log)
    if step is None and any(t.template == "token-select-greedy" for t in types):
        step = argmax_step()
    rows = []
    for w in workloads:
        mine = [t for t in types if t.workload == w["id"]]
        N_vu = sum(t.vus for t in mine)
        N_sub = sum(t.vus * t.per_vu for t in mine)
        for t in mine:
            s = subs[_key(t.template, t.params)]
            per_sub: dict[str, Any] | None = None
            if "_fit" in s:
                e = int(t.parameters["NB"]) - 1
                lin, curv = s.get("_block") or ({m: 0 for m in METRICS}, {m: 0 for m in METRICS})

                def head(T: Fraction | int) -> dict[str, Fraction]:
                    return {m: v + e * lin[m] + Fraction(e * (e - 1), 2) * curv[m] for m, v in at(s["_fit"], T).items()}
                per_sub = {m: float(v) for m, v in head(Fraction(t.T["mean"])).items()}
                per_sub_range = {b: float(head(int(t.T[b]))["nnz_AB"] + head(int(t.T[b]))["ands"]) for b in ("min", "max")}
            elif "counts" in s:
                per_sub = dict(s["counts"])
                per_sub_range = None
            else:
                per_sub_range = None
            gen = _generic(per_sub) if per_sub is not None else None
            reg = regs.get(_key(t.template, t.params), {"none": "excluded" if t.template in EXCLUDED else "not lowered"})
            row: dict[str, Any] = {"workload": w["id"], "unit_type": t.label, "vu_family": t.family, "parameters": t.parameters,
                                   "template": t.template, "subcircuit_parameters": t.params, "subcircuits_per_vu": t.per_vu,
                                   "method": s["method"], "reason": s.get("reason"), "T": t.T}
            if "unit" in reg:
                row.update(source="registered", lowering=reg["lowering"], registered_unit=reg["unit"])
                if t.T:
                    prim = _registered_sub(reg, t, t.T["mean"], t.T["ceil16_mean"])
                    prim_range = {b: _registered_sub(reg, t, int(t.T[b]))["nnz"] for b in ("min", "max")}
                else:
                    prim, prim_range = _registered_sub(reg, t), None
            else:
                row.update(source="generic", registered_none=reg["none"])
                prim = dict(gen) if gen is not None else None
                prim_range = per_sub_range
            if t.template == "token-select-greedy" and step:
                V = int(t.params["V"])
                row["nnz_step_model"] = {"step": dict(step), "steps": V - 1,
                                         "nnz": (V - 1) * (step["nnz_a"] + step["nnz_b"] + step["rows"]),
                                         "committed": (V - 1) * step["rows"]}
            wk = work(t.template, t.params, t.T["mean"] if t.T else None)
            for gran, n, N, mult in (("vu", t.vus, N_vu, t.per_vu), ("subcircuit", t.vus * t.per_vu, N_sub, 1)):
                mean, q = draw_count(N, n, K)
                g: dict[str, Any] = {"units": n, "population": N, "share": n / N, "sampled_mean": mean, "sampled_q": q,
                                     "ops": None if wk["ops_per_input"] is None else wk["ops_per_input"] * mult,
                                     "bytes": None if wk["bytes_per_input"] is None else wk["bytes_per_input"] * mult}
                if prim is not None:
                    c = {m: v * mult for m, v in prim.items()}
                    c["nnz_generic"] = None if gen is None else gen["nnz"] * mult
                    if gen is not None:
                        c["generic"] = {m: v * mult for m, v in gen.items()}
                    g.update(per_unit=c, audit_nnz_mean=mean * c["nnz"], audit_nnz_q=q * c["nnz"])
                    if prim_range:
                        g["nnz_range"] = {b: v * mult for b, v in prim_range.items()}
                row[gran] = g
            rows.append(row)
    for s in subs.values():
        s.pop("_fit", None)
        s.pop("_block", None)
    return {"format": "verity/f6-unit-costs/v2", "epsilon": str(EPSILON), "delta": "2^-40", "K": K,
            "draw": "uniform K-subset of all units per workload and granularity (a hidden partition cannot stratify by type)",
            "quantile": "least k with P(X > k) <= 2^-40, hypergeometric", "exact_nnz": exact_nnz, "attention_t": list(attention_t),
            "nnz": "registered lowering's layout where one exists (nnz_a + nnz_b + rows, C = I over rows), else the generic "
                   "whole lowering (nnz_AB + ANDs); nnz_generic is the generic count",
            "subcircuits": [dict(v, params=v["params"]) for v in subs.values()],
            "registered": [{"template": k[0], "params": dict(k[1]), **v} for k, v in regs.items()],
            "argmax_step": step, "reconciliation": reconcile(regs, subs) if regs else [], "rows": rows}


# --- rendering -------------------------------------------------------------------------------------------------------------

def _si(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    v = float(v)
    for unit, d in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(v) >= d:
            return f"{v / d:.3g}{unit}"
    return f"{v:.3g}"


METHOD_MARK = {"exact": "E", "affine": "A", "scan law": "S", "model": "M", "missing": "–", "excluded": "x", "not built": "nb"}
SOURCE_MARK = {"registered": "R", "generic": "G", "tree": "T"}


def render(inv: Mapping[str, Any]) -> str:
    lines = [f"# F6 unit costs: K = {inv['K']} draws (ε = {inv['epsilon']}, δ = {inv['delta']}), uniform over all units", "",
             "Method: E exact build at the census parameters, A exact affine extrapolation (checked on 4 builds; attention: "
             "affine in T within a key block and in the block count, checked past the first block), S token-select's scan "
             "law (read off every V in 2..139, checked by an exact build at V = 1030), M attention model (affine in T within "
             "a key block; past the first block, the metrics not affine in the block count are fitted with an unchecked "
             "curvature), – missing, x excluded, nb not built.", "",
             "src: R the template's registered C-Flock lowering (units x its flock-ir-unit/v2 layout; committed = rows, "
             "nnz = nnz_a + nnz_b + rows, C = I over rows), G the generic whole lowering (no registered one; committed = "
             "inputs + ANDs + output copies, nnz = nnz_AB + ANDs). nnz_generic is the generic count for every row. "
             "q = the 1 − 2⁻⁴⁰ hypergeometric quantile of the sampled count. Ops and bytes are the census work model.", ""]
    if inv.get("reconciliation"):
        lines += ["## Against circuits' F3 unit nonzeros", "",
                  "circuits' figures are nnz_a + nnz_b of one registered unit (`benchmarks/private_circuit/nonzero_cost.py`, "
                  "cursor/f3-nonzero-cost-f253 @76e405893).", "",
                  "| unit | circuits | registered unit nnz_AB | units per subcircuit | subcircuit nnz_AB (registered) | "
                  "subcircuit nnz_AB (generic) | generic / registered | unit |", "|" + "---|" * 8]
        for r in inv["reconciliation"]:
            g, s = r.get("generic_subcircuit_nnz_AB"), r.get("subcircuit_nnz_AB")
            lines.append(f"| {r['unit']} | {r['circuits']:,} | {r.get('unit_nnz_AB', 0):,} | {r.get('units', '–')} | "
                         f"{_si(s)} | {_si(g)} | {f'{g / s:.3f}' if g and s else '–'} | {r['what']} |")
        lines.append("")
    trees = {(r["template"], r["subcircuit_parameters"]["V"]): r.get("tree") or r.get("select_tree") for r in inv["rows"]
             if r.get("tree") or r.get("select_tree")}
    if trees:
        lines += ["token-select (src T) is priced at `TokenSelect_v2`'s argmax tree: every stage is separable, so it costs its "
                  "members' layouts (`ArgmaxLeaf_v1`, `ArgmaxMerge_v1`) times their multiplicities. The generic scan count "
                  "(quadratic in V) is superseded; nnz_generic still shows it. gumbel keeps its registered lane; its select's "
                  "F32 tree is listed for comparison.", "",
                  "| template | V | tree | members | rows | ANDs | nnz | nnz per logit |", "|---|---|---|---|---|---|---|---|"]
        for (tm, V), t in sorted(trees.items()):
            lines.append(f"| {tm} | {V:,} | {t['stage']} | {t['units']:,} | {_si(t['rows'])} | {_si(t['ands'])} | {_si(t['nnz'])} | "
                         f"{t['nnz_per_logit']:.1f} |")
        lines.append("")
    elif inv.get("argmax_step"):
        lines += [f"token-select has no registered lowering: its rows use the generic count (quadratic in V; the running "
                  f"best is never re-committed). A cut lowering's middle ArgmaxStep unit is {inv['argmax_step']}, so "
                  f"about (V − 1) x {inv['argmax_step']['nnz_a'] + inv['argmax_step']['nnz_b'] + inv['argmax_step']['rows']:,} "
                  "nonzeros (``nnz_step_model``).", ""]
    for gran in ("vu", "subcircuit"):
        lines += [f"## Unit = one {'VU' if gran == 'vu' else 'subcircuit'}", ""]
        for wid in dict.fromkeys(r["workload"] for r in inv["rows"]):
            rows = sorted((r for r in inv["rows"] if r["workload"] == wid), key=lambda r: -(r[gran].get("audit_nnz_mean") or 0))
            g0 = rows[0][gran]
            tot = sum(r[gran].get("audit_nnz_mean") or 0 for r in rows)
            lines += [f"### {wid}", "", f"N = {g0['population']:,} units; sampled nonzeros per audit (mean, built types) "
                      f"{_si(tot)}", "",
                      "| unit type | m | src | units | share | ANDs | committed | lookups | nnz_AB | nnz | nnz_generic | "
                      "sampled mean | q | audit nnz | ops | bytes |", "|" + "---|" * 16]
            for r in rows:
                g = r[gran]
                c = g.get("per_unit") or {}
                lines.append("| " + " | ".join([
                    r["unit_type"], METHOD_MARK.get(r["method"], r["method"]), SOURCE_MARK.get(r.get("source"), "G"),
                    f"{g['units']:,}", f"{g['share']:.2%}",
                    _si(c.get("ands")), _si(c.get("committed")), _si(c.get("lookups")), _si(c.get("nnz_AB")), _si(c.get("nnz")),
                    _si(c.get("nnz_generic")), f"{g['sampled_mean']:.1f}", str(g["sampled_q"]), _si(g.get("audit_nnz_mean")),
                    _si(g["ops"]), _si(g["bytes"])]) + " |")
            lines.append("")
    lines += ["## Subcircuits", "", "| template | parameters | method | definition | detail |", "|---|---|---|---|---|"]
    for s in inv["subcircuits"]:
        detail = s.get("reason") or ""
        if s.get("points"):
            detail = f"{s.get('variable')} at {s['points']}" + (f"; exact check {s['exact_check']}" if s.get("exact_check") else "")
        if s.get("per_extra_block"):
            detail += f"; built at T = {list(s['block_checks'])}, per extra block {s['per_extra_block']}"
            if s.get("block_curvature"):
                detail += f", curvature {s['block_curvature']} (unchecked)"
        elif s.get("block_checks"):
            detail += f"; past the first block: {s['block_checks']}"
        if s.get("tables"):
            detail += "; tables " + ", ".join(f"{k} x{v['reads']}" for k, v in s["tables"].items())
        lines.append(f"| {s['template']} | {json.dumps(s['params'], sort_keys=True)} | {s['method']} | {s.get('definition', '')} | {detail} |")
    lines += ["", "## Registered lowerings", "", "| template | parameters | lowering | units | unit rows | unit ANDs | unit nnz_a | "
              "unit nnz_b | none |", "|" + "---|" * 9]
    for r in inv.get("registered", []):
        u = r.get("unit") or {}
        lines.append(f"| {r['template']} | {json.dumps(r['params'], sort_keys=True)} | {r.get('lowering', '–')} | {r.get('units', '–')} | "
                     f"{u.get('rows', '–')} | {u.get('ands', '–')} | {u.get('nnz_a', '–')} | {u.get('nnz_b', '–')} | "
                     f"{r.get('none', '')} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory for unit-costs.json and unit-costs.md")
    ap.add_argument("--exact-nnz", type=float, default=0, help="also build at the census parameters when the nonzeros are at most this")
    ap.add_argument("--jobs", type=int, default=1, help="parallel builds (spawned processes)")
    ap.add_argument("--workload", action="append", default=[], help="only workloads whose id contains this (repeatable)")
    ap.add_argument("--attention-t", default=",".join(map(str, ATTENTION_T)), help="attention's key counts within the first block")
    ap.add_argument("--reuse", help="an earlier run's unit-costs.json: take its subcircuit results instead of building")
    ap.add_argument("--retree", help="an earlier run's unit-costs.json: only re-price its samplers at the argmax tree (retree)")
    a = ap.parse_args(argv)
    from pathlib import Path

    if a.retree:
        inv = json.loads(Path(a.retree).read_text())
        inv["retreed"] = a.retree
    else:
        workloads = [w for w in census.raw("workloads") if not a.workload or any(s in w["id"] for s in a.workload)]
        inv = inventory(workloads, a.exact_nnz, a.jobs, tuple(int(x) for x in a.attention_t.split(",")),
                        log=lambda m: print(m, flush=True), subs=reuse(a.reuse) if a.reuse else None)
        if a.reuse:
            inv["reused"] = a.reuse
    retree(inv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "unit-costs.json").write_text(json.dumps(inv, indent=1, default=str) + "\n")
    (out / "unit-costs.md").write_text(render(inv))
    print(f"wrote {out / 'unit-costs.json'} and {out / 'unit-costs.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
