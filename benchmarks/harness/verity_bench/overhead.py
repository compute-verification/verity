"""`verity bench overhead`'s rules: the cards, the operating points' rule, the prover overhead γ and its interval, and what
γ means per monitoring period. Standard library only, and loadable alone by its path, but for γ's pairing, which is the
harness kit's (`kit.paired`): the node's sweep (`overhead_node`) and the store side (`overhead_records`) both read it.

**The operating points** belong to a card, a model and a precision. Over one layer of the model's linears (`MODELS`), the
fastest stock kernel at each m is the fastest of every library and scaling the card runs (`Gpu.libraries`, the precision's
scalings in `Gpu.precisions`, by wall time), per linear, and t(m) is the layer's sum. Throughput is the layer's FLOP over t(m).

- **C**, compute-bound: the smallest m at which throughput reaches `C_SHARE` of the best throughput sampled, and every m
  sampled above it does too.
- **L**, latency, memory-bound: the largest m whose t(m), and every t(m') sampled below it, stays within `L_SLACK` of t(1).

The sweep (`next_ms`) takes every power of two from 1 to `M_MAX`, then bisects each point's bracket on multiples of `GRID`
until the bracket is one step wide; C and L are always multiples of `GRID`.

**γ** (`layer_gamma`) is a scheme's (`SCHEMES`) cost over the stock arm's for the summed layer, less one: the median of the
reps' paired ratios, with a 95% interval from the ratios' order statistics (`kit.paired`; the 6th and 15th of 20). The
numerator is the scheme's busy SM-time (SMs Active over a traced pass, `nsys`) and the host CPU core-seconds it spends above
stock's at the card's weight (`host_cost`, from the timed pass); the denominator is the stock arm's busy SM-time. The stock
arm is each linear's fastest kernel of the scheme's scaling by wall time, with the least-SM-time kernel's γ beside it; the
`stock` scheme's γ is the A/A control, one kernel against its own twin. A figure of record has at least `MIN_REPS` reps,
an interval within `HALF_WIDTH` of the median ratio (`of_record_why`) and passes the SM-time cross-check
(`cross_check_why`); the scheme gate is γ ≤ `GAMMA_CAP` at each point. `periods` states it per monitoring period.

**What a γ covers** (`covers`) is its model's entry: one layer's linears, and not what `LEFT_OUT` names. **y's commitment**: a
scheme's entry says whether its arm's timed call commits y, the call's output (`Scheme.commits_y`); a timed run records it,
and a figure from a run whose arm committed no y carries `WITHOUT_Y` (`y_note`).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

POINTS_SCHEMA = "verity-bench/overhead-points/v1"
RUN_SCHEMA = "verity-bench/overhead/v1"
#: the points' rule: C's share of the best throughput, L's slack over t(1)
C_SHARE, L_SLACK = 0.9, 0.1
#: the sweep: powers of two up to M_MAX, then bisection on multiples of GRID (the kernels' tile granularity)
M_MAX, GRID = 32768, 16
#: what a sweep records of the rule, and points of record must have been swept under
RULE = {"c_share": C_SHARE, "l_slack": L_SLACK, "grid": GRID, "m_max": M_MAX}
#: a figure of record: at least MIN_REPS paired reps, its 95% interval within HALF_WIDTH of the median ratio, and its
#: busy SM-time at most CROSS_CHECK over its kernels' durations x the SMs their blocks occupy
MIN_REPS, HALF_WIDTH, CROSS_CHECK = 20, 0.05, 0.05
#: the scheme gate: γ at most this at each point
GAMMA_CAP = 10.0
POINTS = ("C", "L")


@dataclass(frozen=True)
class Gpu:
    """A GPU the door measures on (its `--card`): what its device reports (`name` within the device's name, `capability`, `sms`), the
    harness's build target, the L2 the operand sets must pass twice, host CPU core-seconds' weight in SM-seconds (the
    node's CPU capacity over its GPUs'), and the scalings and libraries its
    fastest stock kernel is chosen among; `bridge`: its CUTLASS and DeepGEMM are vLLM's, through torch (`bench.py --bridge`),
    since the harness's own CUTLASS is sm_120's; `datasheet`: its dense FP8 rate's id in `benchmarks/census/hardware.json`
    (the H200's is the H100's: the same compute with more HBM)."""

    gpu: str
    name: str
    capability: str
    sms: int
    arch: str
    l2_bytes: int
    cpu_sm_s_per_core_s: float
    precisions: Mapping[str, tuple[str, ...]]
    libraries: tuple[str, ...]
    bridge: bool
    datasheet: str


#: each precision's scalings, as the PoUW harness names its baseline families: per-tensor, rowwise and 128-block FP8
FP8 = ("fp8-e4m3", "fp8-e4m3-rowwise", "fp8-blk128")
CARDS = {card.gpu: card for card in (
    Gpu("h200", "H200", "9.0", 132, "sm_90a", 50 << 20, 0.48, {"fp8": FP8},
         ("cublaslt", "cutlass", "deepgemm"), True, "h100-sxm5-80gb/e4m3"),
    Gpu("rtx-pro-6000", "RTX PRO 6000", "12.0", 188, "sm_120a", 128 << 20, 1.64,
         {"fp8": FP8, "nvfp4": ("nvfp4",)}, ("cublaslt", "cutlass"), False, "rtx-pro-6000-server/e4m3"),
)}

#: one layer's linears, (n, k) each: qkv's n is the query width and two KV widths, gate_up's twice the intermediate size
MODELS = {
    "llama3.1-70b": {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (57344, 8192), "down": (8192, 28672)},
}
#: the kernel points' model
KERNEL_MODEL = "llama3.1-70b"
#: what a model's entry leaves out of the layer its γ is over, though serving the model computes it
LEFT_OUT = {"llama3.1-70b": ("the LM head",)}
#: the note of a figure from a run whose arm committed no y (`y_note`): where the served call commits y, the γ leaves y's tree
#: out, and is understated by it
WITHOUT_Y = "without y's commitment (y tree not timed)"


@dataclass(frozen=True)
class Scheme:
    """A scheme the door times against stock. `arm` is its PoUW harness arm (`module:Class` in `arm_path`, as `bench.py
    --arm` and `--arm-path` take them), timed beside stock kernels of `family`, the stock scaling its useful output's
    quantization matches. `epsilon` (`module:function` in `arm_path`) is its soundness error at a shape on a card,
    `(m, n, k, card) -> {"epsilon", "delta", "basis"}`, recorded beside its γ. `commits_y`: whether the arm's timed call
    commits y, the call's output (y's tree), false unless the entry states it; a timed run records it beside its arm
    (`overhead.json`'s `arm.commits_y`), and its figures carry `WITHOUT_Y` while it is false (`y_note`). `stock` has no arm:
    its γ is the A/A control every run carries (`bench.py --twins`), the fastest stock kernel against its own twin."""

    id: str
    family: str
    arm: str | None = None
    arm_path: str | None = None
    epsilon: str | None = None
    commits_y: bool = False


#: Pearl+'s useful output is rowwise-scaled E4M3, so its stock, and stock's own A/A control, are that family's
SCHEMES = {s.id: s for s in (
    Scheme("stock", "fp8-e4m3-rowwise"),
    Scheme("pearl-plus", "fp8-e4m3-rowwise", "overhead_arm:PearlPlus", "benchmarks/pouw/pearl_c_sm90",
           "overhead_arm:epsilon", commits_y=True),
)}


def scheme(name: str) -> Scheme:
    try:
        return SCHEMES[name]
    except KeyError:
        raise ValueError(f"no scheme {name!r}: the door times {', '.join(SCHEMES)} (verity_bench.overhead.SCHEMES)") from None


def y_note(arm: Mapping | None) -> str | None:
    """The note a run's figures carry, from the `arm` record of its `overhead.json`: `WITHOUT_Y` unless the record says its
    arm committed y (`commits_y`, the scheme's entry as the run recorded it; a record without the field timed no y tree).
    None for a run with no arm: stock's A/A control commits nothing."""

    return None if arm is None or arm.get("commits_y") else WITHOUT_Y


def covers(model: str = KERNEL_MODEL) -> str:
    """What a γ of `model` is over, in one sentence: its entry's linears, and what the entry leaves out (`LEFT_OUT`)."""

    linears, out = MODELS[model], LEFT_OUT.get(model, ())
    return (f"Each γ is over the {len(linears)} linears of one {model} layer ({', '.join(linears)})"
            + (f" and leaves {' and '.join(out)} out" if out else "") + ".")


def card(gpu: str) -> Gpu:
    try:
        return CARDS[gpu]
    except KeyError:
        raise ValueError(f"no card {gpu!r}: the door measures on {', '.join(CARDS)} (verity_bench.overhead.CARDS)") from None


#: the libraries a card's bridge serves (`bench.py --bridge`), by the bridge's name for each
BRIDGED = {"cutlass": "cutlass-vllm", "deepgemm": "deepgemm"}


def libraries_why(c: Gpu, result: Mapping) -> str | None:
    """Why a `bench.py` result didn't time every library the card's stock kernel is chosen among (a bridged library that
    didn't load), or None."""

    if not c.bridge:
        return None
    if "bridge" not in result:
        return f"bench.py ran without --bridge: the {c.name}'s {', '.join(c.libraries)} are the stock kernel's libraries"
    errors = result["bridge"].get("errors") or {}
    off = {lib: errors.get(BRIDGED[lib]) or errors.get("torch") for lib in c.libraries if lib in BRIDGED}
    if off := {lib: why for lib, why in off.items() if why}:
        return "; ".join(f"the {c.name}'s {lib} didn't load ({why})" for lib, why in off.items())
    return None


def label(m: int, n: int, k: int) -> str:
    """A shape's label as the PoUW harness keys it (`shapes.key`), which `bench.py --shapes` takes for any m, n and k."""

    return f"m{m}-n{n}-k{k}"


def layer_flop(model: str, m: int) -> int:
    return sum(2 * m * n * k for n, k in MODELS[model].values())


# ---- the points --------------------------------------------------------------------------------------------------------------


def powers() -> list[int]:
    return [1 << i for i in range(M_MAX.bit_length())]


def _grid_between(lo: int, hi: int) -> list[int]:
    """The multiples of GRID strictly between lo and hi."""

    first = (lo // GRID + 1) * GRID
    return list(range(first, hi, GRID))


def _mid(lo: int, hi: int) -> int | None:
    """The multiple of GRID nearest the bracket's middle, strictly inside it; None when the bracket holds none."""

    inside = _grid_between(lo, hi)
    if not inside:
        return None
    return min(inside, key=lambda m: (abs(2 * m - lo - hi), m))


@dataclass(frozen=True)
class Points:
    """The rule's verdict on the m's sampled so far: C and L (None until sampled), the best throughput, and the bracket each
    lies in (its sampled neighbour on the side it could still move to, None at the sweep's end)."""

    c: int | None
    l: int | None
    best_flop_per_ms: float
    c_bracket: tuple[int | None, int | None]
    l_bracket: tuple[int | None, int | None]


def rule(times: Mapping[int, float], model: str = KERNEL_MODEL) -> Points:
    """C and L from the layer's time at each sampled m (ms, the fastest stock kernel of each linear, summed). C and L are
    taken only on multiples of GRID: a sampled m off the grid (1, 2, 4, 8) counts for the conditions, never as a point."""

    if 1 not in times:
        raise ValueError("the rule needs t(1): sample m = 1")
    ms = sorted(times)
    flops = {m: layer_flop(model, m) / times[m] for m in ms}
    best = max(flops.values())
    c = None
    for m in reversed(ms):
        if flops[m] < C_SHARE * best:
            break
        if m % GRID == 0:
            c = m
    l = None
    for m in ms:
        if times[m] > (1 + L_SLACK) * times[1]:
            break
        if m % GRID == 0:
            l = m
    below = max((m for m in ms if c is not None and m < c), default=None)
    above = min((m for m in ms if l is not None and m > l), default=None)
    if l is None:
        above = min((m for m in ms if m > 1 and times[m] > (1 + L_SLACK) * times[1]), default=None)
    return Points(c, l, best, (below, c), (l, above))


def next_ms(times: Mapping[int, float], model: str = KERNEL_MODEL) -> list[int]:
    """The m's the sweep samples next: the powers of two not yet sampled, else one m inside each point's bracket wider than
    one GRID step; [] once both brackets are one step wide (the points are final)."""

    missing = [m for m in powers() if m not in times]
    if missing:
        return missing
    p = rule(times, model)
    out = []
    lo, hi = p.c_bracket
    if hi is not None and lo is not None and (mid := _mid(lo, hi)) is not None:
        out.append(mid)
    lo, hi = p.l_bracket
    if lo is None and hi is not None:
        lo = 0
    if lo is not None and hi is not None and (mid := _mid(lo, hi)) is not None:
        out.append(mid)
    return sorted(set(out) - set(times))


def final(times: Mapping[int, float], model: str = KERNEL_MODEL) -> Points:
    """The points of a finished sweep; refuses one whose brackets are still open, or whose rule finds no point."""

    if more := next_ms(times, model):
        raise ValueError(f"the sweep isn't finished: it still samples m = {', '.join(map(str, more))}")
    p = rule(times, model)
    if p.c is None or p.l is None:
        raise ValueError(f"the rule finds no {'C' if p.c is None else 'L'} on multiples of {GRID} in the sweep")
    return p


def fastest(row: Mapping[str, float]) -> tuple[str, float]:
    """The fastest (candidate, ms) of one linear at one m, over every library and scaling timed (candidate -> ms)."""

    name = min(row, key=lambda c: (row[c], c))
    return name, row[name]


def timed(row: Mapping, kind: str = "baseline", method: str = "graph") -> dict[str, float]:
    """A `bench.py` shape row's timed entries of `kind`, `family/candidate` -> median ms per call by `method`."""

    out = {}
    for key, by in (row.get("summary") or {}).items():
        k, _, fam, name = key.split("/", 3)
        if k == kind and method in by:
            out[f"{fam}/{name}"] = by[method]["median"]
    return out


def sweep_times(results: Sequence[Mapping], model: str = KERNEL_MODEL,
                method: str = "graph") -> tuple[dict[int, float], dict[int, dict]]:
    """t(m) at each m the sweep's `bench.py` results timed (each linear's fastest stock kernel, summed), and each linear's
    kernel. Refuses an m at which a linear has no timed stock kernel."""

    linears = MODELS[model]
    rows = {row["label"]: row for r in results for row in r.get("shapes", [])}
    ms = sorted({row["m"] for row in rows.values() if (row["n"], row["k"]) in set(linears.values())})
    times, chosen = {}, {}
    for m in ms:
        per = {}
        for linear, (n, k) in linears.items():
            row = rows.get(label(m, n, k))
            got = timed(row, method=method) if row else {}
            if not got:
                why = "not timed" if row is None else row.get("skipped") or "no stock kernel passed its gate"
                raise ValueError(f"{linear} at m = {m} ({label(m, n, k)}): {why}")
            name, ms_ = fastest(got)
            per[linear] = {"label": label(m, n, k), "kernel": name, "ms": ms_}
        times[m], chosen[m] = sum(p["ms"] for p in per.values()), per
    return times, chosen


# ---- γ ------------------------------------------------------------------------------------------------------------------------


def of_record_why(gamma: Mapping) -> str | None:
    """Why a γ isn't a figure of record (too few reps, an interval too wide), or None."""

    if gamma.get("n", 0) < MIN_REPS:
        return f"{gamma.get('n', 0)} reps, fewer than {MIN_REPS}"
    if gamma.get("half_width") is None or gamma["half_width"] > HALF_WIDTH:
        return f"its 95% interval is ±{(gamma.get('half_width') or math.inf):.1%} of the ratio, wider than ±{HALF_WIDTH:.0%}"
    return None


def traced_cost(item: Mapping, busy: Mapping | None) -> dict | None:
    """One traced item's busy SM-time a call (SMs Active) and its kernels' occupied SM-time (the cross-check), in
    SM-seconds; None when nsys saw no kernel in it."""

    if not busy or busy.get("sm_ns") is None:
        return None
    calls = item["burst"]
    return {"sm_s": busy["sm_ns"] * 1e-9 / calls, "occupied_sm_s": busy["occupied_sm_ns"] * 1e-9 / calls,
            "sm_ns": busy["sm_ns"], "occupied_sm_ns": busy["occupied_sm_ns"]}


def host_cost(item: Mapping, card: Gpu) -> float:
    """One timed burst item's host CPU a call, in SM-seconds: the core-seconds the host spent enqueueing it at the card's
    weight."""

    return card.cpu_sm_s_per_core_s * item.get("cpu_s", 0.0) / item["burst"]


def _median(xs: Sequence[float]) -> float | None:
    return statistics.median(xs) if xs else None


def layer_gamma(result: Mapping, busy: Mapping[str, Mapping], card: Gpu, m: int, sch: Scheme,
                model: str = KERNEL_MODEL) -> dict:
    """γ of one summed layer at `m` from a `bench.py --traced --twins` result and each traced item's busy SM-time
    (`nsys.report`'s `busy`, by range text "item <index in traced_items>"). Each linear's stock arm is the fastest of `sch.family`'s
    timed kernels by wall time (the timed pass's graph median); `gamma_least_sm` is over the one of least busy SM-time
    instead. Rep r of the layer is every linear's rep r, summed: the scheme's busy SM-time (the traced pass's rep r) and its
    host CPU above the stock side's (the timed pass's burst rep r), over the stock side's busy SM-time. `aa` is the A/A
    control, each linear's twinned kernel (the tune's fastest finalist) over its original, and the `stock` scheme's γ."""

    graph_ms: dict[tuple, list[float]] = {}
    host: dict[tuple, float] = {}
    for it in result.get("items") or []:
        key = tuple(it["key"])
        if it["method"] == "graph":
            graph_ms.setdefault(key, []).append(it["ms_per_call"])
        elif it["method"] == "burst":
            host[(key, it["rep"])] = host_cost(it, card)
    cost: dict[tuple, dict] = {}
    for i, it in enumerate(result.get("traced_items") or []):
        key = (tuple(it["key"]), it["rep"])
        if key in host and (c := traced_cost(it, busy.get(f"item {i}"))) is not None:
            cost[key] = c | {"cpu_sm_s": host[key]}
    arm = next(iter(result.get("arms") or {}), None) if sch.arm else None
    missing: list[str] = []
    if sch.arm and arm is None:
        missing.append("the result has no arm")
    sides: dict[str, list[tuple]] = {"fastest": [], "least_sm": [], "twin": [], "original": [], "arm": [], "weight": []}
    timed_keys = {key for key, _ in cost}
    linears = {}

    def med(key: tuple, field: str = "sm_s") -> float | None:
        return _median([c[field] for (k2, _), c in cost.items() if k2 == key])

    def side(key: tuple) -> dict:
        return {"sm_s": med(key), "cpu_sm_s": med(key, "cpu_sm_s")}

    for linear, (n, k) in MODELS[model].items():
        lab = label(m, n, k)
        walled = sorted(key for key in graph_ms if key[:3] == ("baseline", lab, sch.family))
        if not walled:
            missing.append(f"{linear} ({lab}): no timed {sch.family} kernel")
            continue
        fast = min(walled, key=lambda key: (_median(graph_ms[key]), key))
        rec: dict = {"label": lab, "stock_fastest": {"kernel": fast[3], "graph_ms": _median(graph_ms[fast]), **side(fast)}}
        if fast not in timed_keys:
            missing.append(f"{linear} ({lab}): the fastest stock kernel {fast[3]} has no busy SM-time")
        else:
            sides["fastest"].append(fast)
        traced = [key for key in walled if key in timed_keys]
        if traced:
            least = min(traced, key=lambda key: (med(key), key))
            sides["least_sm"].append(least)
            rec["stock_least_sm"] = {"kernel": least[3], **side(least)}
        twin = next((key for key in sorted(timed_keys) if key[:3] == ("twin", lab, sch.family)
                     and ("baseline", *key[1:]) in timed_keys), None)
        if twin is None:
            missing.append(f"{linear} ({lab}): no twin with a busy SM-time beside its original (bench.py --twins)")
        else:
            sides["twin"].append(twin)
            sides["original"].append(("baseline", *twin[1:]))
            rec["twin"] = {"kernel": twin[3], **side(twin)}
        if arm is not None:
            for kind in ("arm", "weight"):
                key = (kind, lab, arm, "hash")
                if key not in timed_keys:
                    if kind == "arm":
                        missing.append(f"{linear} ({lab}): no timed {arm} hash call with a busy SM-time (out of domain, "
                                       "rejected or not gated)")
                    continue
                sides[kind].append(key)
                rec[kind] = side(key)
        linears[linear] = rec

    def series(keys: list[tuple], field: str) -> dict[int, float]:
        reps = set.intersection(*({r for (k2, r) in cost if k2 == key} for key in keys)) if keys else set()
        return {r: sum(cost[(key, r)][field] for key in keys) for r in sorted(reps)}

    def whole(keys: list[tuple]) -> bool:
        return len(keys) == len(MODELS[model])

    def pair(a: list[tuple], b: list[tuple], field: str = "sm_s") -> dict | None:
        """a's `field` SM-time and its host CPU above b's, over b's `field` SM-time, rep by rep."""

        # imported here, so the rules above load alone by this file's path, with no package beside them
        from verity_bench.kit import paired

        if not a or not whole(b):
            return None
        sa, sb, ca, cb = series(a, field), series(b, field), series(a, "cpu_sm_s"), series(b, "cpu_sm_s")
        reps = sorted(set(sa) & set(sb))
        return paired([sa[r] + ca[r] - cb[r] for r in reps], [sb[r] for r in reps]) if reps else None

    out: dict = {"m": m, "passes": {"sm_s": "traced_items", "cpu_sm_s": "items (burst)", "wall": "items (graph)"},
                 "scheme": sch.id, "family": sch.family, "arm": arm, "linears": linears,
                 "aa": pair(sides["twin"], sides["original"]) if whole(sides["twin"]) else None}
    if sch.arm:
        armed = whole(sides["arm"])
        out |= {"gamma": pair(sides["arm"], sides["fastest"]) if armed else None,
                "gamma_least_sm": pair(sides["arm"], sides["least_sm"]) if armed else None,
                "gamma_occupancy": pair(sides["arm"], sides["fastest"], "occupied_sm_s") if armed else None,
                "gamma_weight_side_bound": pair(sides["arm"] + sides["weight"], sides["fastest"])
                if armed and whole(sides["weight"]) else None}
    else:
        out |= {"gamma": out["aa"],
                "gamma_occupancy": pair(sides["twin"], sides["original"], "occupied_sm_s") if whole(sides["twin"]) else None}
    ratios = {}
    for key in {key for keys in sides.values() for key in keys}:
        sm = sum(c["sm_ns"] for (k2, _), c in cost.items() if k2 == key)
        occ = sum(c["occupied_sm_ns"] for (k2, _), c in cost.items() if k2 == key)
        ratios["/".join(key)] = sm / occ if occ else None
    out["cross_check"] = {"ratios": ratios, "bound": CROSS_CHECK}
    out["missing"] = missing
    return out


def cross_check_why(ratios: Mapping[str, float | None], bound: float = CROSS_CHECK) -> str | None:
    """Why the SMs Active reading can't stand: an entry whose busy SM-time exceeds its kernels' occupied SM-time (each
    kernel's duration times min(SMs, blocks), the most SMs Active can read while it runs: `nsys.occupied_ns`) by more than
    `bound`, or one without kernels. Below it, the ratio reads the SMs the kernels left idle (a partial last wave, a
    persistent grid's uneven tiles, a block that waits on another), which hides no work: SMs Active counts every cycle an
    SM holds a warp."""

    off = {k: r for k, r in ratios.items() if r is None or r - 1 > bound}
    if not off:
        return None
    worst = max(off, key=lambda k: math.inf if off[k] is None else off[k])
    return (f"{len(off)} entries' SMs Active SM-time exceeds their kernels' occupied SM-time by over {bound:.0%} "
            f"({worst}: {off[worst]})")


def point_whys(p: Mapping) -> list[str]:
    """Why one point's γ isn't of record: a linear missing a side, too few reps or too wide an interval, the cross-check,
    an arm row the reference verifier didn't accept."""

    whys = list(p.get("missing") or [])
    g = p.get("gamma")
    whys.append("no γ" if g is None else of_record_why(g))
    whys.append(cross_check_why((p.get("cross_check") or {}).get("ratios") or {}))
    for linear, v in (p.get("verified") or {}).items():
        if not v.get("accepted"):
            whys.append(f"{linear}: the reference verifier didn't accept its transcript ({v.get('why')})")
    return [w for w in whys if w]


# ---- per monitoring period ------------------------------------------------------------------------------------------------------

PERIODS_H = {"week": 7 * 24, "month": 30 * 24, "year": 365 * 24}


def periods(gamma: float, gpus: int, window_s: float, world_flop_per_year: float, flop_per_s_per_gpu: float,
            gates_per_flop: float) -> dict:
    """What γ costs a window and each monitoring period, pooled over all compute: `share`, γ/(1 + γ), of the compute a
    prover spends goes to proving. The datacenter's (`gpus` of the card's datasheet rate, audited in windows of `window_s`):
    the GPU-hours spent proving, and their compute in FLOP and gates. The world's (the adversary resource model's FLOP a
    year as useful work): the compute proving it adds, γ times it, in FLOP and gates."""

    share = gamma / (1 + gamma)
    year_h = PERIODS_H["year"]

    def at(hours: float) -> dict:
        dc_h = gpus * hours * share
        world = world_flop_per_year * hours / year_h * gamma
        return {"datacenter_gpu_h": dc_h, "datacenter_gates": dc_h * 3600 * flop_per_s_per_gpu * gates_per_flop,
                "world_flop": world, "world_gates": world * gates_per_flop}

    return {"share": share, "window": at(window_s / 3600)} | {name: at(hours) for name, hours in PERIODS_H.items()}



def paired(scheme: Sequence[float], stock: Sequence[float], confidence: float = 0.95) -> dict:
    """`kit.paired` by this module's name, which serving cost's `benchmarks/serve_tax` calls and checks until it calls
    `kit.paired`. The kit is imported on the call, so this file still loads alone by its path."""

    from verity_bench import kit

    return kit.paired(scheme, stock, confidence)
