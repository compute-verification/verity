"""numpy models of GLM-5.3's and DeepSeek-V4-Pro's bf16 GEMMs on sm_90, the arithmetic `ll_bf16` and `cublas_bf16` compute, and
vLLM d9105ea80's choice of kernel for the MoE gate (`GateLinear.forward`, `LLBf16Gemm.dispatch`):

* `gate_rows`: x (M, K) bf16 against the gate's W (N, K) -> f32 words (M, N); M <= 16 is `ll_bf16_gemm` (`dispatch_sm90`), past
  it cuBLAS's ``torch.mm(x, W.T, out_dtype=float32)`` (`f32out_rows`);
* `dotprod_rows`: `LlBf16DotprodGemm_v1` per token row: BS lane chains from +0 (FFMA, or FMUL then FADD) over `dotprod_rounds`,
  each warp's butterfly, then the warp sums in order from +0;
* `splitk_rows`: `LlBf16SplitKGemm_v1{DOT=HopperBF16WgmmaDot16_v1}` per token row: each (cluster rank, MMA warp)'s k16 blocks
  ascend, so its chain is `DotBf16` over those columns (`mla_check.dot_chain`), then the warps' and the ranks' ordered sums;
* `f32out_rows`: `GemmF32Out_v1{DOT=HopperBF16WgmmaDot16_v1}` (also DeepSeek-V4-Pro's compressor GEMMs); `linear_rows`: core's
  `Gemm_v2` (lm_head, GLM-5.3's indexer `wk_weights_proj`, DeepSeek-V4-Pro's `weights_proj`); `bmm_rows`: `BatchedGemm_v1`,
  `mla_check.bmm_rows`;
* `cublas_splitk_rows`: cuBLAS 12.9.1's H200 route in `cublas_bf16`: equal 64-element-aligned contiguous chunks, a final
  remainder, f32 partials and a serial increasing-order f32 reduction. `default_cublas_route` reads the selected route at
  every decode batch to `CUBLAS_ROUTES["m_max"]` tokens out of `cublas_routes.json`, the capture's decode-batch scan
  (`glm53_bf16_gemm_capture_gpu.scan_routes`), in which `fit_cublas_split_k` names the split counts whose route computes a
  launch's observed words exactly, so the split count is read out of the words (Kineto reports grid.z = 1 for these launches,
  so no profile field carries it).  Any other model's bf16-output routes are the cc 9.0 device pin's (`pinned_routes`,
  `verity_vllm/pins/`), measured by `served routes`' probe, which the predicted served Program reads too.  The routes hold
  under the scan's conditions only (`ROUTE_CONDITIONS`): `route_departures` names how a serving process's facts differ, which a
  Commit's verdict, TP1's and each TP rank's, refuses for a Program that binds one of the `ROUTE_DEFINITIONS`
  (`check.cublas_route_pin`).  So a torch or cuBLAS upgrade in the serving venv refuses every such Program until the scan
  reruns under it, on an H200 with the serving venv's python, as `CUBLAS_ROUTES["run"]` ran:
  `tests/properties/glm53_bf16_gemm_capture_gpu.py --scan-run --served-default --models glm53,dsv4-pro --scratch DIR --out OUT` with
  `--only linear,bmm,f32out --scan-m 1:2048`, then `--only router --scan-m 17:2048`; its cases' `routes`, its `switch` facts and the
  device's are then copied into `cublas_routes.json` (no tool writes it);
* the cc 12.0 routes beside it: `cublas_splitk_rows(partial="bf16")` (a bf16 workspace), `cublas_aligned_splitk_rows` (the
  same over grid.z slices at an alignment, each slice's k-tiles residue first), `cublas_serial_splitk_rows` (slices meeting through the bf16 output) and
  `cublas_gemv_rows` (M = 1 gemvx lanes); `cublas_route_rows` is a device pin route's twin, and `cublas_routes_at` with
  `route_probe_rows` what the probe tells apart.

NaN payloads are not part of any comparison (`mla_check._words_equal`): the GPU's FFMA and FADD write 0x7FFFFFFF, numpy keeps
an operand's.
"""
from __future__ import annotations

import json
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.registry import cublas_bf16 as CB
from verity.ml.definitions.registry import mla_check as MK

U32 = np.uint32
WARP = 32
#: `LLBf16Gemm`'s defaults and its sm_90 tables by gate (K, N), vLLM d9105ea80 `cute_dsl/ll_bf16.py`
DOTPROD_MAX_M, DOTPROD_BS, DOTPROD_VEC = 4, 128, 8
SPLITK_DEFAULT, SPLITK_TILE_K, SPLITK_WARPS = 6, 256, 4
SM90_DOTPROD_BS = {(4096, 256): {1: 256, 3: 256}, (7168, 384): {1: 256, 2: 256}}
SM90_SPLITK = {(4096, 256): {**{m: 8 for m in range(5, 8)}, 10: 8, 11: 8, 13: 8, 14: 8, 15: 8, 16: 8},
               (7168, 256): {8: 6},
               (6144, 256): {9: 8, 11: 8},
               (7168, 384): {m: (8 if m in (6, 8, 12) else 7) for m in range(6, 17)}}
LL_MAX_M = 16
DEFAULT_LL_BF16_FMA = 1
#: cuBLAS 12.9.1's default route on the H200 by "phase K N": [m0, m1, split] runs over M (split None: the non-split serial
#: chain), as the capture's decode-batch scan (run `run`) mapped them
CUBLAS_ROUTES = json.loads(Path(__file__).with_name("cublas_routes.json").read_text())
_ROUTES = {(p, int(k), int(n)): runs for (p, k, n), runs in ((key.split(), v) for key, v in CUBLAS_ROUTES["routes"].items())}
#: what the scan's process ran on and under, as `glm53_bf16_gemm_capture_gpu.switch` read it (its device beside): the device, torch,
#: the cuBLAS it mapped, its reduced-precision flags, preferred backend and workspace variables.  cuBLAS picks a route from these
#: and the launch's shape, so the routes are a serving process's own only when its facts of record (`cublas` in
#: `check.replay.driver.engine_facts_of_record`) are these
ROUTE_CONDITIONS: dict[str, Any] = CUBLAS_ROUTES["conditions"]
#: the Definitions a Program binds only at a route of the pin (the vocabulary's `cublas_*` roles, which `vllm_bindings.bf16_gemm`, the
#: fold and the predictor bind at `default_cublas_route`); `Gemm_v2` is bound at the pin's non-split linear routes too, but also by
#: every other rule, so its presence does not say the pin was read
ROUTE_DEFINITIONS = (CB.CublasSplitKGemmWord, CB.CublasSplitKGemmF32OutWord, CB.GemmF32OutWord)


def route_bound(by_family: dict[str, int]) -> dict[str, int]:
    """The entries of `by_family` (a Program's row counts by Definition id) whose Definition is one of `ROUTE_DEFINITIONS`."""
    ids = {d.id for d in ROUTE_DEFINITIONS}
    return {f: n for f, n in sorted(by_family.items()) if f in ids}


def route_departures(engine: dict[str, Any] | None) -> list[str]:
    """Each way a serving process's facts of record (`engine_facts_of_record`) are not `ROUTE_CONDITIONS`, by name: a fact recorded
    otherwise, or not recorded (a fact the process could not read is recorded as 'unavailable: <why>', which no condition is);
    [] when it ran under them."""
    seen = (engine or {}).get("cublas")
    if not isinstance(seen, dict):
        return [f"no cuBLAS facts recorded (engine_of_record.cublas; the pin's: {ROUTE_CONDITIONS})"]
    env = seen.get("env") if isinstance(seen.get("env"), dict) else {}
    pairs = [(k, seen.get(k, KeyError), w) for k, w in ROUTE_CONDITIONS.items() if k != "env"]
    pairs += [(v, env.get(v, KeyError), w) for v, w in ROUTE_CONDITIONS["env"].items()]
    return [f"{k} not recorded (pin: {w!r})" if got is KeyError else f"{k} {got!r} (pin: {w!r})" for k, got, w in pairs if got != w]


def _route_run(phase: str, M: int, K: int, N: int) -> list | None:
    return next((r for r in _ROUTES.get((phase, K, N), ()) if r[0] <= M <= r[1]), None)


def default_cublas_split_k(phase: str, M: int, K: int, N: int) -> int | None:
    """The split count selected by cuBLAS 12.9.1 on the pinned H200, or None for its non-split routes."""
    run = _route_run(phase, M, K, N)
    return run[2] if run else None


def _pins():
    """Each pinned device's measured facts, `<its name in vLLM's file-name form>.json`: its compute capability, multiprocessor count,
    library versions and cuBLAS routes.  They are the integration's, read when asked for, so loading the Definitions imports
    nothing of it."""
    return resources.files("verity_vllm") / "pins"


@cache
def device_pin(device: str) -> dict[str, Any]:
    """The pin of the device that reports `device` (`torch.cuda.get_device_name`)."""
    path = _pins() / f"{device.replace(' ', '_')}.json"
    if not path.is_file():
        raise KeyError(f"no device pin {path.name} in verity_vllm/pins")
    return json.loads(path.read_text())


@cache
def capability_pin(cc: tuple[int, int]) -> dict[str, Any]:
    """The one pinned device of compute capability `cc`."""
    pins = [json.loads(p.read_text()) for p in _pins().iterdir() if p.name.endswith(".json")]
    got = [p for p in pins if tuple(p["compute_capability"]) == tuple(cc)]
    if len(got) != 1:
        raise KeyError(f"{len(got)} device pins of compute capability {cc} in verity_vllm/pins; one is required")
    return got[0]


def pinned_routes() -> dict[str, dict[str, Any]]:
    """The cc 9.0 device's bf16-output cuBLAS route at each probed (M, K, N), "M,K,N" -> {"kernel": "nosplit"} or {"kernel": "splitk",
    "SPLIT_K": s}: `torch.mm(x, w.t())`, which an unbiased `aten.linear` and the served wrapper's extern mm both run."""
    return capability_pin((9, 0))["cublas"]


def default_cublas_route(phase: str, M: int, K: int, N: int) -> tuple[str, dict[str, int]] | None:
    """The captured default cuBLAS route, else (a linear) the cc 9.0 device pin's, distinguishing a known non-split launch from a
    shape/M with no record."""
    run = _route_run(phase, M, K, N)
    if run is not None:
        return ("splitk", {"SPLIT_K": run[2]}) if run[2] is not None else ("non_split", {})
    if phase == "linear" and (r := pinned_routes().get(f"{M},{K},{N}")) is not None:
        return ("splitk", {"SPLIT_K": r["SPLIT_K"]}) if r["kernel"] == "splitk" else ("non_split", {})
    return None


SPLITK_CANDIDATES = range(2, 17)


def fit_cublas_split_k(x: np.ndarray, w: np.ndarray, got: np.ndarray, *, out: str,
                       candidates: range = SPLITK_CANDIDATES) -> list[int]:
    """The split counts in `candidates` under which the modeled route (`cublas_splitk_rows` at the default hypothesis,
    `CUBLAS_SPLITK_HYPOTHESIS`) computes exactly the observed words `got` of x's rows against every row of w; [] names a route
    no candidate models.  Under `splitk_count_probe` rows the fit is unique by construction; elsewhere more than one fit means
    these rows cannot tell the candidates apart."""
    x, w, got = np.asarray(x, np.uint16), np.asarray(w, np.uint16), np.asarray(got)
    fits = []
    for s in candidates:
        if not 1 < s <= x.shape[1] // 16:
            continue
        want = cublas_splitk_rows(x, w, s, out=out)
        if not (~MK._words_equal(got, want, f32_words=out == "f32")).any():
            fits.append(int(s))
    return fits


def splitk_count_probe(K: int, max_rows: int, candidates: range = SPLITK_CANDIDATES) -> np.ndarray:
    """W rows that tell every candidate split count apart through an all-ones activation row, under either output rounding.

    Each row tests one chunk boundary t of some candidate's partition: -2^25 at t - 16, +2^25 at t and +1 at t + 16 (zeros
    elsewhere).  A serial chain that crosses no boundary at t cancels the pair exactly before adding the +1; a partition split
    at t reduces two +-2^25 partials whose chunk chains absorbed (lost) the +1.  The 1-against-0 difference survives the bf16
    rounding, unlike the low-bit reassociation noise random rows leave, so `fit_cublas_split_k` over these rows reads the
    served partition out of the words at any output type.  A greedy cover picks boundaries until every pair of candidates with
    distinct partitions differs somewhere and every candidate has a probed boundary (so the unsplit serial chain fits no
    candidate), and the builder verifies those distinctions on the models, raising otherwise."""
    cands = [s for s in candidates if 1 < s <= K // 16]
    sets = {s: frozenset(int(c[0]) for c in _split_cols(K, s, "contiguous")[1:]) for s in cands}
    usable = sorted({t for b in sets.values() for t in b if 16 <= t <= K - 32})
    pairs = {(a, b) for i, a in enumerate(cands) for b in cands[i + 1:] if sets[a] != sets[b]}
    chosen: list[int] = []
    while pairs:
        t = max(usable, key=lambda t: sum((t in sets[a]) != (t in sets[b]) for a, b in pairs))
        covered = {(a, b) for a, b in pairs if (t in sets[a]) != (t in sets[b])}
        if not covered:
            raise ValueError(f"split count probe cannot separate {sorted(pairs)} at K {K}")
        chosen.append(t)
        pairs -= covered
    for s in cands:
        if not sets[s] & set(chosen):
            hit = next((t for t in usable if t in sets[s]), None)
            if hit is None:
                raise ValueError(f"split count probe has no usable boundary for {s} splits at K {K}")
            chosen.append(hit)
    if len(chosen) > max_rows:
        raise ValueError(f"split count probe needs {len(chosen)} rows at K {K}, only {max_rows} available")
    w = np.zeros((len(chosen), K), np.uint16)
    for i, t in enumerate(chosen):
        w[i, t - 16], w[i, t], w[i, t + 16] = 0xCC00, 0x4C00, 0x3F80
    x = np.full((1, K), 0x3F80, np.uint16)
    for out in ("f32", "bf16"):
        serial = MK.dot_chain(x, w) if out == "f32" else MK.bf16_of(MK.dot_chain(x, w))
        models = {s: cublas_splitk_rows(x, w, s, out=out) for s in cands}
        for i, a in enumerate(cands):
            if np.array_equal(models[a], serial):
                raise ValueError(f"split count probe does not tell {a} splits from the serial chain at K {K} ({out})")
            for b in cands[i + 1:]:
                if sets[a] != sets[b] and np.array_equal(models[a], models[b]):
                    raise ValueError(f"split count probe does not tell {a} from {b} apart at K {K} ({out})")
    return w


def dispatch_sm90(M: int, K: int, N: int) -> tuple[str, dict]:
    """The kernel `GateLinear` runs for a bf16 gate on sm_90 at M tokens: ``("dotprod", {BS, VEC})``, ``("splitk", {SPLIT_K,
    TILE_K, WARPS})`` (`ll_bf16_gemm`), or ``("cublas", {})`` (tier 4).  The split-K tables' second entry, ``num_stages``, only
    pipelines the loads."""
    if M <= 0:
        raise ValueError(f"M = {M}: no token rows")
    if M > LL_MAX_M:
        return "cublas", {}
    if M <= DOTPROD_MAX_M or K < 2048:
        return "dotprod", {"BS": SM90_DOTPROD_BS.get((K, N), {}).get(M, DOTPROD_BS), "VEC": DOTPROD_VEC}
    return "splitk", {"SPLIT_K": SM90_SPLITK.get((K, N), {}).get(M, SPLITK_DEFAULT), "TILE_K": SPLITK_TILE_K,
                      "WARPS": SPLITK_WARPS}


def _ordered_sum(parts) -> np.ndarray:
    out = np.zeros(np.shape(parts[0]), U32)
    for p in parts:
        out = MK.add(out, p)
    return out


def _reduce(parts: list[np.ndarray], order: str) -> np.ndarray:
    """F32-add partials in a hypothesized reduction order."""
    if order == "forward":
        return _ordered_sum(parts)
    if order == "reverse":
        return _ordered_sum(parts[::-1])
    if order == "tree":
        level = list(parts)
        while len(level) > 1:
            level = [MK.add(level[i], level[i + 1]) if i + 1 < len(level) else level[i] for i in range(0, len(level), 2)]
        return level[0]
    raise ValueError(f"unknown reduction order {order!r}")


def _split_cols(K: int, splits: int, partition: str) -> list[np.ndarray]:
    """k16-block columns under the contiguous or round-robin partition hypothesis."""
    if K <= 0 or K % 16 or not 1 < splits <= K // 16:
        raise ValueError(f"cublas split-k: K {K}, splits {splits}")
    blocks = K // 16
    if partition == "contiguous":
        chunk = ((K + splits - 1) // splits + 63) // 64 * 64
        bounds = list(range(0, K, chunk)) + [K]
        if len(bounds) != splits + 1:
            bounds = [16 * (z * blocks // splits) for z in range(splits)] + [K]
        return [np.arange(lo, hi) for lo, hi in zip(bounds, bounds[1:])]
    if partition == "strided":
        return [np.concatenate([np.arange(16 * b, 16 * b + 16) for b in range(z, blocks, splits)]) for z in range(splits)]
    raise ValueError(f"unknown split partition {partition!r}")


def cublas_splitk_rows(x: np.ndarray, w: np.ndarray, SPLIT_K: int, *, partition: str = "contiguous",
                       partial: str = "f32", reduction: str = "forward", out: str = "f32") -> np.ndarray:
    """One cuBLAS split-K hypothesis: k16 chains from +0, then a reduction of the partials and the requested output rounding."""
    x, w = np.asarray(x, np.uint16), np.asarray(w, np.uint16)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError(f"cublas split-k: x {x.shape}, W {w.shape}")
    parts = [MK.dot_chain(x[:, cols], w[:, cols]) for cols in _split_cols(x.shape[1], SPLIT_K, partition)]
    if partial == "bf16":
        parts = [DR.bf16_to_f32_words(MK.bf16_of(p)) for p in parts]
    elif partial != "f32":
        raise ValueError(f"unknown partial precision {partial!r}")
    result = _reduce(parts, reduction)
    if out == "bf16":
        return MK.bf16_of(result)
    if out != "f32":
        raise ValueError(f"unknown output precision {out!r}")
    return result


CUBLAS_SPLITK_HYPOTHESIS = "contiguous-f32-forward"
CUBLAS_SPLITK_ALTERNATIVES = (
    CUBLAS_SPLITK_HYPOTHESIS,
    "strided-f32-forward",
    "contiguous-f32-reverse",
    "contiguous-f32-tree",
    "contiguous-bf16-forward",
)


def cublas_splitk_hypotheses(x: np.ndarray, w: np.ndarray, SPLIT_K: int, *, out: str) -> dict[str, np.ndarray]:
    """The capture probe's distinct split partition, partial precision and reduction-order alternatives.

    With two partials, reverse order is the same single commutative add as forward order; with two or three partials, this
    pairwise tree has exactly forward order's parenthesization.  Those are not distinct hypotheses at those split counts.
    """
    ans = {}
    for name in CUBLAS_SPLITK_ALTERNATIVES:
        partition, partial, reduction = name.split("-")
        if (reduction == "reverse" and SPLIT_K == 2) or (reduction == "tree" and SPLIT_K <= 3):
            continue
        ans[name] = cublas_splitk_rows(x, w, SPLIT_K, partition=partition, partial=partial, reduction=reduction, out=out)
    return ans


def cublas_serial_splitk_rows(x: np.ndarray, w: np.ndarray, SPLIT_K: int) -> np.ndarray:
    """`CublasSerialSplitKGemm_v1{K, N, SPLIT_K, HopperBF16WgmmaDot16_v1}` of every row of x (M, K) against w (N, K): the contiguous
    chunks in order, each chunk's k16 chain from +0 added in f32 to the previous chunk's bf16 output and rounded to bf16 again
    (the first rounded alone): bf16 words (M, N)."""
    x, w = np.asarray(x, np.uint16), np.asarray(w, np.uint16)
    out = None
    for cols in _split_cols(x.shape[1], SPLIT_K, "contiguous"):
        p = MK.dot_chain(x[:, cols], w[:, cols])
        out = MK.bf16_of(p if out is None else MK.add(p, DR.bf16_to_f32_words(out)))
    return out


def cublas_aligned_splitk_rows(x: np.ndarray, w: np.ndarray, SPLIT_K: int, ALIGN: int, TILE_K: int) -> np.ndarray:
    """`CublasSplitKGemmBf16PartialsAligned_v1{K, N, SPLIT_K, ALIGN, TILE_K, HopperBF16WgmmaDot16_v2}` of every row of x (M, K)
    against w (N, K): `cublas_bf16.aligned_bounds`' chunks, each chunk's k16 chain over its k-tiles residue first
    (`cublas_bf16.residue_first`), partials rounded to bf16 and widened, a serial increasing-order f32 reduction and one bf16
    rounding: bf16 words (M, N)."""
    x, w = np.asarray(x, np.uint16), np.asarray(w, np.uint16)
    parts = []
    for lo, hi in CB.aligned_bounds(x.shape[1], SPLIT_K, ALIGN):
        r, z = CB.residue_first(lo, hi, TILE_K)
        xs, ws = (np.hstack([a[:, lo:lo + r], np.zeros((a.shape[0], z), np.uint16), a[:, lo + r:hi]]) for a in (x, w))
        parts.append(DR.bf16_to_f32_words(MK.bf16_of(MK.dot_chain(xs, ws))))
    return MK.bf16_of(_ordered_sum(parts))


def cublas_gemv_rows(x: np.ndarray, w: np.ndarray, LANES: int, FMA: int = 1) -> np.ndarray:
    """`CublasGemv_v1{K, N, LANES, FMA}` of every row of x (M, K) against w (N, K): lane t of LANES runs one f32 chain from +0 over
    k = t, t + LANES, ... (FFMA when FMA, else FMUL then FADD), the lanes meet in a butterfly (offsets LANES/2, ..., 1: lane i
    adds lane i + o), and lane 0's sum, added to +0 (so a -0 sum leaves as +0), is rounded to bf16: bf16 words (M, N)."""
    x, w = np.asarray(x, np.uint16), np.asarray(w, np.uint16)
    (M, K), N = x.shape, w.shape[0]
    xf, wf = DR.bf16_to_f32_words(x), DR.bf16_to_f32_words(w)
    acc = np.zeros((M, N, LANES), U32)
    for k0 in range(0, K, LANES):
        n = min(LANES, K - k0)
        a, b, c = xf[:, None, k0:k0 + n], wf[None, :, k0:k0 + n], acc[..., :n]
        acc[..., :n] = MK.fma(a, b, c) if FMA else MK.add(c, MK.mul(a, b))
    o = LANES // 2
    while o:
        acc = MK.add(acc[..., :o], acc[..., o:2 * o])
        o //= 2
    return MK.bf16_of(MK.add(np.zeros((M, N), U32), acc[..., 0]))


def cublas_route_rows(route: dict[str, Any], x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The bf16 words of `torch.mm(x, w.t())` (x (M, K), w (N, K)) along a probed cuBLAS route, a device pin's ``cublas`` entry."""
    kind = route["kernel"]
    if kind == "nosplit":
        return linear_rows(x, w)
    if kind == "splitk" and "align" in route:
        if route.get("partial") != "bf16":
            raise ValueError(f"unknown cuBLAS route {route}")
        return cublas_aligned_splitk_rows(x, w, int(route["SPLIT_K"]), int(route["align"]), int(route["tile_k"]))
    if kind == "splitk":
        return cublas_splitk_rows(x, w, int(route["SPLIT_K"]), partial=route.get("partial", "f32"), out="bf16")
    if kind == "serialk":
        return cublas_serial_splitk_rows(x, w, int(route["SPLIT_K"]))
    if kind == "gemv":
        return cublas_gemv_rows(x, w, int(route["LANES"]))
    raise ValueError(f"unknown cuBLAS route {route}")


def cublas_routes_at(fits: list[int]) -> list[dict[str, Any]]:
    """The routes `splitk_count_probe`'s words leave: the unsplit ones (the k16 chain, the gemvx lanes at 16 and 32, whose
    count-probe words need not be the chain's) and, at each fitting split count, the f32- and bf16-workspace split-K and the serial
    split-K."""
    return ([{"kernel": "nosplit"}, {"kernel": "gemv", "LANES": 16}, {"kernel": "gemv", "LANES": 32}]
            + [r for s in fits for r in ({"kernel": "splitk", "SPLIT_K": s}, {"kernel": "splitk", "SPLIT_K": s, "partial": "bf16"},
                                         {"kernel": "serialk", "SPLIT_K": s})])


def route_probe_rows(M: int, K: int, C: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """Two probes of a route at (M, K), each (x (M, K), w (C, K)) bf16 words sampled from `rng`: spread rows (x of exponents -6 to 6,
    w of -10 to 2), and cancelling rows (x all ones; each W row small words and one +-2^8..2^13 pair, so which chain each product
    lands in decides what the chains absorb, and the small sum shows it through the bf16 rounding)."""
    def words(shape, lo, hi):
        f = rng.standard_normal(shape, dtype=np.float32) * np.exp2(rng.integers(lo, hi, shape)).astype(np.float32)
        return (f.view(U32) >> 16).astype(np.uint16)
    wc = DR.bf16_to_f32_words(words((C, K), -12, -5)).view(np.float32)
    k1 = rng.integers(0, K, C)
    k2 = (k1 + rng.integers(1, K, C)) % K
    big = np.exp2(rng.integers(8, 14, C)).astype(np.float32)
    wc[np.arange(C), k1], wc[np.arange(C), k2] = big, -big
    return [(words((M, K), -6, 7), words((C, K), -10, 3)), (np.full((M, K), 0x3F80, np.uint16), (wc.view(U32) >> 16).astype(np.uint16))]


def dotprod_rounds(K: int, BS: int, VEC: int) -> list[np.ndarray]:
    """`LLBf16Dotprod`'s K loop as rounds, each the k every thread adds then (-1: none): the vector loads at width VEC over K's
    whole tiles of VEC BS, then at width VEC/2 over what is left (its 64-bit tail at VEC 8), then scalar rounds of BS, the last
    ragged."""
    t, rounds, k0 = np.arange(BS), [], 0
    for width in (VEC, VEC // 2):
        tiles = (K - k0) // (width * BS) if width else 0
        rounds += [k0 + width * (BS * i + t) + v for i in range(tiles) for v in range(width)]
        k0 += tiles * width * BS
    rounds += [np.where(s + t < K, s + t, -1) for s in range(k0, K, BS)]
    return rounds


def dotprod_rows(x: np.ndarray, w: np.ndarray, BS: int = DOTPROD_BS, VEC: int = DOTPROD_VEC, FMA: int = 1) -> np.ndarray:
    """`LlBf16DotprodGemm_v1{N, K, BS, VEC, FMA}` of every row of x (M, K) against w (N, K): f32 words (M, N)."""
    x, w = np.asarray(x, np.uint16), np.asarray(w, np.uint16)
    (M, K), N = x.shape, w.shape[0]
    if K < BS or BS % WARP or VEC < 1 or w.shape[1] != K:
        raise ValueError(f"dotprod: K {K} at least BS {BS}, BS of 32, VEC {VEC} >= 1 and W [N, K] (W {w.shape})")
    xf, wf = DR.bf16_to_f32_words(x), DR.bf16_to_f32_words(w)
    acc = np.zeros((M, N, BS), U32)
    for ks in dotprod_rounds(K, BS, VEC):
        a, b = xf[:, None, np.maximum(ks, 0)], wf[None, :, np.maximum(ks, 0)]
        acc = np.where(ks >= 0, MK.fma(a, b, acc) if FMA else MK.add(acc, MK.mul(a, b)), acc)
    lanes = acc.reshape(M, N, BS // WARP, WARP)
    o = WARP // 2
    while o:
        lanes = MK.add(lanes[..., :o], lanes[..., o:2 * o])
        o //= 2
    return _ordered_sum([lanes[:, :, j, 0] for j in range(BS // WARP)])


def splitk_rows(x: np.ndarray, w: np.ndarray, SPLIT_K: int, TILE_K: int = SPLITK_TILE_K, WARPS: int = SPLITK_WARPS) -> np.ndarray:
    """`LlBf16SplitKGemm_v1{N, K, SPLIT_K, TILE_K, WARPS, HopperBF16WgmmaDot16_v1}` of every row of x (M, K) against w (N, K)."""
    x, w = np.asarray(x, np.uint16), np.asarray(w, np.uint16)
    K = x.shape[1]
    tiles, blocks = K // TILE_K, TILE_K // 16
    if K % TILE_K or TILE_K % 16 or blocks % WARPS or not 0 < SPLIT_K <= tiles or w.shape[1] != K:
        raise ValueError(f"splitk: K {K}, TILE_K {TILE_K}, WARPS {WARPS}, SPLIT_K {SPLIT_K} (W {w.shape})")
    ranks = []
    for z in range(SPLIT_K):
        parts = []
        for j in range(WARPS):
            cols = np.concatenate([np.arange(16) + TILE_K * t + 16 * (j + WARPS * ki)
                                   for t in range(z, tiles, SPLIT_K) for ki in range(blocks // WARPS)])
            parts.append(MK.dot_chain(x[:, cols], w[:, cols]))
        ranks.append(_ordered_sum(parts))
    return _ordered_sum(ranks)


def f32out_rows(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`GemmF32Out_v1{K, N, HopperBF16WgmmaDot16_v1}`: x (M, K), w (N, K) -> f32 words (M, N)."""
    return MK.dot_chain(x, w)


def linear_rows(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """core's `Gemm_v2{K, N, HopperBF16WgmmaDot16_v1}`: x (M, K), w (N, K) -> bf16 words (M, N)."""
    return MK.bf16_of(MK.dot_chain(x, w))


def bmm_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`BatchedGemm_v1{B, K, N, HopperBF16WgmmaDot16_v1}`: a (B, M, K), b (B, K, N) -> bf16 words (B, M, N)."""
    return MK.bmm_rows(a, b)


def gate_rows(x: np.ndarray, w: np.ndarray, FMA: int = 1) -> np.ndarray:
    """A bf16 MoE gate on sm_90: the kernel `dispatch_sm90` names at x's M and the gate's (K, N), as the Definition that is its
    function."""
    kind, st = dispatch_sm90(x.shape[0], x.shape[1], w.shape[0])
    if kind == "dotprod":
        return dotprod_rows(x, w, FMA=FMA, **st)
    if kind == "splitk":
        return splitk_rows(x, w, **st)
    return f32out_rows(x, w)
