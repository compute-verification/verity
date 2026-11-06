"""Torch-free recomputation of vLLM d9105ea80's Gumbel-max sampler noise -- the executable half of an `rng` root class.

vLLM's sampler (`vllm/v1/worker/gpu/sample/gumbel.py::_gumbel_sample_kernel`) draws no random tensor from a torch
generator: for every logits row it evaluates, inside the kernel, a *pure function* of three integers --

    gumbel_seed = tl.randint(seed, pos)                    # Philox4x32-10, key = request seed (int64), counter = position
    u          = tl.rand(gumbel_seed, vocab_index)         # Philox4x32-10, key = gumbel_seed (uint32), counter = vocab index
    u          = max(u, 4.6566127342e-10)                  # fp32
    g          = -log(-log1p(-u))                          # fp32 (libdevice logf / log1pf on the GPU)
    token      = argmax_v(logits[v] / T + g[v])            # fp32, first index on ties

-- where `seed` is the request's `SamplingParams.seed` (or a host-side `np.random.randint` draw when None:
`sample/states.py`), and `pos` is the absolute position of the sampled token.  This module recomputes the chain with numpy
integers and IEEE fp32 arithmetic (Triton's `python/triton/language/random.py`, v3.7.1, transcribed line by line), so a
validator can decide whether a captured sampled token is *explained* by (seed, position, logits) alone.

Exactness budget (what is bit-exact and what is not):
  * Philox, the int32 -> fp32 conversion and the `* 2**-31` scaling are exact (integer arithmetic and single IEEE ops).
  * `logf` / `log1pf` are libdevice on the GPU (<= 1 ulp each per the CUDA math manual) vs the platform libm here: the noise may
    differ by a few fp32 ulps, so a recomputed token is certified only when the winner's margin over the runner-up exceeds
    the tolerance -- exactly the margin discipline `TokenSelect` already applies to argmax ties.
  * `logits / T` inside the kernel (APPLY_TEMPERATURE=True) is Triton's `div.full.f32` (approximate, <= 2 ulp); vLLM's default
    path divides in `_temperature_kernel` first (fp32, stored) and calls the sampler with APPLY_TEMPERATURE=False, in which
    case the division result is *in the captured logits* and the argmax input is exact.

CLI (laptop, CPU-only, needs the run's log + snapshots):
    verity-vllm sampling-rng --run RUN_DIR [--out RUN_DIR/rng_check.json]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from dataclasses import dataclass

from verity_vllm.config import option

# Triton `random.py` (32-bit path) and vLLM `gumbel.py` constants
PHILOX_KEY_A = np.uint32(0x9E3779B9)
PHILOX_KEY_B = np.uint32(0xBB67AE85)
PHILOX_ROUND_A = np.uint32(0xD2511F53)
PHILOX_ROUND_B = np.uint32(0xCD9E8D57)
TL_RAND_SCALE_F32 = np.float32(4.6566127342e-10)     # `uint_to_uniform_float`: (2**23 - 1) / 2**23 * 2**-31, rounded to fp32
TL_RAND_MIN_F32 = np.float32(4.6566127342e-10)       # vLLM `_TL_RAND_MIN` (clamp before the flipped Gumbel transform)
TL_RAND64_SCALE = np.float64(5.421010862427522170037e-20)
F64_TINY = np.float64(2.2250738585072014e-308)
DRAFT_NOISE_SALT = 1 << 30
MASK32 = 0xFFFFFFFF


def philox4x32_10(c0, c1, c2, c3, k0, k1, rounds: int = 10):
    """Triton `philox_impl` for uint32 state: `rounds` Philox rounds of counter (c0..c3) under key (k0, k1).  Vectorised over
    numpy uint32 arrays (broadcast); identical to Random123's Philox4x32 (same multipliers, Weyl constants and lane order)."""
    c0, c1, c2, c3 = (np.asarray(x, dtype=np.uint32) for x in (c0, c1, c2, c3))
    k0, k1 = np.uint32(k0), np.uint32(k1)
    with np.errstate(over="ignore"):
        for _ in range(rounds):
            _c0, _c2 = c0, c2
            hi_b = ((_c2.astype(np.uint64) * np.uint64(PHILOX_ROUND_B)) >> np.uint64(32)).astype(np.uint32)
            hi_a = ((_c0.astype(np.uint64) * np.uint64(PHILOX_ROUND_A)) >> np.uint64(32)).astype(np.uint32)
            c0 = hi_b ^ c1 ^ k0
            c2 = hi_a ^ c3 ^ k1
            c1 = _c2 * PHILOX_ROUND_B          # low 32 bits (uint32 wraparound), Triton `tl.mul(..., sanitize_overflow=False)`
            c3 = _c0 * PHILOX_ROUND_A
            k0 = np.uint32((int(k0) + int(PHILOX_KEY_A)) & MASK32)
            k1 = np.uint32((int(k1) + int(PHILOX_KEY_B)) & MASK32)
    return c0, c1, c2, c3


def tl_randint4x(seed: int, offsets, *, seed_bits: int = 64):
    """Triton `randint4x(seed, offset)` -> four uint32 blocks.  `seed` is reinterpreted as uint64 (int64 two's complement when
    `seed_bits == 64`; a uint32 scalar such as `gumbel_seed` zero-extends, so its key high word is 0) and split into the Philox
    key (lo, hi); the offsets become the counter (lo word, hi word for >32-bit offsets, 0, 0)."""
    offsets = np.asarray(offsets)
    seed_u = int(seed) & (MASK32 if seed_bits == 32 else 0xFFFFFFFFFFFFFFFF)
    k0, k1 = seed_u & MASK32, (seed_u >> 32) & MASK32
    if offsets.dtype.itemsize > 4:
        off = offsets.astype(np.int64)
        lo = (off & MASK32).astype(np.uint32)
        hi = ((off >> 32) & MASK32).astype(np.uint32)      # arithmetic shift of the int64, then truncation to uint32
    else:
        lo = offsets.astype(np.int64).__and__(MASK32).astype(np.uint32)
        hi = np.zeros_like(lo)
    zero = np.zeros_like(lo)
    return philox4x32_10(lo, hi, zero, zero, k0, k1)


def tl_randint(seed: int, offsets, *, seed_bits: int = 64) -> np.ndarray:
    """Triton `randint`: the first Philox output word (uint32)."""
    return tl_randint4x(seed, offsets, seed_bits=seed_bits)[0]


def uint_to_uniform_float(x: np.ndarray) -> np.ndarray:
    """Triton `uint_to_uniform_float` (32-bit): bitcast to int32, fold negatives (`-x - 1`), scale by 2**-31-ish in fp32."""
    xi = np.asarray(x, dtype=np.uint32).view(np.int32)
    with np.errstate(over="ignore"):
        xi = np.where(xi < 0, -xi - np.int32(1), xi).astype(np.int32)
    return (xi.astype(np.float32) * TL_RAND_SCALE_F32).astype(np.float32)


def tl_rand(seed: int, offsets, *, seed_bits: int = 64) -> np.ndarray:
    """Triton `rand(seed, offset)`: fp32 uniforms in [0, 1)."""
    return uint_to_uniform_float(tl_randint(seed, offsets, seed_bits=seed_bits))


def gumbel_seed_of(seed: int, pos: int, *, is_drafting: bool = False) -> int:
    """`gumbel_seed = tl.randint(seed, pos)` (vLLM `gumbel_noised_argmax`): the per-(request seed, position) stream key."""
    p = int(pos) + (DRAFT_NOISE_SALT if is_drafting else 0)
    return int(tl_randint(int(seed), np.asarray([p], dtype=np.int64))[0])


def gumbel_noise(seed: int, pos: int, vocab_size: int, *, is_drafting: bool = False, use_fp64: bool = False) -> np.ndarray:
    """The noise vector the kernel adds to row `pos` of a request with `seed`: fp32[vocab_size] (or fp64 when USE_FP64).
    `keys` = the vocab index, so the same token draws the same noise wherever it appears (kernel docstring)."""
    gs = gumbel_seed_of(seed, pos, is_drafting=is_drafting)
    keys = np.arange(int(vocab_size), dtype=np.int32)
    if use_fp64:
        lo, hi, _, _ = tl_randint4x(gs, keys, seed_bits=32)
        r = (hi.astype(np.uint64) << np.uint64(32)) | lo.astype(np.uint64)
        u = np.maximum(r.astype(np.float64) * TL_RAND64_SCALE, F64_TINY)
        return (-np.log(-np.log(u))).astype(np.float64)
    u = np.maximum(tl_rand(gs, keys, seed_bits=32), TL_RAND_MIN_F32).astype(np.float32)
    with np.errstate(divide="ignore"):
        return (-np.log(-np.log1p(-u))).astype(np.float32)


def noised_argmax(logits: np.ndarray, noise: np.ndarray, *, temperature: float | None = None) -> dict:
    """`argmax_v(logits[v] (/ T) + noise[v])` in fp32 with first-index tie-break (Triton `tl.max(return_indices=True)` per
    1024-block + torch `argmax` over the block maxima == global first-index argmax).  Returns the token, the runner-up and
    the fp32 margin (in value and in ulps of the winner)."""
    x = np.asarray(logits, dtype=np.float32)
    if temperature is not None and float(temperature) not in (0.0, 1.0):
        x = (x / np.float32(temperature)).astype(np.float32)     # IEEE division here; the kernel's `/` is div.full.f32 (<= 2 ulp)
    s = (x + noise.astype(x.dtype)).astype(x.dtype)
    best = int(np.argmax(s))
    rest = np.delete(s, best)
    second = int(np.argmax(rest))
    second = second + (1 if second >= best else 0)
    margin = float(s[best] - s[second])
    ulp = float(np.spacing(np.abs(s[best]).astype(s.dtype)))
    return {"token": best, "runner_up": second, "margin": margin, "margin_ulps": margin / ulp if ulp > 0 else float("inf"),
            "value": float(s[best])}


def sampled_token(logits_row: np.ndarray, seed: int, pos: int, *, temperature: float, apply_temperature: bool,
                  is_drafting: bool = False, use_fp64: bool = False) -> dict:
    """One captured sampler row, recomputed: returns `noised_argmax(...)` plus the stream key."""
    V = int(np.asarray(logits_row).shape[-1])
    if float(temperature) == 0.0:
        out = noised_argmax(logits_row, np.zeros(V, dtype=np.float32))
    else:
        noise = gumbel_noise(seed, pos, V, is_drafting=is_drafting, use_fp64=use_fp64)
        out = noised_argmax(logits_row, noise, temperature=temperature if apply_temperature else None)
    out["gumbel_seed"] = gumbel_seed_of(seed, pos, is_drafting=is_drafting)
    return out


# --- run check (log + snapshots) ---------------------------------------------------------------------------------------------

def check_run(log_path: Path, snapshots_dir: Path) -> dict:
    """For every snapshotted step with a `_gumbel_sample_kernel` launch: recompute each logits row's token from the captured
    logits (the kernel's `logits_ptr` snapshot), the request seed and the row position, and compare with `SampleEnd`."""
    from verity_vllm.check.fold_compare import SnapshotStore
    from verity_vllm.observe.events import DispatchEnter, SampleEnd, StepStart, TritonLaunch, ViewDesc
    from verity_vllm.observe.log import read_log

    events = list(read_log(log_path))
    header = events[0]
    wl = header.workload
    seeds_by_req: dict[str, int | None] = {}
    for r in wl.get("requests", []):
        seeds_by_req[r["request_id"]] = r.get("seed")
    for c in wl.get("children", []) or []:
        seeds_by_req[c["request_id"]] = c.get("seed")
    temperature = float((wl.get("sampling") or {}).get("temperature", 0.0))
    store = SnapshotStore(snapshots_dir)
    rows: list[dict] = []
    per_step: dict[int, dict] = {}
    cur: StepStart | None = None
    ops_sample: dict[int, list[str]] = {}
    for ev in events[1:]:
        if isinstance(ev, StepStart):
            cur = ev
            per_step[ev.step] = {"launch": None, "sample_end": None}
        elif isinstance(ev, (TritonLaunch, DispatchEnter)) and ev.phase == "sample":
            ops_sample.setdefault(ev.step, []).append(ev.fn if isinstance(ev, TritonLaunch) else ev.op)
            if isinstance(ev, TritonLaunch) and ev.fn.endswith("_gumbel_sample_kernel"):
                per_step[ev.step]["launch"] = (ev, cur)
        elif isinstance(ev, SampleEnd):
            per_step[ev.step]["sample_end"] = ev
    n_ok = n_bad = n_skipped = 0
    for step, d in sorted(per_step.items()):
        if d["launch"] is None or d["sample_end"] is None:
            continue
        launch, ss = d["launch"]
        logits_view = launch.args.get("logits_ptr")
        if not isinstance(logits_view, ViewDesc):
            n_skipped += 1
            continue
        snap = store.find(logits_view, at=launch.id) or store.find(logits_view, before=launch.id + 1)
        if snap is None:
            rows.append({"step": step, "status": "no-snapshot"})
            n_skipped += 1
            continue
        words = store.extract(snap, logits_view)
        logits = _as_f32(words, logits_view.dtype).reshape(logits_view.shape)
        cx = launch.constexprs or {}
        apply_t = bool(cx.get("APPLY_TEMPERATURE", False))
        fp64 = bool(cx.get("USE_FP64", False))
        drafting = bool(cx.get("IS_DRAFTING", False))
        seeds_snap = _small_int_arg(store, launch, "seeds_ptr")
        pos_snap = _small_int_arg(store, launch, "pos_ptr")
        temp_snap = _small_arg(store, launch, "temp_ptr", np.float32)
        map_snap = _small_int_arg(store, launch, "expanded_idx_mapping_ptr")
        for row, (rid, pos) in enumerate(ss.logits_rows):
            seed = seeds_by_req.get(rid)
            src = "workload"
            if seeds_snap is not None and map_snap is not None and row < len(map_snap):
                seed, src = int(seeds_snap[int(map_snap[row])]), "snapshot(seeds_ptr)"
            if pos_snap is not None and row < len(pos_snap):
                pos = int(pos_snap[row])
            T = temperature
            if temp_snap is not None and map_snap is not None and row < len(map_snap):
                T = float(temp_snap[int(map_snap[row])])
            rec = {"step": step, "row": row, "request_id": rid, "pos": int(pos), "seed": seed, "seed_source": src, "temperature": T,
                   "apply_temperature_in_kernel": apply_t, "use_fp64": fp64, "observed": int(d["sample_end"].sampled_token_ids[row])}
            if seed is None and T != 0.0:
                rec.update(status="unseeded", note="seed=None -> vLLM drew np.random.randint on the host; not recomputable from the log")
                n_skipped += 1
            else:
                out = sampled_token(logits[row], int(seed or 0), int(pos), temperature=T, apply_temperature=apply_t,
                                    is_drafting=drafting, use_fp64=fp64)
                rec.update(out)
                rec["status"] = "match" if out["token"] == rec["observed"] else "mismatch"
                n_ok += rec["status"] == "match"
                n_bad += rec["status"] == "mismatch"
            rows.append(rec)
    return {"schema": "verity-gen-sampling/rng-check/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "log": str(log_path), "snapshots": str(snapshots_dir), "temperature": temperature,
            "sample_phase_ops_by_step": {str(k): v for k, v in sorted(ops_sample.items())},
            "summary": {"match": n_ok, "mismatch": n_bad, "skipped": n_skipped}, "rows": rows}


def _as_f32(words: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bfloat16":
        return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if dtype == "float16":
        return words.view(np.float16).astype(np.float32)
    if dtype == "float32":
        return words.view(np.float32)
    if dtype == "float64":
        return words.view(np.float64).astype(np.float32)
    raise ValueError(dtype)


def _small_arg(store, launch, name: str, dtype):
    from verity_vllm.observe.events import ViewDesc
    v = launch.args.get(name)
    if not isinstance(v, ViewDesc):
        return None
    snap = store.find(v, at=launch.id) or store.find(v, before=launch.id + 1)
    if snap is None:
        return None
    w = store.extract(snap, v)
    if dtype is np.float32:
        return _as_f32(w, v.dtype).reshape(-1)
    return w.reshape(-1)


def _small_int_arg(store, launch, name: str):
    from verity_vllm.observe.events import ViewDesc
    v = launch.args.get(name)
    w = _small_arg(store, launch, name, None)
    if w is None or not isinstance(v, ViewDesc):
        return None
    np_dt = {"int64": np.int64, "int32": np.int32, "uint32": np.uint32, "int16": np.int16, "uint8": np.uint8}.get(v.dtype)
    return w.view(np_dt).astype(np.int64) if np_dt is not None else None


@dataclass(frozen=True, kw_only=True)
class Options:
    run: str = option("--run", help="run dir (RUN_DIR/capture/{log.jsonl.gz,snapshots.json,values/})")
    log: str | None = option("--log", default=None)
    snapshots: str | None = option("--snapshots", default=None)
    out: str | None = option("--out", default=None)


def main(a: Options) -> int:
    run = Path(a.run)
    log = Path(a.log) if a.log else run / "capture" / "log.jsonl.gz"
    snaps = Path(a.snapshots) if a.snapshots else log.parent
    rep = check_run(log, snaps)
    out = Path(a.out) if a.out else run / "rng_check.json"
    out.write_text(json.dumps(rep, indent=1, default=str))
    s = rep["summary"]
    print(f"rng_check: match={s['match']} mismatch={s['mismatch']} skipped={s['skipped']} -> {out}")
    for r in rep["rows"]:
        if r.get("status") in ("match", "mismatch"):
            print(f"  step {r['step']} row {r['row']} {r['request_id'][:8]} pos={r['pos']} seed={r['seed']} T={r['temperature']}: "
                  f"observed={r['observed']} recomputed={r['token']} [{r['status']}] margin={r['margin']:.4g} ({r['margin_ulps']:.0f} ulp)")
        else:
            print(f"  step {r['step']} row {r.get('row')}: {r['status']} {r.get('note', '')}")
    return 0 if s["mismatch"] == 0 else 1
