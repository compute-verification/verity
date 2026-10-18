"""A served H1 + fx1 window (`eps1-sm120-h1fx1`) on vLLM's int8-exact row, one process on one GPU, in the static form (Daniel,
20:52Z): the window's tiles, rows and positions are a fixed function of public parameters alone.

The receipts bind extra work beside cuBLASLt's served y, not the served kernel's.  vllm's PoUW option
(``pouw:eps1-sm120-h1fx1``, `verity_vllm.protocol_options.pouw_h1`) hands over each int8-exact linear call's s8 rows and scales
and its layer, and holds every layer's root(B), the weights' root and the epoch's 64-byte salt from the challenger's salt coin
(`pouw_h1.program_salt`).  Its own per-call positions and padding follow the step's real rows, so this window doesn't use
them.  Instead:

* **Public parameters**: the model, the batch slots (vLLM's ``max_num_seqs``), ``T`` rows a step (``max_num_batched_tokens``:
  the scheduler never puts more of a step's tokens in a linear), ``S`` steps and the max tokens.
* **A side**: each (step s, linear w) call's A table holds all T slot rows (`commit`'s a-table: K s8 bytes, then the f32
  scale).  The step's real rows fill slots 0 .. m - 1, and every other slot is an empty row, committed like the rest: zero
  bytes with scale 0.  Since a served row's scale is never 0, "empty" is read from the committed row.  Steps past the last
  served one are all empty.  A row count never comes from the step's tokens, its batch or EOS.
* **Units and positions**: the call's units are its T/32 row blocks by N/32 column blocks, at positions
  s·U + off(w) + u.  Here U = Σ_w (T/32)(N_w/32), off(w) is the sum over the linears before w, and u runs row block by row block.
  The units' count, S·U, is fixed by the public parameters.
* **Receipts**: `h1tile`'s unit on the GPU (`receipts.call_receipts`), on every unit of every call, empty rows included.
  Seeds come from the salt, the position and the call's (root(A), root(B)).  There is one receipt tree per call.
* **Coins** (``--coins``): seeded, the option's one salt, opened at install before serving; or live, a round a step from
  the proof service's challenger (`pouw.window.live_challenger`, `sampled_proofs.challenger.LiveRecord`).  Step s's round
  root, over its A-table roots, is recorded before round s's coin opens, and the step's receipts take the round's salt
  (`window.round_key`), so every candidate a round's noise reads is committed before that noise's coin.
* **Credit**: static (coordinator, 21:38Z): every unit of the S·U is credited as compute, a count fixed by the public
  parameters, so no selection decides credit.  H1's dither makes an empty row's unit real work (X = wrap8(0 + b)).  The
  served rows (the slots whose committed scale is nonzero) are usefulness, reported beside the credit and never part of it.
* **Not covered**: attention (QKᵀ, P·V) and lm_head.

In order, into OUT (`record.json` throughout, `manifest.json` at the end, the option's `pouw-h1-calls.json`):

  timing    `noise.cu` over a call's per-unit noise against `torch._int_mm` of the same call, CUDA events, K = 4096 and 14336
  engine    the int8 pin and `engine.build.build_engine(target=from_device() + int8_exact)`, eager, prefix caching off, T and slots
  wrap      the fraction of weights that W' = B + c + U mod 2^8 wraps, per layer and site (`wrap_counts`)
  reference the prompts generated with no option installed
  install   the option: every layer's root(B), the weights' root, the challenger's salt coin opened after it, the salt
  serve     the prompts again with the option; inside its `record`, each call's A table and receipts; then the empty steps;
            the token ids must be the reference's
  receipts  one receipt tree per call (one leaf per unit, `commit`)
  prover    the prover's data (the table key, each call's real rows, scales and receipts) as SCRATCH/prover.tar, before the
            gate so a gate failure keeps it; `serve.sh` puts it in the store once the GPU is released, else moves it into
            OUT, which the run's custody uploads
  gate      drawn units, real and empty, against `h1tile.reference` (receipt) and the served integer product (z); controls: the
            unit at the next position, a flipped receipt bit, root(A) rebuilt (and a tampered row, real or empty), root(B)
            rebuilt (and a tampered row), the served rows read back from the committed scales

    cd integrations/vllm; PYTHONPATH=.:../..:../../tools/research/src:../../benchmarks/pouw/served_h1
    $PY ../../benchmarks/pouw/served_h1/serve.py --out OUT --cubin noise.cubin --at 2026-10-18T00:00:00Z --by compute-accounting
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import commit as CM  # noqa: E402

SCHEMA = "verity/pouw-h1-pass/v1"
RECORD = "verity/pouw-served-h1/record/v1"
SCHEME = "eps1-sm120-h1fx1"
FRAMING = "the receipts bind extra work beside cuBLASLt's served y, not the served kernel's"
NOT_COVERED = ["attention (QKᵀ, P·V)", "lm_head"]
SITES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def site_of(name: str) -> tuple[int, str]:
    return int(name.split(".layers.")[1].split(".")[0]), name.rsplit(".", 1)[1]


class Record:
    def __init__(self, out: Path):
        self.out, self.doc = out, {"schema": RECORD, "argv": sys.argv, "steps": {}}

    def step(self, name: str, value: Any) -> None:
        self.doc["steps"][name] = value
        self.save()
        print(f"[served-h1] {name}: {json.dumps(value, default=str)[:600]}", flush=True)

    def save(self) -> None:
        (self.out / "record.json").write_text(json.dumps(self.doc, indent=1, default=str) + "\n")


# -- timing -------------------------------------------------------------------------------------------------------------

def timing(cubin, reps: int = 10) -> dict:
    """Per-unit noise on the GPU against the int8 GEMM of the same M x N x K call, both from CUDA events."""
    import torch
    import receipts as RC

    def clock(fn, n: int) -> float:
        fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / n

    out = []
    for m, n, k in ((4096, 4096, 4096), (4096, 4096, 14336)):
        a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
        b = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        gemm = clock(lambda: torch._int_mm(a, b.t()), reps)
        units = (m // 32) * (n // 32)
        lay = RC.layout(k)
        blocks = lay.blocks + RC.LANES
        seeds = torch.randint(-2 ** 62, 2 ** 62, (units, 8), dtype=torch.int64, device="cuda")
        sink = torch.empty(units, blocks, dtype=torch.int32, device="cuda")
        hashed = clock(lambda: cubin.launch(seeds, blocks, sink, sink=True), reps)
        full = torch.empty(units, blocks, 16, dtype=torch.int32, device="cuda")
        written = clock(lambda: cubin.launch(seeds, blocks, full), reps)
        del full, sink
        torch.cuda.empty_cache()
        out.append({"m": m, "n": n, "k": k, "units": units, "blocks_per_unit": blocks, "noise_blocks": lay.blocks,
                    "key_blocks": RC.LANES, "gemm_ms": round(gemm, 4), "noise_hash_ms": round(hashed, 4),
                    "noise_written_ms": round(written, 4), "ratio_hash": round(hashed / gemm, 3),
                    "ratio_written": round(written / gemm, 3), "ns_per_unit_hash": round(1e6 * hashed / units, 2),
                    "gemm_tops": round(2 * m * n * k / gemm / 1e9, 1),
                    "compressions_per_s": round(units * blocks / hashed * 1e3 / 1e9, 2)})
    return {"device": torch.cuda.get_device_name(0), "reps": reps, "shapes": out,
            "what": "noise_hash: SHA-512 of every noise and key block of every 32 x 32 unit, each block's halves XORed to one "
                    "word (no stream written); noise_written: the same, writing the stream receipts.py reads; gemm: "
                    "torch._int_mm (cuBLASLt s8, s32 accumulate) of the call; the seeds' 4 compressions a unit and the fold are "
                    "not in either"}


# -- weights ------------------------------------------------------------------------------------------------------------

def wrap_prob() -> np.ndarray:
    """P(B + c + U leaves [-128, 127]) for each B in [-128, 127]: c a fair bit, U = -64 + (byte mod 129), byte uniform."""
    pu = np.zeros(129)
    for byte in range(256):
        pu[byte % 129] += 1 / 256
    p = np.zeros(256)
    for v in range(-128, 128):
        for c in (0, 1):
            for r in range(129):
                if not -128 <= v + c + r - 64 <= 127:
                    p[v + 128] += 0.5 * pu[r]
    return p


def wrap_counts(mods: dict) -> dict:
    import torch
    p = torch.tensor(wrap_prob(), dtype=torch.float64)
    per_layer: dict[int, dict] = {}
    for name, m in mods.items():
        layer, site = site_of(name)
        h = torch.bincount((m.weight.to(torch.int64) + 128).flatten(), minlength=256).cpu().to(torch.float64)
        tot = float(h.sum())
        big = float(h[:128 - 63].sum() + h[128 + 64:].sum())
        d = per_layer.setdefault(layer, {"weights": 0, "abs_gt_63": 0, "expected_wrapped": 0.0, "sites": {}})
        d["weights"] += int(tot)
        d["abs_gt_63"] += int(big)
        d["expected_wrapped"] += float((h * p).sum())
        d["sites"][site] = {"weights": int(tot), "frac_abs_gt_63": round(big / tot, 6),
                            "frac_expected_wrapped": round(float((h * p).sum()) / tot, 6), "abs_max": int(m.weight.to(torch.int16).abs().max())}
    for d in per_layer.values():
        d["frac_abs_gt_63"] = round(d["abs_gt_63"] / d["weights"], 6)
        d["frac_expected_wrapped"] = round(d["expected_wrapped"] / d["weights"], 6)
    tot = sum(d["weights"] for d in per_layer.values())
    return {"rule": "W' = B + c + U mod 2^8 wraps where B + c + U is outside [-128, 127]: possible only for |B| > 63; "
                    "expected_wrapped weights each B by P(wrap | B) over c (a fair bit) and U (-64 + byte mod 129)",
            "frac_abs_gt_63": round(sum(d["abs_gt_63"] for d in per_layer.values()) / tot, 6),
            "frac_expected_wrapped": round(sum(d["expected_wrapped"] for d in per_layer.values()) / tot, 6),
            "layers": {str(k): v for k, v in sorted(per_layer.items())}}


def padded(t, rows: int):
    """`t` [m, k] with zero rows appended up to `rows`."""
    import torch
    if t.shape[0] == rows:
        return t
    out = torch.zeros(rows, t.shape[1], dtype=t.dtype, device=t.device)
    out[:t.shape[0]] = t
    return out


# -- the run ------------------------------------------------------------------------------------------------------------

def static_layout(ns: list[int], t: int) -> tuple[list[int], int]:
    """off(w) of each linear and U, the units of one step: linear w's call is (t / 32) row blocks by (N_w / 32) column blocks."""
    offs, at = [], 0
    for n in ns:
        if n % 32 or t % 32:
            raise ValueError(f"N = {n} or T = {t} is not a multiple of 32")
        offs.append(at)
        at += (t // 32) * (n // 32)
    return offs, at


def chunk_of(k: int) -> int:
    return max(16, (1 << 22) // k)


def run(a: argparse.Namespace) -> dict:
    import multiprocessing as mp
    import torch

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = Record(out)
    workers = a.workers or max(2, min(48, len(os.sched_getaffinity(0)) - 2))
    pool = mp.get_context("spawn").Pool(workers)
    import receipts as RC

    cubin = RC.Cubin(a.cubin) if a.cubin else None
    torch.backends.cuda.matmul.allow_tf32 = False
    if cubin is not None and not a.skip_timing:
        rec.step("timing", timing(cubin))
    if cubin is not None:
        probe = [os.urandom(64) for _ in range(3)]
        rec.step("noise_kernel_vs_hashlib", {"equal": bool(torch.equal(RC.noise_blocks(probe, 40, "cuda", cubin).cpu(),
                                                                       RC.noise_blocks_cpu(probe, 40)))})

    from tests.engine import int8_exact_gpu as G
    from verity.core.protocols.pouw.coins import Challenger
    from verity_vllm import protocol_options as PO
    from verity_vllm.engine import int8_exact as E
    from verity_vllm.engine import int8_ops
    from verity_vllm.engine.build import build_engine, target_mismatches
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.protocol_options import pouw
    from verity_vllm.protocol_options import pouw_h1 as PH

    t_max, s_max = a.rows, a.steps or a.max_tokens + 2
    target = TargetProfile.from_device().with_overrides(int8_exact=a.int8_exact)
    prompts = list(G.PROMPTS)
    with tempfile.TemporaryDirectory() as scratch:
        entry, how = G.pin(a)
        checkpoints = G.scratch_checkpoints(a.checkpoints, entry, Path(scratch))
        eng = build_engine(checkpoints, case=a.role, target=target, max_num_seqs=len(prompts),
                           engine_args={"gpu_memory_utilization": a.gpu_mem, "max_num_batched_tokens": t_max,
                                        "enable_chunked_prefill": True})
    public = {"model": {k: entry.get(k) for k in ("repo", "revision", "recipe")}, "slots": len(prompts), "rows": t_max,
              "steps": s_max, "max_tokens": a.max_tokens}
    rec.step("engine", {"pin": {k: entry.get(k) for k in ("repo", "revision", "recipe", "local_path")}, "pin_source": how,
                        "build_s": round(eng.build_s, 1), "target_mismatches": target_mismatches(eng),
                        "enforce_eager": eng.kwargs.get("enforce_eager"), "prefix_caching": eng.kwargs.get("enable_prefix_caching"),
                        "int8_exact": a.int8_exact, "public": public, "target": target.to_json(), "effective": eng.effective})

    layers = PH.layers(eng.model, pouw.linear_types())
    ids = {id(m): w for w, m in enumerate(layers)}
    names = {ids[id(m)]: n for n, m in eng.model.named_modules() if id(m) in ids}
    offs, u_step = static_layout([int(m.weight.shape[0]) for m in layers], t_max)
    rec.step("wrap", wrap_counts({names[w]: m for w, m in enumerate(layers)}))

    reference = G.generate(eng.llm, prompts, a.max_tokens)
    rec.step("reference", {"requests": len(reference), "tokens": sum(len(r["token_ids"]) for r in reference)})

    option = f"pouw:{PH.SCHEME}"
    ctx = PO.Context(stage="serve", enabled=(option,), settings={"pouw": {"coins": Challenger(a.epoch), "retain_steps": 0,
                                                                         "transcript_dir": str(out)}})
    t0 = time.perf_counter()
    comp = PO.install(eng.model, ctx)
    h1 = pouw.h1_run(option)
    rec.step("install", {"s": round(time.perf_counter() - t0, 1), "option": option, "salt": h1.salt.hex(),
                         "salt_bytes": len(h1.salt), "weights_root": h1.commits.get("weights_root"),
                         "coins": h1.commits.get("coins"), "salt_opening": h1.commits.get("salt_opening"),
                         "weight_roots": {names[w]: r.hex() for w, r in sorted(h1.weight_roots.items())}, "roots": PH.ROOTS})

    window = os.environ.get("RESEARCH_RUN_ID") or "local"
    key = os.urandom(64)
    dev = layers[0].weight.device
    st = {"on": True, "step": -1, "last": None, "order_ok": True, "refused": []}
    stash: dict[int, tuple] = {}
    calls: list[dict] = []
    data: list[dict] = []
    a_check = {"equal": 0, "mismatch": 0, "untapped": 0}
    orig_linear, orig_record = int8_ops.linear, h1.record

    def sync() -> None:
        if dev.type == "cuda":
            torch.cuda.synchronize()

    live = a.coins == "live"
    pending: list[tuple] = []
    round_log: list[dict] = []
    if live:
        from verity.core.protocols.pouw import window as Wd
        from verity.core.protocols.sampled_proofs import challenger as Ch
        lch = Wd.live_challenger(a.epoch, s_max)
        live_salt = lch.open(Wd.SALT, 0, bytes.fromhex(h1.commits["weights_root"]))
        rounds = Ch.LiveRecord(lch, Wd.ROUND)

    def receipts(i: int, slots, salt: bytes) -> None:
        c = calls[i]
        sync()
        t = time.perf_counter()
        rc = RC.call_receipts(RC.Call(slots, layers[c["linear"]].weight), salt, c["base"], bytes.fromhex(c["root_a"]),
                              bytes.fromhex(c["root_b"]), cubin, chunk=chunk_of(c["k"]))
        sync()
        c.update(units=int(rc.shape[0]), receipt_words=int(rc.shape[1]), receipts_s=round(time.perf_counter() - t, 3),
                 salt=salt.hex())
        data[i]["receipts"] = rc

    def round_root(s: int, roots_a: list[bytes]) -> tuple[dict, bytes]:
        fields = {"window": window, "step": s, "linears": len(roots_a)}
        return fields, CM.levels(CM.domain(fields, len(roots_a), "round"), roots_a)[-1][0]

    def close_round(s: int) -> None:
        """Round s (step s) under live coins: its root over the step's A-table roots, recorded before its coin opens; then
        the step's receipts under the round's 64-byte salt, `round_key`'s stream."""
        fields, root = round_root(s, [bytes.fromhex(calls[i]["root_a"]) for i, _ in pending])
        opening = rounds.record(root)
        salt = Wd.round_key(lch.commitment, opening.coin, s, root).stream(0, 64)
        round_log.append({"round": s, "root": root.hex(), "fields": fields, "opening": opening.to_json()})
        for i, slots in pending:
            receipts(i, slots, salt)
        pending.clear()

    def table(s: int, w: int, xq, sa: np.ndarray) -> None:
        """Call (s, w): its T-slot A table from the step's m real rows `xq` [m, K] (on the device) and scales; its receipts
        now (seeded) or once the step's round closes (live)."""
        m, k = int(xq.shape[0]), int(layers[w].weight.shape[1])
        n = int(layers[w].weight.shape[0])
        name = names[w]
        layer, site = site_of(name)
        slots = torch.zeros(t_max, k, dtype=torch.int8, device=dev)
        slots[:m] = xq
        scales = np.zeros(t_max, dtype=np.float32)
        scales[:m] = sa
        fields = {"window": window, "step": s, "linear": w, "module": name, "slots": t_max, "k": k, "schema": CM.SCHEMA}
        dom = CM.domain(fields, t_max, "a-table")
        rows = CM.s8_rows(slots.cpu().numpy(), scales)
        lv = CM.levels(dom, [CM.leaf(dom, key, f"a/{s}/{w}", r, bytes(row))[0] for r, row in enumerate(rows)])
        calls.append({"step": s, "linear": w, "module": name, "layer": layer, "site": site, "k": k, "n": n, "slots": t_max,
                      "base": s * u_step + offs[w], "units": (t_max // 32) * (n // 32), "root_a": lv[-1][0].hex(),
                      "root_b": h1.weight_roots[w].hex(), "a_domain": dom.domain_id.hex(), "a_fields": fields})
        data.append({"m": m, "xq": slots[:m].cpu().numpy(), "sa": scales[:m].copy(), "receipts": None})
        if not live:
            receipts(len(calls) - 1, slots, h1.salt)
            return
        pending.append((len(calls) - 1, slots))
        if w == len(layers) - 1:
            close_round(s)

    def tap(x2, w, s_w):
        xq, s_a, y = orig_linear(x2, w, s_w)
        if st["on"]:
            stash[w.data_ptr()] = (xq, s_a)
        return xq, s_a, y

    def record(w: int, xq, sa, n: int, y=None):
        call = orig_record(w, xq, sa, n, y)
        served = stash.pop(layers[w].weight.data_ptr(), None)
        sa = np.asarray(sa, dtype=np.float32).reshape(-1)
        if served is None:
            a_check["untapped"] += 1
            rows = torch.from_numpy(np.asarray(xq)).to(dev)
        else:
            same = (np.array_equal(served[0].cpu().numpy(), xq)
                    and np.array_equal(served[1].reshape(-1).cpu().numpy(), sa))
            a_check["equal" if same else "mismatch"] += 1
            rows = served[0]
        if w == 0:
            st["step"] += 1
        elif st["last"] is None or w != st["last"] + 1:
            st["order_ok"] = False
        st["last"] = w
        if st["step"] >= s_max or call.m > t_max:
            st["refused"].append({"step": st["step"], "linear": w, "m": call.m})
            return call
        table(st["step"], w, rows, sa)
        return call

    int8_ops.linear, h1.record = tap, record
    t0 = time.perf_counter()
    try:
        served = G.generate(eng.llm, prompts, a.max_tokens)
    finally:
        int8_ops.linear = orig_linear
        st["on"] = False
    serve_s = time.perf_counter() - t0
    done = st["step"] + 1
    complete = st["last"] == len(layers) - 1
    t0 = time.perf_counter()
    if pending:
        close_round(calls[pending[0][0]]["step"])
    for s in range(done, s_max):
        for w in range(len(layers)):
            table(s, w, torch.zeros(0, int(layers[w].weight.shape[1]), dtype=torch.int8, device=dev), np.zeros(0, np.float32))
    static_ok = ([(c["step"], c["linear"], c["base"], c["units"]) for c in calls]
                 == [(s, w, s * u_step + offs[w], (t_max // 32) * (int(layers[w].weight.shape[0]) // 32))
                     for s in range(s_max) for w in range(len(layers))])
    credited = {"units": sum(c["units"] for c in calls), "rows": len(calls) * t_max,
                "macs": sum(t_max * c["n"] * c["k"] for c in calls)}
    useful = {"rows": sum(d["m"] for d in data), "macs": sum(d["m"] * c["n"] * c["k"] for c, d in zip(calls, data))}
    useful["frac_of_credited_macs"] = round(useful["macs"] / max(1, credited["macs"]), 6)
    rec.step("serve", {"steps_served": done, "steps": s_max, "last_step_complete": complete, "order_ok": st["order_ok"],
                       "refused": st["refused"], "calls": len(calls), "units": sum(c["units"] for c in calls),
                       "units_static": s_max * u_step, "static": static_ok, "u_step": u_step, "s": round(serve_s, 1),
                       "empty_steps_s": round(time.perf_counter() - t0, 1), "a_served_vs_recorded": a_check,
                       "receipts_s": round(sum(c["receipts_s"] for c in calls), 1),
                       "real_rows_per_step": [max((d["m"] for c, d in zip(calls, data) if c["step"] == s), default=0)
                                              for s in range(s_max)], "credited": credited, "useful": useful})
    equal = [s_["token_ids"] == r["token_ids"] for s_, r in zip(served, reference)]
    rec.step("tokens", {"equal": all(equal), "per_prompt": equal, "served": served, "reference": reference})

    t0 = time.perf_counter()
    rtasks = []
    for c, d in zip(calls, data):
        fields = {"window": window, "step": c["step"], "linear": c["linear"], "module": c["module"], "base": c["base"],
                  "units": c["units"], "words": c["receipt_words"], "schema": CM.SCHEMA}
        dom = CM.domain(fields, c["units"])
        c.update(receipt_domain=dom.domain_id.hex(), receipt_fields=fields)
        rtasks.append((dom, CM.receipt_rows(d["receipts"])))
    rl = pool.map(CM.leaves, [(dom.binding, dom.count, key, f"r/{c['step']}/{c['linear']}", 0, rows)
                              for c, (dom, rows) in zip(calls, rtasks)], chunksize=1)
    levels = []
    for c, (dom, _), leaves in zip(calls, rtasks, rl):
        lv = CM.levels(dom, CM.split(leaves))
        levels.append(lv)
        c["receipt_root"] = lv[-1][0].hex()
    del rtasks, rl
    rec.step("receipt_trees", {"calls": len(calls), "leaves": sum(c["units"] for c in calls),
                               "s": round(time.perf_counter() - t0, 1)})
    comp.close()

    coins = {"mode": a.coins, "ok": True}
    if live:
        roots = [bytes.fromhex(r["root"]) for r in round_log]
        openings = [Ch.LiveOpening.from_json(r["opening"]) for r in round_log]
        commitment = Ch.LiveCommitment.from_json(lch.commitment.to_json())
        by_step: dict[int, list[dict]] = {}
        for c in calls:
            by_step.setdefault(c["step"], []).append(c)
        recomputed = [round_root(r["round"], [bytes.fromhex(c["root_a"]) for c in by_step[r["round"]]])[1] == root
                      and {c["salt"] for c in by_step[r["round"]]}
                      == {Wd.round_key(commitment, o.coin, r["round"], root).stream(0, 64).hex()}
                      for r, root, o in zip(round_log, roots, openings)]
        coins.update(scheme=Wd.LIVE_SCHEME, commitment=lch.commitment.to_json(), salt_coin=live_salt.to_json(),
                     rounds=round_log, rounds_reason=Wd.rounds_reason(commitment, roots, openings),
                     rounds_cover_steps=[r["round"] for r in round_log] == list(range(s_max)),
                     roots_and_keys_recomputed=all(recomputed),
                     rule="round s = step s: its root is the frame-v3-sha512 tree over the step's A-table roots in linear "
                          "order (domain identity_digest_sha512('verity/pouw/served-h1/round/v0', {window, step, linears})), "
                          "recorded through sampled_proofs.challenger.LiveRecord before round s's pouw/round coin opens; "
                          "the step's 64-byte H1 salt is window.round_key(commitment, coin_s, s, root_s).stream(0, 64)")
        coins["ok"] = bool(coins["rounds_reason"] is None and coins["rounds_cover_steps"] and all(recomputed))
        rec.step("coins", {k: v for k, v in coins.items() if k != "rounds"})

    manifest = {"schema": SCHEMA, "scheme": SCHEME, "tile": [RC.R, RC.C_], "framing": FRAMING, "not_covered": NOT_COVERED,
                "window": window, "public": public, "epoch": a.epoch, "option": option, "commitment": h1.commits,
                "coins": coins, "salt": h1.salt.hex(),
                "salt_rule": ("live: each call's salt is its round's (coins.rule); manifest.salt is the option's, unused"
                              if live else
                              "pouw_h1.program_salt: SHA-512(b'verity/pouw/h1/program-salt/v1\\0' || the epoch's 32-byte "
                              "salt), the epoch's salt audit.Epoch.start's from the challenger's salt coin opened after the "
                              "weights' root (pouw.weights_root over every layer's pouw_h1.weight_root)"),
                "roots": {"a": "the call's a-table root (this window's, T slot rows; not yet the Commit's Int8RowQuant_v1 tap)",
                          "b": "the option's pouw_h1.weight_root of the layer (a stand-in, pouw_h1.ROOTS)"},
                "positions": "call (s, w) at base s U + off(w), U = sum_w (T / 32)(N_w / 32), off(w) the sum over linears "
                             "before w; unit u = slot rows 32 (u // (N_w / 32)) .., weight rows 32 (u % (N_w / 32)) ..",
                "u_step": u_step, "offsets": offs, "units": s_max * u_step,
                "leaves": {"a": "slot row i: K s8 bytes then its f32 scale LE; an empty slot zero bytes and scale 0",
                           "r": "unit u: its receipt's 2 (KS + m) u32 words LE (per set y then z)",
                           "schema": CM.SCHEMA, "tree": "frame-v3-sha512, one per call and kind",
                           "domain": "identity_digest_sha512('verity/pouw/served-h1/<a-table|receipts>/v0', fields), owner "
                                     "-1, ranks 0 .. count - 1",
                           "salts": "SHA-512(SALT_TAG || key || u32be len(label) || label || u64be i || u8 c), c = 0..2, "
                                    "labels a/<s>/<w> and r/<s>/<w>; the key is the prover's"},
                "credited": {**credited, "rule": "static: every unit of the S U, a count fixed by the public parameters, "
                                                 "credited as compute; no selection decides credit"},
                "useful": {**useful, "rule": "the slots whose committed scale is nonzero (a served row's scale is never 0); "
                                             "reported beside the credit, never part of it"},
                "calls": calls, "tokens_equal": all(equal)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    rec.step("prover", prover(a, out, key, calls, data))
    try:
        rec.step("gate", gate(a, calls, data, levels, layers, h1, key, E, PH, t_max, done))
    except Exception as e:  # noqa: BLE001
        rec.step("gate", {"ok": False, "error": f"{type(e).__name__}: {e}"[:4000]})
    s = rec.doc["steps"]["serve"]
    rec.doc["ok"] = bool(all(equal) and calls and not a_check["mismatch"] and not a_check["untapped"] and not s["refused"]
                         and s["order_ok"] and s["last_step_complete"] and s["static"] and rec.doc["steps"]["gate"]["ok"]
                         and coins["ok"])
    rec.save()
    pool.close()
    print(f"SERVED-H1 {'OK' if rec.doc['ok'] else 'FAIL'} calls={len(calls)} units={s['units']} steps={done}/{s_max} "
          f"tokens_equal={all(equal)} static={s['static']} gate={rec.doc['steps']['gate']['ok']}", flush=True)
    return rec.doc


def gate(a, calls: list[dict], data: list[dict], levels: list, layers: list, h1, key: bytes, E, PH, t_max: int,
         done: int) -> dict:
    """Drawn units against `h1tile.reference`, and the controls: per site real steps' calls, then calls of empty steps."""
    import torch
    import receipts as RC

    rng = random.Random(os.urandom(16))
    picks = []
    for site in SITES:
        real = [i for i, c in enumerate(calls) if c["site"] == site and c["step"] < done]
        for _ in range(a.gate_per_site if site != "down_proj" else max(1, a.gate_per_site // 2)):
            if real:
                picks.append(rng.choice(real))
    empty = [i for i, c in enumerate(calls) if c["step"] >= done]
    picks += rng.sample(empty, min(2, len(empty)))
    units, ok, b_roots = [], True, {}
    for i in picks:
        c, d = calls[i], data[i]
        u = rng.randrange(c["units"])
        weight = layers[c["linear"]].weight
        slots = padded(torch.from_numpy(d["xq"]), t_max)
        scales = np.zeros(t_max, dtype=np.float32)
        scales[:d["m"]] = d["sa"]
        call = RC.Call(slots, weight.cpu())
        root_a, root_b = bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"])
        salt = bytes.fromhex(c["salt"])
        z, want = RC.reference(call, salt, c["base"], root_a, root_b, u)
        got = [int(v) for v in d["receipts"][u]]
        ri, cj = call.units[u]
        y = E.int_mm(slots[32 * ri:32 * ri + 32].to(weight.device), weight[32 * cj:32 * cj + 32]).cpu().to(torch.int64)
        z_ok = [v & 0xFFFFFFFF for v in y.flatten().tolist()] == z
        nxt = RC.reference(call, salt, c["base"] + 1, root_a, root_b, u)[1]
        dom = CM.domain(c["receipt_fields"], c["units"])
        root_r = bytes.fromhex(c["receipt_root"])
        rows = CM.receipt_rows(d["receipts"][u:u + 1])[0]
        label = f"r/{c['step']}/{c['linear']}"
        leaf = CM.leaf(dom, key, label, u, bytes(rows))[0]
        sibs = CM.path(levels[i], u)
        flipped = bytearray(rows)
        flipped[rng.randrange(len(flipped))] ^= 1 << rng.randrange(8)
        leaf_f = CM.leaf(dom, key, label, u, bytes(flipped))[0]
        a_dom = CM.domain(c["a_fields"], t_max, "a-table")
        a_rows = CM.s8_rows(slots.numpy(), scales)
        a_label = f"a/{c['step']}/{c['linear']}"
        a_root = CM.levels(a_dom, [CM.leaf(a_dom, key, a_label, r, bytes(x))[0] for r, x in enumerate(a_rows)])[-1][0]
        row = 32 * ri + rng.randrange(32)
        a_rows[row, rng.randrange(a_rows.shape[1])] ^= 1
        a_tampered = CM.levels(a_dom, [CM.leaf(a_dom, key, a_label, r, bytes(x))[0] for r, x in enumerate(a_rows)])[-1][0]
        committed = CM.s8_rows(slots.numpy(), scales)
        useful = sum(int(np.frombuffer(bytes(x[-4:]), "<f4")[0] != 0) for x in committed)
        if c["linear"] not in b_roots:
            b_rows = weight.cpu().numpy().copy()
            rebuilt = PH.weight_root(c["linear"], b_rows)
            b_rows[rng.randrange(c["n"]), rng.randrange(c["k"])] ^= 1
            b_roots[c["linear"]] = (rebuilt, PH.weight_root(c["linear"], b_rows))
        r = {"step": c["step"], "linear": c["linear"], "module": c["module"], "k": c["k"], "unit": u,
             "position": c["base"] + u, "real_rows_in_tile": max(0, min(32, d["m"] - 32 * ri)),
             "receipt_equal_reference": got == want, "z_equal_served_product": z_ok, "next_position_differs": nxt != got,
             "receipt_opens": CM.verify(dom, root_r, u, leaf, sibs),
             "flipped_receipt_refused": not CM.verify(dom, root_r, u, leaf_f, sibs),
             "a_root_rebuilt": a_root == root_a, "tampered_a_row_changes_root": a_tampered != root_a,
             "tampered_row_was_empty": row >= d["m"], "served_rows_from_scales": useful == d["m"],
             "b_root_rebuilt": b_roots[c["linear"]][0] == root_b,
             "tampered_b_row_changes_root": b_roots[c["linear"]][1] != root_b}
        r["ok"] = all(v for k_, v in r.items() if isinstance(v, bool) and k_ != "tampered_row_was_empty")
        ok &= r["ok"]
        units.append(r)
        print(f"[served-h1] gate {json.dumps(r)}", flush=True)
    return {"ok": bool(ok and units), "units": units}


def prover(a, out: Path, key: bytes, calls: list[dict], data: list[dict]) -> dict:
    """The prover's data: the table key, each call's real rows (the rest of its slots are empty), scales and receipts, as
    SCRATCH/prover.tar for `serve.sh` to put in the store after the lease (OUT/prover with no --scratch)."""
    dest = Path(a.scratch) if a.scratch else out / "prover"
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest) as td:
        p = Path(td) / "prover"
        p.mkdir()
        (p / "prover.json").write_text(json.dumps({"key": key.hex(), "calls": [[c["step"], c["linear"], d["m"]]
                                                                              for c, d in zip(calls, data)]}) + "\n")
        for c, d in zip(calls, data):
            np.savez(p / f"s{c['step']:03d}w{c['linear']:03d}.npz", xq=d["xq"], sa=d["sa"], receipts=d["receipts"])
        tar = dest / "prover.tar"
        with tarfile.open(tar, "w") as t:
            t.add(p, arcname="prover")
    h = hashlib.sha256()
    with tar.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return {"bytes": tar.stat().st_size, "sha256": h.hexdigest(), "tar": str(tar),
            "put": "serve.sh, after the lease: OUT/prover-put.txt, else OUT/prover.tar"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cubin", help="noise.cu built for this device (else hashlib, and no timing)")
    ap.add_argument("--scratch", default=None, help="where the prover's tar is built")
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--base-role", default="LLAMA31_8B")
    ap.add_argument("--role", default="LLAMA31_8B_INT8")
    ap.add_argument("--repo", default="verity-int8/Llama-3.1-8B-INT8")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--at", required=True, help="resolved_at of a new int8 pin entry (UTC, ISO 8601)")
    ap.add_argument("--by", required=True, help="resolved_by of a new int8 pin entry")
    ap.add_argument("--int8-exact", default="all", choices=["linears", "attention", "all"])
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--rows", type=int, default=64, help="T, the public rows a step: vLLM's max_num_batched_tokens")
    ap.add_argument("--steps", type=int, default=0, help="S, the public steps (default max tokens + 2)")
    ap.add_argument("--gpu-mem", type=float, default=0.6, help="vLLM's gpu_memory_utilization: the rest is the side computation's")
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--gate-per-site", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--coins", default="seeded", choices=["seeded", "live"],
                    help="live: a round a step, its coin opened after its A tables' root (the proof service's challenger)")
    return 0 if run(ap.parse_args(argv))["ok"] else 1


if __name__ == "__main__":
    try:
        rc = main()
    except BaseException:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        rc = 2
    sys.stdout.flush()
    sys.stderr.flush()
    # the in-process engine and the spawn pool can keep an exiting interpreter alive past the lease
    os._exit(rc)
