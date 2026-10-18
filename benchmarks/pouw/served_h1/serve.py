"""A served H1 + fx1 window (`eps1-sm120-h1fx1`) on vLLM's int8-exact row, one process on one GPU, through vllm's PoUW option
(``pouw:eps1-sm120-h1fx1``, `verity_vllm.protocol_options.pouw_h1`).

The receipts bind extra work beside cuBLASLt's served y, not the served kernel's.  The option records every int8-exact
linear call as a `pouw_h1.Call`: its positions (a running base, the call's units row block by row block over its grid padded
to 32 x 32 tiles, compile's unit order), its seed group's roots (root(A) over the call's s8 rows, root(B) its layer's weight
root; vllm's stand-ins, `pouw_h1.ROOTS`) and the epoch's 64-byte salt (`pouw_h1.program_salt`).  On each prefill call
(M >= 32) this script computes `h1tile`'s unit on the GPU (`receipts.call_receipts`) for every unit of that grid, on the served
A (the `verity_int8::row_quant` output the linear's `int8_ops.linear` returned, held equal to the rows the option recorded) and
the served weight B; the last row block's pad rows are zero, as the Program's are.  Decode calls (M < 32), attention and
lm_head are not covered.

In order, into OUT (`record.json` throughout, `manifest.json` at the end, the option's `pouw-h1-calls.json`):

  timing    `noise.cu` over a call's per-unit noise (seed_j's blocks and σ_R's key blocks) against `torch._int_mm` of the same
            call, CUDA events (`timing`), at K = 4096 and 14336
  engine    the int8 pin and `engine.build.build_engine(target=from_device() + int8_exact)`, eager, prefix caching off
  wrap      the fraction of weights that W' = B + c + U mod 2^8 wraps, per layer and site (`wrap_counts`)
  reference the prompts generated with no option installed
  install   the option: every layer's root(B), the weights' root, the challenger's salt coin opened after it, the salt
  serve     the prompts again with the option, and the receipts computed inside its `record` for every prefill call; the
            token ids must be the reference's
  receipts  one receipt tree per covered call (one hm96-sha512/row-seg/v1 leaf per unit, `commit`)
  prover    the prover's data (the option's A salt key, the receipt key, each covered call's A rows, scales and receipts) into
            the store (`research data put`, else OUT/prover), before the gate so a gate failure keeps it
  gate      drawn units of each site against `h1tile.reference` (receipt) and the served integer product (z); controls: the
            unit at the next position, a receipt with one bit flipped, root(A) and root(B) rebuilt, a tampered A row and B row

    cd integrations/vllm; PYTHONPATH=.:../..:../../tools/research/src:../../benchmarks/pouw/served_h1
    $PY ../../benchmarks/pouw/served_h1/serve.py --out OUT --cubin noise.cubin --at 2026-10-18T00:00:00Z --by compute-accounting
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
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
NOT_COVERED = ["decode calls (M < 32)", "attention (QKᵀ, P·V)", "lm_head"]
LONG = ("In the early morning light, the old lighthouse keeper climbed the spiral staircase one last time, counting each of "
        "the two hundred and twelve steps as he had done every day for forty years, and when he reached the lamp room he "
        "paused to look out over the grey water toward the harbor where")
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

    target = TargetProfile.from_device().with_overrides(int8_exact=a.int8_exact)
    prompts = G.PROMPTS + [LONG]
    with tempfile.TemporaryDirectory() as scratch:
        entry, how = G.pin(a)
        checkpoints = G.scratch_checkpoints(a.checkpoints, entry, Path(scratch))
        eng = build_engine(checkpoints, case=a.role, target=target, max_num_seqs=len(prompts),
                           engine_args={"gpu_memory_utilization": a.gpu_mem})
    rec.step("engine", {"pin": {k: entry.get(k) for k in ("repo", "revision", "recipe", "local_path")}, "pin_source": how,
                        "build_s": round(eng.build_s, 1), "target_mismatches": target_mismatches(eng),
                        "enforce_eager": eng.kwargs.get("enforce_eager"), "prefix_caching": eng.kwargs.get("enable_prefix_caching"),
                        "int8_exact": a.int8_exact, "target": target.to_json(), "effective": eng.effective})

    layers = PH.layers(eng.model, pouw.linear_types())
    ids = {id(m): w for w, m in enumerate(layers)}
    names = {ids[id(m)]: n for n, m in eng.model.named_modules() if id(m) in ids}
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
    st = {"on": True}
    stash: dict[int, tuple] = {}
    calls: list[dict] = []
    data: list[dict] = []
    decode = {"calls": 0, "rows": 0, "units_positioned": 0}
    a_check = {"equal": 0, "mismatch": 0, "untapped": 0}
    orig_linear, orig_record = int8_ops.linear, h1.record

    def tap(x2, w, s_w):
        xq, s_a, y = orig_linear(x2, w, s_w)
        if st["on"]:
            stash[w.data_ptr()] = (xq, s_a)
        return xq, s_a, y

    def record(w: int, xq, sa, n: int, y=None):
        call = orig_record(w, xq, sa, n, y)
        lin = layers[w]
        served = stash.pop(lin.weight.data_ptr(), None)
        if served is None:
            a_check["untapped"] += 1
            return call
        same = (np.array_equal(served[0].cpu().numpy(), xq)
                and np.array_equal(served[1].reshape(-1).cpu().numpy(), np.asarray(sa).reshape(-1)))
        a_check["equal" if same else "mismatch"] += 1
        if call.m < RC.R:
            decode["calls"] += 1
            decode["rows"] += call.m
            decode["units_positioned"] += call.units
            return call
        name = names[w]
        layer, site = site_of(name)
        mp_, np_ = PH.padded(call.m, call.n)
        torch.cuda.synchronize()
        t = time.perf_counter()
        rc = RC.call_receipts(RC.Call(padded(served[0], mp_), padded(lin.weight, np_)), h1.salt, call.base, *call.roots, cubin)
        torch.cuda.synchronize()
        calls.append({"index": call.index, "module": name, "layer": layer, "site": site, "linear": w, "m": call.m,
                      "k": call.k, "n": call.n, "padded": [mp_, np_], "base": call.base, "units": int(rc.shape[0]),
                      "units_match_option": int(rc.shape[0]) == call.units, "root_a": call.roots[0].hex(),
                      "root_b": call.roots[1].hex(), "receipt_words": int(rc.shape[1]),
                      "receipts_s": round(time.perf_counter() - t, 3), "a_served_equal": same,
                      "x_at_127": int((served[0] == 127).sum())})
        data.append({"xq": np.asarray(xq), "sa": np.asarray(sa).reshape(-1), "receipts": rc})
        return call

    int8_ops.linear, h1.record = tap, record
    t0 = time.perf_counter()
    try:
        served = G.generate(eng.llm, prompts, a.max_tokens)
    finally:
        int8_ops.linear = orig_linear
        st["on"] = False
    serve_s = time.perf_counter() - t0
    rec.step("serve", {"calls_recorded": len(h1.calls), "units_positioned": h1.next_base, "calls_covered": len(calls),
                       "units_covered": sum(c["units"] for c in calls), "s": round(serve_s, 1), "decode_uncovered": decode,
                       "a_served_vs_recorded": a_check, "receipts_s": round(sum(c["receipts_s"] for c in calls), 1),
                       "prefill_m": sorted({c["m"] for c in calls}),
                       "units_match_option": all(c["units_match_option"] for c in calls),
                       "consecutive": [c.base for c in h1.calls] == [0, *np.cumsum([c.units for c in h1.calls])[:-1].tolist()]})
    equal = [s["token_ids"] == r["token_ids"] for s, r in zip(served, reference)]
    rec.step("tokens", {"equal": all(equal), "per_prompt": equal, "served": served, "reference": reference})

    t0 = time.perf_counter()
    rtasks = []
    for c, d in zip(calls, data):
        fields = {"window": window, "call": c["index"], "module": c["module"], "base": c["base"], "units": c["units"],
                  "words": c["receipt_words"], "schema": CM.SCHEMA}
        dom = CM.domain(fields, c["units"])
        c.update(receipt_domain=dom.domain_id.hex(), receipt_fields=fields, receipt_schema=CM.SCHEMA)
        rtasks.append((dom, CM.receipt_rows(d["receipts"])))
    rl = pool.map(CM.leaves, [(dom.binding, dom.count, key, f"r/{c['index']}", 0, rows)
                              for c, (dom, rows) in zip(calls, rtasks)], chunksize=1)
    levels = []
    for c, (dom, _), leaves in zip(calls, rtasks, rl):
        lv = CM.levels(dom, CM.split(leaves))
        levels.append(lv)
        c["receipt_root"] = lv[-1][0].hex()
    rec.step("receipt_trees", {"calls": len(calls), "leaves": sum(c["units"] for c in calls),
                               "s": round(time.perf_counter() - t0, 1)})

    comp.close()

    manifest = {"schema": SCHEMA, "scheme": SCHEME, "tile": [RC.R, RC.C_], "framing": FRAMING, "not_covered": NOT_COVERED,
                "window": window, "model": rec.doc["steps"]["engine"]["pin"], "epoch": a.epoch, "option": option,
                "commitment": h1.commits, "calls_file": PH.CALLS_FILE, "calls_record": PH.RECORD,
                "salt": h1.salt.hex(),
                "salt_rule": "pouw_h1.program_salt: SHA-512(b'verity/pouw/h1/program-salt/v1\\0' || the epoch's 32-byte salt), "
                             "the epoch's salt audit.Epoch.start's from the challenger's salt coin opened after the weights' "
                             "root (pouw.weights_root over every layer's pouw_h1.weight_root)",
                "roots": PH.ROOTS,
                "positions": "the option's: call c's units at base_c + u, u row block by row block over its grid padded to "
                             "32 x 32 tiles (compile's unit order; unit u = rows 32 (u // (N' / 32)) .., weight rows "
                             "32 (u % (N' / 32)) .., N' the padded N)",
                "covered": "every unit of each call with M >= 32 (the last row block's pad rows zero, as the Program's)",
                "leaves": {"r": "unit u: its receipt's 2 (KS + m) u32 words LE (per set y then z), hm96-sha512/row-seg/v1, one "
                                "frame-v3-sha512 tree a covered call",
                           "domain": "identity_digest_sha512('verity/pouw/served-h1/receipts/v0', receipt_fields), owner -1, "
                                     "ranks 0 .. units - 1",
                           "salts": "SHA-512(SALT_TAG || key || u32be len(label) || label || u64be u || u8 c), c = 0..2, "
                                    "label r/<call index>; the key is the prover's"},
                "calls": calls, "units_covered": sum(c["units"] for c in calls), "units_positioned": h1.next_base,
                "tokens_equal": all(equal)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    rec.step("prover", prover(a, out, key, h1, calls, data))
    try:
        rec.step("gate", gate(a, calls, data, levels, layers, h1, key, E, PH))
    except Exception as e:  # noqa: BLE001
        rec.step("gate", {"ok": False, "error": f"{type(e).__name__}: {e}"[:4000]})
    s = rec.doc["steps"]["serve"]
    rec.doc["ok"] = bool(all(equal) and calls and not a_check["mismatch"] and not a_check["untapped"]
                         and s["units_match_option"] and s["consecutive"] and rec.doc["steps"]["gate"]["ok"])
    rec.save()
    pool.close()
    print(f"SERVED-H1 {'OK' if rec.doc['ok'] else 'FAIL'} calls={len(calls)} units={manifest['units_covered']} "
          f"tokens_equal={all(equal)} gate={rec.doc['steps']['gate']['ok']}", flush=True)
    return rec.doc


def gate(a, calls: list[dict], data: list[dict], levels: list, layers: list, h1, key: bytes, E, PH) -> dict:
    """Drawn units against `h1tile.reference`, and the controls."""
    import torch
    import receipts as RC

    rng = random.Random(os.urandom(16))
    picks = []
    for site in SITES:
        cs = [c for c in calls if c["site"] == site]
        for _ in range(a.gate_per_site if site != "down_proj" else max(1, a.gate_per_site // 2)):
            if cs:
                c = rng.choice(cs)
                picks.append((calls.index(c), rng.randrange(c["units"])))
    units, ok, b_roots = [], True, {}
    for i, u in picks:
        c, d = calls[i], data[i]
        weight = layers[c["linear"]].weight
        mp_, np_ = c["padded"]
        xq = padded(torch.from_numpy(d["xq"]), mp_)
        call = RC.Call(xq, padded(weight.cpu(), np_))
        root_a, root_b = bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"])
        z, want = RC.reference(call, h1.salt, c["base"], root_a, root_b, u)
        got = [int(v) for v in d["receipts"][u]]
        ri, cj = call.units[u]
        y = E.int_mm(xq[32 * ri:32 * ri + 32].cuda(), padded(weight, np_)[32 * cj:32 * cj + 32]).cpu().to(torch.int64)
        z_ok = [v & 0xFFFFFFFF for v in y.flatten().tolist()] == z
        nxt = RC.reference(call, h1.salt, c["base"] + 1, root_a, root_b, u)[1]
        dom = CM.domain(c["receipt_fields"], c["units"])
        root_r = bytes.fromhex(c["receipt_root"])
        rows = CM.receipt_rows(d["receipts"][u:u + 1])[0]
        leaf = CM.leaf(dom, key, f"r/{c['index']}", u, bytes(rows))[0]
        sibs = CM.path(levels[i], u)
        flipped = bytearray(rows)
        flipped[rng.randrange(len(flipped))] ^= 1 << rng.randrange(8)
        leaf_f = CM.leaf(dom, key, f"r/{c['index']}", u, bytes(flipped))[0]
        a_rows = d["xq"].copy()
        a_rebuilt = PH.activation_root(h1.salt, c["index"], a_rows, h1.salt_key)
        a_rows[min(32 * ri, c["m"] - 1), rng.randrange(c["k"])] ^= 1
        a_tampered = PH.activation_root(h1.salt, c["index"], a_rows, h1.salt_key)
        if c["linear"] not in b_roots:
            b_rows = weight.cpu().numpy()
            rebuilt = PH.weight_root(c["linear"], b_rows)
            b_rows[rng.randrange(c["n"]), rng.randrange(c["k"])] ^= 1
            b_roots[c["linear"]] = (rebuilt, PH.weight_root(c["linear"], b_rows))
        r = {"call": c["index"], "module": c["module"], "k": c["k"], "unit": u, "position": c["base"] + u,
             "receipt_equal_reference": got == want, "z_equal_served_product": z_ok,
             "next_position_differs": nxt != got, "receipt_opens": CM.verify(dom, root_r, u, leaf, sibs),
             "flipped_receipt_refused": not CM.verify(dom, root_r, u, leaf_f, sibs),
             "a_root_rebuilt": a_rebuilt == root_a, "tampered_a_row_changes_root": a_tampered != root_a,
             "b_root_rebuilt": b_roots[c["linear"]][0] == root_b,
             "tampered_b_row_changes_root": b_roots[c["linear"]][1] != root_b}
        r["ok"] = all(v for k_, v in r.items() if isinstance(v, bool))
        ok &= r["ok"]
        units.append(r)
        print(f"[served-h1] gate {json.dumps(r)}", flush=True)
    return {"ok": bool(ok and units), "units": units}


def prover(a, out: Path, key: bytes, h1, calls: list[dict], data: list[dict]) -> dict:
    """The prover's data: the option's A salt key, the receipt key, each covered call's A rows, scales and receipts, into the
    store (or OUT when the store refuses)."""
    with tempfile.TemporaryDirectory(dir=a.scratch) as td:
        p = Path(td) / "prover"
        p.mkdir()
        (p / "prover.json").write_text(json.dumps({"a_salt_key": h1.salt_key.hex(), "receipt_key": key.hex(),
                                                   "calls": [c["index"] for c in calls]}) + "\n")
        for c, d in zip(calls, data):
            np.savez(p / f"call{c['index']:05d}.npz", xq=d["xq"], sa=d["sa"], receipts=d["receipts"])
        tar = Path(td) / "prover.tar"
        with tarfile.open(tar, "w") as t:
            t.add(p, arcname="prover")
        size = tar.stat().st_size
        h = hashlib.sha256()
        with tar.open("rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
        cli = [shutil.which("research")] if shutil.which("research") else ["python3", "-m", "research"]
        env = {**os.environ, "PYTHONPATH": f"{HERE.parents[2] / 'tools' / 'research' / 'src'}:{os.environ.get('PYTHONPATH', '')}"}
        try:
            r = subprocess.run([*cli, "data", "put", "--kind", "pouw-h1-prover", "--file", str(tar), "--preserve"],
                               capture_output=True, text=True, env=env)
            rc, so, se = r.returncode, r.stdout, r.stderr
        except OSError as e:
            rc, so, se = -1, "", repr(e)
        doc = {"bytes": size, "sha256": h.hexdigest(), "cli": cli, "put_rc": rc, "put_out": so[-800:], "put_err": se[-800:]}
        if rc:
            (out / "prover").mkdir(exist_ok=True)
            os.replace(tar, out / "prover" / "prover.tar")
            doc["in_out"] = "prover/prover.tar"
        return doc


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
    ap.add_argument("--gpu-mem", type=float, default=0.6, help="vLLM's gpu_memory_utilization: the rest is the side computation's")
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--gate-per-site", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--skip-timing", action="store_true")
    return 0 if run(ap.parse_args(argv))["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
