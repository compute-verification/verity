"""A served H1 + fx1 window (`eps1-sm120-h1fx1`) on vLLM's int8-exact row, one process on one GPU.

The receipts bind extra work beside cuBLASLt's served y, not the served kernel's: after each int8-exact linear of a prefill,
`receipts.call_receipts` computes `h1tile`'s unit on the GPU on the served A (the `verity_int8::row_quant` output the
linear's `int8_ops.linear` returned) and the served weight B, for every full 32 × 32 tile.  Decode calls (M < 32), a call's
rows past its last full 32-row block, attention and lm_head are not covered.

In order, into OUT (`record.json` throughout, `manifest.json` at the end):

  timing    `noise.cu` over a call's per-unit noise (seed_j's blocks and σ_R's key blocks) against `torch._int_mm` of the same
            call, CUDA events (`timing`), at K = 4096 and 14336
  engine    the int8 pin and `engine.build.build_engine(target=from_device() + int8_exact)`, eager, prefix caching off
  weights   every int8 linear's B table (`commit`: one hm96-sha512/row/v2 leaf per output row), one root per weight tensor; the
            weights' root; the fraction of weights that W' = B + c + U mod 2^8 wraps, per layer (`wrap`)
  coin      the salt coin of a `pouw.coins.Challenger` epoch, opened after the weights' root; the 64-byte salt
  serve     the prompts generated with the side computation hooked into every int8 linear: each prefill call's A table (AROW's
            leaf: K s8 bytes then s_a as f32, hm96-sha512/row-seg/v1), its units at positions base + u, their receipts
  reference the same prompts again with no hook: the token ids must be the served run's
  receipts  one receipt tree per call (one hm96-sha512/row-seg/v1 leaf per unit)
  gate      drawn units of each site against `h1tile.reference` (receipt) and the served integer product (z); controls: the
            unit at the next position, a receipt with one bit flipped and a tampered A row against their roots
  prover    the prover's data (the salt key, each call's A rows and receipts) into the store (`research data put`)

    cd integrations/vllm; PYTHONPATH=.:../..:../../tools/research/src:../../benchmarks/pouw/served_h1
    $PY ../../benchmarks/pouw/served_h1/serve.py --out OUT --cubin noise.cubin --at 2026-10-18T00:00:00Z --by compute-accounting
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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
WEIGHTS_TAG = b"verity/pouw/served-h1/weights/v0\0"
SALT_BYTES = 64
FRAMING = "the receipts bind extra work beside cuBLASLt's served y, not the served kernel's"
NOT_COVERED = ["decode calls (M < 32)", "a call's rows past its last full 32-row block", "attention (QKᵀ, P·V)", "lm_head"]
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
                            "frac_expected_wrapped": round(float((h * p).sum()) / tot, 6), "abs_max": int(m.weight.abs().max())}
    for d in per_layer.values():
        d["frac_abs_gt_63"] = round(d["abs_gt_63"] / d["weights"], 6)
        d["frac_expected_wrapped"] = round(d["expected_wrapped"] / d["weights"], 6)
    tot = sum(d["weights"] for d in per_layer.values())
    return {"rule": "W' = B + c + U mod 2^8 wraps where B + c + U is outside [-128, 127]: possible only for |B| > 63; "
                    "expected_wrapped weights each B by P(wrap | B) over c (a fair bit) and U (-64 + byte mod 129)",
            "frac_abs_gt_63": round(sum(d["abs_gt_63"] for d in per_layer.values()) / tot, 6),
            "frac_expected_wrapped": round(sum(d["expected_wrapped"] for d in per_layer.values()) / tot, 6),
            "layers": {str(k): v for k, v in sorted(per_layer.items())}}


def b_tables(pool, mods: dict, window: str, key: bytes, chunk: int = 2048) -> tuple[dict, dict]:
    """Every weight's B table root, and its rows (n, K + 4) uint8 on the host."""
    rows, tasks, meta = {}, [], {}
    for name, m in mods.items():
        q = m.weight.cpu().numpy()
        s = m.weight_scale.reshape(-1).cpu().numpy()
        rows[name] = CM.s8_rows(q, s)
        n, k = q.shape
        fields = {"window": window, "module": name, "n": n, "k": k, "schema": CM.SCHEMA["b"]}
        dom = CM.domain("b", fields, n)
        meta[name] = {"dom": dom, "fields": fields, "n": n, "k": k}
        for at in range(0, n, chunk):
            tasks.append((name, at, (dom.binding, n, "b", key, f"b/{name}", at, rows[name][at:at + chunk])))
    out = pool.map(CM.leaves, [t[2] for t in tasks], chunksize=1)
    got: dict[str, list[bytes]] = {name: [] for name in mods}
    for (name, at, _), leaves in zip(tasks, out):
        got[name] += CM.split(leaves)
    roots = {}
    for name, mt in meta.items():
        lv = CM.levels(mt["dom"], got[name])
        roots[name] = {"root_b": lv[-1][0].hex(), "domain": mt["dom"].domain_id.hex(), "n": mt["n"], "k": mt["k"],
                       "schema": CM.SCHEMA["b"], "fields": mt["fields"]}
    return roots, rows


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
    from verity_vllm.engine import int8_exact as E
    from verity_vllm.engine import int8_ops
    from verity_vllm.engine.build import build_engine, target_mismatches
    from verity_vllm.program.frontend.target_profile import TargetProfile

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

    mods = {n: m for n, m in eng.model.named_modules()
            if type(getattr(m, "quant_method", None)).__name__ == "Int8PlainLinearMethod"}
    window = (os.environ.get("RESEARCH_RUN_ID") or "local") + "/" + os.urandom(8).hex()
    key = os.urandom(64)
    rec.step("wrap", wrap_counts(mods))
    t0 = time.perf_counter()
    weights, b_rows = b_tables(pool, mods, window, key)
    weights_root = hashlib.sha512(WEIGHTS_TAG + b"".join(bytes.fromhex(weights[n]["root_b"]) for n in mods)).digest()
    rec.step("weights", {"tensors": len(weights), "rows": sum(w["n"] for w in weights.values()),
                         "weights_root": weights_root.hex(), "s": round(time.perf_counter() - t0, 1)})

    from verity.core.protocols.pouw.audit import EPOCH_SALT_DOMAIN
    from verity.core.protocols.pouw.coins import Challenger as Coins
    from verity.ml.randomness import derive
    coins = Coins(a.epoch)
    opening = coins.open("salt", weights_root)
    salt = derive(opening.coin, EPOCH_SALT_DOMAIN, {"epoch": a.epoch, "coins": coins.commitment.root,
                                                    "weights": weights_root}).stream("salt", SALT_BYTES)
    rec.step("coin", {"commitment": coins.commitment.to_json(), "salt": salt.hex()})

    by_ptr = {m.weight.data_ptr(): n for n, m in mods.items()}
    st = {"on": True, "base": 0}
    calls: list[dict] = []
    data: list[dict] = []
    decode = {"calls": 0, "rows": 0}
    partial = {"rows": 0}
    requant_bad = [0]
    orig = int8_ops.linear

    def side(name: str, x2, xq, s_a, w, s_w) -> None:
        xq2, s2 = E.quantize_rows(x2)
        requant_bad[0] += not (torch.equal(xq2, xq) and torch.equal(s2, s_a))
        m, k = xq.shape
        n = w.shape[0]
        if m < RC.R:
            decode["calls"] += 1
            decode["rows"] += m
            return
        i = len(calls)
        layer, site = site_of(name)
        rows = CM.s8_rows(xq.cpu().numpy(), s_a.cpu().numpy())
        fields = {"window": window, "call": i, "module": name, "m": m, "k": k, "schema": CM.SCHEMA["a"]}
        dom = CM.domain("a", fields, m)
        lv = CM.levels(dom, [CM.leaf(dom, "a", key, f"a/{i}", r, bytes(row))[0] for r, row in enumerate(rows)])
        root_a, root_b = lv[-1][0], bytes.fromhex(weights[name]["root_b"])
        torch.cuda.synchronize()
        t = time.perf_counter()
        rc = RC.call_receipts(RC.Call(xq, w), salt, st["base"], root_a, root_b, cubin)
        torch.cuda.synchronize()
        units = rc.shape[0]
        calls.append({"call": i, "module": name, "layer": layer, "site": site, "m": m, "k": k, "n": n, "root_a": root_a.hex(),
                      "a_domain": dom.domain_id.hex(), "a_fields": fields, "a_schema": CM.SCHEMA["a"], "root_b": root_b.hex(),
                      "base": st["base"], "units": units, "rows_covered": RC.R * (m // RC.R), "receipt_words": int(rc.shape[1]),
                      "receipts_s": round(time.perf_counter() - t, 3),
                      "x_wrap_entries": int((xq == 127).sum())})
        partial["rows"] += m % RC.R
        data.append({"a_rows": rows, "receipts": rc, "xq": xq.clone(), "w": name})
        st["base"] += units

    def hooked(x2, w, s_w):
        xq, s_a, y = orig(x2, w, s_w)
        name = by_ptr.get(w.data_ptr())
        if st["on"] and name is not None:
            side(name, x2, xq, s_a, w, s_w)
        return xq, s_a, y

    int8_ops.linear = hooked
    t0 = time.perf_counter()
    try:
        served = G.generate(eng.llm, prompts, a.max_tokens)
    finally:
        int8_ops.linear = orig
        st["on"] = False
    serve_s = time.perf_counter() - t0
    rec.step("serve", {"calls": len(calls), "units": st["base"], "s": round(serve_s, 1), "decode_uncovered": decode,
                       "partial_rows_uncovered": partial["rows"], "a_requantized_mismatch": requant_bad[0],
                       "receipts_s": round(sum(c["receipts_s"] for c in calls), 1),
                       "prefill_m": sorted({c["m"] for c in calls})})
    reference = G.generate(eng.llm, prompts, a.max_tokens)
    equal = [s["token_ids"] == r["token_ids"] for s, r in zip(served, reference)]
    rec.step("tokens", {"equal": all(equal), "per_prompt": equal, "served": served, "reference": reference})

    t0 = time.perf_counter()
    rtasks = []
    for c, d in zip(calls, data):
        fields = {"window": window, "call": c["call"], "module": c["module"], "units": c["units"],
                  "words": c["receipt_words"], "schema": CM.SCHEMA["r"]}
        dom = CM.domain("r", fields, c["units"])
        c.update(receipt_domain=dom.domain_id.hex(), receipt_fields=fields, receipt_schema=CM.SCHEMA["r"])
        rtasks.append((dom, CM.receipt_rows(d["receipts"])))
    rl = pool.map(CM.leaves, [(dom.binding, dom.count, "r", key, f"r/{c['call']}", 0, rows)
                              for c, (dom, rows) in zip(calls, rtasks)], chunksize=1)
    levels = []
    for c, (dom, _), leaves in zip(calls, rtasks, rl):
        lv = CM.levels(dom, CM.split(leaves))
        levels.append(lv)
        c["receipt_root"] = lv[-1][0].hex()
    rec.step("receipt_trees", {"calls": len(calls), "leaves": sum(c["units"] for c in calls),
                               "s": round(time.perf_counter() - t0, 1)})

    rec.step("gate", gate(a, calls, data, levels, b_rows, mods, salt, key, E))

    manifest = {"schema": SCHEMA, "scheme": SCHEME, "tile": [RC.R, RC.C_], "framing": FRAMING, "not_covered": NOT_COVERED,
                "window": window, "model": rec.doc["steps"]["engine"]["pin"], "epoch": a.epoch,
                "coins": coins.commitment.to_json(), "salt_opening": opening.to_json(), "salt": salt.hex(),
                "salt_rule": "derive(coin, EPOCH_SALT_DOMAIN, {epoch, coins: the commitment's root, weights: the weights' root})"
                             ".stream('salt', 64), the coin opened after the weights' root (core_check's 32-byte rule at 64)",
                "weights_rule": "SHA-512(WEIGHTS_TAG || each weight's root_b in the order of `weights`)",
                "weights_tag": WEIGHTS_TAG.hex(), "weights_root": weights_root.hex(),
                "weights": [{"module": n, **{k: v for k, v in w.items() if k != "fields"}, "fields": w["fields"]}
                            for n, w in weights.items()],
                "positions": "position = base + u: calls in serving order, a call's units by row block then column block "
                             "(unit u = rows 32 (u // (N / 32)) .., columns 32 (u % (N / 32)) ..)",
                "leaves": {"a": "token row i: K s8 bytes then s_a[i] f32 LE, hm96-sha512/row-seg/v1 (AROW's)",
                           "b": "output row j: K s8 bytes then s_w[j] f32 LE, hm96-sha512/row/v2",
                           "r": "unit u: its receipt's 2 (KS + m) u32 words LE (per set y then z), hm96-sha512/row-seg/v1",
                           "domains": "identity_digest_sha512('verity/pouw/served-h1/<a-table|b-table|receipts>/v0', "
                                      "fields), owner -1, ranks 0 .. count - 1",
                           "salts": "SHA-512(SALT_TAG || key || u32be len(label) || label || u64be i || u8 c), c = 0..2: "
                                    "labels a/<call>, b/<module>, r/<call>; the key is the prover's"},
                "calls": calls, "units": st["base"],
                "tokens_equal": all(equal)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    rec.step("prover", prover(a, out, key, calls, data))
    rec.doc["ok"] = bool(all(equal) and not requant_bad[0] and calls and rec.doc["steps"]["gate"]["ok"])
    rec.save()
    pool.close()
    print(f"SERVED-H1 {'OK' if rec.doc['ok'] else 'FAIL'} calls={len(calls)} units={st['base']} tokens_equal={all(equal)} "
          f"gate={rec.doc['steps']['gate']['ok']}", flush=True)
    return rec.doc


def gate(a, calls: list[dict], data: list[dict], levels: list, b_rows: dict, mods: dict, salt: bytes, key: bytes, E) -> dict:
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
                picks.append((c["call"], rng.randrange(c["units"])))
    units, ok = [], True
    for i, u in picks:
        c, d = calls[i], data[i]
        b = torch.from_numpy(b_rows[c["module"]][:, :c["k"]].copy().view(np.int8))
        call = RC.Call(d["xq"].cpu(), b)
        root_a, root_b = bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"])
        z, want = RC.reference(call, salt, c["base"], root_a, root_b, u)
        got = [int(v) for v in d["receipts"][u]]
        ri, cj = call.units[u]
        y = E.int_mm(d["xq"][32 * ri:32 * ri + 32], mods[c["module"]].weight[32 * cj:32 * cj + 32]).cpu().to(torch.int64)
        z_ok = [v & 0xFFFFFFFF for v in y.flatten().tolist()] == z
        nxt = RC.reference(call, salt, c["base"] + 1, root_a, root_b, u)[1]
        dom = CM.domain("r", c["receipt_fields"], c["units"])
        root_r = bytes.fromhex(c["receipt_root"])
        rows = CM.receipt_rows(d["receipts"][u:u + 1])[0]
        leaf = CM.leaf(dom, "r", key, f"r/{i}", u, bytes(rows))[0]
        sibs = CM.path(levels[i], u)
        flipped = bytearray(rows)
        flipped[rng.randrange(len(flipped))] ^= 1 << rng.randrange(8)
        leaf_f = CM.leaf(dom, "r", key, f"r/{i}", u, bytes(flipped))[0]
        a_dom = CM.domain("a", c["a_fields"], c["m"])
        row = bytearray(d["a_rows"][32 * ri])
        row[rng.randrange(c["k"])] ^= 1
        a_leaf_t = CM.leaf(a_dom, "a", key, f"a/{i}", 32 * ri, bytes(row))[0]
        a_lv = CM.levels(a_dom, [CM.leaf(a_dom, "a", key, f"a/{i}", r, bytes(x))[0] for r, x in enumerate(d["a_rows"])])
        r = {"call": i, "module": c["module"], "k": c["k"], "unit": u, "position": c["base"] + u,
             "receipt_equal_reference": got == want, "z_equal_served_product": z_ok,
             "next_position_differs": nxt != got, "receipt_opens": CM.verify(dom, root_r, u, leaf, sibs),
             "flipped_receipt_refused": not CM.verify(dom, root_r, u, leaf_f, sibs),
             "a_root_rebuilt": a_lv[-1][0] == root_a,
             "tampered_a_row_refused": not CM.verify(a_dom, root_a, 32 * ri, a_leaf_t, CM.path(a_lv, 32 * ri))}
        r["ok"] = all(v for k_, v in r.items() if isinstance(v, bool))
        ok &= r["ok"]
        units.append(r)
        print(f"[served-h1] gate {json.dumps(r)}", flush=True)
    return {"ok": bool(ok and units), "units": units}


def prover(a, out: Path, key: bytes, calls: list[dict], data: list[dict]) -> dict:
    """The prover's data: the salt key, each call's A rows and receipts, into the store (or OUT when the store refuses)."""
    with tempfile.TemporaryDirectory(dir=a.scratch) as td:
        p = Path(td) / "prover"
        p.mkdir()
        (p / "prover.json").write_text(json.dumps({"key": key.hex(), "calls": len(calls)}) + "\n")
        for c, d in zip(calls, data):
            np.savez(p / f"call{c['call']:04d}.npz", a_rows=d["a_rows"], receipts=d["receipts"])
        tar = Path(td) / "prover.tar"
        with tarfile.open(tar, "w") as t:
            t.add(p, arcname="prover")
        size = tar.stat().st_size
        sha = hashlib.sha256(tar.read_bytes()).hexdigest()
        r = subprocess.run(["research", "data", "put", "--kind", "pouw-h1-prover", "--file", str(tar), "--preserve"],
                           capture_output=True, text=True)
        doc = {"bytes": size, "sha256": sha, "put_rc": r.returncode, "put_out": r.stdout[-800:], "put_err": r.stderr[-800:]}
        if r.returncode:
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
