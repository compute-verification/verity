"""GPU run: vLLM's read-back of an `fp8_ds_mla` cache row on its dense prefill (``ops.cp_gather_and_upconvert_fp8_kv_cache``,
whose token function is `verity.definitions.registry.fp8_ds_mla_upconvert`'s docstring's), every bf16 word it writes held to
``Fp8DsMlaUpconvertRow_v1{DN=512,DR=64}``.

Three phases, each a process of its own:

- ``--phase capture`` (the card): first the op called directly on rows of planted bytes (`planted_rows`: every vector of
  `HAND_VECTORS` as a whole row, every E4M3 code under `EDGE_SCALES`, uniform bytes, and uniform codes under scales of
  every f32 exponent), in a cache whose rows sit behind a shuffled block table at offsets (``seq_starts``), the
  workspace poisoned with 0xA5A5 first; then the stand-in of `fp8_cache_replay_gpu` (GLM-5.3 cut to `LAYERS` layers,
  `HEADS` heads, tensor-parallel 1, eager) served with ``kv_cache_dtype="fp8"`` on redrawn weights (``--weights
  scaled``), its latent norm's gain set per 128-value group to `GAINS` so the cache write quantises groups of very
  different sizes. It serves `PLAN` (requests arriving between steps, some sharing a cached prefix, more tokens than a
  step takes), then `SWEEP` one request at a time. Every call of the op is wrapped: after it, the rows the kernel read
  (by its own task map, `sources`) and the workspace it wrote are copied out, and each word is held to the Definition's
  numpy kernel at once. The calls of the first attention layer are saved whole (``calls/*.npz``), the others by
  SHA-512. Each step's batch (each request's computed and scheduled tokens) and the first layer's route decisions
  (``_use_sparse_mha`` with ``prefill_max_seq_len``, the longest query and whether the step has cached context) are
  recorded, which says from a run which request lengths take this route.
- ``--phase compare`` (no card): every saved row, planted and served, through the Definition's own evaluator
  (``evaluate_batch(..., kernel="reference")``), its words against the kernel's; what the rows hold (NaN codes, subnormal
  codes and scales, subnormal and infinite products); and the ``_C_stable_libtorch.abi3.so`` the run loaded, by sha256,
  with the counts of the conversion and multiply instructions in its sm_90 SASS of the kernel where ``cuobjdump`` is found.
- without ``--phase``: both, capture first.

Each writes OUT/<phase>.json with rows compared and rows that differ; it exits 1 when a row differs or an output row has
no source, 0 otherwise. On one idle card, from the tree's root, in the pod's venv:

    PYTHONHASHSEED=0 $PY adapters/vllm/tests/properties/fp8_read_back_capture_gpu.py --phase capture --out DIR
    $PY adapters/vllm/tests/properties/fp8_read_back_capture_gpu.py --phase compare --out DIR
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ADAPTER = Path(__file__).resolve().parents[2]
#: ``cp_gather_and_upconvert_fp8_kv_cache_page``'s mangled name in vLLM's ``_C_stable_libtorch.abi3.so``
KERNEL = "_ZN4vllm41cp_gather_and_upconvert_fp8_kv_cache_pageEPKhP13__nv_bfloat16PKiS5_iiillllS5_"
#: the instructions counted in its SASS: the E4M3 -> f16 unpack, f16 -> f32, the multiply, the f32 -> bf16 pack, and what
#: would contract or flush the multiply
SASS_OPS = ("F2FP.F16.E4M3.UNPACK_B", "HADD2.F32", "FMUL", "FFMA", "FMUL.FTZ", "F2FP.BF16.F32.PACK_AB")
#: the latent norm's gain on each 128-value group of the NoPE part: 1; 2^-120, whose group's scale is an f32 subnormal and
#: whose small codes give subnormal products; per value 2^-14 to 2^2 (`spread`), so one group's codes span E4M3's range
#: down to its subnormals; 2^-40
GAINS = (1.0, 2.0 ** -120, "spread", 2.0 ** -40)
#: the served requests: prompt length, the step it arrives before, and whether it starts with the shared prefix
PLAN = [(24, 0, False), (150, 0, True), (610, 1, False), (330, 2, True), (70, 2, False), (900, 3, True), (610, 4, True),
        (1200, 4, False), (40, 6, True), (512, 7, False)]
SHARED = 256
#: prompt lengths served alone after `PLAN`, around GLM-5.3's ``index_topk`` (2048); a negative one repeats the prompt
#: of that length, so it starts from a prefix-cache hit
SWEEP = [600, 1500, 2047, 2048, 2049, 2100, -1500, -2048, -2049]
#: the workspace's poison, as a bf16 word
POISON = 0xA5A5


def sources(block_table: np.ndarray, ws: np.ndarray, starts: np.ndarray | None, batch: int, total: int,
            block_size: int) -> np.ndarray:
    """The kernel's task map (``map_gather_page_task<false>``, cache_kernels.cu:1139-1176): output row o of request r, for o
    in [ws[r], ws[r + 1]) (the last request's end is ``total``), reads source position starts[r] + (o - ws[r]) at block
    block_table[r, pos // block_size], slot pos % block_size. Returns (total, 4): request, position, block, slot; -1 where
    no request writes the row."""
    out = np.full((total, 4), -1, np.int64)
    for r in range(batch):
        b = min(int(ws[r]), total)
        e = min(int(ws[r + 1]) if r + 1 < batch else total, total)
        n = max(e - b, 0)
        pos = (0 if starts is None else int(starts[r])) + np.arange(n, dtype=np.int64)
        out[b:b + n, 0] = r
        out[b:b + n, 1] = pos
        out[b:b + n, 2] = block_table[r, pos // block_size]
        out[b:b + n, 3] = pos % block_size
    return out


def held(rows: np.ndarray, words: np.ndarray) -> dict[str, Any]:
    """The kernel's words against the Definition's numpy kernel on the rows it read."""
    from verity.definitions.registry import fp8_ds_mla_upconvert as U

    want = U.upconvert_rows(*U.split_rows(rows, **U.FP8_DS_MLA_GLM53))[0]
    bad = want != words
    rows_bad = np.flatnonzero(bad.any(axis=1))
    out: dict[str, Any] = {"rows": int(len(rows)), "rows_differing": int(len(rows_bad)), "words_differing": int(bad.sum())}
    if len(rows_bad):
        r = int(rows_bad[0])
        w = int(np.flatnonzero(bad[r])[0])
        out["first"] = {"row": r, "word": w, "kernel": int(words[r, w]), "definition": int(want[r, w])}
    return out


def _words(t: Any) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def planted_rows(rng: np.random.Generator) -> np.ndarray:
    """Rows of planted bytes (N, 656): each hand vector across a whole row; every code under each four of `EDGE_SCALES`,
    forward and reversed; 512 rows of uniform bytes; 512 of uniform codes under scales of uniform sign, exponent and
    mantissa. The RoPE words are uniform, NaN payloads among them."""
    from verity.definitions.registry import fp8_ds_mla_upconvert as U

    DN, DR = U.FP8_DS_MLA_GLM53["DN"], U.FP8_DS_MLA_GLM53["DR"]
    G = DN // U.QT

    def row(nope, scale) -> np.ndarray:
        rope = rng.integers(0, 1 << 16, DR, dtype=np.uint16)
        return np.concatenate([np.asarray(nope, np.uint8), np.asarray(scale, "<u4").view(np.uint8), rope.astype("<u2").view(np.uint8)])

    out = [row([b] * DN, [s] * G) for b, s, _, _ in U.HAND_VECTORS]
    codes = np.tile(np.arange(256, dtype=np.uint8), DN // 256)
    for k in range(0, len(U.EDGE_SCALES), G):
        s = U.EDGE_SCALES[k:k + G]
        out += [row(codes, s), row(codes[::-1], s[::-1])]
    out += list(rng.integers(0, 256, (512, U.row_bytes(DN, DR)), dtype=np.uint8))
    for _ in range(512):
        s = (rng.integers(0, 2, G, dtype=np.uint32) << 31) | (rng.integers(0, 255, G, dtype=np.uint32) << 23) \
            | rng.integers(0, 1 << 23, G, dtype=np.uint32)
        out.append(row(rng.integers(0, 256, DN, dtype=np.uint8), s))
    return np.stack(out)


def planted(out: Path) -> dict[str, Any]:
    """The op on `planted_rows`: three requests at source offsets 0, 37 and 69 behind a shuffled block table, the rest of
    the cache random bytes, the workspace poisoned; then the first request alone without ``seq_starts``."""
    import torch
    from vllm import _custom_ops as ops

    rng = np.random.default_rng(20261112)
    rows = planted_rows(rng)
    N, RB, BS = len(rows), rows.shape[1], 64
    parts = [N // 3, N // 3, N - 2 * (N // 3)]
    starts = np.array([0, 37, 69], np.int32)
    need = [-(-(int(s) + n) // BS) for s, n in zip(starts, parts)]
    nblocks = sum(need) + 16
    perm = rng.permutation(nblocks).astype(np.int32)
    table = np.zeros((3, max(need)), np.int32)
    host = rng.integers(0, 256, (nblocks, BS, RB), dtype=np.uint8)
    at, used = 0, 0
    for r, (s, n) in enumerate(zip(starts, parts)):
        table[r, :need[r]] = perm[used:used + need[r]]
        used += need[r]
        pos = int(s) + np.arange(n)
        host[table[r, pos // BS], pos % BS] = rows[at:at + n]
        at += n
    ws = np.array([0, parts[0], parts[0] + parts[1]], np.int32)
    expect = sources(table, ws, starts, 3, N, BS)
    cache = torch.from_numpy(host).cuda()
    d = out / "planted"
    d.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"rows": N, "requests": parts, "seq_starts": starts.tolist(), "calls": []}
    for name, batch, total, st in (("three", 3, N, starts), ("first, no seq_starts", 1, parts[0], None)):
        dst = torch.empty((total, 576), dtype=torch.bfloat16, device="cuda")
        dst.view(torch.int16).fill_(POISON - (1 << 16))
        ops.cp_gather_and_upconvert_fp8_kv_cache(
            src_cache=cache, dst=dst, block_table=torch.from_numpy(table[:batch]).cuda(),
            workspace_starts=torch.from_numpy(ws[:batch]).cuda(), batch_size=batch,
            seq_starts=None if st is None else torch.from_numpy(st[:batch]).cuda())
        torch.cuda.synchronize()
        words = _words(dst)
        src = sources(table, ws, st, batch, total, BS)
        read = host[src[:, 2], src[:, 3]]
        same_map = batch == 1 or bool((src == expect).all())
        entry = {"call": name, "batch": batch, "total": total, "unmapped": int((src[:, 0] < 0).sum()),
                 "map_is_planted": same_map and bool((read == rows[:total]).all()),
                 "poison_rows": int((words == POISON).all(axis=1).sum()), **held(read, words)}
        doc["calls"].append(entry)
        np.savez_compressed(d / f"{len(doc['calls']) - 1:02d}.npz", rows=read, words=words, block_table=table[:batch],
                            workspace_starts=ws[:batch], seq_starts=np.array([] if st is None else st[:batch], np.int32))
    doc["ok"] = all(c["rows_differing"] == 0 and c["unmapped"] == 0 and c["map_is_planted"] for c in doc["calls"])
    return doc


def spread(rng: random.Random, n: int) -> list[float]:
    return [2.0 ** rng.uniform(-14.0, 2.0) for _ in range(n)]


def set_gains(model: Any) -> dict[str, Any]:
    """Every attention layer's latent norm gain (``kv_a_layernorm.weight``, 512 values) to `GAINS`, group by group."""
    import torch

    rng = random.Random(7)
    hit = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("kv_a_layernorm.weight") and p.numel() == 512:
                g = []
                for x in GAINS:
                    g += spread(rng, 128) if x == "spread" else [x] * 128
                p.copy_(torch.tensor(g, dtype=torch.float32).to(p.dtype))
                hit.append(name)
    if not hit:
        raise SystemExit("fp8-read-back: no kv_a_layernorm.weight of 512 values in the stand-in: the gains are not set")
    return {"layers": len(hit), "first": hit[0], "gains": [x if isinstance(x, str) else float(x) for x in GAINS]}


def serve(out: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ADAPTER))
    import torch
    from vllm import _custom_ops as ops

    from tests.properties import fp8_cache_replay_gpu as R
    from verity_vllm.engine import env, handle
    from verity_vllm.engine.hooks import Hooks

    env.apply_engine(None)
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
    model = Path(tempfile.mkdtemp(prefix="fp8-read-back-stand-in-"))
    R.stand_in(model)
    cfg = {**R.SERVE, "kv_cache_dtype": "fp8"}
    eng = R.engine(model, cfg, eager=True)
    runner = handle.model_runner(eng)
    redrawn = R.rescale(runner)
    gains = set_gains(runner.model)
    layer = next((m for m in R.kv_layers(runner.model).values() if callable(getattr(m, "_compute_prefill_context", None))), None)
    if layer is None:
        raise SystemExit("fp8-read-back: no attention layer with _compute_prefill_context")
    d = out / "calls"
    d.mkdir(parents=True, exist_ok=True)
    st: dict[str, Any] = {"step": -1, "in_step": False, "first_layer": False}
    calls: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []

    def gather(orig: Any) -> Any:
        def wrapped(src_cache: Any, dst: Any, block_table: Any, workspace_starts: Any, batch_size: int,
                    seq_starts: Any = None) -> Any:
            res = orig(src_cache=src_cache, dst=dst, block_table=block_table, workspace_starts=workspace_starts,
                       batch_size=batch_size, seq_starts=seq_starts)
            bt = block_table.cpu().numpy()
            ws = workspace_starts.cpu().numpy()
            ss = None if seq_starts is None else seq_starts.cpu().numpy()
            total, bs = int(dst.shape[0]), int(src_cache.shape[1])
            src = sources(bt, ws, ss, int(batch_size), total, bs)
            ok = src[:, 0] >= 0
            idx = torch.from_numpy(src[ok]).to(src_cache.device)
            rows = src_cache[idx[:, 2], idx[:, 3]].contiguous().view(torch.uint8).cpu().numpy()
            rows = rows.reshape(int(ok.sum()), int(src_cache.shape[-1]) * src_cache.element_size())
            words = _words(dst)[ok]
            n = len(calls)
            entry = {"call": n, "step": st["step"], "in_step": st["in_step"], "first_layer": st["first_layer"],
                     "batch": int(batch_size), "total": total, "unmapped": int((~ok).sum()), "block_size": bs,
                     "requests": [{"workspace_start": int(ws[r]), "seq_start": None if ss is None else int(ss[r]),
                                   "rows": int((src[:, 0] == r).sum())} for r in range(int(batch_size))],
                     **held(rows, words)}
            if st["first_layer"]:
                np.savez_compressed(d / f"{n:05d}.npz", rows=rows, words=words, block_table=bt, workspace_starts=ws,
                                    seq_starts=np.array([] if ss is None else ss, np.int32))
                entry["saved"] = f"calls/{n:05d}.npz"
            else:
                entry["sha512"] = {"rows": hashlib.sha512(rows.tobytes()).hexdigest(),
                                   "words": hashlib.sha512(words.tobytes()).hexdigest()}
            calls.append(entry)
            return res
        return wrapped

    def context(orig: Any) -> Any:
        def wrapped(*args: Any, **kw: Any) -> Any:
            st["first_layer"] = True
            try:
                return orig(*args, **kw)
            finally:
                st["first_layer"] = False
        return wrapped

    def route(orig: Any) -> Any:
        def wrapped(attn_metadata: Any, *args: Any, **kw: Any) -> Any:
            dense = orig(attn_metadata, *args, **kw)
            p = getattr(attn_metadata, "prefill", None)
            routes.append({"step": st["step"], "in_step": st["in_step"], "dense": bool(dense),
                           "prefill_max_seq_len": getattr(attn_metadata, "prefill_max_seq_len", None),
                           "max_query_len": getattr(p, "max_query_len", None),
                           "cached_context": getattr(p, "chunked_context", None) is not None,
                           "use_dense_mha": getattr(p, "use_dense_mha", None)})
            return dense
        return wrapped

    def execute_model(orig: Any) -> Any:
        def wrapped(scheduler_output: Any, *args: Any, **kw: Any) -> Any:
            real = not kw.get("dummy_run") and int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0) > 0
            if real:
                st["step"] += 1
                st["in_step"] = True
            try:
                res = orig(scheduler_output, *args, **kw)
            finally:
                st["in_step"] = False
            if real:
                ib = runner.execute_model_state.input_batch
                k = len(ib.req_ids)
                qsl = np.asarray(ib.query_start_loc_np[:k + 1], dtype=np.int64)
                nct = np.asarray(ib.num_computed_tokens_np[:k], dtype=np.int64)
                steps.append({"step": st["step"], "batch": [
                    {"request": str(r), "computed": int(nct[i]), "scheduled": int(qsl[i + 1] - qsl[i])}
                    for i, r in enumerate(ib.req_ids[:k])]})
            return res
        return wrapped

    hooks = Hooks(tag="fp8-read-back")
    hooks.wrap(ops, "cp_gather_and_upconvert_fp8_kv_cache", gather)
    hooks.wrap(layer, "_compute_prefill_context", context)
    hooks.wrap(layer, "_use_sparse_mha", route)
    hooks.wrap(runner, "execute_model", execute_model)
    rng = random.Random(0)

    def ids(n: int) -> list[int]:
        return [rng.randrange(1000, 150000) for _ in range(n)]

    shared = ids(SHARED)
    reqs: dict[str, list[int]] = {}
    due = []
    for i, (n, at, pre) in enumerate(PLAN):
        rid = f"plan-{i:02d}"
        reqs[rid] = shared + ids(max(n - SHARED, 8)) if pre else ids(n)
        due.append((at, rid))
    greedy = R.params({"temperature": 0.0, "seed": 0}, 8)
    it = 0
    pending = sorted(due)
    while pending or eng.has_unfinished_requests():
        while pending and pending[0][0] <= it:
            rid = pending.pop(0)[1]
            eng.add_request(rid, {"prompt_token_ids": reqs[rid]}, greedy)
        eng.step()
        it += 1
    swept: dict[int, str] = {}
    one = R.params({"temperature": 0.0, "seed": 0}, 2)
    sweep_runs = []
    for k, n in enumerate(SWEEP):
        rid = f"sweep-{k:02d}-{'repeat-' if n < 0 else ''}{abs(n)}"
        reqs[rid] = reqs[swept[-n]] if n < 0 else ids(n)
        if n > 0:
            swept[n] = rid
        first = st["step"] + 1
        eng.add_request(rid, {"prompt_token_ids": reqs[rid]}, one)
        while eng.has_unfinished_requests():
            eng.step()
        sweep_runs.append({"request": rid, "prompt": len(reqs[rid]), "repeat": n < 0, "steps": [first, st["step"]]})
    hooks.uninstall()
    shutil.rmtree(model, ignore_errors=True)

    def per_step(s: int) -> dict[str, Any]:
        rs = [r for r in routes if r["step"] == s]
        cs = [c for c in calls if c["step"] == s and c["first_layer"]]
        b = next((x["batch"] for x in steps if x["step"] == s), [])
        return {"step": s, "batch": [{**x, "seq_len": x["computed"] + x["scheduled"]} for x in b],
                "routes": [{k: r[k] for k in ("dense", "prefill_max_seq_len", "max_query_len", "cached_context")} for r in rs],
                "gather_rows_first_layer": sum(c["rows"] for c in cs)}

    for x in sweep_runs:
        x["by_step"] = [per_step(s) for s in range(x["steps"][0], x["steps"][1] + 1)]
    real = [c for c in calls if c["in_step"]]
    doc = {"chosen": R.chosen(eng, runner), "redrawn": redrawn, "gains": gains, "steps": len(steps),
           "device": {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))},
           "vllm": __import__("vllm").__version__, "calls": calls, "routes": routes, "batches": steps, "sweep": sweep_runs,
           "plan_steps": [per_step(s) for s in range(0, (sweep_runs[0]["steps"][0] if sweep_runs else st["step"] + 1))],
           "rows": sum(c["rows"] for c in real), "rows_differing": sum(c["rows_differing"] for c in real),
           "rows_first_layer": sum(c["rows"] for c in real if c["first_layer"]),
           "unmapped": sum(c["unmapped"] for c in calls), "calls_outside_steps": len(calls) - len(real)}
    doc["ok"] = bool(real) and doc["rows_differing"] == 0 and doc["unmapped"] == 0 \
        and all(c["rows_differing"] == 0 for c in calls)
    return doc


def capture(out: Path) -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("fp8-read-back: PYTHONHASHSEED=0 (a set's order decides a batch's row order)")
    doc: dict[str, Any] = {}
    try:
        doc["planted"] = planted(out)
        print(f"fp8-read-back: planted {[{k: c[k] for k in ('call', 'rows', 'rows_differing')} for c in doc['planted']['calls']]}",
              flush=True)
    except Exception as e:  # noqa: BLE001 - recorded, and the serve still runs
        doc["planted"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    try:
        doc["serve"] = serve(out)
        s = doc["serve"]
        print(f"fp8-read-back: served {len(s['calls'])} calls, {s['rows']} rows, {s['rows_differing']} differing", flush=True)
    except Exception as e:  # noqa: BLE001 - recorded
        import traceback

        doc["serve"] = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
    doc["ok"] = bool(doc["planted"].get("ok")) and bool(doc["serve"].get("ok"))
    (out / "capture.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if doc["ok"] else 1


def coverage(nope: np.ndarray, scale: np.ndarray, words: np.ndarray, DN: int) -> dict[str, int]:
    e = (scale >> 23) & 0xFF
    w = words[:, :DN]
    we = (w >> 7) & 0xFF
    return {"nan_codes": int(((nope & 0x7F) == 0x7F).sum()),
            "subnormal_codes": int(((((nope >> 3) & 0xF) == 0) & ((nope & 0x7) != 0)).sum()),
            "zero_codes": int(((nope & 0x7F) == 0).sum()), "subnormal_scales": int(((e == 0) & ((scale & 0x7FFFFF) != 0)).sum()),
            "zero_scales": int(((scale & 0x7FFFFFFF) == 0).sum()), "nonfinite_scales": int((e == 0xFF).sum()),
            "distinct_scale_exponents": int(len(np.unique(e))), "subnormal_products": int(((we == 0) & ((w & 0x7F) != 0)).sum()),
            "infinite_products": int(((w & 0x7FFF) == 0x7F80).sum()), "nan_products": int((w == 0x7FFF).sum())}


def sass(lib: Path) -> dict[str, Any]:
    """The kernel's sm_90 SASS in ``lib`` by `cuobjdump`, where one is found: its instruction counts for `SASS_OPS`."""
    tools = [shutil.which("cuobjdump"), *sorted(glob.glob("/usr/local/cuda*/bin/cuobjdump"))]
    tool = next((t for t in tools if t), None)
    if tool is None:
        return {"cuobjdump": None}
    p = subprocess.run([tool, "-sass", "-arch", "sm_90", "-fun", KERNEL, str(lib)], capture_output=True, text=True)
    text = p.stdout
    opcodes = [m.group(1) for m in re.finditer(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)", text)]
    counts = {op: opcodes.count(op) for op in SASS_OPS}
    counts["FFMA*"] = sum(o.startswith("FFMA") for o in opcodes)
    counts[".FTZ outside F2I"] = sum(".FTZ" in o and not o.startswith("F2I") for o in opcodes)
    return {"cuobjdump": tool, "rc": p.returncode, "instructions": len(opcodes), "counts": counts, "text": text}


def compare(out: Path) -> int:
    from verity.circuits.evaluation import evaluate_batch
    from verity.definitions.registry import fp8_ds_mla_upconvert as U

    D = U.glm53_row()
    DN, DR = U.FP8_DS_MLA_GLM53["DN"], U.FP8_DS_MLA_GLM53["DR"]
    doc: dict[str, Any] = {"definition": D.id, "sets": {}}
    for name, pat in (("planted", "planted/*.npz"), ("served", "calls/*.npz")):
        files = sorted(out.glob(pat))
        rows = np.concatenate([np.load(f)["rows"] for f in files]) if files else np.zeros((0, U.row_bytes(DN, DR)), np.uint8)
        words = np.concatenate([np.load(f)["words"] for f in files]) if files else np.zeros((0, DN + DR), np.uint16)
        nope, scale, rope = U.split_rows(rows, DN, DR)
        want = evaluate_batch(D, nope, scale, rope, kernel="reference") if len(rows) else words
        bad = np.asarray(want, np.uint16) != words
        doc["sets"][name] = {"files": len(files), "rows": int(len(rows)), "rows_differing": int(bad.any(axis=1).sum()),
                             "words": int(bad.size), "words_differing": int(bad.sum()),
                             "coverage": coverage(nope, scale, words, DN) if len(rows) else {}}
        print(f"fp8-read-back: {name}: {doc['sets'][name]['rows']} rows through {D.id}, "
              f"{doc['sets'][name]['rows_differing']} differing", flush=True)
    try:
        import vllm

        lib = Path(vllm.__file__).parent / "_C_stable_libtorch.abi3.so"
        doc["library"] = {"path": str(lib), "sha256": hashlib.sha256(lib.read_bytes()).hexdigest(), "vllm": vllm.__version__}
        s = sass(lib)
        if s.get("text"):
            (out / "sass_sm90.txt").write_text(s.pop("text"))
        doc["sass"] = s
    except Exception as e:  # noqa: BLE001 - recorded, the comparison stands without it
        doc["library"] = {"error": f"{type(e).__name__}: {e}"}
    sets = doc["sets"].values()
    doc["ok"] = all(x["rows"] > 0 and x["rows_differing"] == 0 for x in sets)
    (out / "compare.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if doc["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=("capture", "compare"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.phase == "capture":
        return capture(out)
    if a.phase == "compare":
        return compare(out)
    rc = subprocess.run([sys.executable, __file__, "--out", str(out), "--phase", "capture"]).returncode
    return max(rc, subprocess.run([sys.executable, __file__, "--out", str(out), "--phase", "compare"]).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
