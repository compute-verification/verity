"""GPU run of the FA-tap exactness property (`verity_vllm.check.fa_tap_exactness`): check a tap build on this GPU and write the sealed
record `<out>/fa_tap_exactness.json`.  Exit status 1 unless ok; `--verify RECORD` re-derives a record's digest and verdict without a GPU.

The launch arguments are the ones `vllm.vllm_flash_attn.flash_attn_varlen_func` passes to the installed op (recorded at the op), so the
check follows the installed interface.  It lives under tests/ because it seeds a random generator, wraps the installed op and parses
arguments, which library code may not do yet (lints P3, P9, P6).

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as gate (b)
    $PY tests/check/fa_tap_exactness_gpu.py --so /workspace/cp/fa2/build/matReq/verity_fa2_matReq.so --out DIR          # FA2 (L40S)
    $PY tests/check/fa_tap_exactness_gpu.py --so /workspace/cp/fa2/build/fa3_matReq/verity_fa3_matReq.so --out DIR      # FA3 (H100)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.check.fa_tap_exactness import (FILL, INSTALLED, PAGE, QUICK, RECORD, SCHEMA, case_ok, cases, expected_written, geometry,
                                                negative_ok, negatives, seal, verify_record, visited_rows)
from verity_vllm.commit.hidden_stream import StreamLayout


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _environment() -> dict[str, Any]:
    import torch

    env = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0)))}
    for mod in ("vllm", "triton"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            env[mod] = f"unavailable: {e}"[:160]
    return env


def _inputs(c: dict[str, Any], D: int, seed: int) -> dict[str, Any]:
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)
    b, H, Hk = len(c["seqlens_q"]), c["H"], c["Hk"]
    per_seq = -(-max(c["seqlens_k"]) // PAGE)
    blocks = b * per_seq + 8
    q = torch.randn(sum(c["seqlens_q"]), H, D, device="cuda", generator=g).to(torch.bfloat16)
    k = torch.randn(blocks, PAGE, Hk, D, device="cuda", generator=g).to(torch.bfloat16)
    v = torch.randn(blocks, PAGE, Hk, D, device="cuda", generator=g).to(torch.bfloat16)
    block_table = torch.randperm(blocks, device="cuda", generator=g).to(torch.int32)[: b * per_seq].reshape(b, per_seq).contiguous()
    cu_q = torch.zeros(b + 1, dtype=torch.int32, device="cuda")
    cu_q[1:] = torch.cumsum(torch.tensor(c["seqlens_q"], device="cuda"), 0)
    return {"q": q, "k": k, "v": v, "cu_seqlens_q": cu_q, "seqused_k": torch.tensor(c["seqlens_k"], dtype=torch.int32, device="cuda"),
            "block_table": block_table, "max_seqlen_q": max(c["seqlens_q"]), "max_seqlen_k": max(c["seqlens_k"]),
            "scale": float(np.float32(D ** -0.5))}


def _vllm_args(fa_version: int, c: dict[str, Any], t: dict[str, Any]) -> tuple:
    """The positional arguments vLLM's `flash_attn_varlen_func` hands the installed op for this launch."""
    import torch
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    ns_name, op_name = INSTALLED[fa_version]
    ns = getattr(torch.ops, ns_name)
    op = getattr(ns, op_name)
    seen: list[tuple[tuple, dict]] = []

    def record(*args, **kwargs):
        seen.append((args, kwargs))
        return op(*args, **kwargs)

    setattr(ns, op_name, record)
    try:
        flash_attn_varlen_func(t["q"], t["k"], t["v"], max_seqlen_q=t["max_seqlen_q"], cu_seqlens_q=t["cu_seqlens_q"],
                               max_seqlen_k=t["max_seqlen_k"], seqused_k=t["seqused_k"], softmax_scale=t["scale"], causal=True,
                               window_size=list(c["window"]), softcap=c["softcap"], block_table=t["block_table"], num_splits=1,
                               fa_version=fa_version)
    finally:
        setattr(ns, op_name, op)
    if len(seen) != 1 or seen[0][1]:
        raise RuntimeError(f"flash_attn_varlen_func made {len(seen)} {ns_name}.{op_name} call(s), keyword arguments "
                           f"{sorted(seen[0][1]) if seen else []}: the hidden sources wrap exactly one positional call")
    return seen[0][0]


def _call(op, args: tuple) -> tuple:
    """`op` on private copies of the tensor arguments (the ops write `out` and may write `q`), returning (out, softmax_lse)."""
    import torch

    r = op(*(a.clone() if isinstance(a, torch.Tensor) else a for a in args))
    return r[0], r[1]


def _same(x, y) -> bool:
    import torch

    def bits(t):
        return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)

    return x.shape == y.shape and x.dtype == y.dtype and bool(torch.equal(bits(x), bits(y)))


def _armed(src, args: tuple, buf, decl: dict[str, int], tag: int) -> tuple[tuple, int]:
    """The tapped launch with the stream armed for `decl`; returns ((out, lse), launches the kernel skipped)."""
    import torch

    skipped0 = int(src.ns.verity_tap_skipped())
    src.arm(buf, decl["HB"], decl["M"], decl["NB"], decl["BN"], decl["D"], tag)
    try:
        res = _call(src.tapped_op, args)
        torch.cuda.synchronize()
    finally:
        src.uninstall()
    return res, int(src.ns.verity_tap_skipped()) - skipped0


def check_case(src, fa_version: int, c: dict[str, Any], D: int, seed: int) -> dict[str, Any]:
    import torch

    geo = geometry(fa_version, c, D)
    rec = {"case": c["name"], "D": D, "H": c["H"], "Hk": c["Hk"], "seqlens_q": c["seqlens_q"], "seqlens_k": c["seqlens_k"],
           "softcap": c["softcap"], "window": c["window"], **{k: geo[k] for k in ("HB", "M", "NB", "BN", "kBM", "swapped")}}
    t = _inputs(c, D, seed)
    args = _vllm_args(fa_version, c, t)
    layout = StreamLayout(geo["HB"], geo["M"], geo["NB"], geo["BN"], D)
    buf = torch.full((layout.padded_words(),), FILL, dtype=torch.int32, device="cuda")
    out_i, lse_i = _call(src.orig_op, args)
    (out_t, lse_t), skipped = _armed(src, args, buf, {k: geo[k] for k in ("HB", "M", "NB", "BN", "D")}, 7)
    out_u, lse_u = _call(src.tapped_op, args)
    torch.cuda.synchronize()
    words = buf.cpu().numpy().view(np.uint32)
    expected = expected_written(layout, src.src, visited_rows(c, geo))
    written = words != FILL
    rec.update(words=int(layout.total_words), expected_written=int(expected.sum()), written=int(written.sum()),
               spill_outside_layout=int(np.count_nonzero(written & ~expected)), expected_but_unwritten=int(np.count_nonzero(expected & ~written)),
               skipped=skipped, out_lse_eq_installed=_same(out_t, out_i) and _same(lse_t, lse_i),
               out_lse_eq_untapped=_same(out_t, out_u) and _same(lse_t, lse_u))
    rec["ok"] = case_ok(rec)
    return rec


def check_negative(src, fa_version: int, c: dict[str, Any], D: int, decl: dict[str, int], why: str, seed: int) -> dict[str, Any]:
    import torch

    geo = geometry(fa_version, c, D)
    rec = {"case": c["name"], "D": D, "why": why, "declared": decl, "kernel": {k: geo[k] for k in ("HB", "M", "NB", "BN", "D")}}
    t = _inputs(c, D, seed)
    args = _vllm_args(fa_version, c, t)
    words = max(StreamLayout(decl["HB"], decl["M"], decl["NB"], decl["BN"], decl["D"]).padded_words(),
                StreamLayout(geo["HB"], geo["M"], geo["NB"], geo["BN"], D).padded_words())
    buf = torch.full((words,), FILL, dtype=torch.int32, device="cuda")
    out_i, _ = _call(src.orig_op, args)
    (out_w, _), skipped = _armed(src, args, buf, decl, 9)
    rec.update(skipped=skipped, buffer_untouched=bool((buf == FILL).all().item()), out_eq_installed=_same(out_w, out_i))
    rec["ok"] = negative_ok(rec)
    return rec


def _guarded(fn, base: dict[str, Any], *args) -> dict[str, Any]:
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        return {**base, "error": f"{type(e).__name__}: {e}"[:400], "ok": False}


def run(so: str, out: str, hdims: list[int] | None = None, quick: bool = False) -> dict[str, Any]:
    """Check the tap build at `so` on this GPU and write the sealed record to `<out>/fa_tap_exactness.json`."""
    import torch
    import vllm.vllm_flash_attn as vfa

    from verity_vllm.acquire.sources.hidden_source import make_hidden_source

    torch.ops.load_library(so)
    fa_version = 3 if hasattr(getattr(torch.ops, Path(so).stem), "verity_set_tap_v9") else 2
    installed = glob.glob(os.path.join(os.path.dirname(vfa.__file__), f"{INSTALLED[fa_version][0]}*.so"))
    if installed:
        torch.ops.load_library(installed[0])
    src = make_hidden_source(None, so, flash_attn_version=fa_version)
    layout = {2: 8, 3: 9}[fa_version]
    if src.stats["tap_layout"] != layout:
        raise SystemExit(f"{so}: tap layout v{src.stats['tap_layout']}; the declared-tile check needs v{layout}")
    doc: dict[str, Any] = {
        "schema": SCHEMA, "backend": f"fa{fa_version}", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv,
        "environment": _environment(),
        "tap": {"so": so, "sha256": _sha256(so), "ns": src.ns_name, "mode": src.mode, "src": src.src, "layout": src.stats["tap_layout"],
                "hdims": src.stats["hdims"]},
        "installed": {"op": ".".join(INSTALLED[fa_version]), "so": installed[0] if installed else None,
                      "sha256": _sha256(installed[0]) if installed else None},
        "cases": [], "negatives": []}
    for D in hdims or src.stats["hdims"]:
        for i, c in enumerate(cases(fa_version, quick)):
            rec = _guarded(check_case, {"case": c["name"], "D": D}, src, fa_version, c, D, 1000 * D + i)
            doc["cases"].append(rec)
            print(f"[fa-tap] fa{fa_version} hd{D} {c['name']:30s} " + (rec.get("error") or
                  f"tile {rec['kBM']}x{rec['BN']} HB={rec['HB']} M={rec['M']} NB={rec['NB']} skipped={rec['skipped']} "
                  f"==installed {rec['out_lse_eq_installed']} ==untapped {rec['out_lse_eq_untapped']} spill={rec['spill_outside_layout']} "
                  f"unwritten={rec['expected_but_unwritten']}") + f" -> {'OK' if rec['ok'] else 'FAIL'}", flush=True)
        for j, (c, decl, why) in enumerate(negatives(fa_version, D)):
            rec = _guarded(check_negative, {"case": c["name"], "D": D, "why": why}, src, fa_version, c, D, decl, why, 7 * D + j)
            doc["negatives"].append(rec)
            print(f"[fa-tap] fa{fa_version} hd{D} negative {c['name']} ({why}): " + (rec.get("error") or
                  f"skipped={rec['skipped']} untouched={rec['buffer_untouched']} ==installed {rec['out_eq_installed']}")
                  + f" -> {'OK' if rec['ok'] else 'FAIL'}", flush=True)
    doc = seal(doc)
    path = Path(out) / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"FA-TAP-EXACTNESS {'OK' if doc['ok'] else 'FAIL'} {doc['digest']} {path}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FA-tap exactness: the tap build computes the installed vLLM FlashAttention's result")
    ap.add_argument("--so", help="the tap build: verity_fa2_matReq.so (pod_fa2_tap.sh) or verity_fa3_matReq.so (pod_fa3_tap.sh)")
    ap.add_argument("--out", help=f"directory for {RECORD}")
    ap.add_argument("--hdims", default=None, help="comma list of head dims (default: every head dim the build compiles)")
    ap.add_argument("--quick", action="store_true", help=f"only the cases {', '.join(QUICK)}")
    ap.add_argument("--verify", metavar="RECORD", help="check a record's digest and verdict (no GPU)")
    a = ap.parse_args(argv)
    if a.verify:
        doc = json.loads(Path(a.verify).read_text())
        problems = verify_record(doc)
        print(f"{a.verify}: {'intact, ok=' + str(doc.get('ok')) if not problems else '; '.join(problems)} (digest {doc.get('digest')})")
        return 1 if problems or not doc.get("ok") else 0
    if not (a.so and a.out):
        ap.error("--so and --out are required (or --verify RECORD)")
    doc = run(a.so, a.out, [int(h) for h in a.hdims.split(",")] if a.hdims else None, a.quick)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
