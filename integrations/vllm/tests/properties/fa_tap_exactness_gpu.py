"""GPU run of the FA-tap exactness property (`verity_vllm.properties.fa_tap_exactness`): check a tap build on this GPU and write the sealed
record `<out>/fa_tap_exactness.json`.  Exit status 1 unless ok; `--verify RECORD` re-derives a record's digest and verdict without a GPU.

The launch arguments are the ones `vllm.vllm_flash_attn.flash_attn_varlen_func` passes to the installed op (recorded at the op), so the
check follows the installed interface.  It lives under tests/ because it seeds a random generator, wraps the installed op and parses
arguments, which library code may not do yet (lints P3, P9, P6).

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as gate (b)
    $PY tests/properties/fa_tap_exactness_gpu.py --so /workspace/cp/fa2/build/matReq/verity_fa2_matReq.so --out DIR          # FA2 (L40S)
    $PY tests/properties/fa_tap_exactness_gpu.py --so /workspace/cp/fa2/build/fa3_matReq/verity_fa3_matReq.so --out DIR      # FA3 (H100)

A guarded-max build (`verity_tap_row3() == 1`, and its MS class with `verity_tap_src() & 64`) is checked against the record of the
default build on the same GPU, `--baseline`:

    $PY tests/properties/fa_tap_exactness_gpu.py --so /workspace/cp/fa2/build/matReqG/verity_fa2_matReqG.so --baseline DIR/fa_tap_exactness.json --out DIR_G

An FA3 build is checked under FA3's per-iteration `Check_inf` chain with `--construction check-inf-per-iteration` (property (g)): the MS
words against `AttnBlock_v4` and every output row against `Attention_v4` on the launch's inputs.  `--against-decode` runs the prefill
cases only and also checks every row against the decode-shaped Definitions (`Attention_v4` at the one-row decode's tile and
`MASKED_FROM`, which the window circuit binds at prompt positions too), ending in a `FA3-PREFILL-VS-DECODE EXACT|INEXACT` line:

    $PY tests/properties/fa_tap_exactness_gpu.py --so .../verity_fa3_matReq.so --hdims 64 --construction check-inf-per-iteration --against-decode --out DIR
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

from verity_vllm.properties.fa_tap_exactness import (CONSTRUCTIONS, FILL, INSTALLED, PAGE, QUICK, RECORD, SCHEMA, case_ok, cases, check_ms,
                                                check_row3, decode_shaped, decode_tile, expected_written, geometry, masked_digest,
                                                masked_from, negative_ok, negatives, prefill_cases, scale_log2_bits, seal, verify_record,
                                                visited_rows)
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
    if c.get("special"):                     # k[..., 0] < 0 everywhere: q[t, :, 0] = +inf / -inf / NaN makes every score of t -inf / +inf / NaN
        k[..., 0] = -(k[..., 0].abs() + 0.5)
        for kind, toks in c["special"].items():
            q[toks, :, 0] = {"-inf": float("inf"), "+inf": float("-inf"), "nan": float("nan")}[kind]
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


def _words(x) -> np.ndarray:
    """A bf16 tensor's words as uint16 (CPU)."""
    import torch

    return x.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def out_vs_ir(c: dict[str, Any], D: int, t: dict[str, Any], out, geo: dict[str, Any], decode: bool = False) -> dict[str, Any]:
    """Every output row of an FA3 case against `Attention_v4` on the launch's inputs: per query token, each head's query over the T keys
    its row sees (paged K/V gathered through the block table), block b guarded iff b >= the row's tile's `n_block_min_causal_local_mask`.
    The twin (`derived_rows.attention_head_states`) evaluates the heads it covers; a head with a non-finite operand or accumulator goes to
    the reference evaluator (`AttentionHead_v4`, gate by gate).  `decode`: each row against the decode-shaped Definition instead, the
    one-row decode's tile (`decode_tile`) and `MASKED_FROM` (its row 0's).  Mismatched rows are counted, those of a token whose every score
    is special (`SPECIAL`) apart, with the fewest keys a mismatched row sees."""
    from verity.ml.circuits.evaluation.reference import evaluate
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.numerics import derived_rows as DR
    from verity.ml.definitions.registry import fa3_check_inf as F3
    from verity.ml.definitions.registry import attention_vocab
    from verity.ml.definitions.registry import hopper

    q, k, v, o = _words(t["q"]), _words(t["k"]), _words(t["v"]), _words(out)
    bt = t["block_table"].cpu().numpy()
    H, Hk = c["H"], c["Hk"]
    g = H // Hk
    kBM, BN = decode_tile(D, g) if decode else (geo["kBM"], geo["BN"])
    kvh = np.arange(H) // g
    special = {tk for toks in (c.get("special") or {}).values() for tk in toks}
    dot16, inv = DR.TC_DOT16_BY_PRIM["HopperBF16WgmmaDot16_v1"], DR.INV_SUM_BY_PRIM["Fa3InvSum_v1"]
    rec: dict[str, Any] = {"rows": 0, "heads": 0, "mismatch": 0, "mismatch_rows": 0, "special_mismatch_rows": 0, "min_T_mismatch": None,
                           "reference_heads": 0, "unevaluated": 0, **({"tile": [kBM, BN]} if decode else {})}      # mismatch: heads
    tok = 0
    for seq, (sq, sk) in enumerate(zip(c["seqlens_q"], c["seqlens_k"])):
        keys = np.arange(sk)
        kseq, vseq = k[bt[seq, keys // PAGE], keys % PAGE], v[bt[seq, keys // PAGE], keys % PAGE]      # (sk, Hk, D)
        for r in range(sq):
            T = r + 1 + sk - sq
            mf = attention_vocab.fa3_masked_from(T, BN, 0 if decode else r, kBM, g)
            kk = np.ascontiguousarray(np.transpose(kseq[:T], (1, 0, 2))[kvh])                            # (H, T, D)
            vv = np.ascontiguousarray(np.transpose(vseq[:T], (1, 0, 2))[kvh])
            st = DR.attention_head_states(q[tok + r], kk, vv, BN, dot16=dot16, inv=inv, masked_from=mf)
            want = np.asarray(st["out"], dtype=np.uint16)
            for h in np.flatnonzero(st["nonfinite"]):
                spec = bind(F3.AttentionHeadV4, T=T, D=D, BN=BN, DOT=hopper.HopperBF16WgmmaDot16, INV=hopper.Fa3InvSum, MASKED_FROM=mf)
                try:
                    want[h] = np.asarray(evaluate(spec, q[tok + r, h].tolist(), kk[h].reshape(-1).tolist(), vv[h].reshape(-1).tolist()),
                                         dtype=np.uint16)
                    rec["reference_heads"] += 1
                except Exception as e:  # noqa: BLE001
                    rec["unevaluated"] += 1
                    rec.setdefault("first_unevaluated", {"token": tok + r, "head": int(h), "error": f"{type(e).__name__}: {e}"[:200]})
            bad = np.flatnonzero((want != o[tok + r]).any(axis=1))
            rec["rows"] += 1
            rec["heads"] += H
            rec["mismatch"] += int(bad.size)
            if bad.size:
                rec["mismatch_rows"] += 1
                rec["special_mismatch_rows"] += int(tok + r in special)
                rec["min_T_mismatch"] = T if rec["min_T_mismatch"] is None else min(rec["min_T_mismatch"], T)
            if bad.size and "first_mismatch" not in rec:
                h = int(bad[0])
                rec["first_mismatch"] = {"token": tok + r, "head": h, "T": T, "masked_from": mf,
                                         "ir": [int(x) for x in want[h][:4]], "kernel": [int(x) for x in o[tok + r, h][:4]]}
        tok += sq
    return rec


def check_case(src, fa_version: int, c: dict[str, Any], D: int, seed: int, baseline: dict[tuple[str, int], str] | None = None,
               construction: str = CONSTRUCTIONS[0], against_decode: bool = False) -> dict[str, Any]:
    """One case at head dim `D`; a guarded-max build (`src.stats["row3"]`) also checks ROW word 3 and compares the rest of the six classes
    with the default build's (`baseline`: (case, D) -> its `stream_sha256_masked`); a build with the MS class (`src.ms`) checks it;
    `against_decode` also checks every output row against the decode-shaped Definitions (`out_vs_decode`)."""
    import torch

    geo = geometry(fa_version, c, D)
    row3 = int(src.stats.get("row3") or 0)
    rec = {"case": c["name"], "D": D, "H": c["H"], "Hk": c["Hk"], "seqlens_q": c["seqlens_q"], "seqlens_k": c["seqlens_k"],
           "softcap": c["softcap"], "window": c["window"], **({"special": c["special"]} if c.get("special") else {}),
           **{k: geo[k] for k in ("HB", "M", "NB", "BN", "kBM", "swapped")}}
    t = _inputs(c, D, seed)
    args = _vllm_args(fa_version, c, t)
    layout = StreamLayout(geo["HB"], geo["M"], geo["NB"], geo["BN"], D, ms=src.ms)
    buf = torch.full((layout.padded_words(),), FILL, dtype=torch.int32, device="cuda")
    out_i, lse_i = _call(src.orig_op, args)
    (out_t, lse_t), skipped = _armed(src, args, buf, {k: geo[k] for k in ("HB", "M", "NB", "BN", "D")}, 7)
    out_u, lse_u = _call(src.tapped_op, args)
    torch.cuda.synchronize()
    words = buf.cpu().numpy().view(np.uint32)
    expected = expected_written(layout, src.src, visited_rows(c, geo), row3=row3)
    written = words != FILL
    rec.update(words=int(layout.total_words), expected_written=int(expected.sum()), written=int(written.sum()),
               spill_outside_layout=int(np.count_nonzero(written & ~expected)), expected_but_unwritten=int(np.count_nonzero(expected & ~written)),
               skipped=skipped, out_lse_eq_installed=_same(out_t, out_i) and _same(lse_t, lse_i),
               out_lse_eq_untapped=_same(out_t, out_u) and _same(lse_t, lse_u), stream_sha256_masked=masked_digest(words, layout))
    if row3:
        mine = (baseline or {}).get((c["name"], D))
        rec["row3"] = {**check_row3(words, layout, visited_rows(c, geo)),
                       "stream_eq_baseline": None if mine is None else mine == rec["stream_sha256_masked"]}
    per_iteration = fa_version == 3 and construction == "check-inf-per-iteration"
    if src.ms:
        rec["ms"] = check_ms(words, layout, visited_rows(c, geo), scale_log2_bits(c, D), masked_from=masked_from(c, geo) if per_iteration else None)
    if per_iteration:
        rec["out_vs_ir"] = out_vs_ir(c, D, t, out_t, geo)
    if against_decode:
        rec["out_vs_decode"] = out_vs_ir(c, D, t, out_t, geo, decode=True)
    rec["ok"] = case_ok(rec)
    return rec


def check_negative(src, fa_version: int, c: dict[str, Any], D: int, decl: dict[str, int], why: str, seed: int) -> dict[str, Any]:
    import torch

    geo = geometry(fa_version, c, D)
    rec = {"case": c["name"], "D": D, "why": why, "declared": decl, "kernel": {k: geo[k] for k in ("HB", "M", "NB", "BN", "D")}}
    t = _inputs(c, D, seed)
    args = _vllm_args(fa_version, c, t)
    words = max(StreamLayout(decl["HB"], decl["M"], decl["NB"], decl["BN"], decl["D"], ms=src.ms).padded_words(),
                StreamLayout(geo["HB"], geo["M"], geo["NB"], geo["BN"], D, ms=src.ms).padded_words())
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


def _baseline(path: str | None, fa_version: int, env: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[tuple[str, int], str]]:
    """The default build's record a guarded-max build is compared with: intact, ok, the same backend on the same device capability."""
    if not path:
        return None, {}
    doc = json.loads(Path(path).read_text())
    problems = verify_record(doc)
    if problems or not doc.get("ok") or (doc.get("tap") or {}).get("row3") or doc.get("backend") != f"fa{fa_version}" \
            or (doc.get("environment") or {}).get("capability") != env["capability"]:
        raise SystemExit(f"--baseline {path}: not an intact ok default-build fa{fa_version} record of this device capability ({problems[:3]})")
    ref = {"record": path, "digest": doc["digest"], "so_sha256": (doc.get("tap") or {}).get("sha256")}
    return ref, {(e["case"], int(e["D"])): e.get("stream_sha256_masked") for e in doc.get("cases") or []}


def run(so: str, out: str, hdims: list[int] | None = None, quick: bool = False, baseline: str | None = None,
        construction: str = CONSTRUCTIONS[0], against_decode: bool = False) -> dict[str, Any]:
    """Check the tap build at `so` on this GPU (a guarded-max build against the default build's record `baseline`) and write the sealed
    record to `<out>/fa_tap_exactness.json`.  `against_decode` (FA3, `check-inf-per-iteration`): the prefill cases only
    (`prefill_cases`), each row also against the decode-shaped Definitions, and a `FA3-PREFILL-VS-DECODE` line."""
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
    env = _environment()
    row3 = int(src.stats.get("row3") or 0)
    if row3 and not baseline:
        raise SystemExit(f"{so} is a guarded-max build (verity_tap_row3 {row3}): give --baseline <the default build's record>")
    base_ref, base = _baseline(baseline, fa_version, env)
    if construction != CONSTRUCTIONS[0] and fa_version != 3:
        raise SystemExit(f"--construction {construction}: only an FA3 build has another attention construction")
    if against_decode and construction != "check-inf-per-iteration":
        raise SystemExit("--against-decode: the decode-shaped Definitions are Attention_v4's, so give an FA3 build and "
                         "--construction check-inf-per-iteration")
    doc: dict[str, Any] = {
        "schema": SCHEMA, "backend": f"fa{fa_version}", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv,
        "environment": env,
        "tap": {"so": so, "sha256": _sha256(so), "ns": src.ns_name, "mode": src.mode, "src": src.src, "layout": src.stats["tap_layout"],
                "hdims": src.stats["hdims"], **({"row3": row3} if row3 else {})},
        "installed": {"op": ".".join(INSTALLED[fa_version]), "so": installed[0] if installed else None,
                      "sha256": _sha256(installed[0]) if installed else None},
        **({"baseline": base_ref} if base_ref else {}), **({"construction": construction} if construction != CONSTRUCTIONS[0] else {}),
        **({"against_decode": True} if against_decode else {}), "cases": [], "negatives": []}
    for D in hdims or src.stats["hdims"]:
        for i, c in enumerate(prefill_cases(quick) if against_decode else cases(fa_version, quick)):
            rec = _guarded(check_case, {"case": c["name"], "D": D}, src, fa_version, c, D, 1000 * D + i, base, construction, against_decode)
            doc["cases"].append(rec)
            g, ms, oi, od = rec.get("row3"), rec.get("ms"), rec.get("out_vs_ir"), rec.get("out_vs_decode")
            print(f"[fa-tap] fa{fa_version} hd{D} {c['name']:30s} " + (rec.get("error") or
                  f"tile {rec['kBM']}x{rec['BN']} HB={rec['HB']} M={rec['M']} NB={rec['NB']} skipped={rec['skipped']} "
                  f"==installed {rec['out_lse_eq_installed']} ==untapped {rec['out_lse_eq_untapped']} spill={rec['spill_outside_layout']} "
                  f"unwritten={rec['expected_but_unwritten']}" + (f" row3 guards={g['guard_words']} !=guard {g['guard_mismatch']} first!=0 "
                  f"{g['first_block_nonzero']} !=ir {g['ir_mismatch']}/{g['ir_sampled']} ==baseline {g['stream_eq_baseline']}" if g else "")
                  + (f" ms words={ms['ms_words']} !=ir-twin {ms['ms_mismatch']} (row_max -inf {ms['ms_mismatch_neg_inf_max']}) !=ir "
                     f"{ms['ir_mismatch']}/{ms['ir_sampled']}" + (f" unguarded {ms['unguarded_words']}" if "unguarded_words" in ms else "") if ms else "")
                  + (f" out-vs-IR rows={oi['rows']} !={oi['mismatch']} ref-heads={oi['reference_heads']} uneval={oi['unevaluated']}" if oi else "")
                  + (f" vs-decode {od['tile'][0]}x{od['tile'][1]} rows={od['rows']} !={od['mismatch']} heads in {od['mismatch_rows']} rows "
                     f"(special {od['special_mismatch_rows']}) fewest keys {od['min_T_mismatch']} uneval={od['unevaluated']}" if od else ""))
                  + f" -> {'OK' if rec['ok'] else 'FAIL'}", flush=True)
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
    if against_decode:
        s = decode_shaped(doc)
        print(f"FA3-PREFILL-VS-DECODE {'EXACT' if s['exact'] else 'INEXACT'} rows={s['rows']} heads={s['heads']} mismatched heads={s['mismatch']} "
              f"in rows={s['mismatch_rows']} (special-score rows {s['special_mismatch_rows']}) fewest keys mismatched={s['min_T_mismatch']} "
              f"uneval={s['unevaluated']} cases={s['cases']} record {doc['digest']}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FA-tap exactness: the tap build computes the installed vLLM FlashAttention's result")
    ap.add_argument("--so", help="the tap build: verity_fa2_matReq.so (pod_fa2_tap.sh) or verity_fa3_matReq.so (pod_fa3_tap.sh)")
    ap.add_argument("--out", help=f"directory for {RECORD}")
    ap.add_argument("--hdims", default=None, help="comma list of head dims (default: every head dim the build compiles)")
    ap.add_argument("--quick", action="store_true", help=f"only the cases {', '.join(QUICK)}")
    ap.add_argument("--baseline", metavar="RECORD", default=None,
                    help="a guarded-max build: the default build's record on this GPU, whose six classes (but ROW word 3) it must equal")
    ap.add_argument("--construction", choices=CONSTRUCTIONS, default=CONSTRUCTIONS[0],
                    help="FA3: the attention construction the MS words and outputs are checked against (check-inf-per-iteration: Attention_v4)")
    ap.add_argument("--against-decode", action="store_true",
                    help="FA3, check-inf-per-iteration: only the prefill cases, every row also against the decode-shaped Definitions "
                         "(Attention_v4 at the one-row decode's tile and MASKED_FROM)")
    ap.add_argument("--verify", metavar="RECORD", help="check a record's digest and verdict (no GPU)")
    a = ap.parse_args(argv)
    if a.verify:
        doc = json.loads(Path(a.verify).read_text())
        problems = verify_record(doc)
        print(f"{a.verify}: {'intact, ok=' + str(doc.get('ok')) if not problems else '; '.join(problems)} (digest {doc.get('digest')})")
        return 1 if problems or not doc.get("ok") else 0
    if not (a.so and a.out):
        ap.error("--so and --out are required (or --verify RECORD)")
    doc = run(a.so, a.out, [int(h) for h in a.hdims.split(",")] if a.hdims else None, a.quick, a.baseline, a.construction,
              a.against_decode)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
