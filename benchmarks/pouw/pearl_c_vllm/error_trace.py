"""An untimed per-linear error trace of Llama-3.1-8B-Instruct in vLLM on the RTX PRO 6000: for every linear, on the same
inputs, the output error of stock vLLM BF16, stock vLLM FP8 and Pearl-C v1 as served against the float product x·Wᵀ, and
Pearl-C's error split exactly into its parts.

  python benchmarks/pouw/pearl_c_vllm/error_trace.py --model DIR --kernel K --cubin C --out OUT/trace.json [--chunk-min 6]

``window.sh`` with ``MODE=trace`` runs it (untimed, one GPU, as node-2 fill: exit 99 while linears remain).

- **The inputs:** the quality run's first WikiText-2 window (BOS and 2,047 tokens, `e2e.wikitext_windows`), one prefill in the
  arm engine with the wrapper off, so every linear's x (m = 2,048 rows) is the BF16 model's own.  Forward pre-hooks store
  them under OUT/scratch.
- **The reference:** the FP32 product of the BF16 words, TF32 off.
- **bf16:** the layer's stock apply (``UnquantizedLinearMethod``).
- **fp8:** stock vLLM ``quantization="fp8"`` (``Fp8PerTensorOnlineLinearMethod``: a per-tensor weight scale, and a per-token
  dynamic activation scale where CUTLASS runs), exactly: the FP8 engine, in a child process, applies each of its own linears'
  ``quant_method.apply`` to the captured inputs.  Its first linear's input from its own forward must equal the captured one.
- **pearlc:** `PearlCDeviceRun.gemm` as served (v1, the fused chain).
- **Pearl-C's split** (`split`), exact in FP64: with A′ and B̃ the codes the forming wrote (read in column order: form_s5
  stores them in K32 order, `natural`), N_A = E′_A·F_Aᵀ and N_B = E′_B·F_Bᵀ the injected noise from its own factors, and
  s = 1/(α_A ⊗ α_B):
  - ``cast_noise`` = (A′ − N_A)(B̃ − N_B)ᵀ·s − x·Wᵀ: the E4M3 cast of the noisy operands, everything after it exact;
  - ``chain`` = (C̃ − A′B̃ᵀ)·s: the accumulator path (the unfused ``gemm_pearlc`` on the same buffers stores C̃ and U, and its
    y must equal the served y bit for bit);
  - ``peel`` = ((U − C̃) − ((A′ − N_A)(B̃ − N_B)ᵀ − A′B̃ᵀ))·s: the clean-up's BF16 factors and its accumulation;
  - ``out`` = y − U·s: the glue's rounding to BF16.
  They sum to y − x·Wᵀ exactly.
- **The cast, further:** ``adopt`` is Pearl-C's own quantizer with no noise, e4m3(α·x) on both sides (GPU 1's ``quant_s5``),
  multiplied exactly (``adopt_dev``: ``gemm_plain``'s y on the card).  Each operand alone, with and without the noise
  (``a_clean``, ``a_noise``, ``b_clean``, ``b_noise``).  ``forming``: the fraction of codes the atom wrote that differ from
  e4m3 of the exact α·x + E′F.
- **The noise against its law:** RMS(N_A)/(α·ρ) per row (β's definition gives 1), and the same for B's rows.
- **The expected size** (`noise_gain`): with E4M3's rounding error proportional to the value it rounds, the noise raises A's
  cast error variance by Σ_il (x² + (N_A/α)²)·w2_l / Σ_il x²·w2_l, w2_l = Σ_j W_jl², and B's likewise; so
  cast_noise² ≈ a_clean²·gain_A + b_clean²·gain_B.

Every error is a relative RMS against x·Wᵀ, over all rows and without row 0 (BOS, whose massive activations dominate its
scale), with row quantiles.  Results go to OUT/trace-parts.jsonl a linear at a time, and a chunk restarts after the last.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

SCHEME = "pearl-c-sm120-v1-h1"
#: a fixed beacon, so every chunk's install draws the same salt and weight noise
BEACON = hashlib.sha256(b"pearl-c-vllm error trace").digest()
METHODS = ("bf16", "fp8", "adopt", "adopt_dev", "cast_noise", "pearlc")
PARTS = ("cast_noise", "chain", "peel", "out")
ONE_SIDED = ("a_clean", "a_noise", "b_clean", "b_noise")


def linears(model) -> list[tuple[str, object]]:
    """Every LinearBase in `e2e.install`'s order (``model.modules()``), with its name."""
    from vllm.model_executor.layers.linear import LinearBase
    return [(name, m) for name, m in model.named_modules() if isinstance(m, LinearBase)]


def kind(name: str) -> str:
    return name.rsplit(".", 1)[-1]


# ---- the arithmetic (pure torch, CPU-testable) -------------------------------------------------------------------------

def rel(e, ref) -> float:
    return float(e.double().norm() / ref.double().norm())


def rows(e, ref) -> dict:
    """Relative RMS over all rows and without row 0, row 0's own, and the row quantiles."""
    import torch
    r = (e.double().norm(dim=1) / ref.double().norm(dim=1).clamp_min(1e-300)).float()
    q = torch.quantile(r, torch.tensor([0.5, 0.99], dtype=torch.float32, device=r.device)).tolist()
    return {"all": rel(e, ref), "no_bos": rel(e[1:], ref[1:]), "bos": float(r[0]), "row_p50": q[0], "row_p99": q[1],
            "row_max": float(r.max())}


#: form_s5's K32 order (pearl_c_sm120.cu): within each aligned 32 of k, byte 8q + 2b + s holds column 8b + 2q + s.  It is
#: its own inverse, and a product over k of two operands in it is unchanged.
K32 = [8 * b + 2 * q + s for q in range(4) for b in range(4) for s in range(2)]


def natural(codes):
    """Codes in K32 order (form_s5's A′ and B̃) in column order, or the reverse."""
    return codes.reshape(*codes.shape[:-1], -1, 32)[..., K32].reshape(codes.shape)


def split(ref, y, c, u, at, bt, epa, epb, fa, fb, ia_a, ia_b) -> dict:
    """Pearl-C's y − ref as its four exact parts (this module's docstring), in FP64: {part: tensor}, and the clean operands.
    `at` and `bt` are form_s5's codes, in K32 order; the noise and the reference are in column order."""
    import torch
    f = torch.float64
    a1, b1 = natural(at.to(f)), natural(bt.to(f))
    na, nb = epa.to(f) @ fa.to(f).T, epb.to(f) @ fb.to(f).T
    ca, cb = a1 - na, b1 - nb
    s = ia_a.to(f)[:, None] * ia_b.to(f)[None, :]
    exact = a1 @ b1.T
    clean = ca @ cb.T
    ct, ut = c.to(f), u.to(f)
    parts = {"cast_noise": clean * s - ref.to(f), "chain": (ct - exact) * s, "peel": ((ut - ct) - (clean - exact)) * s,
             "out": y.to(f) - ut * s}
    return {"parts": parts, "total": y.to(f) - ref.to(f), "na": na, "nb": nb, "ca": ca, "cb": cb}


def noise_gain(x, w, na_x, nb_w) -> dict:
    """The factor by which the noise raises each operand's cast error variance when E4M3's error is proportional to the
    value it rounds: `na_x` and `nb_w` are the noise in x's and W's units (N/α)."""
    x2, w2 = x.double() ** 2, w.double() ** 2
    wl, xl = w2.sum(0), x2.sum(0)
    return {"a": float(((x2 + na_x.double() ** 2) @ wl).sum() / (x2 @ wl).sum()),
            "b": float(((w2 + nb_w.double() ** 2) @ xl).sum() / (w2 @ xl).sum())}


def noise_law(x, na_x) -> dict:
    """RMS(N/α) per row over ρ (every 8th position's RMS, as ``stats_s5`` takes it): 1 by β's definition."""
    import torch
    rho = (x.double()[:, ::8] ** 2).mean(1).sqrt()
    r = (na_x.double() ** 2).mean(1).sqrt() / rho.clamp_min(1e-300)
    q = torch.quantile(r.float(), torch.tensor([0.01, 0.5, 0.99], device=r.device)).tolist()
    s = x.double().abs().amax(1) / rho.clamp_min(1e-300)
    return {"p01": q[0], "p50": q[1], "p99": q[2], "max_over_rho_p50": float(s.median()), "max_over_rho_max": float(s.max())}


def e4m3(v):
    """Saturating nearest-even E4M3, as the forming's cast."""
    import torch
    return v.clamp(-448.0, 448.0).to(torch.float32).to(torch.float8_e4m3fn)


def summarize(parts: list[dict]) -> dict:
    """Over linears: each method's RMS of the per-linear relative errors, by kind and in all; the Pearl-C/FP8 ratio; each
    part's mean share of Pearl-C's error variance; the expected cast_noise against the measured one."""
    import math
    import statistics

    def rms(xs):
        return math.sqrt(sum(v * v for v in xs) / len(xs)) if xs else None

    def scope(sel: list[dict]) -> dict:
        d = {}
        for key in ("all", "no_bos"):
            d[key] = {m: rms([p["error"][m][key] for p in sel]) for m in METHODS}
            d[key].update({m: rms([p["one_sided"][m][key] for p in sel]) for m in ONE_SIDED})
        ratio = [p["error"]["pearlc"]["no_bos"] / p["error"]["fp8"]["no_bos"] for p in sel]
        d["pearlc_over_fp8_no_bos"] = {"median": statistics.median(ratio), "min": min(ratio), "max": max(ratio)}
        d["variance_share"] = {x: statistics.fmean(p["share"][x] for p in sel) for x in PARTS}
        d["cast_noise_expected_over_measured"] = statistics.median(p["expected"]["cast_noise_over_measured"] for p in sel)
        d["noise_law_a_p50"] = statistics.median(p["noise_law"]["a"]["p50"] for p in sel)
        d["noise_law_b_p50"] = statistics.median(p["noise_law"]["b"]["p50"] for p in sel)
        return d

    out: dict = {"linears": len(parts), "all": scope(parts),
                 "by_kind": {k: scope([p for p in parts if p["kind"] == k]) for k in sorted({p["kind"] for p in parts})}}
    worst = sorted(parts, key=lambda p: -p["error"]["pearlc"]["no_bos"] / p["error"]["fp8"]["no_bos"])[:5]
    out["worst_ratio"] = [{"w": p["w"], "name": p["name"], "pearlc": p["error"]["pearlc"]["no_bos"],
                           "fp8": p["error"]["fp8"]["no_bos"]} for p in worst]
    return out


# ---- the stock FP8 engine, in a child process ----------------------------------------------------------------------------

def fp8_child(conn, args, scratch: str, names: list[str], window: list[int], count: int) -> None:
    """The stock FP8 engine: its first linear's input on `window` against the captured one, then each linear's own
    ``quant_method.apply`` on the captured inputs, into scratch/fp8-<w>.pt."""
    try:
        import e2e
        import torch
        from vllm import SamplingParams
        from vllm_bench import find_model
        llm = e2e.engine(args, "base")
        layers = linears(find_model(llm))
        if [n for n, _ in layers] != names:
            raise ValueError("the FP8 engine's linears are not the BF16 engine's")
        seen: dict = {}

        def first(_m, a):
            seen.setdefault("x", a[0].detach().reshape(-1, a[0].shape[-1]).cpu())

        hook = layers[0][1].register_forward_pre_hook(first)
        llm.generate([{"prompt_token_ids": window}], SamplingParams(temperature=0.0, max_tokens=1), use_tqdm=False)
        hook.remove()
        d = Path(scratch)
        x0 = torch.load(d / "x-0.pt")
        with torch.inference_mode():
            for w, (_, layer) in enumerate(layers[:count]):
                y = layer.quant_method.apply(layer, torch.load(d / f"x-{w}.pt").cuda(), None)
                torch.save(y.cpu(), d / f"fp8-{w}.pt")
        qm = layers[0][1].quant_method
        conn.send({"same_first_input": bool(torch.equal(seen["x"], x0)),
                   "first_input_max_abs_diff": float((seen["x"].float() - x0.float()).abs().max()),
                   "method": type(qm).__name__, "kernel": type(getattr(qm, "fp8_linear", None)).__name__,
                   "activation_quant": str(getattr(qm, "activation_quant_key", None))})
    except Exception as ex:  # noqa: BLE001 - the parent reports it
        conn.send({"error": f"fp8: {type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-4000:]})


# ---- one linear ------------------------------------------------------------------------------------------------------

def trace_linear(run, kernel, w: int, name: str, layer, x, weight, y_fp8) -> dict:
    """Every method's error on one linear, and Pearl-C's split (this module's docstring)."""
    import torch
    from verity_vllm.protocol_options import pouw_pearl_c_device as PPD
    f8 = torch.float8_e4m3fn
    m, k = x.shape
    n = weight.shape[0]
    xf, wf = x.float(), weight.float()
    ref = xf @ wf.T
    with torch.inference_mode():
        y_bf16 = type(layer.quant_method).apply(layer.quant_method, layer, x, None)
        y_pc = run.gemm(w, x, weight, None)
    sh, wt = run.shapes[(m, n, k)], run.weights[w]
    dev = x.device
    own = {b: torch.zeros(size, dtype=torch.uint8, device=dev)
           for b, size in (("c", m * n * 4), ("u", m * n * 4), ("y", m * n * 2), ("qa_s5", m * k), ("qb_s5", n * k),
                           ("y_adopt", m * n * 2))}
    bufs = {**{b: t.data_ptr() for b, t in wt.b.items()}, **{b: t.data_ptr() for b, t in sh.b.items()},
            **{b: t.data_ptr() for b, t in own.items()}, "line_a": 0, "leaves": 0, "digests": 0}
    xb = wf.contiguous()
    keys = {"dom_a": 0, "dom_b": 0, "dom_t": 0, "line_b": 0, "k_msg": 0, "k_leaf": 0, "unit": 0}
    pipe = kernel.run.Pipeline(kernel.dev, sh.mp, n, k, run.variant, keys, sh.xa.data_ptr(), xb.data_ptr(), bufs=bufs,
                               m_real=m, fused=False)
    try:
        PPD._run(kernel, pipe, ("gemm_pearlc", "quant_b", "quant_a", "gemm_adopt"), PPD._stream(dev))
        torch.cuda.synchronize(dev)
    finally:
        pipe.free()
    c = own["c"].view(torch.float32).view(m, n)
    u = own["u"].view(torch.float32).view(m, n)
    y2 = own["y"].view(torch.bfloat16).view(m, n)
    at = sh.b["at"][:m * k].view(f8).view(m, k)
    bt = wt.b["bt"][:n * k].view(f8).view(n, k)
    epa = sh.b["epa_codes"][:m * 32].view(f8).view(m, 32)
    epb = wt.b["epb_codes"][:n * 32].view(f8).view(n, 32)
    fa = wt.b["fa_codes"][:k * 32].view(f8).view(k, 32)
    fb = wt.b["fb_codes"][:k * 32].view(f8).view(k, 32)
    ia_a = sh.b["ia_a"][:4 * m].view(torch.float32)
    ia_b = wt.b["ia_b"][:4 * n].view(torch.float32)
    al_a = sh.b["alpha_a"][:4 * m].view(torch.float32)
    al_b = wt.b["alpha_b"][:4 * n].view(torch.float32)
    qa = own["qa_s5"].view(f8).view(m, k).float()
    qb = own["qb_s5"].view(f8).view(n, k).float()

    sp = split(ref, y_pc, c, u, at, bt, epa, epb, fa, fb, ia_a, ia_b)
    total = sp["total"]
    identity = float((total - sum(sp["parts"].values())).abs().max() / ref.double().abs().max())
    s = ia_a[:, None] * ia_b[None, :]
    na_x, nb_w = (sp["na"] * ia_a.double()[:, None]).float(), (sp["nb"] * ia_b.double()[:, None]).float()
    ca_x, cb_w = (sp["ca"] * ia_a.double()[:, None]).float(), (sp["cb"] * ia_b.double()[:, None]).float()
    errors = {"bf16": rows(y_bf16.float() - ref, ref), "fp8": rows(y_fp8.float() - ref, ref),
              "adopt": rows((qa @ qb.T) * s - ref, ref), "adopt_dev": rows(own["y_adopt"].view(torch.bfloat16).view(m, n).float() - ref, ref),
              "cast_noise": rows(sp["parts"]["cast_noise"], ref), "pearlc": rows(total, ref)}
    one = {"a_clean": rows((qa * ia_a[:, None]) @ wf.T - ref, ref), "a_noise": rows(ca_x @ wf.T - ref, ref),
           "b_clean": rows(xf @ (qb * ia_b[:, None]).T - ref, ref), "b_noise": rows(xf @ cb_w.T - ref, ref)}
    part_err = {p: rows(t, ref) for p, t in sp["parts"].items()}
    tot2 = float(total.double().pow(2).sum())
    share = {p: float((t * total).sum()) / tot2 for p, t in sp["parts"].items()}
    gain = noise_gain(xf, wf, na_x, nb_w)
    expected = (one["a_clean"]["all"] ** 2 * gain["a"] + one["b_clean"]["all"] ** 2 * gain["b"]) ** 0.5
    form_a = e4m3((al_a[:, None] * xf).double() + sp["na"])
    form_b = e4m3((al_b[:, None] * wf).double() + sp["nb"])
    forming = {"a_codes_differ": float((form_a.view(torch.uint8) != natural(at.view(torch.uint8))).float().mean()),
               "b_codes_differ": float((form_b.view(torch.uint8) != natural(bt.view(torch.uint8))).float().mean())}
    return {"w": w, "name": name, "kind": kind(name), "m": m, "n": n, "k": k, "error": errors, "one_sided": one,
            "parts": part_err, "share": share, "identity_max_rel": identity,
            "served_equals_unfused": bool(torch.equal(y_pc, y2)),
            "noise_law": {"a": noise_law(xf, na_x), "b": noise_law(wf, nb_w)},
            "noise_gain": gain, "expected": {"cast_noise": expected, "cast_noise_over_measured": expected / errors["cast_noise"]["all"]},
            "forming": forming}


# ---- the run -------------------------------------------------------------------------------------------------------

def trace(args) -> int:
    """One chunk: capture (first chunk), the FP8 child (first chunk), then linears until --chunk-min has passed.
    Returns 0 when every linear is traced, 99 when some remain."""
    import e2e
    import torch
    from vllm import SamplingParams
    from vllm_bench import find_model

    t0 = time.perf_counter()
    d = args.out.parent
    scratch = d / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    parts_path = d / "trace-parts.jsonl"
    done = {json.loads(line)["w"] for line in parts_path.read_text().splitlines()} if parts_path.exists() else set()
    args.scheme, args.batch_tokens = SCHEME, e2e.WINDOW
    window = e2e.wikitext_windows(args.model, 1)[0]
    llm = e2e.engine(args, "arm")
    model = find_model(llm)
    state = {"active": False}
    run, weights = e2e.install(model, args, BEACON, state)
    layers = linears(model)
    names = [n for n, _ in layers]
    count = min(args.linears or len(layers), len(layers))
    meta_path = d / "trace-meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if "gate" not in meta:
        gate = e2e.gate(run, weights)
        meta["gate"] = {"ok": all(g["ok"] for g in gate), "rows": gate}
        if not meta["gate"]["ok"]:
            meta_path.write_text(json.dumps(meta, indent=1))
            raise SystemExit("the arm's gate failed")
    if not (scratch / f"x-{count - 1}.pt").exists():
        got: dict = {}
        hooks = [layer.register_forward_pre_hook(
                     lambda _m, a, w=w: got.__setitem__(w, a[0].detach().reshape(-1, a[0].shape[-1]).to("cpu", copy=True)))
                 for w, (_, layer) in enumerate(layers)]
        llm.generate([{"prompt_token_ids": window}], SamplingParams(temperature=0.0, max_tokens=1), use_tqdm=False)
        for h in hooks:
            h.remove()
        if sorted(got) != list(range(len(layers))) or any(v.shape[0] != len(window) for v in got.values()):
            raise SystemExit(f"the capture saw {len(got)} linears, rows {sorted({v.shape[0] for v in got.values()})}")
        for w, x in got.items():
            torch.save(x, scratch / f"x-{w}.pt")
        meta.update(window_tokens=len(window), linears=len(layers))
    if not (scratch / f"fp8-{count - 1}.pt").exists():
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=fp8_child, args=(child, args, str(scratch), names, window, count))
        proc.start()
        if not parent.poll(1200):
            proc.terminate()
            raise SystemExit("the FP8 engine did not answer in 1200 s")
        msg = parent.recv()
        proc.join(120)
        if "error" in msg:
            raise SystemExit(msg["error"] + "\n" + msg.get("trace", ""))
        meta["fp8"] = msg
    meta_path.write_text(json.dumps(meta, indent=1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    run.begin()
    with parts_path.open("a") as out:
        for w, (name, layer) in enumerate(layers[:count]):
            if w in done:
                continue
            if time.perf_counter() - t0 > 60 * args.chunk_min and len(done) > 0:
                run.end()
                return 99
            x = torch.load(scratch / f"x-{w}.pt").cuda()
            y8 = torch.load(scratch / f"fp8-{w}.pt").cuda()
            rec = trace_linear(run, run.kernel, w, name, layer, x, weights[w], y8)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            done.add(w)
            del x, y8
            torch.cuda.empty_cache()
    run.end()
    parts = [json.loads(line) for line in parts_path.read_text().splitlines()]
    doc = {"schema": "verity-pouw/pearl-c-vllm-error-trace/v1", "scheme": SCHEME, "model": args.model,
           "run": os.environ.get("RESEARCH_RUN_ID") or os.environ.get("FILL_JOB"), "meta": meta, "summary": summarize(parts),
           "checks": {"served_equals_unfused": all(p["served_equals_unfused"] for p in parts),
                      "identity_max_rel": max(p["identity_max_rel"] for p in parts),
                      "fp8_same_first_input": meta.get("fp8", {}).get("same_first_input")}}
    args.out.write_text(json.dumps(doc, indent=1))
    shutil.rmtree(scratch, ignore_errors=True)
    print(json.dumps({"summary": doc["summary"]["all"], "checks": doc["checks"]}), flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--kernel", default=str(HERE.parent / "pearl_c_sm120"))
    ap.add_argument("--cubin")
    ap.add_argument("--mem-arm", type=float, default=0.3)
    ap.add_argument("--mem-base", type=float, default=0.2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk-min", type=float, default=6.0, help="start no linear after this many minutes (the fill chunk)")
    ap.add_argument("--linears", type=int, help="trace the first N linears only (a smoke)")
    sys.exit(trace(ap.parse_args()))


if __name__ == "__main__":
    main()
