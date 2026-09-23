"""W-A56-2 on the pinned instance: the served argmax kernel's NaN / tie / signed-zero / -inf rule, MEASURED with crafted rows.

Run INSIDE the pinned instance's environment (GPU pod `eea6d108f985` class: NVIDIA RTX 4090 sm_89, torch 2.13.0+cu129, vLLM
0.28.1rc1.dev472+gd9105ea80 at /workspace/rec-env; VLLM_USE_FLASHINFER_SAMPLER=0, VLLM_BATCH_INVARIANT=1) -- GPU pods only
through the integrator.  The record's capture histogram says the greedy path ran `aten.argmax` over the logits (16 calls per
request: `sample-phase-skipped:aten.argmax`), so the first probe is torch.argmax on CUDA over bf16 and fp32 rows; the second
drives vLLM's own V2 sampler (`vllm.v1.worker.gpu.sample.sampler.Sampler`, greedy) when importable, so the two agree or the
difference is on record.

    python probe_pinned.py --out argmax_rule_pinned.json

Prints one line per case: case, dtype, torch.argmax index, vLLM sampler index (or "n/a"), and the summary rule.  Cases:
  tie_same_block      two equal maxima inside one 1024 block (idx 7, 700)         -> lowest or highest index?
  tie_across_blocks   equal maxima in different blocks (idx 7, 40000)             -> lowest or highest?
  tie_three           three equal maxima (idx 3, 2048, 49000)
  nan_mid             one NaN at idx 1234, real max elsewhere (idx 40000)         -> NaN wins (torch GreaterOrNan) or never wins (program Bf16GtStrict)?
  nan_two             NaN at 1234 and 30000                                       -> which NaN?
  nan_first           NaN at idx 0                                                (the program's init makes idx 0 the default winner)
  nan_all             every logit NaN
  neg_zero_tie        +0.0 at idx 5 and -0.0 at idx 9 as the maxima              -> which sign wins?
  neg_inf_all         every logit -inf                                            -> idx 0?
  pos_inf_and_nan     +inf at idx 100, NaN at idx 200
V = 49152 (the pinned vocabulary).  Nothing is invented: if a case cannot be driven through the vLLM sampler the row says so.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys

V = 49152


def cases():
    yield "tie_same_block", {7: 5.0, 700: 5.0}, 1.0
    yield "tie_across_blocks", {7: 5.0, 40000: 5.0}, 1.0
    yield "tie_three", {3: 5.0, 2048: 5.0, 49000: 5.0}, 1.0
    yield "nan_mid", {1234: math.nan, 40000: 5.0}, 1.0
    yield "nan_two", {1234: math.nan, 30000: math.nan, 40000: 5.0}, 1.0
    yield "nan_first", {0: math.nan, 40000: 5.0}, 1.0
    yield "nan_all", "all_nan", None
    yield "neg_zero_tie", {5: 0.0, 9: -0.0}, -1.0
    yield "neg_inf_all", "all_neg_inf", None
    yield "pos_inf_and_nan", {100: math.inf, 200: math.nan}, 1.0


def build_row(torch, spec, fill, dtype, device):
    if spec == "all_nan":
        return torch.full((V,), math.nan, dtype=dtype, device=device)
    if spec == "all_neg_inf":
        return torch.full((V,), -math.inf, dtype=dtype, device=device)
    row = torch.full((V,), fill, dtype=dtype, device=device)
    for i, v in spec.items():
        row[i] = v
    return row


def vllm_greedy(torch, logits2d):
    """vLLM V2 sampler greedy path on a [B, V] tensor; returns list of ints or None when not importable/drivable."""
    try:
        from vllm.v1.worker.gpu.sample import sampler as S  # type: ignore
    except Exception as e:  # noqa: BLE001
        return None, f"vllm sampler import failed: {type(e).__name__}: {e}"
    # Prefer a module-level greedy helper if the pinned build has one; otherwise report n/a rather than guess.
    for name in ("greedy_sample", "_greedy_sample", "sample_greedy"):
        fn = getattr(S, name, None)
        if fn is not None:
            try:
                out = fn(logits2d)
                return [int(x) for x in out.view(-1).tolist()], f"vllm.v1.worker.gpu.sample.sampler.{name}"
            except Exception as e:  # noqa: BLE001
                return None, f"{name} raised {type(e).__name__}: {e}"
    return None, "no module-level greedy helper in the pinned sampler; drive Sampler.forward with a greedy SamplingMetadata manually"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="argmax_rule_pinned.json")
    a = ap.parse_args()
    import torch  # noqa: PLC0415

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if dev == "cuda" else "CPU (NOT the pinned instance)",
        "cuda": torch.version.cuda,
        "label": "MEASURED on the pinned instance" if dev == "cuda" else "MEASURED on CPU torch -- NOT the pinned kernel",
    }
    try:
        import vllm  # type: ignore

        env["vllm"] = vllm.__version__
    except Exception:  # noqa: BLE001
        env["vllm"] = "not importable"
    rows = []
    for name, spec, fill in cases():
        for dtype in (torch.bfloat16, torch.float32):
            row = build_row(torch, spec, fill, dtype, dev)
            t_idx = int(torch.argmax(row).item())
            v_idx, v_note = vllm_greedy(torch, row.view(1, -1))
            rows.append({"case": name, "dtype": str(dtype).replace("torch.", ""), "torch_argmax": t_idx,
                         "vllm_greedy": None if v_idx is None else v_idx[0], "vllm_note": v_note})
            print(f"{name:18s} {str(dtype).replace('torch.', ''):9s} torch.argmax={t_idx:6d} vllm={v_idx[0] if v_idx else 'n/a':>6}  {v_note}")
    by = {(r["case"], r["dtype"]): r["torch_argmax"] for r in rows}
    summary = {
        "tie_rule_torch": "lowest index" if by[("tie_same_block", "bfloat16")] == 7 and by[("tie_across_blocks", "bfloat16")] == 7 and by[("tie_three", "bfloat16")] == 3
        else "NOT lowest index -- see rows",
        "nan_rule_torch": "NaN wins (first NaN)" if by[("nan_mid", "bfloat16")] == 1234 and by[("nan_two", "bfloat16")] == 1234
        else "NaN never wins" if by[("nan_mid", "bfloat16")] == 40000 else "other -- see rows",
        "neg_zero": "+0 at 5 wins" if by[("neg_zero_tie", "bfloat16")] == 5 else "-0 at 9 wins",
        "all_neg_inf": by[("neg_inf_all", "bfloat16")],
        "all_nan": by[("nan_all", "bfloat16")],
        "vllm_sampler_driven": any(r["vllm_greedy"] is not None for r in rows),
    }
    print(json.dumps({"env": env, "summary": summary}, indent=1))
    with open(a.out, "w") as f:
        json.dump({"env": env, "summary": summary, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
