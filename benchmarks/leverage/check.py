"""Checks `lm.LM` before any measurement reads it, and exits 1 on a failure.

  - its logits against Hugging Face transformers' on two prompts, when transformers is importable
    (`uv run --with transformers ...`);
  - a cached decode against one full forward;
  - a fault at a site's honest value changes nothing, and a fault after the last layer's attention leaves every later
    position's cache, so every later served token, as honest.

  uv run --extra torch-cpu --with tokenizers [--with transformers] python benchmarks/leverage/check.py --model gpt2 --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm import LM, MODELS, Cache, Fault, fetch
from prompts import PROMPTS


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    lm = LM(a.model)
    tok = lm.tokenizer()
    res, ok = {"model": a.model, "revision": MODELS[a.model]["revision"]}, True

    ids = torch.tensor([tok.encode(PROMPTS[0]).ids])
    full = lm.forward(ids, Cache())
    c = Cache()
    lm.forward(ids[:, :-3], c)
    steps = torch.cat([lm.forward(ids[:, i:i + 1], c) for i in range(ids.shape[1] - 3, ids.shape[1])], 1)
    res["cache_vs_full"] = max_abs(steps, full[:, -3:])
    ok &= res["cache_vs_full"] < 1e-3

    try:
        import transformers
    except ImportError:
        res["hf"] = "transformers not importable: skipped"
    else:
        hf = transformers.AutoModelForCausalLM.from_pretrained(fetch(a.model), torch_dtype=torch.float32,
                                                               attn_implementation="eager").eval()
        diffs = []
        for p in PROMPTS[:2]:
            x = torch.tensor([tok.encode(p).ids])
            with torch.no_grad():
                ref = hf(x).logits
            diffs.append(max_abs(lm.forward(x, Cache()), ref))
        res["hf_max_abs_logit_diff"] = diffs
        res["hf_version"] = transformers.__version__
        ok &= max(diffs) < 2e-3

    t0 = ids.shape[1] - 1
    taps = {(s, l): None for s, l in (("final", lm.spec.layers), ("mlph", 3), ("resid", 0))}
    pre = Cache()
    lm.forward(ids[:, :t0], pre)
    honest = lm.forward(ids[:, t0:], pre.expand(1), taps=taps)
    same = []
    for (s, l), x in taps.items():
        f = Fault.of(t0, [(s, l, 5, float(x[0, 0, 5]))])
        cf = pre.expand(1)
        same.append(max_abs(lm.forward(ids[:, t0:], cf, fault=f), honest))
    res["fault_at_honest_value"] = same
    ok &= max(same) < 1e-2

    L = lm.spec.layers
    rows = [("final", L, 3, 1e4), ("resid", L - 1, 3, 1e4), ("mlp", L - 1, 3, 1e4), ("attn", L - 1, 3, 1e4),
            ("mlph", L - 1, 3, 1e4), ("resid", L - 2, 3, 1e4)]
    f = Fault.of(t0, rows)
    cf, ch = pre.expand(len(rows)), pre.expand(len(rows))
    x = ids[:, t0:].expand(len(rows), 1)
    lf, lh = lm.forward(x, cf, fault=f), lm.forward(x, ch)
    y = lf[:, -1].argmax(-1, keepdim=True)
    nf, nh = lm.forward(y, cf), lm.forward(y, ch)
    later = [max_abs(nf[i], nh[i]) for i in range(len(rows))]
    res["first_token_changed"] = [bool(lf[i, -1].argmax() != lh[i, -1].argmax()) for i in range(len(rows))]
    res["next_position_logit_diff"] = dict(zip([f"{s}{l}" for s, l, _, _ in rows], later))
    ok &= max(later[:5]) == 0.0 and later[5] > 0.0

    res["ok"] = bool(ok)
    (a.out / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
