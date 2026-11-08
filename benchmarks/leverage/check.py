"""Checks `lm.LM` before any measurement reads it, and exits 1 on a failure.

  - its logits against Hugging Face transformers' on two prompts, when transformers is importable
    (`uv run --with transformers ...`);
  - a cached decode against one full forward;
  - a fault at a site's honest value changes nothing, and a fault after the last layer's attention leaves every later
    position's cache, so every later served token, as honest;
  - a joint fault writing a whole group at its honest values changes nothing, and neither do boxes no honest value
    reaches;
  - an `Edit` of one down-projection weight matches the model with that weight changed, and a forward started from a
    layer's residual stream (x_in) matches the full one.

  uv run --extra torch-cpu --with tokenizers [--with transformers] python benchmarks/leverage/check.py --model gpt2 --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm import GROUP, LM, MODELS, Boxes, Cache, Edit, Fault, fetch
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

    taps = {("mlph", 2): None, ("ln1", 1): None}
    honest = lm.forward(ids[:, t0:], pre.expand(1), taps=taps)
    joint = []
    for (s, l), x in taps.items():
        g = list(range(GROUP, 2 * GROUP))
        f = Fault.of(t0, [(s, l, g, x[0, 0, g].tolist())], dtype=torch.float32)
        joint.append(max_abs(lm.forward(ids[:, t0:], pre.expand(1), fault=f), honest))
    res["joint_fault_at_honest_values"] = joint
    ok &= max(joint) < 1e-4

    lm.boxes = Boxes({k: torch.full((2, lm.width(k[0])), 2.0 ** 60) for k in lm.sites()})
    res["wide_boxes"] = max_abs(lm.forward(ids, Cache()), full)
    res["wide_boxes_cut"] = sum(lm.boxes.cut.values())
    lm.boxes = None
    ok &= res["wide_boxes"] == 0.0 and res["wide_boxes_cut"] == 0

    l, i, j, delta = 2, 7, 11, 0.05
    W = lm.down_proj(l)
    e = lm.forward(ids.expand(2, -1), Cache(), edit=Edit(l, i, j, torch.tensor([0.0, delta])))
    w0 = float(W[i, j])
    W[i, j] = w0 + delta
    changed = lm.forward(ids, Cache())
    W[i, j] = w0
    res["edit_vs_changed_weight"] = max_abs(e[1], changed[0])
    res["edit_zero_vs_honest"] = max_abs(e[0], full[0])
    ok &= res["edit_vs_changed_weight"] < 1e-3 and res["edit_zero_vs_honest"] < 1e-5

    k = 3
    taps = {("resid", k - 1): None}
    lm.forward(ids, Cache(), taps=taps)
    res["x_in_vs_full"] = max_abs(lm.forward(None, Cache(), x_in=taps[("resid", k - 1)], first_layer=k, pos0=0), full)
    ok &= res["x_in_vs_full"] < 1e-3

    res["ok"] = bool(ok)
    (a.out / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
