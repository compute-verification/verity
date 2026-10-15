"""What the forming costs end to end: Llama-3.1-8B's perplexity on the capture's tokens (WikiText-2 test, window 0) when
every linear of every decoder layer serves `cost.py`'s emulated y in place of the bf16 product.

The forward is `pearlc_capture.py`'s streamed bf16 forward on the CPU (one decoder layer at a time from the safetensors);
a forward hook on each of the seven linears replaces its output with the method's y (rounded to bf16, as the layer
consumes it). The final norm and lm_head stay bf16. The noise lines are `pearl_c.sample_line`'s rule over numpy's random
bytes (`cost.lines_fast`), drawn per linear from one seed shared by every method, so v1 and the fix see the same lines.

    bf16              no hook (the reference)
    fp8               e4m3(alpha x) . e4m3(alpha w) with v1's alpha, no noise
    pearl_v1, pearl_fix         Pearl-C's served y (v1, rho' = max(rho, s/4))
    pearl_v1_split, pearl_fix_split   the same with the floor(k/100) largest channels of the call's x exact
    int8n6, int8n6_chunk        int8 to [-121, 121] per row, or per row and chunk of 170 columns, exact products
    int8n6_split                int8n6 with the split's channels exact

    python ppl.py CAPTURE_SCRIPT MODEL_SNAPSHOT IDS.npy METHOD [--layers N] [--json OUT]
    python ppl.py CAPTURE_SCRIPT --selftest
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers

import cost

LINEARS = [m for ms in cost.ROLES.values() for m in ms]


def method_y(method: str, x: np.ndarray, w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base, split = method.removesuffix("_split"), method.endswith("_split")
    exact = 0.0
    if split:
        out = cost.split_channels(x)
        keep = np.setdiff1d(np.arange(x.shape[1]), out)
        exact = x[:, out] @ w[:, out].T
        x, w = x[:, keep], w[:, keep]
    if base in ("fp8", "int8n6", "int8n6_chunk"):
        return cost.quantized(x, w, base) + exact
    fix = "fix" in base
    pad = (-x.shape[1]) % 128
    if pad:
        x, w = np.pad(x, ((0, 0), (0, pad))), np.pad(w, ((0, 0), (0, pad)))
    k = x.shape[1]
    fa, fb = cost.lines_fast(rng, k, 16.0), cost.lines_fast(rng, k, 16.0)
    ea, eb = cost.lines_fast(rng, x.shape[0], 256.0), cost.lines_fast(rng, w.shape[0], 256.0)
    return cost.serve(x, w, fa, fb, ea, eb, fix)[0] + exact


def hook(method: str, layer: int, name: str, log: list):
    seed = layer * 1000 + LINEARS.index(name)

    def fn(mod, inp, out):
        x = inp[0].reshape(-1, inp[0].shape[-1]).double().numpy()
        w = mod.weight.double().numpy()
        y = method_y(method, x, w, np.random.default_rng(seed))
        ref = out.reshape(-1, out.shape[-1]).double().numpy()
        log.append({"layer": layer, "linear": name, "rel_err_vs_bf16": float(np.linalg.norm(y - ref) / np.linalg.norm(ref))})
        return torch.from_numpy(y).to(out.dtype).reshape(out.shape)
    return fn


def forward(capture, snapshot: Path, ids: torch.Tensor, method: str, n_layers: int | None = None) -> dict:
    cfg = transformers.AutoConfig.from_pretrained(snapshot)
    cfg._attn_implementation = "sdpa"
    S = capture.Stream(snapshot, cfg)
    h, pos, rope, mask = S.start(ids)
    log: list = []
    t0 = time.time()
    for i in range(n_layers or cfg.num_hidden_layers):
        layer = S.load(i)
        hooks = [layer.get_submodule(m).register_forward_hook(hook(method, i, m, log)) for m in LINEARS] if method != "bf16" else []
        with torch.no_grad():
            out = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=rope)
        h = out[0] if isinstance(out, tuple) else out
        for hk in hooks:
            hk.remove()
        print(json.dumps({"layer": i, "seconds": round(time.time() - t0, 1)}), flush=True)
    norm = transformers.models.llama.modeling_llama.LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight = torch.nn.Parameter(S.tensor("model.norm.weight"))
    head = S.tensor("lm_head.weight") if "lm_head.weight" in S.wmap else S.tensor("model.embed_tokens.weight")
    with torch.no_grad():
        logits = (norm(h)[0].float() @ head.float().T)
        nll = torch.nn.functional.cross_entropy(logits[:-1], ids[0, 1:], reduction="none")
    return {"method": method, "layers": n_layers or cfg.num_hidden_layers, "tokens": int(ids.shape[1]),
            "nll": float(nll.mean()), "ppl": float(torch.exp(nll.mean())), "seconds": round(time.time() - t0, 1),
            "per_linear": log}


def load_capture(path: str):
    spec = importlib.util.spec_from_file_location("pearlc_capture", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def selftest(capture) -> dict:
    """On a tiny random Llama: the bf16 method equals the plain forward, and every other method runs."""
    import tempfile
    L = transformers.models.llama.modeling_llama
    torch.manual_seed(0)
    cfg = L.LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=8,
                        num_key_value_heads=2, max_position_embeddings=1024, tie_word_embeddings=False)
    with tempfile.TemporaryDirectory() as d:
        L.LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(d)
        ids = torch.randint(0, 512, (1, 64))
        full = L.LlamaForCausalLM.from_pretrained(d, dtype=torch.bfloat16).eval()
        with torch.no_grad():
            ref = torch.nn.functional.cross_entropy(full(ids).logits[0, :-1].float(), ids[0, 1:])
        res = {m: forward(capture, Path(d), ids, m) for m in ("bf16", "fp8", "pearl_v1", "pearl_fix_split", "int8n6_chunk")}
    return {"full_model_nll": float(ref), **{m: {"nll": r["nll"], "max_rel_err": max((x["rel_err_vs_bf16"] for x in r["per_linear"]), default=0.0)}
                                           for m, r in res.items()}}


def main(argv: list[str]) -> int:
    capture = load_capture(argv[0])
    if argv[1] == "--selftest":
        print(json.dumps(selftest(capture), indent=1))
        return 0
    opt = lambda f, d: argv[argv.index(f) + 1] if f in argv else d  # noqa: E731
    torch.set_num_threads(int(opt("--threads", 12)))
    ids = torch.from_numpy(np.load(argv[2]))[None]
    res = forward(capture, Path(argv[1]), ids, argv[3], int(opt("--layers", 0)) or None)
    if "--json" in argv:
        Path(opt("--json", "")).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_linear"}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
