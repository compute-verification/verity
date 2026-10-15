"""What restricting served GEMM operands to an exactly-accumulating grid costs Llama-3.1-8B: WikiText-2 perplexity on the
capture's tokens (`ppl.py`'s eval: test window 0, 2048 tokens) with every linear of every decoder layer replaced by the
format's product, on one GPU.

Exact formats: integer operands, one scale per activation row and one per weight row (output channel), so the scales
factor out of the k-sum and y = s_a s_w (A_int B_int^T).  The integer product is computed in float64, which is the exact
integer sum (every |sum| < 2^53); `bound` is max over outputs of sum_k |a||b|, which bounds every partial sum in every
summation order, against the accumulator's exact range (2^24 for FP32 accumulation of FP8/FP4 products, 2^31 - 1 for s32).

    int8          integers |x| <= 127 (IMMA s8 x s8 -> s32)
    e4m3i16       E4M3's integers |x| <= 16, all integers
    e4m3ik        E4M3's exact integers (1..16, 18..32 by 2, 36..64 by 4, ...) up to M(k), the largest with k M^2 <= 2^24
    e4m3i448      the same set up to 448 (exact only for k <= 83; a quality reference)
    e2m1i         E2M1's integers {0, 1, 2, 3, 4, 6}

Reference formats (scales vary along k or the grid is not integer, so the FP32 sum rounds; products here in float64):

    fp8_tensor    E4M3, one scale per tensor (activation dynamic)
    fp8_token     E4M3, per activation row and per weight row
    fp8_block     E4M3, activations per row and 128-column block, weights per 128 x 128 block
    nvfp4         E2M1 with an E4M3 scale per 16-element block along k and a per-tensor FP32 scale

Suffixes: `_rot` multiplies x and W by one orthogonal Q per k before quantizing (x Q (W Q)^T = x W^T): random signs times
a Sylvester Hadamard, Kronecker a random 7 x 7 orthogonal for k = 14,336 (QuaRot's randomized Hadamard).  `_wclip`
picks each weight row's clip ratio in {1, 0.95, ..., 0.5} by squared error.

    python exact_grid.py MODEL_SNAPSHOT IDS.npy OUT_DIR METHOD...
    python exact_grid.py --selftest
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers

LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
FP32_EXACT = 2 ** 24
S32_EXACT = 2 ** 31 - 1


def e4m3_ints(top: int) -> list[int]:
    vals = set(range(17))
    for e, step in ((16, 2), (32, 4), (64, 8), (128, 16)):
        vals |= set(range(e, 2 * e, step))
    vals |= set(range(256, 449, 32))
    return sorted(v for v in vals if v <= top)


def e4m3_top(k: int) -> int:
    return max(v for v in e4m3_ints(448) if k * v * v <= FP32_EXACT)


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_INT = [0, 1, 2, 3, 4, 6]


def grid(base: str, k: int) -> tuple[list[float], int]:
    """(nonnegative levels, accumulator exact range) of an exact format at depth k."""
    if base == "int8":
        return list(range(128)), S32_EXACT
    if base == "e4m3i16":
        return list(range(17)), FP32_EXACT
    if base == "e4m3ik":
        return e4m3_ints(e4m3_top(k)), FP32_EXACT
    if base == "e4m3i448":
        return e4m3_ints(448), FP32_EXACT
    if base == "e2m1i":
        return E2M1_INT, FP32_EXACT
    raise KeyError(base)


def to_levels(v: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Nearest level of |v| (clipped to the top), with v's sign."""
    a = v.abs().clamp(max=float(levels[-1]))
    if bool((levels[1:] - levels[:-1] == 1).all()):
        q = torch.round(a)
    else:
        q = levels[torch.bucketize(a, (levels[1:] + levels[:-1]) / 2)]
    return torch.copysign(q, v)


def quant_rows(m: torch.Tensor, levels: torch.Tensor, clip: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer rows and per-row scales: m ~ q * s[:, None]."""
    top = float(levels[-1])
    amax = m.abs().amax(dim=1).clamp(min=1e-30)
    if not clip:
        s = amax / top
        return to_levels(m / s[:, None], levels), s
    best_q, best_s, best_e = None, None, None
    for c in np.linspace(1.0, 0.5, 11):
        s = amax * float(c) / top
        q = to_levels(m / s[:, None], levels)
        e = ((q * s[:, None] - m) ** 2).sum(dim=1)
        if best_e is None:
            best_q, best_s, best_e = q, s, e
        else:
            better = e < best_e
            best_q = torch.where(better[:, None], q, best_q)
            best_s = torch.where(better, s, best_s)
            best_e = torch.minimum(e, best_e)
    return best_q, best_s


def e4m3(v: torch.Tensor) -> torch.Tensor:
    return v.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(v.dtype)


def fp8_dequant(m: torch.Tensor, how: str, side: str) -> torch.Tensor:
    if how == "fp8_tensor":
        s = m.abs().amax().clamp(min=1e-30) / 448.0
        return e4m3(m / s) * s
    if how == "fp8_token":
        s = m.abs().amax(dim=1, keepdim=True).clamp(min=1e-30) / 448.0
        return e4m3(m / s) * s
    r, k = m.shape
    pad = (-k) % 128
    mp = torch.nn.functional.pad(m, (0, pad))
    if side == "x":
        b = mp.reshape(r, -1, 128)
        s = b.abs().amax(dim=2, keepdim=True).clamp(min=1e-30) / 448.0
        return (e4m3(b / s) * s).reshape(r, -1)[:, :k]
    rp = (-r) % 128
    mp = torch.nn.functional.pad(mp, (0, 0, 0, rp))
    R, K = mp.shape
    b = mp.reshape(R // 128, 128, K // 128, 128)
    s = b.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-30) / 448.0
    return (e4m3(b / s) * s).reshape(R, K)[:r, :k]


def nvfp4_dequant(m: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    r, k = m.shape
    g = m.abs().amax().clamp(min=1e-30) / (448.0 * 6.0)
    b = m.reshape(r, k // 16, 16)
    bs = e4m3(b.abs().amax(dim=2, keepdim=True) / 6.0 / g) * g
    bs = torch.where(bs > 0, bs, torch.ones_like(bs))
    return (to_levels(b / bs, levels) * bs).reshape(r, k)


_ROT: dict[int, torch.Tensor] = {}


def hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def rotation(k: int, device) -> torch.Tensor:
    if k not in _ROT:
        g = torch.Generator().manual_seed(k)
        odd = k
        while odd % 2 == 0:
            odd //= 2
        h = hadamard(k // odd)
        if odd > 1:
            r, _ = torch.linalg.qr(torch.randn(odd, odd, generator=g, dtype=torch.float64))
            h = torch.kron(h.contiguous(), r.contiguous())
        signs = torch.randint(0, 2, (k,), generator=g).double() * 2 - 1
        _ROT[k] = (signs[:, None] * h).to(device)
    return _ROT[k]


EXACT = ("int8", "e4m3i16", "e4m3ik", "e4m3i448", "e2m1i")
REFERENCE = ("fp8_tensor", "fp8_token", "fp8_block", "nvfp4")


def product(method: str, x: torch.Tensor, w: torch.Tensor, wcache: dict) -> tuple[torch.Tensor, dict]:
    """The method's y for float64 x (m x k) and W (n x k), and its exactness record."""
    base = method.split("_rot")[0].split("_wclip")[0]
    rot, clip = "_rot" in method, "_wclip" in method
    k = x.shape[1]
    if rot:
        q = rotation(k, x.device)
        x = x @ q
    if base in EXACT:
        lv, acc = grid(base, k)
        levels = torch.tensor(lv, dtype=torch.float64, device=x.device)
        if "w" not in wcache:
            wq = w @ rotation(k, w.device) if rot else w
            wcache["w"] = quant_rows(wq, levels, clip)
        bi, sb = wcache["w"]
        ai, sa = quant_rows(x, levels, False)
        s = ai @ bi.T
        bound = float((ai.abs().float() @ bi.abs().float().T).max())
        a_priori = k * float(lv[-1]) ** 2
        return s * sa[:, None] * sb[None, :], {"bound": bound, "acc": acc, "top": float(lv[-1]), "a_priori": a_priori,
                                                "exact": bound <= acc, "guaranteed": a_priori <= acc}
    if base in ("fp8_tensor", "fp8_token", "fp8_block"):
        if "w" not in wcache:
            wcache["w"] = fp8_dequant(w @ rotation(k, w.device) if rot else w, base, "w")
        return fp8_dequant(x, base, "x") @ wcache["w"].T, {"exact": False}
    if base == "nvfp4":
        levels = torch.tensor(E2M1, dtype=torch.float64, device=x.device)
        if "w" not in wcache:
            wcache["w"] = nvfp4_dequant(w @ rotation(k, w.device) if rot else w, levels)
        return nvfp4_dequant(x, levels) @ wcache["w"].T, {"exact": False}
    raise KeyError(method)


def attach(model, method: str, log: list) -> list:
    hooks = []
    for i, layer in enumerate(model.model.layers):
        for name in LINEARS:
            mod = layer.get_submodule(name)
            wcache: dict = {}

            def fn(mod, inp, out, i=i, name=name, wcache=wcache):
                x = inp[0].reshape(-1, inp[0].shape[-1]).double()
                w = mod.weight.double()
                y, rec = product(method, x, w, wcache)
                ref = x @ w.T
                rec.update(layer=i, linear=name, k=x.shape[1],
                           rel_err=float(torch.linalg.norm(y - ref) / torch.linalg.norm(ref)))
                log.append(rec)
                return y.to(out.dtype).reshape(out.shape)
            hooks.append(mod.register_forward_hook(fn))
    return hooks


def evaluate(model, ids: torch.Tensor, method: str) -> dict:
    log: list = []
    hooks = attach(model, method, log) if method != "bf16" else []
    t0 = time.time()
    try:
        with torch.no_grad():
            logits = model(ids).logits[0].float()
    finally:
        for h in hooks:
            h.remove()
    nll = torch.nn.functional.cross_entropy(logits[:-1], ids[0, 1:], reduction="none")
    worst = max(log, key=lambda r: r["rel_err"]) if log else None
    exact = [r for r in log if "bound" in r]
    return {"method": method, "tokens": int(ids.shape[1]), "nll": float(nll.mean()), "ppl": float(torch.exp(nll.mean())),
            "seconds": round(time.time() - t0, 1), "worst_linear": worst,
            "exact_all": all(r["exact"] for r in log) if log else None,
            "guaranteed_all": all(r.get("guaranteed", False) for r in log) if log else None,
            "max_bound_over_acc": max((r["bound"] / r["acc"] for r in exact), default=None),
            "per_linear": log}


def load(snapshot: str):
    model = transformers.AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16, attn_implementation="sdpa")
    return model.cuda().eval()


def selftest() -> dict:
    """Grids, exactness and rotation on small tensors; every method runs on a tiny random Llama."""
    assert e4m3_ints(32) == list(range(17)) + list(range(18, 33, 2))
    assert e4m3_top(4096) == 64 and e4m3_top(14336) == 32 and max(e4m3_ints(448)) == 448 and len(e4m3_ints(448)) == 55
    for v in e4m3_ints(448):
        assert float(torch.tensor(float(v)).to(torch.float8_e4m3fn).float()) == v
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for n in (28, 256):
        q = rotation(n, dev)
        assert torch.allclose(q @ q.T, torch.eye(n, dtype=torch.float64, device=dev), atol=1e-12)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 256, generator=g, dtype=torch.float64).to(dev)
    w = torch.randn(32, 256, generator=g, dtype=torch.float64).to(dev)
    out = {}
    for m in EXACT + REFERENCE + ("e4m3i16_rot", "e2m1i_rot_wclip"):
        y, rec = product(m, x, w, {})
        out[m] = {"rel_err": float(torch.linalg.norm(y - x @ w.T) / torch.linalg.norm(x @ w.T)), **rec}
    assert out["int8"]["rel_err"] < out["e4m3i16"]["rel_err"] < out["e2m1i"]["rel_err"]
    L = transformers.models.llama.modeling_llama
    torch.manual_seed(0)
    cfg = L.LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=8,
                        num_key_value_heads=2, max_position_embeddings=1024, tie_word_embeddings=False)
    model = L.LlamaForCausalLM(cfg).to(torch.bfloat16).to(dev).eval()
    ids = torch.randint(0, 512, (1, 64), device=dev)
    out["tiny"] = {m: {k: v for k, v in evaluate(model, ids, m).items() if k not in ("per_linear", "worst_linear")}
                   for m in ("bf16", "int8", "e2m1i_rot", "nvfp4", "fp8_block")}
    return out


def main(argv: list[str]) -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    if argv[0] == "--selftest":
        print(json.dumps(selftest(), indent=1))
        return 0
    snapshot, ids, out = argv[0], torch.from_numpy(np.load(argv[1]))[None].cuda(), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    model = load(snapshot)
    for method in argv[3:]:
        res = evaluate(model, ids, method)
        (out / f"ppl_{method}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "per_linear"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
