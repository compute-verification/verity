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
    e4m3ika       the same set, activations to M_a and weights to M_w with k M_a M_w <= 2^24 (`e4m3_split`)
    e4m3i448      the same set up to 448 (exact only for k <= 83; a quality reference)
    e2m1i         E2M1's integers {0, 1, 2, 3, 4, 6}
    e2m1b         E2M1 (halves included) times a power-of-two scale 2^e per 16 along k, e in 0..E(k) with k (12 2^E)^2 <=
                  2^24, under one scale per row: every product an integer in quarter units (nvf4 with UE4M3 scales 2^e)

Reference formats (scales vary along k or the grid is not integer, so the FP32 sum rounds; products here in float64):

    fp8_tensor    E4M3, one scale per tensor (activation dynamic)
    fp8_token     E4M3, per activation row and per weight row
    fp8_block     E4M3, activations per row and 128-column block, weights per 128 x 128 block
    nvfp4         E2M1 with an E4M3 scale per 16-element block along k and a per-tensor FP32 scale

Suffixes: `_rot` multiplies x and W by one orthogonal Q per k before quantizing (x Q (W Q)^T = x W^T): random signs times
a Sylvester Hadamard, Kronecker a random 7 x 7 orthogonal for k = 14,336 (QuaRot's randomized Hadamard).  `_wclip`
picks each weight row's clip ratio in {1, 0.95, ..., 0.5} by squared error.

Unpeeled edits (note:20261015T1826Z-draft-d4-perplexity), `GRID[_rot]@EDIT:ALPHABET[:sSEED]` with GRID int8 or e4m3i16:
EDIT `w` adds one draw from ALPHABET (in LSBs) to each weight integer, fresh per matrix; `a` to each activation integer,
fresh per token and per linear; `wa` both; `GRID@-` is the grid unedited.  An edited side is quantized to top M - max|e|,
so every edited integer stays on the grid (|q| <= 127, or <= 16).  The integer product is torch._int_mm (s8 x s8 -> s32,
exact); x, W and the rotation are fp32.  ALPHABETS lists the alphabets, each drawn uniformly.

    python exact_grid.py MODEL_SNAPSHOT IDS.npy OUT_DIR METHOD...
    python exact_grid.py --selftest
    python exact_grid.py --table RESULTS.jsonl...
    python exact_grid.py --edit-table RESULTS.jsonl...
"""
from __future__ import annotations

import json
import math
import sys
import time
import zlib
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


def e4m3_split(k: int) -> tuple[int, int]:
    """(activation top, weight top) among E4M3's integers: the weight top at most half of sqrt(2^24 / k), the activation
    top the largest with k M_a M_w <= 2^24 (128 and 32 at k = 4096, 72 and 16 at k = 14,336)."""
    mw = max(v for v in e4m3_ints(448) if 2 * v <= math.sqrt(FP32_EXACT / k))
    return max(v for v in e4m3_ints(448) if k * v * mw <= FP32_EXACT), mw


def grid(base: str, k: int) -> tuple[list[float], list[float], int]:
    """(activation levels, weight levels, accumulator exact range) of an exact format at depth k; levels nonnegative."""
    if base == "int8":
        return list(range(128)), list(range(128)), S32_EXACT
    if base == "e4m3i16":
        return list(range(17)), list(range(17)), FP32_EXACT
    if base == "e4m3ik":
        return e4m3_ints(e4m3_top(k)), e4m3_ints(e4m3_top(k)), FP32_EXACT
    if base == "e4m3ika":
        ma, mw = e4m3_split(k)
        return e4m3_ints(ma), e4m3_ints(mw), FP32_EXACT
    if base == "e4m3i448":
        return e4m3_ints(448), e4m3_ints(448), FP32_EXACT
    if base == "e2m1i":
        return E2M1_INT, E2M1_INT, FP32_EXACT
    raise KeyError(base)


def e2m1b_window(k: int) -> int:
    """The largest E with k (12 2^E)^2 <= 2^24 (2 at k = 4096, 1 at 14,336): `e2m1b`'s block exponents 0..E."""
    return max(e for e in range(8) if k * (12 * 2 ** e) ** 2 <= FP32_EXACT)


def quant_blocks(m: torch.Tensor, window: int, levels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`e2m1b`'s integer rows and per-row units: m ~ q * s[:, None], q = 2 e2m1 2^e with one e in 0..window per 16 along
    k (the row's largest block at e = window), so q is an integer of magnitude at most 12 2^window."""
    r, k = m.shape
    u = m.abs().amax(dim=1).clamp(min=1e-30) / (6.0 * 2 ** window)
    b = m.reshape(r, k // 16, 16) / u[:, None, None]
    e = torch.exp2(torch.ceil(torch.log2(b.abs().amax(dim=2, keepdim=True).clamp(min=1e-30) / 6.0)).clamp(0, window))
    return (2 * to_levels(b / e, levels) * e).reshape(r, k), u / 2


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


EXACT = ("int8", "e4m3i16", "e4m3ik", "e4m3ika", "e4m3i448", "e2m1i")
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
        la, lw, acc = grid(base, k)
        if "w" not in wcache:
            wq = w @ rotation(k, w.device) if rot else w
            wcache["w"] = quant_rows(wq, torch.tensor(lw, dtype=torch.float64, device=x.device), clip)
        bi, sb = wcache["w"]
        ai, sa = quant_rows(x, torch.tensor(la, dtype=torch.float64, device=x.device), False)
        s = ai @ bi.T
        bound = float((ai.abs().float() @ bi.abs().float().T).max())
        a_priori = k * float(la[-1]) * float(lw[-1])
        return s * sa[:, None] * sb[None, :], {"bound": bound, "acc": acc, "top": [float(la[-1]), float(lw[-1])],
                                                "a_priori": a_priori, "exact": bound <= acc, "guaranteed": a_priori <= acc}
    if base == "e2m1b":
        window, levels = e2m1b_window(k), torch.tensor(E2M1, dtype=torch.float64, device=x.device)
        if "w" not in wcache:
            wcache["w"] = quant_blocks(w @ rotation(k, w.device) if rot else w, window, levels)
        bi, sb = wcache["w"]
        ai, sa = quant_blocks(x, window, levels)
        bound = float((ai.abs().float() @ bi.abs().float().T).max())
        a_priori = k * (12.0 * 2 ** window) ** 2
        return (ai @ bi.T) * sa[:, None] * sb[None, :], {"bound": bound, "acc": FP32_EXACT, "top": [12.0 * 2 ** window] * 2,
                                                         "a_priori": a_priori, "exact": bound <= FP32_EXACT,
                                                         "guaranteed": a_priori <= FP32_EXACT}
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


ALPHABETS = {"pm1": [-1, 1], "pm13": [-3, -1, 1, 3], "pm57": [-7, -5, 5, 7],
             **{f"u{k}": list(range(-k, k + 1)) for k in (1, 2, 4, 8, 16)}}
EDIT_GRIDS = {"int8": (127, S32_EXACT), "e4m3i16": (16, FP32_EXACT)}
_ROT32: dict[int, torch.Tensor] = {}


def parse_edit(method: str) -> dict:
    head, _, spec = method.partition("@")
    grid_name, rot = head.split("_rot")[0], head.endswith("_rot")
    parts = spec.split(":")
    edit = parts[0]
    if grid_name not in EDIT_GRIDS or edit not in ("-", "w", "a", "wa") or (edit == "-") != (len(parts) == 1):
        raise KeyError(method)
    alphabet = ALPHABETS[parts[1]] if edit != "-" else [0]
    seed = int(parts[2].removeprefix("s")) if len(parts) > 2 else 0
    top = EDIT_GRIDS[grid_name][0]
    room = top - max(abs(v) for v in alphabet)
    if room < 1:
        raise ValueError(f"{method}: alphabet does not fit on {grid_name} (|q| <= {top})")
    return {"grid": grid_name, "rot": rot, "edit": edit, "alphabet": parts[1] if edit != "-" else "-", "values": alphabet,
            "seed": seed, "top_w": room if "w" in edit else top, "top_a": room if "a" in edit else top,
            "acc": EDIT_GRIDS[grid_name][1], "bits": math.log2(len(alphabet))}


def draw(shape: tuple[int, int], values: list[int], tag: str, device) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(zlib.crc32(tag.encode()))
    v = torch.tensor(values, dtype=torch.float32, device=device)
    return v[torch.randint(len(values), shape, generator=g, device=device)]


def int_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a b^T of integer-valued fp32 operands in [-127, 127], exactly, as int64."""
    return torch._int_mm(a.to(torch.int8), b.to(torch.int8).T).long()


def edited_product(method: str, x: torch.Tensor, w: torch.Tensor, wcache: dict, tag: str) -> tuple[torch.Tensor, dict]:
    """The edited grid's y for fp32 x (m x k) and W (n x k); `tag` names the linear, so each matrix draws its own noise."""
    e = parse_edit(method)
    k = x.shape[1]
    if e["rot"]:
        if k not in _ROT32:
            _ROT32[k] = rotation(k, x.device).float()
        x = x @ _ROT32[k]
    if "w" not in wcache:
        wq = w @ _ROT32[k] if e["rot"] else w
        bi, sb = quant_rows(wq, torch.arange(e["top_w"] + 1, dtype=torch.float32, device=x.device), False)
        if "w" in e["edit"]:
            bi = bi + draw(tuple(bi.shape), e["values"], f"{e['seed']}/{tag}/w", x.device)
        wcache["w"] = bi, sb
    bi, sb = wcache["w"]
    ai, sa = quant_rows(x, torch.arange(e["top_a"] + 1, dtype=torch.float32, device=x.device), False)
    if "a" in e["edit"]:
        ai = ai + draw(tuple(ai.shape), e["values"], f"{e['seed']}/{tag}/a", x.device)
    top = EDIT_GRIDS[e["grid"]][0]
    assert float(ai.abs().max()) <= top and float(bi.abs().max()) <= top
    s = int_product(ai, bi)
    bound = float(int_product(ai.abs(), bi.abs()).max())
    a_priori = k * float(top) * float(top)
    y = s.double() * sa.double()[:, None] * sb.double()[None, :]
    return y, {"bound": bound, "acc": e["acc"], "top": [float(top), float(top)], "a_priori": a_priori,
               "exact": bound <= e["acc"], "guaranteed": a_priori <= e["acc"]}


def attach(model, method: str, log: list) -> list:
    hooks = []
    for i, layer in enumerate(model.model.layers):
        for name in LINEARS:
            mod = layer.get_submodule(name)
            wcache: dict = {}

            def fn(mod, inp, out, i=i, name=name, wcache=wcache):
                x = inp[0].reshape(-1, inp[0].shape[-1])
                if "@" in method:
                    x, w = x.float(), mod.weight.float()
                    y, rec = edited_product(method, x, w, wcache, f"{i}/{name}")
                else:
                    x, w = x.double(), mod.weight.double()
                    y, rec = product(method, x, w, wcache)
                ref = (x @ w.T).to(y.dtype)
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
            "mean_rel_err": float(np.mean([r["rel_err"] for r in log])) if log else None,
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
    assert e2m1b_window(4096) == 2 and e2m1b_window(14336) == 1
    assert e4m3_split(4096) == (128, 32) and e4m3_split(14336) == (72, 16)
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
    for m in EXACT + REFERENCE + ("e4m3i16_rot", "e2m1i_rot_wclip", "e2m1b", "e2m1b_rot"):
        y, rec = product(m, x, w, {})
        out[m] = {"rel_err": float(torch.linalg.norm(y - x @ w.T) / torch.linalg.norm(x @ w.T)), **rec}
    assert out["int8"]["rel_err"] < out["e4m3i16"]["rel_err"] < out["e2m1i"]["rel_err"]
    xf, wf = x.float(), w.float()
    for m in ("int8", "int8_rot", "e4m3i16", "e4m3i16_rot"):
        y0, y1 = edited_product(m + "@-", xf, wf, {}, "t")[0], product(m, x, w, {})[0]
        assert float(torch.linalg.norm(y0 - y1) / torch.linalg.norm(y1)) < 1e-3, m
    ai, sa = quant_rows(xf, torch.arange(128 - 16, dtype=torch.float32, device=dev), False)
    bi, sb = quant_rows(wf, torch.arange(128, dtype=torch.float32, device=dev), False)
    ya, _ = edited_product("int8@a:u16", xf, wf, {}, "t")
    noise = draw(tuple(ai.shape), ALPHABETS["u16"], "0/t/a", dev)
    assert set(noise.unique().tolist()) == set(range(-16, 17))
    assert torch.equal(ya, (((ai + noise).double() @ bi.double().T) * sa.double()[:, None] * sb.double()[None, :]))
    assert not torch.equal(noise, draw(tuple(ai.shape), ALPHABETS["u16"], "1/t/a", dev))
    for m in ("int8@w:pm1", "int8@a:pm13", "int8_rot@wa:pm57:s2", "e4m3i16@wa:u8", "e4m3i16_rot@w:u4"):
        y, rec = edited_product(m, xf, wf, {}, "t")
        out[m] = {"rel_err": float(torch.linalg.norm(y - xf.double() @ wf.double().T) / torch.linalg.norm(x @ w.T)), **rec}
        assert rec["guaranteed"] and rec["exact"]
    try:
        parse_edit("e4m3i16@w:u16")
        raise AssertionError("u16 fits e4m3i16")
    except ValueError:
        pass
    L = transformers.models.llama.modeling_llama
    torch.manual_seed(0)
    cfg = L.LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=8,
                        num_key_value_heads=2, max_position_embeddings=1024, tie_word_embeddings=False)
    model = L.LlamaForCausalLM(cfg).to(torch.bfloat16).to(dev).eval()
    ids = torch.randint(0, 512, (1, 64), device=dev)
    out["tiny"] = {m: {k: v for k, v in evaluate(model, ids, m).items() if k not in ("per_linear", "worst_linear")}
                   for m in ("bf16", "int8", "e2m1i_rot", "nvfp4", "fp8_block", "int8@-", "int8_rot@wa:u2:s1")}
    assert abs(out["tiny"]["int8@-"]["ppl"] - out["tiny"]["int8"]["ppl"]) < 1e-3 * out["tiny"]["int8"]["ppl"]
    return out


#: sm_120 dense tensor rate (MACs/SM/clk, note:20261015T0608Z-draft-census-sm120) of the instruction each format runs on
RATE = {"int8": (1024, "IMMA s8"), "e4m3i16": (1024, "QMMA e4m3"), "e4m3ik": (1024, "QMMA e4m3"),
        "e4m3ika": (1024, "QMMA e4m3"), "e4m3i448": (1024, "QMMA e4m3"), "e2m1i": (2048, "OMMA.SF nvf4, unit scales"), "fp8_tensor": (1024, "QMMA e4m3"),
        "fp8_token": (1024, "QMMA e4m3"), "fp8_block": (1024, "QMMA e4m3 + rescale per 128"),
        "nvfp4": (2048, "OMMA.SF nvf4"), "e2m1b": (2048, "OMMA.SF nvf4, power-of-two scales")}


def table(results: list[dict]) -> str:
    rows = ["| format | exact? (k M^2 <= acc; observed bound / acc) | perplexity | worst linear: rel. error (layer.linear) "
            "| tensor rate, MACs/SM/clk |", "|---|---|---:|---|---:|"]
    for r in results:
        m = r["method"]
        base = m.split("_rot")[0].split("_wclip")[0]
        w = r.get("worst_linear")
        if r.get("max_bound_over_acc") is None:
            ex = "n/a" if m == "bf16" else "no"
        else:
            ex = f"{'yes' if r['guaranteed_all'] else 'no'}; {r['max_bound_over_acc']:.3f}"
        worst = f"{w['rel_err']:.4f} ({w['layer']}.{w['linear'].split('.')[-1]})" if w else "-"
        rate = f"{RATE[base][0]:,} ({RATE[base][1]})" if base in RATE else "512 (HMMA bf16)"
        rows.append(f"| {m} | {ex} | {r['ppl']:.3f} | {worst} | {rate} |")
    return "\n".join(rows)


def edit_table(results: list[dict]) -> str:
    """The unpeeled-edit table: perplexity (mean, and min-max over seeds when there are several) with its change against
    the unedited int8 grid and bf16, and whether it is under 1% and 10% of each."""
    ppl = {r["method"]: r for r in results}
    base = {"bf16": ppl["bf16"]["ppl"], "int8": (ppl.get("int8@-") or ppl["int8"])["ppl"]}
    groups: dict[str, dict[int, dict]] = {}
    for m, r in ppl.items():
        if "@" in m:
            e = parse_edit(m)
            groups.setdefault(m.split(":s")[0] if e["edit"] != "-" else m, {})[e["seed"]] = r
    order = {g: i for i, g in enumerate(("int8", "int8_rot", "e4m3i16", "e4m3i16_rot"))}

    def key(g: str):
        e = parse_edit(g)
        return order[g.split("@")[0]], ("-", "w", "a", "wa").index(e["edit"]), e["bits"], max(e["values"])

    def under(d: float) -> str:
        return "<1%" if d < 1 else "<10%" if d < 10 else "-"

    rows = ["| grid | edit | alphabet | max abs e | entropy, bits/entry | e std, LSB | mean linear rel. error | perplexity | seeds "
            "| Δ% vs int8 | Δ% vs bf16 | under (int8 / bf16) |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for g in sorted(groups, key=key):
        e, runs = parse_edit(g), groups[g]
        p = [r["ppl"] for r in runs.values()]
        mean = float(np.mean(p))
        spread = f" ({min(p):.3f}-{max(p):.3f})" if len(p) > 1 else ""
        di, db = 100 * (mean / base["int8"] - 1), 100 * (mean / base["bf16"] - 1)
        std = math.sqrt(float(np.mean(np.square(e["values"]))))
        rel = float(np.mean([r["mean_rel_err"] for r in runs.values()]))
        rows.append(f"| {g.split('@')[0]} | {e['edit']} | {e['alphabet']} | {max(e['values'])} | {e['bits']:.2f} | {std:.2f} "
                    f"| {rel:.4f} | {mean:.3f}{spread} | {len(p)} | {di:+.2f} | {db:+.2f} | {under(di)} / {under(db)} |")
    return "\n".join(rows)


def main(argv: list[str]) -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    if argv[0] == "--selftest":
        print(json.dumps(selftest(), indent=1))
        return 0
    if argv[0] in ("--table", "--edit-table"):
        rows = [json.loads(line) for p in argv[1:] for line in Path(p).read_text().splitlines() if line.startswith("{")]
        print((table if argv[0] == "--table" else edit_table)(rows))
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
