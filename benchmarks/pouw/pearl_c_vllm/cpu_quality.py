"""Pearl+'s output quality on a CPU, the LM head's product included: `e2e.py --perplexity`'s eval (WikiText-2's test split,
`e2e.WIKITEXT`, cut by `e2e.wikitext_windows` into 141 windows of BOS and 2,047 tokens) on Llama 3.2 1B's BF16 checkpoint,
through Hugging Face's Llama in BF16, with each product computed by a PyTorch emulation of ``pearl-c-h100-v1-h3``'s served y.

  python benchmarks/pouw/pearl_c_vllm/cpu_quality.py --model DIR --mode MODE [--windows N] --out DIR
  python benchmarks/pouw/pearl_c_vllm/cpu_quality.py --model DIR --check --out DIR
  python benchmarks/pouw/pearl_c_vllm/cpu_quality.py --summary DIR... --record QUALITY_V1_JSON --out DIR

Modes (`MODES`), each one pass over the same windows, the noise seeded alike in every mode:
- ``bf16``: every product bf16(x·Wᵀ) accumulated in FP32, as vLLM's BF16 GEMM serves it (the run of record's ``bf16``);
- ``pearl``: Pearl+ on the products vLLM's ``LinearBase`` holds (qkv, o, gate_up, down, with q, k, v and gate, up each one
  product on their joined weight as vLLM fuses them), the LM head in BF16 (the run of record's ``pearlc-v1``);
- ``pearl-head``: ``pearl`` with the LM head's product (k 2,048, n 128,256 = 64 x 2,004, the tied embedding) under Pearl+
  too.  Its linears draw the same noise as ``pearl``'s, so the two differ by the head alone.

The emulation, per product (x: m x k BF16, W: n x k BF16), against `pearl_c` (forming v1, -h3):
- exact: each row's s, ρ, α and β in `s5_stats`'s FP32 order; E′ = e4m3(β·E) (`eprime`); the forming's atom per element,
  α·x + E′·F_t as `HOPPER_E4M3_K32` sums it (`atom`: its 14-bit truncating adder, which moves about 0.2% of the codes
  against an FP32 sum), and the cast (`pearl_kw.f32_to_fp8`, saturating at 448), so A′ and B̃ are the scheme's codes; the
  peel factors' chains A′·F_B, B̃·F_A and G (`chain`, promoted every 128 products) and E′_B·G's FMA chain, and their BF16
  roundings; the served unscale RN_f32(RN_f32(U·RN_f32(1/α_A))·RN_f32(1/α_B)) (`PearlC.useful`), then BF16;
- FP32 sums in place of the H100's for C̃ = A′·B̃ᵀ and the peel's four BF16 steps, whose exact form is an elementwise
  replay of the whole GEMM.  ``--check`` measures how far that moves the served y on the model's own tiles against
  `fast_tile`'s exact forms (relative RMS near 1e-3 of y, against Pearl+'s 0.05 from x·Wᵀ), and holds the error against
  x·Wᵀ to the device gate's (`e2e.gate`, relative RMS); ``--atoms fp32`` takes FP32 sums for the forming and chains too;
- the noise lines are `pearl_c.sample_line`'s rule (`line_codes`) over uniform bytes from a seeded generator in place of
  SHA-512's: one E_A line a row of A each call (-h3 seeds each row on its own digest), and per weight E_B a row and the
  bases F_A and F_B, at norms 256 and 16.

Each mode also counts the rows Pearl+'s rules exclude (the noise floor ρ·α ≥ 1 and `live_row`, here in float64): every
weight's B rows once, and the A rows of every call, linears and head apart.  `--summary` reads the modes' files and the
run of record's ``quality-v1.json`` and writes the perplexities, their ratios with a standard error over the windows, and
top-1 agreement, with the BF16 baseline's per-position agreement with the record's as the eval's identity check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from verity.protocols.accounting.pouw.prover.schemes import pearl_c as C  # noqa: E402
from verity.protocols.accounting.pouw.prover.schemes import pearl_kw as P  # noqa: E402

MODES = ("bf16", "pearl", "pearl-head")
SEED = 20261114
E_NORM, F_NORM = P.NOISE_NORM, C.LINE_NORM_V1
_F32 = torch.float32
QMAX = torch.tensor(P.f32_of_bits(C.S5_QMAX), dtype=_F32)
FLOOR = torch.tensor(P.f32_of_bits(C.S5_FLOOR), dtype=_F32)
FOUR = torch.tensor(P.f32_of_bits(C.S5_FOUR), dtype=_F32)
LINE = torch.tensor(P.f32_of_bits(C.S5_LINE), dtype=_F32)


def e4m3(x: torch.Tensor) -> torch.Tensor:
    """Pearl's cast (`pearl_kw.f32_to_fp8`: nearest even, saturating at 448) on FP32 values, returned as FP32 values."""
    if not torch.isfinite(x).all():
        raise ValueError("a non-finite FP32 value reaches the E4M3 cast")
    return x.clamp(-QMAX, QMAX).to(torch.float8_e4m3fn).to(_F32)


def line_codes(raw: torch.Tensor, norm: float) -> torch.Tensor:
    """`pearl_c.sample_line`'s rule on rows of 32 bytes: xs = ±((b & 0x7F) + 1), the BF16 scale
    bf16_div(bf16(norm·32), bf16(isqrt(Σ xs² · 32 · 32))), each code e4m3(bf16_mul(bf16(x), scale)); the codes' values."""
    b = raw.to(torch.int64)
    xs = torch.where(b >= 128, -1, 1) * ((b & 0x7F) + 1)
    ss = (xs * xs).sum(1) * 1024
    root = torch.sqrt(ss.to(torch.float64)).floor().to(torch.int64)
    root = root - (root * root > ss).to(torch.int64) + ((root + 1) * (root + 1) <= ss).to(torch.int64)
    num = torch.tensor(norm * 32, dtype=torch.float64).to(_F32).to(torch.bfloat16).to(torch.float64)
    den = root.to(torch.float64).to(_F32).to(torch.bfloat16).to(torch.float64)
    scale = (num / den).to(_F32).to(torch.bfloat16).to(torch.float64)
    vals = (xs.to(torch.float64) * scale[:, None]).to(_F32).to(torch.bfloat16).to(_F32)
    return e4m3(vals)


def lines(gen: torch.Generator, rows: int, norm: float) -> torch.Tensor:
    return line_codes(torch.randint(0, 256, (rows, 32), generator=gen, dtype=torch.uint8), norm)


_NONE = -1000
#: the H100 E4M3 atom (`verity.circuits.silicon.models.HOPPER_E4M3_K32`): its adder's width and exponent floor
WIDTH, ZERO_EXPONENT = 14, -139
#: elements of an atom's (lanes, 32) terms per block, so a block's temporaries stay near 64 MB
_BLOCK = 1 << 24


def code_exponent(v: torch.Tensor) -> torch.Tensor:
    """An E4M3 value's exponent as the atom's product term reads it (a subnormal's is −6), `_NONE` for zero."""
    return torch.where(v == 0, _NONE, (torch.frexp(v)[1] - 1).clamp(min=-6))


def atom(acc: torch.Tensor, a: torch.Tensor, ea: torch.Tensor, b: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
    """`HOPPER_E4M3_K32.step` on every lane: acc (L,) FP32 values; a, b (L, 32) E4M3 values (broadcast) and their
    `code_exponent`s.  Each term (acc, and each product at exponent ea + eb) is cut toward zero to the grid
    q = 2^(emax − 13) of the largest term's exponent, the cut terms summed exactly, and the sum cut to 14 significant
    bits.  Every step is exact in FP32: a product has 8 significant bits, a cut term is an integer below 2^15 in units of q."""
    if not torch.isfinite(acc).all():
        raise ValueError("a non-finite accumulator reaches the atom")
    ep = ea + eb
    e_acc = torch.where(acc == 0, _NONE, (torch.frexp(acc)[1] - 1).clamp(min=-126))
    emax = torch.maximum(ep.amax(-1), e_acc).clamp(min=ZERO_EXPONENT)
    if bool((emax < -100).logical_and((acc != 0) | (ep.amax(-1) > _NONE // 2)).any()):
        raise ValueError("an atom's terms below 2^-100, outside the emulation's FP32 grid")
    emax = emax.clamp(min=-100)
    q = torch.ldexp(torch.ones_like(acc), emax - (WIDTH - 1))
    t = torch.trunc((a * b) / q[:, None]).sum(-1) + torch.trunc(acc / q)
    mag = t.abs()
    cut = (torch.frexp(mag)[1] - WIDTH).clamp(min=0)
    mag = torch.ldexp(torch.floor(torch.ldexp(mag, -cut)), cut)
    return torch.copysign(mag, t) * q + 0.0


def atoms(acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The forming's atom on every element: D[i, t] = atom(acc[i, t], a[i], b[t]) for acc (R, K), a (R, 32), b (K, 32)."""
    r, k = acc.shape
    ea, eb = code_exponent(a), code_exponent(b)
    out = torch.empty_like(acc)
    rows = max(1, _BLOCK // (32 * k))
    for i in range(0, r, rows):
        n = min(rows, r - i)
        out[i:i + n] = atom(acc[i:i + n].reshape(-1), a[i:i + n, None, :].expand(n, k, 32).reshape(-1, 32),
                            ea[i:i + n, None, :].expand(n, k, 32).reshape(-1, 32),
                            b[None].expand(n, k, 32).reshape(-1, 32), eb[None].expand(n, k, 32).reshape(-1, 32)).view(n, k)
    return out


def chain(a: torch.Tensor, b: torch.Tensor, g: int = 4) -> torch.Tensor:
    """`pearl_c.chain` on every pair: out[i, j] = chain(a[i], b[:, j]) for a (R, K) and b (K, J) E4M3 values, J small: per
    window of `g` atoms (128 products) a +0 accumulator chains them, then one FP32 nearest-even add promotes it into the
    total, and a zero total is +0."""
    r, k = a.shape
    j = b.shape[1]
    ea, eb = code_exponent(a), code_exponent(b)
    out = torch.empty(r, j, dtype=_F32)
    rows = max(1, _BLOCK // (32 * j))
    for i in range(0, r, rows):
        n = min(rows, r - i)
        total = torch.zeros(n * j, dtype=_F32)
        for ws in range(0, k, 32 * g):
            local = torch.zeros(n * j, dtype=_F32)
            for s in range(ws, min(ws + 32 * g, k), 32):
                local = atom(local, a[i:i + n, None, s:s + 32].expand(n, j, 32).reshape(-1, 32),
                             ea[i:i + n, None, s:s + 32].expand(n, j, 32).reshape(-1, 32),
                             b[s:s + 32].T[None].expand(n, j, 32).reshape(-1, 32),
                             eb[s:s + 32].T[None].expand(n, j, 32).reshape(-1, 32))
            total = total + local + 0.0
        out[i:i + n] = total.view(n, j)
    return out


def fma_chain(a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """out[i, q] = the FP32 FMA chain from +0 of a[i, p] · g[p, q] in p order (a BF16, g FP32): each FMA's exact
    a·g + acc in float64, rounded to FP32."""
    acc = torch.zeros(a.shape[0], g.shape[1], dtype=torch.float64)
    a64, g64 = a.to(torch.float64), g.to(torch.float64)
    for p in range(a.shape[1]):
        acc = (a64[:, p, None] * g64[None, p, :] + acc).to(_F32).to(torch.float64)
    return acc.to(_F32)


def row_stats(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """`pearl_c.s5_stats` on each row of FP32 values: s = max|x|; S over every 8th position in 32 FMA chains from +0
    (chain j takes positions 8(j + 32i), i ascending), then the butterfly; ρ = sqrt(S / ⌈k/8⌉); α = 448 / max(s + 4ρ, 2^-32);
    β = (ρ·α)·√r/(256·16).  An FMA is x² + acc in float64 (x² exact there), then rounded to FP32."""
    r, k = x.shape
    s = x.abs().amax(1)
    sub = x[:, ::C.S5_STRIDE].to(torch.float64)
    kk = sub.shape[1]
    pad = -kk % 32
    if pad:
        sub = torch.cat([sub, sub.new_zeros(r, pad)], 1)
    v = sub.reshape(r, -1, 32)
    part = torch.zeros(r, 32, dtype=torch.float64)
    for t in range(v.shape[1]):
        part = (v[:, t] * v[:, t] + part).to(_F32).to(torch.float64)
    p = part.to(_F32)
    lanes = torch.arange(32)
    for o in (16, 8, 4, 2, 1):
        p = p + p[:, lanes ^ o]
    rho = torch.sqrt(p[:, 0] / torch.tensor(float(kk), dtype=_F32))
    alpha = QMAX / torch.maximum(s + FOUR * rho, FLOOR)
    beta = (rho * alpha) * LINE
    return s, rho, alpha, beta


def excluded(x: torch.Tensor, rho: torch.Tensor, alpha: torch.Tensor) -> tuple[int, int]:
    """Rows under the noise floor (ρ·α < 1, the exact product) and rows failing `live_row` (more than k/64 entries with
    x² ≥ 64·ρ², ρ² = Σ_{p ≡ 0 (8)} x_p² / ⌈k/8⌉), in float64."""
    k = x.shape[1]
    floor = (rho.to(torch.float64) * alpha.to(torch.float64) < P.SIGMA_MIN)
    xd = x.to(torch.float64)
    sub = xd[:, ::C.S5_STRIDE]
    rho2 = (sub * sub).sum(1) / sub.shape[1]
    dead = ((xd * xd) >= 64 * rho2[:, None]).sum(1)
    live = 64 * dead <= k
    return int(floor.sum()), int((~live).sum())


#: ``exact``: the forming's atoms and the peel factors' chains as the H100 computes them (`atoms`, `chain`, `fma_chain`);
#: ``fp32``: FP32 sums in their place (`--atoms`)
ATOMS = {"mode": "exact"}


def _chain(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return chain(a, b) if ATOMS["mode"] == "exact" else a @ b


def form(x: torch.Tensor, e: torch.Tensor, f: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """`pearl_c.form_v1`: codes = e4m3(atom(α·x, E′, F_t)) per element, with E′ = e4m3(β·E)."""
    _, rho, alpha, beta = row_stats(x)
    ep = e4m3(beta[:, None] * e)
    d = alpha[:, None] * x
    if ATOMS["mode"] == "exact":
        d = atoms(d, ep, f)
    else:
        d.add_(ep @ f.T)
    return e4m3(d), alpha, ep, rho


class Weight:
    """One weight's side, once: B̃ and α_B (`form` with E_B and F_B), the bases, and P_B = [bf16(E′_B·G − B̃·F_A) ‖ −E′_B]
    with G = F_Bᵀ·F_A, each product a `chain` and E′_B·G an FMA chain (`fma_chain`)."""

    def __init__(self, w: torch.Tensor, gen: torch.Generator, e_b: torch.Tensor | None = None,
                 f_a: torch.Tensor | None = None, f_b: torch.Tensor | None = None):
        n, k = w.shape
        w = w.to(_F32)
        self.f_a = lines(gen, k, F_NORM) if f_a is None else f_a
        self.f_b = lines(gen, k, F_NORM) if f_b is None else f_b
        e_b = lines(gen, n, E_NORM) if e_b is None else e_b
        self.codes, alpha_b, ep_b, rho_b = form(w, e_b, self.f_b)
        self.excluded = excluded(w, rho_b, alpha_b)
        g = _chain(self.f_b.T.contiguous(), self.f_a)
        eg = fma_chain(ep_b, g) if ATOMS["mode"] == "exact" else ep_b @ g
        mid = (eg - _chain(self.codes, self.f_a)).to(torch.bfloat16).to(_F32)
        self.pb = torch.cat([mid, -ep_b], 1)
        self.alpha_b, self.inv_b = alpha_b, 1 / alpha_b

    def product(self, x: torch.Tensor, gen: torch.Generator, e_a: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple]:
        """The served y of x (m x k BF16) as BF16, and the call's excluded A rows.  C̃ = A′·B̃ᵀ and the peel are FP32 sums."""
        xa = x.to(_F32)
        e_a = lines(gen, xa.shape[0], E_NORM) if e_a is None else e_a
        codes, alpha_a, ep_a, rho_a = form(xa, e_a, self.f_a)
        u = codes @ self.codes.T
        pa = torch.cat([ep_a, _chain(codes, self.f_b).to(torch.bfloat16).to(_F32)], 1)
        u.addmm_(pa, self.pb.T)
        u.mul_((1 / alpha_a)[:, None]).mul_(self.inv_b[None, :])
        return u.to(torch.bfloat16), excluded(xa, rho_a, alpha_a)


def _seed(*parts: int) -> int:
    return int.from_bytes(hashlib.sha512(b"cpu-quality/" + b"/".join(str(p).encode() for p in parts)).digest()[:8], "little") >> 1


class Group:
    """One served product: a weight, or the joined weights of one fused vLLM layer (q, k, v; gate, up), whose members each take
    their columns of one product of the shared input."""

    def __init__(self, index: int, kind: str, weights: list[torch.Tensor], state: dict):
        self.index, self.kind, self.state = index, kind, state
        self.w = torch.cat([w.detach() for w in weights], 0).contiguous()
        self.w32 = self.w.to(_F32)
        self.n, self.k = self.w.shape
        self.members, self.left, self._x, self._y = len(weights), 0, None, None
        self.pearl: Weight | None = None
        self.calls = 0
        self.tap = None

    def pearl_on(self) -> bool:
        mode = self.state["mode"]
        return mode == "pearl-head" or (mode == "pearl" and self.kind == "linear")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.tap is not None:
            self.tap(self, x)
        if self.left and x is self._x:
            self.left -= 1
            y = self._y
            if not self.left:
                self._x = self._y = None
            return y
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.k)
        if self.pearl_on():
            if self.pearl is None:
                self.pearl = Weight(self.w, torch.Generator().manual_seed(_seed(SEED, "weight", self.index)))
                self.state["excluded_b"][self.kind] = [a + b for a, b in zip(self.state["excluded_b"][self.kind], self.pearl.excluded)]
            gen = torch.Generator().manual_seed(_seed(SEED, "call", self.state["window"], self.index, self.calls))
            y, ex = self.pearl.product(x2, gen)
            acc = self.state["excluded_a"][self.kind]
            acc[0], acc[1], acc[2] = acc[0] + ex[0], acc[1] + ex[1], acc[2] + x2.shape[0]
        else:
            y = (x2.to(_F32) @ self.w32.T).to(torch.bfloat16)
        self.calls += 1
        y = y.reshape(*lead, self.n)
        if self.members > 1:
            self._x, self._y, self.left = x, y, self.members - 1
        return y


class Member(torch.nn.Module):
    def __init__(self, group: Group, lo: int, hi: int):
        super().__init__()
        self.group, self.lo, self.hi = group, lo, hi

    def forward(self, x):
        y = self.group(x)
        return y if (self.lo, self.hi) == (0, self.group.n) else y[..., self.lo:self.hi]


def install(model, state: dict) -> list[Group]:
    """Every product of the Llama model replaced by a `Group`: per layer qkv, o, gate_up and down (vLLM's ``LinearBase``
    sites, in its order), then the LM head."""
    groups: list[Group] = []

    def fuse(owner, names: list[str], kind: str):
        mods = [getattr(owner, n) for n in names]
        if any(getattr(m, "bias", None) is not None for m in mods):
            raise SystemExit(f"{names}: a bias, which Llama 3.2 has none of")
        g = Group(len(groups), kind, [m.weight for m in mods], state)
        groups.append(g)
        lo = 0
        for name, m in zip(names, mods):
            setattr(owner, name, Member(g, lo, lo + m.weight.shape[0]))
            lo += m.weight.shape[0]

    for layer in model.model.layers:
        fuse(layer.self_attn, ["q_proj", "k_proj", "v_proj"], "linear")
        fuse(layer.self_attn, ["o_proj"], "linear")
        fuse(layer.mlp, ["gate_proj", "up_proj"], "linear")
        fuse(layer.mlp, ["down_proj"], "linear")
    fuse(model, ["lm_head"], "head")
    return groups


def load(model_dir: str):
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.eval()
    return model


def new_state(mode: str) -> dict:
    return {"mode": mode, "window": 0, "excluded_b": {"linear": [0, 0], "head": [0, 0]},
            "excluded_a": {"linear": [0, 0, 0], "head": [0, 0, 0]}}


def run_mode(args) -> int:
    """One forward per window and every mode of ``--mode`` (comma-separated) on it: modes listed together must agree on
    every product but the head (pearl and pearl-head), which then share the layers' Pearl+ products, noise included, and
    differ only in the head's."""
    import e2e
    modes = args.mode.split(",")
    if any(m not in MODES for m in modes) or len(set(modes)) != len(modes) or (
            len(modes) > 1 and not set(modes) <= {"pearl", "pearl-head"}):
        raise SystemExit(f"--mode {args.mode}: one of {MODES}, or pearl,pearl-head (which share every product but the head)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    windows = e2e.wikitext_windows(args.model, args.windows)
    ids_digest = hashlib.sha256(json.dumps(windows).encode()).hexdigest()
    if args.range:
        lo, hi = map(int, args.range.split(":"))
    else:
        shard, shards = map(int, args.part.split("/"))
        if not 0 <= shard < shards:
            raise SystemExit(f"--part {args.part}: I/N with 0 <= I < N")
        lo, hi = shard * len(windows) // shards, (shard + 1) * len(windows) // shards
    if not 0 <= lo < hi <= len(windows):
        raise SystemExit(f"windows {lo}:{hi}: not inside 0:{len(windows)}")
    print(json.dumps({"mode": args.mode, "windows": len(windows), "range": [lo, hi], "tokens": sum(len(w) - 1 for w in windows),
                      "windows_sha256": ids_digest, "threads": torch.get_num_threads()}), flush=True)
    model = load(args.model)
    state = new_state(modes[0])
    groups = install(model, state)
    acc = {m: {"nll": 0.0, "count": 0, "top1": [], "per_window": []} for m in modes}
    t0 = time.perf_counter()
    for i in range(lo, hi):
        w = windows[i]
        state["window"], state["mode"] = i, modes[0]
        for g in groups:
            g.calls = 0
        ids = torch.tensor([w])
        with torch.inference_mode():
            hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
            for mode in modes:
                state["mode"] = mode
                groups[-1].calls = 0
                lp = model.lm_head(hidden)[0, :-1].to(_F32).log_softmax(-1)
                got = lp.gather(1, ids[0, 1:, None])[:, 0].to(torch.float64)
                a = acc[mode]
                a["top1"] += lp.argmax(-1).tolist()
                part = -float(got.sum())
                a["nll"], a["count"] = a["nll"] + part, a["count"] + got.shape[0]
                a["per_window"].append(part)
                print(f"WINDOW {i + 1}/{len(windows)} {mode} nll {part:.3f} ppl_so_far {math.exp(a['nll'] / a['count']):.4f} "
                      f"t {time.perf_counter() - t0:.0f}s", flush=True)
        docs = write(out, modes, acc, state, groups, args, len(windows), ids_digest, lo, i + 1, t0)
    for doc in docs:
        print(json.dumps({k: v for k, v in doc.items() if k not in ("per_window_nll", "top1")}), flush=True)
    return 0


def write(out, modes, acc, state, groups, args, windows, ids_digest, lo, done, t0) -> list[dict]:
    """Each mode's windows lo:done so far, rewritten after every window, so a run cut short keeps what it finished."""
    docs = []
    for mode in modes:
        a = acc[mode]
        head = mode == "pearl-head"
        doc = {"schema": "verity-pouw/pearl-c-cpu-quality/v1", "mode": mode, "scheme": None if mode == "bf16" else
               "pearl-c-h100-v1-h3 (emulated)", "model": args.model, "windows": windows, "windows_sha256": ids_digest,
               "range": [lo, done], "run_with": modes,
               "seed": SEED, "atoms": ATOMS["mode"], "threads": torch.get_num_threads(), "seconds": time.perf_counter() - t0,
               "nll": a["nll"], "tokens": a["count"], "perplexity": math.exp(a["nll"] / a["count"]),
               "per_window_nll": a["per_window"], "top1": a["top1"],
               "excluded_b_rows": {k: {"floor": v[0] if head or k == "linear" else 0, "live": v[1] if head or k == "linear" else 0}
                                   for k, v in state["excluded_b"].items()},
               "excluded_a_rows": {k: dict(zip(("floor", "live", "rows"), v if head or k == "linear" else [0, 0, 0]))
                                   for k, v in state["excluded_a"].items()},
               "rows_b": {"linear": sum(g.n for g in groups if g.kind == "linear"), "head": groups[-1].n}}
        path = out / f"quality-{mode}-w{lo:03d}.json"
        path.with_suffix(".tmp").write_text(json.dumps(doc))
        path.with_suffix(".tmp").replace(path)
        docs.append(doc)
    return docs


def _words(t: torch.Tensor) -> list[list[int]]:
    return (t.to(_F32).contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF).tolist()


def _values(codes: list[list[int]]) -> torch.Tensor:
    return torch.tensor([[P.fp8_to_f32(c) for c in row] for row in codes], dtype=_F32)


def reference_tile(x: torch.Tensor, w: torch.Tensor, tag: bytes) -> dict:
    """One tile (x's rows by w's rows) through `pearl_c`'s forming, chain and peel under `fast_tile`'s exact batch forms, with
    -h3's own noise lines; and through `Weight` on the same lines.  Returns how far the emulation's A′, B̃ and served y are."""
    import fast_tile
    import numpy as np
    k = x.shape[1]
    seed_b = C.S512Seed(hashlib.sha512(b"seed-b/" + tag).digest()[:32])
    e_a = [C.sample_line(C.S512Seed(hashlib.sha512(b"row/" + tag + bytes([i])).digest()[:32]), 0, 0, 0) for i in range(x.shape[0])]
    e_b = [C.sample_line(seed_b, 1, 0, j) for j in range(w.shape[0])]
    fa, fb = C._basis(seed_b, 0, k, F_NORM), C._basis(seed_b, 1, k, F_NORM)
    a = C.form_v1(_words(x), e_a, fa, C.H100)
    b = C.form_v1(_words(w), e_b, fb, C.H100)
    pa = C.peel_factors_a(a.codes, C.scaled_noise(a, e_a), seed_b, k, F_NORM, C.H100)
    pb = C.peel_factors_b(b.codes, C.scaled_noise(b, e_b), seed_b, k, F_NORM, C.H100)
    u = fast_tile.peel_batch(fast_tile._pairs(a.codes, b.codes, C.H100), pa, pb, C.H100)
    f32 = lambda v: torch.from_numpy(np.asarray(v, dtype=np.uint32).view(np.float32).copy())  # noqa: E731
    inv_a, inv_b = 1 / f32(a.alpha), 1 / f32(b.alpha)
    y_ref = ((f32(u) * inv_a[:, None]) * inv_b[None, :]).to(torch.bfloat16).to(_F32)

    emu = Weight(w, torch.Generator(), e_b=_values(e_b), f_a=_values(fa), f_b=_values(fb))
    y_emu, _ = emu.product(x, torch.Generator(), e_a=_values(e_a))
    y_emu = y_emu.to(_F32)
    codes_a, alpha_a, _, _ = form(x.to(_F32), _values(e_a), _values(fa))
    exact = x.to(_F32) @ w.to(_F32).T
    rms = lambda t: float(t.pow(2).mean().sqrt())  # noqa: E731
    differ = [[j, float(emu.alpha_b[j]), float(f32(b.alpha[j])), float(f32(C.s5_stats(_words(w[j:j + 1])[0])[2]))]
              for j in torch.nonzero(emu.alpha_b != f32(b.alpha))[:, 0].tolist()[:8]]
    return {"rows": x.shape[0], "cols": w.shape[0], "k": k, "alpha_b_differ": differ,
            "alpha_a_equal": float((alpha_a == f32(a.alpha)).float().mean()),
            "alpha_b_equal": float((emu.alpha_b == f32(b.alpha)).float().mean()),
            "a_codes_equal": float((codes_a == _values(a.codes)).float().mean()),
            "b_codes_equal": float((emu.codes == _values(b.codes)).float().mean()),
            "y_bf16_equal": float((y_emu == y_ref).float().mean()),
            "y_emu_vs_ref_rel_rms": rms(y_emu - y_ref) / rms(y_ref),
            "y_ref_vs_exact_rel_rms": rms(y_ref - exact) / rms(exact),
            "y_emu_vs_exact_rel_rms": rms(y_emu - exact) / rms(exact)}


def check(args) -> int:
    """The emulation against `pearl_c`'s exact forms on the model's own tiles (its activations at layer 0's qkv and down
    and at the head, window 0), and the device gate's relative RMS (`e2e.gate`: m BF16 rows of N(0, 1) against each distinct
    (n, k)'s first weight, and the head's)."""
    import e2e
    import fast_tile
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    undo = fast_tile.install(C.PearlC(forming="v1"))
    model = load(args.model)
    groups = install(model, new_state("bf16"))
    seen: dict[int, torch.Tensor] = {}
    for g in (groups[0], groups[3], groups[-1]):
        g.tap = lambda g, x: seen.setdefault(g.index, x.reshape(-1, g.k)[:64].clone())
    window = e2e.wikitext_windows(args.model, 1)[0][:64]
    with torch.inference_mode():
        model(input_ids=torch.tensor([window]), use_cache=False)
    tiles = []
    for g, name in ((groups[0], "qkv"), (groups[3], "down"), (groups[-1], "head")):
        x = seen[g.index][16:32]
        cols = [(0, 64)] + ([(g.n - 64, g.n)] if g.kind == "head" else [])
        for lo, hi in cols:
            t0 = time.perf_counter()
            r = reference_tile(x, g.w[lo:hi], f"{name}/{lo}".encode())
            tiles.append({"site": name, "cols": [lo, hi], **r, "seconds": time.perf_counter() - t0})
            print(json.dumps(tiles[-1]), flush=True)
    undo()
    gate, gen = [], torch.Generator().manual_seed(20260930)
    for g in (groups[0], groups[1], groups[2], groups[3], groups[-1]):
        x = torch.randn(512, g.k, generator=gen).to(torch.bfloat16)
        y, _ = Weight(g.w, torch.Generator().manual_seed(g.index)).product(x, torch.Generator().manual_seed(1))
        ref = x.to(_F32) @ g.w32.T
        err = float(((y.to(_F32) - ref).pow(2).sum() / ref.pow(2).sum()).sqrt())
        gate.append({"site": g.index, "kind": g.kind, "n": g.n, "k": g.k, "m": 512, "rel_rms": err})
        print(json.dumps(gate[-1]), flush=True)
    (out / "check.json").write_text(json.dumps({"atoms": ATOMS["mode"], "tiles": tiles, "gate": gate}, indent=1))
    return 0


def join(parts: list[dict]) -> dict:
    """A mode's `--part` outputs as one: their ranges must tile every window once, on one eval set, seed and atoms."""
    parts = sorted(parts, key=lambda d: d["range"][0])
    first = parts[0]
    for key in ("windows", "windows_sha256", "seed", "atoms", "model"):
        if len({json.dumps(d.get(key)) for d in parts}) != 1:
            raise SystemExit(f"{first['mode']}: its parts differ in {key}")
    ranges = [d["range"] for d in parts]
    if ranges[0][0] != 0 or ranges[-1][1] != first["windows"] or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
        raise SystemExit(f"{first['mode']}: its parts' ranges {ranges} don't tile windows 0..{first['windows']}")
    nll, tokens = sum(d["nll"] for d in parts), sum(d["tokens"] for d in parts)
    return {**first, "range": [0, first["windows"]], "nll": nll, "tokens": tokens, "perplexity": math.exp(nll / tokens),
            "seconds": sum(d["seconds"] for d in parts), "parts": len(parts),
            "per_window_nll": [v for d in parts for v in d["per_window_nll"]], "top1": [v for d in parts for v in d["top1"]],
            "excluded_a_rows": {k: {f: sum(d["excluded_a_rows"][k][f] for d in parts) for f in v}
                                for k, v in first["excluded_a_rows"].items()}}


def summary(args) -> int:
    """The modes' perplexities, each ratio to bf16 with a standard error over the windows (the per-window log-ratio's), top-1
    agreement with bf16, and the bf16 baseline's per-position top-1 agreement with the run of record's bf16."""
    parts: dict[str, list[dict]] = {}
    for d in args.summary:
        for p in Path(d).glob("quality-*.json"):
            doc = json.loads(p.read_text())
            parts.setdefault(doc["mode"], []).append(doc)
    docs = {mode: join(ps) for mode, ps in parts.items()}
    record = json.loads(Path(args.record).read_text())["modes"] if args.record else {}
    base = docs["bf16"]
    rows = {}
    for mode, doc in docs.items():
        row = {"perplexity": doc["perplexity"], "tokens": doc["tokens"], "windows_sha256": doc["windows_sha256"],
               "excluded_b_rows": doc["excluded_b_rows"], "excluded_a_rows": doc["excluded_a_rows"], "seconds": doc["seconds"],
               "top1_agreement_vs_bf16": sum(a == b for a, b in zip(doc["top1"], base["top1"], strict=True)) / len(base["top1"])}
        for ref_name, ref in (("bf16", base), ("pearl", docs.get("pearl"))):
            if ref is None or ref is doc:
                continue
            per = [a - b for a, b in zip(doc["per_window_nll"], ref["per_window_nll"], strict=True)]
            nw, tok = len(per), doc["tokens"]
            mean = sum(per) / nw
            sd = math.sqrt(sum((p - mean) ** 2 for p in per) / (nw - 1))
            # log(ppl ratio) = Σ per / tokens; its standard error over the windows
            row[f"vs_{ref_name}"] = {"ratio": math.exp(sum(per) / tok), "added_pct": 100 * (math.exp(sum(per) / tok) - 1),
                                     "added_pct_se": 100 * math.exp(sum(per) / tok) * sd * math.sqrt(nw) / tok}
        rows[mode] = row
    if record:
        rows["bf16"]["top1_agreement_vs_record_bf16"] = sum(
            a == b for a, b in zip(base["top1"], record["bf16"]["top1"], strict=True)) / len(base["top1"])
        if "pearl" in docs:
            rows["pearl"]["top1_agreement_vs_record_pearlc_v1"] = sum(
                a == b for a, b in zip(docs["pearl"]["top1"], record["pearlc-v1"]["top1"], strict=True)) / len(base["top1"])
        rows["record"] = {m: {"perplexity": math.exp(r["nll"] / r["tokens"]), "tokens": r["tokens"]} for m, r in record.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="the checkpoint's directory (unsloth/Llama-3.2-1B at the run of record's revision)")
    ap.add_argument("--mode", help=f"one of {', '.join(MODES)}; or pearl,pearl-head, one pass through the layers for both")
    ap.add_argument("--windows", type=int, help="the first N windows (default all 141)")
    ap.add_argument("--part", default="0/1", metavar="I/N",
                    help="the I-th of N contiguous ranges of the windows; --summary joins a mode's parts")
    ap.add_argument("--range", metavar="A:B", help="windows A to B (exclusive), in place of --part")
    ap.add_argument("--threads", type=int, default=torch.get_num_threads())
    ap.add_argument("--atoms", choices=("exact", "fp32"), default="exact",
                    help="the forming's atoms and the peel factors' chains: the H100's (default), or FP32 sums")
    ap.add_argument("--check", action="store_true", help="the emulation against pearl_c's exact forms, and the gate's error")
    ap.add_argument("--summary", nargs="+", metavar="DIR", help="directories holding the modes' quality-<mode>.json")
    ap.add_argument("--record", help="the run of record's quality-v1.json, for its perplexities and per-position top-1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ATOMS["mode"] = args.atoms
    if args.summary:
        return summary(args)
    if args.check:
        return check(args)
    if not (args.model and args.mode):
        ap.error("--model and --mode (or --check, or --summary)")
    return run_mode(args)


if __name__ == "__main__":
    sys.exit(main())
