"""HT-PROV's audit arithmetic: the zero-knowledge cost of checking one hardening step W_{t+1} = Step(W_t; coins_t,
batch_t) at Llama-3.2-1B's and Llama-3.1-8B's shapes, and of the pass's checked tail.

Every input is a cited constant or a named estimate; the output is arithmetic on them (`python audit.py` prints the
tables and writes `audit.json` beside `--out` if given).

- ANDs a private x private int8 MAC at w = 16: 130.4 (HT-PROTO-U §2, checked there). A bf16 x bf16 MAC into fp32 is
  estimated at `BF16_MAC` (mantissa product plus fp32 alignment, normalisation and rounding).
- Openings: hm96's row_ands(n) = 116,240 + 58,120·⌈(n + 17)/128⌉ for an n-byte leaf (HT-PROTO §2, recalled from
  GD-PROVER), at 4 KiB leaves. SHA-512 expansion of a step's seed into noise: 58,120 ANDs a 128-byte block.
- Prover time: HT-PROTO-U's audit, 14.2G ANDs in 210 s on an RTX PRO 6000 (GD-PROVER's fit), so `S_PER_AND`.

Two ways to check a step:

- **full**: every MAC of the forward and backward over the step's B tokens, and every parameter's update;
- **split**: every parameter's update in full (a corrupted update writes W freely), and the forward and backward
  sampled (`DRAWS` units a step), since a corrupted gradient moves an entry by at most the optimizer's normalised step.

Two one-time checks on the served matrices, priced the same way:

- **end box** (`endbox`): one live dither Z of the committed start, and |q·s − W₀ − Z| ≤ (Δ + ½)·s checked per entry:
  W₀ (int16) and q (int8) opened, `BOX_ARITH` ANDs of fixed-point arithmetic and `BOX_NOISE_BITS` of SHA-512 noise;
  the code-law gate's range counts on the already opened codes at `GATE_ANDS` an entry;
- **spectral gate** (`spectral`): the energy a randomized block Krylov subspace of q captures, seeded by the verifier's
  live coins (k = r + `OVERSAMPLE` probes, one power iteration), which can only underestimate the top-r energy: one
  product with the public probes at `PUBLIC_MAC` ANDs a MAC and three private x private products at `INT8_MAC`.
"""

import argparse
import json
import math
from pathlib import Path

S_PER_AND = 210 / 14.2e9
INT8_MAC = 130.4
BF16_MAC = 800.0
UNIT = 6.69e6
DRAWS = 2773
SHA_BLOCK = 58_120


def row_ands(n):
    return 116_240 + 58_120 * math.ceil((n + 17) / 128)


OPEN = row_ands(4096) / 4096

SHAPES = {
    "llama-3.2-1b": {"layers": 16, "d": 2048, "linears": [(2048, 3072), (2048, 2048), (2048, 16384), (8192, 2048)],
                     "vocab": 128_256, "tied": True},
    "llama-3.1-8b": {"layers": 32, "d": 4096, "linears": [(4096, 6144), (4096, 4096), (4096, 28672), (14336, 4096)],
                     "vocab": 128_256, "tied": False},
}

OPTIMIZERS = {
    # bytes of state a parameter read and written, bytes of gradient read, ANDs of arithmetic, noise bits
    "adamw-fp32": {"state": 12, "grad": 4, "arith": 20_000, "noise_bits": 8},
    "sign-int16": {"state": 2, "grad": 2, "arith": 200, "noise_bits": 2},
}


def shape(name):
    s = SHAPES[name]
    lin = s["layers"] * sum(k * n for k, n in s["linears"])
    emb = s["vocab"] * s["d"] * (1 if s["tied"] else 2)
    return {"linear_params": lin, "params": lin + emb + 2 * s["layers"] * s["d"],
            "macs_token_fwd": lin + s["vocab"] * s["d"], "attn_macs_token_fwd": lambda ctx: s["layers"] * ctx * s["d"]}


def update_ands(opt, noise=True):
    o = OPTIMIZERS[opt]
    return OPEN * (2 * o["state"] + o["grad"]) + o["arith"] + (SHA_BLOCK * o["noise_bits"] / 1024 if noise else 0)


BOX_ARITH = 300
BOX_NOISE_BITS = 16
GATE_ANDS = 60
PUBLIC_MAC = 32
OVERSAMPLE = 10


def endbox(name):
    sh = shape(name)
    per = OPEN * 3 + BOX_ARITH + SHA_BLOCK * BOX_NOISE_BITS / 1024
    return {"ands_per_param": per, "gpu_h": per * sh["params"] * S_PER_AND / 3600,
            "code_law_gpu_h": GATE_ANDS * sh["linear_params"] * S_PER_AND / 3600}


def spectral(name, r):
    sh = shape(name)
    ands = sh["linear_params"] * (r + OVERSAMPLE) * (PUBLIC_MAC + 3 * INT8_MAC)
    return {"r": r, "ands": ands, "gpu_h": ands * S_PER_AND / 3600}


def step(name, opt, tokens, ctx, mac):
    sh = shape(name)
    macs = 3 * tokens * (sh["macs_token_fwd"] + sh["attn_macs_token_fwd"](ctx))
    upd = update_ands(opt) * sh["params"]
    full = macs * mac + upd
    split = upd + DRAWS * UNIT
    return {"params": sh["params"], "linear_params": sh["linear_params"], "macs": macs, "update_ands": upd,
            "full_ands": full, "split_ands": split, "full_gpu_h": full * S_PER_AND / 3600,
            "split_gpu_h": split * S_PER_AND / 3600}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="4096,65536,1048576", help="tokens a step")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--taus", default="1,10,30,100")
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = []
    print(f"openings {OPEN:.1f} ANDs a byte; {S_PER_AND:.3g} s an AND")
    for name in SHAPES:
        for opt in OPTIMIZERS:
            for tok in (int(x) for x in args.tokens.split(",")):
                for mac_name, mac in (("int8", INT8_MAC), ("bf16", BF16_MAC)):
                    r = {"shape": name, "optimizer": opt, "tokens": tok, "mac": mac_name, **step(name, opt, tok, args.ctx, mac)}
                    rows.append(r)
                    print(f"{name} {opt} B={tok} {mac_name}: full {r['full_gpu_h']:.3g} GPU-h a step, split "
                          f"{r['split_gpu_h']:.3g} (update {r['update_ands'] / r['params']:.0f} ANDs a parameter)")
    taus = [int(x) for x in args.taus.split(",")]
    tail = {f"{r['shape']}/{r['optimizer']}": {t: r["split_gpu_h"] * t for t in taus}
            for r in rows if r["tokens"] == 4096 and r["mac"] == "int8"}
    for k, v in tail.items():
        print(f"split tail {k}: " + ", ".join(f"τ {t}: {h:,.0f} GPU-h" for t, h in v.items()))
    once = {name: {"endbox": endbox(name), "spectral": [spectral(name, r) for r in (4, 8, 32)]} for name in SHAPES}
    for name, v in once.items():
        print(f"{name}: end box {v['endbox']['gpu_h']:.1f} GPU-h ({v['endbox']['ands_per_param']:.0f} ANDs a parameter); "
              f"code-law gate {v['endbox']['code_law_gpu_h']:.2f} GPU-h; "
              "spectral gate " + ", ".join(f"r {s['r']}: {s['gpu_h']:.0f} GPU-h" for s in v["spectral"]))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "audit.json").write_text(json.dumps({"rows": rows, "tail": tail, "once": once}, default=str))


if __name__ == "__main__":
    main()
