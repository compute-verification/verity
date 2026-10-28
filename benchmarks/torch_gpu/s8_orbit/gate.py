"""The orbit kernel on the GPU against torch._int_mm (cuBLASLt's exact s32 GEMM), every tile width:

  small   |x|, |w| <= 63, where no form carries: y32 equal to X·Wᵀ, no carry bit, and the bf16 words equal to the exact
          linear's epilogue on X·Wᵀ (cvt.rn.f32, mul.rn s_a, mul.rn s_w, cvt.rn.bf16), at shapes with tails in every dim
  full    int8 X on ±127 against a checkpoint's linears (or W on ±127): y32 plus `orbit_gpu.corrections` equal to X·Wᵀ
          mod 2^32, with carry bits counted

    python gate.py --cubin C --out OUT [--snapshot DIR --layer 0] [--m 2048]

Writes OUT/gate.json; exits 4 on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orbit_gpu as G  # noqa: E402
from cpu_check import LINEARS, Shards  # noqa: E402

SMALL = ((256, 384, 512), (130, 200, 192), (2, 4, 64), (16384, 6144, 4096))


def epilogue(o: torch.Tensor, sa: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    return ((o.float() * sa[:, None]) * sw[None, :]).to(torch.bfloat16)


def small(lib: G.Lib, gen: torch.Generator) -> list[dict]:
    recs = []
    for m, n, k in SMALL:
        x = torch.randint(-63, 64, (m, k), generator=gen, dtype=torch.int8, device="cuda")
        w = torch.randint(-63, 64, (n, k), generator=gen, dtype=torch.int8, device="cuda")
        sa = torch.rand(m, generator=gen, device="cuda") / 64
        sw = torch.rand(n, generator=gen, device="cuda") / 64
        want = (x.double() @ w.double().t()).long() if m < 17 else torch._int_mm(x, w.t()).long()
        wb, bc = G.b_forms(w)
        for bn in G.TILES:
            y32 = torch.full((m, n), 0x5A5A5A5A, dtype=torch.int32, device="cuda")
            lib.reset()
            info = lib.mm(x, wb, y32=y32, bn=bn)
            y = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
            lib.mm(x, wb, y=y, sa=sa, sw=sw, bn=bn)
            torch.cuda.synchronize()
            bad = int((y32.long() != want).sum())
            bad16 = int((y.view(torch.int16) != epilogue(want, sa, sw).view(torch.int16)).sum())
            recs.append({"case": "small", "m": m, "n": n, "k": k, "bn": bn, "mismatches": bad, "bf16_mismatches": bad16,
                         "a_carry_bits": lib.flags, "b_carries": sum(bc), "ok": bad == 0 and bad16 == 0 and lib.flags == 0,
                         **info})
            print(json.dumps(recs[-1]), flush=True)
    return recs


def full(lib: G.Lib, gen: torch.Generator, weights: dict[str, torch.Tensor], m: int) -> list[dict]:
    recs = []
    for name, w in weights.items():
        n, k = w.shape
        x = torch.randint(-127, 128, (m, k), generator=gen, dtype=torch.int8, device="cuda")
        want = torch._int_mm(x, w.t()).long()
        corr = G.corrections(x, w)
        wb, bc = G.b_forms(w)
        for bn in G.TILES:
            y32 = torch.zeros((m, n), dtype=torch.int32, device="cuda")
            lib.reset()
            lib.mm(x, wb, y32=y32, bn=bn)
            torch.cuda.synchronize()
            got = (y32.long() + corr) & 0xFFFFFFFF
            bad = int((got != (want & 0xFFFFFFFF)).sum())
            low = int(((y32.long() & 0xFF) != (want & 0xFF)).sum())
            recs.append({"case": "full", "linear": name, "m": m, "n": n, "k": k, "bn": bn, "mismatches": bad,
                         "mod256_mismatches": low, "a_carry_bits": lib.flags, "b_carries": sum(bc),
                         "uncorrected_mismatch_frac": float((y32.long() != want).float().mean()),
                         "ok": bad == 0 and low == 0 and lib.flags > 0})
            print(json.dumps(recs[-1]), flush=True)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lib = G.Lib(Path(a.cubin))
    gen = torch.Generator(device="cuda").manual_seed(a.seed)
    if a.snapshot:
        sh = Shards(Path(a.snapshot))
        weights = {lin: torch.from_numpy(__import__("numpy").concatenate(
            [sh.get(f"model.layers.{a.layer}.{p}.weight") for p in parts])).cuda() for lin, parts in LINEARS.items()}
    else:
        weights = {"random": torch.randint(-127, 128, (1024, 1024), generator=gen, dtype=torch.int8, device="cuda")}
    recs = small(lib, gen) + full(lib, gen, weights, a.m)
    doc = {"cubin_sha256": lib.sha256, "layout": lib.layout, "sms": lib.sms, "ok": all(r["ok"] for r in recs), "records": recs}
    (out / "gate.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"ok": doc["ok"], "n": len(recs), "failed": [r for r in recs if not r["ok"]][:5]}), flush=True)
    return 0 if doc["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
