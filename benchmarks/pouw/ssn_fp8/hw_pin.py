"""The pinned SSN proof block on silicon: one block of four steps per tile through tc_probe's chain4 wgmma kernels.

``PYTHONPATH=. python -m benchmarks.pouw.ssn_fp8.hw_pin --out DIR [--instances 250 --salts 8]`` on an sm_90a host.  For
each format (E4M3, E5M2: four m64n8k32; BF16: four m64n8k16) it forms ``instances`` data cases (`study.data`: 64
rows, 8 columns, the degenerate rows and columns included) under ``salts`` coin draws each (`design.form`, mode pin,
the partial bound, one block of four steps: K = 121 data entries for FP8, 57 for BF16), runs every tile on the GPU with
the block's bias as the accumulator, and reports: hardware words against `sim` (bit for bit), words outside their
row's binade (``pin_fail``), and words whose value less u times their noise differs between salts of one instance
(``affine_fail``).  Exit 1 on any mismatch, pin or affine failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.tc_probe import tc_probe as TP

from . import design as D
from . import sim
from . import study as St

CASES = {"e4m3": ("sm90.wgmma.m64n8k32.e4m3", 121), "e5m2": ("sm90.wgmma.m64n8k32.e5m2", 121),
         "bf16": ("sm90.wgmma.m64n8k16.bf16", 57)}


def layout_check(probe: TP.Probe, fmt) -> None:
    rng = np.random.default_rng(1)
    a = rng.integers(-8, 9, (64, probe.ins.m, probe.k)).astype(np.float32)
    b = rng.integers(-8, 9, (64, TP.N, probe.k)).astype(np.float32)
    c = rng.integers(-1000, 1001, (64, probe.ins.m, TP.N)).astype(np.float32)
    want = TP.f32_bits((c + np.einsum("nik,njk->nij", a, b)).astype(np.float32))
    got = probe.tiles(fmt.from_f32(a), fmt.from_f32(b), TP.f32_bits(c))
    if not np.array_equal(want, got):
        raise RuntimeError(f"layout check failed: {int(np.count_nonzero(want != got))} words differ")


def one(so: Path, dtype: str, instances: int, salts: int, seed: int, out: Path, signed: bool = False) -> dict:
    ins_id, K = CASES[dtype]
    ins = TP.INSTRUCTIONS[ins_id]
    base = TP.Probe(so, ins)
    TP.layout_check(base, ins)
    probe = TP.Probe(so, ins, kernel=TP.chain_kernel(base), k=TP.CHAIN_STEPS * ins.k)
    probe.set_descriptor(base.descriptor["lbo_bytes"], base.descriptor["sbo_bytes"])
    layout_check(probe, ins.fmt)
    As, Bs, Cs, want, last_noise, us, ps, biases = [], [], [], [], [], [], [], []
    for d in range(instances):
        A, B, *_ = St.data(dtype, 64, 8, K, np.random.default_rng([seed, d]))
        for s in range(salts):
            f = D.form(dtype, A, B, "pin", np.random.default_rng([seed, d, s]), block=4, bound="partial", signed=signed)
            if f.steps != 4 or f.blocks != 1:
                raise RuntimeError(f"expected one block of four steps, got {f.steps} steps in {f.blocks} blocks")
            w, bad = D.run(dtype, f)
            if bad.any():
                raise RuntimeError("the simulator refuses a formed word")
            As.append(f.A)
            Bs.append(f.B)
            Cs.append(f.init[:, :, 0])
            want.append(w[:, :, -1])
            last_noise.append(f.noise[:, :, -1])
            us.append(f.u)
            ps.append(f.p)
            biases.append(f.bias[:, :, 0])
    A, Bt, C, W = map(np.stack, (As, Bs, Cs, want))
    Dw = probe.tiles(A, Bt, C)
    np.savez_compressed(out / f"pin_{dtype}{'_signed' if signed else ''}.npz", A=A, Bt=Bt, C=C, D=Dw)
    mism = int(np.count_nonzero(Dw != W))
    p = np.stack(ps)
    e = ((Dw.astype(np.int64) >> 23) & 0xFF) - 127
    pin_fail = int((e != p[:, :, None]).sum())
    core = sim.fp32_value(Dw) - np.stack(us)[:, :, None] * np.stack(last_noise) - np.stack(biases)
    core = core.reshape(instances, salts, *core.shape[1:])
    affine_fail = int((~(core == core[:, :1]).all(1)).sum())
    noise_span = np.stack(last_noise).reshape(instances, salts, 64, 8)
    varies = int(((noise_span != noise_span[:, :1]).any(1)).sum())
    return {"format": dtype, "signed": signed, "instruction": ins_id, "kernel": probe.kernel, "tiles": int(A.shape[0]), "words": int(Dw.size),
            "data_entries_a_block": K, "mismatches_vs_sim": mism, "pin_fail": pin_fail, "affine_fail": affine_fail,
            "outputs_whose_noise_varies_over_salts": varies, "outputs": instances * 64 * 8}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--instances", type=int, default=250)
    ap.add_argument("--salts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--formats", nargs="+", default=list(CASES))
    ap.add_argument("--signed", action="store_true", help="also run the signed coin lanes (+-2^r)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    so, ident = TP.build("wgmma_tiles.cu", "sm_90a", out)
    probe = TP.Probe(so, TP.INSTRUCTIONS["sm90.wgmma.m64n8k32.e4m3"])
    doc = {"toolchain": ident, "device": probe.identity(), "results": []}
    for dtype in args.formats:
        for signed in (False, True) if args.signed else (False,):
            r = one(so, dtype, args.instances, args.salts, args.seed, out, signed)
            doc["results"].append(r)
            print(json.dumps(r), flush=True)
    (out / "hw_pin.json").write_text(json.dumps(doc, indent=1))
    bad = any(r["mismatches_vs_sim"] or r["pin_fail"] or r["affine_fail"] for r in doc["results"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
