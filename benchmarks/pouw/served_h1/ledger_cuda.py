"""The same-GEMM gate's launch ledger on a CUDA device (`same_gemm.Recorder("cuda")`): where torch.profiler and CUPTI put
each kernel, as the gate reads it.  Two calls of one linear: today's beside path (`torch._int_mm`, cuBLASLt, and its
dequant in the call's range) and a fused stand-in (one raw driver-API launch, `receipts.Cubin`'s `h1_noise`, in the fused
range, the call returning a view of its buffer).  It passes when the beside call's GEMM kernels attach to its `_int_mm`
and the gate refuses it, and the raw launch's kernel is inside the fused range, nothing runs after it (the recorder's
device-to-host reads included) and the gate accepts it.

    python ledger_cuda.py --cubin noise.cubin --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import receipts as RC  # noqa: E402
import same_gemm as SG  # noqa: E402

M, K, N, BLOCKS = 32, 64, 64, 4


def _raw(prof, i: int) -> list[dict]:
    """Every event in call i's range with the kernels CUPTI attached to it, reads included: what the ledger reduces."""
    named = [e for e in prof.events() if e.name == f"{SG.CALL_RANGE}{i}"]
    r = next(e for e in named if SG._on_host(e))
    return [{"name": e.name, "depth": d, "device": str(e.device_type), "kernels": [k.name for k in (getattr(e, "kernels", ()) or ())]}
            for e, d in _walk(r, 0)] + [{"same_name": str(e.device_type), "children": len(e.cpu_children)} for e in named]


def _walk(e, d):
    yield e, d
    for c in e.cpu_children:
        yield from _walk(c, d + 1)


def main(argv: list[str] | None = None) -> int:
    import torch
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cubin", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(M, K, device=dev, generator=g)
    b = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev, generator=g)
    bt, s_w = b.t().contiguous(), torch.rand(N, device=dev, generator=g) * 1e-3
    cub = RC.Cubin(a.cubin)
    seeds = RC.seed_words([hashlib.sha512(bytes([i])).digest() for i in range(M)], dev)
    buf = torch.empty(M, BLOCKS, 16, dtype=torch.int32, device=dev)

    def beside():
        s_a = x.abs().amax(1) / 127
        xq = (x / s_a[:, None]).round().clamp(-127, 127).to(torch.int8)
        return (torch._int_mm(xq, bt).to(torch.float32) * s_a[:, None] * s_w[None, :]).to(torch.bfloat16)

    beside()
    cub.launch(seeds, BLOCKS, buf)
    torch.cuda.synchronize()
    rec = SG.Recorder("cuda")
    with rec:
        with rec.call(0, key=0):
            y0 = beside()
            rec.served(0, y0, M)
        with rec.call(1, key=1):
            with rec.fused(1, ["h1_noise"]) as f:
                cub.launch(seeds, BLOCKS, buf)
                f.outs(rows=M, y=buf, receipts=buf)
            y1 = buf[:M]
            rec.served(1, y1, M)
        rec.committed(1, "receipts", buf)
        torch.cuda.synchronize()
    ledger = rec.ledger()
    raw = {i: _raw(rec.prof, i) for i in (0, 1)}
    e0, e1 = ledger["calls"]
    c0, c1 = {"linear": 0, "step": 0, "module": "beside"}, {"linear": 0, "step": 1, "module": "fused"}
    why0, why1 = SG.call_refusal(c0, e0, ["h1_noise"]), SG.call_refusal(c1, e1, ["h1_noise"])
    gemm0 = [x for x in e0["events"] if x["gemm"]]
    noise1 = [x for x in e1["events"] if x["kind"] == "kernel" and x["name"] == "h1_noise"]
    checks = {
        "beside_gemm_is_int_mm": bool(gemm0) and gemm0[0]["name"] == "aten::_int_mm",
        "beside_gemm_has_kernels": bool(gemm0) and bool(gemm0[0].get("kernels")),
        "beside_refused": why0 is not None and "served y from aten::_int_mm call #" in why0,
        "raw_launch_seen": len(noise1) == 1,
        "raw_launch_in_fused_range": len(noise1) == 1 and noise1[0]["fused"],
        "nothing_after_fused": not [x for x in e1["events"] if x["after_fused"] and (x["kind"] == "kernel" or x.get("writes"))],
        "fused_accepted": why1 is None,
        "no_gemm_outside": ledger["outside"] == [],
    }
    summary = {"ok": all(checks.values()), "checks": checks, "beside_reason": why0, "fused_reason": why1,
               "beside_gemm": gemm0[:1], "fused_kernel": noise1, "device": torch.cuda.get_device_name(),
               "torch": torch.__version__, "cuda": torch.version.cuda}
    (out_dir / "ledger-cuda.json").write_text(json.dumps({**summary, "ledger": ledger, "raw": raw}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
