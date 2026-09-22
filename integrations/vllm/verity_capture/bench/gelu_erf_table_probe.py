"""[lane cov-ln] Measure the exhaustive bf16 table of `aten.gelu(x, approximate='none')` on the pod and write it as the pure
module `verity_vllm/registry/quarantine/ln/gelu_erf_table.py` (zlib + base64 of 65,536 little-endian uint16 words).

    /workspace/venv-cu129/bin/python -m verity_capture.bench.gelu_erf_table_probe --out verity_vllm/registry/quarantine/ln/gelu_erf_table.py

The table is the *definition* of `ln/GeluErfBf16@v1` (see that module): the domain of a bf16 -> bf16 kernel is finite, so the
measured table is the exact semantics of the pinned build (torch version, GPU name and sha256 of the words are recorded in the
module header).  Also prints the sanity facts the lane's test asserts (±0, x -> x above 2^4, -0 below -2^4, NaN -> 0x7FFF).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import zlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import numpy as np
    import torch

    words = torch.arange(65536, dtype=torch.int32, device=a.device).to(torch.int16).view(torch.bfloat16)
    y = torch.nn.functional.gelu(words)                      # approximate='none' (erf)
    out = y.view(torch.int16).to(torch.int32).cpu().numpy().astype(np.int64) & 0xFFFF
    tab = out.astype("<u2")
    raw = tab.tobytes()
    sha = hashlib.sha256(raw).hexdigest()
    enc = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
    dev = torch.cuda.get_device_name(0) if a.device.startswith("cuda") else a.device
    lines = [enc[i:i + 100] for i in range(0, len(enc), 100)]
    body = "\n".join(f'    "{ln}"' for ln in lines)
    src = f'''"""The exhaustive bf16 table of `aten.gelu(x, approximate='none')` (see `gelu_erf_bf16.py`): output word for every input
word 0x0000..0xFFFF, measured by `verity_capture/bench/gelu_erf_table_probe.py`.

    torch {torch.__version__}, device {dev}, sha256(raw little-endian uint16 words) = {sha}
"""
from __future__ import annotations

import base64
import zlib

SHA256 = "{sha}"
TORCH = "{torch.__version__}"
DEVICE = "{dev}"

_B64 = (
{body}
)

#: 131,072 bytes: word i (the kernel's output for input word i) at bytes 2i, 2i+1 (little-endian)
TABLE_WORDS: bytes = zlib.decompress(base64.b64decode(_B64))
assert len(TABLE_WORDS) == 2 * 65536
'''
    with open(a.out, "w") as f:
        f.write(src)
    # sanity facts
    t = tab.astype(np.uint32)
    f32 = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
    print(f"wrote {a.out}: {len(enc)} b64 chars, sha256 {sha}, torch {torch.__version__}, {dev}")
    print("gelu(+0) =", hex(int(t[0x0000])), " gelu(-0) =", hex(int(t[0x8000])))
    big = np.isfinite(f32) & (f32 >= 16.0)
    print("x>=16 -> x:", int((t[big] == np.arange(65536, dtype=np.uint32)[big]).sum()), "/", int(big.sum()))
    neg = np.isfinite(f32) & (f32 <= -16.0)
    print("x<=-16 -> -0:", int((t[neg] == 0x8000).sum()), "/", int(neg.sum()))
    nan = np.isnan(f32)
    print("NaN words -> 0x7FFF:", int((t[nan] == 0x7FFF).sum()), "/", int(nan.sum()))
    print("+inf ->", hex(int(t[0x7F80])), " -inf ->", hex(int(t[0xFF80])))
    # how far the kernel is from the correctly rounded value (informational)
    import math
    xs = f32.astype(np.float64)
    exact = np.array([x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) if math.isfinite(x) else float("nan") for x in xs])
    rn = np.asarray(exact, dtype=np.float32).view(np.uint32)
    rn_bf16 = ((rn + 0x7FFF + ((rn >> 16) & 1)) >> 16) & 0xFFFF
    fin = np.isfinite(f32)
    print("kernel word != RN_bf16(exact real gelu):", int((t[fin] != rn_bf16[fin]).sum()), "/", int(fin.sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
