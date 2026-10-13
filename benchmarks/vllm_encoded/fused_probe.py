"""Each fused launch the stack's install makes, one at a time with a sync after each, logged before it starts: which one
hangs (a debugging probe; not part of the stack)."""

import argparse
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stack as S  # noqa: E402


def log(*a) -> None:
    print(time.strftime("%H:%M:%SZ", time.gmtime()), *a, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt-lib", required=True)
    ap.add_argument("--reader-lib", required=True)
    ap.add_argument("--paged-lib", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    st = S.Stack(types.SimpleNamespace(k=a.k, wt_lib=a.wt_lib, reader_lib=a.reader_lib, paged_lib=a.paged_lib), True)
    log("stack up")
    gen = torch.Generator(device="cuda").manual_seed(1)
    for name, (n, k, bf) in {"kv": (1024, 4096, False), "lm": (2048, 4096, True)}.items():
        w = (torch.randn((n, k), generator=gen, device="cuda") * 0.1).to(torch.bfloat16 if bf else torch.float8_e4m3fn)
        lin = S.StackLinear(st, name, w, None if bf else torch.ones(1, device="cuda"), bf=bf)
        log(name, "encoded", lin.words, "words")
        for m in S.CAPTURE:
            x8 = S.rand_x(lin, m, gen)
            for cfg in lin.candidates(m):
                plan = lin.planned(m, cfg, {})
                log(name, "m", m, "cfg", cfg, "groups", plan[1], "receipts", plan[3], "...")
                lin.sums(x8, plan)
                torch.cuda.synchronize()
                log(name, "m", m, "cfg", cfg, "ok")
    log("done")


if __name__ == "__main__":
    main()
