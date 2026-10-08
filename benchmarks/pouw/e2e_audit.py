"""End to end on one RTX 4090: the GPU kernels prove, `verity.core.protocols.pouw.audit.Verifier` audits with the pure-Python reference.

    python benchmarks/pouw/e2e_audit.py --out RESULT.json

- ``ncp-v1`` on Qwen2.5-3B-shaped matmuls at decode (m = 16): the native u8xs8 route-U kernel starts at c0 and dumps every
  depth-16 checkpoint; each 16 x 16 tile's digest covers them step-major.  The verifier draws work-weighted tiles and
  recomputes each with the reference.
- ``pearl-fp8-v4`` (ada) on a smaller matmul: the e4m3 GEMM of the noisy-quantized operands; the verifier recomputes the
  drawn tiles (noise, quantization, policy, replay, extract, J).
- The negative control: a prover that corrupts half the tiles' checked values; t = 8 draws miss them all with probability 2^-8.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from kernels import load  # noqa: E402
from verity.core.protocols.pouw.audit import Epoch, Prover, Sampled, Verifier, Workload, reference_kernel  # noqa: E402
from verity.core.protocols.pouw.protocol import Matmul, Shape  # noqa: E402
from verity.core.protocols.pouw.schemes import get  # noqa: E402
from verity.core.protocols.pouw.schemes import ncp as N  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_kw as P  # noqa: E402


def ncp_noise_rows(tag: str, count: int, k: int, parts) -> np.ndarray:
    return np.stack([np.frombuffer(N.xof(tag, k, *parts(i)), dtype=np.uint8).astype(np.int16) % 64 - 31 for i in range(count)])


def ncp_gpu_kernel(ext, corrupt: float = 0.0, seed: int = 0):
    rng = random.Random(seed)

    def kernel(scheme, ctx, noise, a_rows, b_rows, tiles):
        u, k = ctx.matmul, ctx.matmul.shape.k
        pi = np.array(noise[0])
        A = np.stack([np.frombuffer(r, dtype=np.int8) for r in a_rows]).astype(np.int16)
        B = np.stack([np.frombuffer(r, dtype=np.int8) for r in b_rows]).astype(np.int16)
        E1 = ncp_noise_rows("verity/pouw/ncp/E1/v1", len(a_rows), k, lambda i: (ctx.salt, u.index, ctx.root_a, u.weight, i))
        F1 = ncp_noise_rows("verity/pouw/ncp/F1/v1", len(b_rows), k, lambda j: (ctx.salt, u.weight, j))
        E2 = E1[:, pi]
        F2 = np.empty_like(F1)
        F2[:, pi] = -F1
        X = np.concatenate([A + E1, A + E2, -(A + E1 + E2)], axis=1).astype(np.int8)
        Y = np.concatenate([B + F1, B + F2, B + F1 + F2], axis=1).astype(np.int8)
        cfg = 1 if len(a_rows) <= 32 else 0
        XU = (X.astype(np.int16) + N.BETA).astype(np.uint8)
        _, ck = ext.ncp_route_u(torch.from_numpy(XU).cuda(), torch.from_numpy(Y.view(np.uint8)).cuda(), cfg)
        ck = ck.cpu().numpy().astype(np.uint32)    # [m, n, T], already route U
        out = []
        for t in tiles:
            block = ck[np.ix_(t.rows, t.cols)]            # [h, w, T]
            if corrupt and rng.random() < corrupt:
                block = block.copy()
                block[0, 0, 0] ^= 1
            words = np.ascontiguousarray(block.transpose(2, 0, 1)).astype("<u4").tobytes()
            digest = hashlib.shake_256(b"verity/pouw/ncp/tile/v1\0" + words).digest(32)
            final = block[:, :, -1].astype(np.int64)
            out.append((digest, np.where(final >= 1 << 31, final - (1 << 32), final).tolist()))
        return out
    return kernel


def pearl_gpu_kernel(ext):
    def kernel(scheme, ctx, noise, a_rows, b_rows, tiles):
        from verity.core.protocols.pouw.protocol import Tile
        full = Tile(tuple(range(len(a_rows))), tuple(range(len(b_rows))))
        ops = scheme.encode(ctx, noise, a_rows, b_rows, full)
        C = ext.fp8_gemm(torch.tensor(ops.a.codes, dtype=torch.uint8).cuda(), torch.tensor(ops.b.codes, dtype=torch.uint8).cuda(), 1).cpu()
        out = []
        for t in tiles:
            tile = [[C[i, j].item() for j in t.cols] for i in t.rows]
            sub = P.Ops(P.Built(*(([getattr(ops.a, f)[i] for i in t.rows]) for f in ("clean", "codes", "alpha", "beta", "l2", "noise"))),
                        P.Built(*(([getattr(ops.b, f)[j] for j in t.cols]) for f in ("clean", "codes", "alpha", "beta", "l2", "noise"))))
            out.append((scheme.digest(ctx, noise, tile), scheme.useful(ctx, noise, sub, tile)))
        return out
    return kernel


def int7_rows(rng: np.random.Generator, count: int, k: int) -> list[bytes]:
    return [rng.integers(-64, 64, k, dtype=np.int8).tobytes() for _ in range(count)]


def run_ncp(ext, t: int) -> dict:
    s = get("ncp-v1")
    rng = np.random.default_rng(1)
    shapes = [((16, 2048, 2560), 0), ((16, 2048, 2048), 1), ((16, 11008, 2048), 2)]
    work = Workload(tuple(Matmul(100 + i, Shape(*sh), w) for i, (sh, w) in enumerate(shapes)))
    weights = {w: int7_rows(rng, n, k) for w, (n, k) in work.weights().items()}
    acts = {u.index: int7_rows(rng, u.shape.m, u.shape.k) for u in work.matmuls}
    out = {}
    for label, corrupt in (("honest", 0.0), ("corrupt-50%", 0.5)):
        prover = Prover(s, weights, ncp_gpu_kernel(ext, corrupt, seed=3))
        epoch = Epoch.unverified(work, prover.weight_roots, b"beacon round 2026-09-28")
        t0 = time.perf_counter()
        tr = prover.prove(epoch, acts)
        t_prove = time.perf_counter() - t0
        if label == "honest":
            for pos, u in enumerate(work.matmuls):      # the useful output is A·Bᵀ exactly
                A = np.stack([np.frombuffer(r, dtype=np.int8) for r in acts[u.index]]).astype(np.int64)
                B = np.stack([np.frombuffer(r, dtype=np.int8) for r in weights[u.weight]]).astype(np.int64)
                assert np.array_equal(np.array(tr.useful[pos]), A @ B.T)
        t0 = time.perf_counter()
        audit = Verifier(s, epoch).audit(prover, tr, Sampled(t), b"verifier secret")
        out[label] = {"accepted": audit.accepted, "drawn": len(audit.selected), "failed": sum(not v.ok for v in audit.verdicts),
                      "prove_s": t_prove, "verify_s": time.perf_counter() - t0, "tiles": len(Verifier(s, epoch).refs),
                      "profile": audit.profile.to_json() if audit.profile else None}
    return out


def run_pearl(ext, t: int) -> dict:
    s = get("pearl-fp8-v4")
    rng = random.Random(2)
    k, n, m = 1024, 128, 16
    rows = lambda cnt: [P.fp10_row([rng.randint(-127, 127) for _ in range(k)], [0x3C00 + rng.randrange(0x200) for _ in range(k // 8)])
                        for _ in range(cnt)]
    work = Workload((Matmul(7, Shape(m, k, n), 3),))
    weights, acts = {3: rows(n)}, {7: rows(m)}
    prover = Prover(s, weights, pearl_gpu_kernel(ext))
    epoch = Epoch.unverified(work, prover.weight_roots, b"beacon round 2026-09-28")
    t0 = time.perf_counter()
    tr = prover.prove(epoch, acts)
    t_prove = time.perf_counter() - t0
    t0 = time.perf_counter()
    audit = Verifier(s, epoch).audit(prover, tr, Sampled(t), b"verifier secret")
    return {"accepted": audit.accepted, "drawn": len(audit.selected), "failed": sum(not v.ok for v in audit.verdicts),
            "reasons": sorted({v.reason for v in audit.verdicts if not v.ok}), "prove_s": t_prove, "verify_s": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "e2e.json")
    ap.add_argument("--t", type=int, default=8)
    args = ap.parse_args()
    ext = load()
    res = {"ncp": run_ncp(ext, args.t), "pearl": run_pearl(ext, max(2, args.t // 4))}
    print(json.dumps(res, indent=1))
    args.out.write_text(json.dumps({"schema": "verity-pouw/e2e-audit/v1", **res}, indent=1))
    import run_result
    ok = res["ncp"]["honest"]["accepted"] and not res["ncp"]["corrupt-50%"]["accepted"] and res["pearl"]["accepted"]
    run_result.write(args.out, ok, "e2e_audit: honest provers accepted, the corrupting prover rejected",
                     json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "profile"} if isinstance(v, dict) else v for k, v in res.items()})[:2000],
                     [("ncp.prove_s", res["ncp"]["honest"]["prove_s"], "s"), ("ncp.verify_s", res["ncp"]["honest"]["verify_s"], "s"),
                      ("pearl.prove_s", res["pearl"]["prove_s"], "s"), ("pearl.verify_s", res["pearl"]["verify_s"], "s")])


if __name__ == "__main__":
    main()
