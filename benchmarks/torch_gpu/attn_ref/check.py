#!/usr/bin/env python3
"""attn_ref on one H200 (sm_90a), through attn_ref.py on torch tensors, against fixture.py's words from the Definition.

0. Identity, before anything runs: one visible device, compute capability 9.0, a name starting with ATTN_EXPECT (default
   "NVIDIA H200"), and its UUID equal to GPU_LEASE_UUID when gpu-lease set it.  A mismatch exits 3 before any launch.
1. The check.  Every batch is made of copies of the pool's sequences (CHECK/pool.npz), so every word of every batch has its
   expected word, and every word is compared.  S and O are filled with 0xA5A5 before each launch, so a word the kernels
   don't write fails too.  At each batch size B of BATCHES:
   - uniform: for each KV length n = 1..32, every sequence holds n keys; the cache and P past n are overwritten with junk
     (NaN, infinities, the largest finite) that must not enter any word.  S must be the pool's S up to n and 0xFF80 past
     it, O the pool's O at length n.
   - mixed (B > 1): each sequence its own pool sequence and length (seeded), the pool's own words past each length; the
     mismatches are counted by length.  The same batch at 1 and 8 warps a block must give the same bytes as at 4.
   The odd case (CHECK/odd.npz: GQA, head dim 128, 37 keys, lengths 0..37) runs at 1, 4 and 8 warps.  A control that
   poisons S and O and launches nothing must be rejected, and the inputs must come back unchanged.  Exit 4 if anything
   fails, before anything is timed.
2. Diagnostic timing (information only): at B = TIME_B, all sequences at 32 keys and the mixed lengths, QK^T and PV here
   and torch's two bf16 batched matmuls (cuBLAS, not the Definition's words), the median of REPS after a warm-up,
   interleaved rep by rep.

One JSON object per line on stdout as each part finishes, a summary on stderr.

  python3 check.py CHECK   (env: BATCHES=1,7,128,20480, REPS=20, TIME_B=20480, ATTN_EXPECT, ATTN_ANY_GPU=1 for a dry run on
                            another card)
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attn_ref as A  # noqa: E402

POISON = 0xA5A5
JUNK = (0x7FC0, 0xFFC1, 0x7F80, 0xFF80, 0x7F7F, 0xFF7F)
FILL = A.NEG_INF


def i16(w: int) -> int:
    return w - 0x10000 if w >= 0x8000 else w


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def identity() -> dict:
    n = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    uuid = str(getattr(props, "uuid", "")).lower().removeprefix("gpu-")
    lease = os.environ.get("GPU_LEASE_UUID", "").lower().removeprefix("gpu-")
    doc = {"devices": n, "name": props.name, "cc": f"{props.major}.{props.minor}", "uuid": uuid, "lease_uuid": lease or None,
           "sms": props.multi_processor_count, "torch": torch.__version__, "cuda": torch.version.cuda,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    want = os.environ.get("ATTN_EXPECT", "NVIDIA H200")
    problems = []
    if n != 1:
        problems.append(f"{n} visible devices, want 1")
    if (props.major, props.minor) != (9, 0):
        problems.append(f"compute capability {props.major}.{props.minor}, want 9.0 (the cubin is sm_90a)")
    if not os.environ.get("ATTN_ANY_GPU"):
        if not props.name.startswith(want):
            problems.append(f"{props.name!r} is not {want!r}")
        if lease and uuid != lease:
            problems.append(f"device {uuid} is not the lease's {lease}")
    doc["problems"] = problems
    return doc


class Pool:
    def __init__(self, path: str, dev: torch.device):
        z = np.load(path)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).to(dev)  # noqa: E731
        self.Q, self.K, self.V, self.P, self.S, self.O = (t(z[k]) for k in ("Q", "K", "V", "P", "S", "O"))
        self.kind = [str(k) for k in z["kind"]]
        self.n = self.Q.shape[0]
        self.L = self.K.shape[1]


def bf(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.bfloat16)


def poisoned(shape: tuple, dev) -> torch.Tensor:
    return torch.full(shape, i16(POISON), dtype=torch.int16, device=dev)


def junk_tails(x: torch.Tensor, n: int, axis: int) -> torch.Tensor:
    """``x`` (int16) with every position >= n along ``axis`` replaced by JUNK words, cycling."""
    x = x.clone()
    idx = [slice(None)] * x.dim()
    idx[axis] = slice(n, None)
    tail = x[tuple(idx)]
    if tail.numel():
        j = torch.tensor([i16(w) for w in JUNK], dtype=torch.int16, device=x.device)
        x[tuple(idx)] = j[torch.arange(tail.numel(), device=x.device) % len(JUNK)].view(tail.shape)
    return x


def expected_s(pool: Pool, sid: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    j = torch.arange(pool.L, device=sid.device)
    return torch.where(j[None, None, :] < lens[:, None, None].long(), pool.S[sid], torch.tensor(i16(FILL), dtype=torch.int16,
                                                                                                 device=sid.device))


def run_pair(Q, K, V, P, lens, warps: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, D = Q.shape
    L = K.shape[1]
    S, O = poisoned((B, H, L), Q.device), poisoned((B, H, D), Q.device)
    A.qk(bf(Q), bf(K), lens, out=bf(S), warps=warps)
    A.pv(bf(P), bf(V), lens, out=bf(O), warps=warps)
    torch.cuda.synchronize()
    return S, O


def uniform(pool: Pool, B: int, dev) -> list[dict]:
    rows = []
    for n in range(1, pool.L + 1):
        sid = (torch.arange(B, device=dev) * 7 + (n if B < pool.n else 0)) % pool.n
        lens = torch.full((B,), n, dtype=torch.int32, device=dev)
        Q, K, V, P = pool.Q[sid], junk_tails(pool.K[sid], n, 1), junk_tails(pool.V[sid], n, 1), junk_tails(pool.P[sid], n, 2)
        before = [x.clone() for x in (Q, K, V, P)]
        S, O = run_pair(Q, K, V, P, lens)
        es, eo = expected_s(pool, sid, lens), pool.O[sid, n - 1]
        same = all(torch.equal(a, b) for a, b in zip(before, (Q, K, V, P)))
        rows.append({"part": "uniform", "B": B, "n": n, "sequences": int(torch.unique(sid).numel()),
                     "qk_words": S.numel(), "qk_mismatches": int((S != es).sum()),
                     "pv_words": O.numel(), "pv_mismatches": int((O != eo).sum()), "inputs_unchanged": same})
    return rows


def mixed(pool: Pool, B: int, dev, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    sid = torch.randint(0, pool.n, (B,), generator=g).to(dev)
    lens = torch.randint(1, pool.L + 1, (B,), generator=g, dtype=torch.int32).to(dev)
    Q, K, V, P = pool.Q[sid], pool.K[sid], pool.V[sid], pool.P[sid]
    before = [x.clone() for x in (Q, K, V, P)]
    S, O = run_pair(Q, K, V, P, lens)
    es, eo = expected_s(pool, sid, lens), pool.O[sid, lens.long() - 1]
    qk_bad, pv_bad = (S != es).sum((1, 2)), (O != eo).sum((1, 2))
    by = lambda w: torch.bincount(lens.long(), weights=w.double(), minlength=pool.L + 1)[1:].long().tolist()  # noqa: E731
    seqs = torch.bincount(lens.long(), minlength=pool.L + 1)[1:].tolist()
    warps = {}
    for w in (1, 8):
        S2, O2 = run_pair(Q, K, V, P, lens, warps=w)
        warps[str(w)] = torch.equal(S2, S) and torch.equal(O2, O)
    S0, O0 = poisoned(S.shape, dev), poisoned(O.shape, dev)
    control = int((S0 != es).sum()) + int((O0 != eo).sum())
    same = all(torch.equal(a, b) for a, b in zip(before, (Q, K, V, P)))
    return {"part": "mixed", "B": B, "seed": seed, "sequences": int(torch.unique(sid).numel()),
            "qk_words": S.numel(), "qk_mismatches": int(qk_bad.sum()), "pv_words": O.numel(), "pv_mismatches": int(pv_bad.sum()),
            "by_length": {"sequences": seqs, "qk_mismatches": by(qk_bad), "pv_mismatches": by(pv_bad)},
            "same_bytes_at_warps": warps, "control_mismatches": control, "control_rejected": control > 0,
            "inputs_unchanged": same}


def odd(path: str, dev) -> dict:
    z = np.load(path)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).to(dev)  # noqa: E731
    Q, K, V, P, S_e, O_e = (t(z[k]) for k in ("Q", "K", "V", "P", "S", "O"))
    lens = torch.from_numpy(z["lens"].astype(np.int32)).to(dev)
    out = {"part": "odd", "shape": {"B": Q.shape[0], "H": Q.shape[1], "Hkv": K.shape[1], "D": Q.shape[2], "L": K.shape[2],
                                    "lens": z["lens"].tolist()}, "qk_words": S_e.numel(), "pv_words": O_e.numel()}
    for w in (1, 4, 8):
        S, O = poisoned(S_e.shape, dev), poisoned(O_e.shape, dev)
        A.qk(bf(Q), bf(K), lens, out=bf(S), warps=w)
        A.pv(bf(P), bf(V), lens, out=bf(O), warps=w)
        torch.cuda.synchronize()
        out[f"warps{w}"] = {"qk_mismatches": int((S != S_e).sum()), "pv_mismatches": int((O != O_e).sum())}
    return out


def timing(pool: Pool, B: int, dev, reps: int) -> list[dict]:
    rows = []
    g = torch.Generator().manual_seed(7)
    sid = torch.randint(0, pool.n, (B,), generator=g).to(dev)
    mixed_lens = torch.randint(1, pool.L + 1, (B,), generator=g, dtype=torch.int32).to(dev)
    Q, K, V, P = (bf(x[sid]) for x in (pool.Q, pool.K, pool.V, pool.P))
    S = torch.empty(B, Q.shape[1], pool.L, dtype=torch.bfloat16, device=dev)
    O = torch.empty(B, Q.shape[1], Q.shape[2], dtype=torch.bfloat16, device=dev)
    Kt = K.transpose(1, 2)
    for label, lens in (("n32", torch.full((B,), pool.L, dtype=torch.int32, device=dev)), ("mixed", mixed_lens)):
        arms = {
            "attn_ref.qk": lambda: A.qk(Q, K, lens, out=S, check_lens=False),
            "attn_ref.pv": lambda: A.pv(P, V, lens, out=O, check_lens=False),
            "torch.matmul QK^T (cuBLAS)": lambda: torch.matmul(Q, Kt, out=S),
            "torch.matmul PV (cuBLAS)": lambda: torch.matmul(P, V, out=O),
        }
        times: dict[str, list[float]] = {k: [] for k in arms}
        for f in arms.values():
            f()
        torch.cuda.synchronize()
        for _ in range(reps):
            for name, f in arms.items():
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                f()
                e1.record()
                e1.synchronize()
                times[name].append(e0.elapsed_time(e1))
        med = {k: statistics.median(v) for k, v in times.items()}
        rows.append({"part": "timing", "B": B, "lens": label, "reps": reps, "median_ms": med,
                     "min_ms": {k: min(v) for k, v in times.items()},
                     "qk_over_cublas": med["attn_ref.qk"] / med["torch.matmul QK^T (cuBLAS)"],
                     "pv_over_cublas": med["attn_ref.pv"] / med["torch.matmul PV (cuBLAS)"]})
    return rows


def main(argv: list[str]) -> int:
    check = argv[1] if len(argv) > 1 else os.path.join(HERE, "check")
    ident = identity()
    emit({"part": "identity", **ident})
    if ident["problems"]:
        print("identity: " + "; ".join(ident["problems"]), file=sys.stderr)
        return 3
    dev = torch.device("cuda", 0)
    lib = A.lib()
    emit({"part": "cubin", "path": str(lib.path), "sha256": lib.sha256, "pv_kpv": A.PV_K})
    pool = Pool(os.path.join(check, "pool.npz"), dev)
    batches = [int(b) for b in os.environ.get("BATCHES", "1,7,128,20480").split(",")]
    failures = []
    for B in batches:
        for row in uniform(pool, B, dev):
            emit(row)
            if row["qk_mismatches"] or row["pv_mismatches"] or not row["inputs_unchanged"]:
                failures.append(f"uniform B={B} n={row['n']}")
        if B > 1:
            row = mixed(pool, B, dev, seed=B)
            emit(row)
            if (row["qk_mismatches"] or row["pv_mismatches"] or not all(row["same_bytes_at_warps"].values())
                    or not row["control_rejected"] or not row["inputs_unchanged"]):
                failures.append(f"mixed B={B}")
    row = odd(os.path.join(check, "odd.npz"), dev)
    emit(row)
    if any(row[f"warps{w}"]["qk_mismatches"] or row[f"warps{w}"]["pv_mismatches"] for w in (1, 4, 8)):
        failures.append("odd")
    if failures:
        print("FAILED: " + ", ".join(failures), file=sys.stderr)
        return 4
    print(f"check: 0 mismatches at B {batches}, every KV length 1..{pool.L}, mixed lengths and the odd case", file=sys.stderr)
    for row in timing(pool, int(os.environ.get("TIME_B", "20480")), dev, int(os.environ.get("REPS", "20"))):
        emit(row)
        print(f"timing {row['lens']}: " + ", ".join(f"{k} {v:.3f} ms" for k, v in row["median_ms"].items()), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
