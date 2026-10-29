#!/usr/bin/env python3
"""The units on one H200 (sm_90a) or RTX PRO 6000 (sm_120a), through their modules on torch tensors, against fixture.py's
words from the twins.

0. Identity, before anything runs: one visible device, a compute capability build.sh has a cubin for (lib.ARCHES), a name
   starting with UNITS_EXPECT (default the arch's card, EXPECT), and its UUID equal to GPU_LEASE_UUID when gpu-lease set it.
   A mismatch exits 3 before any launch.
1. The sweeps: units_sweep's chunk digests of each op the cubin has (sweep.ARCH_OPS) against CHECK/sweep.json; a chunk
   that differs is named, and on sm_120a (units_sweep_words) the first few such chunks' differing words are counted and
   the first one given.  SWEEP_ONLY=1 stops here.
2. The check.  Every batch is made of copies of a pool's rows, so every word of every batch has its expected word, and
   every word of every unit output is compared: RMSNorm's out, chunks, groups and rs; the quantizer's codes, scale and
   absmax tree; softmax's p, m and r; argmax's token and every chunk's best and index; ReLU's and the residual add's words.
   Each output is filled with 0xA5 bytes before its launch, so a word a kernel doesn't write fails too, and a control that
   poisons and launches nothing must be rejected; the inputs must come back unchanged.  At each batch size B of BATCHES,
   the frozen shape's units: RMSNorm of 16384 (both gains), the quantizer at (16384, false) and (65536, true), softmax over
   B sequences of 256 heads at every length n = 1..32 (the words past n overwritten with NaNs and infinities, which must
   not enter any word) and at mixed lengths, argmax over 256, ReLU over B rows of every bf16 word, the residual add of 16384;
   then each pool's other widths once.  The add's row also counts where torch's CUDA bf16 add differs from it (information:
   circuits expects none).  Exit 4 if anything fails, before anything is timed.
3. Diagnostic timing (information only), bench.py's: at each B of TIME_B, each unit and torch's nearest op on the GPU's
   ruler and the host-inclusive one, and a step row, the units' medians summed at a decode step's counts and that step's
   launches measured as one (STEP: per layer 2 RMSNorms, 3 quantizers of 16384, 1 of 65536 with ReLU, 1 softmax at n = 32
   and 2 residual adds, then the final RMSNorm, quantizer and argmax), without the embedding or any matmul.

One JSON object per line on stdout as each part finishes, a summary on stderr.

  python3 check.py CHECK   (env: BATCHES=1,7,128,4096,16384,20480, REPS=20, STEP_REPS=5, TIME_B=4096,16384, UNITS_EXPECT,
                            UNITS_ANY_GPU=1 for a dry run on another card, SWEEP=0 to skip the sweeps, SWEEP_ONLY=1)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import add as AD  # noqa: E402
import argmax as AM  # noqa: E402
import bench as BE  # noqa: E402
import lib as L  # noqa: E402
import quant as Q  # noqa: E402
import relu as RL  # noqa: E402
import rmsnorm as RN  # noqa: E402
import softmax as SM  # noqa: E402
import sweep as SW  # noqa: E402
from verity_config.settings import of  # noqa: E402

settings = of(__file__)
HEADS = BE.HEADS
#: each arch's card, the default UNITS_EXPECT
EXPECT = {"sm_90a": "NVIDIA H200", "sm_120a": "NVIDIA RTX PRO 6000"}
#: how many disagreeing chunks of an op are compared word by word
DIFF_CHUNKS = 4
JUNK = (0x7FC0, 0xFFC1, 0x7F80, 0xFF80, 0x7F7F, 0xFF7F)
POISON = {torch.int8: -0x5B, torch.int16: -0x5A5B, torch.int32: -0x5A5A5A5B}  # 0xA5, 0xA5A5, 0xA5A5A5A5


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def identity() -> dict:
    n = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    uuid = str(getattr(props, "uuid", "")).lower().removeprefix("gpu-")
    lease = (settings.GPU_LEASE_UUID or "").lower().removeprefix("gpu-")
    doc = {"devices": n, "name": props.name, "cc": f"{props.major}.{props.minor}", "uuid": uuid, "lease_uuid": lease or None,
           "sms": props.multi_processor_count, "torch": torch.__version__, "cuda": torch.version.cuda,
           "cuda_visible_devices": settings.CUDA_VISIBLE_DEVICES}
    arch = L.ARCHES.get((props.major, props.minor))
    doc["arch"] = arch
    want = settings.UNITS_EXPECT or EXPECT.get(arch, "")
    problems = []
    if n != 1:
        problems.append(f"{n} visible devices, want 1")
    if arch is None:
        problems.append(f"compute capability {props.major}.{props.minor}: build.sh has cubins only for "
                        f"{', '.join(f'{a} ({c[0]}.{c[1]})' for c, a in L.ARCHES.items())}")
    if not settings.UNITS_ANY_GPU:
        if not props.name.startswith(want):
            problems.append(f"{props.name!r} is not {want!r}")
        if lease and uuid != lease:
            problems.append(f"device {uuid} is not the lease's {lease}")
    doc["problems"] = problems
    return doc


def dev_words(a: np.ndarray, dev) -> torch.Tensor:
    """Unsigned words onto the device as the signed integer type of their width."""
    a = np.ascontiguousarray(a)
    signed = {1: np.int8, 2: np.int16, 4: np.int32}[a.dtype.itemsize]
    return torch.from_numpy(a.view(signed)).to(dev)


def poisoned(shape: tuple, dtype, dev) -> torch.Tensor:
    return torch.full(shape, POISON[dtype], dtype=dtype, device=dev)


def bf(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.bfloat16)


def bad(got: torch.Tensor, want: torch.Tensor) -> int:
    return int((got != want).sum())


def pick(P: int, B: int, seed: int, dev) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, P, (B,), generator=g).to(dev)


def compare(row: dict, pairs: dict) -> dict:
    """``row`` with each output's word count and mismatches; ``pairs`` is {name: (got, want)}."""
    for k, (got, want) in pairs.items():
        row[f"{k}_words"] = got.numel()
        row[f"{k}_mismatches"] = bad(got, want)
    return row


def failed(row: dict) -> bool:
    return (any(v for k, v in row.items() if k.endswith("_mismatches")) or row.get("inputs_unchanged") is False
            or row.get("control_rejected") is False)


# ---- the units ------------------------------------------------------------------------------------------------------------

def rmsnorm_case(z, N: int, B: int | None, dev, seed: int) -> list[dict]:
    X, Wg, OUT = (dev_words(z[f"{N}.{k}"], dev) for k in ("x", "w", "out"))
    CH, GR, RS = (dev_words(z[f"{N}.{k}"], dev) for k in ("chunks", "groups", "rs"))
    P = X.shape[0]
    sid = torch.arange(P, device=dev) if B is None else pick(P, B, seed, dev)
    x = X[sid]
    rows = []
    for k in (0, 1):
        before, wb = x.clone(), Wg[k].clone()
        out = poisoned(x.shape, torch.int16, dev)
        parts = {"chunks": poisoned(CH[sid].shape, torch.int32, dev), "groups": poisoned(GR[sid].shape, torch.int32, dev),
                 "rs": poisoned(RS[sid].shape, torch.int32, dev)}
        RN.gpu(bf(x), bf(Wg[k]), out=bf(out), parts=parts)
        torch.cuda.synchronize()
        row = compare({"part": "rmsnorm", "N": N, "B": len(sid), "gain": k, "rows": int(torch.unique(sid).numel())},
                      {"out": (out, OUT[k][sid]), "chunks": (parts["chunks"], CH[sid]), "groups": (parts["groups"], GR[sid]),
                       "rs": (parts["rs"], RS[sid])})
        row["control_rejected"] = bad(poisoned(x.shape, torch.int16, dev), OUT[k][sid]) > 0
        row["inputs_unchanged"] = torch.equal(before, x) and torch.equal(wb, Wg[k])
        rows.append(row)
    return rows


def quant_case(z, N: int, relu: bool, B: int | None, dev, seed: int) -> dict:
    key = f"{N},{int(relu)}"
    X, CODES, SCALE, TREE = (dev_words(z[f"{key}.{k}"], dev) for k in ("x", "codes", "scale", "tree"))
    P = X.shape[0]
    sid = torch.arange(P, device=dev) if B is None else pick(P, B, seed, dev)
    x = X[sid]
    before = x.clone()
    codes, scale, tree = poisoned(x.shape, torch.int8, dev), poisoned((len(sid),), torch.int32, dev), \
        poisoned((len(sid), TREE.shape[1]), torch.int16, dev)
    Q.gpu(bf(x), relu, codes=codes, scale_out=scale, tree=tree)
    torch.cuda.synchronize()
    row = compare({"part": "quant", "N": N, "relu": relu, "B": len(sid), "rows": int(torch.unique(sid).numel())},
                  {"codes": (codes, CODES[sid]), "scale": (scale, SCALE[sid]), "tree": (tree, TREE[sid])})
    row["control_rejected"] = bad(poisoned(x.shape, torch.int8, dev), CODES[sid]) > 0
    row["inputs_unchanged"] = torch.equal(before, x)
    return row


def junk_tails(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """x (R, L) int16 with every position >= n[row] replaced by JUNK words, cycling."""
    j = torch.arange(x.shape[1], device=x.device)
    junk = torch.tensor([w - 0x10000 if w >= 0x8000 else w for w in JUNK], dtype=torch.int16, device=x.device)
    return torch.where(j[None, :] >= n[:, None], junk[j % len(JUNK)][None, :], x)


def softmax_run(x, lens, Pp, Mm, Rr, sid, nrow) -> dict:
    R, Lw = x.shape
    before = x.clone()
    p = poisoned((R, Lw), torch.int16, x.device)
    parts = {"m": poisoned((R,), torch.int16, x.device), "r": poisoned((R,), torch.int32, x.device)}
    SM.gpu(bf(x), lens, HEADS, out=bf(p), parts=parts)
    torch.cuda.synchronize()
    i = nrow.long() - 1
    row = compare({}, {"p": (p, Pp[sid, i]), "m": (parts["m"], Mm[sid, i]), "r": (parts["r"], Rr[sid, i])})
    row["inputs_unchanged"] = torch.equal(before, x)
    return row


def softmax_case(z, B: int, dev, seed: int) -> list[dict]:
    X, Pp, Mm, Rr = (dev_words(z[k], dev) for k in ("x", "p", "m", "r"))
    P, Lw = X.shape
    R = B * HEADS
    rows = []
    totals = {"p": [0, 0], "m": [0, 0], "r": [0, 0]}
    for n in range(1, Lw + 1):
        sid = pick(P, R, seed + n, dev)
        lens = torch.full((B,), n, dtype=torch.int32, device=dev)
        nrow = lens.repeat_interleave(HEADS)
        r = softmax_run(junk_tails(X[sid], nrow), lens, Pp, Mm, Rr, sid, nrow)
        for k in totals:
            totals[k] = [totals[k][0] + r[f"{k}_words"], totals[k][1] + r[f"{k}_mismatches"]]
        rows.append({"part": "softmax", "B": B, "rows": R, "n": n, **r})
    sid = pick(P, R, seed, dev)
    g = torch.Generator().manual_seed(seed + 99)
    lens = torch.randint(1, Lw + 1, (B,), generator=g, dtype=torch.int32).to(dev)
    nrow = lens.repeat_interleave(HEADS)
    r = softmax_run(X[sid], lens, Pp, Mm, Rr, sid, nrow)
    r["control_rejected"] = bad(poisoned((R, Lw), torch.int16, dev), Pp[sid, nrow.long() - 1]) > 0
    rows.append({"part": "softmax", "B": B, "rows": R, "n": "mixed", **r})
    return rows


def argmax_case(z, V: int, B: int | None, dev, seed: int) -> dict:
    X, TOK, BEST, IDX = (dev_words(z[f"{V}.{k}"], dev) for k in ("x", "token", "best", "idx"))
    P = X.shape[0]
    sid = torch.arange(P, device=dev) if B is None else pick(P, B, seed, dev)
    x = X[sid]
    before = x.clone()
    tok = poisoned((len(sid),), torch.int16, dev)
    parts = {"best": poisoned((len(sid), BEST.shape[1]), torch.int16, dev),
             "idx": poisoned((len(sid), IDX.shape[1]), torch.int16, dev)}
    AM.gpu(bf(x), out=tok, parts=parts)
    torch.cuda.synchronize()
    row = compare({"part": "argmax", "V": V, "B": len(sid), "rows": int(torch.unique(sid).numel())},
                  {"token": (tok, TOK[sid]), "best": (parts["best"], BEST[sid]), "idx": (parts["idx"], IDX[sid])})
    row["control_rejected"] = bad(poisoned((len(sid),), torch.int16, dev), TOK[sid]) > 0
    row["inputs_unchanged"] = torch.equal(before, x)
    return row


def relu_case(z, B: int, dev) -> dict:
    X, Y = dev_words(z["x"], dev), dev_words(z["y"], dev)
    x = X.repeat(B, 1)
    before = x.clone()
    y = poisoned(x.shape, torch.int16, dev)
    RL.gpu(bf(x), out=bf(y))
    torch.cuda.synchronize()
    row = compare({"part": "relu", "B": B, "words_per_row": X.numel()}, {"y": (y, Y.repeat(B, 1))})
    row["control_rejected"] = bad(poisoned(x.shape, torch.int16, dev), Y.repeat(B, 1)) > 0
    row["inputs_unchanged"] = torch.equal(before, x)
    return row


def add_case(z, B: int | None, dev, seed: int) -> dict:
    A, Bw, Y = (dev_words(z[k], dev) for k in ("a", "b", "y"))
    P = A.shape[0]
    sid = torch.arange(P, device=dev) if B is None else pick(P, B, seed, dev)
    a, b = A[sid], Bw[sid]
    ab, bb = a.clone(), b.clone()
    y = poisoned(a.shape, torch.int16, dev)
    AD.gpu(bf(a), bf(b), out=bf(y))
    torch.cuda.synchronize()
    want = Y[sid]
    row = compare({"part": "add", "N": A.shape[1], "B": len(sid), "rows": int(torch.unique(sid).numel())}, {"y": (y, want)})
    row["control_rejected"] = bad(poisoned(a.shape, torch.int16, dev), want) > 0
    row["inputs_unchanged"] = torch.equal(ab, a) and torch.equal(bb, b)
    row["torch_add_differs"] = bad((bf(a) + bf(b)).view(torch.int16), want)
    return row


def sweeps(check: str, arch: str) -> list[dict]:
    with open(os.path.join(check, "sweep.json")) as f:
        want = json.load(f)
    words = "units_sweep_words" in L.lib().fns
    rows = []
    for op in SW.ARCH_OPS[arch]:
        w = [int(h, 16) for h in want[str(op)]["chunks"]]
        got = SW.gpu(op)
        diff = [c for c, (g, e) in enumerate(zip(got, w)) if g != e] + list(range(min(len(got), len(w)), max(len(got), len(w))))
        row = {"part": "sweep", "op": op, "name": SW.OPS[op][0], "n": SW.OPS[op][1], "chunks": len(w),
               "chunks_mismatching": len(diff), "first_mismatching_chunks": diff[:16]}
        if words and diff:
            try:
                row["differences"] = [SW.first_difference(op, c, SW.gpu_words(op, c)) for c in diff[:DIFF_CHUNKS]]
            except Exception as e:  # noqa: BLE001  (the models read verity: the row's verdict stands without its diagnosis)
                row["differences_error"] = repr(e)
        rows.append(row)
    return rows


def main(argv: list[str]) -> int:
    check = argv[1] if len(argv) > 1 else os.path.join(HERE, "check")
    ident = identity()
    emit({"part": "identity", **ident})
    if ident["problems"]:
        print("identity: " + "; ".join(ident["problems"]), file=sys.stderr)
        return 3
    dev = torch.device("cuda", 0)
    lib = L.lib()
    emit({"part": "cubin", "arch": lib.arch, "path": str(lib.path), "sha256": lib.sha256})
    failures = []
    if settings.SWEEP:
        for row in sweeps(check, lib.arch):
            emit(row)
            if row["chunks_mismatching"]:
                failures.append(f"sweep {row['name']}")
    if settings.SWEEP_ONLY:
        if failures:
            print("FAILED: " + ", ".join(failures), file=sys.stderr)
            return 4
        print(f"sweeps: 0 mismatching chunks in ops {list(SW.ARCH_OPS[lib.arch])} on {lib.arch}", file=sys.stderr)
        return 0
    z = {k: np.load(os.path.join(check, f"{k}.npz")) for k in ("rmsnorm", "quant", "softmax", "argmax", "relu", "add")}
    batches = list(settings.BATCHES)

    def run(rows, label: str) -> None:
        for row in rows if isinstance(rows, list) else [rows]:
            emit(row)
            if failed(row):
                failures.append(f"{label} {row.get('n', '')}".strip())

    for B in batches:
        run(rmsnorm_case(z["rmsnorm"], 16384, B, dev, seed=B), f"rmsnorm 16384 B={B}")
        run(quant_case(z["quant"], 16384, False, B, dev, seed=B), f"quant 16384 B={B}")
        run(quant_case(z["quant"], 65536, True, B, dev, seed=B + 1), f"quant 65536 relu B={B}")
        run(softmax_case(z["softmax"], B, dev, seed=B), f"softmax B={B}")
        run(argmax_case(z["argmax"], 256, B, dev, seed=B), f"argmax 256 B={B}")
        run(relu_case(z["relu"], B, dev), f"relu B={B}")
        run(add_case(z["add"], B, dev, seed=B), f"add 16384 B={B}")
    for N in sorted({int(k.split(".")[0]) for k in z["rmsnorm"].files} - {16384}):
        run(rmsnorm_case(z["rmsnorm"], N, None, dev, seed=0), f"rmsnorm {N}")
    for key in sorted({k.split(".")[0] for k in z["quant"].files} - {"16384,0", "65536,1"}):
        N, r = (int(v) for v in key.split(","))
        run(quant_case(z["quant"], N, bool(r), None, dev, seed=0), f"quant {key}")
    for V in sorted({int(k.split(".")[0]) for k in z["argmax"].files} - {256}):
        run(argmax_case(z["argmax"], V, None, dev, seed=0), f"argmax {V}")
    if failures:
        print("FAILED: " + ", ".join(failures), file=sys.stderr)
        return 4
    print(f"check: 0 mismatches at B {batches} (softmax at every length 1..32 and mixed), the pools' other widths, and "
          f"every sweep", file=sys.stderr)
    for B in BE.runs():
        for row in BE.timing(dev, B, settings.REPS, settings.STEP_REPS):
            emit(row)
            print(BE.summary(row), file=sys.stderr)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
