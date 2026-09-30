#!/usr/bin/env python3
"""Where v2's unpromoted chain is exact on one RTX PRO 6000 (sm_120), over whole 8,192² units: the node-2 job for
`no-aligned-exact-region/sm120-unpromoted` (internal/pouw/ttout-restatements.md §7), and `fp8-merge-rate/sm120` at fragment
granularity on the same formed operands (exact.cu; the assessor's measurement, not a bench arm).

A step of word (i, j) at atom t is exact when the atom's FP32 output equals its accumulator plus the exact sum of its 32
products.  The lemma's region is 1,024 rows x 512 columns of a unit's output, all exact on every atom of a 32-atom range.

1. Identity as sm120_chain/run.py (one visible RTX PRO 6000 at cc 12.0 whose UUID is --expect-uuid; exit 3 otherwise).
2. Build: exact.cu for sm_120a (no -ftz, no fast math), and no FADD, FFMA or FMUL with `.FTZ` in its SASS (every `.FTZ`
   op is recorded: the compiler's integer division has an F2I.FTZ); exit 5 otherwise.
3. `refcheck`, the first item (exit 4 on any mismatch): on 128-row units of gaussian, zero-slices and
   aligned-spikes-pm1-r64 at k = 1,024 and 8,192, the card's forming equals `pearlc_census.form_s5` code for code; the
   card's per-(word, atom) flags and final words equal the twin's (`census.step` on the sm_120 atom, the exact sum in int64)
   on every word at k = 1,024 and on the first 64 x 64 tile at k = 8,192; `window_bits` equals the windows recomputed from
   the flags, both orientations; and on that tile the vectorized greedy equals a line-for-line port of
   `internal/pouw/cheap-binding/pearlc_exact_runs.py`'s metrics (steps, run32, tile, frag, win, rect), which are reported.
4. `unit:<family>`, per census family (`unit:<family>@<run>` for the census's draw `run`, --runs): A and B, 8,192 x 8,192 codes each, formed on the card by §5 (the host draws X, s,
   rho, alpha, C and E' exactly as `form_s5` does; `formed` in sm120_chain/run.py at 8,192 rows), gated on 4,096 sampled
   elements; the unpromoted chain with flags (2 GB), gated on 512 sampled words over all 256 atoms (and `window_bits` on the
   same words over every window, both orientations); then for every one of
   the 225 sliding 32-atom windows: the share of words exact over it, the natural 16 x 8 fragments exact over it, and a
   greedy biclique (rows by exact count, columns intersected; then every row holding the chosen columns added), on row
   bitsets and on column bitsets; on the 8 aligned windows and the 2 densest others, `intersect_greedy` too (from the 4
   fullest rows, add the row keeping the most columns; it finds a planted block the count order misses).  The report's
   `lemma_block_found` is whether any search found 1,024 rows x 512 columns in any window.
   The same forming gives the salt-dead share (`Pouw.PearlC.SaltDead.saltDead` at Π's β = σα·f32(√32/(256·16)), E flat at
   ±44 so E' = e4m3(f32(β·44))): over the old support (F flat ±44), over the corrected one (F ±11/4, bc-3006c44a's Ruling
   on F2), and with the census's own flat line (E' = e4m3(σα/16), F ±11/4); per side, all rows and Π's admitted rows,
   the worst admitted row beside P3's 1/64 bound; `dead_mask` on the card, gated on 2,048 elements per screen against
   census.step.  The same A and B give the merge rates: Strassen's diagonal pair (sum and difference both E4M3, as pearlc_merge_rate.py)
   and each of Strassen's 10 pre-adds, per element and per whole fragment (16 x 32 on A, 32 x 8 on B), and random pairs;
   the (A fragment, B fragment) pairs at one k block where all 10 are exact (one Strassen step's condition); and the largest block where the diagonal pair's sum and difference are both E4M3 (`greedy` and `intersect_greedy` on
   A11, A22 and B11, B22 in place: rows and columns are not permuted across quadrants).
5. `acts:<linear>` with --acts DIR (a pearlc_capture directory; --acts-layers filters by regex): the same measurements with
   A = the linear's captured token rows (the first multiple of 64) and B = its weight rows, formed as in 4; the region
   search counts only words whose A row and B row Π admits (the noise-floor and liveness rules, unsplit).
6. Restartable: one JSON per item in --out, a state file, and exit 99 once --budget-min is spent while items remain; exit 0
   after summary.json.  An item started 3 times without finishing fails the job (exit 1).

  gpu-lease 1 --wait -- env CUDA_DEVICE_ORDER=PCI_BUS_ID python3 benchmarks/pouw/sm120_exact/run.py --gpu lease \
      --expect-uuid lease --clock-label locked-2100 --out DIR
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pearlc_census as census  # noqa: E402
import run_result  # noqa: E402
from sm120_chain.run import Dev as ChainDev, Nvml, emit, i32, u64, vp  # noqa: E402

ATOM = "sm120-e4m3-k32"
KERNELS = ("form_codes", "chain_flags", "window_bits", "dead_mask")
DEAD_BETA = np.float32(np.sqrt(32) / (256 * 16))  # Π's β_row = σα·f32(√32/(256·16)) (#449's pearl_c.py, LINE_NORM_V1 = 16)
DEAD_SCREENS = {  # (E side, F's flat code): SaltDead.support before and after bc-3006c44a's Ruling on F2 (10:00Z)
    "old_F44": ("scheme", 44.0), "new_F11/4": ("scheme", 2.75), "census_line_F11/4": ("census", 2.75)}
LINE_NORM = 16.0
WIN = 32
REGION = (1024, 512)
CHECK_FAMILIES = ("gaussian", "zero-slices", "aligned-spikes-pm1-r64")
UNIT_FAMILIES = (*census.FAMILIES, *census.ACTIVATION_FAMILIES, *census.OUTLIER_FAMILIES, *census.FREEZE_FAMILIES)
PAPER = {  # ttout-restatements.md §7, pearlc_exact_runs.py on #449's form_v1 (other operands than these: compare in law)
    (1024, "gaussian"): "97.76% 54.0% | 0/0 | 17.29%/6 | 54.0%-54.0% | 5x20",
    (1024, "zero-slices"): "97.41% 48.9% | 0/0 | 12.70%/6 | 48.9%-48.9% | 3x29",
    (1024, "aligned-spikes-pm1-r64"): "13.14% 0.0% | 0/0 | 0.00%/0 | 0.0%-0.0% | 0x0",
    (8192, "gaussian"): "96.21% 62.4% | 0/0 | 4.49%/5 | 27.8%-55.7% | 3x27",
    (8192, "zero-slices"): "94.98% 51.4% | 0/0 | 2.42%/4 | 19.0%-47.2% | 3x23",
    (8192, "aligned-spikes-pm1-r64"): "2.66% 0.0% | 0/0 | 0.00%/0 | 0.0%-0.0% | 0x0",
}


class Dev(ChainDev):
    def load(self, cubin: bytes) -> None:
        self.mod = vp()
        self.ck("cuModuleLoadData", C.byref(self.mod), cubin)
        for name in KERNELS:
            f = vp()
            self.ck("cuModuleGetFunction", C.byref(f), self.mod, name.encode())
            self.fns[name] = f

    def fetch(self, p: int, count: int, dtype, offset: int = 0) -> np.ndarray:
        out = np.empty(count, dtype)
        self.ck("cuMemcpyDtoH_v2", out.ctypes.data_as(vp), u64(p + offset), out.nbytes)
        return out

    def form(self, Cf: np.ndarray, Ec: np.ndarray, Fc: np.ndarray) -> tuple[int, int]:
        """A' on the card; returns the codes' and C's device pointers."""
        rows, k = Cf.shape
        dc, de, df = self.put(Cf.astype(np.float32)), self.put(Ec), self.put(Fc)
        out = self.alloc(rows * k)
        warps = (rows // 16) * (k // 8)
        self.launch("form_codes", ((warps + 7) // 8, 1, 1), 256, 0, [u64(dc), u64(de), u64(df), u64(out), i32(rows), i32(k)])
        self.sync()
        for p in (de, df):
            self.free(p)
        return out, dc

    def dead(self, dc: int, e: np.ndarray, f: int, rows: int, k: int) -> np.ndarray:
        de, out = self.put(e), self.alloc(rows * k)
        warps = (rows // 16) * (k // 8)
        self.launch("dead_mask", ((warps + 7) // 8, 1, 1), 256, 0, [u64(dc), u64(de), C.c_uint(f), u64(out), i32(rows), i32(k)])
        self.sync()
        mask = self.fetch(out, rows * k, np.uint8).reshape(rows, k)
        self.free(de)
        self.free(out)
        return mask

    def chain(self, da: int, db: int, m: int, n: int, k: int) -> tuple[int, int]:
        flags, out = self.alloc(4 * (k // 1024) * m * n), self.alloc(4 * m * n)
        self.launch("chain_flags", (n // 64, m // 64, 1), 128, 0, [u64(da), u64(db), u64(flags), u64(out), i32(m), i32(n), i32(k)])
        self.sync()
        return flags, out

    def windows(self, flags: int, M: int, m: int, n: int, T: int, t0: int, nt: int, tr: int) -> np.ndarray:
        assert (t0 + nt + 30) // 32 < (t0 >> 5) + 16
        self.launch("window_bits", ((m * n + 255) // 256, 1, 1), 256, 0,
                    [u64(flags), u64(M), i32(m), i32(n), i32(T), i32(t0), i32(nt), i32(tr), i32(t0 >> 5)])
        self.sync()
        rows, cols = (n, m) if tr else (m, n)
        return self.fetch(M, nt * rows * (cols // 32), np.uint32).reshape(nt, rows, cols // 32)


def build(out: Path, nvcc: str) -> tuple[bytes, dict]:
    cuobjdump = str(Path(nvcc).with_name("cuobjdump"))
    cubin = out / "exact.cubin"
    cmd = [nvcc, "-O3", "-std=c++17", "-gencode", "arch=compute_120a,code=sm_120a", "-Xptxas", "-v", "-cubin",
           "-o", str(cubin), str(HERE / "exact.cu")]
    ptx = subprocess.run(cmd, capture_output=True, text=True, check=True).stderr
    sass = subprocess.run([cuobjdump, "-sass", str(cubin)], capture_output=True, text=True, check=True).stdout
    version = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    problems = [ln for ln in ptx.splitlines() if re.search(r"warning|error", ln, re.I)]
    kernels: dict[str, dict[str, int]] = {}
    cur = None
    for line in sass.splitlines():
        fm = re.search(r"Function : (\w+)", line)
        if fm:
            cur = fm.group(1)
            kernels[cur] = {}
            continue
        ins = re.search(r"\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", line)
        if cur and ins:
            op = ins.group(1)
            for key in ("QMMA.16832.F32.E4M3.E4M3", "FADD", "FFMA", "FMUL", "LDL", "STL"):
                if op == key or op.startswith(key + "."):
                    kernels[cur][key] = kernels[cur].get(key, 0) + 1
            if re.match(r"F(ADD|FMA|MUL)\.", op) and ".FTZ" in op:
                problems.append(f"{cur}: {op}")
            if ".FTZ" in op:
                kernels[cur]["ftz:" + op] = kernels[cur].get("ftz:" + op, 0) + 1
    for name in ("form_codes", "chain_flags"):
        if not kernels.get(name, {}).get("QMMA.16832.F32.E4M3.E4M3"):
            problems.append(f"{name}: no QMMA.16832.F32.E4M3.E4M3")
    spills = [ln.strip() for ln in ptx.splitlines() if re.search(r"bytes spill", ln)]
    data = cubin.read_bytes()
    return data, {"nvcc": version, "cmd": " ".join(cmd), "cubin_sha256": hashlib.sha256(data).hexdigest(),
                  "source_sha256": hashlib.sha256((HERE / "exact.cu").read_bytes()).hexdigest(), "sass": kernels,
                  "ptxas_spills": spills, "problems": problems}


# ---- operands -----------------------------------------------------------------------------------------------------------

def prep(X: np.ndarray, Fc: np.ndarray, rng: np.random.Generator):
    """form_s5's host side with delta = 1: C = f32(alpha x) and E' (its first draw), the rows' domain verdicts, and each
    row's flat E' code per salt-dead screen."""
    rows, k = X.shape
    s = np.abs(X).max(1)
    rho = np.sqrt((X[:, ::8] ** 2).mean(1))
    sig = 1.0 * rho
    alpha = np.where(s + 4 * sig > 0, census.Q_MAX / np.maximum(s + 4 * sig, 1e-300), 0.0)
    floor_ok = sig * alpha >= 1.0
    live_ok = np.array([(np.abs(X[r0:r0 + 512]) >= 8 * sig[r0:r0 + 512, None]).sum(1) * 64 <= k
                        for r0 in range(0, rows, 512)]).ravel()
    Cf = np.empty((rows, k), np.float32)
    for r0 in range(0, rows, 512):
        Cf[r0:r0 + 512] = (alpha[r0:r0 + 512, None] * X[r0:r0 + 512]).astype(np.float32)
    Ec = census.enc(census.unit_lines(rows, rng) * (sig * alpha / LINE_NORM)[:, None])
    beta = (sig.astype(np.float32) * alpha.astype(np.float32)).astype(np.float32) * DEAD_BETA
    flat = {"scheme": census.enc((beta * np.float32(44.0)).astype(np.float32).astype(np.float64)),
            "census": census.enc(sig * alpha / LINE_NORM)}
    return Cf, Ec, floor_ok, live_ok, flat


def operands(dev: Dev, fam: str, k: int, rows: int, run: int = 1):
    """A and B's codes formed on the card (device pointers, with C's), with the host inputs kept for the gates."""
    XA, XB, _, _ = census.family_rows(fam, k, 1.0, rows, rows, run)
    return form_sides(dev, XA, XB, fam, k, run)


def form_sides(dev: Dev, XA: np.ndarray, XB: np.ndarray, key: str, k: int, run: int = 1):
    """§5's forming of activation rows XA and weight rows XB, with the census's seeds under `key`."""
    FcA, FcB = census.s5_lines(k, run, LINE_NORM)
    sides = {}
    for side, X, Fc in (("A", XA, FcA), ("B", XB, FcB)):
        Cf, Ec, floor_ok, live_ok, flat = prep(X, Fc, census._rng(run, key, k, 1.0, f"unit-{side}"))
        ptr, cptr = dev.form(Cf, Ec, Fc)
        sides[side] = {"ptr": ptr, "cptr": cptr, "C": Cf, "E": Ec, "F": Fc, "row_ok": floor_ok & live_ok, "flat": flat,
                       "rows": X.shape[0], "floor_ok": bool(floor_ok.all()), "live_ok": bool(live_ok.all())}
    return sides


def capture_linear(rd: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """One linear of a pearlc_capture directory: X (its tokens, sha256-checked) and W from the manifest's snapshot."""
    import pearlc_activations as PA
    man = json.loads((rd / "manifest.json").read_text())
    ent = man["inputs"][name]
    f = rd / ent["file"]
    if PA.sha256(f) != ent["sha256"]:
        raise SystemExit(f"{f}: sha256 differs from {rd / 'manifest.json'}")
    snap = Path(man["snapshot"])
    wmap = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    wname = name + ".weight"
    return np.load(f).astype(np.float64), PA.safetensor(snap / wmap[wname], wname).astype(np.float64)


def dead_twin(Cf: np.ndarray, e: np.ndarray, f: int, r: np.ndarray, l: np.ndarray) -> np.ndarray:
    """SaltDead over the flat pairs for elements (r, l), by census.step: the value is the same for j = 0 .. 32."""
    P = len(r)
    vals = []
    for j in range(33):
        a = np.repeat(e[r][:, None], 32, 1)
        b = np.full((P, 32), f, np.uint8)
        b[:, :j] |= 0x80
        D, _ = census.step(Cf[r, l].astype(np.float64), a, b)
        vals.append(census.val(census.enc(D)))
    v = np.stack(vals)
    return (v == v[0]).all(0)


def dead_stats(dev: Dev, side: dict, rows: int, k: int, rng: np.random.Generator) -> dict:
    """Per screen: the salt-dead share of the side's elements, over all rows and over the rows Π admits, the worst admitted
    row's share (P3's bound: 1/64), gated on 2,048 sampled elements against dead_twin."""
    out, bad = {}, 0
    ok = side["row_ok"]
    for name, (eside, fval) in DEAD_SCREENS.items():
        f = int(census.enc(np.array([fval]))[0])
        e = side["flat"][eside]
        mask = dev.dead(side["cptr"], e, f, rows, k)
        r, l = rng.integers(0, rows, 2048), rng.integers(0, k, 2048)
        bad += int((mask[r, l].astype(bool) != dead_twin(side["C"], e, f, r, l)).sum())
        per_row = mask.sum(1, dtype=np.int64) / k
        out[name] = {"share": round(float(per_row.mean()), 8),
                     "share_admitted": round(float(per_row[ok].mean()), 8) if ok.any() else None,
                     "worst_row_admitted": round(float(per_row[ok].max()), 8) if ok.any() else None,
                     "admitted_rows_over_1/64": int((per_row[ok] > 1 / 64).sum()), "admitted_rows": int(ok.sum()),
                     "f_code": hex(f)}
    out["gate_mismatches"] = bad
    return out


def form_twin(Cf, Ec, Fc, r, l) -> np.ndarray:
    D, _ = census.step(Cf[r, l].astype(np.float64), Ec[r], Fc[l])
    return census.enc(D)


def ival(codes: np.ndarray) -> np.ndarray:
    return np.rint(census.val(codes) * 512).astype(np.int64)


def chain_twin(A: np.ndarray, B: np.ndarray):
    """(P, k) codes of P words: per atom whether the step is exact (value), whether no bit was lost, and the final word."""
    P, k = A.shape
    T = k // 32
    ia, ib = ival(A), ival(B)
    acc = np.zeros(P)
    flags, nolost = np.zeros((P, T), bool), np.zeros((P, T), bool)
    for t in range(T):
        sl = slice(32 * t, 32 * t + 32)
        D, ex = census.step(acc, A[:, sl], B[:, sl])
        s = (ia[:, sl] * ib[:, sl]).sum(1)
        flags[:, t] = np.rint(D * 2.0 ** 18).astype(np.int64) == np.rint(acc * 2.0 ** 18).astype(np.int64) + s
        nolost[:, t] = ex
        acc = D
    return flags, nolost, acc.astype(np.float32).view(np.uint32)


def card_flags(dev: Dev, flags: int, m: int, n: int, T: int, ii: np.ndarray, jj: np.ndarray, tile: int = 0):
    """The card's flags for words (ii, jj) as (P, T) bools, the (tile, tile, T) corner, and the exact-step count."""
    planes = T // 32
    got = np.zeros((len(ii), T), bool)
    corner = np.zeros((tile, tile, T), bool)
    total = 0
    for p in range(planes):
        pl = dev.fetch(flags, m * n, np.uint32, 4 * p * m * n)
        total += int(popc(pl).sum())
        w = pl[ii * n + jj]
        got[:, 32 * p:32 * p + 32] = ((w[:, None] >> np.arange(32, dtype=np.uint32)) & 1).astype(bool)
        if tile:
            c = pl.reshape(m, n)[:tile, :tile]
            corner[:, :, 32 * p:32 * p + 32] = ((c[:, :, None] >> np.arange(32, dtype=np.uint32)) & 1).astype(bool)
        del pl
    return got, corner, total


def popc(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    b = x.view(np.uint8)
    return np.unpackbits(b, axis=-1).reshape(*x.shape, -1).sum(-1)


def same_words(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x == y) | (((x & 0x7FFFFFFF) == 0) & ((y & 0x7FFFFFFF) == 0))


# ---- the region search ----------------------------------------------------------------------------------------------------

def greedy(bits: np.ndarray) -> dict:
    """bits: (rows, cols / 32) uint32 row bitsets of one window.  Rows by exact count (stable), columns intersected: the
    best rows x cols over prefixes (`plain`, pearlc_exact_runs.greedy_rect's rule); then for the prefixes at the best and
    at 16 .. 1,024 rows, every row holding the prefix's columns (`grown`)."""
    rows = bits.shape[0]
    M = bits.view(np.uint64) if bits.shape[1] % 2 == 0 else bits
    cnt = popc(M).sum(1, dtype=np.int64)
    order = np.argsort(-cnt, kind="stable")
    acc = np.bitwise_and.accumulate(M[order], axis=0)
    cols = popc(acc).sum(1, dtype=np.int64)
    r = np.arange(1, rows + 1)
    b = int((r * cols).argmax())
    plain = (int(r[b]), int(cols[b])) if cols[b] else (0, 0)
    grown = plain
    for c in sorted({b, *(x - 1 for x in (16, 64, 256, 512, 1024) if x <= rows)}):
        if not cols[c]:
            continue
        full = int(((M & acc[c]) == acc[c]).all(1).sum())
        if full * int(cols[c]) > grown[0] * grown[1]:
            grown = (full, int(cols[c]))
    at = {str(x): int(cols[x - 1]) for x in (16, 64, 256, 512, 1024) if x <= rows}
    return {"plain": plain, "grown": grown, "cols_at_rows": at,
            "rows_with_cols_ge_%d" % REGION[1]: int((cols >= REGION[1]).sum()),
            "rows_with_cols_ge_%d" % REGION[0]: int((cols >= REGION[0]).sum())}


def intersect_greedy(bits: np.ndarray, seeds: int = 4, steps: int = 64) -> tuple[int, int]:
    """A stronger search than `greedy`: from each of the `seeds` rows with most exact words, repeatedly add the row that keeps
    the most columns; at every step every row holding the current columns counts.  The best rows x cols found."""
    M = bits.view(np.uint64) if bits.shape[1] % 2 == 0 else bits
    cnt = popc(M).sum(1, dtype=np.int64)
    best = (0, 0)
    for s in np.argsort(-cnt, kind="stable")[:seeds]:
        cur = M[s].copy()
        used = np.zeros(len(M), bool)
        used[s] = True
        for _ in range(steps):
            inter = M & cur
            keep = popc(inter).sum(1, dtype=np.int64)
            c = int(popc(cur).sum())
            if c == 0:
                break
            full = int((keep == c).sum())
            if full * c > best[0] * best[1]:
                best = (full, c)
            keep[used | (keep == c)] = -1
            nxt = int(keep.argmax())
            if keep[nxt] <= 0:
                break
            cur &= M[nxt]
            used[nxt] = True
    return best


def fragments_exact(bits: np.ndarray) -> int:
    """Natural 16 x 8 fragments all exact over the window, from (m, n / 32) row bitsets."""
    m = bits.shape[0]
    g = np.bitwise_and.reduce(bits.reshape(m // 16, 16, -1), axis=1)
    return int((g.view(np.uint8) == 0xFF).sum())


def host_windows(planes: np.ndarray, t0: int, nt: int) -> np.ndarray:
    """(P, T/32) uint32 flag words -> (nt, P) bools: exact over [t, t + 32)."""
    P, np_ = planes.shape
    ext = np.concatenate([planes, np.zeros((P, 1), np.uint32)], 1).astype(np.uint64)
    out = np.zeros((nt, P), bool)
    for q, t in enumerate(range(t0, t0 + nt)):
        p, s = t >> 5, t & 31
        x = ((ext[:, p + 1] << np.uint64(32)) | ext[:, p]) >> np.uint64(s)
        out[q] = (x & np.uint64(0xFFFFFFFF)) == np.uint64(0xFFFFFFFF)
    return out


def pack_rows(ok: np.ndarray) -> np.ndarray:
    """(rows, cols) bools -> (rows, cols / 32) uint32, bit j % 32 of word j / 32 = column j."""
    return np.ascontiguousarray(np.packbits(np.ascontiguousarray(ok), axis=1, bitorder="little")).view(np.uint32)


# ---- pearlc_exact_runs.py's metrics, ported line for line (E[i][j][t] for one T x T tile) ----------------------------------

def _longest(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def _greedy_rect(ok, T):
    rows = sorted(range(T), key=lambda i: -sum(ok[i]))
    best = (0, 0)
    for r in range(1, T + 1):
        cols = sum(all(ok[i][j] for i in rows[:r]) for j in range(T))
        if r * cols > best[0] * best[1]:
            best = (r, cols)
    return best


def reference_metrics(E: list, T: int, W: int) -> dict:
    steps = sum(sum(sum(x) for x in row) for row in E) / (T * T * W)
    inrun = 0
    for i in range(T):
        for j in range(T):
            cur = 0
            for t in range(W + 1):
                if t < W and E[i][j][t]:
                    cur += 1
                else:
                    if cur >= 32:
                        inrun += cur
                    cur = 0
    run32 = inrun / (T * T * W)
    tile = [all(E[i][j][t] for i in range(T) for j in range(T)) for t in range(W)]
    frags = [[all(E[i][j][t] for i in range(fi, fi + 16) for j in range(fj, fj + 8)) for t in range(W)]
             for fi in range(0, T, 16) for fj in range(0, T, 8)]
    fshare = sum(sum(f) for f in frags) / (len(frags) * W)
    flong = max(_longest(f) for f in frags)
    dens, rects = [], []
    for t0 in range(0, W - 31, 32):
        ok = [[all(E[i][j][t0:t0 + 32]) for j in range(T)] for i in range(T)]
        dens.append(sum(map(sum, ok)) / (T * T))
        rects.append(_greedy_rect(ok, T))
    best = max(rects, key=lambda x: x[0] * x[1]) if rects else (0, 0)
    return {"steps": steps, "run32": run32, "tile_atoms": sum(tile), "tile_longest": _longest(tile), "frag_share": fshare,
            "frag_longest": flong, "win_min": min(dens) if dens else 0, "win_max": max(dens) if dens else 0,
            "rect": list(best), "rects": [list(x) for x in rects], "dens": dens,
            "row": f"{steps:.2%} {run32:.1%} | {sum(tile)}/{_longest(tile)} | {fshare:.2%}/{flong} | "
                   f"{min(dens) if dens else 0:.1%}-{max(dens) if dens else 0:.1%} | {best[0]}x{best[1]}"}


def vectorized_tile(E: np.ndarray) -> dict:
    """The same tile's aligned windows through this job's vectorized path (host_windows + greedy's `plain`)."""
    T = E.shape[0]
    W = E.shape[2]
    planes = np.packbits(E.reshape(T * T, W), axis=1, bitorder="little").view(np.uint32)
    t0s = list(range(0, W - 31, 32))
    dens, rects = [], []
    for t0 in t0s:
        ok = host_windows(planes, t0, 1)[0].reshape(T, T)
        dens.append(float(ok.mean()))
        rects.append(list(greedy(pack_rows(ok))["plain"]))
    return {"dens": dens, "rects": rects}


# ---- merge rates ----------------------------------------------------------------------------------------------------------

VALS = census.val(np.arange(256))
FINITE = ~np.isnan(VALS)
_SET = np.unique(VALS[FINITE])


def _is_e4m3(x: np.ndarray) -> np.ndarray:
    i = np.clip(np.searchsorted(_SET, x), 0, len(_SET) - 1)
    return _SET[i] == x


_VA, _VB = np.meshgrid(VALS, VALS, indexing="ij")
_OKF = FINITE[:, None] & FINITE[None, :]
LUT_SUM = _OKF & _is_e4m3(np.where(_OKF, _VA + _VB, 0.0))
LUT_DIFF = _OKF & _is_e4m3(np.where(_OKF, _VA - _VB, 0.0))
LUT_BOTH = LUT_SUM & LUT_DIFF


def frag_mask(ok: np.ndarray, fr: int, fc: int) -> np.ndarray:
    r, c = ok.shape
    return ok[: r - r % fr, : c - c % fc].reshape(r // fr, fr, c // fc, fc).all(axis=(1, 3))


def frag_count(ok: np.ndarray, fr: int, fc: int) -> tuple[int, int]:
    f = frag_mask(ok, fr, fc)
    return int(f.sum()), int(f.size)


def merge_stats(A: np.ndarray, Bs: np.ndarray, seed: int) -> dict:
    """A: (m, k) codes; Bs: (n, k) codes, B = Bs^T (k x n).  Strassen's quadrants A11 = A[:m/2, :k/2] .. and B11 = B[:k/2,
    :n/2] = Bs[:n/2, :k/2]^T ..; fragments 16 x 32 of A and 32 x 8 of B (8 rows x 32 columns of Bs)."""
    m, k = A.shape
    n = Bs.shape[0]
    h, kh, nh = m // 2, k // 2, n // 2
    A11, A12, A21, A22 = A[:h, :kh], A[:h, kh:], A[h:, :kh], A[h:, kh:]
    # in Bs's layout (rows j of n, columns l of k): B11 = Bs[:nh, :kh], B12 = Bs[nh:, :kh], B21 = Bs[:nh, kh:], B22 = Bs[nh:, kh:]
    B11, B12, B21, B22 = Bs[:nh, :kh], Bs[nh:, :kh], Bs[:nh, kh:], Bs[nh:, kh:]
    out = {}
    all5 = {"A": None, "B": None}
    for name, x, y, lut, fr in (("A11+A22", A11, A22, LUT_SUM, 16), ("A21+A22", A21, A22, LUT_SUM, 16),
                                ("A11+A12", A11, A12, LUT_SUM, 16), ("A21-A11", A21, A11, LUT_DIFF, 16),
                                ("A12-A22", A12, A22, LUT_DIFF, 16), ("B11+B22", B11, B22, LUT_SUM, 8),
                                ("B12-B22", B12, B22, LUT_DIFF, 8), ("B21-B11", B21, B11, LUT_DIFF, 8),
                                ("B11+B12", B11, B12, LUT_SUM, 8), ("B21+B22", B21, B22, LUT_SUM, 8)):
        ok = lut[x, y]
        f = frag_mask(ok, fr, 32)
        all5[name[0]] = f if all5[name[0]] is None else all5[name[0]] & f
        out[name] = {"element": round(float(ok.mean()), 6), "frag_hits": int(f.sum()), "frags": int(f.size)}
    # one Strassen step on a (16-row A fragment, 8-column B fragment) pair at one 32-deep k block needs all 10 pre-adds exact
    per_kb = all5["A"].sum(0).astype(np.int64) * all5["B"].sum(0)
    strassen = [int(per_kb.sum()), all5["A"].shape[0] * all5["B"].shape[0] * all5["A"].shape[1]]
    both_a = LUT_BOTH[A11, A22]
    both_b = LUT_BOTH[B11, B22]
    blocks = {}
    for name, ok in (("A11,A22", both_a), ("B11,B22", both_b)):
        bits = pack_rows(ok)
        g = greedy(bits)
        blocks[name] = {"plain": list(g["plain"]), "grown": list(g["grown"]), "intersect": list(intersect_greedy(bits)),
                        "need_fragment": [16 if name[0] == "A" else 8, 32]}
    rng = np.random.default_rng(seed)
    q = 1 << 20
    ra = LUT_BOTH[A.ravel()[rng.integers(0, A.size, q)], A.ravel()[rng.integers(0, A.size, q)]]
    fa, fna = frag_count(both_a, 16, 32)
    fb, fnb = frag_count(both_b, 8, 32)
    hits = sum(v["frag_hits"] for v in out.values())
    frags = sum(v["frags"] for v in out.values())
    return {"pair_A": round(float(both_a.mean()), 6), "pair_B": round(float(both_b.mean()), 6),
            "rand_A": round(float(ra.mean()), 6), "frag_both_A": [fa, fna], "frag_both_B": [fb, fnb],
            "preadds": out, "preadd_frag_hits": hits, "preadd_frags": frags, "pair_blocks": blocks,
            "all5_A": [int(all5["A"].sum()), int(all5["A"].size)], "all5_B": [int(all5["B"].sum()), int(all5["B"].size)],
            "strassen_frag_pairs": strassen,
            "zero_share_A": round(float(((A & 0x7F) == 0).mean()), 6), "zero_share_B": round(float(((Bs & 0x7F) == 0).mean()), 6)}


# ---- items ------------------------------------------------------------------------------------------------------------------

def refcheck(dev: Dev, fh) -> dict:
    res = {"item": "refcheck", "gates": [], "tiles": []}
    ok = True
    for k in (1024, 8192):
        T = k // 32
        for fam in CHECK_FAMILIES:
            rows = 128
            sides = operands(dev, fam, k, rows)
            A = dev.fetch(sides["A"]["ptr"], rows * k, np.uint8).reshape(rows, k)
            B = dev.fetch(sides["B"]["ptr"], rows * k, np.uint8).reshape(rows, k)
            XA, XB, _, _ = census.family_rows(fam, k, 1.0, rows, rows, 1)
            FcA, FcB = census.s5_lines(k, 1, LINE_NORM)
            refA = census.form_s5(XA, FcA, 1.0, LINE_NORM, census._rng(1, fam, k, 1.0, "unit-A"))[0]
            refB = census.form_s5(XB, FcB, 1.0, LINE_NORM, census._rng(1, fam, k, 1.0, "unit-B"))[0]
            g = {"gate": f"refcheck/{fam}/k{k}", "form_mismatches": int((A != refA).sum() + (B != refB).sum())}
            flags, out = dev.chain(sides["A"]["ptr"], sides["B"]["ptr"], rows, rows, k)
            words = dev.fetch(out, rows * rows, np.uint32)
            tile = 64
            if k == 1024:
                ii, jj = (x.ravel() for x in np.meshgrid(np.arange(rows), np.arange(rows), indexing="ij"))
            else:
                ii, jj = (x.ravel() for x in np.meshgrid(np.arange(tile), np.arange(tile), indexing="ij"))
            got, corner, total = card_flags(dev, flags, rows, rows, T, ii, jj, tile)
            want, nolost, fin = chain_twin(refA[ii], refB[jj])
            g["words"] = int(len(ii))
            g["flag_mismatches"] = int((got != want).sum())
            g["word_mismatches"] = int((~same_words(words[ii * rows + jj], fin)).sum())
            g["steps_exact_value"] = round(float(want.mean()), 6)
            g["steps_no_bit_lost"] = round(float(nolost.mean()), 6)
            planes = np.stack([dev.fetch(flags, rows * rows, np.uint32, 4 * p * rows * rows) for p in range(T // 32)], 1)
            M = dev.alloc(4 * (T - WIN + 1) * rows * (rows // 32))
            wb = 0
            for tr in (0, 1):
                card = dev.windows(flags, M, rows, rows, T, 0, T - WIN + 1, tr)
                host = host_windows(planes, 0, T - WIN + 1).reshape(-1, rows, rows)
                if tr:
                    host = host.transpose(0, 2, 1)
                wb += int((card != np.stack([pack_rows(h) for h in host])).sum())
            g["window_mismatches"] = wb
            dev.free(M)
            for p in (flags, out, *(sides[s][x] for s in "AB" for x in ("ptr", "cptr"))):
                dev.free(p)
            E = corner.tolist()
            refm = reference_metrics(E, tile, T)
            vec = vectorized_tile(corner)
            g["port_vs_vectorized_mismatches"] = int(sum(a != b for a, b in zip(refm["rects"], vec["rects"])) +
                                                     sum(abs(a - b) > 1e-12 for a, b in zip(refm["dens"], vec["dens"])))
            g["ok"] = all(g[x] == 0 for x in ("form_mismatches", "flag_mismatches", "word_mismatches", "window_mismatches",
                                              "port_vs_vectorized_mismatches"))
            ok &= g["ok"]
            emit(fh, g)
            res["gates"].append(g)
            row = {"k": k, "family": fam, "this_run": refm["row"], "restatements_s7": PAPER.get((k, fam)),
                   "forming": "census form_s5 (this run) vs #449 form_v1 (s7): other operands, compare in law"}
            emit(fh, {"tile": row})
            res["tiles"].append(row)
    res["ok"] = ok
    return res


def unit(dev: Dev, fh, item: str, k: int, rows: int, seed: int) -> dict:
    fam, _, run = item.split(":", 1)[1].partition("@")
    t_start = time.time()
    sides = operands(dev, fam, k, rows, int(run or 1))
    res = measure(dev, fh, item, fam, sides, k, seed, t_start, mask=False)
    res["run"] = int(run or 1)
    return res


def acts_unit(dev: Dev, fh, rd: Path, name: str, seed: int) -> dict:
    """A captured linear as a unit: A = its token rows (the first multiple of 64), B = its weight rows; the region search
    counts only words whose A row and B row Π admits (the unsplit rules)."""
    t_start = time.time()
    X, W = capture_linear(rd, name)
    X = X[: X.shape[0] - X.shape[0] % 64]
    sides = form_sides(dev, X, W, name, X.shape[1])
    del X, W
    res = measure(dev, fh, f"acts:{name}", name, sides, sides["A"]["C"].shape[1], seed, t_start, mask=True)
    res["capture"] = str(rd)
    return res


def measure(dev: Dev, fh, item: str, label: str, sides: dict, k: int, seed: int, t_start: float, mask: bool) -> dict:
    m, n = sides["A"]["rows"], sides["B"]["rows"]
    T = k // 32
    rng = np.random.default_rng(seed)
    okA, okB = sides["A"]["row_ok"], sides["B"]["row_ok"]
    res = {"item": item, "family": label, "m": m, "n": n, "k": k, "atom": ATOM, "forming": "s5, delta 1, line norm 16",
           "rules_ok": all(sides[s]["floor_ok"] and sides[s]["live_ok"] for s in "AB"),
           "rejected_by": [f"{r}-{s}" for s in "AB" for r, v in (("noise-floor", sides[s]["floor_ok"]),
                                                                    ("liveness", sides[s]["live_ok"])) if not v],
           "rows_admitted": {"A": int(okA.sum()), "B": int(okB.sum())}, "region_search_admitted_only": mask}
    codes = {}
    bad = 0
    res["salt_dead"] = {}
    for s in "AB":
        rows = sides[s]["rows"]
        codes[s] = dev.fetch(sides[s]["ptr"], rows * k, np.uint8).reshape(rows, k)
        r, l = rng.integers(0, rows, 2048), rng.integers(0, k, 2048)
        bad += int((codes[s][r, l] != form_twin(sides[s]["C"], sides[s]["E"], sides[s]["F"], r, l)).sum())
        res["salt_dead"][s] = dead_stats(dev, sides[s], rows, k, rng)
        sides[s]["C"] = None
        dev.free(sides[s]["cptr"])
    res["form_gate"] = {"elements": 4096, "mismatches": bad}
    dead_bad = sum(res["salt_dead"][s]["gate_mismatches"] for s in "AB")
    t_form = time.time()
    flags, out = dev.chain(sides["A"]["ptr"], sides["B"]["ptr"], m, n, k)
    t_chain = time.time()
    ii, jj = rng.integers(0, m, 512), rng.integers(0, n, 512)
    got, corner, total = card_flags(dev, flags, m, n, T, ii, jj, 64)
    want, nolost, fin = chain_twin(codes["A"][ii], codes["B"][jj])
    words = dev.fetch(out, m * n, np.uint32)
    res["chain_gate"] = {"words": 512, "flag_mismatches": int((got != want).sum()),
                         "word_mismatches": int((~same_words(words[ii * n + jj], fin)).sum()),
                         "sampled_steps_no_bit_lost": round(float(nolost.mean()), 6)}
    del words
    res["ok"] = (bad == 0 and dead_bad == 0 and res["chain_gate"]["flag_mismatches"] == 0 and
                 res["chain_gate"]["word_mismatches"] == 0)
    res["steps_exact"] = round(total / (m * n * T), 6)
    res["tile64"] = {k_: v for k_, v in reference_metrics(corner.tolist(), 64, T).items() if k_ not in ("rects", "dens")}
    if not res["ok"]:
        emit(fh, {"gate_failed": res})
        return res
    dev.free(out)
    nt = 15
    M = dev.alloc(4 * nt * m * (n // 32))
    wins, densest, strong = [], [], []
    colA, colB = pack_rows(okA[None, :])[0], pack_rows(okB[None, :])[0]
    wgot = np.lib.stride_tricks.sliding_window_view(got, WIN, axis=1).all(2)
    win_bad = 0
    for t0 in range(0, T - WIN + 1, nt):
        cnt = min(nt, T - WIN + 1 - t0)
        rb = dev.windows(flags, M, m, n, T, t0, cnt, 0)
        cb = dev.windows(flags, M, m, n, T, t0, cnt, 1)
        want_w = wgot[:, t0:t0 + cnt].T
        win_bad += int((((rb[:, ii, jj >> 5] >> (jj & 31)) & 1).astype(bool) != want_w).sum())
        win_bad += int((((cb[:, jj, ii >> 5] >> (ii & 31)) & 1).astype(bool) != want_w).sum())
        dens = popc(rb).sum((1, 2)) / (m * n)
        if mask:
            rb[:, ~okA] = 0
            rb &= colB
            cb[:, ~okB] = 0
            cb &= colA
        for q in range(cnt):
            g_r, g_c = greedy(rb[q]), greedy(cb[q])
            d = float(dens[q])
            wins.append({"t": t0 + q, "density": round(d, 6), "frags16x8": fragments_exact(rb[q]), "by_rows": g_r, "by_cols": g_c})
            if (t0 + q) % WIN == 0:
                strong.append({"t": t0 + q, "density": round(d, 6), "rows_x_cols": list(intersect_greedy(rb[q])),
                               "cols_x_rows": list(intersect_greedy(cb[q]))})
            elif not densest or d > densest[-1][0] or len(densest) < 2:
                densest = sorted([*densest, (d, t0 + q, rb[q].copy(), cb[q].copy())], key=lambda x: -x[0])[:2]
        del rb, cb
    for d, t, rbits, cbits in densest:
        r_, c_ = intersect_greedy(rbits), intersect_greedy(cbits)
        strong.append({"t": t, "density": round(d, 6), "rows_x_cols": list(r_), "cols_x_rows": list(c_)})
    del densest
    dev.free(M)
    for p in (flags, sides["A"]["ptr"], sides["B"]["ptr"]):
        dev.free(p)
    res["window_gate"] = {"words": 512, "windows": T - WIN + 1, "orientations": 2, "mismatches": win_bad}
    res["ok"] = win_bad == 0
    if not res["ok"]:
        emit(fh, {"gate_failed": {"item": item, "window_gate": res["window_gate"]}})
        return res
    t_win = time.time()

    def area(x):
        return x[0] * x[1]

    best_r = max(wins, key=lambda w: area(w["by_rows"]["grown"]))
    best_c = max(wins, key=lambda w: area(w["by_cols"]["grown"]))
    lemma = any(w["by_rows"][f"rows_with_cols_ge_{REGION[1]}"] >= REGION[0] or
                w["by_cols"][f"rows_with_cols_ge_{REGION[0]}"] >= REGION[1] or
                (w["by_rows"]["grown"][0] >= REGION[0] and w["by_rows"]["grown"][1] >= REGION[1]) or
                (w["by_cols"]["grown"][0] >= REGION[1] and w["by_cols"]["grown"][1] >= REGION[0]) for w in wins)
    res["windows"] = {
        "count": len(wins),
        "density_min": min(w["density"] for w in wins), "density_max": max(w["density"] for w in wins),
        "aligned_density": [w["density"] for w in wins if w["t"] % 32 == 0],
        "frags16x8_max": max(w["frags16x8"] for w in wins), "frags16x8_total": sum(w["frags16x8"] for w in wins),
        "frags16x8_per_window": (m // 16) * (n // 8),
        "best_by_rows": {"t": best_r["t"], "rows_x_cols": best_r["by_rows"]["grown"], "plain": best_r["by_rows"]["plain"]},
        "best_by_cols": {"t": best_c["t"], "cols_x_rows": best_c["by_cols"]["grown"], "plain": best_c["by_cols"]["plain"]},
        "max_cols_at_rows": {r: max(w["by_rows"]["cols_at_rows"].get(r, 0) for w in wins) for r in ("16", "64", "256", "512", "1024")},
        "max_rows_at_cols": {r: max(w["by_cols"]["cols_at_rows"].get(r, 0) for w in wins) for r in ("16", "64", "256", "512", "1024")},
        "intersect_greedy": strong,
        "lemma_block_found": lemma or any(s["rows_x_cols"][0] >= REGION[0] and s["rows_x_cols"][1] >= REGION[1] or
                                          s["cols_x_rows"][0] >= REGION[1] and s["cols_x_rows"][1] >= REGION[0] for s in strong),
    }
    res["per_window"] = wins
    res["merge"] = merge_stats(codes["A"], codes["B"], seed)
    res["seconds"] = {"form": round(t_form - t_start, 1), "chain": round(t_chain - t_form, 1),
                      "windows": round(t_win - t_chain, 1), "total": round(time.time() - t_start, 1)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="lease")
    ap.add_argument("--expect-uuid", required=True)
    ap.add_argument("--clock-label", default="unlocked")
    ap.add_argument("--families", default=",".join(UNIT_FAMILIES))
    ap.add_argument("--runs", default="1", help="the census draws per family: run 1 is unit:<family>, run r unit:<family>@r")
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--budget-min", type=float, default=5.5)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--nvcc", default=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
    ap.add_argument("--out", default=None)
    ap.add_argument("--acts", default=None, help="a pearlc_capture directory: each captured linear is a unit")
    ap.add_argument("--acts-layers", default=".", help="regex over the capture's linear names")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / "log.jsonl", "a")
    census.set_atom(ATOM)
    items = ["refcheck", *(f"unit:{f}" if r == "1" else f"unit:{f}@{r}"
                           for r in args.runs.split(",") for f in args.families.split(",") if f)]
    if args.acts:
        names = json.loads((Path(args.acts) / "manifest.json").read_text())["inputs"]
        items += [f"acts:{x}" for x in names if re.search(args.acts_layers, x)]
    state_path = out_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": [], "starts": {}}
    todo = [x for x in items if x not in state["done"]]
    if not todo:
        write_summary(out_dir, items, fh)
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    index = visible if args.gpu == "lease" and visible.isdigit() else args.gpu
    if visible != index or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        emit(fh, {"refused": f"need CUDA_VISIBLE_DEVICES={args.gpu} (one index) and CUDA_DEVICE_ORDER=PCI_BUS_ID"})
        sys.exit(3)
    expect = args.expect_uuid
    if expect == "lease":
        expect = subprocess.run(["nvidia-smi", "-i", index, "--query-gpu=uuid", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.strip()
    dev = Dev()
    ident = {"name": dev.name, "uuid": dev.uuid, "cc": list(dev.cc), "sms": dev.sms, "cuda_driver_api": dev.driver,
             "visible_devices": dev.count, "pci_index": index, "expect_uuid": expect}
    if dev.count != 1 or dev.uuid != expect or dev.cc != (12, 0) or "RTX PRO 6000" not in dev.name:
        emit(fh, {"refused": "identity", **ident})
        sys.exit(3)
    nvml = Nvml(dev.uuid)
    emit(fh, {"identity": {**ident, "nvml_driver": nvml.driver, "numpy": np.__version__, "atom": ATOM,
                           "clock_label": args.clock_label, "clocks": nvml.read(), "start": time.strftime("%FT%TZ", time.gmtime())}})
    cubin, rec = build(out_dir, args.nvcc)
    emit(fh, {"build": rec})
    if rec["problems"]:
        sys.exit(5)
    dev.load(cubin)

    unit_secs: list[float] = []
    for item in todo:
        est = max(unit_secs) if unit_secs else 90.0
        if time.time() - t0 + est > 60 * args.budget_min:
            emit(fh, {"chunk_end": item, "remaining": len([x for x in items if x not in state["done"]]),
                      "seconds": round(time.time() - t0, 1)})
            sys.exit(99)
        state["starts"][item] = state["starts"].get(item, 0) + 1
        state_path.write_text(json.dumps(state))
        if state["starts"][item] > 3:
            emit(fh, {"failed": item, "reason": "started 3 times without finishing"})
            sys.exit(1)
        ts = time.time()
        if item == "refcheck":
            res = refcheck(dev, fh)
        elif item.startswith("acts:"):
            res = acts_unit(dev, fh, Path(args.acts), item.split(":", 1)[1], args.seed)
        else:
            res = unit(dev, fh, item, args.k, args.rows, args.seed)
        res.update({"identity": ident, "build_cubin_sha256": rec["cubin_sha256"], "clocks": nvml.read(),
                    "clock_label": args.clock_label})
        if not res["ok"]:
            (out_dir / f"{item.replace(':', '-')}.failed.json").write_text(json.dumps(res, indent=1) + "\n")
            sys.exit(4)
        (out_dir / f"{item.replace(':', '-')}.json").write_text(json.dumps(res, indent=1) + "\n")
        brief = {x: res[x] for x in ("item", "steps_exact", "rules_ok") if x in res}
        if "windows" in res:
            brief.update({x: res["windows"][x] for x in ("density_min", "density_max", "frags16x8_max", "best_by_rows",
                                                          "best_by_cols", "intersect_greedy", "lemma_block_found")})
            brief["merge"] = {x: res["merge"][x] for x in ("pair_A", "rand_A", "frag_both_A", "frag_both_B",
                                                            "preadd_frag_hits", "preadd_frags")}
            brief["salt_dead_A"] = {n: res["salt_dead"]["A"][n]["share"] for n in DEAD_SCREENS}
            brief["seconds"] = res["seconds"]
        emit(fh, {"done": brief})
        state["done"].append(item)
        state_path.write_text(json.dumps(state))
        if item != "refcheck":
            unit_secs.append(time.time() - ts)
    write_summary(out_dir, items, fh)


def write_summary(out_dir: Path, items: list[str], fh) -> None:
    rows = []
    for item in items:
        if item == "refcheck":
            continue
        r = json.loads((out_dir / f"{item.replace(':', '-')}.json").read_text())
        w, mg = r["windows"], r["merge"]
        rows.append({"item": item, "family": r["family"], "run": r.get("run"), "rules_ok": r["rules_ok"],
                     "rows_admitted": r["rows_admitted"], "steps_exact": r["steps_exact"],
                     "density": [w["density_min"], w["density_max"]], "frags16x8_max": w["frags16x8_max"],
                     "best_by_rows": w["best_by_rows"]["rows_x_cols"], "best_by_cols": w["best_by_cols"]["cols_x_rows"],
                     "intersect_best": max([x for s in w["intersect_greedy"] for x in (s["rows_x_cols"], s["cols_x_rows"][::-1])],
                                           key=lambda x: x[0] * x[1], default=[0, 0]),
                     "lemma_block_found": w["lemma_block_found"], "tile64": r["tile64"]["row"],
                     "merge_pair_A": mg["pair_A"], "merge_frag_hits": mg["preadd_frag_hits"], "merge_frags": mg["preadd_frags"],
                     "merge_hits_by_preadd": {n: v["frag_hits"] for n, v in mg["preadds"].items() if v["frag_hits"]},
                     "strassen_frag_pairs": mg["strassen_frag_pairs"], "pair_blocks": mg["pair_blocks"],
                     "salt_dead": {s: {n: [v["share"], v["share_admitted"], v["worst_row_admitted"]]
                                       for n, v in r["salt_dead"][s].items() if n != "gate_mismatches"} for s in "AB"}})
    doc = {"tool": "pouw_sm120_exact", "rows": rows,
           "refcheck": json.loads((out_dir / "refcheck.json").read_text())["tiles"]}
    path = out_dir / "summary.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    emit(fh, {"summary": rows})
    dom = [r for r in rows if r["rules_ok"] or r["item"].startswith("acts:")]
    hits = sum(r["merge_frag_hits"] for r in dom)
    frags = sum(r["merge_frags"] for r in dom)
    sp = [sum(r["strassen_frag_pairs"][i] for r in dom) for i in (0, 1)]
    found = sum(r["lemma_block_found"] for r in dom)
    biggest = max((x[0] * x[1] for r in dom for x in (r["best_by_rows"], r["best_by_cols"], r["intersect_best"])), default=0)
    run_result.write(path, True, "sm120-exact",
                     f"{len(rows)} units, {len(dom)} admitted; there: lemma block found in {found}, largest block {biggest} "
                     f"words, pre-add fragments exact {hits}/{frags}, Strassen fragment pairs {sp[0]}/{sp[1]}",
                     [("units", len(rows), "count"), ("units_admitted", len(dom), "count"), ("lemma_block_units", found, "count"),
                      ("largest_block_words", biggest, "words"), ("merge_fragment_hits", hits, "count"),
                      ("merge_fragments", frags, "count"), ("strassen_frag_pair_hits", sp[0], "count"),
                      ("strassen_frag_pairs", sp[1], "count")])


if __name__ == "__main__":
    main()
