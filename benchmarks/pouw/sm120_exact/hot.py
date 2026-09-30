#!/usr/bin/env python3
"""v2-hot's full-unit rerun, the GPU half (one RTX PRO 6000, sm_120): where the unpromoted chain is exact from v2-hot's salted
start H_i, beside the +0 chain, on whole 8,192² units (internal/pouw/rtx-pro/fill-candidate-aligned-exact-regions.md,
"Update, 12:30Z"; v2-hot-start-rule.md; server.md 11:50Z item 4).  `hot_search.py` is the CPU half, the window search over
what this writes.

A step of word (i, j) at atom t is exact when the atom's FP32 output equals its accumulator plus the exact sum of its 32
products (pearlc_hot_start.chain's test).  Every word of row i starts at H_i, built by pearlc_hot_start.hot_words with the
public-constant rule (`const`, c0 = 64); the +0 chain starts every word at +0.

1. Identity and build as run.py (one visible RTX PRO 6000 at cc 12.0 whose UUID is --expect-uuid, exit 3; exact.cu for
   sm_120a, no .FTZ FADD/FFMA/FMUL in its SASS, exit 5).
2. `selftest`, the first item (exit 4 on any mismatch), at k = 1,024 on 128 x 128 units:
   - near-cap: this job's unit equals pearlc_hot_start.units("near-cap", 1024, 128, 128) (both sides' codes, α and ρ
     words) and its H_i equal hot_words(..., "const") on that reference's rows;
   - near-cap, rank1 and aligned-spikes-r64: the card's forming equals pearl_c.form_v1 on SM120_UNPROMOTED on every row of
     both sides; the card's per-(word, atom) flags and final words, from H_i and from +0, equal pearlc_hot_start.chain on
     every word (near-cap) or the first 64 x 64 (the others); both bitset layouts equal the flags.
3. `unit:<family>@<u>` for each family (--families: the census's ACTIVATION_FAMILIES and FREEZE_FAMILIES, and near-cap) and
   u = 1 .. --units: A and B, 8,192 rows of k = 8,192 each, formed on the card by pearl_c.form_v1's rule (the host computes
   s5_stats, E' and C = α·x with the scheme's own functions; the card runs the atom and Pearl's cast).  Rows: near-cap as
   pearlc_hot_start.units draws them (columns first, then rows; unit 1 at its salt "near-cap/8192/compat-wide", unit u at
   ".../u"); a census family from census.family_rows(fam, k, 1.0, 8192, 8192, u) at salt "<fam>/8192/hot-full/<u>";
   `cancel-pair@t4` (opt-in) as the assessor's hot_late_start.py draws it, from a generator seeded by that salt's SHA-256.
   Then both chains, and per chain each atom's exactness as row bitsets and column bitsets:
   `<out>/units/<fam>@<u>/{hot,zero}.{rows,cols}.u32` (T x 8,192 x 256 uint32 each, 2 GiB), `okA.npy`/`okB.npy` (the rows
   and columns Π admits: noise floor and liveness, pearl_c.PearlC.admissible's per-row rule; `unit.json`'s `rows_live`
   counts liveness alone, pearl_c.live_row), `H.npy`.  Gated per unit:
   2 rows per side and 4,096 elements through form_v1 / the device atom, 64 words over all 256 atoms of both chains through
   pearlc_hot_start.chain (both layouts), and H_i against v2-hot-start-rule.md's closed form (e_i = E(t_i) + 10 +
   [M(t_i) >= 0x3504F4], t_i = fl32(α_i·ρ_i)); `unit.json`, written last, has the per-atom exact counts of both chains.
4. Restartable: one directory per unit, a state file, and exit 99 once --budget-min is spent while units remain; exit 0
   after summary.json.  A unit started 3 times without finishing fails the job (exit 1).

--mode v1hot is v1-hot's full-unit run (ttout-restatements.md §9; the fill candidate's "Update, 14:40Z"): the same units,
and per v1 group g (atoms 4g .. 4g + 3, 128 deep) the chain restarted from H_{i,g} (chain `v1hot`, pearlc_v1_hot's
hot_group_words, publicConst 64) or from +0 (chain `v1`, v1's groups).  A plane is then a group, exact when all 4 of its
steps are: `{v1hot,v1}.{rows,cols}.u32` are k / 128 x 8,192 x 256 uint32 (512 MiB), and `Hg.npy` holds H_{i,g}.  Gated as
above per group (pearlc_hot_start.chain over each group from its start), plus H_{i,0} = H_i and every group's exponent
against the closed form; the self-test also checks H_{i,g} against hot_group_words on the reference rows.

  gpu-lease 1 --wait -- env CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=<04734fb5>/packages/verity/src:<04734fb5>/protocols/pouw \
      python3 benchmarks/pouw/sm120_exact/hot.py --gpu lease --expect-uuid lease --ref <pearlc_hot_start.py's dir> --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))
import pearlc_census as census  # noqa: E402
import run_result  # noqa: E402
from run import Dev, build, popc  # noqa: E402
from sm120_chain.run import Nvml, emit  # noqa: E402

HS = HV = C = D = P = None  # pearlc_hot_start, pearlc_v1_hot and the scheme's modules, imported from --ref and PYTHONPATH
FAMILIES = (*census.ACTIVATION_FAMILIES, *census.FREEZE_FAMILIES, "near-cap")
CANCEL_FAMILIES = ("cancel-pair@t4",)  # opt-in (--families), not in the default set
CHAINS = ("hot", "zero")
MODE_CHAINS = {"v2hot": CHAINS, "v1hot": ("v1hot", "v1")}
GROUP = 128                # v1's group depth: G = 4 atoms of 32
SELFTEST = (("near-cap", 128), ("rank1", 64), ("aligned-spikes-r64", 64))
REF_FILES = ("pearlc_hot_start.py", "pearlc_exact_runs.py", "pearlc_promotion_fusion.py", "pearlc_row_compat.py")


def jdump(x, **kw) -> str:
    """json.dumps that writes numpy scalars as the Python numbers they hold."""
    def scalar(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"{type(o).__name__} is not JSON serializable")
    return json.dumps(x, default=scalar, **kw)


def load_ref(ref: str, mode: str = "v2hot") -> dict:
    global HS, HV, C, D, P
    sys.path.insert(0, ref)
    import pearlc_hot_start as hs
    from verity_pouw.schemes import pearl_c, pearl_c_device, pearl_kw
    HS, C, D, P = hs, pearl_c, pearl_c_device, pearl_kw
    if not hasattr(C, "form_v1") or not hasattr(D, "SM120_UNPROMOTED"):
        raise SystemExit(f"{C.__file__}: no form_v1 or SM120_UNPROMOTED; PYTHONPATH must name the scheme's tree (04734fb5)")
    files = {m.__name__: m.__file__ for m in (pearl_c, pearl_c_device, pearl_kw)}
    files.update({f: str(Path(ref) / f) for f in REF_FILES})
    if mode == "v1hot":
        import pearlc_v1_hot as hv
        HV = hv
        files["pearlc_v1_hot.py"] = str(Path(ref) / "pearlc_v1_hot.py")
    return {name: {"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for name, p in files.items()}


# ---- a unit's host side --------------------------------------------------------------------------------------------------

def words(X: np.ndarray) -> np.ndarray:
    W = np.ascontiguousarray(X.astype(np.float32)).view(np.uint32)
    if ((W >> 23) & 0xFF == 0xFF).any():
        raise SystemExit("a family row is not finite in FP32")
    return W


def cancel_pair_t4(rng: np.random.Generator, rows: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """internal/pouw/red-team/hot_late_start.py's `cancel-pair@t4`: B's columns Student-t(4) with B[2l+1] = B[2l]; A's rows
    A[2l] = v, A[2l+1] = -v, v's first half random signs and the rest N(0, 0.1²); B drawn first."""
    B = rng.standard_t(4, (rows, k))
    v = rng.standard_normal((rows, k // 2)) * 0.1
    v[:, : k // 4] = np.sign(rng.standard_normal((rows, k // 4)))
    A = np.empty((rows, k))
    A[:, 0::2], A[:, 1::2] = v, -v
    B[:, 1::2] = B[:, 0::2]
    return A, B


def draw(fam: str, k: int, rows: int, u: int) -> tuple[np.ndarray, np.ndarray, str]:
    """X_A and X_B (rows x k FP32 words) and the unit's salt."""
    if fam in CANCEL_FAMILIES:
        salt = f"{fam}/{k}/hot-full/{u}"
        XA, XB = cancel_pair_t4(np.random.default_rng(list(hashlib.sha256(salt.encode()).digest())), rows, k)
        return words(XA), words(XB), salt
    if fam == "near-cap":
        salt = f"near-cap/{k}/compat-wide" + ("" if u == 1 else f"/{u}")
        rng = random.Random(salt)
        pos = sorted(rng.sample([l for l in range(k) if l % 8], k // 64))
        xb = np.array([[HS.f32(v) for v in HS.fam_row(fam, k, rng, pos)] for _ in range(rows)], np.uint32)
        xa = np.array([[HS.f32(v) for v in HS.fam_row(fam, k, rng, pos)] for _ in range(rows)], np.uint32)
        return xa, xb, salt
    XA, XB, _, _ = census.family_rows(fam, k, 1.0, rows, rows, u)
    return words(XA), words(XB), f"{fam}/{k}/hot-full/{u}"


def _row_stats(task):
    """Per row: s5_stats, the E line and E', and Π's admission (pearl_c.PearlC.admissible's per-row rule)."""
    seed, side, i0, X = task
    out = []
    for q, x in enumerate(X.tolist()):
        s, rho, alpha, beta = C.s5_stats(x)
        e = P.sample_line(seed, side, 0, i0 + q)
        live = bool(C.live_row(x))
        ok = P.f32_of_bits(rho) * P.f32_of_bits(alpha) >= P.SIGMA_MIN and live
        out.append((s, rho, alpha, beta, C.eprime(beta, e), e, ok, live))
    return out


def _draw_task(args):
    return draw(*args)


def _basis_task(args):
    seed_b, side, k = args
    return np.array(C._basis(seed_b, side, k, C.LINE_NORM_V1), np.uint8)


def prep(pool: Pool, fam: str, k: int, rows: int, u: int, mode: str = "v2hot") -> dict:
    """The unit's host side: both sides' FP32 rows, s5 words, E', C = α·x, F, admission, H_i, and in v1hot mode H_{i,g}
    (rows x k / 128)."""
    xa, xb, salt = pool.apply(_draw_task, ((fam, k, rows, u),))
    seed_b = C._labelled(salt.encode() + b"B", "seed-B")
    seed_a = C._labelled(salt.encode() + b"A", "seed-A")
    basis = pool.map_async(_basis_task, [(seed_b, 0, k), (seed_b, 1, k)])
    step = max(1, rows // 64)
    sides = {}
    for name, X, seed, side in (("A", xa, seed_a, 0), ("B", xb, seed_b, 1)):
        st = [r for part in pool.map(_row_stats, [(seed, side, i, X[i:i + step]) for i in range(0, rows, step)]) for r in part]
        alpha = np.array([r[2] for r in st], np.uint32)
        Cf = X.view(np.float32) * alpha.view(np.float32)[:, None]
        if (~np.isfinite(Cf)).any():
            raise SystemExit("α·x is not finite")
        sides[name] = {"x": X, "C": Cf, "s": np.array([r[0] for r in st], np.uint32),
                       "rho": np.array([r[1] for r in st], np.uint32), "alpha": alpha,
                       "beta": np.array([r[3] for r in st], np.uint32), "ep": np.array([r[4] for r in st], np.uint8),
                       "e": np.array([r[5] for r in st], np.uint8), "ok": np.array([r[6] for r in st], bool),
                       "live": np.array([r[7] for r in st], bool)}
    sides["A"]["F"], sides["B"]["F"] = basis.get()
    a = sides["A"]
    # Below the noise floor (α·ρ < 1, e.g. ρ = 0) a row is outside the domain and H_i has no size; such rows are left out of
    # the search, so they start at H_i sized as if α·ρ = 1.
    floor = (a["alpha"].view(np.float32).astype(np.float64) * a["rho"].view(np.float32) >= P.SIGMA_MIN)
    one = np.uint32(P.f32_bits(1.0))
    sized = SimpleNamespace(codes=[None] * rows, alpha=np.where(floor, a["alpha"], one).tolist(),
                            rho=np.where(floor, a["rho"], one).tolist())
    H, c = HS.hot_words(seed_a, sized, None, "const")
    out = {"fam": fam, "u": u, "k": k, "rows": rows, "salt": salt, "seed_a": seed_a, "seed_b": seed_b, "sides": sides,
           "H": np.array(H, np.uint32), "c0": c, "below_floor": ~floor}
    if mode == "v1hot":
        out["Hg"] = np.array(HV.hot_group_words(seed_a, sized, k // GROUP), np.uint32)
    return out


def closed_form_h(alpha: np.ndarray, rho: np.ndarray, H: np.ndarray, below_floor: np.ndarray) -> int:
    """Rows in the domain where H_i's exponent differs from v2-hot-start-rule.md's e_i = E(t_i) + 10 + [M(t_i) >= 0x3504F4],
    with t_i = fl32(α_i·ρ_i)."""
    t = (alpha.view(np.float32) * rho.view(np.float32)).view(np.uint32)
    e = ((t >> 23) & 0xFF).astype(np.int64) - 127 + 10 + ((t & 0x7FFFFF) >= 0x3504F4)
    return int(((((H >> 23) & 0xFF).astype(np.int64) - 127 != e) & ~below_floor).sum())


def hg_gates(u: dict) -> dict:
    """H_{i,g} against v2-hot: group 0's word is H_i (block 0 of the same keyed line hash, the same exponent), and every
    group's exponent is the closed form's."""
    A, Hg = u["sides"]["A"], u["Hg"]
    return {"Hg0_vs_H_mismatches": int((Hg[:, 0] != u["H"]).sum()),
            "Hg_closed_form_mismatches": sum(closed_form_h(A["alpha"], A["rho"], Hg[:, g], u["below_floor"])
                                             for g in range(Hg.shape[1]))}


# ---- the card ------------------------------------------------------------------------------------------------------------

def form_side(dev: Dev, side: dict) -> int:
    ptr, cptr = dev.form(side["C"], side["ep"], side["F"], pearl=True)
    dev.free(cptr)
    return ptr


def chain_bits(dev: Dev, da: int, db: int, m: int, n: int, k: int, h: int, sink=None,
               grouped: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """One chain; its planes' row and column bitsets go to sink(layout, p0, (nt, rows, cols / 32) uint32), 32 planes at a
    time.  A plane is an atom, or with grouped a v1 group (restarting from h's H_{i,g}, or +0), exact on all 4 of its
    atoms.  Returns the final words and each plane's exact count."""
    w = GROUP // 32 if grouped else 1
    flags, out = dev.chain(da, db, m, n, k, h, k // GROUP if grouped else 0)
    T = k // 32 // w
    M = dev.alloc(4 * 32 * m * (n // 32))
    counts = np.zeros(T, np.int64)
    for p0 in range(0, T, 32):
        nt = min(32, T - p0)
        for tr, layout in ((0, "rows"), (1, "cols")):
            dev.atom_bits(flags, M, m, n, p0 * w, nt, tr, w)
            rows_, cols_ = (n, m) if tr else (m, n)
            planes = dev.fetch(M, nt * rows_ * (cols_ // 32), np.uint32).reshape(nt, rows_, cols_ // 32)
            if not tr:
                counts[p0:p0 + nt] = popc(planes).sum((1, 2))
            if sink:
                sink(layout, p0, planes)
            del planes
    fin = dev.fetch(out, m * n, np.uint32).reshape(m, n)
    dev.free(M)
    dev.free(flags)
    dev.free(out)
    return fin, counts


def bit(planes_rows: np.ndarray, t: int, i: int, j: int) -> bool:
    return bool((planes_rows[t, i, j >> 5] >> (j & 31)) & 1)


def ref_chain(a: list, b: list, h: int, T: int) -> tuple[list, int]:
    return HS.chain(a, b, D.SM120_UNPROMOTED, h, T)


def ref_planes(mode: str, chain: str, a: list, b: list, u: dict, i: int, T: int) -> tuple[list, int]:
    """pearlc_hot_start.chain's per-plane exactness for word (i, ·) and its final word: per atom from H_i or +0 (v2hot);
    per v1 group, all 4 steps exact from H_{i,g} or +0, and the last group's word (v1hot)."""
    if mode == "v2hot":
        return ref_chain(a, b, int(u["H"][i]) if chain == "hot" else 0, T)
    got, fin = [], 0
    for g in range(len(a) // GROUP):
        lo = GROUP * g
        fl, fin = ref_chain(a[lo:lo + GROUP], b[lo:lo + GROUP], int(u["Hg"][i, g]) if chain == "v1hot" else 0, GROUP // 32)
        got.append(all(fl))
    return got, fin


def h_ptr(dev: Dev, chain: str, u: dict) -> int:
    if chain == "hot":
        return dev.put(u["H"])
    if chain == "v1hot":
        return dev.put(np.ascontiguousarray(u["Hg"]))
    return 0


# ---- items ---------------------------------------------------------------------------------------------------------------

def selftest(dev: Dev, pool: Pool, fh, mode: str) -> dict:
    k, rows = 1024, 128
    grouped = mode == "v1hot"
    T = k // GROUP if grouped else k // 32
    res = {"item": "selftest", "mode": mode, "gates": []}
    ok = True
    for fam, words_side in SELFTEST:
        u = prep(pool, fam, k, rows, 1, mode)
        g = {"gate": f"selftest/{fam}/k{k}", "rows": rows}
        A, B = u["sides"]["A"], u["sides"]["B"]
        fa, fb = form_host(A, rows, k), form_host(B, rows, k)
        if fam == "near-cap":
            FA, FB, _live, _dev, seed_a = HS.units(fam, k, rows, rows)
            g["unit_vs_hot_start_units"] = int(
                (np.array(FA.codes, np.uint8) != fa).sum() + (np.array(FB.codes, np.uint8) != fb).sum() +
                (np.array(FA.alpha, np.uint32) != A["alpha"]).sum() + (np.array(FA.rho, np.uint32) != A["rho"]).sum() +
                (np.array(FB.alpha, np.uint32) != B["alpha"]).sum() + (seed_a != u["seed_a"]))
            Hr, _ = HS.hot_words(seed_a, FA, FB, "const")
            g["H_vs_hot_words"] = int((np.array(Hr, np.uint32) != u["H"]).sum())
            if grouped:
                Hgr = HV.hot_group_words(seed_a, FA, k // GROUP)
                g["Hg_vs_hot_group_words"] = int((np.array(Hgr, np.uint32) != u["Hg"]).sum())
        da, db = form_side(dev, A), form_side(dev, B)
        ca = dev.fetch(da, rows * k, np.uint8).reshape(rows, k)
        cb = dev.fetch(db, rows * k, np.uint8).reshape(rows, k)
        g["form_vs_form_v1"] = int((ca != fa).sum() + (cb != fb).sum())
        g["H_closed_form_mismatches"] = closed_form_h(A["alpha"], A["rho"], u["H"], u["below_floor"])
        if grouped:
            g.update(hg_gates(u))
        for chain in MODE_CHAINS[mode]:
            got = {}
            hp = h_ptr(dev, chain, u)
            fin, counts = chain_bits(dev, da, db, rows, rows, k, hp,
                                     lambda layout, t0, planes: got.__setitem__(layout, planes.copy()), grouped)
            if hp:
                dev.free(hp)
            bad_f = bad_w = bad_t = 0
            for i in range(words_side):
                for j in range(words_side):
                    want, wfin = ref_planes(mode, chain, ca[i].tolist(), cb[j].tolist(), u, i, T)
                    for t in range(T):
                        bad_f += bit(got["rows"], t, i, j) != want[t]
                        bad_t += bit(got["cols"], t, j, i) != want[t]
                    bad_w += not same_word(int(fin[i, j]), wfin)
            g[f"{chain}_flag_mismatches"], g[f"{chain}_cols_mismatches"], g[f"{chain}_word_mismatches"] = bad_f, bad_t, bad_w
            g[f"{chain}_words_checked"] = words_side * words_side
            g[f"{chain}_planes_exact"] = round(float(counts.sum() / (rows * rows * T)), 6)
        dev.free(da)
        dev.free(db)
        g["ok"] = all(v == 0 for x, v in g.items() if x.endswith(("mismatches", "_vs_hot_start_units", "_vs_hot_words",
                                                                   "_vs_hot_group_words", "form_vs_form_v1")))
        ok &= g["ok"]
        emit(fh, g)
        res["gates"].append(g)
    res["ok"] = ok
    return res


def form_host(side: dict, rows: int, k: int) -> np.ndarray:
    """pearl_c.form_v1 on SM120_UNPROMOTED for the side's rows (all of them)."""
    f = side["F"].tolist()
    out = C.form_v1(side["x"][:rows].tolist(), side["e"][:rows].tolist(), f, D.SM120_UNPROMOTED)
    return np.array(out.codes, np.uint8)


def same_word(x: int, y: int) -> bool:
    return x == y or (x & 0x7FFFFFFF) == 0 == (y & 0x7FFFFFFF)


def unit(dev: Dev, pool: Pool, fh, out_dir: Path, u: dict, seed: int, mode: str) -> dict:
    fam, k, m = u["fam"], u["k"], u["rows"]
    grouped = mode == "v1hot"
    n, T = m, (k // GROUP if grouped else k // 32)
    name = f"{fam}@{u['u']}"
    t_start = time.time()
    rng = np.random.default_rng([seed, u["u"], *hashlib.sha256(fam.encode()).digest()[:4]])
    A, B = u["sides"]["A"], u["sides"]["B"]
    ud = out_dir / "units" / name
    if ud.exists():
        shutil.rmtree(ud)
    ud.mkdir(parents=True)
    np.save(ud / "okA.npy", A["ok"])
    np.save(ud / "okB.npy", B["ok"])
    np.save(ud / "H.npy", u["H"])
    if grouped:
        np.save(ud / "Hg.npy", u["Hg"])
    res = {"item": f"unit:{name}", "mode": mode, "family": fam, "unit": u["u"], "salt": u["salt"], "m": m, "n": n, "k": k,
           "planes": T, "atoms_per_plane": GROUP // 32 if grouped else 1,
           "forming": "pearl_c.form_v1 on SM120_UNPROMOTED (host: s5_stats, eprime, alpha*x; card: the atom, Pearl's cast)",
           "hot_rule": "const", "c0": u["c0"], "rows_admitted": {"A": int(A["ok"].sum()), "B": int(B["ok"].sum())},
           "rows_live": {"A": int(A["live"].sum()), "B": int(B["live"].sum())},
           "A_rows_below_floor_sized_at_1": int(u["below_floor"].sum()),
           "H_exponents": [int(x) for x in np.percentile(((u["H"] >> 23) & 0xFF).astype(int) - 127, [0, 50, 100])]}
    da, db = form_side(dev, A), form_side(dev, B)
    ca = dev.fetch(da, m * k, np.uint8).reshape(m, k)
    cb = dev.fetch(db, n * k, np.uint8).reshape(n, k)
    bad = 0
    for side, codes in ((A, ca), (B, cb)):
        for r in rng.integers(0, m, 2):
            one = {x: side[x][r:r + 1] for x in ("x", "e")} | {"F": side["F"]}
            bad += int((form_host(one, 1, k)[0] != codes[r]).sum())
        r, l = rng.integers(0, m, 4096), rng.integers(0, k, 4096)
        atom = D.SM120_UNPROMOTED.atom
        f = side["F"]
        for q in range(len(r)):
            w = C._check_finite(atom.step(int(side["C"][r[q], l[q]].view(np.uint32)), side["ep"][r[q]].tolist(), f[l[q]].tolist()))
            bad += int(P.f32_to_fp8(P.f32_of_bits(w)) != int(codes[r[q], l[q]]))
    res["form_gate"] = {"rows": 4, "elements": 8192, "mismatches": bad}
    res["H_closed_form_mismatches"] = closed_form_h(A["alpha"], A["rho"], u["H"], u["below_floor"])
    if grouped:
        res.update(hg_gates(u))
    t_form = time.time()
    ii, jj = rng.integers(0, m, 64), rng.integers(0, n, 64)
    res["chains"] = {}
    for chain in MODE_CHAINS[mode]:
        files = {lay: open(ud / f"{chain}.{lay}.u32", "wb") for lay in ("rows", "cols")}
        sample = {"rows": np.zeros((64, T), bool), "cols": np.zeros((64, T), bool)}

        def sink(layout, p0, planes):
            planes.tofile(files[layout])
            nt = planes.shape[0]
            if layout == "rows":
                sample["rows"][:, p0:p0 + nt] = ((planes[:, ii, jj >> 5] >> (jj & 31)) & 1).T.astype(bool)
            else:
                sample["cols"][:, p0:p0 + nt] = ((planes[:, jj, ii >> 5] >> (ii & 31)) & 1).T.astype(bool)

        hp = h_ptr(dev, chain, u)
        fin, counts = chain_bits(dev, da, db, m, n, k, hp, sink, grouped)
        if hp:
            dev.free(hp)
        for fobj in files.values():
            fobj.close()
        bf = bc = bw = 0
        for q in range(64):
            want, wfin = ref_planes(mode, chain, ca[ii[q]].tolist(), cb[jj[q]].tolist(), u, int(ii[q]), T)
            want = np.array(want, bool)
            bf += int((sample["rows"][q] != want).sum())
            bc += int((sample["cols"][q] != want).sum())
            bw += not same_word(int(fin[ii[q], jj[q]]), wfin)
        res["chains"][chain] = {"gate": {"words": 64, "planes": T, "flag_mismatches": bf, "cols_mismatches": bc,
                                         "word_mismatches": bw},
                                "exact_per_plane": counts.tolist(),
                                "planes_exact": round(float(counts.sum() / (m * n * T)), 6),
                                "bytes_per_layout": T * m * n // 8}
        if not grouped:
            res["chains"][chain]["exact_per_atom"] = res["chains"][chain]["exact_per_plane"]
            res["chains"][chain]["steps_exact"] = res["chains"][chain]["planes_exact"]
        del fin
    dev.free(da)
    dev.free(db)
    res["ok"] = (bad == 0 and all(v == 0 for x, v in res.items() if x.endswith("mismatches")) and
                 all(v == 0 for c in res["chains"].values() for x, v in c["gate"].items() if x.endswith("mismatches")))
    res["seconds"] = {"form_and_gate": round(t_form - t_start, 1), "total": round(time.time() - t_start, 1)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="lease")
    ap.add_argument("--expect-uuid", required=True)
    ap.add_argument("--clock-label", default="unlocked")
    ap.add_argument("--ref", required=True, help="the directory holding pearlc_hot_start.py and its imports")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--units", type=int, default=4)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--budget-min", type=float, default=5.5)
    ap.add_argument("--mode", choices=tuple(MODE_CHAINS), default="v2hot",
                    help="v2hot: per atom, from H_i and +0; v1hot: per v1 group, from H_{i,g} and +0")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--nvcc", default=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / "log.jsonl", "a")
    code = load_ref(args.ref, args.mode)
    items = ["selftest", *(f"unit:{f}@{u}" for u in range(1, args.units + 1) for f in args.families.split(",") if f)]
    state_path = out_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": [], "starts": {}}
    todo = [x for x in items if x not in state["done"]]
    if not todo:
        write_summary(out_dir, items, fh, args.mode)
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
    emit(fh, {"identity": {**ident, "nvml_driver": nvml.driver, "numpy": np.__version__, "code": code,
                           "clock_label": args.clock_label, "clocks": nvml.read(), "start": time.strftime("%FT%TZ", time.gmtime())}})
    cubin, rec = build(out_dir, args.nvcc)
    emit(fh, {"build": rec})
    if rec["problems"]:
        sys.exit(5)
    dev.load(cubin)

    pool = Pool(args.procs, initializer=load_ref, initargs=(args.ref, args.mode))
    ahead = ThreadPoolExecutor(1)
    unit_secs: list[float] = []

    def prep_of(item):
        fam, _, u = item.split(":", 1)[1].partition("@")
        return ahead.submit(prep, pool, fam, args.k, args.rows, int(u), args.mode)

    def fits(units: int) -> bool:
        return time.time() - t0 + units * (max(unit_secs) if unit_secs else 150.0) <= 60 * args.budget_min

    nxt = None
    for q, item in enumerate(todo):
        if not fits(1):
            if nxt:
                nxt.result()
            emit(fh, {"chunk_end": item, "remaining": len([x for x in items if x not in state["done"]]),
                      "seconds": round(time.time() - t0, 1)})
            pool.close()
            sys.exit(99)
        state["starts"][item] = state["starts"].get(item, 0) + 1
        state_path.write_text(json.dumps(state))
        if state["starts"][item] > 3:
            emit(fh, {"failed": item, "reason": "started 3 times without finishing"})
            sys.exit(1)
        ts = time.time()
        if item == "selftest":
            res = selftest(dev, pool, fh, args.mode)
        else:
            u = (nxt or prep_of(item)).result()
            nxt = prep_of(todo[q + 1]) if q + 1 < len(todo) and fits(2) else None
            res = unit(dev, pool, fh, out_dir, u, args.seed, args.mode)
            del u
        res.update({"identity": ident, "build_cubin_sha256": rec["cubin_sha256"], "code": code, "clocks": nvml.read(),
                    "clock_label": args.clock_label})
        stem = item.replace(":", "-")
        if not res["ok"]:
            (out_dir / f"{stem}.failed.json").write_text(jdump(res, indent=1) + "\n")
            emit(fh, {"gate_failed": {x: res[x] for x in res if x not in ("identity", "code")}})
            sys.exit(4)
        if item != "selftest":
            (out_dir / "units" / item.split(":", 1)[1] / "unit.json").write_text(jdump(res, indent=1) + "\n")
        (out_dir / f"{stem}.json").write_text(jdump(res, indent=1) + "\n")
        emit(fh, {"done": {x: res.get(x) for x in ("item", "rows_admitted", "H_exponents", "seconds")} |
                  {c: res["chains"][c]["planes_exact"] for c in res.get("chains", {})}})
        state["done"].append(item)
        state_path.write_text(json.dumps(state))
        if item != "selftest":
            unit_secs.append(time.time() - ts)
    pool.close()
    write_summary(out_dir, items, fh, args.mode)


def write_summary(out_dir: Path, items: list[str], fh, mode: str) -> None:
    rows = []
    for item in items[1:]:
        r = json.loads((out_dir / f"{item.replace(':', '-')}.json").read_text())
        rows.append({"item": item, "family": r["family"], "unit": r["unit"], "rows_admitted": r["rows_admitted"],
                     "rows_live": r.get("rows_live"), "H_exponents": r["H_exponents"],
                     **{f"planes_exact_{c}": r["chains"][c]["planes_exact"] for c in MODE_CHAINS[mode]}})
    doc = {"tool": "pouw_sm120_exact_hot", "mode": mode, "rows": rows,
           "selftest": json.loads((out_dir / "selftest.json").read_text())["gates"]}
    path = out_dir / "summary.json"
    path.write_text(jdump(doc, indent=1) + "\n")
    emit(fh, {"summary": len(rows)})
    run_result.write(path, True, "sm120-exact-hot", f"{len(rows)} units of both chains written for hot_search.py",
                     [("units", len(rows), "count")])


if __name__ == "__main__":
    main()
