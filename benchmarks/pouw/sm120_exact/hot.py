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
   ".../u"); a census family from census.family_rows(fam, k, 1.0, 8192, 8192, u) at salt "<fam>/8192/hot-full/<u>".  Then
   both chains, and per chain each atom's exactness as row bitsets and column bitsets:
   `<out>/units/<fam>@<u>/{hot,zero}.{rows,cols}.u32` (T x 8,192 x 256 uint32 each, 2 GiB), `okA.npy`/`okB.npy` (the rows
   and columns Π admits: noise floor and liveness, pearl_c.PearlC.admissible's per-row rule), `H.npy`.  Gated per unit:
   2 rows per side and 4,096 elements through form_v1 / the device atom, 64 words over all 256 atoms of both chains through
   pearlc_hot_start.chain (both layouts), and H_i against v2-hot-start-rule.md's closed form (e_i = E(t_i) + 10 +
   [M(t_i) >= 0x3504F4], t_i = fl32(α_i·ρ_i)); `unit.json`, written last, has the per-atom exact counts of both chains.
4. Restartable: one directory per unit, a state file, and exit 99 once --budget-min is spent while units remain; exit 0
   after summary.json.  A unit started 3 times without finishing fails the job (exit 1).

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

HS = C = D = P = None      # pearlc_hot_start and the scheme's modules, imported in main() from --ref and PYTHONPATH
FAMILIES = (*census.ACTIVATION_FAMILIES, *census.FREEZE_FAMILIES, "near-cap")
CHAINS = ("hot", "zero")
SELFTEST = (("near-cap", 128), ("rank1", 64), ("aligned-spikes-r64", 64))
REF_FILES = ("pearlc_hot_start.py", "pearlc_exact_runs.py", "pearlc_promotion_fusion.py", "pearlc_row_compat.py")


def load_ref(ref: str) -> dict:
    global HS, C, D, P
    sys.path.insert(0, ref)
    import pearlc_hot_start as hs
    from verity_pouw.schemes import pearl_c, pearl_c_device, pearl_kw
    HS, C, D, P = hs, pearl_c, pearl_c_device, pearl_kw
    if not hasattr(C, "form_v1") or not hasattr(D, "SM120_UNPROMOTED"):
        raise SystemExit(f"{C.__file__}: no form_v1 or SM120_UNPROMOTED; PYTHONPATH must name the scheme's tree (04734fb5)")
    files = {m.__name__: m.__file__ for m in (pearl_c, pearl_c_device, pearl_kw)}
    files.update({f: str(Path(ref) / f) for f in REF_FILES})
    return {name: {"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for name, p in files.items()}


# ---- a unit's host side --------------------------------------------------------------------------------------------------

def words(X: np.ndarray) -> np.ndarray:
    W = np.ascontiguousarray(X.astype(np.float32)).view(np.uint32)
    if ((W >> 23) & 0xFF == 0xFF).any():
        raise SystemExit("a family row is not finite in FP32")
    return W


def draw(fam: str, k: int, rows: int, u: int) -> tuple[np.ndarray, np.ndarray, str]:
    """X_A and X_B (rows x k FP32 words) and the unit's salt."""
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
        ok = P.f32_of_bits(rho) * P.f32_of_bits(alpha) >= P.SIGMA_MIN and C.live_row(x)
        out.append((s, rho, alpha, beta, C.eprime(beta, e), e, ok))
    return out


def _draw_task(args):
    return draw(*args)


def _basis_task(args):
    seed_b, side, k = args
    return np.array(C._basis(seed_b, side, k, C.LINE_NORM_V1), np.uint8)


def prep(pool: Pool, fam: str, k: int, rows: int, u: int) -> dict:
    """The unit's host side: both sides' FP32 rows, s5 words, E', C = α·x, F, admission, and H_i."""
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
                       "e": np.array([r[5] for r in st], np.uint8), "ok": np.array([r[6] for r in st], bool)}
    sides["A"]["F"], sides["B"]["F"] = basis.get()
    a = sides["A"]
    # Below the noise floor (α·ρ < 1, e.g. ρ = 0) a row is outside the domain and H_i has no size; such rows are left out of
    # the search, so they start at H_i sized as if α·ρ = 1.
    floor = (a["alpha"].view(np.float32).astype(np.float64) * a["rho"].view(np.float32) >= P.SIGMA_MIN)
    one = np.uint32(P.f32_bits(1.0))
    H, c = HS.hot_words(seed_a, SimpleNamespace(codes=[None] * rows, alpha=np.where(floor, a["alpha"], one).tolist(),
                                                rho=np.where(floor, a["rho"], one).tolist()), None, "const")
    return {"fam": fam, "u": u, "k": k, "rows": rows, "salt": salt, "seed_a": seed_a, "seed_b": seed_b, "sides": sides,
            "H": np.array(H, np.uint32), "c0": c, "below_floor": ~floor}


def closed_form_h(alpha: np.ndarray, rho: np.ndarray, H: np.ndarray, below_floor: np.ndarray) -> int:
    """Rows in the domain where H_i's exponent differs from v2-hot-start-rule.md's e_i = E(t_i) + 10 + [M(t_i) >= 0x3504F4],
    with t_i = fl32(α_i·ρ_i)."""
    t = (alpha.view(np.float32) * rho.view(np.float32)).view(np.uint32)
    e = ((t >> 23) & 0xFF).astype(np.int64) - 127 + 10 + ((t & 0x7FFFFF) >= 0x3504F4)
    return int(((((H >> 23) & 0xFF).astype(np.int64) - 127 != e) & ~below_floor).sum())


# ---- the card ------------------------------------------------------------------------------------------------------------

def form_side(dev: Dev, side: dict) -> int:
    ptr, cptr = dev.form(side["C"], side["ep"], side["F"], pearl=True)
    dev.free(cptr)
    return ptr


def chain_bits(dev: Dev, da: int, db: int, m: int, n: int, k: int, h: int, sink=None) -> tuple[np.ndarray, np.ndarray]:
    """One chain; its atoms' row and column bitsets go to sink(layout, t0, (32, rows, cols / 32) uint32) per flag plane.
    Returns the final words and each atom's exact count."""
    flags, out = dev.chain(da, db, m, n, k, h)
    T = k // 32
    M = dev.alloc(4 * 32 * m * (n // 32))
    counts = np.zeros(T, np.int64)
    for t0 in range(0, T, 32):
        for tr, layout in ((0, "rows"), (1, "cols")):
            dev.atom_bits(flags, M, m, n, t0, 32, tr)
            rows_, cols_ = (n, m) if tr else (m, n)
            planes = dev.fetch(M, 32 * rows_ * (cols_ // 32), np.uint32).reshape(32, rows_, cols_ // 32)
            if not tr:
                counts[t0:t0 + 32] = popc(planes).sum((1, 2))
            if sink:
                sink(layout, t0, planes)
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


# ---- items ---------------------------------------------------------------------------------------------------------------

def selftest(dev: Dev, pool: Pool, fh) -> dict:
    k, rows, T = 1024, 128, 32
    res = {"item": "selftest", "gates": []}
    ok = True
    for fam, words_side in SELFTEST:
        u = prep(pool, fam, k, rows, 1)
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
        da, db = form_side(dev, A), form_side(dev, B)
        ca = dev.fetch(da, rows * k, np.uint8).reshape(rows, k)
        cb = dev.fetch(db, rows * k, np.uint8).reshape(rows, k)
        g["form_vs_form_v1"] = int((ca != fa).sum() + (cb != fb).sum())
        g["H_closed_form_mismatches"] = closed_form_h(A["alpha"], A["rho"], u["H"], u["below_floor"])
        hp = dev.put(u["H"])
        for chain, h in (("hot", hp), ("zero", 0)):
            got = {}
            fin, counts = chain_bits(dev, da, db, rows, rows, k, h,
                                     lambda layout, t0, planes: got.__setitem__(layout, planes.copy()))
            bad_f = bad_w = bad_t = 0
            for i in range(words_side):
                for j in range(words_side):
                    want, wfin = ref_chain(ca[i].tolist(), cb[j].tolist(), int(u["H"][i]) if chain == "hot" else 0, T)
                    for t in range(T):
                        bad_f += bit(got["rows"], t, i, j) != want[t]
                        bad_t += bit(got["cols"], t, j, i) != want[t]
                    bad_w += not same_word(int(fin[i, j]), wfin)
            g[f"{chain}_flag_mismatches"], g[f"{chain}_cols_mismatches"], g[f"{chain}_word_mismatches"] = bad_f, bad_t, bad_w
            g[f"{chain}_words_checked"] = words_side * words_side
            g[f"{chain}_steps_exact"] = round(float(counts.sum() / (rows * rows * T)), 6)
        dev.free(hp)
        dev.free(da)
        dev.free(db)
        g["ok"] = all(v == 0 for x, v in g.items() if x.endswith(("mismatches", "_vs_hot_start_units", "_vs_hot_words",
                                                                   "form_vs_form_v1")))
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


def unit(dev: Dev, pool: Pool, fh, out_dir: Path, u: dict, seed: int) -> dict:
    fam, k, m = u["fam"], u["k"], u["rows"]
    n, T = m, k // 32
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
    res = {"item": f"unit:{name}", "family": fam, "unit": u["u"], "salt": u["salt"], "m": m, "n": n, "k": k,
           "forming": "pearl_c.form_v1 on SM120_UNPROMOTED (host: s5_stats, eprime, alpha*x; card: the atom, Pearl's cast)",
           "hot_rule": "const", "c0": u["c0"], "rows_admitted": {"A": int(A["ok"].sum()), "B": int(B["ok"].sum())},
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
            bad += P.f32_to_fp8(P.f32_of_bits(w)) != codes[r[q], l[q]]
    res["form_gate"] = {"rows": 4, "elements": 8192, "mismatches": bad}
    res["H_closed_form_mismatches"] = closed_form_h(A["alpha"], A["rho"], u["H"], u["below_floor"])
    t_form = time.time()
    ii, jj = rng.integers(0, m, 64), rng.integers(0, n, 64)
    hp = dev.put(u["H"])
    res["chains"] = {}
    for chain, h in (("hot", hp), ("zero", 0)):
        files = {lay: open(ud / f"{chain}.{lay}.u32", "wb") for lay in ("rows", "cols")}
        sample = {"rows": np.zeros((64, T), bool), "cols": np.zeros((64, T), bool)}

        def sink(layout, t0, planes):
            planes.tofile(files[layout])
            if layout == "rows":
                sample["rows"][:, t0:t0 + 32] = ((planes[:, ii, jj >> 5] >> (jj & 31)) & 1).T.astype(bool)
            else:
                sample["cols"][:, t0:t0 + 32] = ((planes[:, jj, ii >> 5] >> (ii & 31)) & 1).T.astype(bool)

        fin, counts = chain_bits(dev, da, db, m, n, k, h, sink)
        for fobj in files.values():
            fobj.close()
        bf = bc = bw = 0
        for q in range(64):
            want, wfin = ref_chain(ca[ii[q]].tolist(), cb[jj[q]].tolist(), int(u["H"][ii[q]]) if chain == "hot" else 0, T)
            want = np.array(want, bool)
            bf += int((sample["rows"][q] != want).sum())
            bc += int((sample["cols"][q] != want).sum())
            bw += not same_word(int(fin[ii[q], jj[q]]), wfin)
        res["chains"][chain] = {"gate": {"words": 64, "atoms": T, "flag_mismatches": bf, "cols_mismatches": bc,
                                         "word_mismatches": bw},
                                "exact_per_atom": counts.tolist(), "steps_exact": round(float(counts.sum() / (m * n * T)), 6),
                                "bytes_per_layout": T * m * n // 8}
        del fin
    dev.free(hp)
    dev.free(da)
    dev.free(db)
    res["ok"] = (bad == 0 and res["H_closed_form_mismatches"] == 0 and
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
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--nvcc", default=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / "log.jsonl", "a")
    code = load_ref(args.ref)
    items = ["selftest", *(f"unit:{f}@{u}" for u in range(1, args.units + 1) for f in args.families.split(",") if f)]
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
    emit(fh, {"identity": {**ident, "nvml_driver": nvml.driver, "numpy": np.__version__, "code": code,
                           "clock_label": args.clock_label, "clocks": nvml.read(), "start": time.strftime("%FT%TZ", time.gmtime())}})
    cubin, rec = build(out_dir, args.nvcc)
    emit(fh, {"build": rec})
    if rec["problems"]:
        sys.exit(5)
    dev.load(cubin)

    pool = Pool(args.procs, initializer=load_ref, initargs=(args.ref,))
    ahead = ThreadPoolExecutor(1)
    unit_secs: list[float] = []

    def prep_of(item):
        fam, _, u = item.split(":", 1)[1].partition("@")
        return ahead.submit(prep, pool, fam, args.k, args.rows, int(u))

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
            res = selftest(dev, pool, fh)
        else:
            u = (nxt or prep_of(item)).result()
            nxt = prep_of(todo[q + 1]) if q + 1 < len(todo) and fits(2) else None
            res = unit(dev, pool, fh, out_dir, u, args.seed)
            del u
        res.update({"identity": ident, "build_cubin_sha256": rec["cubin_sha256"], "code": code, "clocks": nvml.read(),
                    "clock_label": args.clock_label})
        stem = item.replace(":", "-")
        if not res["ok"]:
            (out_dir / f"{stem}.failed.json").write_text(json.dumps(res, indent=1) + "\n")
            emit(fh, {"gate_failed": {x: res[x] for x in res if x not in ("identity", "code")}})
            sys.exit(4)
        if item != "selftest":
            (out_dir / "units" / item.split(":", 1)[1] / "unit.json").write_text(json.dumps(res, indent=1) + "\n")
        (out_dir / f"{stem}.json").write_text(json.dumps(res, indent=1) + "\n")
        emit(fh, {"done": {x: res.get(x) for x in ("item", "rows_admitted", "H_exponents", "seconds")} |
                  {c: res["chains"][c]["steps_exact"] for c in res.get("chains", {})}})
        state["done"].append(item)
        state_path.write_text(json.dumps(state))
        if item != "selftest":
            unit_secs.append(time.time() - ts)
    pool.close()
    write_summary(out_dir, items, fh)


def write_summary(out_dir: Path, items: list[str], fh) -> None:
    rows = []
    for item in items[1:]:
        r = json.loads((out_dir / f"{item.replace(':', '-')}.json").read_text())
        rows.append({"item": item, "family": r["family"], "unit": r["unit"], "rows_admitted": r["rows_admitted"],
                     "H_exponents": r["H_exponents"], **{f"steps_exact_{c}": r["chains"][c]["steps_exact"] for c in CHAINS}})
    doc = {"tool": "pouw_sm120_exact_hot", "rows": rows,
           "selftest": json.loads((out_dir / "selftest.json").read_text())["gates"]}
    path = out_dir / "summary.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    emit(fh, {"summary": len(rows)})
    run_result.write(path, True, "sm120-exact-hot", f"{len(rows)} units of both chains written for hot_search.py",
                     [("units", len(rows), "count")])


if __name__ == "__main__":
    main()
