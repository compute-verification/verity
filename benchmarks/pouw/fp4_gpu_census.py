#!/usr/bin/env python3
"""Pearl-C4's enforced tile rule (#580 at 14d6f1bb, through `fp4_tile_gpu`'s port, which its gate holds bit for bit to
the reference) on Qwen2.5-7B: all 28 layers, every linear, under none|pc, rotb8s|pc and rotb8s|al (`fp4_kt_census`'s
weights), against `fp4_census`'s 15 activation families, 16 tiles of 64 x 64 per (variant, family, layer, linear).

Three fill jobs, each queuing the next (--then) once its stage is done:
  prep   (CPU)  per (variant, layer): each linear's B rows, 16 keyed 64-row blocks of the transformed W (k_proj and v_proj
                have 8, all of W), and each family's A rows, 64 (128 for k_proj and v_proj, so every linear has 16
                tiles), as BF16; the reference's parse_row and stats per row, the E lines and both F bases
  gpu    (GPU)  per (variant, layer): every side's forming and salt-dead codes (`Engine` on CUDA); then per family every
                word of A x B through `fp4_replay.cu` (identity atoms, sub-grid blocks), a few of them again on its CPU
                build and a few rows re-formed on the CPU, which must agree
  judge  (CPU)  per (variant, layer, linear): each tile through `fp4_tile_gpu.judge` (D-SB, D-NF, R1, the debit's items,
                debit / cap, the reject), with each row's `row_terms` (salt-dead codes, F1′) evaluated once for all the
                tiles it is in and the reference's per-row dead_blocks, row_passes and spread_ok memoized per row; one
                tile per task again from scratch (a fresh memo, nothing memoized per row), and one per (variant, layer)
                through the reference's own code end to end (`tile_outcome(port=False)`), each of which must agree

The noise: one salt and B root per (variant, layer, linear) and one A root per family, keyed by labels, so B̃ is formed
once per linear, as a deployment forms it once per salt.  A tile of a family is an A block x a B block of the unit
m x k x n (m the family's A rows, n the linear's), with the E lines of its A rows' and W rows' own indices.

Shortcuts: no split (k_unit = k: a deployed split is fixed per linear, and real.py's stand-in is chosen per X); B is 16
keyed blocks of W, not all of it; m is 64 (128), not a deployed batch; the noise is keyed by labels, not an epoch's beacon.

    python fp4_gpu_census.py prep  --out DIR --rule-tree TREE [--then JOB]
    python fp4_gpu_census.py gpu   --out DIR --rule-tree TREE [--then JOB]
    python fp4_gpu_census.py judge --out DIR --rule-tree TREE

Each exits 99 once --budget-min is spent with work left (one fill chunk; every unit's output is its checkpoint), and 0
when its stage is done, after copying --then (a fill job script) into the fill queue once; judge writes DIR/summary.json
and exits 1 if a check disagreed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fp4_census as C
import fp4_kt_census as KT
import fp4_tile_gpu as G

TILES, TILE = 16, 64
LAYERS = tuple(range(28))
DIMS = {"self_attn.q_proj": (3584, 3584), "self_attn.k_proj": (512, 3584), "self_attn.v_proj": (512, 3584),
        "self_attn.o_proj": (3584, 3584), "mlp.gate_proj": (18944, 3584), "mlp.up_proj": (18944, 3584),
        "mlp.down_proj": (3584, 18944)}                  # (n, k) of each linear
FILL_QUEUE = Path(os.environ.get("FILL_DIR", "/workspace/pouw/fill")) / "queue"
F = P = W = None


def setup(tree: str) -> Path:
    global F, P, W
    root = G.use_tree(tree)
    F, P, W = G.F, G.P, G.W
    return root


def short(lin: str) -> str:
    return lin.split(".")[1]


def unit_name(variant: str, layer: int) -> str:
    return f"{KT.tag(variant)}--L{layer:02d}"


def label(*parts) -> bytes:
    return hashlib.sha256("/".join(["fp4-gpu-census", *map(str, parts)]).encode()).digest()


def shape(lin: str) -> tuple[int, int, int, int]:
    """(n, k, B blocks, A blocks) of a linear: every linear has TILES tiles, A blocks x B blocks."""
    n, k = DIMS[lin]
    nb = min(TILES, n // TILE)
    return n, k, nb, -(-TILES // nb)


def context(variant: str, layer: int, lin: str, fam: str):
    from verity_pouw.protocol import Context, Matmul, Shape
    n, k, _, ma = shape(lin)
    li = KT.LINEARS.index(lin)
    return Context(label(variant, layer, lin, "salt"), Matmul(8 * layer + li, Shape(ma * TILE, k, n), 0),
                   label(variant, layer, lin, fam, "root_a"), label(variant, layer, lin, "root_b"))


def tile_of(lin: str, t: int) -> tuple[range, range]:
    """Tile t's A rows and its B rows (positions in the unit's sampled B)."""
    _, _, nb, _ = shape(lin)
    a, b = divmod(t, nb)
    return range(a * TILE, (a + 1) * TILE), range(b * TILE, (b + 1) * TILE)


def bf16_codes(x) -> np.ndarray:
    import torch
    return torch.as_tensor(np.asarray(x, np.float32)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def queue_next(out: Path, job: str) -> None:
    """Copy the fill job `job` into the fill queue, once."""
    mark = out / f"queued-{Path(job).name}"
    if not job or mark.exists():
        return
    tmp = FILL_QUEUE / f".{Path(job).name}.tmp"
    tmp.write_bytes(Path(job).read_bytes())
    tmp.chmod(0o755)
    tmp.replace(FILL_QUEUE / Path(job).name)
    mark.touch()


def pool_run(tasks: list, fn, args, est_min) -> int:
    """Run `fn` over `tasks` (each (name, arg), its output DIR/<name>), longest first, starting a task only while the
    budget has room for its estimate; 99 if tasks remain."""
    t0 = time.monotonic()
    pending = sorted(tasks, key=lambda t: -est_min(t[1]))
    left = len(pending)
    with ProcessPoolExecutor(args.workers, initializer=_worker_init, initargs=(args.rule_tree,)) as pool:
        flight = {}
        while pending or flight:
            while len(flight) < args.workers:
                room = args.budget_min - (time.monotonic() - t0) / 60
                fits = [t for t in pending if est_min(t[1]) <= room]
                if not fits:
                    break
                pending.remove(fits[0])
                flight[pool.submit(fn, fits[0][1])] = fits[0][0]
            if not flight:
                break
            for fut in wait(flight, return_when=FIRST_COMPLETED)[0]:
                flight.pop(fut)
                fut.result()
                left -= 1
    if left:
        print(f"{left} tasks remain", flush=True)
        return 99
    return 0


def _worker_init(tree: str) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    setup(tree)
    import torch
    torch.set_num_threads(1)


# ---- prep (CPU) ---------------------------------------------------------------------------------------------------

def side_rows(codes: np.ndarray, e_lines: list, k: int) -> dict:
    """One side's rows through the reference's parse_row and stats: α, β, a_E per row, and why a row fails (the
    reference's P.Reject, with the step: "parse" or "stats"), "" when it doesn't."""
    stats, fail = [], []
    for row in codes:
        try:
            x = F.parse_row(row.tobytes(), k)
        except P.Reject as e:
            stats.append((0, 0, 0)), fail.append(f"parse: {e}")
            continue
        try:
            st = F.stats(F.NVFP4, x)
        except P.Reject as e:
            stats.append((0, 0, 0)), fail.append(f"stats: {e}")
            continue
        stats.append((st[2], st[3], st[4])), fail.append("")
    return {"codes": codes, "stats": np.array(stats, np.int64), "e": np.array(e_lines, np.uint8), "fail": fail}


def prep_unit(task: tuple) -> None:
    variant, layer, out, prep, model = task
    dst = out / "rows" / f"{unit_name(variant, layer)}.npz"
    s = F.PearlC4("nvfp4")
    ws = KT.layer_weights(KT.Shards(model), layer, variant, np.load(prep), blocks=TILES)
    arrs, fails = {}, {}
    for lin in KT.LINEARS:
        w, side = ws[lin]
        n, k, nb, ma = shape(lin)
        idx = KT.w_blocks(f"{layer}.{lin}", n, TILES)
        if w.shape != (nb * TILE, k) or len(idx) != nb * TILE:
            raise SystemExit(f"{lin}: W rows {w.shape}, expected {(nb * TILE, k)}")
        seed_b = s.noise(context(variant, layer, lin, KT.FAMILIES[0])).seed_b
        sides = {"b": side_rows(bf16_codes(w), [F.line(seed_b, F.ROLE_EB, int(j)) for j in idx], k)}
        for fam in KT.FAMILIES:
            x = C.family(fam, ma * TILE, k, np.random.default_rng([KT.SEED, layer, k, *fam.encode()]))[0]
            if side is not None:
                x[:, side] = 0.0
            seed_a = s.noise(context(variant, layer, lin, fam)).seed_a
            sides[f"a.{fam}"] = side_rows(bf16_codes(x), [F.line(seed_a, F.ROLE_EA, i) for i in range(ma * TILE)], k)
        nm = short(lin)
        arrs |= {f"{nm}.b_idx": idx, f"{nm}.f_a": np.array(F.basis(seed_b, F.ROLE_FA, k), np.uint8),
                 f"{nm}.f_b": np.array(F.basis(seed_b, F.ROLE_FB, k), np.uint8)}
        for sd, v in sides.items():
            arrs |= {f"{nm}.{sd}.codes": v["codes"], f"{nm}.{sd}.stats": v["stats"], f"{nm}.{sd}.e": v["e"]}
            fails[f"{nm}.{sd}"] = {i: f for i, f in enumerate(v["fail"]) if f}
    tmp = dst.with_name(dst.stem + ".tmp.npz")
    np.savez(tmp, **arrs)
    dst.with_suffix(".json").write_text(json.dumps({"variant": variant, "layer": layer, "fails": fails}))
    tmp.replace(dst)


def cmd_prep(args) -> int:
    out = Path(args.out)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    tasks = [(unit_name(v, i), (v, i, out, args.prep, args.model)) for v in args.variants for i in args.layers
             if not (out / "rows" / f"{unit_name(v, i)}.npz").exists()]
    rc = pool_run(tasks, prep_unit, args, lambda a: 3.0)
    if rc == 0:
        queue_next(out, args.then)
    return rc


# ---- gpu ----------------------------------------------------------------------------------------------------------

def form_side(eng, rows, nm: str, sd: str, fails: dict, device: str) -> dict:
    """One side's codes, scales and salt-dead codes on `Engine` (a failing row formed as zeros: its tiles reject)."""
    import torch
    codes16, st, e = rows[f"{nm}.{sd}.codes"], rows[f"{nm}.{sd}.stats"].copy(), rows[f"{nm}.{sd}.e"]
    f = rows[f"{nm}.f_b" if sd == "b" else f"{nm}.f_a"]
    x = codes16.astype(np.int64) << 16
    bad = [int(i) for i in fails.get(f"{nm}.{sd}", {})]
    x[bad], st[bad] = 0, (0x3F800000, 0, 0x38)
    R, k = x.shape
    chunk = max(1, (1 << 21) // k)
    ft = torch.tensor(f.astype(np.int64), device=device)
    out = {"codes": [], "scales": [], "dead": [], "dead_ok": []}
    for r0 in range(0, R, chunk):
        xs, al, ae, es = (torch.tensor(v[r0:r0 + chunk], device=device)
                          for v in (x, st[:, 0], st[:, 2], e.astype(np.int64)))
        c, s = eng.form(xs, al, ae, es, ft)
        d, ok = eng.salt_dead(xs, al, ae)
        for kk, v in (("codes", c), ("scales", s), ("dead", d), ("dead_ok", ok)):
            out[kk].append(v.cpu().numpy())
    return {"codes": np.concatenate(out["codes"]).astype(np.uint8), "scales": np.concatenate(out["scales"]).astype(np.uint8),
            "dead": np.packbits(np.concatenate(out["dead"]), axis=1), "dead_ok": np.concatenate(out["dead_ok"])}


def spot_form(rows, formed, nm: str, sd: str, fails: dict) -> None:
    """Two rows of a side re-formed by `Engine` on the CPU: codes, scales and salt-dead codes must agree."""
    cpu = G.engine("cpu")
    got = form_side(cpu, {kk: (rows[kk][:2] if kk.startswith(f"{nm}.{sd}.") else rows[kk])
                          for kk in (f"{nm}.{sd}.codes", f"{nm}.{sd}.stats", f"{nm}.{sd}.e", f"{nm}.f_a", f"{nm}.f_b")},
                    nm, sd, {f"{nm}.{sd}": {i: m for i, m in fails.get(f"{nm}.{sd}", {}).items() if int(i) < 2}}, "cpu")
    for kk in ("codes", "scales", "dead", "dead_ok"):
        if not np.array_equal(got[kk], formed[f"{nm}.{sd}.{kk}"][:2]):
            raise SystemExit(f"{nm}.{sd}: CUDA {kk} differ from the CPU's on its first rows")


def gpu_unit(out: Path, variant: str, layer: int, eng, rep, cpu_rep, deadline: float, device: str = "cuda") -> bool:
    """One (variant, layer): its formed sides, then each family's words; False once the deadline passes."""
    name = unit_name(variant, layer)
    rows = np.load(out / "rows" / f"{name}.npz")
    fails = json.loads((out / "rows" / f"{name}.json").read_text())["fails"]
    fpath = out / "formed" / f"{name}.npz"
    if not fpath.exists():
        arrs = {}
        for lin in KT.LINEARS:
            nm = short(lin)
            for sd in ["b"] + [f"a.{fam}" for fam in KT.FAMILIES]:
                arrs |= {f"{nm}.{sd}.{kk}": v for kk, v in form_side(eng, rows, nm, sd, fails, device).items()}
        for lin in KT.LINEARS:
            for sd in ("b", f"a.{KT.FAMILIES[layer % len(KT.FAMILIES)]}"):
                spot_form(rows, arrs, short(lin), sd, fails)
        tmp = fpath.with_name(fpath.stem + ".tmp.npz")
        np.savez(tmp, **arrs)
        tmp.replace(fpath)
    formed = np.load(fpath)
    for fam in KT.FAMILIES:
        wpath = out / "words" / f"{KT.tag(variant)}--{fam}--L{layer:02d}.npz"
        if wpath.exists():
            continue
        if time.monotonic() > deadline:
            return False
        arrs = {}
        for lin in KT.LINEARS:
            nm = short(lin)
            a = (formed[f"{nm}.a.{fam}.codes"], formed[f"{nm}.a.{fam}.scales"])
            b = (formed[f"{nm}.b.codes"], formed[f"{nm}.b.scales"])
            ident, sub, err = rep.run(a[0], a[1], b[0], b[1])
            ci, cs, ce = cpu_rep.run(a[0][:2], a[1][:2], b[0][:8], b[1][:8])
            if not (np.array_equal(ci, ident[:2, :8]) and np.array_equal(cs, sub[:2, :8]) and np.array_equal(ce, err[:2, :8])):
                raise SystemExit(f"{name} {fam} {nm}: CUDA words differ from the CPU build's")
            if ident.max() >= 1 << 16 or sub.max() >= 1 << 16 or err.max() >= 1 << 8:
                raise SystemExit(f"{name} {fam} {nm}: a count past its storage")
            arrs |= {f"{nm}.ident": ident.astype(np.uint16), f"{nm}.sub": sub.astype(np.uint16),
                     f"{nm}.err": err.astype(np.uint8)}
        tmp = wpath.with_name(wpath.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrs)
        tmp.replace(wpath)
    return True


def cmd_gpu(args) -> int:
    out = Path(args.out)
    for d in ("formed", "words"):
        (out / d).mkdir(parents=True, exist_ok=True)
    eng, rep, cpu_rep = G.engine(args.device), G.Replay("cuda" if args.device == "cuda" else "cpu"), G.Replay("cpu")
    deadline = time.monotonic() + 60 * args.budget_min
    t0 = time.monotonic()
    for v in args.variants:
        for i in args.layers:
            if all((out / "words" / f"{KT.tag(v)}--{fam}--L{i:02d}.npz").exists() for fam in KT.FAMILIES):
                continue
            if time.monotonic() > deadline or not gpu_unit(out, v, i, eng, rep, cpu_rep, deadline, args.device):
                print(f"gpu: budget spent after {(time.monotonic() - t0) / 60:.1f} min", flush=True)
                return 99
    queue_next(out, args.then)
    return 0


# ---- judge (CPU) --------------------------------------------------------------------------------------------------

class RowMemo:
    """The reference's per-row functions (dead_blocks, row_passes, spread_ok), each evaluated once per row object for
    as long as this memo lives, answered while it is entered."""

    NAMES = ("dead_blocks", "row_passes", "spread_ok")

    def __init__(self):
        self.orig = {nm: getattr(F, nm) for nm in self.NAMES}
        self.memo = {nm: self._memo(fn) for nm, fn in self.orig.items()}

    def __enter__(self):
        for nm, fn in self.memo.items():
            setattr(F, nm, fn)
        return self

    def __exit__(self, *exc):
        for nm, fn in self.orig.items():
            setattr(F, nm, fn)

    @staticmethod
    def _memo(fn):
        cache: dict = {}

        def memo(fmt, row):
            hit = cache.get(id(row))
            if hit is None or hit[0] is not row or hit[1] is not fmt:
                hit = cache[id(row)] = (row, fmt, fn(fmt, row))
            return hit[2]

        return memo


class TileMemo(dict):
    """`row_terms`'s memo for one tile, keyed by tile position, over terms kept per row of the side: a row's terms are
    a function of the row alone (its words, α, a_E, E line, scales and salt-dead codes), so every tile it is in reads
    one evaluation."""

    def __init__(self, a: dict, b: dict, rows, cols):
        super().__init__()
        self.terms = {"A": (a, list(rows)), "B": (b, list(cols))}

    def _at(self, key):
        side, p = key
        terms, pos = self.terms[side]
        return terms, pos[p]

    def __contains__(self, key):
        terms, row = self._at(key)
        return row in terms

    def __getitem__(self, key):
        terms, row = self._at(key)
        return terms[row]

    def __setitem__(self, key, value):
        terms, row = self._at(key)
        terms[row] = value


class Side:
    """One side's rows as the reference's Formed rows (lists, one object per row, shared by every tile)."""

    def __init__(self, rows, formed, nm: str, sd: str, fails: dict):
        codes16 = rows[f"{nm}.{sd}.codes"]
        st = rows[f"{nm}.{sd}.stats"]
        self.bytes = [r.tobytes() for r in codes16]
        self.x = (codes16.astype(np.int64) << 16).tolist()
        self.codes = formed[f"{nm}.{sd}.codes"].astype(np.int64).tolist()
        self.scales = formed[f"{nm}.{sd}.scales"].astype(np.int64).tolist()
        self.alpha, self.beta, self.ae = (st[:, c].tolist() for c in range(3))
        self.e = rows[f"{nm}.{sd}.e"].astype(np.int64).tolist()
        k = codes16.shape[1]
        dead = np.unpackbits(formed[f"{nm}.{sd}.dead"], axis=1, count=k).astype(bool).tolist()
        self.dead = [d if ok else None for d, ok in zip(dead, formed[f"{nm}.{sd}.dead_ok"].tolist())]
        self.fail = {int(i): m for i, m in fails.get(f"{nm}.{sd}", {}).items()}

    def formed(self, idx):
        return F.Formed([self.codes[i] for i in idx], [self.scales[i] for i in idx], [self.alpha[i] for i in idx],
                        [self.beta[i] for i in idx], [self.ae[i] for i in idx], [self.x[i] for i in idx])

    def reject(self, idx) -> str:
        """The reference's P.Reject while forming these rows (`PearlC4._side`): parse_row over every row, then stats row
        by row, then D-SB; "" when none."""
        for step in ("parse", "stats"):
            for i in idx:
                if self.fail.get(i, "").startswith(step):
                    return self.fail[i].split(": ", 1)[1]
        try:
            F.check_scale_bytes(F.NVFP4, self.formed(idx))
        except P.Reject as e:
            return str(e)
        return ""


def judge_tile(s, ctx, A: Side, B: Side, rows, cols, words, terms=None) -> dict:
    """One tile's outcome, `fp4_tile_gpu.tile_outcome`'s record, from the formed sides; `terms` the (A, B) row terms
    kept across tiles (a fresh memo when None)."""
    for side, sel in ((A, rows), (B, cols)):
        why = side.reject(sel)
        if why:
            return {"rejected": why}
    a, b = A.formed(rows), B.formed(cols)
    e_a, e_b = [A.e[i] for i in rows], [B.e[j] for j in cols]
    dead = [A.dead[i] for i in rows], [B.dead[j] for j in cols]
    memo = None if terms is None else TileMemo(*terms, rows, cols)
    try:
        return {"operands": G.operand_digest(a, b)} | G.judge(s, ctx.matmul, ctx.matmul.shape.k, a, e_a, b, e_b, words,
                                                              dead, memo)
    except Exception as e:  # noqa: BLE001        as fp4_tile_gpu's gate records it
        return {"raised": f"{type(e).__name__}: {e}"}


def judge_linear(task: tuple) -> None:
    variant, layer, lin, out, ref_tile = task
    name, nm = unit_name(variant, layer), short(lin)
    dst = out / "judge" / f"{name}--{nm}.json"
    rows = np.load(out / "rows" / f"{name}.npz")
    formed = np.load(out / "formed" / f"{name}.npz")
    fails = json.loads((out / "rows" / f"{name}.json").read_text())["fails"]
    idx = rows[f"{nm}.b_idx"]
    s = F.PearlC4("nvfp4")
    B = Side(rows, formed, nm, "b", fails)
    h = int.from_bytes(label(variant, layer, lin, "check")[:4], "little")
    check_fam, check_t = KT.FAMILIES[h % len(KT.FAMILIES)], (h >> 8) % TILES
    tiles, checks, terms_b, row_memo = [], [], {}, RowMemo()
    for fam in KT.FAMILIES:
        ctx = context(variant, layer, lin, fam)
        A = Side(rows, formed, nm, f"a.{fam}", fails)
        with np.load(out / "words" / f"{KT.tag(variant)}--{fam}--L{layer:02d}.npz") as z:
            words = [z[f"{nm}.{kk}"].astype(np.int64) for kk in ("ident", "sub", "err")]
        terms_a: dict = {}
        for t in range(TILES):
            r, c = tile_of(lin, t)
            wds = tuple(v[r.start:r.stop, c.start:c.stop] for v in words)
            c0 = time.perf_counter()
            with row_memo:
                got = judge_tile(s, ctx, A, B, list(r), list(c), wds, (terms_a, terms_b))
            tiles.append({"family": fam, "tile": t, "rows": [r.start, r.stop], "b_rows": [c.start, c.stop],
                          "w_rows": [int(idx[c.start]), int(idx[c.stop - 1]) + 1],
                          "seconds": round(time.perf_counter() - c0, 2), **got})
            if (fam, t) != (check_fam, check_t):
                continue
            fresh = judge_tile(s, ctx, A, B, list(r), list(c), wds)
            checks.append({"kind": "fresh", "family": fam, "tile": t, "agree": fresh == got})
            if ref_tile:
                tile = SimpleNamespace(rows=r, cols=[int(idx[j]) for j in c])
                c0 = time.perf_counter()
                ref = G._outcome(G.tile_outcome, s, ctx, [A.bytes[i] for i in r], [B.bytes[j] for j in c], tile,
                                 ctx.matmul.shape.k, port=False)
                checks.append({"kind": "reference", "family": fam, "tile": t, "agree": ref == got,
                               "seconds": round(time.perf_counter() - c0, 1)})
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps({"variant": variant, "layer": layer, "linear": lin, "shape": list(shape(lin)),
                               "tiles": tiles, "checks": checks}))
    tmp.replace(dst)


def summarize(out: Path, files: list[Path]) -> dict:
    per: dict = {}
    checks = Counter()
    for f in files:
        d = json.loads(f.read_text())
        for c in d["checks"]:
            checks[(c["kind"], c["agree"])] += 1
        for t in d["tiles"]:
            key = f"{d['variant']}|{short(d['linear'])}|{t['family']}"
            p = per.setdefault(key, {"tiles": 0, "rejected": 0, "raised": 0, "nothing_credited": 0, "debited": 0,
                                     "over_cap": 0, "r1_rows": 0, "dnf_fail_rows": 0, "max_ratio": "0", "ratios": [],
                                     "items": Counter(), "rejects": Counter()})
            p["tiles"] += 1
            if t.get("rejected"):
                p["rejected"] += 1
                p["rejects"][t["rejected"]] += 1
                continue
            if "raised" in t:
                p["raised"] += 1
                p["rejects"][t["raised"]] += 1
                continue
            p["r1_rows"] += sum(len(v) for v in t["r1"])
            p["dnf_fail_rows"] += sum(t["dnf_fail"])
            if "debit" not in t:
                p["nothing_credited"] += 1
                continue
            p["debited"] += 1
            p["over_cap"] += t["reject"]
            p["ratios"].append(Fraction(t["ratio"]))
            for kk, v in t["debit"].items():
                if kk != "counts" and isinstance(v, (int, str)):
                    try:
                        p["items"][kk] += Fraction(v)
                    except (ValueError, ZeroDivisionError):
                        pass
    for p in per.values():
        r = sorted(p.pop("ratios"))
        p["max_ratio"] = float(r[-1]) if r else None
        p["median_ratio"] = float(r[len(r) // 2]) if r else None
        p["items"] = {kk: float(v) for kk, v in p["items"].items()}
        p["rejects"] = dict(p["rejects"])
    return {"schema": "pouw/fp4-gpu-census/v0", "rule": "#580 at 14d6f1bb (pearl_c4, pearl_c_work)",
            "checks": {f"{kk[0]}:{'agree' if kk[1] else 'DISAGREE'}": v for kk, v in sorted(checks.items())},
            "check_disagreements": sum(v for kk, v in checks.items() if not kk[1]),
            "tiles": sum(p["tiles"] for p in per.values()), "over_cap": sum(p["over_cap"] for p in per.values()),
            "rejected": sum(p["rejected"] for p in per.values()), "raised": sum(p["raised"] for p in per.values()),
            "cells": per}


def cmd_judge(args) -> int:
    out = Path(args.out)
    (out / "judge").mkdir(parents=True, exist_ok=True)
    tasks = []
    short_k = [lin for lin in KT.LINEARS if DIMS[lin][1] == min(k for _, k in DIMS.values())]
    for v in args.variants:
        for i in args.layers:
            h = int.from_bytes(label(v, i, "reference-linear")[:4], "little")
            ref_lin = short_k[h % len(short_k)]
            for lin in KT.LINEARS:
                name = f"{unit_name(v, i)}--{short(lin)}"
                if not (out / "judge" / f"{name}.json").exists():
                    tasks.append((name, (v, i, lin, out, lin == ref_lin)))
    rc = pool_run(tasks, judge_linear, args, lambda a: (14.0 if DIMS[a[2]][1] > 4096 else 4.5) + 2.0 * a[4])
    if rc:
        return rc
    files = [out / "judge" / f"{unit_name(v, i)}--{short(lin)}.json" for v in args.variants for i in args.layers
             for lin in KT.LINEARS]
    summary = summarize(out, files)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({kk: summary[kk] for kk in ("tiles", "over_cap", "rejected", "raised", "checks",
                                                  "check_disagreements")}), flush=True)
    return 1 if summary["check_disagreements"] else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["prep", "gpu", "judge"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule-tree", default="")
    ap.add_argument("--model", default=KT.MODEL)
    ap.add_argument("--prep", default="/workspace/pouw/keyed-transforms/s7b/prep.npz")
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--variants", default=",".join(KT.VARIANTS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--budget-min", type=float, default=20.0)
    ap.add_argument("--then", default="")
    ap.add_argument("--device", default="cuda", help="gpu: the forming's torch device and the replay's build")
    args = ap.parse_args(argv)
    args.rule_tree = str(setup(args.rule_tree))
    args.layers = [int(v) for v in args.layers.split(",")]
    args.variants = args.variants.split(",")
    return {"prep": cmd_prep, "gpu": cmd_gpu, "judge": cmd_judge}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
