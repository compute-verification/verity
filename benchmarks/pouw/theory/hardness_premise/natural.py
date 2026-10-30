"""Fast-matmul routes on natural weights (HT-PREMISE round 3): the cheapest exact schedule for E·Wᵀ, W a trained matrix.

W is HT-PUBLIC's sample (`benchmarks/hardness_public/natural.py` at a0dd00f0d5: Qwen2.5 and Llama-3.x projections at
layers 0, L/2 and L − 1, per-output-channel symmetric int8), range-read from Hugging Face; E is the noise over `a`
values (b_E = log₂a), and the output is needed exactly (w = 16 or more). The schedule family is `search.py`'s, decided
on W's actual entries instead of their range:

- Strassen or Winograd levels, chosen per product (each of a node's seven products picks its own subtree), to `--depth`;
  W's forms are free (W is fixed before E), E's forms are counted as `execute.py` counts them;
- an s8 leaf takes its W-form C as lo + 256·hi with lo in [−128, 127], so only the entries that overflow int8 have
  hi ≠ 0. 256·(A·hi) is added by the cheapest exact route: an s8 product over the rows and columns of hi that hold a
  nonzero (gathered; MMA tile granularity ignored), one IMAD per noise entry and nonzero (ALU, plus loading the noise
  rows it reads), b1 planes (H200) or FP4 (sm_120); plus one write a touched output;
- at the root, the plane routes: W in s low b1 planes against E's two planes (H200) or in FP4 digits (sm_120), with
  the rest of W as a sparse high digit.

Every count is `execute.py`'s (W1: s8 1 a MAC, b1 0.125, nvf4 0.5, 32 a register write; `fast` = IADD3 and four noise
lanes a register, `plain` = IADD and two), E's affine map included (one write an output), and per MAC of E·Wᵀ. The
honest kernel is 1 (its leaf 1 + 32/K). `--kcaps` splits K into ⌈K/kcap⌉ equal chunks, each its own product and committed output.

Each matrix's chosen tree is executed exactly on a crop against E·Wᵀ (`check`), and the bookkeeping is checked against
`execute.run` on a crop small enough that nothing overflows (`check_counts`).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

import execute
from execute import PROGRAMS, SCHEMES, _eval, _form, _shift
from search import CHIPS

# HT-PUBLIC's loader, copied (its module imports torch at import time): the same models, tensors and quantization.
MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "llama3.2-1b": "unsloth/Llama-3.2-1B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "llama3.1-8b": "unsloth/Meta-Llama-3.1-8B",
}
PROJ = [("self_attn", p) for p in ("q_proj", "k_proj", "v_proj", "o_proj")] + [
    ("mlp", p) for p in ("gate_proj", "up_proj", "down_proj")
]
HF = "https://huggingface.co"


def fetch(url, start=None, end=None):
    headers = {"Range": f"bytes={start}-{end - 1}"} if start is not None else {}
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as f:  # noqa: S310  (HF's https URL)
                return f.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or attempt == 4:
                raise
        except Exception:
            if attempt == 4:
                raise
        time.sleep(2**attempt)


def safetensors_header(url):
    n = int.from_bytes(fetch(url, 0, 8), "little")
    return 8 + n, json.loads(fetch(url, 8, 8 + n))


def load_tensor(url, base, entry):
    b, e = entry["data_offsets"]
    raw = fetch(url, base + b, base + e)
    if entry["dtype"] == "BF16":
        a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif entry["dtype"] == "F16":
        a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        a = np.frombuffer(raw, dtype=np.float32).copy()
    return a.reshape(entry["shape"])


def quantize(w):
    s = np.abs(w).max(axis=1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    return np.clip(np.rint(w / s), -127, 127).astype(np.int16)


def model_tasks(name):
    repo = MODELS[name]
    cfg = json.loads(fetch(f"{HF}/{repo}/resolve/main/config.json"))
    nl = cfg["num_hidden_layers"]
    layers = sorted({0, nl // 2, nl - 1})
    try:
        idx = json.loads(fetch(f"{HF}/{repo}/resolve/main/model.safetensors.index.json"))["weight_map"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        idx = None
    wanted = [f"model.layers.{i}.{blk}.{p}.weight" for i in layers for blk, p in PROJ]
    files = sorted({idx[t] for t in wanted}) if idx else ["model.safetensors"]
    headers = {f: safetensors_header(f"{HF}/{repo}/resolve/main/{f}") for f in files}
    tasks = []
    for t in wanted:
        f = idx[t] if idx else "model.safetensors"
        base, h = headers[f]
        tasks.append((name, t, f"{HF}/{repo}/resolve/main/{f}", base, h[t]))
    return tasks


INF = float("inf")
S8 = (-128, 127)


def hi_stats(hi):
    nz = hi != 0
    return {"S": int(nz.any(axis=1).sum()), "T": int(nz.any(axis=0).sum()), "nnz": int(nz.sum()),
            "hmax": int(np.abs(hi).max())}


def forms(C, lvl, depth):
    """W's side of every schedule to `depth`: per node its shape, the statistics of its overflow digit (None if the
    form fits int8), and per scheme the seven products' subtrees. W's forms depend only on W, so one tree serves every
    chip and reading."""
    Kp, n = C.shape
    st = {"shape": (Kp, n), "hi": None}
    if C.min() < S8[0] or C.max() > S8[1]:
        st["hi"] = hi_stats((C + 128) >> 8)
    if lvl < depth and Kp % 2 == 0 and n % 2 == 0 and Kp >= 64 and n >= 16:
        h, w = Kp // 2, n // 2
        Bb = [C[:h, :w], C[:h, w:], C[h:, :w], C[h:, w:]]
        st["kids"] = {}
        for sname, (_U, V, _W) in SCHEMES.items():
            kids = []
            for v in V:
                Cv = None
                for q, c in enumerate(v):
                    if c:
                        Cv = c * Bb[q] if Cv is None else Cv + c * Bb[q]
                kids.append(forms(Cv, lvl + 1, depth))
            st["kids"][sname] = kids
    return st


def plane_stats(C):
    """The root's digit splits of W: s low digits for b1 planes, one or two FP4 digits, and the rest's statistics."""
    C32 = C.astype(np.int32)
    out = {}
    for s in range(1, 8):
        lo = ((C32 + 2 ** (s - 1)) & (2**s - 1)) - 2 ** (s - 1)
        hi = (C32 - lo) >> s
        out[f"b1_s{s}"] = hi_stats(hi) if hi.any() else None
    rest = C32
    for d in (1, 2):
        lo = ((rest + 4) & 7) - 4
        rest = (rest - lo) >> 3
        out[f"nvf4_d{d}"] = hi_stats(rest) if rest.any() else None
    return out


class Natural:
    """The cheapest schedule for A·C, A the noise side (m×K) and C = Wᵀ's form (K×n), priced on one chip and reading."""

    def __init__(self, chip: str, reading: str, a: int = 4):
        self.chip, self.a = chip, a
        self.iadd3 = reading == "fast"
        self.lanes = 4 if self.iadd3 else 2
        self.b1, self.nvf4 = "b1" in CHIPS[chip], "nvf4" in CHIPS[chip]

    def corr(self, hs, ia, m):
        """W1 to add 256·(A·hi) exactly into the outputs hi touches, hi the overflow digit."""
        S, T, nnz, hmax = hs["S"], hs["T"], hs["nnz"], hs["hmax"]
        opts = [(m * S * T + 32 * m * T, "gather"), (32 * m * nnz + 32 * m * S / self.lanes, "alu")]
        if self.b1:
            pE = max(1, math.ceil(math.log2(ia[1] - ia[0] + 1)))
            pH = 2 * math.ceil(math.log2(hmax + 1))
            opts.append((0.125 * pE * pH * m * S * T + 32 * m * T * (pE * pH + (ia[0] != 0))
                         + 32 * pE * m * S / self.lanes, "b1"))
        if self.nvf4 and hmax <= 4:
            s4 = _shift(ia, (-4, 4))
            if s4 is not None:
                opts.append((0.5 * m * S * T + 32 * m * T * (2 + (s4 != 0)), "nvf4"))
        return min(opts)

    def atom(self, st, ia, m):
        Kp, n = st["shape"]
        sa = _shift(ia, S8)
        if sa is None:
            return INF, None
        cost = m * Kp * n + (32 * m * n if sa else 0)
        node = {"atom": "s8", "sa": int(sa)}
        if st["hi"] is None:
            return cost, node
        c, how = self.corr(st["hi"], ia, m)
        return cost + c, dict(node, corr=how, **st["hi"])

    def node(self, st, ia, lvl):
        """(cost, tree) of the best schedule and of the best leaf, for A·C at level lvl (m = 2^−lvl of the root's)."""
        m = 0.5**lvl
        a_cost, a_tree = self.atom(st, ia, m)
        best, alt = (a_cost, a_tree), (INF, None)
        Kp, n = st["shape"]
        if "kids" in st:
            h, w = Kp // 2, n // 2
            ms = m / 2
            for sname, (U, _V, _W) in SCHEMES.items():
                kids = [self.node(k, _form([ia] * 4, u), lvl + 1) for k, u in zip(st["kids"][sname], U)]
                e_adds = sum(sum(map(abs, u)) - 1 for u in U) * ms * h
                for pi, prog in enumerate(PROGRAMS[sname]):
                    chained = {step[2] for step in prog if step[0] == "chain"}
                    tot, trees = 0.0, []
                    for t, (k_cost, k_tree, k_atom, k_atree) in enumerate(kids):
                        if t in chained and k_atom <= k_cost + 32 * ms * w:
                            tot += k_atom
                            trees.append(k_atree)
                        else:
                            tot += k_cost + (32 * ms * w if t in chained else 0)
                            trees.append(k_tree)
                    writes = sum((-(-(len(step[2]) - 1) // 2) if self.iadd3 else len(step[2]) - 1)
                                 for step in prog if step[0] == "sum")
                    tot += 32 * writes * ms * w + 32 * e_adds / self.lanes
                    if tot < alt[0]:
                        alt = (tot, {"scheme": sname, "program": pi, "kids": trees})
            best = min(best, alt, key=lambda x: x[0])
        if lvl == 0:
            self.root_alt = alt
        return best[0], best[1], a_cost, a_tree

    def planes(self, ps, Kp, n):
        """The root's plane routes: s low digits of W on b1 (H200) or d FP4 digits (sm_120), the rest a correction."""
        ia = (0, self.a - 1)
        pE = max(1, math.ceil(math.log2(self.a)))
        out = {}
        if self.b1:
            for s in range(1, 8):
                cost = 0.125 * pE * s * Kp * n + 32 * n * (pE * s + 1) + 32 * pE * Kp / self.lanes
                if ps[f"b1_s{s}"]:
                    cost += self.corr(ps[f"b1_s{s}"], ia, 1.0)[0]
                out[f"b1_s{s}"] = cost
        if self.nvf4:
            for d in (1, 2):
                cost = d * 0.5 * Kp * n + 32 * n * (3 * d - 1)  # per digit two drain writes, one combining write
                if ps[f"nvf4_d{d}"]:
                    cost += self.corr(ps[f"nvf4_d{d}"], ia, 1.0)[0]
                out[f"nvf4_d{d}"] = cost
        return out


def depth_of(t):
    return 1 + max(depth_of(k) for k in t["kids"]) if "scheme" in t else 0


def mix(t, lvl=0, acc=None):
    """Fractions of the root's MACs by leaf kind, and the corrections' MACs."""
    acc = acc if acc is not None else {}
    if "scheme" in t:
        for k in t["kids"]:
            mix(k, lvl + 1, acc)
    else:
        key = "s8+" + t["corr"] if t.get("corr") else "s8"
        acc[key] = acc.get(key, 0) + 8.0**-lvl
    return acc


def run(t, A, C):
    """A·C executed exactly by tree t (int64), asserting every MMA operand fits its format."""
    if "scheme" in t:
        U, V, _W = SCHEMES[t["scheme"]]
        h, k2, n2 = A.shape[0] // 2, A.shape[1] // 2, C.shape[1] // 2
        Ab = [A[:h, :k2], A[:h, k2:], A[h:, :k2], A[h:, k2:]]
        Bb = [C[:k2, :n2], C[:k2, n2:], C[k2:, :n2], C[k2:, n2:]]
        M = [run(kid, sum(c * Ab[p] for p, c in enumerate(u) if c), sum(c * Bb[q] for q, c in enumerate(v) if c))
             for kid, u, v in zip(t["kids"], U, V)]
        Y = _eval(PROGRAMS[t["scheme"]][t["program"]], M)
        return np.block([[Y[0], Y[1]], [Y[2], Y[3]]])
    sa = t["sa"]
    A2 = A - sa
    assert A2.min() >= S8[0] and A2.max() <= S8[1]
    lo = ((C + 128) & 255) - 128
    hi = (C - lo) >> 8
    Y = A2 @ lo + 256 * (A2 @ hi)
    if t.get("corr"):
        rows, cols = np.flatnonzero((hi != 0).any(axis=1)), np.flatnonzero((hi != 0).any(axis=0))
        assert (len(rows), len(cols)) == (t["S"], t["T"]) and np.abs(hi).max() <= 127
    else:
        assert not hi.any()
    return Y + sa * C.sum(axis=0)[None, :]


def check(q, chip, depth, a, seed=0):
    """The chosen tree on a crop of q (n 256, K 512): exact against E·Wᵀ."""
    n, K = min(256, q.shape[0]), min(512, q.shape[1])
    C = q[:n, :K].T
    _cost, tree, _ac, _at = Natural(chip, "fast", a).node(forms(C, 0, depth), (0, a - 1), 0)
    E = np.random.default_rng(seed).integers(0, a, (32, K), dtype=np.int64)
    C64 = C.astype(np.int64)
    return bool(np.array_equal(run(tree, E, C64), E @ C64)), depth_of(tree)


def check_full(q, chip, reading, depth, a, seed=0):
    """The served-K best and best non-plain trees on all of q, against E·Wᵀ for 2^depth noise rows."""
    C = q.T
    nat = Natural(chip, reading, a)
    _cost, tree, _ac, _at = nat.node(forms(C, 0, depth), (0, a - 1), 0)
    E = np.random.default_rng(seed).integers(0, a, (2**depth, C.shape[0]), dtype=np.int64)
    C64 = C.astype(np.int64)
    ref = E @ C64
    out = {"best": bool(np.array_equal(run(tree, E, C64), ref)), "best_depth": depth_of(tree)}
    if nat.root_alt[1] is not None:
        out.update(alt=bool(np.array_equal(run(nat.root_alt[1], E, C64), ref)), alt_depth=depth_of(nat.root_alt[1]))
    return out


def to_execute(t):
    if "scheme" in t:
        return {"scheme": t["scheme"], "roles": "AB", "w": 16, "kids": [to_execute(k) for k in t["kids"]]}
    return {"atom": "s8", "w": 16}


def check_counts(chip, reading, depth, a, seed=0):
    """On a W small enough that no form overflows, this module's count equals `execute.run`'s for the same tree."""
    C = np.random.default_rng(seed).integers(-1, 2, (512, 256)).astype(np.int16)
    nat = Natural(chip, reading, a)
    nat.node(forms(C, 0, depth), (0, a - 1), 0)
    cost, tree = nat.root_alt
    t = to_execute(tree)
    execute.heights(t)
    o = execute.Ops()
    m = 2**depth * 4
    execute.run(t, None, None, (0, a - 1), (int(C.min()), int(C.max())), (m, 512, 256), o)
    return cost * m, o.w1(reading)


def price(q, chips, readings, depth, a, kcap):
    """W1 per MAC of E·qᵀ: the best schedule, the best one that isn't the plain leaf, and the plane routes."""
    n, K = q.shape
    chunks = math.ceil(K / kcap) if kcap else 1
    while chunks > 1 and (K % chunks or (K // chunks) % 2**depth):
        chunks += 1
    kc = K // chunks
    sts, pss = [], []
    for c in range(chunks):
        C = q[:, c * kc:(c + 1) * kc].T
        sts.append(forms(C, 0, depth))
        pss.append(plane_stats(C))
    macs, rows = n * K, []
    for chip in chips:
        for reading in readings:
            nat = Natural(chip, reading, a)
            best = alt = 0.0
            planes, trees = {}, []
            for st, ps in zip(sts, pss):
                cost, tree, _ac, _at = nat.node(st, (0, a - 1), 0)
                best += cost + 32 * n
                alt += nat.root_alt[0] + 32 * n
                for k, v in nat.planes(ps, kc, n).items():
                    planes[k] = planes.get(k, 0.0) + v + 32 * n
                trees.append((tree, nat.root_alt[1]))
            rows.append({"chip": chip, "reading": reading, "depth": depth, "kcap": kcap, "chunks": chunks,
                         "K_chunk": kc, "plain": 1 + 32 * chunks / K, "best": best / macs, "alt": alt / macs,
                         "planes": {k: v / macs for k, v in planes.items()}, "best_depth": depth_of(trees[0][0]),
                         "best_mix": mix(trees[0][0]), "alt_depth": depth_of(trees[0][1]),
                         "alt_mix": mix(trees[0][1])})
    return rows


def overflow_profile(q, depth):
    """How often W's forms overflow int8: per level, the fraction of entries of the Strassen and Winograd forms."""
    C = q.T
    out = {}
    for sname, (_U, V, _W) in SCHEMES.items():
        level, frac = [C], []
        for _ in range(depth):
            nxt, over, tot = [], 0, 0
            for X in level:
                h, w = X.shape[0] // 2, X.shape[1] // 2
                Bb = [X[:h, :w], X[:h, w:], X[h:, :w], X[h:, w:]]
                for v in V:
                    Cv = sum(c * Bb[i] for i, c in enumerate(v) if c)
                    over += int(((Cv < S8[0]) | (Cv > S8[1])).sum())
                    tot += Cv.size
                    nxt.append(Cv)
            frac.append(over / tot)
            level = nxt
        out[sname] = frac
    return out


def work(task, chips, readings, depth, a, kcaps, full=False):
    name, tensor, url, base, entry = task
    t0 = time.time()
    q = quantize(load_tensor(url, base, entry))
    rec = {"model": name, "tensor": tensor, "shape": list(q.shape), "macs_per_token": int(q.size),
           "sigma_q": float(q.astype(np.float64).std()), "rows": []}
    for chip in chips:
        rec[f"exact_{chip}"] = check(q, chip, depth, a)
        if full:
            for reading in readings:
                rec[f"exact_full_{chip}_{reading}"] = check_full(q, chip, reading, depth, a)
    rec["overflow"] = overflow_profile(q[:, : q.shape[1] // 2**depth * 2**depth], min(depth, 2))
    for kcap in kcaps:
        if kcap and kcap >= q.shape[1] and 0 in kcaps:
            continue
        rec["rows"] += price(q, chips, readings, depth, a, kcap)
    rec["seconds"] = time.time() - t0
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--chips", default="h200 sm120")
    ap.add_argument("--readings", default="fast plain")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("-a", type=int, default=4, help="noise values (b_E = log2 a)")
    ap.add_argument("--kcaps", default="0 4096", help="0 = the served K")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--self-check", action="store_true", help="only the bookkeeping check against execute.run")
    ap.add_argument("--full-check", action="store_true", help="also execute the served-K trees on whole matrices")
    args = ap.parse_args(argv)
    chips, readings = args.chips.split(), args.readings.split()
    for chip in chips:
        for reading in readings:
            ours, theirs = check_counts(chip, reading, 2, args.a)
            print(f"check_counts {chip} {reading}: natural {ours:.1f} execute {theirs:.1f}", flush=True)
            assert abs(ours - theirs) < 1e-6 * theirs, (ours, theirs)
    if args.self_check:
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [t for name in args.models.split(",") for t in model_tasks(name)]
    tasks.sort(key=lambda t: -math.prod(t[4]["shape"]))
    kcaps = [int(k) for k in args.kcaps.split()]
    ok = True
    with concurrent.futures.ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, t, chips, readings, args.depth, args.a, kcaps, args.full_check) for t in tasks]
        with open(args.out / "matrices.jsonl", "w") as f:
            for fut in concurrent.futures.as_completed(futs):
                rec = fut.result()
                ok &= all(rec[f"exact_{c}"][0] for c in chips)
                ok &= all(v for k, x in rec.items() if k.startswith("exact_full") for kk, v in x.items()
                          if kk in ("best", "alt"))
                f.write(json.dumps(rec) + "\n")
                f.flush()
                fast = [r for r in rec["rows"] if r["reading"] == "fast"]
                print(rec["model"], rec["tensor"], rec["shape"], " ".join(
                    f"{r['chip']}/K{r['kcap'] or 'served'}: best {r['best']:.4f} alt {r['alt']:.4f} d{r['best_depth']}"
                    for r in fast), f"{rec['seconds']:.0f}s", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
