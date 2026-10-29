"""N-RED round 3: N-LEAN's per-entry cancellation (A8) on `fix2`-type designs, under W1 as stated and at warp grain.

Every design here keeps `fix2`'s step 1 (row tags against G, a₁·C) and adds, at each later step t, X_t(i)·w_t[j]:
X_t(i) from the call's coins, w_t[j] from coins drawn after B is committed (once a window).

- `fix2`: X = 1 + b_t ∈ {1, 2}, w ∈ {±1}; `fix2w2`: w ∈ {±1, ±2}; `fix2w13`: w ∈ {±1, ±3}.
- `uw4`, `uw6`, `uw8`: X and w both uniform over the nonzero integers in ±(2^(bits−1)) (±8, ±32, ±127).

A8 (chosen data, N-LEAN interim 1 §2b): zero data past step 1 except one lane a step with A = 1 and B = −v*, v* the
likeliest value of X·w. An entry's step-t word equals its step t − 1 word whenever X_t(i)·w_t[j] = v*, which has
probability p* = max_v Pr[X·w = v]. B is committed before w, so v* is a constant, which is all A8 needs.

- **Model P** (W1 as N-W1 states it: a price per fresh word written, data-dependent branches allowed): an unchanged
  accumulator isn't rewritten. Saving ≈ p*·(S − 1)/S of the words. One pointer a row-step is charged anyway.
- **Model W** (a stricter premise: a register write is charged for all 32 lanes of its warp, predicated or not, and
  the lane layout is fixed for the window because a per-step re-layout costs a write a word): a 1-row × 32-column
  warp-register is skipped only when all 32 entries are unchanged. The adversary groups columns once a window knowing
  w (`group_columns`, greedy), so that each group is uniform on as many steps as it can; rows whose X then gives v*
  skip that group's write. This is a lower bound on what model W leaves: other layouts could only add.

Each constructive adversary computes every changed word as its previous word plus X·w + the known data dot, keeps the
unchanged ones, and its words must equal the honest words (`np.array_equal`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noise_sweep as ns  # noqa: E402
import nred_attacks as na  # noqa: E402
import nred_round2 as r2  # noqa: E402

WARP = 32
D1 = 24


def _sym(top: int) -> tuple[int, ...]:
    return tuple(v for v in range(-top, top + 1) if v)


DESIGNS = {
    "fix2": ((1, 2), (-1, 1)),
    "fix2w2": ((1, 2), (-2, -1, 1, 2)),
    "fix2w13": ((1, 2), (-3, -1, 1, 3)),
    "uw4": (_sym(8), _sym(8)),
    "uw6": (_sym(32), _sym(32)),
    "uw8": (_sym(127), _sym(127)),
}
COIN_BITS = {"fix2": (1, 1), "fix2w2": (1, 2), "fix2w13": (1, 2), "uw4": (4, 4), "uw6": (6, 6), "uw8": (8, 8)}


def point_mass(design: str) -> tuple[int, float]:
    """v* and p* = max_v Pr[X·w = v], X and w uniform and independent."""
    xs, ws = DESIGNS[design]
    prod = np.multiply.outer(np.array(xs), np.array(ws)).ravel()
    vals, cnt = np.unique(prod, return_counts=True)
    i = int(np.argmax(cnt))
    return int(vals[i]), float(cnt[i] / prod.size)


def draw(design: str, m: int, n: int, s: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    xs, ws = DESIGNS[design]
    return rng.choice(xs, (m, s - 1)).astype(np.int64), rng.choice(ws, (s - 1, n)).astype(np.int64)


def instance(design: str, m: int, n: int, s: int, rng: np.random.Generator, cancel: bool = True):
    """Chosen data (A8's cancelling lane, or plain zero tails), fix2's step 1, and the design's later increments."""
    pub = r2.public("fix2", n, s, m, rng)
    co = r2.coins("fix2", m, n, s, rng)
    k = D1 + 31 * (s - 1)
    a = np.clip(np.rint(rng.normal(0, 30, (m, k))), -127, 127).astype(np.int64)
    b = np.clip(np.rint(rng.normal(0, 25, (k, n))), -127, 127).astype(np.int64)
    a[:, D1:] = 0
    v, _ = point_mass(design)
    lanes = [D1 + 31 * (t - 1) for t in range(1, s)]
    if cancel:
        for lane in lanes:
            a[:, lane] = 1
            b[lane] = -v
    x, w = draw(design, m, n, s, rng)
    n1 = co["a1"] @ pub["c"] + pub["tags"] @ pub["g"]
    inc = x[:, :, None] * w[None]
    cum = np.concatenate([np.zeros((m, 1, n), dtype=np.int64), np.cumsum(inc, axis=1)], axis=1)
    honest = r2.partials(a, b, D1, s) + n1[:, :, None] + cum.transpose(0, 2, 1)
    return {"a": a, "b": b, "x": x, "w": w, "lanes": lanes, "honest": honest}


def step_increment(inst: dict, t: int) -> np.ndarray:
    """What the adversary adds at step t: its own data's lane (it chose the data) plus X_t·w_t, no k32 chain."""
    lane = inst["lanes"][t - 1]
    a, b = inst["a"], inst["b"]
    return np.multiply.outer(a[:, lane], b[lane]) + inst["x"][:, t - 1][:, None] * inst["w"][t - 1][None]


def build_entries(design: str, inst: dict) -> tuple[np.ndarray, int, int]:
    """Model P: step 1 written; at each later step an entry whose coins give X·w = v* keeps its word (a copy), every
    other entry is rewritten as its previous word plus the step's increment."""
    h = inst["honest"]
    m, n, s = h.shape
    v, _ = point_mass(design)
    out = np.empty_like(h)
    out[:, :, 0] = h[:, :, 0]
    writes = m * n
    for t in range(1, s):
        keep = inst["x"][:, t - 1][:, None] * inst["w"][t - 1][None] == v
        out[:, :, t] = np.where(keep, out[:, :, t - 1], out[:, :, t - 1] + step_increment(inst, t))
        writes += int((~keep).sum())
    return out, writes, m * (s - 1)


def group_columns(design: str, w: np.ndarray, rng: np.random.Generator) -> tuple[list[np.ndarray], np.ndarray]:
    """Greedy groups of WARP columns, chosen once a window knowing w: seed a group, then add the column that keeps the
    most weighted steps uniform. A uniform step's weight is the share of X values that make X·w_t = v* there.
    Returns the groups and, per group and step, the X value that skips it (0 when none does)."""
    xs, _ = DESIGNS[design]
    v, _ = point_mass(design)
    n = w.shape[1]
    need = np.where(v % w == 0, v // w, 0)
    need = np.where(np.isin(need, xs), need, 0)
    weight = (need != 0) / len(xs)
    free = np.ones(n, dtype=bool)
    order = rng.permutation(n)
    groups, skip = [], []
    for seed in order:
        if not free[seed]:
            continue
        if free.sum() < WARP:
            break
        free[seed] = False
        members = [seed]
        alive = need[:, seed] != 0
        for _ in range(WARP - 1):
            cand = np.flatnonzero(free)
            eq = w[alive][:, cand] == w[alive, seed][:, None]
            score = (eq * weight[alive, seed][:, None]).sum(0)
            j = int(cand[int(np.argmax(score))])
            free[j] = False
            members.append(j)
            alive &= w[:, j] == w[:, seed]
        groups.append(np.array(members))
        skip.append(np.where(alive, need[:, seed], 0))
    return groups, np.array(skip)


def build_warps(design: str, inst: dict, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    """Model W: a 1 × WARP warp-register (row, group) is rewritten whole unless every entry is unchanged."""
    h = inst["honest"]
    m, n, s = h.shape
    groups, skip = group_columns(design, inst["w"], rng)
    cols = np.concatenate(groups)
    out = np.empty_like(h)
    out[:, :, 0] = h[:, :, 0]
    rest = np.setdiff1d(np.arange(n), cols)
    writes = m * n
    for t in range(1, s):
        out[:, :, t] = out[:, :, t - 1] + step_increment(inst, t)
        for g, members in enumerate(groups):
            if not skip[g, t - 1]:
                writes += WARP * m
                continue
            kept = inst["x"][:, t - 1] == skip[g, t - 1]
            out[np.ix_(kept, members, [t])] = out[np.ix_(kept, members, [t - 1])]
            writes += WARP * int((~kept).sum())
        writes += m * len(rest)
    return out, writes, 0


def small(seed: int = 53) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    m, n, s = 96, 256, 12
    res, det = [], {}
    for design in DESIGNS:
        inst = instance(design, m, n, s, rng)
        h = inst["honest"]
        unchanged = float(np.mean(h[:, :, 1:] == h[:, :, :-1]))
        pe, wr, pt = build_entries(design, inst)
        pw, ww, _ = build_warps(design, inst, rng)
        rows_rep = int(sum((h[:, :, t] == h[:, :, t - 1]).all(axis=1).sum() for t in range(1, s)))
        plain = instance(design, m, n, s, rng, cancel=False)["honest"]
        det[design] = {"p_star": point_mass(design)[1], "v_star": point_mass(design)[0],
                       "unchanged_entry_steps": unchanged,
                       "model_P_equal": bool(np.array_equal(pe, h)), "model_P_saving": na.saving(h.size, wr, pt),
                       "model_W_equal": bool(np.array_equal(pw, h)), "model_W_saving": na.saving(h.size, ww, 0),
                       "row_repeats": rows_rep,
                       "plain_zero_tail_unchanged": float(np.mean(plain[:, :, 1:] == plain[:, :, :-1])),
                       "m": m, "n": n, "S": s}

    def ok(name, cond, d):
        res.append({"check": name, "pass": bool(cond), "detail": d})

    for design in ("fix2", "fix2w2"):
        d = det[design]
        ok(f"A8 model P {design}: the cancelling lane leaves ~1/4 of later entry-steps unchanged, words equal, > 15% "
           "saved", d["model_P_equal"] and abs(d["unchanged_entry_steps"] - 0.25 * (s - 1) / s) < 0.03
           and d["model_P_saving"] > 0.15, d)
    ok("A8 model P fix2w13: ~1/8 unchanged, words equal, > 7% saved",
       det["fix2w13"]["model_P_equal"] and det["fix2w13"]["model_P_saving"] > 0.07, det["fix2w13"])
    ok("A8 model W: every design's warp-grain adversary reproduces the honest words",
       all(det[d]["model_W_equal"] for d in DESIGNS), {d: det[d]["model_W_saving"] for d in DESIGNS})
    ok("A8 model W fix2: column groups fixed once a window still skip > 1% of the words",
       det["fix2"]["model_W_saving"] > 0.01, det["fix2"])
    ok("uw6 and uw8: < 1% of entry-steps unchanged under A8, and no row vector repeats",
       all(det[d]["unchanged_entry_steps"] < 0.01 and det[d]["row_repeats"] == 0 for d in ("uw6", "uw8")),
       {d: det[d]["unchanged_entry_steps"] for d in ("uw6", "uw8")})
    ok("plain zero tails (no cancelling lane): no entry-step is unchanged, since X·w is never 0",
       all(det[d]["plain_zero_tail_unchanged"] == 0 for d in DESIGNS),
       {d: det[d]["plain_zero_tail_unchanged"] for d in DESIGNS})
    inst = instance("fix2", m, n, s, rng)
    bad = dict(inst, w=inst["w"].copy())
    bad["w"][0, :] = -bad["w"][0, :]
    pe, _, _ = build_entries("fix2", bad)
    ok("mutation: an adversary keeping step 2's words on the wrong columns (w_2 negated) fails equality",
       not np.array_equal(pe, inst["honest"]), "")
    return res, det


def at_scale(seed: int = 59) -> dict:
    """A8's saving as a share of stock over Llama's linears (16 layers alike), models P and W, m 64 / 512 / 8,192.
    Model W groups each linear's columns once (its w drawn once a window)."""
    rng = np.random.default_rng(seed)
    out = {}
    grouped = {}
    for design in DESIGNS:
        v, _ = point_mass(design)
        for lin, (n, k) in ns.LLAMA.items():
            s = ns.steps1l(k)
            _, w = draw(design, 1, n, s, rng)
            groups, skip = group_columns(design, w, rng)
            grouped[(design, lin)] = (w, skip, len(groups))
        for m in na.MS:
            sp = sw = st = 0.0
            for lin, (n, k) in ns.LLAMA.items():
                s = ns.steps1l(k)
                w, skip, ng = grouped[(design, lin)]
                x, _ = draw(design, m, 1, s, rng)
                zeros = 0
                for t in range(s - 1):
                    xv, xc = np.unique(x[:, t], return_counts=True)
                    for xi, c in zip(xv, xc):
                        if v % xi == 0:
                            zeros += int(c) * int((w[t] == v // xi).sum())
                sp += ns.H32 * (zeros - m * (s - 1))
                kept = sum(int((x[:, t][:, None] == skip[None, :, t]).sum()) for t in range(s - 1))
                sw += ns.H32 * WARP * kept
                st += m * n * k
            out[f"{design}/m{m}"] = {"model_P": sp / st, "model_W": sw / st}
    out["groups_uniform_steps"] = {f"{d}/{lin}": float((grouped[(d, lin)][1] != 0).sum(1).mean())
                                   for d in DESIGNS for lin in ns.LLAMA}
    return out


def costs() -> dict:
    """Per design: p*, served noise std at K 2,048 / 8,192 (m 512, n 256, de-meaned per column), coin bits a call and a
    window over Llama, and the epilogue that would remove the later steps' noise (a rank-(S − 1) product, as a share of
    stock) — needed when the std is too large to serve."""
    rng = np.random.default_rng(61)
    out = {}
    for design in DESIGNS:
        bx, bw = COIN_BITS[design]
        rec = {"v_star": point_mass(design)[0], "p_star": point_mass(design)[1]}
        for k in (2048, 8192):
            s = ns.steps1l(k)
            x, w = draw(design, 512, 256, s, rng)
            tot = r2._exact_matmul(x, w)
            rec[f"later_std_K{k}"] = float((tot - tot.mean(0)).std())
        rec["coin_bits_a_call_per_row"] = sum((ns.steps1l(k) - 1) * bx for _, k in ns.LLAMA.values()) * ns.LLAMA_LAYERS
        rec["coin_bits_a_window"] = sum(n * (ns.steps1l(k) - 1) * bw for n, k in ns.LLAMA.values()) * ns.LLAMA_LAYERS
        rec["epilogue_share_of_stock"] = (sum(n * (ns.steps1l(k) - 1) for n, k in ns.LLAMA.values())
                                          / sum(n * k for n, k in ns.LLAMA.values()))
        out[design] = rec
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="write the report here")
    args = ap.parse_args(argv)
    checks, det = small()
    sc = at_scale()
    cs = costs()
    checks.append({"check": "A8 at scale, model P: fix2 and fix2w2 > 20% of stock at every m; uw8 < 0.1%",
                   "pass": min(sc[f"{d}/m{m}"]["model_P"] for d in ("fix2", "fix2w2") for m in na.MS) > 0.2
                   and max(sc[f"uw8/m{m}"]["model_P"] for m in na.MS) < 0.001,
                   "detail": {k: v for k, v in sc.items() if "/m" in k}})
    checks.append({"check": "A8 at scale, model W: fix2 > 1% of stock at every m",
                   "pass": min(sc[f"fix2/m{m}"]["model_W"] for m in na.MS) > 0.01, "detail": ""})
    rep = {"checks": checks, "small": det, "at_scale": sc, "costs": cs,
           "periods": {f"A8_P_{d}_m8192": ns.periods(sc[f"{d}/m8192"]["model_P"], 1.044) for d in ("fix2", "fix2w2")}
           | {f"A8_W_{d}_m8192": ns.periods(sc[f"{d}/m8192"]["model_W"], 1.044) for d in ("fix2", "fix2w2")}}
    text = json.dumps(rep, indent=1, default=float)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text)
    for ch in checks:
        print(("PASS " if ch["pass"] else "FAIL ") + ch["check"])
    print(json.dumps({"small": det, "at_scale": sc, "costs": cs}, indent=1, default=float))
    return 0 if all(ch["pass"] for ch in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
