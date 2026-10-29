"""The tag's column coin G at decode: what each layout of G costs in W1, and what holding G a window does to copies.

    python g_layout.py THEORY_DIR OUT.json [--calls 32] > table.md

The lead's step 1 carries the row tag T(i) on A's lanes 24–27 against G_j on B′'s, four digits in [−3, 3] a column.
Layouts (`LAYOUTS`):

- `gpu-call`: G drawn per call on the GPU from a 32-bit coin word (v = ⌊u·2401/2³²⌋, base-7 digits minus 3) and
  stored into B′'s step-1 lanes 24–27: one u32 register write a column a call.
- `cert-call`: the certifier streams G's four digit bytes a column a call, drawn exactly uniform by rejection, and the
  kernel places them with the 4-byte cp.async the lane coins already use: no register write. The stream is unchanged,
  4 bytes a column a call.
- `window-b`: G drawn once a window after B's root and written into B′'s step-1 lanes once, as the window coins
  c_t[j] of decode's coin-free steps are: one u32 write a column a window.

Prices in W1 (a MAC 1, a register write 32) are shares of the linears' stock at m rows, Llama-3.2-1B's linears
(THEORY_DIR's `noise_sweep.LLAMA`).

Copies, on THEORY_DIR's round-4 harness (`nred_round4.lead_instance`, run unchanged): chosen zero data, so every row
of a tile is A1's identical row, with A2c's or A2r's cancel lane. G is fed through the generator the harness draws
from, either fresh each call or held for `calls` calls, and the draws are recorded so the words can be rebuilt
(checked against the harness's own within-call counts). Counted for each:

- the harness's within-call grains (entry, xrow, tilecol, column, lag1), averaged over calls. At m 64 they come from
  the harness and the rebuilt words must match them; below 64 rows the harness's tile doesn't fit, so the same draws
  are made in the same order and counted on a partial tile, with the full tile's three lanes at every step;
- `xcall_word`: a word equal to one held in its column from an earlier call, and not earlier in its own call;
- `xcall_entry`: a word equal to one its own (row, column) held in an earlier call, at any step;
- `step1_entry`: a step-1 word equal to its entry's step-1 word in an earlier call;
- `tile` and `xcall_tile`: a 16 × 8 tile of one step (16 rows, 8 columns, the least the tensor pipe skips) equal to
  any tile held earlier, within the call or from an earlier call.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

LAYOUTS = ("gpu-call", "cert-call", "window-b")
MS = (1, 16, 64)
TILE_ROWS, TILE_COLS = 16, 8
LEAD = "blc4a-t64-e7"
REG_WRITE = 32.0


class Draws:
    """The generator `lead_instance` draws from, with G replaced by `g` when given and every draw recorded."""

    def __init__(self, rng: np.random.Generator, g: np.ndarray | None, g_args: tuple):
        self.rng, self.g, self.g_args, self.out = rng, g, g_args, []

    def integers(self, low, high, size):
        if self.g is not None and (low, high, tuple(size)) == self.g_args and not self.out:
            x = self.g.copy()
        else:
            x = self.rng.integers(low, high, size)
        self.out.append(x)
        return x

    def choice(self, a, size):
        x = self.rng.choice(a, size)
        self.out.append(x)
        return x


def gpu_digits(u: np.ndarray) -> np.ndarray:
    """The GPU's G from a 32-bit coin word: v = ⌊u·2401/2³²⌋, base-7 digits minus 3, lane e = digit e."""
    v = (u.astype(np.uint64) * 2401) >> np.uint64(32)
    return np.stack([(v // 7 ** e) % 7 for e in range(4)]).astype(np.int64) - 3


def cert_digits(rng: np.random.Generator, n: int) -> np.ndarray:
    """The certifier's G: v uniform on [0, 2401) by rejection on 12-bit draws, the same digits."""
    v = np.empty(0, dtype=np.int64)
    while v.size < n:
        x = rng.integers(0, 4096, 2 * n)
        v = np.concatenate([v, x[x < 2401]])
    v = v[:n]
    return np.stack([(v // 7 ** e) % 7 for e in range(4)]).astype(np.int64) - 3


def prices(ns, steps: int, calls_window: float) -> dict:
    """G's forming in W1 a call, as a share of the linears' stock at m rows, for each layout."""
    sum_n = sum(n for n, _ in ns.LLAMA.values())
    sum_kn = sum(n * k for n, k in ns.LLAMA.values())
    out = {}
    for m in MS:
        per_call = REG_WRITE * sum_n / (m * sum_kn)
        out[f"m{m}"] = {
            "gpu-call": per_call,
            "cert-call": 0.0,
            "window-b": per_call / calls_window,
            "cert-call_loads_a_column_a_call": 1,
            "lane_coin_loads_a_column_a_call_at_least": min(steps, -(-steps * m // 64)),
            "stream_bytes_a_token_G": 4 * sum_n * ns.LLAMA_LAYERS,
        }
    return out


def words(r4, na, ns, draws: list, m: int, n: int, k: int, pattern: str) -> np.ndarray:
    """The adversary's words from the recorded draws, as `lead_instance` forms them."""
    tile, e, lanes = r4.LAYOUTS[LEAD]
    s = ns.steps1l(k)
    g, a1, b, coin = draws
    ntiles = -(-m // tile)
    tile_of = np.arange(m) // tile
    c = np.where((np.arange(s - 1) % 2 == 1) & (pattern == "a2r"), 2, 1)
    wts = np.ones(lanes, dtype=np.int64)
    beta = na.bias(na.code(n), s, True)
    d0 = na.row_tags(m) @ g + a1 @ na.code(n) - beta[None]
    rpos = np.concatenate([np.zeros((m, 1), dtype=np.int64), np.cumsum(1 + b - c[None], axis=1)], axis=1)
    cpos = np.concatenate([np.zeros((ntiles, 1, n), dtype=np.int64),
                           np.cumsum(np.einsum("l,ltsn->tsn", wts, coin), axis=1)], axis=1)
    return d0[:, None, :] + rpos[:, :, None] + cpos[tile_of]


def tiles(w: np.ndarray) -> list[bytes]:
    m, s, n = w.shape
    if m < TILE_ROWS:
        return []
    t = w[: m // TILE_ROWS * TILE_ROWS].reshape(m // TILE_ROWS, TILE_ROWS, s, n // TILE_COLS, TILE_COLS)
    t = t.transpose(0, 2, 3, 1, 4).reshape(-1, TILE_ROWS * TILE_COLS)
    return [row.tobytes() for row in np.ascontiguousarray(t)]


def within(r4, w: np.ndarray, tile: int) -> dict:
    """The harness's within-call grains, on a tile of `tile` rows (a partial tile at m < 64)."""
    m, s, n = w.shape
    per = m * n
    t4 = w.reshape(-1, tile, s, n)
    return {"entry": float((s - r4._distinct(w, 1)).sum()) / per,
            "xrow": float((tile - r4._distinct(t4, 1)).sum()) / per,
            "tilecol": float((tile * s - r4._distinct(t4.reshape(-1, tile * s, n), 1)).sum()) / per,
            "column": float((m * s - r4._distinct(w.reshape(m * s, n), 0)).sum()) / per,
            "lag1": float((np.diff(w, axis=1) == 0).sum()) / per}


def keyed(w: np.ndarray, by_row: bool) -> np.ndarray:
    """One int64 key a word: its column (and row, if `by_row`) above its value."""
    m, s, n = w.shape
    pos = np.broadcast_to(np.arange(n)[None, None, :], w.shape)
    if by_row:
        pos = pos + n * np.broadcast_to(np.arange(m)[:, None, None], w.shape)
    assert np.abs(w).max() < 2 ** 30
    return (pos.astype(np.int64) << 31) + (w + 2 ** 30)


def copies(r4, na, ns, m: int, n: int, k: int, pattern: str, calls: int, window: bool, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    g_args = (-na.G_MAX, na.G_MAX + 1, (na.TAG_LANES, n))
    g_window = rng.integers(*g_args) if window else None
    saved, rebuilt_ok = [], True
    col_held = entry_held = step1_held = np.empty(0, dtype=np.int64)
    tile_held: set[bytes] = set()
    cnt = {"xcall_word": 0, "xcall_entry": 0, "step1_entry": 0, "tile": 0, "xcall_tile": 0}
    words_total = tiles_total = step1_total = 0
    tile, e, lanes = r4.LAYOUTS[LEAD]
    s = ns.steps1l(k)
    for call in range(calls):
        d = Draws(rng, g_window, g_args)
        if m % tile == 0:
            inst = r4.lead_instance(LEAD, k, m, n, d, pattern)
        else:
            d.integers(*g_args)
            d.integers(0, 2, (m, na.TAG_LANES))
            d.integers(0, 2, (m, s - 1))
            d.choice(np.arange(-(2 ** e - 1), 2 ** e, 2), (lanes, 1, s - 1, n))
            inst = None
        w = words(r4, na, ns, d.out, m, n, k, pattern)
        sv = within(r4, w, min(m, tile))
        if inst is not None:
            rebuilt_ok &= inst["equal_to_honest"] and all(abs(sv[key] - inst["saved"][key]) < 1e-12 for key in sv)
        saved.append(sv)
        col_keys, entry_keys = keyed(w, False).ravel(), keyed(w, True).ravel()
        step1_keys = keyed(w[:, :1], True).ravel()
        if call:
            first = np.zeros(col_keys.size, dtype=bool)
            first[np.unique(col_keys, return_index=True)[1]] = True
            cnt["xcall_word"] += int((first & np.isin(col_keys, col_held)).sum())
            cnt["xcall_entry"] += int(np.isin(entry_keys, entry_held).sum())
            cnt["step1_entry"] += int(np.isin(step1_keys, step1_held).sum())
            words_total += w.size
            step1_total += m * n
        own: set[bytes] = set()
        for key in tiles(w):
            if key in tile_held:
                cnt["xcall_tile"] += 1
            elif key in own:
                cnt["tile"] += 1
            own.add(key)
            tiles_total += 1
        tile_held |= own
        col_held = np.union1d(col_held, col_keys)
        entry_held = np.union1d(entry_held, entry_keys)
        step1_held = np.union1d(step1_held, step1_keys)
    mean = {key: float(np.mean([sv[key] for sv in saved])) for key in saved[0]}
    return {"m": m, "k": k, "n": n, "pattern": pattern, "calls": calls, "window_g": window,
            "rebuilt_equal_to_harness": bool(rebuilt_ok), "within_call_saved": mean,
            "xcall_word": cnt["xcall_word"] / max(words_total, 1),
            "xcall_entry": cnt["xcall_entry"] / max(words_total, 1),
            "step1_entry": cnt["step1_entry"] / max(step1_total, 1),
            "tile": cnt["tile"] / max(tiles_total, 1), "xcall_tile": cnt["xcall_tile"] / max(tiles_total, 1),
            "tiles": tiles_total}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("theory_dir")
    ap.add_argument("out")
    ap.add_argument("--calls", type=int, default=32)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--window-s", type=float, default=600.0)
    ap.add_argument("--step-ms", type=float, default=4.49, help="the slowest decode step, for the fewest calls")
    args = ap.parse_args(argv)
    sys.path.insert(0, args.theory_dir)
    import noise_sweep as ns
    import nred_attacks as na
    import nred_round4 as r4

    calls_window = args.window_s * 1000.0 / args.step_ms
    lead_steps = 1 + -(-(2048 - 21) // 28)
    rep = {"calls_window": calls_window, "prices": prices(ns, lead_steps, calls_window)}

    u = np.random.default_rng(1).integers(0, 2 ** 32, 1 << 16, dtype=np.uint64)
    gd = gpu_digits(u)
    cd = cert_digits(np.random.default_rng(2), 1 << 16)
    rep["digits"] = {"gpu_range": [int(gd.min()), int(gd.max())], "cert_range": [int(cd.min()), int(cd.max())],
                     "gpu_tv_bound": 2401 / 2 ** 32, "cert_tv": 0.0}

    rep["copies"] = []
    seed = 0
    for k in (2048,):
        for m in MS:
            for pattern in ("a2c", "a2r"):
                for window in (False, True):
                    seed += 1
                    rep["copies"].append(copies(r4, na, ns, m, args.n, k, pattern, args.calls, window, seed))
    rep["a5"] = {key: max(v["gap_share_of_stock"] for v in val.values()) for key, val in r4.a5_lead().items()}

    checks = []

    def ok(name, cond, detail):
        checks.append({"check": name, "pass": bool(cond), "detail": detail})

    ok("the rebuilt words match the harness's honest chain and entry count in every call",
       all(c["rebuilt_equal_to_harness"] for c in rep["copies"]), None)
    ok("the GPU's and the certifier's G take the same values, [−3, 3] a digit",
       rep["digits"]["gpu_range"] == rep["digits"]["cert_range"] == [-3, 3], rep["digits"])
    ok("no 16 × 8 tile repeats, within a call or across calls, under either G",
       all(c["tile"] == 0 and c["xcall_tile"] == 0 for c in rep["copies"] if c["tiles"]),
       [(c["m"], c["pattern"], c["window_g"], c["tiles"]) for c in rep["copies"]])
    pairs = {}
    for c in rep["copies"]:
        pairs.setdefault((c["m"], c["pattern"]), {})[c["window_g"]] = c
    worst = max(abs(p[True]["within_call_saved"][key] - p[False]["within_call_saved"][key])
                / (0.1 + p[False]["within_call_saved"][key])
                for p in pairs.values() for key in p[False]["within_call_saved"])
    ok("holding G a window leaves every within-call grain within 10% (+0.1 a word an entry) of G drawn per call",
       worst < 0.1, worst)
    ok("held G is held: step-1 words repeat their entry's from an earlier call more often under the window's G",
       all(p[True]["step1_entry"] > p[False]["step1_entry"] for p in pairs.values()),
       {f"m{m}/{pat}": [p[False]["step1_entry"], p[True]["step1_entry"]] for (m, pat), p in pairs.items()})
    ok("A5 on the lead with lane coins per call saves 0 (G's timing doesn't enter it)", rep["a5"]["call"] == 0,
       rep["a5"])
    rep["checks"] = checks

    with open(args.out, "w") as fh:
        json.dump(rep, fh, indent=1)

    print(f"calls a window: {calls_window:.0f}\n")
    print("| m | gpu-call | cert-call | window-b | lane coin loads a column a call (at least) |")
    print("|---|---|---|---|---|")
    for m in MS:
        p = rep["prices"][f"m{m}"]
        print(f"| {m} | {p['gpu-call']:.4%} | {p['cert-call']:.4%} | {p['window-b']:.2e} | "
              f"{p['lane_coin_loads_a_column_a_call_at_least']} |")
    print("\n| m | pattern | G | entry | xrow | tilecol | column | xcall word | xcall entry | step-1 entry "
          "| tiles: within / across |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in rep["copies"]:
        sv = c["within_call_saved"]
        print(f"| {c['m']} | {c['pattern']} | {'window' if c['window_g'] else 'call'} | {sv['entry']:.4f} | "
              f"{sv['xrow']:.4f} | {sv['tilecol']:.4f} | {sv['column']:.4f} | "
              f"{c['xcall_word']:.4f} | {c['xcall_entry']:.4f} | {c['step1_entry']:.4f} | "
              f"{c['tile']:.4f} / {c['xcall_tile']:.4f} ({c['tiles']}) |")
    print(f"\nA5 (round 4's search, max over m): lane coins per call {rep['a5']['call']:.4f}, "
          f"a window {rep['a5']['window']:.4f}\n")
    for ch in checks:
        print(f"{'ok  ' if ch['pass'] else 'FAIL'} {ch['check']}")
    return 0 if all(ch["pass"] for ch in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
