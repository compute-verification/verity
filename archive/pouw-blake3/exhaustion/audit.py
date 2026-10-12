#!/usr/bin/env python3
"""The CPU audit of a salt-to-deadline window (`window.py`'s OUT directory), and the window's certified fraction of the cards'
FP8 capacity.

1. The clock: the salt is derive(source, SEED_DOMAIN, {t_s, the cards' UUIDs}) through `Epoch.unverified` on B's root, as
   start.json records it; t_d = t_s + the duration; every card's close was sent and received by t_d, and the window root
   (recomputed from the closes) was computed by t_d.
2. The draw: re-derived from draws.json's `source` (drawn after the window root was written) over the window root and the
   salt, uniform over every credited tile, and it must be the tiles the cards opened.
3. Each drawn tile: its GEMM's entry tagged(ENTRY, card, i, root_A, root_T) opens to its card's root (RFC 6962 path), B's
   dumped tree has the window's B root, and `verity.core.protocols.pouw.audit.Verifier(scheme, epoch).check_tile` accepts it, with the GEMM's
   one-unit epoch (the window's salt, B's root, matmul index card << 32 | i), its transcript (root_A, the tile tree's
   commitment) and the opening read from the dumps (tree_A, tree_T, the tile leaves, tree_B) and the fill's rows (A's set
   i mod 2, B).  The verifier recomputes the tile through the scheme on the CPU (about 11 s per 1,024 of k).
4. The controls, both of which must reject: `tampered`, the card's re-run of draw 0's GEMM with its first message-digest byte
   flipped between the product and the hashes, checked at the tile whose leaf the flip changed against the transcript that
   re-run commits; `wrong salt`, draw 0's honest opening checked under the epoch of another salt (a transcript computed before
   or apart from this window's salt).
5. The fraction: certified = (1 − γ)(1 − ε_s)·N, N the credited GEMMs; capacity = Σ_cards duration / t_base, t_base the
   card's cuBLASLt FP8 time per GEMM at the shape (the fastest `lt*` baseline's median, harness bench.py, before and after the
   window, averaged); fraction = certified / capacity.  γ is the Lean-pinned price twin `price_twins.lookup` publishes for
   pearl-c-sm120/v1 on the cast-8.72 path at 8,192³ (`../harness/price_twins.json`), never typed in; ε_s is the
   least (on a grid of 10⁻⁶) with (1 − ε_s)^t ≤ δ_s = 1/20 for the window's t draws, and t ≥ 59 (ε_s ≤ 1/20) is required.

  python audit.py OUT [--bench-before DIR] [--bench-after DIR] [--jobs J] [--commit SHA]

DIR holds card<g>/bench.json per card.  Writes OUT/audit.json and prints one line per check; exit 0 only if every check holds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import re
import statistics
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL = HERE.parent / "pearl_c_sm120"
HARNESS = HERE.parent / "harness"


def _load(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


W = _load("exhaustion_window", HERE / "window.py")
V = _load("pearl_c_sm120_verify", KERNEL / "verify.py")
PT = _load("pouw_harness_price_twins", HARNESS / "price_twins.py")

from verity.ml.commitments import Commitment  # noqa: E402
from verity.core.protocols.pouw.audit import TileOpening, TileRef, Transcript, Verifier  # noqa: E402

EPS_S = DELTA_S = Fraction(1, 20)
STATEMENT, PATH = "pearl-c-sm120/v1", "cast-8.72"
LT = re.compile(r"^lt\d*_")


def draws_needed(eps: Fraction = EPS_S, delta: Fraction = DELTA_S) -> int:
    """The least t with (1 − ε_s)^t ≤ δ_s."""
    t = max(1, math.ceil(math.log(delta) / math.log(1 - eps)))
    while (1 - eps) ** t > delta:
        t += 1
    while t > 1 and (1 - eps) ** (t - 1) <= delta:
        t -= 1
    return t


def eps_for(t: int, delta: Fraction = DELTA_S, den: int = 10**6) -> Fraction:
    """The least ε_s on a grid of 1/den with (1 − ε_s)^t ≤ δ_s: what t draws establish (at most 1/20 from t = 59)."""
    e = Fraction(math.ceil((1 - float(delta) ** (1 / t)) * den), den)
    while e > Fraction(1, den) and (1 - (e - Fraction(1, den))) ** t <= delta:
        e -= Fraction(1, den)
    while (1 - e) ** t > delta:
        e += Fraction(1, den)
    return e


def gamma(n: int, k: int) -> dict:
    """γ for the window's statement at (n, k), as `price_twins.lookup` publishes it (the larger of its two FADD prices)."""
    found = PT.lookup(PT.load(), STATEMENT, PATH, n, k)
    if found["published"] is None:
        return {"gamma": None, "missing": found["missing"]}
    pick = found["by_price"][found["published_at"]]
    return {"gamma": found["published"], "pin": pick["pin"], "status": pick["status"], "fadd": found["published_at"],
            "source": pick["source"], "type_hash": pick["type_hash"], "cap": found["cap"]}


def baseline(bench: dict, label: str) -> dict:
    """A bench.py result's plain baselines at `label`: each one's median ms per call (its graph items, else every item),
    the fastest cuBLASLt (`lt*`) and the fastest of all."""
    by: dict[str, dict[str, list[float]]] = {}
    for it in bench.get("items", []):
        key = it.get("key", [])
        if len(key) >= 4 and key[0] == "baseline" and key[1] == label:
            by.setdefault(key[3], {}).setdefault(it.get("method", ""), []).append(it["ms_per_call"])
    med = {name: statistics.median(ms.get("graph") or [x for v in ms.values() for x in v]) for name, ms in by.items()}
    lt = {n: v for n, v in med.items() if LT.match(n)}
    pick = lambda d: (min(d, key=d.get), d[min(d, key=d.get)]) if d else (None, None)  # noqa: E731
    return {"exit": bench.get("exit"), "cublaslt": pick(lt), "fastest": pick(med), "medians_ms": med}


def fraction(counts: dict[int, int], base_ms: dict[int, float], duration_s: float, gamma_: Fraction | None,
             eps: Fraction = EPS_S) -> dict:
    """The certified fraction: (1 − γ)(1 − ε_s)·N over Σ_cards duration / t_base (GEMMs of the baseline in the window)."""
    capacity = sum(Fraction(duration_s) * 1000 / Fraction(base_ms[g]) for g in counts)
    n = sum(counts.values())
    out = {"credited_gemms": n, "per_card": {str(g): {"gemms": counts[g], "base_ms": base_ms[g],
                                                         "capacity_gemms": float(Fraction(duration_s) * 1000 / Fraction(base_ms[g]))}
                                            for g in sorted(counts)},
           "capacity_gemms": float(capacity), "uncertified_fraction": float(n / capacity) if capacity else None,
           "slowdown": float(capacity / n) if n else None, "eps_s": str(eps)}
    if gamma_ is not None and capacity:
        cert = (1 - gamma_) * (1 - eps) * n
        out.update(certified_gemms=float(cert), fraction=float(cert / capacity), gamma=str(gamma_))
    return out


def _tile_of(sch, shape, tile_id: int):
    return sch.tiles(W.mshape(shape))[tile_id]


def check(job: dict) -> dict:
    """One tile through `Verifier.check_tile` from the dumps (module docstring, step 3); `job` names the dumps and the epoch."""
    out, shape = Path(job["out"]), tuple(job["shape"])
    m, n, k = shape
    sch = W.scheme(job["hashing"])
    card, i, tid = job["card"], job["i"], job["tile"]
    mm = W.matmul(card, i, shape)
    try:
        man = job["manifest"]
        base = out / job["dir"]
        files = {name: V.load(base, spec) for name, spec in man["files"].items()}
        tree_b = V.load(out / job["weights_dir"], job["weights"])
        tile = _tile_of(sch, shape, tid)
        a_rows = W.rows(W.a_seed(card, i & 1), tile.rows[0], len(tile.rows), k)
        b_rows = W.rows(W.SEED_B, tile.cols[0], len(tile.cols), k)
        count = man["tiles"]
        opening = TileOpening(TileRef(tid, 0, tile),
                              tuple(V.opening(files["tree_a"], m, r, row) for r, row in zip(tile.rows, a_rows)),
                              tuple(V.opening(tree_b, n, c, row) for c, row in zip(tile.cols, b_rows)),
                              V.opening(files["tree_t"], count, tid, bytes(files["leaves"][32 * tid:32 * tid + 32])))
        transcript = Transcript((bytes.fromhex(man["root_a"]),), Commitment(bytes.fromhex(man["root_t"]), count), ())
        salt = bytes.fromhex(job["salt"])
        v = Verifier(sch, W.gemm_epoch(salt, bytes.fromhex(job["root_b"]), mm)).check_tile(transcript, opening)
        return {**{x: job[x] for x in ("label", "card", "i", "tile")}, "ok": v.ok, "reason": v.reason}
    except Exception as ex:  # noqa: BLE001 - a tile that can't be opened is rejected, with the reason
        return {**{x: job[x] for x in ("label", "card", "i", "tile")}, "ok": False, "reason": f"{type(ex).__name__}: {ex}"}


def changed_tiles(a: bytes, b: bytes) -> list[int]:
    return [t for t in range(len(a) // 32) if a[32 * t:32 * t + 32] != b[32 * t:32 * t + 32]]


def audit(out: Path, before: Path | None, after: Path | None, jobs: int, commit: str) -> dict:
    start = json.loads((out / "start.json").read_text())
    win = json.loads((out / "window.json").read_text())
    dr = json.loads((out / "draws.json").read_text())
    shape = (start["shape"]["m"], start["shape"]["n"], start["shape"]["k"])
    checks: list[dict] = []

    def note(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    hashing = start.get("hashing", "h1")
    sch = W.scheme(hashing)
    note("scheme", start.get("scheme", sch.name) == sch.name, sch.name)
    uuids = [c["uuid"] for c in start["cards"]]
    root_b = bytes.fromhex(start["root_b"])
    salt = W.salt_of(W.window_seed(bytes.fromhex(start["source"]), start["t_s_ns"], uuids), root_b, shape)
    t_s, t_d = start["t_s_ns"], start["t_d_ns"]
    note("salt", salt.hex() == start["salt"], f"derive(source {start['source'][:16]}…, t_s {start['t_s']}, {len(uuids)} cards)")
    note("deadline", t_d == t_s + int(start["duration_s"] * 1e9), f"t_s {start['t_s']}, t_d {start['t_d']}")
    closes = win["cards"]
    on_time = [c for c in closes if c["t_close_ns"] <= t_d and c["t_recv_ns"] <= t_d]
    note("closes by t_d", len(on_time) == len(closes) == len(uuids) and not win["late"],
         f"{len(on_time)} of {len(uuids)} cards closed by t_d; late {len(win['late'])}")
    root = W.window_root(salt, t_s, t_d, closes)
    note("window root", root.hex() == win["root"] and win["t_root_ns"] <= t_d,
         f"{win['root'][:16]}… at {win['t_root']} (t_d {start['t_d']})")
    tiles = dr["tiles_per_gemm"]
    note("tiles per GEMM", tiles == len(sch.tiles(W.mshape(shape))), str(tiles))
    picked = W.draws(bytes.fromhex(dr["source"]), root, salt, [(c["card"], c["count"]) for c in closes], tiles, len(dr["draws"]))
    note("draw", picked == dr["draws"] and dr["t_source_ns"] >= win["t_root_ns"],
         f"{len(picked)} tiles from source {dr['source'][:16]}… drawn after the root")
    need = draws_needed()
    eps = eps_for(len(picked)) if picked else EPS_S
    note("draws enough", len(picked) >= need, f"t = {len(picked)} (ε_s = 1/20 at δ_s = 1/20 needs t ≥ {need}): ε_s = {eps} "
         f"({float(eps):.4%}) at δ_s = {DELTA_S}")
    by_card = {c["card"]: c for c in closes}
    weights = {}
    for c in closes:
        wdir = out / "weights" / f"card{c['card']}"
        opened = next((o for o in dr["opened"] if o.get("card") == c["card"]), {})
        spec = opened.get("weights")
        good = bool(spec) and (wdir / spec["path"]).exists() and spec["root"] == start["root_b"]
        weights[c["card"]] = (f"weights/card{c['card']}", spec)
        if any(d["card"] == c["card"] for d in picked):
            note(f"card {c['card']} B tree", good, spec["root"][:16] if spec else "no dump")
    work = []
    for j, d in enumerate(picked):
        rel = f"draws/card{d['card']}-gemm{d['i']}"
        mpath = out / rel / "manifest.json"
        if not mpath.exists():
            note(f"draw {j}", False, f"no dump {rel}")
            continue
        man = json.loads(mpath.read_text())
        e = W.entry(d["card"], d["i"], bytes.fromhex(man["root_a"]), bytes.fromhex(man["root_t"]))
        got = W.root_from_path(e, d["i"], by_card[d["card"]]["count"], [bytes.fromhex(h) for h in man["path"] or []])
        note(f"draw {j} entry", got is not None and got.hex() == by_card[d["card"]]["root"],
             f"card {d['card']} GEMM {d['i']}: root_A {man['root_a'][:16]}…, root_T {man['root_t'][:16]}…")
        wd, ws = weights[d["card"]]
        work.append({"label": f"draw {j}", "out": str(out), "shape": shape, "card": d["card"], "i": d["i"], "tile": d["tile"],
                     "dir": rel, "manifest": man, "weights_dir": wd, "weights": ws, "salt": salt.hex(), "root_b": start["root_b"],
                     "hashing": hashing})
    controls = []
    tam = out / "controls" / "tampered" / "manifest.json"
    if work and tam.exists():
        tman = json.loads(tam.read_text())
        honest = V.load(out / work[0]["dir"], work[0]["manifest"]["files"]["leaves"])
        flipped = V.load(out / "controls" / "tampered", tman["files"]["leaves"])
        moved = changed_tiles(bytes(honest), bytes(flipped))
        note("tampered control changed a leaf", len(moved) >= 1, f"tiles {moved[:4]}")
        if moved:
            controls.append({**work[0], "label": "tampered", "tile": moved[0], "dir": "controls/tampered", "manifest": tman,
                             "i": tman["i"]})
    else:
        note("tampered control", False, "no tampered dump")
    if work:
        other = W.salt_of(os.urandom(32), root_b, shape)
        controls.append({**work[0], "label": "wrong salt", "salt": other.hex()})
    unique = {}
    for w in work:
        unique.setdefault((w["card"], w["i"], w["tile"]), w)
    todo = list(unique.values()) + controls
    if jobs <= 1 or len(todo) <= 1:
        results = [check(x) for x in todo]
    else:
        with multiprocessing.get_context("fork").Pool(min(jobs, len(todo))) as pool:
            results = pool.map(check, todo, chunksize=1)
    verdict = {(r["card"], r["i"], r["tile"]): r for r in results[:len(unique)]}
    tiles_out = []
    for w in work:
        r = verdict[(w["card"], w["i"], w["tile"])]
        tiles_out.append({**r, "label": w["label"]})
        note(f"{w['label']} tile", r["ok"], f"card {w['card']} GEMM {w['i']} tile {w['tile']}: "
             + ("accepted" if r["ok"] else r["reason"]))
    ctl = results[len(unique):]
    for r in ctl:
        note(f"control {r['label']}", not r["ok"], f"card {r['card']} GEMM {r['i']} tile {r['tile']}: "
             + (f"rejected ({r['reason']})" if not r["ok"] else "ACCEPTED"))
    note("both controls", len(ctl) == 2 and not any(r["ok"] for r in ctl), f"{len(ctl)} run")
    base = {}
    for when, d in (("before", before), ("after", after)):
        if d is None:
            continue
        for c in closes:
            p = d / f"card{c['card']}" / "bench.json"
            if p.exists():
                base.setdefault(c["card"], {})[when] = baseline(json.loads(p.read_text()), f"m{shape[0]}-n{shape[1]}-k{shape[2]}")
    base_ms, base_fast = {}, {}
    for c in closes:
        b = base.get(c["card"], {})
        lt = [x["cublaslt"][1] for x in b.values() if x["cublaslt"][1] is not None and x["exit"] == 0]
        fast = [x["fastest"][1] for x in b.values() if x["fastest"][1] is not None and x["exit"] == 0]
        if lt:
            base_ms[c["card"]] = statistics.mean(lt)
            base_fast[c["card"]] = statistics.mean(fast)
    counts = {c["card"]: c["count"] for c in closes}
    g = gamma(shape[1], shape[2])
    note("gamma", g["gamma"] is not None, f"{g['gamma']} ({g.get('pin')}, {g.get('status')}, {g.get('source')})" if g["gamma"]
         else f"no published price twin at n {shape[1]}, k {shape[2]}: {g.get('missing')}")
    frac = None
    if counts and set(base_ms) == set(counts):
        gm = None if g["gamma"] is None else Fraction(g["gamma"])
        frac = fraction(counts, base_ms, start["duration_s"], gm, eps)
        frac["vs_fastest_plain"] = fraction(counts, base_fast, start["duration_s"], gm, eps)
    note("baseline", frac is not None, f"cuBLASLt before/after on {len(base_ms)} of {len(counts)} cards")
    ok = all(c["ok"] for c in checks)
    return {"schema": W.SCHEMA + "/audit", "verifier_commit": commit, "ok": ok, "scheme": sch.name, "shape": shape,
            "t_s": start["t_s"], "t_d": start["t_d"], "source": start["source"], "salt": start["salt"], "window_root": win["root"],
            "draw_source": dr["source"], "gemms": counts, "tiles": tiles_out, "controls": ctl, "baseline": base,
            "gamma": g, "fraction": frac, "checks": checks}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--bench-before", type=Path)
    p.add_argument("--bench-after", type=Path)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--commit", default=os.environ.get("RESEARCH_SOURCE_SHA", "unknown"))
    a = p.parse_args(argv)
    res = audit(a.out, a.bench_before, a.bench_after, a.jobs, a.commit)
    blob = json.dumps(res, indent=1, default=str) + "\n"
    (a.out / "audit.json").write_text(blob)
    for c in res["checks"]:
        print(f"{'PASS' if c['ok'] else 'FAIL'} {c['check']}: {c['detail']}", flush=True)
    f = res["fraction"] or {}
    print(f"{'ACCEPT' if res['ok'] else 'REJECT'} window {res['window_root'][:16]}… ({res['t_s']} to {res['t_d']}): "
          f"{f.get('credited_gemms')} GEMMs credited, capacity {f.get('capacity_gemms')}, certified fraction "
          f"{f.get('fraction')} (uncertified {f.get('uncertified_fraction')}, slowdown {f.get('slowdown')}); "
          f"audit.json sha256:{hashlib.sha256(blob.encode()).hexdigest()}; verifier {a.commit}", flush=True)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
