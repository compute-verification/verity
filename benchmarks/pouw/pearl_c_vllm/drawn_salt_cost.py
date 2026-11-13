"""The honest cost of the drawn-salt forms of rev1's two debit terms (`pearl_c_debit.word_mend`) on a served Pearl+ case's
committed words, at the served device's step and at sm_120's.

`word_mend` reads the support's two ends: a dead pair (an element salt-dead in both rows) and a forced zero (a product the
adder cuts at both ends).  Their drawn-salt forms read the codes the challenge's salt formed instead:

- **a drawn cut**: a product the adder cuts on its word's chain at the drawn A′ and B̃ codes (`pearl_c_debit.cut_flags`
  along `chain_cuts`, from +0 at each promotion group's first atom);
- **a drawn dead pair**: an element whose drawn code is a code a salt-free program could write, in both rows.  The
  reference (``centre``) is the support's centre, F = 0: the forming atom from f32(α·x) with no noise product, cast.
  ``any`` also takes either end's code (`pearl_c_debit.end_codes`) as a reference, so it counts at least as many.

Each sampled tile (``--per-call`` seeded tiles with credited cells of every call whose A and B rows the case kept, from
the verifier's own tiling, `clear_audit.Verifier.refs`) is formed at its pass's salt as the verifier forms it
(`PearlC.encode` on `fast_tile`'s batch forms, the pass's epoch re-started from its coins, `manifest_epoch`).  At the
served scheme its C̃ and U are replayed and its leaf held to the committed transcript leaf where the case kept the
leaves, so the codes measured are the run's.  Per tile and device, over its credited words:

- the rule's debit and cap now (`clear_work.tile_cap`'s terms; every ``--cross-check``-th tile also through `tile_cap`
  itself, stopping on any difference), and each credited A row's debit against the row cap where the format has one;
- the drawn terms' counts, and the debit with them in `word_mend`'s place, a product of several kinds counted once:
  ``mends_centre`` (both ends' kinds, drawn cuts, centre dead pairs) and ``mends_any`` (``any`` dead pairs instead);
- the debit over the cap and the largest row's debit over its row cap, with and without the drawn terms.

sm_120: the same committed words formed at the same salt and roots under ``pearl-c-sm120-v1-h3`` (its atom, step, prices
and cap 1/1,000).  Statistics only: nothing here is a rule.

  python drawn_salt_cost.py CASE_DIR OUT_DIR [--jobs N] [--seed S] [--per-call N] [--cross-check N] [--devices h100,sm120]

Writes ``OUT_DIR/tiles.jsonl`` (a line a tile as each finishes) and ``OUT_DIR/summary.json``.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC / "benchmarks/pouw/pearl_c_vllm"))
sys.path.insert(0, str(SRC / "verity/protocols/pouw/clear"))
import fast_tile as FT  # noqa: E402
import clear_work as CW  # noqa: E402
from clear_audit import Verifier  # noqa: E402
from verity.protocols.pouw.prover.schemes import pearl_c_debit as D  # noqa: E402
from verity.protocols.pouw.prover.schemes import pearl_c_work as PW  # noqa: E402
from verity.protocols.pouw.verifier.coins import CoinCommitment  # noqa: E402
from verity.protocols.pouw.verifier.protocol import Context, Matmul, Shape, Tile  # noqa: E402
from verity.protocols.pouw.verifier.window import LIVE_SCHEME  # noqa: E402
from verity.protocols.spot.device.challenger import LiveCommitment  # noqa: E402
from verity_vllm.protocols.pouw import pearl_c as PD  # noqa: E402

LEAF_BYTES = 64
SM120_SCHEME = next(s for s in PD.SCHEMES if s.startswith("pearl-c-sm120-v1"))
#: per pass: (manifest, epoch, records, {device: scheme}, the files' blob paths by path)
PASSES: dict[int, tuple] = {}


def _same(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Two E4M3 code arrays of one value: equal, or both zeros (`pearl_c_debit.ends_agree`)."""
    return (x == y) | (((x | y) & 0x7F) == 0)


def _rows_of(path: Path, k: int, idx) -> list[bytes]:
    """Rows `idx` of a BF16 row file as the committed FP32 words' bytes (`verify_run.f32_row`)."""
    out = []
    with path.open("rb") as f:
        for i in idx:
            f.seek(2 * k * i)
            out.append((np.frombuffer(f.read(2 * k), "<u2").astype("<u4") << 16).tobytes())
    return out


def _references(formed, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per element of each formed row (α, x, β): its two ends' codes, whether they agree (salt-dead), and its centre code
    (the forming atom from f32(α·x) with zero noise products, cast)."""
    plus, minus, dead, centre = [], [], [], []
    for alpha, x, beta in formed:
        xs = np.asarray(x, np.int64)
        p, m, d = FT._ends(alpha, xs, beta, device)
        acc = FT._finite(FT._mul(np.full(len(xs), alpha, np.uint32), xs.astype(np.uint32)))
        z = np.zeros((len(xs), device.atom.k), np.int64)
        plus.append(p), minus.append(m), dead.append(d), centre.append(FT._fp8(FT._step(device.atom, acc, z, z)))
    return tuple(np.stack(v) for v in (plus, minus, dead, centre))


def _word_counts(A, B, ref_a, ref_b, device) -> dict[str, np.ndarray]:
    """Per word (A's rows by B's, row-major), the products of each kind and of each union, along every word's chain at
    the drawn codes and at both ends."""
    pa, ma, da, ca = ref_a
    pb, mb, db, cb = ref_b
    fixc_a, fixc_b = _same(A, ca), _same(B, cb)
    fixa_a = fixc_a | _same(A, pa) | _same(A, ma)
    fixa_b = fixc_b | _same(B, pb) | _same(B, mb)
    na, nb, k = A.shape[0], B.shape[0], A.shape[1]
    ia, ib = np.repeat(np.arange(na), nb), np.tile(np.arange(nb), na)
    atom, g_ = device.atom.k, device.g
    kinds = ("ends", "cut_drawn", "dead_centre", "dead_any", "mends_centre", "mends_any", "nonzero")
    out = {kind: np.zeros(len(ia), np.int64) for kind in kinds}
    acc, acc_p, acc_m = (np.zeros(len(ia), np.uint32) for _ in range(3))
    for t, s in enumerate(range(0, k, atom)):
        if t % g_ == 0:
            acc[:], acc_p[:], acc_m[:] = 0, 0, 0
        sl = slice(s, s + atom)
        a, b = A[ia, sl], B[ib, sl]
        a_p, b_p, a_m, b_m = pa[ia, sl], pb[ib, sl], ma[ia, sl], mb[ib, sl]
        ends = (da[ia, sl] & db[ib, sl]) | (FT._cut(acc_p, a_p, b_p, device) & FT._cut(acc_m, a_m, b_m, device))
        cut = FT._cut(acc, a, b, device)
        dc, dn = fixc_a[ia, sl] & fixc_b[ib, sl], fixa_a[ia, sl] & fixa_b[ib, sl]
        for kind, m in (("ends", ends), ("cut_drawn", cut), ("dead_centre", dc), ("dead_any", dn),
                        ("mends_centre", ends | cut | dc), ("mends_any", ends | cut | dn),
                        ("nonzero", ((a & 0x7F) != 0) & ((b & 0x7F) != 0))):
            out[kind] += m.sum(axis=1)
        acc, acc_p, acc_m = (FT._step(device.atom, acc, a, b), FT._step(device.atom, acc_p, a_p, b_p),
                             FT._step(device.atom, acc_m, a_m, b_m))
    return {kind: v.reshape(na, nb) for kind, v in out.items()}


def measure(scheme, record, ops, pa: list[int], pb: list[int], cross_check: bool) -> dict:
    d, n = scheme.device, record.matmul.shape.n
    A = np.array([ops.a.codes[p] for p in pa], np.int64)
    B = np.array([ops.b.codes[p] for p in pb], np.int64)
    formed_a = [(ops.a.alpha[p], ops.a.x[p], ops.a.beta[p]) for p in pa]
    formed_b = [(ops.b.alpha[p], ops.b.x[p], ops.b.beta[p]) for p in pb]
    na, nb, k = A.shape[0], B.shape[0], A.shape[1]
    flags, zeros = FT._flags(A, B, d)
    flags, zeros = flags.reshape(na, nb, -1), zeros.reshape(na, nb)
    ref_a, ref_b = _references(formed_a, d), _references(formed_b, d)
    every = flags.all(axis=1)
    elements = np.array([k if every[i].all() else int((np.repeat(every[i], d.atom.k) | ref_a[2][i]).sum())
                         for i in range(na)], np.int64)
    counts = _word_counts(A, B, ref_a, ref_b, d)
    atoms_i, zeros_i = flags.sum(axis=(1, 2)), zeros.sum(axis=1)

    def debit(rows, mends) -> Fraction:
        return D.debit_units(int(atoms_i[rows].sum()), Fraction(nb * int(elements[rows].sum()), n), scheme.prices, d,
                             zeros=int(zeros_i[rows].sum()), mends=int(mends[rows].sum()))

    cell = CW.cell_credit(scheme, record)
    cap = scheme.cap * cell * na * nb
    every_row = np.arange(na)
    out = {"words": na * nb, "products": na * nb * k, "cap": float(cap)}
    for kind, v in counts.items():
        out[kind] = int(v.sum())
        out[f"{kind}_max_word"] = int(v.max())
    out["fixed_centre_a"] = int(_same(A, ref_a[3]).sum())
    out["fixed_centre_b"] = int(_same(B, ref_b[3]).sum())
    out["elements_a"], out["elements_b"] = int(A.size), int(B.size)
    for label, mends in (("now", counts["ends"]), ("centre", counts["mends_centre"]), ("any", counts["mends_any"])):
        dt = debit(every_row, mends.sum(axis=1))
        out[f"debit_{label}"] = float(dt)
        out[f"ratio_{label}"] = float(dt / cap)
    out["drawn_share_of_cap_centre"] = float(Fraction(out["mends_centre"] - out["ends"]) / cap)
    out["drawn_share_of_cap_any"] = float(Fraction(out["mends_any"] - out["ends"]) / cap)
    if CW.row_cap(scheme):
        row_cap = scheme.cap * cell * nb * CW.row_cap_multiple(scheme, k)
        for label, mends in (("now", counts["ends"]), ("centre", counts["mends_centre"]), ("any", counts["mends_any"])):
            m_i = mends.sum(axis=1)
            out[f"row_ratio_max_{label}"] = max(float(debit(np.array([i]), m_i) / row_cap) for i in range(na))
    if cross_check:
        ref_debit, ref_cap = CW.tile_cap(scheme, record, A.tolist(), B.tolist(), formed_a, formed_b)
        if ref_debit != debit(every_row, counts["ends"].sum(axis=1)) or ref_cap != cap:
            raise AssertionError(f"tile_cap's ({ref_debit}, {ref_cap}) is not this replay's")
        out["cross_checked"] = True
    return out


def tile_job(job) -> dict:
    pi, pos, tid, rows, cols, check = job
    man, epoch, records, schemes, blobs = PASSES[pi]
    u, rec, record = epoch.workload.matmuls[pos], man["matmuls"][pos], records[pos]
    wt = man["weights"][str(u.weight)]
    k = u.shape.k
    a_bytes = _rows_of(blobs[rec["rows"]["path"]], k, rows)
    b_bytes = _rows_of(blobs[wt["rows"]["path"]], k, cols)
    tile = Tile(tuple(rows), tuple(cols))
    ctx = Context(epoch.salt, u, bytes.fromhex(rec["root_a"]), epoch.weight_roots[u.weight])
    cr, cc = record.credited(tile)
    pos_a, pos_b = {i: p for p, i in enumerate(rows)}, {j: p for p, j in enumerate(cols)}
    pa, pb = [pos_a[i] for i in cr], [pos_b[j] for j in cc]
    out = {"at": [pi, pos, tid], "weight": u.weight, "shape": [u.shape.m, k, u.shape.n], "rows": len(cr), "cols": len(cc)}
    for i, (name, scheme) in enumerate(schemes.items()):
        noise = scheme.noise(ctx)
        ops = scheme.encode(ctx, noise, a_bytes, b_bytes, tile)
        if ops is None:
            out[name] = {"formed": False}
            continue
        got = {"formed": True}
        if i == 0:
            leaves = blobs.get(man["leaves"]["path"])
            if leaves is not None:
                with leaves.open("rb") as f:
                    f.seek(LEAF_BYTES * tid)
                    committed = f.read(LEAF_BYTES)
                got["leaf_matches"] = scheme.digest(ctx, noise, scheme.checked(ctx, ops)) == committed
        got.update(measure(scheme, record, ops, pa, pb, check))
        out[name] = got
    return out


def _coins(man: dict):
    live = isinstance(man.get("coins"), dict) and man["coins"].get("scheme") == LIVE_SCHEME
    return (LiveCommitment if live else CoinCommitment).from_json(man["coins"])


def summarize(lines: list[dict], names: list[str]) -> dict:
    doc = {}
    for name in names:
        got = [ln[name] for ln in lines if ln.get(name, {}).get("formed")]
        agg = {"tiles": len(got), "not_formed": sum(1 for ln in lines if not ln.get(name, {}).get("formed")),
               "leaf_checked": sum(1 for g in got if "leaf_matches" in g),
               "leaf_matches": sum(1 for g in got if g.get("leaf_matches")),
               "cross_checked": sum(1 for g in got if g.get("cross_checked"))}
        for key in ("words", "products", "nonzero", "ends", "cut_drawn", "dead_centre", "dead_any", "mends_centre",
                    "mends_any", "fixed_centre_a", "fixed_centre_b", "elements_a", "elements_b"):
            agg[key] = sum(g[key] for g in got)
        for key in ("cut_drawn", "dead_centre", "dead_any", "mends_centre", "mends_any"):
            agg[f"{key}_max_word"] = max((g[f"{key}_max_word"] for g in got), default=0)
        cap = sum(g["cap"] for g in got)
        agg["cap"] = cap
        for label in ("now", "centre", "any"):
            agg[f"debit_{label}"] = sum(g[f"debit_{label}"] for g in got)
            agg[f"pooled_ratio_{label}"] = agg[f"debit_{label}"] / cap if cap else None
            worst = max(range(len(got)), key=lambda i: got[i][f"ratio_{label}"]) if got else None
            agg[f"ratio_max_{label}"] = got[worst][f"ratio_{label}"] if got else None
            agg[f"ratio_max_{label}_at"] = [ln for ln in lines if ln.get(name, {}).get("formed")][worst]["at"] if got else None
            agg[f"tiles_over_cap_{label}"] = sum(1 for g in got if g[f"ratio_{label}"] > 1)
            if got and f"row_ratio_max_{label}" in got[0]:
                agg[f"row_ratio_max_{label}"] = max(g[f"row_ratio_max_{label}"] for g in got)
                agg[f"tiles_row_over_{label}"] = sum(1 for g in got if g[f"row_ratio_max_{label}"] > 1)
        for label in ("centre", "any"):
            agg[f"drawn_share_of_cap_{label}_max"] = max((g[f"drawn_share_of_cap_{label}"] for g in got), default=None)
            agg[f"drawn_share_of_cap_{label}_pooled"] = (agg[f"mends_{label}"] - agg["ends"]) / cap if cap else None
        doc[name] = agg
    return doc


def main() -> int:
    case, out_dir = Path(sys.argv[1]), Path(sys.argv[2])

    def opt(name: str, default):
        return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default

    jobs = opt("--jobs", len(os.sched_getaffinity(0)))
    seed, per_call, every = opt("--seed", 20261113), opt("--per-call", 1), opt("--cross-check", 8)
    devices = opt("--devices", "h100,sm120").split(",")
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = json.loads((case / "parts.json").read_text())
    files = parts["evidence"]["files"]
    rng = np.random.default_rng(seed)
    picks, names = [], None
    for pi, man in enumerate(parts["manifests"]):
        served = PD.scheme_of(man["scheme"], **{k: tuple(v) if isinstance(v, list) else v
                                                for k, v in man.get("instance", {}).items()})
        schemes = {served.device.name: served}
        if "sm120" in devices and served.device.name != "sm120":
            schemes["sm120"] = PD.scheme_of(SM120_SCHEME, **{k: tuple(v) if isinstance(v, list) else v
                                                            for k, v in man.get("instance", {}).items()})
        schemes = {n_: s for n_, s in schemes.items() if n_ in devices}
        names = names or list(schemes)
        matmuls = tuple(Matmul(u["index"], Shape(u["m"], u["k"], u["n"]), u["weight"]) for u in man["matmuls"])
        epoch, _ = PD.manifest_epoch(man, _coins(man))
        records = [PW.UnitRecord(mm, u["real_rows"], u["k_kept"], tuple(u["excluded_a"]),
                                 tuple(man["weights"][str(mm.weight)]["excluded_b"]), tuple(u.get("voluntary_a", ())),
                                 tuple(man["weights"][str(mm.weight)].get("voluntary_b", ())))
                   for mm, u in zip(matmuls, man["matmuls"])]
        blobs = {path: case / "blobs" / h for path, h in files[pi].items()}
        PASSES[pi] = (man, epoch, records, schemes, blobs)
        by_pos: dict[int, list] = {}
        for ref in Verifier(CW._TileRuleView(served), epoch).refs:
            cr, cc = records[ref.matmul].credited(ref.tile)
            if cr and cc:
                by_pos.setdefault(ref.matmul, []).append(ref)
        for pos, rec in enumerate(man["matmuls"]):
            wt = man["weights"][str(rec["weight"])]
            if rec["rows"]["path"] not in blobs or wt["rows"]["path"] not in blobs or pos not in by_pos:
                continue
            for r in rng.choice(len(by_pos[pos]), min(per_call, len(by_pos[pos])), replace=False):
                ref = by_pos[pos][int(r)]
                picks.append((pi, pos, ref.id, list(ref.tile.rows), list(ref.tile.cols)))
    FT.install(next(iter(PASSES[0][3].values())))
    work = [(*p, i % every == 0) for i, p in enumerate(picks)]
    print(f"{len(work)} tiles over {len(PASSES)} passes, devices {names}, {jobs} jobs", flush=True)
    lines = []
    with (out_dir / "tiles.jsonl").open("w") as f, multiprocessing.get_context("fork").Pool(jobs) as pool:
        for line in pool.imap_unordered(tile_job, work):
            lines.append(line)
            f.write(json.dumps(line) + "\n")
            f.flush()
            print(f"{len(lines)}/{len(work)} {json.dumps(line)}", flush=True)
    doc = {"case": str(case), "seed": seed, "per_call": per_call, "served": [m["scheme"] for m in parts["manifests"]],
           "devices": summarize(lines, names)}
    (out_dir / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")
    print("SUMMARY " + json.dumps(doc), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
