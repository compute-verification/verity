"""The H100 noise revision's honest replay: a served Pearl+ case's sampled tiles (`drawn_salt_cost.py`'s, HONESTCOST's
`r20261113-133854-0092`) formed under candidate noise revisions, and the drawn-salt debit of each at three salts.

Each setting is (δ, G, layout) at a salt:

- **δ**, the noise scale: forming v1 (`pearl_c.form_v1`, `s5_stats`) with σ = δ·ρ in β_row and the headroom scaled the same
  way, α = 448/max(s + 4δρ, 2^−32), both FP32 nearest-even in v1's order: t = s + (4δ)·ρ, β = ((δ·ρ)·α)·line.  δ = 1 is
  v1, and the replay asserts it equals the served scheme's `encode` on every tile at every salt.
- **G**, atoms per promotion group: the H100 record (`pearl_c_device.H100`) with ``g`` replaced, its own credit, cap and
  debit prices (`clear_work.cell_credit`, `pearl_c_debit.atom_units`), so G = 1 credits four times the promotion adds.
- **layout**: ``served``, or ``outlier-last``, k permuted offline so that each linear's channels sit in ascending order of
  their mean |x|/rms over rows of *other* passes (out of sample; a pass whose class no other pass kept falls back to its
  own rows, and the tile line says so).  The permutation is one the model admits: one for the residual stream (qkv and
  gate_up, every layer: their inputs are RMSNorm of one stream), one per layer for down_proj's intermediate dimension,
  and for o_proj one head_dim permutation per KV group applied within each of its heads.  A and B rows are permuted
  alike, so each word's product is the same.  The noise lines stay those of the served rows' seeds (a permuted model's
  rows have other digests, so other but identically drawn noise).
- **salt**: 0 is the pass's own (where the replay's δ = 1 leaf is held to the committed transcript leaf); 1 and 2 are
  SHA-512("noise-replay/salt" ‖ salt ‖ i), truncated to the salt's length.

Per setting and tile, over its credited words (`drawn_salt_cost.measure`'s terms, mend 2's drawn-salt debit being
``mends`` = both ends' kinds ∪ drawn cuts ∪ centre dead pairs, one MAC each, added to the rule's debit):

- ``cut``: products the 14-bit adder cuts on the word's chain at the drawn codes (`pearl_c_debit.cut_flags` along
  `chain_cuts`, from +0 at each group's first atom), attributed by what set the cut product's alignment exponent:
  ``zero`` (the product is 0: a code that rounded to ±0), else the largest product of its own atom (``own_*``) or the
  running sum from the group's earlier atoms (``earlier_*``, the group's largest earlier product), each by whether that
  product's A or B element is signal-large (|x| > 4σ = 4δρ of its row: ``a``, ``b``, ``ab``) or noise-scale (``n``);
- ``dead``: centre dead pairs (both elements' drawn codes equal the F = 0 code);
- the debit with ``mends`` over the tile cap and the largest row's over its row cap;
- precision: the e4m3 rounding error of every formed element in x's units, (code − D)/α, over its row's rms; and the
  codes saturated at ±448, and the rows off v1's noise floor ρ·α ≥ 1.

Statistics only: nothing here is a rule.

  python noise_replay.py CASE_DIR OUT_DIR [--jobs N] [--seed S] [--per-call N] [--cross-check N] [--deltas 1,2,3,4]
                         [--gs 4,2,1] [--layouts served,outlier-last] [--salts 3] [--limit N]

Writes ``OUT_DIR/tiles.jsonl`` (a line a tile and salt as each finishes) and ``OUT_DIR/summary.json``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import multiprocessing
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC / "benchmarks/pouw/pearl_c_vllm"))
sys.path.insert(0, str(SRC / "verity/protocols/accounting/pouw/prover/clear"))
import fast_tile as FT  # noqa: E402
import clear_work as CW  # noqa: E402
from clear_audit import Verifier  # noqa: E402
from verity.circuits.silicon import fp32  # noqa: E402
from verity.protocols.accounting.pouw.prover.schemes import pearl_c as C  # noqa: E402
from verity.protocols.accounting.pouw.prover.schemes import pearl_c_debit as D  # noqa: E402
from verity.protocols.accounting.pouw.prover.schemes import pearl_c_work as PW  # noqa: E402
from verity.protocols.accounting.pouw.prover.schemes import pearl_kw as P  # noqa: E402
from verity.protocols.accounting.pouw.verifier.coins import CoinCommitment  # noqa: E402
from verity.protocols.accounting.pouw.verifier.protocol import Context, Matmul, Shape, Tile  # noqa: E402
from verity.protocols.accounting.pouw.verifier.window import LIVE_SCHEME  # noqa: E402
from verity.protocols.proofs.spot.device.challenger import LiveCommitment  # noqa: E402
from verity_vllm.protocols.pouw import pearl_c as PD  # noqa: E402

LEAF_BYTES = 64
LAYOUTS = ("served", "outlier-last")
#: 614b's model, Llama 3.2 1B: (k, n) of qkv, o, gate_up and down, and o_proj's input as heads x head_dim
SHAPES = {(2048, 3072): "qkv", (2048, 2048): "o", (2048, 16384): "gate_up", (8192, 2048): "down"}
HEADS, HEAD_DIM, Q_PER_KV = 32, 64, 4
CAUSES = ("zero",) + tuple(f"{w}_{c}" for w in ("own", "earlier") for c in ("n", "a", "b", "ab"))
FP8_VALUE = np.array([np.nan if c & 0x7F == 0x7F else P.fp8_to_f32(c) for c in range(256)], np.float64)
#: per pass: (manifest, epoch, records, served scheme, the files' blob paths by path)
PASSES: dict[int, tuple] = {}
#: per (pass, matmul position): the outlier-last permutation of k, and where its score came from
PERMS: dict[tuple[int, int], tuple[np.ndarray, str]] = {}
OPTS: dict = {}


def _same(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Two E4M3 code arrays of one value: equal, or both zeros (`pearl_c_debit.ends_agree`)."""
    return (x == y) | (((x | y) & 0x7F) == 0)


def _words(path: Path, k: int, idx) -> tuple[list[bytes], np.ndarray]:
    """Rows `idx` of a BF16 row file: the committed FP32 words' bytes (`verify_run.f32_row`) and the words."""
    out = []
    with path.open("rb") as f:
        for i in idx:
            f.seek(2 * k * i)
            out.append(np.frombuffer(f.read(2 * k), "<u2").astype("<u4") << 16)
    return [w.tobytes() for w in out], np.stack(out).astype(np.uint32)


def _f32(word: int) -> int:
    return P.f32_bits(P.f32(float(word)))


def form(x: np.ndarray, stats, e_rows, f: np.ndarray, delta: int, device) -> dict:
    """Forming v1 at noise scale δ on rows of FP32 words `x` (r, k), each row's (s, ρ) from `s5_stats`: α, β, the
    pre-cast atom words D and the codes."""
    four, dw = _f32(4 * delta), _f32(delta)
    r, k = x.shape
    alpha, beta, codes, pre = [], [], np.empty((r, k), np.int64), np.empty((r, k), np.uint32)
    for i, ((s, rho), e_row) in enumerate(zip(stats, e_rows)):
        t = C._check_finite(fp32.add(s, fp32.mul(four, rho)))
        al = fp32.div(C.S5_QMAX, max(t, C.S5_FLOOR))
        be = C._check_finite(fp32.mul(fp32.mul(fp32.mul(dw, rho), al), C.S5_LINE))
        ep = np.array(C.eprime(be, e_row), np.int64)
        acc = FT._finite(FT._mul(np.full(k, al, np.uint32), x[i]))
        pre[i] = FT._step(device.atom, acc, np.broadcast_to(ep, (k, C.SLICE)), f)
        codes[i] = FT._fp8(pre[i])
        alpha.append(al), beta.append(be)
    return {"alpha": alpha, "beta": beta, "codes": codes, "pre": pre, "x": x}


def precision(formed: dict, stats) -> dict:
    """The e4m3 rounding error in x's units over each row's rms, summed squared; saturated codes; rows off ρ·α ≥ 1."""
    x = formed["x"].view(np.float32).astype(np.float64)
    rms = np.sqrt((x * x).mean(axis=1))
    alpha = np.array([P.f32_of_bits(a) for a in formed["alpha"]])
    err = (FP8_VALUE[formed["codes"]] - formed["pre"].view(np.float32).astype(np.float64)) / alpha[:, None]
    live = rms > 0
    rel = err[live] / rms[live, None]
    floor = sum(1 for (_, rho), a in zip(stats, formed["alpha"]) if P.f32_of_bits(rho) * P.f32_of_bits(a) < P.SIGMA_MIN)
    return {"err2": float((rel * rel).sum()), "n": int(rel.size), "saturated": int(((formed["codes"] & 0x7F) == 0x7E).sum()),
            "off_floor": floor}


def _ends(alpha: int, x: np.ndarray, beta: int, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`pearl_c_debit`'s support ends on one formed row: each element's codes from f32(α·x) over [c_β]³² and
    [±11/4]³², and whether they agree (salt-dead)."""
    k, atom = x.shape[0], device.atom.k
    acc = FT._finite(FT._mul(np.full(k, alpha, np.uint32), x.astype(np.uint32)))
    cb = np.full((k, atom), D.flat_code(beta), np.int64)
    c = [FT._fp8(FT._step(device.atom, acc, cb, np.full((k, atom), f, np.int64))) for f in D.F_ENDS]
    return c[0], c[1], (c[0] == c[1]) | ((c[0] | c[1]) & 0x7F == 0)


def references(formed: dict, device) -> tuple[np.ndarray, ...]:
    """Per element of each formed row: its two ends' codes, whether they agree, and its centre code (the forming atom
    from f32(α·x) with zero noise products, cast)."""
    plus, minus, dead, centre = [], [], [], []
    for alpha, x, beta in zip(formed["alpha"], formed["x"], formed["beta"]):
        p, m, d = _ends(alpha, x, beta, device)
        acc = FT._finite(FT._mul(np.full(len(x), alpha, np.uint32), x.astype(np.uint32)))
        z = np.zeros((len(x), device.atom.k), np.int64)
        plus.append(p), minus.append(m), dead.append(d), centre.append(FT._fp8(FT._step(device.atom, acc, z, z)))
    return tuple(np.stack(v) for v in (plus, minus, dead, centre))


def _terms(acc: np.ndarray, a: np.ndarray, b: np.ndarray, atom):
    """Per lane of one atom from accumulator `acc`: each product's cut flag (`pearl_c_debit.cut_flags`: its significand
    on the adder shifted right by its distance below the largest exponent of the lane's nonzero products and
    accumulator, at least the floor, is 0), its magnitude key (exponent, significand; −1 for a zero product), and whether
    the accumulator's exponent is above every product's."""
    if len(atom.groups) != 1 or atom.width < 7 or ((a & 0x7F) == 0x7F).any() or ((b & 0x7F) == 0x7F).any():
        raise ValueError("the replay takes one group of E4M3 products and no NaN codes")
    ea, eb = (a >> 3) & 0xF, (b >> 3) & 0xF
    sig = ((a & 7) | np.where(ea > 0, 8, 0)) * ((b & 7) | np.where(eb > 0, 8, 0))
    e = np.maximum(ea, 1) + np.maximum(eb, 1) - 14
    f, m = (acc.astype(np.int64) >> 23) & 0xFF, acc.astype(np.int64) & 0x7FFFFF
    e_acc = np.where((f == 0) & (m == 0), atom.zero_exponent, np.where(f == 0, -126, f - 127))
    top_prod = np.maximum(np.where(sig > 0, e, atom.zero_exponent).max(axis=1), atom.zero_exponent)
    top = np.maximum(top_prod, e_acc)
    cut = ((sig << (atom.width - 7)) >> np.clip(top[:, None] - e, 0, 62)) == 0
    key = np.where(sig > 0, (e + 1024) * 256 + sig, -1)
    return cut, key, e_acc > top_prod


def word_counts(A, B, ref_a, ref_b, big_a, big_b, device) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per word (A's rows by B's, row-major), the products of each kind along every word's chain at the drawn codes and
    at both ends, and the drawn cuts by cause (`CAUSES`) over the tile."""
    pa, ma, da, ca = ref_a
    pb, mb, db, cb = ref_b
    fix_a, fix_b = _same(A, ca), _same(B, cb)
    na, nb, k = A.shape[0], B.shape[0], A.shape[1]
    ia, ib = np.repeat(np.arange(na), nb), np.tile(np.arange(nb), na)
    atom, g_ = device.atom, device.g
    kinds = ("ends", "cut", "dead", "mends")
    out = {kind: np.zeros(len(ia), np.int64) for kind in kinds}
    causes = np.zeros(len(CAUSES), np.int64)
    acc, acc_p, acc_m = (np.zeros(len(ia), np.uint32) for _ in range(3))
    g_key, g_cls = np.full(len(ia), -1, np.int64), np.zeros(len(ia), np.int64)
    lanes = np.arange(len(ia))
    for t, s in enumerate(range(0, k, atom.k)):
        if t % g_ == 0:
            acc[:], acc_p[:], acc_m[:] = 0, 0, 0
            g_key[:], g_cls[:] = -1, 0
        sl = slice(s, s + atom.k)
        a, b = A[ia, sl], B[ib, sl]
        a_p, b_p, a_m, b_m = pa[ia, sl], pb[ib, sl], ma[ia, sl], mb[ib, sl]
        ends = (da[ia, sl] & db[ib, sl]) | (_terms(acc_p, a_p, b_p, atom)[0] & _terms(acc_m, a_m, b_m, atom)[0])
        cut, key, raised = _terms(acc, a, b, atom)
        dead = fix_a[ia, sl] & fix_b[ib, sl]
        for kind, m in (("ends", ends), ("cut", cut), ("dead", dead), ("mends", ends | cut | dead)):
            out[kind] += m.sum(axis=1)
        cls = big_a[ia, sl].astype(np.int64) | (big_b[ib, sl].astype(np.int64) << 1)
        top = key.argmax(axis=1)
        own_cls = cls[lanes, top]
        zero = key < 0
        causes[0] += int((cut & zero).sum())
        live = cut & ~zero
        per_lane = live.sum(axis=1)
        for w, src in ((1, own_cls), (5, g_cls)):
            pick = ~raised if w == 1 else raised
            causes[w:w + 4] += np.bincount(src[pick], weights=per_lane[pick], minlength=4).astype(np.int64)
        top_key = key[lanes, top]
        up = top_key > g_key
        g_key, g_cls = np.where(up, top_key, g_key), np.where(up, own_cls, g_cls)
        acc, acc_p, acc_m = (FT._step(atom, acc, a, b), FT._step(atom, acc_p, a_p, b_p), FT._step(atom, acc_m, a_m, b_m))
    return {kind: v.reshape(na, nb) for kind, v in out.items()}, causes


def measure(scheme, record, fa: dict, fb: dict, ref_a, ref_b, big_a, big_b) -> dict:
    """The drawn-salt terms and debits of one setting's codes on the tile's credited words at `scheme`'s device."""
    d, n = scheme.device, record.matmul.shape.n
    A, B = fa["codes"], fb["codes"]
    na, nb, k = A.shape[0], B.shape[0], A.shape[1]
    flags, zeros = FT._flags(A, B, d)
    flags, zeros = flags.reshape(na, nb, -1), zeros.reshape(na, nb)
    every = flags.all(axis=1)
    elements = np.array([k if every[i].all() else int((np.repeat(every[i], d.atom.k) | ref_a[2][i]).sum())
                         for i in range(na)], np.int64)
    counts, causes = word_counts(A, B, ref_a, ref_b, big_a, big_b, d)
    atoms_i, zeros_i = flags.sum(axis=(1, 2)), zeros.sum(axis=1)

    def debit(rows, mends) -> Fraction:
        return (D.debit_units(int(atoms_i[rows].sum()), Fraction(nb * int(elements[rows].sum()), n), scheme.prices, d,
                              zeros=int(zeros_i[rows].sum())) + int(mends[rows].sum()))

    cell = CW.cell_credit(scheme, record)
    cap = scheme.cap * cell * na * nb
    row_cap = scheme.cap * cell * nb * CW.row_cap_multiple(scheme, k)
    every_row = np.arange(na)
    out = {"products": na * nb * k, "cap": float(cap), "causes": causes.tolist()}
    for kind, v in counts.items():
        out[kind] = int(v.sum())
    out["debit_rule"] = float(debit(every_row, np.zeros(na, np.int64)))
    for label, mends in (("ends", counts["ends"]), ("mends", counts["mends"])):
        m_i = mends.sum(axis=1)
        dt = debit(every_row, m_i)
        out[f"debit_{label}"] = float(dt)
        out[f"ratio_{label}"] = float(dt / cap)
        out[f"row_ratio_max_{label}"] = max(float(debit(np.array([i]), m_i) / row_cap) for i in range(na))
    return out


def _salt(salt: bytes, i: int) -> bytes:
    return salt if i == 0 else hashlib.sha512(b"noise-replay/salt\0" + salt + bytes([i])).digest()[:len(salt)]


def _device(g: int):
    return D.H100 if g == D.H100.g else dataclasses.replace(D.H100, name=f"h100-g{g}", g=g)


def tile_job(job) -> dict:
    pi, pos, tid, rows, cols, si, check = job
    man, epoch, records, served, blobs = PASSES[pi]
    u, rec, record = epoch.workload.matmuls[pos], man["matmuls"][pos], records[pos]
    wt = man["weights"][str(u.weight)]
    k = u.shape.k
    a_bytes, xa_all = _words(blobs[rec["rows"]["path"]], k, rows)
    b_bytes, xb_all = _words(blobs[wt["rows"]["path"]], k, cols)
    tile = Tile(tuple(rows), tuple(cols))
    ctx = Context(_salt(epoch.salt, si), u, bytes.fromhex(rec["root_a"]), epoch.weight_roots[u.weight])
    cr, cc = record.credited(tile)
    pos_a, pos_b = {i: p for p, i in enumerate(rows)}, {j: p for p, j in enumerate(cols)}
    pa, pb = [pos_a[i] for i in cr], [pos_b[j] for j in cc]
    noise = served.noise(ctx)
    e_a = [C.sample_line(s, 0, 0, 0) for s in served.row_seeds(ctx, a_bytes, tile.rows)]
    e_b = [C.sample_line(noise.seed_b, 1, 0, i) for i in tile.cols]
    f_a = np.asarray(C._basis(noise.seed_b, 0, k, served.line_norm), np.int64)
    f_b = np.asarray(C._basis(noise.seed_b, 1, k, served.line_norm), np.int64)
    perm, calibration = PERMS[(pi, pos)]
    out = {"at": [pi, pos, tid], "salt": si, "linear": SHAPES[(k, u.shape.n)], "shape": [u.shape.m, k, u.shape.n],
           "rows": len(cr), "cols": len(cc), "calibration": calibration, "settings": {}}
    schemes = {g: C.PearlC(k_min=served.k_min, tile=served.tile, forming="v1", device=_device(g), hashing=served.hashing)
               for g in OPTS["gs"]}
    ops = None
    for layout in OPTS["layouts"]:
        order = perm if layout == "outlier-last" else np.arange(k)
        xa, xb = xa_all[pa][:, order], xb_all[pb][:, order]
        stats_a = [C.s5_stats(row.tolist())[:2] for row in xa]
        stats_b = [C.s5_stats(row.tolist())[:2] for row in xb]
        for delta in OPTS["deltas"]:
            fa = form(xa, stats_a, [e_a[p] for p in pa], f_a, delta, served.device)
            fb = form(xb, stats_b, [e_b[p] for p in pb], f_b, delta, served.device)
            if delta == 1 and layout == "served":
                ops = served.encode(ctx, noise, a_bytes, b_bytes, tile)
                if ops is None or (fa["codes"].tolist() != [ops.a.codes[p] for p in pa]
                                   or fb["codes"].tolist() != [ops.b.codes[p] for p in pb]):
                    raise AssertionError(f"{out['at']} salt {si}: the δ = 1 replay is not the served scheme's encode")
                out["encode_matches"] = True
                if si == 0:
                    leaves = blobs.get(man["leaves"]["path"])
                    if leaves is not None:
                        with leaves.open("rb") as f:
                            f.seek(LEAF_BYTES * tid)
                            committed = f.read(LEAF_BYTES)
                        out["leaf_matches"] = served.digest(ctx, noise, served.checked(ctx, ops)) == committed
            rho_a = np.array([P.f32_of_bits(r) for _, r in stats_a])[:, None]
            rho_b = np.array([P.f32_of_bits(r) for _, r in stats_b])[:, None]
            big_a = np.abs(xa.view(np.float32)) > 4 * delta * rho_a
            big_b = np.abs(xb.view(np.float32)) > 4 * delta * rho_b
            ref_a, ref_b = references(fa, served.device), references(fb, served.device)
            prec = {"a": precision(fa, stats_a), "b": precision(fb, stats_b)}
            for g, scheme in schemes.items():
                got = measure(scheme, record, fa, fb, ref_a, ref_b, big_a, big_b)
                got["precision"] = prec
                if check and delta == 1 and layout == "served" and g == served.device.g:
                    ref_debit, ref_cap = CW.tile_cap(served, record, fa["codes"].tolist(), fb["codes"].tolist(),
                                                     [(fa["alpha"][i], fa["x"][i].tolist(), fa["beta"][i])
                                                      for i in range(len(pa))])
                    if float(ref_debit) != got["debit_rule"] or float(ref_cap) != got["cap"]:
                        raise AssertionError(f"{out['at']}: tile_cap's ({ref_debit}, {ref_cap}) is not this replay's")
                    out["cross_checked"] = True
                out["settings"][f"d{delta}/g{g}/{layout}"] = got
    return out


def _coins(man: dict):
    live = isinstance(man.get("coins"), dict) and man["coins"].get("scheme") == LIVE_SCHEME
    return (LiveCommitment if live else CoinCommitment).from_json(man["coins"])


def _class(man: dict, pos: int) -> tuple:
    u = man["matmuls"][pos]
    kind = SHAPES[(u["k"], u["n"])]
    return ("residual",) if kind in ("qkv", "gate_up") else (kind, u["weight"])


def calibrate(parts: dict, case: Path) -> None:
    """`PERMS`: per (pass, matmul) the outlier-last order of k, from each channel's mean |x|/rms over the real rows of
    its class's A rows in the other passes (its own pass's when no other pass kept the class)."""
    sums: dict[tuple, dict[int, list]] = {}
    files = parts["evidence"]["files"]
    for pi, man in enumerate(parts["manifests"]):
        blobs = {path: case / "blobs" / h for path, h in files[pi].items()}
        for pos, u in enumerate(man["matmuls"]):
            if u["rows"]["path"] not in blobs:
                continue
            x = (np.fromfile(blobs[u["rows"]["path"]], "<u2").astype("<u4") << 16).view(np.float32)
            x = x.reshape(-1, u["k"])[:u["real_rows"]].astype(np.float64)
            rms = np.sqrt((x * x).mean(axis=1))
            x = x[rms > 0] / rms[rms > 0, None]
            got = sums.setdefault(_class(man, pos), {}).setdefault(pi, [np.zeros(u["k"]), 0])
            got[0] += np.abs(x).sum(axis=0)
            got[1] += x.shape[0]
    for pi, man in enumerate(parts["manifests"]):
        for pos, u in enumerate(man["matmuls"]):
            key = _class(man, pos)
            by_pass = sums.get(key, {})
            use, source = [v for q, v in by_pass.items() if q != pi], "other passes"
            if not use:
                use, source = ([by_pass[pi]], "own pass") if pi in by_pass else ([], "none")
            if not use:
                PERMS[(pi, pos)] = (np.arange(u["k"]), source)
                continue
            score = sum(v[0] for v in use) / sum(v[1] for v in use)
            if key[0] == "o":
                group = score.reshape(HEADS // Q_PER_KV, Q_PER_KV, HEAD_DIM).mean(axis=1)
                perm = np.concatenate([h * HEAD_DIM + np.argsort(group[h // Q_PER_KV], kind="stable")
                                       for h in range(HEADS)])
            else:
                perm = np.argsort(score, kind="stable")
            PERMS[(pi, pos)] = (perm, source)


def _agg(lines: list[dict], key: str, salt: int | None) -> dict:
    got = [ln["settings"][key] for ln in lines if (salt is None or ln["salt"] == salt) and key in ln["settings"]]
    cap = sum(g["cap"] for g in got)
    out = {"tiles": len(got)}
    for kind in ("products", "cut", "dead", "ends", "mends"):
        out[kind] = sum(g[kind] for g in got)
    out["cut_share"] = out["cut"] / out["products"]
    out["dead_share"] = out["dead"] / out["products"]
    causes = np.sum([g["causes"] for g in got], axis=0)
    out["causes"] = dict(zip(CAUSES, causes.tolist()))
    for label in ("ends", "mends"):
        out[f"pooled_ratio_{label}"] = sum(g[f"debit_{label}"] for g in got) / cap
        out[f"ratio_max_{label}"] = max(g[f"ratio_{label}"] for g in got)
        out[f"tiles_over_{label}"] = sum(1 for g in got if g[f"ratio_{label}"] > 1)
        out[f"row_ratio_max_{label}"] = max(g[f"row_ratio_max_{label}"] for g in got)
        out[f"tiles_row_over_{label}"] = sum(1 for g in got if g[f"row_ratio_max_{label}"] > 1)
    for side in ("a", "b"):
        n = sum(g["precision"][side]["n"] for g in got)
        out[f"rms_err_{side}"] = (sum(g["precision"][side]["err2"] for g in got) / n) ** 0.5
        out[f"saturated_{side}"] = sum(g["precision"][side]["saturated"] for g in got)
        out[f"off_floor_{side}"] = sum(g["precision"][side]["off_floor"] for g in got)
    return out


def summarize(lines: list[dict]) -> dict:
    keys = sorted({k for ln in lines for k in ln["settings"]}, key=lambda s: (int(s.split("/")[0][1:]),
                                                                           -int(s.split("/")[1][1:]), s.split("/")[2]))
    salts = sorted({ln["salt"] for ln in lines})
    doc = {}
    for key in keys:
        per = {str(s): _agg(lines, key, s) for s in salts}
        base = f"d1/{key.split('/')[1]}/served"
        for s in salts:
            ref = _agg(lines, base, s)
            for side in ("a", "b"):
                per[str(s)][f"bits_lost_{side}"] = float(np.log2(per[str(s)][f"rms_err_{side}"] / ref[f"rms_err_{side}"]))
        worst = {"tiles_over_mends": max(p["tiles_over_mends"] for p in per.values()),
                 "row_ratio_max_mends": max(p["row_ratio_max_mends"] for p in per.values()),
                 "pooled_ratio_mends": max(p["pooled_ratio_mends"] for p in per.values()),
                 "cut_share": max(p["cut_share"] for p in per.values()),
                 "dead": max(p["dead"] for p in per.values())}
        worst["passes"] = (worst["tiles_over_mends"] == 0 and worst["row_ratio_max_mends"] <= 0.5
                           and worst["pooled_ratio_mends"] <= 1 / 3)
        doc[key] = {"salts": per, "worst": worst}
    return doc


def main() -> int:
    case, out_dir = Path(sys.argv[1]), Path(sys.argv[2])

    def opt(name: str, default):
        return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default

    jobs = opt("--jobs", len(os.sched_getaffinity(0)))
    seed, per_call, every = opt("--seed", 20261113), opt("--per-call", 1), opt("--cross-check", 8)
    OPTS["deltas"] = [int(x) for x in opt("--deltas", "1,2,3,4").split(",")]
    OPTS["gs"] = [int(x) for x in opt("--gs", "4,2,1").split(",")]
    OPTS["layouts"] = opt("--layouts", ",".join(LAYOUTS)).split(",")
    n_salts, limit = opt("--salts", 3), opt("--limit", 0)
    if 1 not in OPTS["deltas"] or "served" not in OPTS["layouts"] or not set(OPTS["layouts"]) <= set(LAYOUTS):
        raise SystemExit("--deltas takes 1 and --layouts takes served: the replay holds δ = 1 to the served encode")
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = json.loads((case / "parts.json").read_text())
    files = parts["evidence"]["files"]
    rng = np.random.default_rng(seed)
    picks = []
    for pi, man in enumerate(parts["manifests"]):
        served = PD.scheme_of(man["scheme"], **{k: tuple(v) if isinstance(v, list) else v
                                                for k, v in man.get("instance", {}).items()})
        if served.device != D.H100:
            raise SystemExit(f"pass {pi}: {served.name} is not the H100's; the revision is the H100's")
        matmuls = tuple(Matmul(u["index"], Shape(u["m"], u["k"], u["n"]), u["weight"]) for u in man["matmuls"])
        epoch, _ = PD.manifest_epoch(man, _coins(man))
        records = [PW.UnitRecord(mm, u["real_rows"], u["k_kept"], tuple(u["excluded_a"]),
                                 tuple(man["weights"][str(mm.weight)]["excluded_b"]), tuple(u.get("voluntary_a", ())),
                                 tuple(man["weights"][str(mm.weight)].get("voluntary_b", ())))
                   for mm, u in zip(matmuls, man["matmuls"])]
        blobs = {path: case / "blobs" / h for path, h in files[pi].items()}
        PASSES[pi] = (man, epoch, records, served, blobs)
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
    calibrate(parts, case)
    FT.install(PASSES[0][3])
    picks = picks[:limit] if limit else picks
    work = [(*p, si, i % every == 0) for i, p in enumerate(picks) for si in range(n_salts)]
    print(f"{len(picks)} tiles x {n_salts} salts over {len(PASSES)} passes, δ {OPTS['deltas']}, G {OPTS['gs']}, "
          f"layouts {OPTS['layouts']}, {jobs} jobs", flush=True)
    lines = []
    with (out_dir / "tiles.jsonl").open("w") as f, multiprocessing.get_context("fork").Pool(jobs) as pool:
        for line in pool.imap_unordered(tile_job, work):
            lines.append(line)
            f.write(json.dumps(line) + "\n")
            f.flush()
            print(f"{len(lines)}/{len(work)} {line['at']} salt {line['salt']}", flush=True)
    doc = {"case": str(case), "seed": seed, "per_call": per_call, "salts": n_salts,
           "served": [m["scheme"] for m in parts["manifests"]],
           "leaf_checked": sum(1 for ln in lines if "leaf_matches" in ln),
           "leaf_matches": sum(1 for ln in lines if ln.get("leaf_matches")),
           "encode_matches": sum(1 for ln in lines if ln.get("encode_matches")),
           "cross_checked": sum(1 for ln in lines if ln.get("cross_checked")),
           "calibration": {s: sum(1 for ln in lines if ln["calibration"] == s and ln["salt"] == 0)
                           for s in ("other passes", "own pass", "none")},
           "settings": summarize(lines)}
    (out_dir / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")
    print("SUMMARY " + json.dumps({k: v for k, v in doc.items() if k != "settings"}), flush=True)
    for key, v in doc["settings"].items():
        print(f"{key}: {json.dumps(v['worst'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
