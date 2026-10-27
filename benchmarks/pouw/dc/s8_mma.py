"""The integer tensor-core step on Hopper (sm_90a) and datacenter Blackwell (sm_100a) against
``verity.ml.circuits.silicon.integer``, one instruction per tile (`probes/s8_mma.cu`).

    s8_mma.py gen --family F --out CASES --meta META     the case file s8_mma.cu reads for family F (mma, wg, wg64, wg128,
                                                         wg256 or tc)
    s8_mma.py check --family F --cases CASES --meta META --words DIR --sass SASS --json OUT [--structured OUT2]
    s8_mma.py evidence --family F --out-dir RUN_OUT --instruction ID --run RUN --json OUT
                                                         one instruction's tc-evidence/v1 meta (benchmarks/tc_probe/trust.py)

Families and their tiles (M x N, K = 32): ``mma`` mma.sync m16n8k32 (16 x 8, both cards), ``wg`` wgmma m64n8k32 (64 x 8,
sm_90a), ``wg64``, ``wg128`` and ``wg256`` wgmma m64nNk32 at the N an s8 GEMM's warpgroup tile issues (64 x N, sm_90a), ``tc``
tcgen05.mma kind::i8 (128 x 32, sm_100a). Eight forms each (s8/u8 x s8/u8, saturating or not) run over the
same tiles, each reading the operand bytes in its own formats. The cases are sm_120's IMMA probe's (run
r20261015-084000-06d8, which pinned `integer.BLACKWELL_SM120_S8_M16N8K32`) at the family's tile shape:

- sweeps: ``random_zero_acc``, ``random_random_acc`` (C uniform over all 2^32 words), ``edge_acc`` (C within 2^17 or 2^21
  of an s32 end, so sums leave the range);
- ``zero_products_nonzero_acc`` (one operand all zero: D must be C), ``acc_only_lsb_walk`` (one product of +-1 or 255
  against C = +-2^b + {-1, 0, 1}), ``large_cancellation`` (operands from +-127 and -128), ``code_m128`` (codes from
  {-128, 127, -127, -1, 0, 1}), ``extremes`` (constant tiles of -128, 127, 255 and 0 against edge accumulators);
- where a clamp sits: ``clamp_block_a`` / ``clamp_block_b`` (products +-16129 in sign blocks of 1, 2, 4, 8 or 16 along
  k, on the A or the B side) and ``clamp_single_a`` / ``clamp_single_b`` (one product of the other sign at each k), all
  against accumulators near +-2^31. Saturating once at the end, after every product, or after every group of G products
  in either k order give different words on these tiles.

``check`` compares every word with the sm_120 model's `IntegerMma.step` at the form's formats and saturation on the
structured families and the first tiles of each sweep, and with its vectorized equivalent everywhere; lists the clamp
hypotheses each saturating form's words allow; and reads the SASS for one tensor instruction per kernel. A form that
matches is the sm_120 step's function on this card.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

S32_MAX, S32_MIN = (1 << 31) - 1, -(1 << 31)
FORMS = [f"{a}{b}_{m}" for m in ("sat", "wrap") for a in ("s8", "u8") for b in ("s8", "u8")]
SHAPES = {"mma": (16, 8), "wg": (64, 8), "wg64": (64, 64), "wg128": (64, 128), "wg256": (64, 256), "tc": (128, 32)}
POS, NEG = 0x7F, 0x81            # +127 and -127 as s8 codes (127 and 129 as u8)
EDGES = [S32_MAX - 1000, S32_MAX - 20000, S32_MAX - 300000, S32_MAX, S32_MIN + 1000, S32_MIN + 20000, S32_MIN + 300000, S32_MIN]
SWEEP_WORDS = 22_000 * 128       # words per sweep family: three of them give >= 8.4M words a form, the whole set >= 10M
DRAFT_WORDS = 100 * 128          # words per sweep family checked through IntegerMma.step itself
STRUCTURED = ("extremes", "clamp_block_a", "clamp_block_b", "clamp_single_a", "clamp_single_b")
HEADER = b"S8MMAPR1"


def words(values) -> np.ndarray:
    return (np.asarray(values, dtype=np.int64) & 0xFFFFFFFF).astype(np.uint32)


def sign_patterns() -> dict[str, list[int]]:
    pats: dict[str, list[int]] = {}
    for b in (1, 2, 4, 8, 16):
        for first in (1, -1):
            pats[f"blocks{b}{'+' if first > 0 else '-'}"] = [first if (k // b) % 2 == 0 else -first for k in range(32)]
    pats.update({"all+": [1] * 32, "all-": [-1] * 32, "first+": [1] + [-1] * 31, "first-": [-1] + [1] * 31,
                 "last+": [-1] * 31 + [1], "last-": [1] * 31 + [-1]})
    return pats


def single_patterns() -> dict[str, list[int]]:
    pats = {}
    for lone in (1, -1):
        for k0 in range(32):
            pats[f"lone{'+' if lone > 0 else '-'}@{k0}"] = [lone if k == k0 else -lone for k in range(32)]
    return pats


def codes(pattern: list[int]) -> np.ndarray:
    return np.array([POS if s > 0 else NEG for s in pattern], dtype=np.uint8)


def side_tiles(patterns: dict[str, list[int]], side: str, m: int, n: int) -> tuple[list, list[list[str]]]:
    """Tiles carrying one sign pattern per A row (side a, C[i][j] = EDGES[j % 8]) or per B column (side b, C[i][j] =
    EDGES[i % 8]); the other operand is all +127. Returns the tiles and, per tile, the pattern of each row (a) or column (b)."""
    names = list(patterns)
    per = m if side == "a" else n
    tiles, labels = [], []
    for start in range(0, len(names), per):
        chunk = names[start:start + per]
        chunk += [chunk[-1]] * (per - len(chunk))
        A = np.full((m, 32), POS, np.uint8)
        B = np.full((n, 32), POS, np.uint8)
        if side == "a":
            for i, p in enumerate(chunk):
                A[i] = codes(patterns[p])
            C = np.array([[EDGES[j % 8] for j in range(n)] for _ in range(m)], np.int64)
        else:
            for j, p in enumerate(chunk):
                B[j] = codes(patterns[p])
            C = np.array([[EDGES[i % 8]] * n for i in range(m)], np.int64)
        tiles.append((A, B, words(C)))
        labels.append(chunk)
    return tiles, labels


def generate(family: str, seed: int = 20261015) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    m, n = SHAPES[family]
    tw = m * n
    rng = np.random.default_rng(seed)
    fams: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, list | None]] = []

    def operands(t: int) -> tuple[np.ndarray, np.ndarray]:
        return rng.integers(0, 256, (t, m, 32), dtype=np.uint8), rng.integers(0, 256, (t, n, 32), dtype=np.uint8)

    def acc(t: int) -> np.ndarray:
        return rng.integers(0, 1 << 32, (t, m, n), dtype=np.uint64).astype(np.uint32)

    sw = -(-SWEEP_WORDS // tw)
    fams.append(("random_zero_acc", *operands(sw), np.zeros((sw, m, n), np.uint32), None))
    fams.append(("random_random_acc", *operands(sw), acc(sw), None))
    u = np.where(rng.integers(0, 2, (sw, m, n)) == 0, rng.integers(0, 1 << 17, (sw, m, n)), rng.integers(0, 1 << 21, (sw, m, n)))
    fams.append(("edge_acc", *operands(sw), words(np.where(rng.integers(0, 2, (sw, m, n)) == 0, S32_MAX - u, S32_MIN + u)), None))

    t = -(-4000 * 128 // tw)
    A, B = operands(t)
    A[: t // 2] = 0
    B[t // 2:] = 0
    c = acc(t)
    c[::4] = words(rng.choice(EDGES, (len(c[::4]), m, n)))
    fams.append(("zero_products_nonzero_acc", A, B, c, None))

    t = -(-2000 * 128 // tw)
    A = np.zeros((t, m, 32), np.uint8)
    B = np.zeros((t, n, 32), np.uint8)
    k0 = rng.integers(0, 32, t)
    A[np.arange(t), :, k0] = rng.choice(np.array([0x01, 0xFF], np.uint8), (t, m))
    B[np.arange(t), :, k0] = rng.choice(np.array([0x01, 0xFF], np.uint8), (t, n))
    bit = rng.integers(0, 32, (t, m, n))
    sgn = rng.choice([1, -1], (t, m, n))
    fams.append(("acc_only_lsb_walk", A, B, words(sgn * (np.int64(1) << bit) + rng.integers(-1, 2, (t, m, n))), None))

    t = -(-4000 * 128 // tw)
    A = rng.choice(np.array([POS, NEG, 0x80], np.uint8), (t, m, 32))
    B = rng.choice(np.array([POS, NEG, 0x80, 0x01, 0xFF], np.uint8), (t, n, 32))
    c = acc(t)
    c[::2] = words(rng.choice(EDGES, (len(c[::2]), m, n)))
    fams.append(("large_cancellation", A, B, c, None))

    pool = np.array([0x80, 0x7F, 0x81, 0xFF, 0x00, 0x01], np.uint8)
    c = acc(t)
    c[::2] = words(rng.choice(EDGES + [0, 1, -1], (len(c[::2]), m, n)))
    fams.append(("code_m128", rng.choice(pool, (t, m, 32)), rng.choice(pool, (t, n, 32)), c, None))

    ext_a, ext_b, ext_c = [], [], []
    table = np.array([[EDGES[j % 8] for j in range(n)] for _ in range(m)], np.int64)
    table[m // 2:] = rng.choice(EDGES + [0, 1, -1, 1 << 20, -(1 << 20)], (m - m // 2, n))
    for ca in (0x80, 0x7F, 0xFF, 0x00):
        for cb in (0x80, 0x7F, 0x81, 0xFF):
            ext_a.append(np.full((m, 32), ca, np.uint8))
            ext_b.append(np.full((n, 32), cb, np.uint8))
            ext_c.append(words(table))
    fams.append(("extremes", np.stack(ext_a), np.stack(ext_b), np.stack(ext_c), None))

    for name, pats, side in (("clamp_block_a", sign_patterns(), "a"), ("clamp_block_b", sign_patterns(), "b"),
                             ("clamp_single_a", single_patterns(), "a"), ("clamp_single_b", single_patterns(), "b")):
        tiles, labels = side_tiles(pats, side, m, n)
        fams.append((name, np.stack([x[0] for x in tiles]), np.stack([x[1] for x in tiles]), np.stack([x[2] for x in tiles]),
                     labels))

    meta, start = [], 0
    for name, A, B, C, labels in fams:
        meta.append({"family": name, "start": start, "tiles": len(A), **({"labels": labels} if labels else {})})
        start += len(A)
    return (np.concatenate([f[1] for f in fams]), np.concatenate([f[2] for f in fams]), np.concatenate([f[3] for f in fams]),
            meta)


def write_cases(path: Path, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(HEADER + np.array([len(A)], "<u4").tobytes() + np.array([A.shape[1], B.shape[1]], "<u2").tobytes())
        f.write(np.ascontiguousarray(A).tobytes())
        f.write(np.ascontiguousarray(B).tobytes())
        f.write(np.ascontiguousarray(C.astype("<u4")).tobytes())


def read_cases(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.fromfile(path, np.uint8)
    if raw[:8].tobytes() != HEADER:
        raise ValueError(f"{path} is not an s8_mma case file")
    t = int(raw[8:12].view("<u4")[0])
    m, n = (int(v) for v in raw[12:16].view("<u2"))
    a, b = 16 + t * m * 32, 16 + t * (m + n) * 32
    return raw[16:a].reshape(t, m, 32), raw[a:b].reshape(t, n, 32), raw[b:].view("<u4").reshape(t, m, n)


def decode(fmt: str, x: np.ndarray) -> np.ndarray:
    return x.astype(np.int8).astype(np.int64) if fmt == "s8" else x.astype(np.int64)


def exact_sums(A, B, fa, fb, chunk_words=1 << 20) -> np.ndarray:
    t, m, _ = A.shape
    n = B.shape[1]
    out = np.empty((t, m, n), np.int64)
    step = max(1, chunk_words // (m * n))
    for s in range(0, t, step):
        out[s:s + step] = np.matmul(decode(fa, A[s:s + step]), decode(fb, B[s:s + step]).transpose(0, 2, 1))
    return out


def as_s32(w: np.ndarray) -> np.ndarray:
    return w.astype(np.int64) - ((w.astype(np.int64) >> 31) << 32)


def hypotheses(c: int, prods: list[int]) -> dict[str, int]:
    """D under each place a saturating unit could clamp: after every group of G products (G = 1 ... 32; G = 32 is "once,
    at the end"), with C added first, in ascending or descending k order."""
    out = {}
    for order in ("asc", "desc"):
        seq = prods if order == "asc" else prods[::-1]
        for G in (1, 2, 4, 8, 16, 32):
            acc = c
            for s in range(0, 32, G):
                acc = min(max(acc + sum(seq[s:s + G]), S32_MIN), S32_MAX)
            out[f"G{G}-{order}" if G < 32 else "end"] = acc
    return out


TENSOR_OP = re.compile(r"(I|H|Q|O)MMA|IDP|[A-Z]*GMMA|UTC\w*MMA")


def sass_kernels(text: str) -> dict[str, dict]:
    """Each kernel's tensor instructions (IMMA, IGMMA, UTCIMMA, ...) from `cuobjdump -sass`."""
    kernels: dict[str, dict] = {}
    name = None
    for line in text.splitlines():
        if m := re.search(r"Function : (\S+)", line):
            name = m.group(1)
            kernels[name] = {"tensor": []}
            continue
        # opcodes carry lowercase shape fields (IGMMA.64x8x32.S8.S8)
        if name and (m := re.search(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Za-z0-9_.]*)\s", line)):
            if TENSOR_OP.match(m.group(1)):
                kernels[name]["tensor"].append(line.split("/*", 2)[1].split("*/", 1)[1].strip().rstrip(";").strip())
    for k in kernels.values():
        k["count"] = len(k["tensor"])
        k["opcodes"] = sorted({s.split()[0] for s in k["tensor"]})
    return kernels


def check_words(family: str, A, B, C, meta, device: dict[str, np.ndarray]) -> tuple[dict, dict, bool]:
    """Each form's words (``device[form]``, (t, M, N) u32) against the sm_120 model at the form's formats and saturation:
    (report of the forms, the clamp tiles' words by pattern, every form exact)."""
    from verity.ml.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32 as SAT
    from verity.ml.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32_WRAP as WRAP

    t, m, n = C.shape
    Cv = as_s32(C)
    fam_of = np.empty(t, object)
    for f in meta:
        fam_of[f["start"]:f["start"] + f["tiles"]] = f["family"]
    draft_tiles = np.concatenate([np.arange(f["start"], f["start"] + (f["tiles"] if f["family"] in STRUCTURED else
                                                                      min(-(-DRAFT_WORDS // (m * n)), f["tiles"])))
                                  for f in meta])
    forms: dict = {}
    structured: dict = {}
    sums: dict = {}
    ok = True
    for form in FORMS:
        fa, fb, mode = form[:2], form[2:4], form.split("_")[1]
        model = replace(SAT if mode == "sat" else WRAP, name=f"probe_{family}_{form}", dtype=fa, b_dtype=fb)
        D = device[form]
        S = sums.setdefault(form[:4], exact_sums(A, B, fa, fb))
        tot = Cv + S
        vec = words(np.clip(tot, S32_MIN, S32_MAX)) if mode == "sat" else words(tot)
        draft_mismatch, vec_vs_draft, examples = 0, 0, []
        for tile in draft_tiles:
            for i in range(m):
                for j in range(n):
                    w = model.step(int(C[tile, i, j]), A[tile, i].tolist(), B[tile, j].tolist())
                    vec_vs_draft += int(w != vec[tile, i, j])
                    if w != D[tile, i, j]:
                        draft_mismatch += 1
                        if len(examples) < 8:
                            examples.append({"family": fam_of[tile], "tile": int(tile), "i": i, "j": j, "C": int(Cv[tile, i, j]),
                                             "exact": int(tot[tile, i, j]), "device": int(as_s32(D[tile, i, j])),
                                             "model": int(as_s32(np.uint32(w)))})
        bad = D != vec
        overflow = (tot > S32_MAX) | (tot < S32_MIN)
        rec = {"model": f"{(SAT if mode == 'sat' else WRAP).name} at a={fa}, b={fb}", "words": int(D.size),
               "mismatches": int(bad.sum()),
               "mismatches_by_family": {f["family"]: int(bad[f["start"]:f["start"] + f["tiles"]].sum()) for f in meta},
               "model_step_checked": int(len(draft_tiles) * m * n), "model_step_mismatches": draft_mismatch,
               "vectorized_vs_model_step": vec_vs_draft, "out_of_range_words": int(overflow.sum()),
               "unwritten_words": int((D == 0xA5A5A5A5).sum()), "first_model_step_mismatches": examples}
        if bad.any() and not examples:
            tile, i, j = map(int, np.argwhere(bad)[0])
            rec["first_mismatch"] = {"family": fam_of[tile], "C": int(Cv[tile, i, j]), "exact": int(tot[tile, i, j]),
                                     "device": int(as_s32(D[tile, i, j])), "model": int(as_s32(vec[tile, i, j]))}
        ok &= rec["mismatches"] == 0 and draft_mismatch == 0 and vec_vs_draft == 0
        if mode == "sat":
            alive = None
            for f in meta:
                if not f["family"].startswith("clamp_"):
                    continue
                side = f["family"][-1]
                for ti, labels in enumerate(f["labels"]):
                    tile = f["start"] + ti
                    for i in range(m):
                        for j in range(n):
                            prods = (decode(fa, A[tile, i]) * decode(fb, B[tile, j])).tolist()
                            h = hypotheses(int(Cv[tile, i, j]), prods)
                            d = int(as_s32(D[tile, i, j]))
                            fits = {k for k, v in h.items() if v == d}
                            alive = fits if alive is None else alive & fits
                            label = labels[i] if side == "a" else labels[j]
                            structured.setdefault(form, {}).setdefault(f["family"], {}).setdefault(label, {})[
                                str(int(Cv[tile, i, j]))] = d
            rec["clamp_hypotheses_consistent"] = sorted(alive or [])
        forms[form] = rec
    return forms, structured, bool(ok)


def check(args) -> int:
    sys.path.insert(0, str(Path.cwd()))
    A, B, C = read_cases(Path(args.cases))
    meta = json.loads(Path(args.meta).read_text())
    t, m, n = C.shape
    if (m, n) != SHAPES[args.family]:
        raise SystemExit(f"{args.cases} holds {m} x {n} tiles, not family {args.family}'s {SHAPES[args.family]}")
    device = {f: np.fromfile(Path(args.words) / f"d_{f}.bin", "<u4").reshape(t, m, n) for f in FORMS}
    forms, structured, ok = check_words(args.family, A, B, C, meta, device)
    report: dict = {"family": args.family, "tile": [m, n, 32], "tiles": t, "words_per_form": t * m * n, "forms": forms,
                    "families": {f["family"]: f["tiles"] for f in meta}}
    sass = sass_kernels(Path(args.sass).read_text())
    mine = {k: v for k, v in sass.items() if k.startswith(f"{args.family}_")}
    report["sass"] = {k: {"count": v["count"], "opcodes": v["opcodes"], "lines": v["tensor"]} for k, v in mine.items()}
    report["sass_single_instruction"] = all(mine.get(f"{args.family}_{f}", {}).get("count") == 1 for f in FORMS)
    ok &= report["sass_single_instruction"]
    report["ok"] = bool(ok)
    Path(args.json).write_text(json.dumps(report, indent=1))
    if args.structured:
        Path(args.structured).write_text(json.dumps(structured, indent=0))
    for form, rec in forms.items():
        print(f"{args.family}_{form}: {rec['mismatches']} / {rec['words']} mismatches, model step {rec['model_step_mismatches']} / "
              f"{rec['model_step_checked']}, out of range {rec['out_of_range_words']}, unwritten {rec['unwritten_words']}, "
              f"clamp {rec.get('clamp_hypotheses_consistent', '-')}")
    for k, v in report["sass"].items():
        print(f"{k}: {v['count']} x {v['opcodes']}")
    print("ok" if ok else "MISMATCH")
    return 0 if ok else 1


def evidence(args) -> int:
    """The ``tc-evidence/v1`` meta (``benchmarks/tc_probe/trust.py``'s nested ``tc_evidence``) of one family's s8.s8 .satfinite
    form in a run's outputs: its families with their word and mismatch counts against the registry instruction's model, and
    its SASS."""
    from verity.ml.circuits.silicon.instructions import INSTRUCTIONS

    out = Path(args.out_dir)
    ins = INSTRUCTIONS[args.instruction]
    rep = json.loads((out / f"check-{args.family}.json").read_text())
    meta = json.loads((out / f"meta-{args.family}.json").read_text())
    form = rep["forms"]["s8s8_sat"]
    sass = rep["sass"][f"{args.family}_s8s8_sat"]
    gpu = (out / "gpu.txt").read_text().splitlines()[1].split(", ")
    m, n, _ = rep["tile"]
    model = ins.model.name
    fams = {f["family"]: {"elements": f["tiles"] * m * n, "asserted": True,
                          "mismatches": {model: form["mismatches_by_family"][f["family"]]}} for f in meta}
    fams["specials"] = {**fams["extremes"], "asserted": False,
                        "note": "integer operands have no special encodings: the extremes family's constant tiles of -128, 127, 255 and 0"}
    te = {"instruction": ins.id, "sku": gpu[2], "arch": ins.sm, "driver": gpu[3], "gpu_uuid": gpu[1],
          "nvcc": (out / "nvcc.txt").read_text().split("\n")[0].strip(), "seed": 20261015, "families": fams,
          "total_mismatches_by_model": {model: form["mismatches"]},
          "sass": {"single_instruction": sass["count"] == 1, "tile_kernel_mma_count": sass["count"], "expected_mma_count": 1,
                   "ok": sass["count"] == 1, "opcode": sass["opcodes"]},
          "clamp_hypotheses_consistent": form.get("clamp_hypotheses_consistent"), "form": "s8s8_sat",
          "compared_against": form["model"]}
    doc = {"run_id": args.run, "tc_evidence": te,
           "workload_fingerprint": {"instruction": ins.id, "model": model, "probe": "benchmarks/pouw/dc/s8_mma.py"}}
    Path(args.json).write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in te.items() if k != "families"}))
    return 0 if form["mismatches"] == 0 and rep["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--family", required=True, choices=sorted(SHAPES))
    g.add_argument("--out", required=True)
    g.add_argument("--meta", required=True)
    g.add_argument("--seed", type=int, default=20261015)
    c = sub.add_parser("check")
    c.add_argument("--family", required=True, choices=sorted(SHAPES))
    for a in ("--cases", "--meta", "--words", "--sass", "--json"):
        c.add_argument(a, required=True)
    c.add_argument("--structured")
    e = sub.add_parser("evidence")
    e.add_argument("--family", required=True, choices=sorted(SHAPES))
    for a in ("--out-dir", "--instruction", "--run", "--json"):
        e.add_argument(a, required=True)
    a = ap.parse_args()
    if a.cmd == "evidence":
        return evidence(a)
    if a.cmd == "gen":
        A, B, C, meta = generate(a.family, a.seed)
        write_cases(Path(a.out), A, B, C)
        Path(a.meta).write_text(json.dumps(meta))
        print(f"{len(A)} tiles of {A.shape[1]} x {B.shape[1]}, {C.size:,} words a form")
        return 0
    return check(a)


if __name__ == "__main__":
    sys.exit(main())
