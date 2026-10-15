"""The integer ``mma.sync`` m16n8k32 on the device against ``verity.ml.circuits.silicon.integer``, one instruction per tile.

    imma_probe.py gen --out CASES --meta META       the case file imma_probe.cu reads (seeded, about 80k tiles)
    imma_probe.py check --cases CASES --meta META --words DIR --sass SASS --json OUT [--structured OUT2]
    imma_probe.py evidence --out-dir RUN/out --form s8s8_sat --instruction ID --model NAME --run RUN --json META
                                                    one form's tc-evidence/v1 meta (benchmarks/tc_probe/trust.py)

Eight forms (s8/u8 x s8/u8, with and without ``.satfinite``) run over the same tiles; each form reads the operand bytes in
its own formats.  Families:

- sweeps: ``random_zero_acc``, ``random_random_acc`` (C uniform over all 2^32 words), ``edge_acc`` (C within 2^17 or 2^21
  of an s32 end, so sums leave the range);
- ``zero_products_nonzero_acc`` (one operand all zero: D must be C), ``acc_only_lsb_walk`` (one product of +-1 or 255
  against C = +-2^b + {-1, 0, 1}), ``large_cancellation`` (operands from +-127 and -128), ``code_m128`` (codes from
  {-128, 127, -127, -1, 0, 1}), ``extremes`` (constant tiles of -128, 127, 255 and 0 against edge accumulators);
- where the clamp sits: ``clamp_block_a`` / ``clamp_block_b`` (products +-16129 in sign blocks of 1, 2, 4, 8 or 16 along
  k, on the A or the B side) and ``clamp_single_a`` / ``clamp_single_b`` (one product of the other sign at each k), all
  against accumulators near +-2^31.  Saturating once at the end, after every product, or after every group of G
  products in either k order give different words on these tiles.

``check`` compares every word with the draft's ``IntegerMma.step`` itself on the structured families and the first tiles
of each sweep, and with its vectorized equivalent (checked against ``step`` on those) on everything; reads the clamp
hypotheses each saturating form's words allow; and reads the SASS for one MMA-class instruction per kernel.
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
POS, NEG = 0x7F, 0x81            # +127 and -127 as s8 codes (127 and 129 as u8)
EDGES = [S32_MAX - 1000, S32_MAX - 20000, S32_MAX - 300000, S32_MAX, S32_MIN + 1000, S32_MIN + 20000, S32_MIN + 300000, S32_MIN]
SWEEP = 22_000                   # tiles per sweep family: three of them give >= 8.4M elements, the whole set >= 10M
DRAFT_TILES = 100                # tiles per sweep family checked through IntegerMma.step itself
STRUCTURED = ("extremes", "clamp_block_a", "clamp_block_b", "clamp_single_a", "clamp_single_b")


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


def side_tiles(patterns: dict[str, list[int]], side: str) -> tuple[list, list[str]]:
    """Tiles carrying one sign pattern per A row (side a, C[i][j] = EDGES[j]) or per B column (side b, C[i][j] = EDGES[i % 8]);
    the other operand is all +127.  Returns the tiles and, per tile, the pattern name of each row (a) or column (b)."""
    names = list(patterns)
    per = 16 if side == "a" else 8
    tiles, labels = [], []
    for start in range(0, len(names), per):
        chunk = names[start:start + per]
        chunk += [chunk[-1]] * (per - len(chunk))
        A = np.full((16, 32), POS, np.uint8)
        B = np.full((8, 32), POS, np.uint8)
        if side == "a":
            for i, n in enumerate(chunk):
                A[i] = codes(patterns[n])
            C = np.array([[EDGES[j] for j in range(8)] for _ in range(16)], np.int64)
        else:
            for j, n in enumerate(chunk):
                B[j] = codes(patterns[n])
            C = np.array([[EDGES[i % 8]] * 8 for i in range(16)], np.int64)
        tiles.append((A, B, words(C)))
        labels.append(chunk)
    return tiles, labels


def generate(seed: int = 20261015) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    fams: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, list | None]] = []

    def sweep(name, n, c):
        fams.append((name, rng.integers(0, 256, (n, 16, 32), dtype=np.uint8), rng.integers(0, 256, (n, 8, 32), dtype=np.uint8),
                     c, None))

    sweep("random_zero_acc", SWEEP, np.zeros((SWEEP, 16, 8), np.uint32))
    sweep("random_random_acc", SWEEP, rng.integers(0, 1 << 32, (SWEEP, 16, 8), dtype=np.uint64).astype(np.uint32))
    u = np.where(rng.integers(0, 2, (SWEEP, 16, 8)) == 0, rng.integers(0, 1 << 17, (SWEEP, 16, 8)), rng.integers(0, 1 << 21, (SWEEP, 16, 8)))
    sweep("edge_acc", SWEEP, words(np.where(rng.integers(0, 2, (SWEEP, 16, 8)) == 0, S32_MAX - u, S32_MIN + u)))

    n = 4000
    A = rng.integers(0, 256, (n, 16, 32), dtype=np.uint8)
    B = rng.integers(0, 256, (n, 8, 32), dtype=np.uint8)
    A[: n // 2] = 0
    B[n // 2:] = 0
    c = rng.integers(0, 1 << 32, (n, 16, 8), dtype=np.uint64).astype(np.uint32)
    c[::4] = words(rng.choice(EDGES, (n // 4, 16, 8)))
    fams.append(("zero_products_nonzero_acc", A, B, c, None))

    n = 2000
    A = np.zeros((n, 16, 32), np.uint8)
    B = np.zeros((n, 8, 32), np.uint8)
    k0 = rng.integers(0, 32, n)
    A[np.arange(n), :, k0] = rng.choice(np.array([0x01, 0xFF], np.uint8), (n, 16))
    B[np.arange(n), :, k0] = rng.choice(np.array([0x01, 0xFF], np.uint8), (n, 8))
    bit = rng.integers(0, 32, (n, 16, 8))
    sgn = rng.choice([1, -1], (n, 16, 8))
    fams.append(("acc_only_lsb_walk", A, B, words(sgn * (np.int64(1) << bit) + rng.integers(-1, 2, (n, 16, 8))), None))

    n = 4000
    A = rng.choice(np.array([POS, NEG, 0x80], np.uint8), (n, 16, 32))
    B = rng.choice(np.array([POS, NEG, 0x80, 0x01, 0xFF], np.uint8), (n, 8, 32))
    c = rng.integers(0, 1 << 32, (n, 16, 8), dtype=np.uint64).astype(np.uint32)
    c[::2] = words(rng.choice(EDGES, (n // 2, 16, 8)))
    fams.append(("large_cancellation", A, B, c, None))

    n = 4000
    pool = np.array([0x80, 0x7F, 0x81, 0xFF, 0x00, 0x01], np.uint8)
    c = rng.integers(0, 1 << 32, (n, 16, 8), dtype=np.uint64).astype(np.uint32)
    c[::2] = words(rng.choice(EDGES + [0, 1, -1], (n // 2, 16, 8)))
    fams.append(("code_m128", rng.choice(pool, (n, 16, 32)), rng.choice(pool, (n, 8, 32)), c, None))

    ext_a, ext_b, ext_c = [], [], []
    table = np.array(EDGES * 16, np.int64).reshape(16, 8)
    table[8:] = rng.choice(EDGES + [0, 1, -1, 1 << 20, -(1 << 20)], (8, 8))
    for ca in (0x80, 0x7F, 0xFF, 0x00):
        for cb in (0x80, 0x7F, 0x81, 0xFF):
            ext_a.append(np.full((16, 32), ca, np.uint8))
            ext_b.append(np.full((8, 32), cb, np.uint8))
            ext_c.append(words(table))
    fams.append(("extremes", np.stack(ext_a), np.stack(ext_b), np.stack(ext_c), None))

    for name, pats, side in (("clamp_block_a", sign_patterns(), "a"), ("clamp_block_b", sign_patterns(), "b"),
                             ("clamp_single_a", single_patterns(), "a"), ("clamp_single_b", single_patterns(), "b")):
        tiles, labels = side_tiles(pats, side)
        fams.append((name, np.stack([t[0] for t in tiles]), np.stack([t[1] for t in tiles]), np.stack([t[2] for t in tiles]), labels))

    meta, start = [], 0
    for name, A, B, C, labels in fams:
        meta.append({"family": name, "start": start, "tiles": len(A), **({"labels": labels} if labels else {})})
        start += len(A)
    return (np.concatenate([f[1] for f in fams]), np.concatenate([f[2] for f in fams]), np.concatenate([f[3] for f in fams]), meta)


def write_cases(path: Path, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(b"IMMAPRB1" + np.array([len(A), 0], "<u4").tobytes())
        f.write(np.ascontiguousarray(A).tobytes())
        f.write(np.ascontiguousarray(B).tobytes())
        f.write(np.ascontiguousarray(C.astype("<u4")).tobytes())


def read_cases(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.fromfile(path, np.uint8)
    assert raw[:8].tobytes() == b"IMMAPRB1", path
    n = int(raw[8:12].view("<u4")[0])
    a, b = 16 + n * 512, 16 + n * 768
    return raw[16:a].reshape(n, 16, 32), raw[a:b].reshape(n, 8, 32), raw[b:].view("<u4").reshape(n, 16, 8)


def decode(fmt: str, x: np.ndarray) -> np.ndarray:
    return x.astype(np.int8).astype(np.int64) if fmt == "s8" else x.astype(np.int64)


def exact_sums(A, B, fa, fb, chunk=8000) -> np.ndarray:
    out = np.empty((len(A), 16, 8), np.int64)
    for s in range(0, len(A), chunk):
        out[s:s + chunk] = np.matmul(decode(fa, A[s:s + chunk]), decode(fb, B[s:s + chunk]).transpose(0, 2, 1))
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


def sass_kernels(text: str) -> dict[str, dict]:
    kernels: dict[str, dict] = {}
    name = None
    for line in text.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            name = m.group(1)
            kernels[name] = {"mma_class": []}
            continue
        if name and (m := re.search(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s", line)):
            op = m.group(1)
            if re.match(r"(I|H|Q|O)MMA|IDP|[A-Z]*GMMA", op):
                kernels[name]["mma_class"].append(line.split("/*", 2)[1].split("*/", 1)[1].strip().rstrip(";").strip())
    for k in kernels.values():
        k["count"] = len(k["mma_class"])
        k["opcodes"] = sorted({s.split()[0] for s in k["mma_class"]})
    return kernels


def check(args) -> int:
    sys.path.insert(0, str(Path.cwd()))
    from verity.ml.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32 as SAT
    from verity.ml.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32_WRAP as WRAP

    A, B, C = read_cases(Path(args.cases))
    meta = json.loads(Path(args.meta).read_text())
    Cv = as_s32(C)
    fam_of = np.empty(len(A), object)
    for f in meta:
        fam_of[f["start"]:f["start"] + f["tiles"]] = f["family"]
    draft_tiles = np.concatenate([np.arange(f["start"], f["start"] + (f["tiles"] if f["family"] in STRUCTURED else min(DRAFT_TILES, f["tiles"])))
                                  for f in meta])
    report: dict = {"tiles": len(A), "elements_per_form": len(A) * 128, "forms": {}, "families": {f["family"]: f["tiles"] for f in meta}}
    structured: dict = {}
    sums_cache: dict = {}
    ok = True
    for form in FORMS:
        fa, fb, mode = form[:2], form[2:4], form.split("_")[1]
        model = replace(SAT if mode == "sat" else WRAP, name=f"probe_{form}", dtype=fa, b_dtype=fb)
        D = np.fromfile(Path(args.words) / f"d_{form}.bin", "<u4").reshape(len(A), 16, 8)
        S = sums_cache.setdefault(form[:4], exact_sums(A, B, fa, fb))
        tot = Cv + S
        vec = words(np.clip(tot, S32_MIN, S32_MAX)) if mode == "sat" else words(tot)
        draft_mismatch, vec_vs_draft = 0, 0
        examples = []
        for n in draft_tiles:
            for i in range(16):
                for j in range(8):
                    w = model.step(int(C[n, i, j]), A[n, i].tolist(), B[n, j].tolist())
                    vec_vs_draft += int(w != vec[n, i, j])
                    if w != D[n, i, j]:
                        draft_mismatch += 1
                        if len(examples) < 8:
                            examples.append({"family": fam_of[n], "tile": int(n), "i": i, "j": j, "C": int(Cv[n, i, j]),
                                             "exact": int(tot[n, i, j]), "device": int(as_s32(D[n, i, j])), "draft": int(as_s32(np.uint32(w)))})
        bad = D != vec
        per_family = {f["family"]: int(bad[f["start"]:f["start"] + f["tiles"]].sum()) for f in meta}
        overflow = (tot > S32_MAX) | (tot < S32_MIN)
        rec = {"model": f"{(SAT if mode == 'sat' else WRAP).name} at a={fa}, b={fb}", "elements": int(D.size),
               "mismatches": int(bad.sum()), "mismatches_by_family": per_family,
               "draft_step_checked": int(len(draft_tiles) * 128), "draft_step_mismatches": draft_mismatch,
               "vectorized_vs_draft_step": vec_vs_draft, "out_of_range_elements": int(overflow.sum()),
               "unwritten_words": int((D == 0xA5A5A5A5).sum()), "first_draft_mismatches": examples}
        if bad.any() and not examples:
            n, i, j = map(int, np.argwhere(bad)[0])
            rec["first_mismatch"] = {"family": fam_of[n], "C": int(Cv[n, i, j]), "exact": int(tot[n, i, j]), "device": int(as_s32(D[n, i, j])),
                                     "model": int(as_s32(vec[n, i, j]))}
        ok &= rec["mismatches"] == 0 and draft_mismatch == 0 and vec_vs_draft == 0
        if mode == "sat":
            alive = None
            for f in meta:
                if not f["family"].startswith("clamp_"):
                    continue
                side = f["family"][-1]
                for t, labels in enumerate(f["labels"]):
                    n = f["start"] + t
                    for i in range(16):
                        for j in range(8):
                            prods = (decode(fa, A[n, i]) * decode(fb, B[n, j])).tolist()
                            h = hypotheses(int(Cv[n, i, j]), prods)
                            d = int(as_s32(D[n, i, j]))
                            fits = {k for k, v in h.items() if v == d}
                            alive = fits if alive is None else alive & fits
                            label = labels[i] if side == "a" else labels[j]
                            structured.setdefault(form, {}).setdefault(f["family"], {}).setdefault(label, {})[str(int(Cv[n, i, j]))] = d
            rec["clamp_hypotheses_consistent"] = sorted(alive or [])
        report["forms"][form] = rec
    sass = sass_kernels(Path(args.sass).read_text())
    report["sass"] = {k: {"count": v["count"], "opcodes": v["opcodes"], "lines": v["mma_class"]} for k, v in sass.items()}
    report["sass_single_instruction"] = all(sass.get(f"imma_{f}", {}).get("count") == 1 for f in FORMS)
    ok &= report["sass_single_instruction"]
    report["ok"] = bool(ok)
    Path(args.json).write_text(json.dumps(report, indent=1))
    if args.structured:
        Path(args.structured).write_text(json.dumps(structured, indent=0))
    for form, rec in report["forms"].items():
        print(f"{form}: {rec['mismatches']} / {rec['elements']} mismatches, draft step {rec['draft_step_mismatches']} / {rec['draft_step_checked']}, "
              f"out of range {rec['out_of_range_elements']}, clamp {rec.get('clamp_hypotheses_consistent', '-')}")
    for k, v in report["sass"].items():
        print(f"{k}: {v['count']} x {v['opcodes']}")
    print("ok" if ok else "MISMATCH")
    return 0 if ok else 1


def evidence(args) -> int:
    """The ``tc-evidence/v1`` meta (``benchmarks/tc_probe/trust.py``'s nested ``tc_evidence``) of one form of a check run:
    its families with their element and mismatch counts against the registry model, and the form's SASS."""
    out = Path(args.out_dir)
    rep = json.loads((out / "check.json").read_text())
    meta = json.loads((out / "meta.json").read_text())
    form = rep["forms"][args.form]
    gpu = (out / "gpu.txt").read_text().splitlines()[1].split(", ")
    sass = rep["sass"][f"imma_{args.form}"]
    fams = {}
    for f in meta:
        n = f["tiles"] * 128
        fams[f["family"]] = {"elements": n, "asserted": True, "mismatches": {args.model: form["mismatches_by_family"][f["family"]]}}
    ext = fams["extremes"]["elements"]
    fams["specials"] = {"elements": ext, "asserted": False, "mismatches": {args.model: form["mismatches_by_family"]["extremes"]},
                        "note": "integer operands have no special encodings: the extremes family's constant tiles of -128, 127, 255 and 0"}
    te = {"instruction": args.instruction, "sku": gpu[2], "arch": "sm_120", "driver": gpu[3], "gpu_uuid": gpu[1],
          "nvcc": (out / "nvcc.txt").read_text().split("\n")[0].strip(), "seed": 20261015, "families": fams,
          "total_mismatches_by_model": {args.model: form["mismatches"]},
          "sass": {"single_instruction": sass["count"] == 1, "tile_kernel_mma_count": sass["count"], "expected_mma_count": 1,
                   "ok": sass["count"] == 1, "opcode": sass["opcodes"]},
          "clamp_hypotheses_consistent": form.get("clamp_hypotheses_consistent"), "form": args.form}
    doc = {"run_id": args.run, "tc_evidence": te,
           "workload_fingerprint": {"instruction": args.instruction, "model": args.model, "probe": "benchmarks/fp8_instantiation/imma_probe.py"}}
    Path(args.json).write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in te.items() if k != "families"}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--meta", required=True)
    g.add_argument("--seed", type=int, default=20261015)
    c = sub.add_parser("check")
    for a in ("--cases", "--meta", "--words", "--sass", "--json"):
        c.add_argument(a, required=True)
    c.add_argument("--structured")
    e = sub.add_parser("evidence")
    for a in ("--out-dir", "--form", "--instruction", "--model", "--run", "--json"):
        e.add_argument(a, required=True)
    a = ap.parse_args()
    if a.cmd == "evidence":
        return evidence(a)
    if a.cmd == "gen":
        A, B, C, meta = generate(a.seed)
        write_cases(Path(a.out), A, B, C)
        Path(a.meta).write_text(json.dumps(meta))
        print(f"{len(A)} tiles, {len(A) * 128:,} elements a form")
        return 0
    return check(a)


if __name__ == "__main__":
    sys.exit(main())
