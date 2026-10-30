"""EMU8: Dither's int8 words from E4M3 MMAs, bit-exactly. The host's half of `emu8.cu`.

  python emu8.py recheck GATE_DIR [--json OUT]   # every dumped prefix word of every arm against the host's int64 words
  python emu8.py summarize TIME_JSONL            # the timed items: ms an arm, ratios to imma and to cuBLASLt int8
  python emu8.py identities                      # the k3 and k4 limb identities over every int8 pair, through E4M3 codes

The limbs: x = 16 h + l with l in [-8, 7] (h = (x + 8) >> 4), m = h + l. k3 (Karatsuba) computes a k32 step's 32 MACs as
15 T1 + 16 T2 with T1 = sum(4h 4h') + sum(-l l') from C1 = 0x4B7BAB6C (ulp 1) and T2 = sum(m m') from C2 = 0x49E07A56
(ulp 1/8), so that the step's increment is 15 bits(acc1) + 2 bits(acc2) mod 2^32 (15 C1 + 2 C2 = 0); k4 chains the four products (16h)(16h'), (16h) l', l (16h'), l l' into
one accumulator from C0 = 0x4B400000. `recheck` regenerates every case's operands from emu8.cu's counter hash, checks the device's bytes
against them, recomputes every k32 prefix word in int64 and compares each arm's words with them; it also replays sampled
k32 steps of k3 and k4 through `verity.ml.circuits.silicon`'s E4M3 model, so the words are shown to be the model's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

DISTS = ("uniform", "maxlimb", "samesign", "runs", "mixed", "dither", "zero_a")
ARMS = ("imma", "k3", "k4")
C0_BITS, C1_BITS, C2_BITS = 0x4B400000, 0x4B7BAB6C, 0x49E07A56   # emu8.cu's: k4's binade, k3's acc1 and acc2
MAXLIMB = np.array([127, -127, 120, -121, 119, -120, -128, 104, -104, 8, -8, -9, 7, 112, -113, 121], np.int64)
M64 = (1 << 64) - 1


# ---- emu8.cu's counter hash and distributions --------------------------------------------------------------------------

def smix(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def hkey(seed: int, side: int, r: np.ndarray, k: np.ndarray) -> np.ndarray:
    s = np.uint64((seed * 0x9E3779B97F4A7C15) & M64)
    return smix(s ^ np.uint64(side << 62) ^ (r.astype(np.uint64) << np.uint64(24)) ^ k.astype(np.uint64))


def draw(dist: str, seed: int, side: int, rows: int, K: int) -> np.ndarray:
    """emu8.cu's `draw` for a whole operand: rows x K int8."""
    r, k = np.meshgrid(np.arange(rows, dtype=np.uint64), np.arange(K, dtype=np.uint64), indexing="ij")
    v = hkey(seed, side, r, k)
    if dist == "uniform":
        x = (v % np.uint64(255)).astype(np.int64) - 127
    elif dist == "maxlimb":
        x = MAXLIMB[(v & np.uint64(15)).astype(np.int64)]
    elif dist == "samesign":
        S = np.array([-120, -127, -128, -120], np.int64)
        x = S[(v & np.uint64(1 if side else 3)).astype(np.int64)]
    elif dist == "runs":
        s = hkey(seed ^ 0x5151, side, r, k >> np.uint64(10))
        S = np.array([127, 120, 119, 104, 121, 112, 127, 120], np.int64)
        mag = S[(v & np.uint64(7)).astype(np.int64)]
        x = np.where((s & np.uint64(1)) == 1, -mag, mag)
    elif dist == "mixed":
        S = np.array([127, 120, 119, 104], np.int64)
        mag = S[(v & np.uint64(3)).astype(np.int64)]
        x = np.where(((v >> np.uint64(8)) & np.uint64(1)) == 1, -mag, mag)
    elif dist in ("dither", "zero_a"):
        if side == 0:
            bit = ((v >> np.uint64(40)) & np.uint64(1)).astype(np.int64)
            x = bit if dist == "zero_a" else (v % np.uint64(254)).astype(np.int64) - 127 + bit
        else:
            u = hkey(seed ^ 0x7777, 2, r, k & np.uint64(31))
            um = (u & np.uint64(63)).astype(np.int64) + 1
            s = np.where(((u >> np.uint64(6)) & np.uint64(1)) == 1, -um, um)
            x = s if dist == "zero_a" else (v % np.uint64(127)).astype(np.int64) - 63 + s
    else:
        raise ValueError(f"emu8: no distribution {dist!r}")
    return x.astype(np.int8)


# ---- limbs and E4M3 ----------------------------------------------------------------------------------------------------

def e4m3_value(code: int) -> float | None:
    """The value of an E4M3 (OCP FN) byte; None for NaN."""
    s, e, f = code >> 7, (code >> 3) & 15, code & 7
    if e == 15 and f == 7:
        return None
    v = (f / 8) * 2.0 ** -6 if e == 0 else (1 + f / 8) * 2.0 ** (e - 7)
    return -v if s else v


E4M3_CODE = {}
for _c in range(256):
    _v = e4m3_value(_c)
    if _v is not None and _v == int(_v) and int(_v) not in E4M3_CODE:
        E4M3_CODE[int(_v)] = _c   # +0 (0x00) before -0 (0x80)


def limbs(x):
    x = np.asarray(x, np.int64)
    h = (x + 8) >> 4
    return h, x - 16 * h


def planes(x, arm: str, side: int):
    """The integer planes the split writes for one operand (k3: A [4h, m, -l], B [4h', m', l']; k4: [16h, l])."""
    h, l = limbs(x)
    if arm == "k3":
        return [4 * h, h + l, l if side else -l]
    return [16 * h, l]


def codes(v) -> np.ndarray:
    """E4M3 bytes of integer values; KeyError when one isn't an E4M3 integer."""
    return np.vectorize(lambda t: E4M3_CODE[int(t)], otypes=[np.uint8])(v)


def identities() -> dict:
    """k3 and k4 over every pair of int8 values: each plane value an E4M3 integer, and the recombination the product."""
    x = np.arange(-128, 128, dtype=np.int64)
    X, Y = np.meshgrid(x, x, indexing="ij")
    out = {}
    for arm in ("k3", "k4"):
        pa, pb = planes(X, arm, 0), planes(Y, arm, 1)
        for p in pa + pb:
            codes(np.unique(p))  # every plane value has an E4M3 code
        if arm == "k3":
            word = 15 * (pa[0] * pb[0] + pa[2] * pb[2]) + 16 * (pa[1] * pb[1])
            per_step = {"acc1": int(32 * np.abs(pa[0] * pb[0] + pa[2] * pb[2]).max()),
                        "acc2": int(32 * np.abs(pa[1] * pb[1]).max())}
        else:
            word = (pa[0] + pa[1]) * (pb[0] + pb[1])
            per_step = {"acc": int(32 * np.abs(word).max())}
        out[arm] = {"pairs": int(X.size), "wrong": int((word != X * Y).sum()), "max_abs_T_a_step": per_step,
                    "binade_room": 1 << 22}
    return out


# ---- the recheck --------------------------------------------------------------------------------------------------------

def exact_words(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """words[l, i, j] = sum over k < 32 (l + 1) of A[i, k] B[j, k], as s32 (int64 sums wrapped)."""
    M, K = A.shape
    N = B.shape[0]
    a = A.astype(np.int64).reshape(M, K // 32, 32)
    b = B.astype(np.int64).reshape(N, K // 32, 32)
    seg = np.einsum("ilm,jlm->lij", a, b)
    run = np.cumsum(seg, axis=0)
    return (run & 0xFFFFFFFF).astype(np.uint32).view(np.int32)


def silicon_model():
    try:
        from verity.ml.circuits.silicon import models as m
    except ImportError:  # the node's python3 runs this from the source tree, outside the workspace's environment
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from verity.ml.circuits.silicon import models as m
    return m.BLACKWELL_SM120_E4M3_M16N8K32


def replay(A: np.ndarray, B: np.ndarray, samples: int, seed: int = 0) -> dict:
    """Sampled k32 steps of k3 and k4 through the silicon model of QMMA m16n8k32: each step's increment rebuilt from the
    model's FP32 words, as emu8.cu rebuilds it, against the exact int64 increment."""
    mdl = silicon_model()
    rng = np.random.default_rng(seed)
    M, K = A.shape
    N = B.shape[0]
    out = {"model": mdl.name, "steps": 0, "k3_agree": 0, "k4_agree": 0}
    for _ in range(samples):
        i, j, l = int(rng.integers(M)), int(rng.integers(N)), int(rng.integers(K // 32))
        a, b = A[i, 32 * l:32 * l + 32].astype(np.int64), B[j, 32 * l:32 * l + 32].astype(np.int64)
        exact = int((a * b).sum())
        pa, pb = [[int(c) for c in codes(p)] for p in planes(a, "k3", 0)], [[int(c) for c in codes(p)] for p in planes(b, "k3", 1)]
        f1 = mdl.step(mdl.step(C1_BITS, pa[0], pb[0]), pa[2], pb[2])
        f2 = mdl.step(C2_BITS, pa[1], pb[1])
        w3 = (15 * f1 + 2 * f2) & 0xFFFFFFFF
        qa, qb = [[int(c) for c in codes(p)] for p in planes(a, "k4", 0)], [[int(c) for c in codes(p)] for p in planes(b, "k4", 1)]
        f = C0_BITS
        for x, y in ((0, 0), (0, 1), (1, 0), (1, 1)):
            f = mdl.step(f, qa[x], qb[y])
        w4 = (f - C0_BITS) & 0xFFFFFFFF
        out["steps"] += 1
        out["k3_agree"] += w3 == exact & 0xFFFFFFFF
        out["k4_agree"] += w4 == exact & 0xFFFFFFFF
    return out


CASE = re.compile(r"c(\d+)-([a-z_]+)-(\d+)x(\d+)x(\d+)$")
GATE_SEED = 0xE8A5


def recheck_case(d: Path, samples: int) -> dict:
    m = CASE.match(d.name)
    if not m:
        raise SystemExit(f"emu8: {d.name} is not a gate case")
    dist, M, N, K = m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
    A = np.fromfile(d / "A.bin", np.int8)
    B = np.fromfile(d / "B.bin", np.int8)
    v = {"case": d.name, "dist": dist, "M": M, "N": N, "K": K, "refusals": []}
    if A.size != M * K or B.size != N * K:
        v["refusals"].append("operands short")
        return v
    A, B = A.reshape(M, K), B.reshape(N, K)
    if not (np.array_equal(A, draw(dist, GATE_SEED, 0, M, K)) and np.array_equal(B, draw(dist, GATE_SEED, 1, N, K))):
        v["refusals"].append("the device's operands aren't the counter hash's")
    ref = exact_words(A, B)
    v["words"] = int(ref.size)
    v["max_abs_word"] = int(np.abs(ref.astype(np.int64)).max())
    h, l = limbs(A)
    v["limbs"] = {"h": [int(h.min()), int(h.max())], "l": [int(l.min()), int(l.max())],
                  "m": [int((h + l).min()), int((h + l).max())]}
    for arm in ARMS:
        w = np.fromfile(d / f"{arm}.bin", np.int32)
        if w.size != ref.size:
            v["refusals"].append(f"{arm}: {w.size} words, {ref.size} expected")
            continue
        w = w.reshape(ref.shape)
        bad = int((w != ref).sum())
        poisoned = int((w.view(np.uint32) == 0xA5A5A5A5).sum())
        v[arm] = {"wrong": bad, "poisoned": poisoned, "sha512": hashlib.sha512(w.tobytes()).hexdigest()}
        if bad:
            v["refusals"].append(f"{arm}: {bad} words differ from the int64 words")
    if all(a in v for a in ARMS):
        v["bit_identical"] = v["k3"]["sha512"] == v["imma"]["sha512"] == v["k4"]["sha512"]
        if not v["bit_identical"]:
            v["refusals"].append("the arms' words differ")
    if samples:
        v["replay"] = r = replay(A, B, samples)
        if r["k3_agree"] != r["steps"] or r["k4_agree"] != r["steps"]:
            v["refusals"].append("the silicon model's words aren't the exact increments")
    return v


def recheck(gate_dir: Path, samples: int) -> list[dict]:
    cases = sorted(p for p in gate_dir.iterdir() if p.is_dir())
    if not cases:
        raise SystemExit(f"emu8: no gate cases in {gate_dir}")
    return [recheck_case(c, samples) for c in cases]


def gate_lines(path: Path) -> list[dict]:
    """The device's gate lines; refused unless every arm of every case passed with its control rejected."""
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.startswith("{")]
    return [r for r in rows if r.get("emu8") == "gate"]


# ---- the timing summary -------------------------------------------------------------------------------------------------

GK = re.compile(r"GK arm=(\S+) reps=(\d+) ms=([\d.]+) sm_mhz=(\S+) clock_gap=([\d.]+)%(.*)")


def summarize(path: Path) -> list[dict]:
    shapes, cur = [], None
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            if r.get("emu8") == "shape":
                cur = {"M": r["M"], "N": r["N"], "K": r["K"], "steady": r["steady"], "arms": {}}
                shapes.append(cur)
        elif (g := GK.match(line)) and cur is not None:
            cur["arms"][g.group(1)] = {"ms": float(g.group(3)), "sm_mhz": g.group(4), "clock_gap": float(g.group(5)),
                                       "tags": g.group(6).strip()}
    for s in shapes:
        macs = s["M"] * s["N"] * s["K"]
        a = s["arms"]
        for name, x in a.items():
            x["tops"] = 2 * macs / (x["ms"] * 1e-3) / 1e12
            if "imma" in a:
                x["vs_imma"] = x["ms"] / a["imma"]["ms"]
            if "cublaslt_i8" in a:
                x["vs_cublaslt_i8"] = x["ms"] / a["cublaslt_i8"]["ms"]
    return shapes


def table(shapes: list[dict]) -> str:
    rows = ["| shape (M, N, K) | arm | ms | TOPS (int8-equiv.) | / imma | / cuBLASLt int8 | tags |", "|---|---|---|---|---|---|---|"]
    for s in shapes:
        for name, x in s["arms"].items():
            rows.append(f"| {s['M']}x{s['N']}x{s['K']} | {name} | {x['ms']:.4f} | {x['tops']:.1f} | "
                        f"{x.get('vs_imma', float('nan')):.3f} | {x.get('vs_cublaslt_i8', float('nan')):.3f} | {x['tags']} |")
    return "\n".join(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recheck")
    r.add_argument("gate_dir", type=Path)
    r.add_argument("--gate-log", type=Path, help="the device's gate.jsonl: every arm accepted, every control rejected")
    r.add_argument("--samples", type=int, default=64, help="k32 steps a case replayed through the silicon model")
    r.add_argument("--json", type=Path)
    s = sub.add_parser("summarize")
    s.add_argument("time_jsonl", type=Path)
    s.add_argument("--json", type=Path)
    sub.add_parser("identities")
    a = p.parse_args(argv)
    if a.cmd == "identities":
        res = identities()
        print(json.dumps(res, indent=1))
        return 0 if all(v["wrong"] == 0 for v in res.values()) else 9
    if a.cmd == "summarize":
        shapes = summarize(a.time_jsonl)
        if a.json:
            a.json.write_text(json.dumps(shapes, indent=1))
        print(table(shapes))
        return 0
    verdicts = recheck(a.gate_dir, a.samples)
    bad = [v for v in verdicts if v["refusals"]]
    if a.gate_log:
        lines = gate_lines(a.gate_log)
        dev_bad = [x for x in lines if not (x["ok"] and x["accepted"] and x["control_rejected"])]
        if len(lines) != len(verdicts) * len(ARMS) or dev_bad:
            bad.append({"case": "device", "refusals": [f"{len(dev_bad)} device gate lines not ok of {len(lines)}"]})
    if a.json:
        a.json.write_text(json.dumps(verdicts, indent=1))
    for v in verdicts:
        rp = v.get("replay", {})
        print(f"{v['case']}: words {v.get('words')} max|w| {v.get('max_abs_word')} limbs {v.get('limbs')} "
              f"identical {v.get('bit_identical')} replay k3 {rp.get('k3_agree')}/{rp.get('steps')} "
              f"k4 {rp.get('k4_agree')}/{rp.get('steps')} {'REFUSED ' + '; '.join(v['refusals']) if v['refusals'] else 'ok'}")
    print(f"RECHECK {'FAIL' if bad else 'PASS'} ({len(verdicts)} cases)")
    return 9 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
