"""ISA-ESCAPE-JOINT: Lemma J, the slot-step identity, per device, mechanically (note:20261019T0745Z-draft-fp-joint).

FPX-MACHINE (`isa_escape_fp.py`) leaves open the joint routes on the tensor forms faster than the credited row: 19, 9 and
21 (form, word) pairs on sm_120, H200 and B200 whose rate over the credited row exceeds every pass bound a counting or rank
argument gives (`rank_gap > 1`, worst 2.47x), and the rank floor's closures on FP kinds (21, 18, 25), which read an
exactness no vendor states. Those are gaps in multiply-accumulates. The TS model counts words written (A6,
`WordsWritten`): an MMA heads at most one committed word a slot. So what bounds a form's words a second is its time a
slot. An instruction of form κ at logical depth K (P = K physical positions a slot dense, K/2 for 2:4) at its kind's
sheet rate R (dense MACs an SM a clock, A3) holds the tensor pipe 4·P/R SMSP-cycles a slot (`Sheet.machine`'s tensor
row over its slots).

(J1) On each device every form's full-depth instruction (its kind's deepest K) takes exactly the credited instruction's
time a slot, P/R = K_c/R_c; FP64 is slower. Shapes shallower than their kind's full depth take half a step or less at
A3's rate, so for them (J1) is premise (M): such an instruction holds the pipe a full step anyway. (M) is measured on
sm_120 for IMMA.16816 (k16 and m8n8k16), IMMA.SP.16832 and HMMA.1684 (TENSOR M1), not for the other shallow sm_120 shapes,
and not at all on H200 and B200 ((M')). On sm_89 it is false (IMMA.16816 runs at the MAC rate), which is why it is a
premise. (J3) With one word a slot (A6) a word costs at least one step on any form, and is credited L/R_c, so its gain
is at most L/K_c: at L = K_c every form at best ties, joint encodings and several elements a slot included.

Each pair's verdict: TIE (full depth exactly the step, and every shallower shape at least a step at A3's rate or by
measured (M)); TIE-ON-M (the same, but a shallower shape rests on (M) unmeasured on this device); CLOSED (slower a slot);
OPEN (a form faster a slot than the step). It reads only the vendor shapes (PTX ISA 9.4: mma / mma.sp, wgmma, tcgen05.mma),
the sheets' rates (`isa_escape_fp.RATES`) and M1's measured cycles; no FP semantics, no rank floor, no premise (a).

S14's class is what J leaves: a packed slot, w > 1 committed words through one slot's output, each extracted by a lane
instruction (a vector word, at least ½ a lane instruction by A6). Packing trades tensor slots for issue: the route's
load a word is max(step/w, the MMA's issue share/w + (fold + decode)/32) against the credited route's at w = 1 (and on
sm_120 the VF3e joint rows). B200: closed by A6 alone, since ½/32 is the step. H200: closed iff the fold costs at least
½ a word (FOLD-OPT's 3 does). sm_120: open for decodes between ½ and the exact break-even, 1 − 1/(4w) on the sheet at
FOLD-OPT's fold. The one construction, the chained radix packing (3 words in 2 IMMA slots, 5 SASS instructions to
decode), is decoded bit-exactly here and loses on all three; on sm_120 that loss rests on FOLD-OPT.

Run: `python isa_escape_joint.py [--mutations] [--verdict PATH]`. Stdlib + exact_kit and its sibling `isa_escape_fp`;
reads nothing.
"""
from __future__ import annotations

import functools
import random
import sys
from fractions import Fraction as Q

import isa_escape_fp as fp
from exact_kit.checks import Checks

DEVICES = fp.DEVICES

# ---- the vendor shapes (PTX ISA 9.4), as (kind, 2:4) -> each shape's logical m, n, k -------------------------------------
# mma.sync / mma.sp on sm_80 and later (e4m3/e5m2 from sm_89, f64 m16n8k{4,8,16} from sm_90; s4/b1 have no sheet rate on
# these three devices and are isa_escape.py's). A 2:4 shape's k is logical: twice its physical positions.
MMA_SYNC = {
    ("s8", False): ("m16n8k32", "m16n8k16", "m8n8k16"),
    ("f16", False): ("m16n8k16", "m16n8k8", "m8n8k4"),
    ("bf16", False): ("m16n8k16", "m16n8k8"),
    ("tf32", False): ("m16n8k8", "m16n8k4"),
    ("f64", False): ("m16n8k16", "m16n8k8", "m16n8k4", "m8n8k4"),
    ("e4m3", False): ("m16n8k32", "m16n8k16"),
    ("e5m2", False): ("m16n8k32", "m16n8k16"),
    ("s8", True): ("m16n8k64", "m16n8k32"),
    ("f16", True): ("m16n8k32", "m16n8k16"),
    ("bf16", True): ("m16n8k32", "m16n8k16"),
    ("tf32", True): ("m16n8k16", "m16n8k8"),
    ("e4m3", True): ("m16n8k64",),
    ("e5m2", True): ("m16n8k64",),
}
# sm_120a's mma.sync .kind:: forms: f8f6f4 (k32, .sp k64), mxf8f6f4 .block_scale (k32, .sp k64), mxf4 and mxf4nvf4
# .block_scale (k64, .sp k128).
MMA_SM120A = {**{(k, False): ("m16n8k32",) for k in ("e4m3", "e5m2", "e3m2", "e2m3", "e2m1", "mxf8")},
              **{(k, True): ("m16n8k64",) for k in ("e4m3", "e5m2", "e3m2", "e2m3", "e2m1", "mxf8")},
              **{(k, False): ("m16n8k64",) for k in ("mxf4", "nvf4")},
              **{(k, True): ("m16n8k128",) for k in ("mxf4", "nvf4")}}
# wgmma (sm_90a): m64nNk<K>, N up to 256, one K per type; .sp at twice it. (b1 has no sheet rate.)
WGMMA = {"f16": 16, "bf16": 16, "tf32": 8, "e4m3": 32, "e5m2": 32, "s8": 32}
# tcgen05.mma (sm_100a): M up to 128 (cta_group::1), N up to 256, one K per kind (kind::tf32 8, kind::f16 16,
# kind::f8f6f4 and kind::i8 32, kind::mxf8f6f4 32, kind::mxf4 and kind::mxf4nvf4 64); .sp at twice it.
TCGEN05 = {"tf32": 8, "f16": 16, "bf16": 16, "e4m3": 32, "e5m2": 32, "e3m2": 32, "e2m3": 32, "e2m1": 32, "s8": 32,
           "mxf8": 32, "mxf4": 64, "nvf4": 64}

# The sheet rates are isa_escape_fp's (A3). sm_120's table lacks the block-scaled FP8 kind (QMMA.SF, kind::mxf8f6f4),
# which the sheet runs at its FP8 rate; FP64 has no tensor row on sm_120 (DMMA.8x8x4 is 584 SMSP-cycles a MMA, M1).
EXTRA_RATES = {"sm_120": {"mxf8": 1024}}

# Premise (M), measured: SMSP-cycles a MMA alone, and its slots. TENSOR M1 (note:20261017T0321Z-draft-eps1-tensor; runs
# r20261017-040443-f33c, r20261017-041103-d35d, r20261017-041754-7630, all 188 SMs, SASS read per loop). The control is
# sm_89's (EPS1-ADA, r20261017-063245-fc25), where the shallow form runs at the MAC rate.
MEASURED = {
    "sm_120": {("s8", False, "m16n8k16"): ("IMMA.16816", Q("16.25"), 128),
               ("s8", False, "m8n8k16"): ("IMMA.16816, 64 of its 128 slots kept", Q("17.25"), 64),
               ("s8", True, "m16n8k32"): ("IMMA.SP.16832", Q("16.25"), 128),
               ("tf32", False, "m16n8k4"): ("HMMA.1684.F32.TF32", Q("16.38"), 128)},
}
SM89 = {"m16n8k32": ("IMMA.16832", Q("16.005"), 128), "m16n8k16": ("IMMA.16816", Q("8.006"), 128)}
CREDITED = {"sm_120": ("s8", "e4m3", "nvf4"), "sm_90a": ("s8", "e4m3"), "sm_100a": ("s8", "e4m3", "nvf4")}


def mnk(shape: str) -> tuple[int, int, int]:
    m, rest = shape[1:].split("n")
    n, k = rest.split("k")
    return int(m), int(n), int(k)


def rate(dev: str, kind: str) -> int:
    return EXTRA_RATES.get(dev, {}).get(kind, fp.RATES[dev].get(kind, 0))


def shapes(dev: str) -> dict[tuple[str, bool], list[tuple[str, str]]]:
    """Every (kind, 2:4) form on `dev` with its shapes, as (family, shape)."""
    out: dict[tuple[str, bool], list[tuple[str, str]]] = {}

    def add(kind, sparse, family, shape):
        out.setdefault((kind, sparse), []).append((family, shape))

    for (kind, sparse), ss in MMA_SYNC.items():
        for s in ss:
            add(kind, sparse, "mma.sp" if sparse else "mma.sync", s)
    if dev == "sm_120":
        for (kind, sparse), ss in MMA_SM120A.items():
            for s in ss:
                add(kind, sparse, "mma.sp .kind" if sparse else "mma.sync .kind", s)
    if dev == "sm_90a":
        for kind, k in WGMMA.items():
            add(kind, False, "wgmma", f"m64n256k{k}")
            add(kind, True, "wgmma.sp", f"m64n256k{2 * k}")
    if dev == "sm_100a":
        for kind, k in TCGEN05.items():
            add(kind, False, "tcgen05.mma", f"m128n256k{k}")
            add(kind, True, "tcgen05.mma.sp", f"m128n256k{2 * k}")
    return out


def positions(shape: str, sparse: bool) -> int:
    """Physical positions a slot: the logical k, halved for 2:4."""
    k = mnk(shape)[2]
    return k // 2 if sparse else k


def slot_time(dev: str, kind: str, sparse: bool, shape: str) -> Q | None:
    """SMSP-cycles a slot at the sheet rate (A3): 4·P/R, R the dense rate. None: the sheet gives the kind no rate."""
    r = rate(dev, kind)
    return Q(4 * positions(shape, sparse), r) if r else None


def step(dev: str, credit: str) -> Q:
    """The credited instruction's time a slot: its kind's full dense depth over its rate."""
    k_c = max(mnk(s)[2] for _, s in shapes(dev)[(credit, False)])
    return Q(4 * k_c, rate(dev, credit))


def full_depth(dev: str, kind: str, sparse: bool) -> int:
    return max(mnk(s)[2] for _, s in shapes(dev)[(kind, sparse)])


def grade_form(dev: str, kind: str, sparse: bool, credit: str) -> dict:
    """Lemma J on one form against one credited row: its full-depth slot time against the step, and each shallower
    shape's source (A3 alone, measured (M), or (M) unmeasured)."""
    st = step(dev, credit)
    forms = shapes(dev)[(kind, sparse)]
    k_full = full_depth(dev, kind, sparse)
    full = [(f, s) for f, s in forms if mnk(s)[2] == k_full]
    t_full = slot_time(dev, kind, sparse, full[0][1])
    shallow = []
    for fam, s in forms:
        if mnk(s)[2] == k_full:
            continue
        t = slot_time(dev, kind, sparse, s)
        if t is None or t >= st:
            src = "A3"
        else:
            meas = MEASURED.get(dev, {}).get((kind, sparse, s))
            src = "M measured" if meas and meas[1] / meas[2] >= st else "M unmeasured"
        shallow.append({"family": fam, "shape": s, "a3_slot": t, "source": src})
    if t_full is None:
        outcome = "NO-ROW"
    elif t_full < st:
        outcome = "OPEN"
    elif t_full > st:
        outcome = "CLOSED"
    elif any(x["source"] == "M unmeasured" for x in shallow):
        outcome = "TIE-ON-M"
    else:
        outcome = "TIE"
    return {"device": dev, "form": kind + (".sp" if sparse else ""), "credit": credit, "step": st, "K_full": k_full,
            "full_families": sorted({f for f, _ in full}), "full_slot": t_full, "shallow": shallow, "outcome": outcome,
            "gain_at_Kc": None if t_full is None else st / t_full}


@functools.lru_cache(maxsize=None)
def fpx_pairs(dev: str, gap: str = "rank_gap") -> tuple[tuple[str, str], ...]:
    """FPX-MACHINE's open joint pairs (form, word) on `dev`: `rank_gap > 1` (the 19/9/21), or with `gap="fp_rank"` the
    rank floor's closures on FP kinds (21/18/25), which read an exactness no vendor states. Read once, at the sheets'
    rates: a mutant perturbs the facts J reads, not FPX's list."""
    rows = [r for r in fp.verdicts() if r["device"] == dev]
    if gap == "rank_gap":
        return tuple(sorted({(r["route"], r["word"]) for r in rows if r["rank_gap"] > 1}))
    return tuple(sorted({(r["route"], r["word"]) for r in rows
                         if r["joint_gap"] > 1 and r["rank_gap"] <= 1 and r["route"].split(".")[0] != "s8"}))


def pair_verdict(dev: str, route: str, word: str) -> dict:
    kind, sparse = route.split(".")[0], route.endswith(".sp")
    g = grade_form(dev, kind, sparse, fp.credit_row(word, dev))
    return g | {"word": word}


def fmt_pair(v: dict) -> str:
    bits = f"{v['form']}:{v['word']} vs {v['credit']} (K_c {full_depth(v['device'], v['credit'], False)}) -> {v['outcome']}"
    m = [f"{x['family']} {x['shape']} ({x['source']})" for x in v["shallow"] if x["source"] != "A3"]
    return bits + (f" [{'; '.join(m)}]" if m else "")


#: Lemma J's verdicts on FPX's joint pairs, per device: (TIE, TIE-ON-M). Nothing else occurs.
EXPECT = {"sm_120": (19, 0), "sm_90a": (6, 3), "sm_100a": (19, 2)}


def families_disagree(dev: str) -> list[str]:
    """Kinds where some family's deepest shape is not the kind's full depth (mma.sync, wgmma and tcgen05 must agree)."""
    fam_k: dict[tuple[str, bool, str], set[int]] = {}
    for (kind, sparse), fs in shapes(dev).items():
        for fam, s in fs:
            fam_k.setdefault((kind, sparse, fam.split(" ")[0].split(".")[0]), set()).add(mnk(s)[2])
    return sorted(f"{k}{'.sp' if sp else ''} ({fam})" for (k, sp, fam), ks in fam_k.items()
                  if max(ks) != full_depth(dev, k, sp))


def pair_counts(dev: str) -> tuple[int, int, int]:
    """(TIE, TIE-ON-M, anything else) over FPX's joint pairs on `dev`."""
    outs = [pair_verdict(dev, r, w)["outcome"] for r, w in fpx_pairs(dev)]
    return outs.count("TIE"), outs.count("TIE-ON-M"), len(outs) - outs.count("TIE") - outs.count("TIE-ON-M")


def j_holds(dev: str) -> bool:
    """Lemma J's claims on `dev` as recorded: the families agree, no form is faster a slot than the step, and FPX's pairs
    get exactly EXPECT's verdicts."""
    no_faster = all(grade_form(dev, k, sp, "s8")["outcome"] != "OPEN" for (k, sp) in shapes(dev))
    return not families_disagree(dev) and no_faster and pair_counts(dev) == (*EXPECT[dev], 0)


# ---- S14's class: packed slots (more than one committed word through one slot's output) -------------------------------
# A packed word's head is the instruction that extracts it (Definition 2), a lane instruction, so it is a vector word:
# A6 gives it at least 1/2 lane-instruction (a lane writes at most 64 bits), ScalarRoute (I3) prices it at q = 2.03.
#
# The chained radix packing (G2): rows r1, r2, r3 and packed A rows x1 + R·x2 and x2 − R·x3, both int8, against one B
# column write s1 = y1 + R·y2 and s2 = y2 − R·y3 in two IMMA slots (G2 writes + in s2; the − makes t's quotient y3).
# t = s1 − R·s2 = y1 + R²·y3, so the map has kernel Z·(R², −R, −1) and is injective on any box where y1 spreads less
# than R². The decode below is a straight-line program in SASS's integer forms (IMAD: 32-bit a·b + c; IMAD.WIDE: signed
# 32x32 + 64 -> 64, a register pair; SHF.R.S32.HI: an arithmetic shift of the high word); its constants (M, the pair C)
# are set once a kernel. Each tree of packed rows gains at most one word over its edges, so 3/2 is this family's best.

INSTANCES = {  # name: (R, operand entries, B entries, L, the words' range)
    "H1 zero instance, L = 96": (126, (0, 1), range(-64, 66), 96, (-64 * 96, 65 * 96)),
    "H1 zero instance, L = 32": (126, (0, 1), range(-64, 66), 32, (-64 * 32, 65 * 32)),
    "design A, 2-bit entries": (41, (0, 1, 2, 3), range(0, 4), 32, (0, 288)),
}
FOLD = 3   # duo's fold step: three lane instructions a word (FOLD-OPT's per-word kernel, fold_opt.py)
JI3E = {"JI3e_0": (Q("8.115163161"), Q(1)), "JI3e_1": (Q("4510.17215313632") / Q("300.08"), Q("0.317063107")),
        "JI3e_2": (Q("4662.85212194032") / Q("300.08"), Q("0.171692378"))}   # sm_120's VF3e rows (tensor_kinds.JI3E)
MMA_ISSUE = {"sm_120": Q(1, 128), "sm_90a": Q(4, 64 * 256), "sm_100a": Q(1, 128 * 256)}   # warp instructions a slot
S32 = 1 << 32


def wrap32(v: int) -> int:
    v %= S32
    return v - S32 if v >= S32 // 2 else v


def imad(a: int, b: int, c: int) -> int:
    return wrap32(a * b + c)


def imad_wide(a: int, b: int, c64: int) -> tuple[int, int]:
    v = (wrap32(a) * wrap32(b) + c64) % (1 << 64)
    hi = v >> 32
    return wrap32(v), (hi - S32 if hi >= S32 // 2 else hi)


def magic(r2: int, lo: int, lo3: int, hi3: int) -> tuple[int, int, int, int]:
    """(M, K, B, C): floor((t − lo)/R²) = floor((t·M + C)/2^K) for every t = y1 + R²·y3, y1 − lo in [0, R²), y3 in
    [lo3, hi3]. B shifts the numerator nonnegative (n = t − lo + R²·B < T), and the pair C absorbs it:
    floor(X/2^K) − B = floor((X − B·2^K)/2^K). Exact while T·(M·R² − 2^K) < 2^K (n = qR² + r, worst r = R² − 1)."""
    b = -lo3
    t_max = r2 * (hi3 + b + 1)
    for k in range(32, 64):
        m = -(-(1 << k) // r2)
        if m < 1 << 31 and t_max * (m * r2 - (1 << k)) < 1 << k:
            return m, k, b, (r2 * b - lo) * m - (b << k)
    raise ValueError("no magic")


def decode(s1: int, s2: int, R: int, M: int, K: int, C: int) -> tuple[list[int], int]:
    """The five-instruction decode: (y1, y2, y3) and its instruction count."""
    t = imad(s2, -R, s1)                       # IMAD          t = y1 + R²·y3
    _, hi = imad_wide(t, M, C)                 # IMAD.WIDE     (t·M + C) / 2^32
    y3 = hi >> (K - 32)                        # SHF.R.S32.HI  floor((t − lo)/R²) = y3
    y2 = imad(y3, R, s2)                       # IMAD          s2 + R·y3
    y1 = imad(y3, -R * R, t)                   # IMAD          t − R²·y3
    return [y1, y2, y3], 5


def chained_ok(name: str, n: int, dm: int = 0) -> tuple[bool, dict]:
    """The packing on one instance: int8 operands, injectivity, the division at every quotient, and n cases decoded
    bit-exactly (the box's 4 corners, then random). `dm` offsets the magic multiplier (a mutant's)."""
    R, xs, bs, L, (lo, hi) = INSTANCES[name]
    rng = random.Random(14)
    packs = {x1 + R * x2 for x1 in xs for x2 in xs} | {x2 - R * x3 for x2 in xs for x3 in xs}
    if not all(-128 <= p <= 127 for p in packs) or not all(-128 <= v <= 127 for v in bs):
        return False, {"int8": sorted(packs)}
    if hi - lo >= R * R:
        return False, {"spread": hi - lo, "R2": R * R}
    M, K, B, C = magic(R * R, lo, lo, hi)
    M += dm
    C += dm * (R * R * B - lo)
    # the division, at every quotient's two worst numerators (r = 0 and r = R² − 1) and the range's ends
    bad = [q for q in range(lo, hi + 1) for y1 in (lo, lo + R * R - 1)
           if (imad_wide(y1 + R * R * q, M, C)[1] >> (K - 32)) != q]
    worst = 0
    for case in range(n):
        if case < 4:   # the box's corners: every product at its extreme
            ext = (min(xs), max(xs))
            rows = [[ext[(case >> i) & 1]] * L for i in range(3)]
            col = [(min(bs), max(bs))[case & 1]] * L
        else:
            rows = [[rng.choice(xs) for _ in range(L)] for _ in range(3)]
            col = [rng.choice(bs) for _ in range(L)]
        y = [sum(a * b for a, b in zip(r, col)) for r in rows]
        s1 = wrap32(sum((a + R * b) * c for a, b, c in zip(rows[0], rows[1], col)))
        s2 = wrap32(sum((a - R * b) * c for a, b, c in zip(rows[1], rows[2], col)))
        got, _ = decode(s1, s2, R, M, K, C)
        worst = max(worst, abs(s1), abs(s2), abs(y[0] + R * R * y[2]))
        if got != y:
            bad.append((y, got))
    return not bad and worst < 1 << 31, {"R": R, "M": M, "K": K, "cases": n, "wrong": bad[:3], "max_s32": worst}


def collision(R: int, lo: int, hi: int) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Two word triples in the box [lo, hi]³ with the same two slots, once y1's spread reaches R²: a and
    a + (R², −R, −1), the map's kernel."""
    if hi - lo < R * R:
        return None
    a, b = (lo, hi, hi), (lo + R * R, hi - R, hi - 1)
    slots = [(y[0] + R * y[1], y[1] - R * y[2]) for y in (a, b)]
    return (a, b) if slots[0] == slots[1] else None


def per_word(dev: str, w: Q | None, e: Q, f: Q, joint: bool) -> dict[str, Q]:
    """A route's load a word, in SMSP-cycles: the tensor facet (Lemma J: the step a slot, over w words a slot) and A2's
    issue (1 a warp instruction: the MMA's share of its slots, and the fold's f and the decode's e lane instructions
    over 32 lanes); on sm_120 with `joint`, also the VF3e joint rows (a an MMA, b a warp instruction of lane
    instructions). `w=None` is the limit w → ∞, which every packing's load is above (each row falls with w)."""
    mma = Q(0) if w is None else MMA_ISSUE[dev] / w
    out = {"tensor": Q(0) if w is None else step(dev, "s8") / w, "issue": mma + (f + e) / 32}
    if joint:
        out |= {name: a * mma + b * (f + e) / 32 for name, (a, b) in JI3E.items()}
    return out


def packing_gain(dev: str, w: Q | None, e: Q, f: Q = Q(FOLD), joint: bool = False) -> Q:
    """The credited route's load a word over the packed route's, on these rows: > 1 only where packing could beat the
    credited route. Every other row of `Sheet.machine` (the pipes, shared memory) is charged by lane instructions,
    which the decode only adds, so a gain <= 1 here is a loss on the data-sheet machine unless the MMA itself binds a
    row these leave out; a gain > 1 is a window, not a construction."""
    u = max(per_word(dev, Q(1), Q(0), f, joint).values())
    return u / max(per_word(dev, w, e, f, joint).values())


def breakeven(dev: str, w: Q | None, f: Q = Q(FOLD), joint: bool = False) -> Q | None:
    """The decode cost a word below which packing at w words a slot gains, exactly (every row is affine in e). None: no
    decode cost gains, since a row the decode doesn't touch already binds."""
    u = max(per_word(dev, Q(1), Q(0), f, joint).values())
    p0, p1 = per_word(dev, w, Q(0), f, joint), per_word(dev, w, Q(1), f, joint)
    if any(p0[r] >= u for r in p0 if p1[r] == p0[r]):
        return None
    return min((u - p0[r]) / (p1[r] - p0[r]) for r in p0 if p1[r] > p0[r])


A6_FLOOR = Q(1, 2)            # a lane instruction writes at most 64 bits (A6): a vector-headed word costs at least ½ of one
I3_Q = Q("2.03")              # ScalarRoute (I3)'s price of a vector word, in lane instructions
CHAIN = (Q(3, 2), Q(5, 3))    # the chained packing: 3 words in 2 slots, the five-instruction decode for the 3
CASES = 600


def closed(dev: str, f: Q) -> bool:
    """No packing gains on `dev` at fold f: the gain's supremum (w → ∞, the decode at A6's floor) is at most 1."""
    return packing_gain(dev, None, A6_FLOOR, f) <= 1


def s14_claims() -> list[tuple[str, bool, object]]:
    """S14's packed-slot class, as (claim, holds, detail)."""
    out = []
    for name in INSTANCES:
        ok, info = chained_ok(name, CASES)
        out.append((f"S14 chained packing, {name}: three words in two IMMA slots, every packed A entry int8, y1's spread "
                    f"below R², the division exact at every quotient, and {CASES} cases (the box's corners first) decoded "
                    "bit-exactly by the five-instruction decode with no s32 overflow", ok, info))
    R = INSTANCES["H1 zero instance, L = 96"][0]
    lmax = max(L for L in range(1, 4 * R) if 129 * L < R * R)
    hit = collision(R, -64 * (lmax + 1), 65 * (lmax + 1))
    out.append((f"S14: on H1's zero instance (R = {R}) the packing is injective exactly up to L = {lmax} (129·L < R²); at "
                f"L = {lmax + 1} two word triples share both slots", lmax == 123 and hit is not None
                and collision(R, -64 * lmax, 65 * lmax) is None, hit))

    w, e = CHAIN
    out.append(("S14: the chained packing's decode is 5/3 lane instructions a word, above A6's floor of 1/2 and below "
                "ScalarRoute (I3)'s q = 2.03, so I3 is false for these vector-headed words", A6_FLOOR <= e < I3_Q, e))
    gains = {d: {"sheet": packing_gain(d, w, e)} for d in DEVICES}
    gains["sm_120"]["VF3e"] = packing_gain("sm_120", w, e, joint=True)
    out.append(("S14: the chained packing loses on all three devices at FOLD-OPT's fold of 3 a word (credited over packed "
                "load a word at most 1; sm_120 on the sheet's rows and on VF3e's joint rows)",
                all(g <= 1 for gs in gains.values() for g in gs.values()),
                {d: {k: f"{float(v):.4f}" for k, v in gs.items()} for d, gs in gains.items()}))
    out.append(("S14: on H200 and B200 it loses with no fold at all (its decode outweighs the slot it saves); on sm_120 its "
                "loss rests on FOLD-OPT: at a fold of 1/2 a word it gains 3/2 on the sheet",
                all(packing_gain(d, w, e, Q(0)) <= 1 for d in ("sm_90a", "sm_100a"))
                and packing_gain("sm_120", w, e, Q(1, 2)) == Q(3, 2), packing_gain("sm_120", w, e, Q(1, 2))))

    smem = {"sm_90a": (64, 256, 32), "sm_100a": (128, 256, 32)}   # wgmma, tcgen05.mma s8: M, N, K
    per_slot = {d: Q((m + n) * k, m * n) / 32 for d, (m, n, k) in smem.items()}
    out.append(("S14: the rows left out don't bind the credited route on H200 and B200: Sheet's shared-memory row (A5), "
                "which wgmma and tcgen05.mma load with their s8 operands, (M + N)·K bytes an instruction read once, over "
                "32 a slot, is below the step; every pipe row is charged by lane instructions, which a decode only adds",
                all(per_slot[d] < step(d, "s8") for d in per_slot), {d: str(v) for d, v in per_slot.items()}))

    st, folds = step("sm_100a", "s8"), (Q(0), Q(1, 9), Q(1, 2), Q(1), Q(FOLD))
    out.append(("S14 closed on B200 by A6 alone: a packed word's decode, at least 1/2 a lane instruction, is 1/64 "
                "SMSP-cycle of issue, the step a slot it saves, and more than the MMA's issue share, so no packing gains "
                "at any words a slot (w → ∞) or fold; at best a tie",
                A6_FLOOR / 32 >= st and A6_FLOOR / 32 > MMA_ISSUE["sm_100a"]
                and all(closed("sm_100a", f) for f in folds), st))
    fstar = 32 * step("sm_90a", "s8") - A6_FLOOR
    out.append(("S14 on H200: no packing gains iff the fold costs at least 1/2 lane instruction a word, which puts a "
                "packed word's issue at a full step; FOLD-OPT's 3 gives it, DuoFoldFanIn's floor gives it only for "
                "three-source instructions (sm_90's widest, f = 10, give 1/9 a word)",
                fstar == Q(1, 2) and closed("sm_90a", fstar) and not closed("sm_90a", fstar - Q(1, 64))
                and closed("sm_90a", Q(FOLD)) and not closed("sm_90a", Q(1, 9)), fstar))

    be = {(j, w_): breakeven("sm_120", w_, joint=j) for j in (False, True) for w_ in (Q(3, 2), Q(2), None)}
    a0, b0 = JI3E["JI3e_0"]
    q_star = 32 * a0 / (b0 * 128)
    top = {(j, w_): packing_gain("sm_120", w_, A6_FLOOR, joint=j) for j in (False, True) for w_ in (Q(3, 2), None)}
    out.append(("S14 open on sm_120: at FOLD-OPT's fold, a packing at w words a slot gains iff its decode is under "
                "1 − 1/(4w) a word on the sheet (5/6 at the chain's 3/2, 1 as w → ∞) and under VF3e's break-even (0.676 "
                "at 3/2, VectorThresholdD1's q* = 32α/(βω) = 2.029 as w → ∞); A6 floors a decode only at 1/2, so decodes "
                "in [1/2, break-even) stay open, worth at most 12/11 (sheet) and 1.036 (VF3e) at w = 3/2, 8/7 and 1.437 "
                "as w → ∞",
                be[(False, Q(3, 2))] == Q(5, 6) and be[(False, Q(2))] == Q(7, 8) and be[(False, None)] == 1
                and be[(True, None)] == q_star and abs(q_star - Q("2.029")) < Q(1, 2000)
                and abs(be[(True, Q(3, 2))] - Q("0.676")) < Q(1, 2000) and all(v > A6_FLOOR for v in be.values())
                and all(e > be[(j, w)] for j in (False, True)) and top[(False, Q(3, 2))] == Q(12, 11)
                and top[(False, None)] == Q(8, 7),
                {"break-even": {f"{'VF3e' if j else 'sheet'} w={w_ or 'inf'}": f"{float(v):.4f}" for (j, w_), v in be.items()},
                 "gain at e=1/2": {f"{'VF3e' if j else 'sheet'} w={w_ or 'inf'}": f"{float(v):.4f}"
                                   for (j, w_), v in top.items()}}))
    return out


def main() -> int:
    checks = Checks()

    # -- the shapes and the step ----------------------------------------------------------------------------------
    for dev in DEVICES:
        steps = {c: step(dev, c) for c in CREDITED[dev]}
        checks.check(f"{dev}: every credited row has one time a slot, 4·K_c/R_c SMSP-cycles",
                     len(set(steps.values())) == 1, {c: str(s) for c, s in steps.items()})
        disagree = families_disagree(dev)
        checks.check(f"{dev}: every family's deepest shape of a kind is the kind's full depth (mma.sync, wgmma and "
                     "tcgen05 agree)", not disagree, disagree)
        sparse_k = [k for (k, sp) in shapes(dev) if sp and (k, False) in shapes(dev)
                    and full_depth(dev, k, True) != 2 * full_depth(dev, k, False)]
        checks.check(f"{dev}: every 2:4 form's full depth is twice its dense kind's (the same physical positions)",
                     not sparse_k, sparse_k)
    checks.check("the steps: 1/8 SMSP-cycle a slot on sm_120 (16 a m16n8k32), 1/32 on H200, 1/64 on B200",
                 [step(d, "s8") for d in DEVICES] == [Q(1, 8), Q(1, 32), Q(1, 64)])

    # -- (J1) at full depth, every form on every device -------------------------------------------------------------
    for dev in DEVICES:
        graded = [grade_form(dev, k, sp, "s8") for (k, sp) in sorted(shapes(dev))]
        ties = sorted(g["form"] for g in graded if g["outcome"] in ("TIE", "TIE-ON-M"))
        slower = sorted(g["form"] for g in graded if g["outcome"] == "CLOSED")
        norow = sorted(g["form"] for g in graded if g["outcome"] == "NO-ROW")
        faster = sorted(g["form"] for g in graded if g["outcome"] == "OPEN")
        checks.check(f"{dev} (J1): every tensor form's full-depth instruction takes exactly the step a slot, FP64 "
                     "slower; none faster", not faster and all(f.startswith("f64") for f in slower + norow),
                     {"ties": len(ties), "slower": slower, "no_sheet_row": norow, "faster": faster})
    checks.check("FP64 is slower a slot where the sheet rates it: K/R = 16/256 on H200 and B200 (generous rate)",
                 all(grade_form(d, "f64", False, "s8")["outcome"] == "CLOSED" for d in ("sm_90a", "sm_100a")))
    checks.check("sm_120 has no FP64 tensor row: Sheet.machine charges a class of peak 0 nothing on the tensor facet "
                 "(x/0 = 0), so a statement over the sheet needs peak > 0 for every MMA class; M1's DMMA.8x8x4 at 584 "
                 "SMSP-cycles for 64 slots is 73 steps a slot (measured)",
                 grade_form("sm_120", "f64", False, "s8")["outcome"] == "NO-ROW" and Q(584, 64) / step("sm_120", "s8") > 1)

    # -- premise (M): the shallower shapes --------------------------------------------------------------------------
    for dev in DEVICES:
        rows = []
        for (k, sp) in sorted(shapes(dev)):
            g = grade_form(dev, k, sp, "s8")
            rows += [(g["form"], x) for x in g["shallow"]]
        meas = sorted(f"{f} {x['shape']}" for f, x in rows if x["source"] == "M measured")
        unmeas = sorted(f"{f} {x['family']} {x['shape']}" for f, x in rows if x["source"] == "M unmeasured")
        half = all(x["a3_slot"] is None or x["a3_slot"] >= step(dev, "s8") or x["a3_slot"] * 2 <= step(dev, "s8")
                   for _, x in rows)
        checks.check(f"{dev} (M): the shallower shapes take at most half a step at A3's rate; {len(meas)} rest on "
                     f"measured (M), {len(unmeas)} on (M) unmeasured", half, {"measured": meas, "unmeasured": unmeas})
    for (kind, sparse, s), (sass, cyc, slots) in MEASURED["sm_120"].items():
        checks.check(f"(M) on sm_120, {sass} ({kind}{'.sp' if sparse else ''} {s}): {cyc} SMSP-cycles for {slots} "
                     "slots is at least a step a slot", cyc / slots >= step("sm_120", "s8"), str(cyc / slots))
    checks.check("control: (M) is false on sm_89 (EPS1-ADA): IMMA.16816 runs at the MAC rate, 8.006 SMSP-cycles for "
                 "128 slots against IMMA.16832's 16.005, half a step a slot, so (M) is a premise, not a law",
                 SM89["m16n8k16"][1] / 128 < Q(1, 2) * SM89["m16n8k32"][1] / 128 * Q(101, 100))

    # -- FPX's open joint pairs, per device ----------------------------------------------------------------------------
    expect = {"sm_120": 19, "sm_90a": 9, "sm_100a": 21}
    for dev in DEVICES:
        pairs = fpx_pairs(dev)
        vs = [pair_verdict(dev, r, w) for r, w in pairs]
        by = {o: [fmt_pair(v) for v in vs if v["outcome"] == o] for o in ("TIE", "TIE-ON-M", "CLOSED", "OPEN", "NO-ROW")}
        checks.check(f"{dev}: FPX's {expect[dev]} open joint pairs (rank_gap > 1) are listed", len(pairs) == expect[dev],
                     len(pairs))
        checks.check(f"{dev}: Lemma J on FPX's joint pairs: {len(by['TIE'])} TIE, {len(by['TIE-ON-M'])} TIE-ON-M (a "
                     "shallower shape rests on (M) unmeasured here), none OPEN; each gains at most L/K_c",
                     j_holds(dev) and all(v["gain_at_Kc"] == 1 for v in vs), {k: v for k, v in by.items() if v})
    sm120 = [pair_verdict("sm_120", r, w) for r, w in fpx_pairs("sm_120")]
    checks.check("sm_120: all 19 tie, by vendor data except s8.sp's m16n8k32, which rests on measured (M) "
                 "(IMMA.SP.16832)", all(v["outcome"] == "TIE" for v in sm120)
                 and {v["form"] for v in sm120 if any(x["source"] == "M measured" for x in v["shallow"])} == {"s8.sp"})
    for dev in ("sm_90a", "sm_100a"):
        vs = [pair_verdict(dev, r, w) for r, w in fpx_pairs(dev)]
        onm = sorted({v["form"] for v in vs if v["outcome"] == "TIE-ON-M"})
        checks.check(f"{dev}: the only pairs resting on (M') are s8.sp's (legacy mma.sp m16n8k32 s8, 16 physical "
                     "positions); the rest tie by vendor data alone", onm == ["s8.sp"], onm)

    # -- the rank floor's closures on FP kinds (RT-FP-6), which J reaches without the rank floor ------------------------
    expect_fp = {"sm_120": 21, "sm_90a": 18, "sm_100a": 25}
    for dev in DEVICES:
        pairs = fpx_pairs(dev, "fp_rank")
        vs = [pair_verdict(dev, r, w) for r, w in pairs]
        by = {o: sorted({v["form"] for v in vs if v["outcome"] == o}) for o in ("TIE", "TIE-ON-M", "CLOSED", "OPEN")}
        checks.check(f"{dev}: the rank floor's {expect_fp[dev]} FP-kind closures under J: none OPEN, no rank floor or FP "
                     f"exactness read; forms resting on (M) unmeasured: {by['TIE-ON-M']}",
                     len(pairs) == expect_fp[dev] and not by["OPEN"], by)

    # -- S14: packed slots, on every device ---------------------------------------------------------------------------
    for claim, ok, detail in s14_claims():
        checks.check(claim, ok, detail)

    if checks.mutations:
        def under(name: str, dev: str, table: dict, key, value):
            with checks.mutant(name):
                saved = table[key]
                table[key] = value
                try:
                    checks.check(f"{dev}: Lemma J's claims hold as recorded", j_holds(dev))
                finally:
                    table[key] = saved

        under("sm_120's NVF4 at 4x the FP8 rate (the sheet says 2x)", "sm_120", fp.RATES["sm_120"], "nvf4", 4096)
        under("sm_120's E4M3 at twice the sheet's FP8 rate", "sm_120", fp.RATES["sm_120"], "e4m3", 2048)
        under("B200's tcgen05 kind::mxf4nvf4 at K = 32 (PTX says 64)", "sm_100a", TCGEN05, "nvf4", 32)
        under("H200's wgmma s8 at K = 16 (PTX says 32)", "sm_90a", WGMMA, "s8", 16)
        under("H200's legacy mma.sp e4m3 at m16n8k32 (PTX has only k64): a shallower 2:4 shape", "sm_90a", MMA_SYNC,
              ("e4m3", True), ("m16n8k64", "m16n8k32"))
        under("sm_120's IMMA.SP.16832 at sm_89's behaviour (8.0 SMSP-cycles, the MAC rate)", "sm_120",
              MEASURED["sm_120"], ("s8", True, "m16n8k32"), ("IMMA.SP.16832", Q(8), 128))
        under("B200's NVF4 at 4x its FP8 rate", "sm_100a", fp.RATES["sm_100a"], "nvf4", 4 * 8192)
        checks.mutation("sm_89's measured IMMA.16816 in sm_120's (M) table: (M) fails on it",
                        SM89["m16n8k16"][1] / SM89["m16n8k16"][2] < step("sm_120", "s8"))

        def under_s14(name: str, table: dict, key, value):
            with checks.mutant(name):
                saved = table[key]
                table[key] = value
                try:
                    for claim, ok, _ in s14_claims():
                        checks.check(claim, ok)
                finally:
                    table[key] = saved

        h1 = "H1 zero instance, L = 96"
        under_s14("S14 at R = 127: a packed A entry is 1 + 127 = 128, outside int8", INSTANCES, h1,
                  (127, *INSTANCES[h1][1:]))
        under_s14("S14 at L = 128 on H1's zero instance: y1 spreads 129·128 >= 126²", INSTANCES, h1,
                  (126, (0, 1), range(-64, 66), 128, (-64 * 128, 65 * 128)))
        under_s14("B200's IMMA-class s8 at half its sheet rate: the step doubles past a decode's 1/64", fp.RATES["sm_100a"],
                  "s8", 4096)
        under_s14("H200's wgmma s8 at half its sheet rate: a fold of 1/2 no longer covers the step", fp.RATES["sm_90a"],
                  "s8", fp.RATES["sm_90a"]["s8"] // 2)
        under_s14("sm_120's VF3e joint row JI3e_0 at a = 9.6: its break-even moves off VectorThresholdD1's q*", JI3E,
                  "JI3e_0", (Q("9.6"), Q(1)))
        with checks.mutant("S14's magic multiplier one below ceil(2^K/R²): the decode's division is off by one"):
            for name in INSTANCES:
                checks.check(f"{name} decodes exactly", chained_ok(name, CASES, dm=-1)[0])
        with checks.mutant("S14 at FOLD-OPT's fold replaced by 1/2 a word on sm_120: the chained packing gains"):
            w, e = CHAIN
            checks.check("the chained packing loses on sm_120", packing_gain("sm_120", w, e, Q(1, 2)) <= 1)

    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
