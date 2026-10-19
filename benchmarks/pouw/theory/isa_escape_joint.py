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

Run: `python isa_escape_joint.py [--mutations] [--verdict PATH]`. Stdlib + exact_kit and its sibling `isa_escape_fp`;
reads nothing.
"""
from __future__ import annotations

import functools
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

    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
