"""CPU witnesses on the H200 (sm_90): the machine model's ε when the cheater may use a wgmma form the census
already prices, and the data-sheet vector's cliffs in A2u and the fold.

    python forms_gap.py            the measured reading with IGMMA_S8_N256_RS offered beside the honest form
    python forms_gap.py --sheet    DS-90's ε against the fold's cost and against A2u's uniform issue cost
    python forms_gap.py --b1       BGMMA_B1_N256 offered at split 8 (the least: t binary products separate
                                   at most 2^t values of an s8 operand) and 64 (bit planes); DS-90 prices it on issue alone
    python forms_gap.py --rs-fit   RS offered at SS's joint coefficients times the H200's matched-pair ratio
    python forms_gap.py --honest-true  γ with honest timed by the loop that folds the wgmma's accumulators,
                                   and each loop at the toml's 122.41 / f and on the clock, 128.01 / f
    python forms_gap.py --fan-in   ε at the fold costs FoldFanIn proves (1/(f − 1) a word), without FoldOpt
    python forms_gap.py --b1-fit   b1 offered at splits 8 and 64 at SS's joint lines times the H200's ratio
    python forms_gap.py --menu     each s8-exact sibling (N128, N64, mma.sync m16n8k32) offered beside it,
                                   then every form FIT has fitted from the H200, at once
    python forms_gap.py --menu-fp  each FP kind (f16, bf16, tf32, e4m3, e5m2) at the lane default, by MACs,
                                   by tensor time and at its H200 fit; then all 22 fitted forms at once, and each kind's
                                   break-even

sm_90 has no data/forms.toml, so `eps` restricts the cheater's tensor route to the honest's form, IGMMA_S8_N256 (wgmma
m64n256k32 s8, A and B from shared memory). The census also prices IGMMA_S8_N256_RS, the same shape with A from
registers: the same fresh words and s8 depth, the same tensor line (M 0.0312468), but on the async-joint rows JI90_0 and
JI90_1 the default a lane instruction gets (0.98, 0.92) where the honest form's measured lines are 62.74 and 66.94. Each
witness copies the device's data into a scratch directory, edits only the copy and runs `machine-model eps` on it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "machine_model"
SM90 = MODEL / "data" / "devices" / "sm_90"
DESIGN = "dither-2s-licpw-redux-l32"
FORMS = """[IGMMA_S8_N256]
shape = [16, 256, 32]
split = 1
exact = "s8.s8 into s32, A and B from shared memory by descriptor"

[IGMMA_S8_N256_RS]
shape = [16, 256, 32]
split = 1
exact = "s8.s8 into s32, A from registers, B from shared memory"
"""


def b1(split: int) -> str:
    return f"""[IGMMA_S8_N256]
shape = [16, 256, 32]
split = 1
exact = "s8.s8 into s32"

[BGMMA_B1_N256]
shape = [16, 256, 256]
split = {split}
exact = "b1 and.popc: s8 operands in bit planes, {split} b1 products an s8 product"
"""


def scratch(root: Path, *, rs: bool, forms: str = FORMS) -> Path:
    """A copy of sm_90's data under `root`; with `rs`, a forms.toml (`forms`, by default the honest form and the RS form)
    and each honest naming its form."""
    d = root / "sm_90"
    shutil.copytree(SM90, d)
    if rs:
        (d / "forms.toml").write_text(forms)
        for p in (d / "honest").glob("*.toml"):
            t = p.read_text()
            p.write_text(re.sub(r'^(class = "IGMMA_S8_N256")$', r'\1\nform = "IGMMA_S8_N256"', t, flags=re.M))
    return d


def eps(data: Path, vector: str, q: float, *extra: str) -> dict:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(MODEL), os.environ.get("PYTHONPATH")]))}
    cmd = [sys.executable, "-m", "verity_machine_model.cli", "--data", str(data), "eps", DESIGN, "--credit", "floor",
           "--operands", "free", "--vector", vector, "--q", str(q), "--json", *extra]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def line(label: str, d: dict) -> str:
    c = d["cheat"]
    forms = {k: round(v, 4) for k, v in (c.get("forms") or {}).items()}
    words = {k: round(v, 4) for k, v in c["words"].items() if v}
    return (f"{label:<34} core {d['core'] * 100:7.3f}%  ε_machine {d['terms']['machine'] * 100:7.3f}%  "
            f"cheater {c['cycles_per_mma']:7.2f} SMSP-cycles a MMA on {','.join(c['binding'])}  forms {forms}  words {words}")


def forms_gap() -> None:
    with tempfile.TemporaryDirectory() as t:
        base, rs = scratch(Path(t) / "base", rs=False), scratch(Path(t) / "rs", rs=True)
        for vector in ("DS-90", "DC-90", "DC-90-ri"):
            for q in (1.22, 0.5):
                print(line(f"{vector} q*={q} honest form only", eps(base, vector, q)))
                print(line(f"{vector} q*={q} + IGMMA_S8_N256_RS", eps(rs, vector, q)))


def set_census(data: Path, kind: str, row: str, u: float) -> None:
    """Set the one census line of `kind` on `row`, in census.csv or an epoch's census/*.csv (DS-90's)."""
    hits = 0
    for p in [data / "census.csv", *sorted((data / "census").glob("*.csv"))]:
        if p.is_file():
            t, n = re.subn(rf"^{kind},{row},,,[^,]*,", f"{kind},{row},,,{u},", p.read_text(), flags=re.M)
            if n:
                p.write_text(t)
            hits += n
    assert hits == 1, (kind, row, hits)


def set_uni_issue(data: Path, cost: float) -> None:
    set_census(data, "UNI", "DS_issue", cost)


def sheet() -> None:
    with tempfile.TemporaryDirectory() as t:
        d = scratch(Path(t), rs=False)
        for r in (1.0546875, 0.95, 0.93, 0.92, 0.91, 0.9, 0.75):
            print(line(f"DS-90 q*=0.5 fold {r}", eps(d, "DS-90", 0.5, "--fold-cost", str(r))))
        for r in (0.9, 0.75):
            print(line(f"DS-90 q*=1.22 fold {r}", eps(d, "DS-90", 1.22, "--fold-cost", str(r))))
        for c in (1, 0.05, 0.031, 0.025, 0.02, 0.0001862, 0):
            set_uni_issue(d, c)
            print(line(f"DS-90 q*=0.5 UNI on DS_issue {c}", eps(d, "DS-90", 0.5)))
        v = d / "vectors" / "DS-90.toml"
        v.write_text(v.read_text().replace('"DS_shfl"]', '"DS_shfl", "UP"]'))
        for qu in (0.5, 8):
            print(line(f"DS-90+UP q*=0.5 UNI 0 q*_U {qu}", eps(d, "DS-90", 0.5, "--q-uniform", str(qu))))


SS_JI = {"JI90_0": 62.7407638, "JI90_1": 66.93849, "JI90_2": 121.566}
# Each census form the H200 runs has been timed against: its shape (a warp's share, m 16), its least sound split, its
# N256-equivalents an instruction (rs.py's n256), the run rs.py's `fit` read, and the fit, each joint row's coefficient
# over IGMMA_S8_N256's (the least over the matched pairs: alone, beside the LI fold, at the IMAD and LOP3 points), or
# None before its run is read. RS's least pair is the LOP3 point, where it runs 1.2% faster than SS; beside the LI fold
# it is 1.9% slower (1.031, 1.030, 1.019).
FIT = {
    "IGMMA_S8_N256_RS": ([16, 256, 32], 1, 1, "r20261019-070001-df11", {"JI90_0": 0.9676, "JI90_1": 0.9694, "JI90_2": 0.9868}),
    "IGMMA_U8S8_N256": ([16, 256, 32], 1, 1, "r20261019-070001-df11", {"JI90_0": 1.0, "JI90_1": 1.0, "JI90_2": 1.0}),
    "BGMMA_B1_N256": ([16, 256, 256], 8, 1, "r20261019-073926-b0ae", {"JI90_0": 1.0, "JI90_1": 1.0, "JI90_2": 1.0}),
    "IGMMA_S8_N128": ([16, 128, 32], 1, 1 / 2, "r20261019-081829-5dff", {"JI90_0": 0.9773, "JI90_1": 0.978, "JI90_2": 0.9842}),
    "IGMMA_S8_N64": ([16, 64, 32], 1, 1 / 4, "r20261019-081829-5dff", {"JI90_0": 1.0, "JI90_1": 1.0, "JI90_2": 1.0}),
    "IMMA16832_S8": ([16, 8, 32], 1, 1 / 32, "r20261019-081829-5dff", {"JI90_0": 1.2241, "JI90_1": 1.2177, "JI90_2": 1.1563}),
    "IGMMA_S8_N8": ([16, 8, 32], 1, 1 / 32, "r20261019-081829-5dff", {"JI90_0": 3.8839, "JI90_1": 3.8012, "JI90_2": 2.9675}),
    "IGMMA_S8_N16": ([16, 16, 32], 1, 1 / 16, "r20261019-081829-5dff", {"JI90_0": 2.1078, "JI90_1": 2.076, "JI90_2": 1.7722}),
    "IGMMA_S8_N32": ([16, 32, 32], 1, 1 / 8, "r20261019-081829-5dff", {"JI90_0": 1.3866, "JI90_1": 1.3755, "JI90_2": 1.2695}),
    "IGMMA_S8_N192": ([16, 192, 32], 1, 3 / 4, "r20261019-081829-5dff", {"JI90_0": 1.0, "JI90_1": 1.0, "JI90_2": 1.0}),
    "IMMA16816_S8": ([16, 8, 16], 1, 1 / 64, "r20261019-081829-5dff", {"JI90_0": 1.8645, "JI90_1": 1.8397, "JI90_2": 1.6027}),
    "BMMA168256_B1": ([16, 8, 256], 8, 1 / 32, "r20261019-081829-5dff", {"JI90_0": 1.2241, "JI90_1": 1.2177, "JI90_2": 1.1563}),
    "HGMMA_F16_N256": ([16, 256, 16], 1, 1 / 2, "r20261019-090330-b1db", {"JI90_0": 1.4604, "JI90_1": 1.4472, "JI90_2": 1.321}),
    "HGMMA_BF16_N256": ([16, 256, 16], 1, 1 / 2, "r20261019-090330-b1db", {"JI90_0": 1.4604, "JI90_1": 1.4472, "JI90_2": 1.321}),
    "HGMMA_TF32_N256": ([16, 256, 8], 1, 1 / 4, "r20261019-090330-b1db", {"JI90_0": 3.3947, "JI90_1": 3.326, "JI90_2": 2.6695}),
    "QGMMA_E4M3_N256": ([16, 256, 32], 3, 1, "r20261019-090330-b1db", {"JI90_0": 1.0, "JI90_1": 1.0, "JI90_2": 1.0}),
    "HGMMA_BF16_N128": ([16, 128, 16], 1, 1 / 4, "r20261019-092853-a64d", {"JI90_0": 1.6522, "JI90_1": 1.6334, "JI90_2": 1.4546}),
    "QGMMA_E4M3_N128": ([16, 128, 32], 3, 1 / 2, "r20261019-092853-a64d", {"JI90_0": 0.9767, "JI90_1": 0.9774, "JI90_2": 0.9838}),
    "QGMMA_E5M2_N256": ([16, 256, 32], 3, 1, "r20261019-092853-a64d", {"JI90_0": 1.0, "JI90_1": 1.0, "JI90_2": 1.0}),
    "HMMA16816_F16": ([16, 8, 16], 1, 1 / 64, "r20261019-092853-a64d", {"JI90_0": 2.6749, "JI90_1": 2.6269, "JI90_2": 2.1677}),
    "HMMA16816_BF16": ([16, 8, 16], 1, 1 / 64, "r20261019-092853-a64d", {"JI90_0": 2.6749, "JI90_1": 2.6269, "JI90_2": 2.1677}),
    "HMMA1688_TF32": ([16, 8, 8], 1, 1 / 128, "r20261019-092853-a64d", {"JI90_0": 5.5764, "JI90_1": 5.4452, "JI90_2": 4.1667}),
}
SIBLINGS = ("IGMMA_S8_N128", "IGMMA_S8_N64", "IMMA16832_S8", "IGMMA_S8_N192", "IGMMA_S8_N32", "IGMMA_S8_N16",
            "IGMMA_S8_N8", "IMMA16816_S8", "BMMA168256_B1")
# The census's FP tensor kinds (every one into FP32, as DC's probes time them): shape, least split (forms.toml's rules:
# f16, bf16 and tf32 hold every s8 and a 32-step word is exact in FP32, so 1; e4m3 3, by Karatsuba on balanced nibbles;
# e5m2 3 as well, the cheater's favour, though its 3 significant bits miss the nibbles' odd sums past 8 and schoolbook
# takes 4), N256-equivalents an instruction by MACs, and by tensor time (IGMMA_S8_N256's M line over the kind's). Those
# timed beside the LI fold are in FIT too.
FP = {
    "HGMMA_F16_N256": ([16, 256, 16], 1, 1 / 2, 1.0),
    "HGMMA_BF16_N256": ([16, 256, 16], 1, 1 / 2, 1.0),
    "HGMMA_BF16_N128": ([16, 128, 16], 1, 1 / 4, 0.0312468 / 0.0624906),
    "HGMMA_TF32_N256": ([16, 256, 8], 1, 1 / 4, 1.0),
    "QGMMA_E4M3_N256": ([16, 256, 32], 3, 1, 1.0),
    "QGMMA_E4M3_N128": ([16, 128, 32], 3, 1 / 2, 0.0312468 / 0.0624906),
    "QGMMA_E5M2_N256": ([16, 256, 32], 3, 1, 1.0),
    "HMMA16816_F16": ([16, 8, 16], 1, 1 / 64, 0.0312468 / 0.656135),
    "HMMA16816_BF16": ([16, 8, 16], 1, 1 / 64, 0.0312468 / 0.65613),
    "HMMA1688_TF32": ([16, 8, 8], 1, 1 / 128, 0.0312468 / 0.65613),
}


set_line = set_census


def form_toml(kind: str, shape: list[int], split: int) -> str:
    return f'[{kind}]\nshape = {shape}\nsplit = {split}\nexact = "s8 products, {split} a product"\n\n'


def offer(root: Path, kinds: dict[str, float | None]) -> Path:
    """sm_90's data with the honest form and each of `kinds` offered, each at IGMMA's joint lines by its MACs times the
    given scale, or at the census's lane default when the scale is None."""
    spec = {**FP, **FIT}
    honest = form_toml("IGMMA_S8_N256", [16, 256, 32], 1)
    d = scratch(root, rs=True, forms=honest + "".join(form_toml(k, spec[k][0], spec[k][1]) for k in kinds))
    for k, scale in kinds.items():
        if scale is not None:
            for row, u in SS_JI.items():
                set_line(d, k, row, u * spec[k][2] * (FIT[k][4][row] if scale == "fit" else scale))
    return d


def rs_fit() -> None:
    with tempfile.TemporaryDirectory() as t:
        for scale in (1.0, "fit", 0.95, 0.9):
            d = offer(Path(t) / f"fit-{scale}", {"IGMMA_S8_N256_RS": scale})
            label = f"{FIT['IGMMA_S8_N256_RS'][3]}'s fit" if scale == "fit" else f"{scale}× SS's"
            for vector, q in (("DC-90", 1.22), ("DC-90-ri", 1.22), ("DC-90-ri", 0.5)):
                print(line(f"{vector} q*={q} RS at {label}", eps(d, vector, q)))


def menu() -> None:
    """The form escape beyond RS: each form the census prices at s8's rate offered beside the honest form at the
    census's lane default, at IGMMA's joint lines by its MACs, and at its fit once read; then every fitted form at
    once."""
    with tempfile.TemporaryDirectory() as t:
        for kind in SIBLINGS:
            for scale in (None, 1.0, "fit"):
                if scale == "fit" and FIT[kind][4] is None:
                    continue
                d = offer(Path(t) / f"{kind}-{scale}", {kind: scale})
                label = {None: "the census default", 1.0: "IGMMA's lines by MACs"}.get(scale, f"{FIT[kind][3]}'s fit")
                print(line(f"DC-90-ri q*=1.22 +{kind} at {label}", eps(d, "DC-90-ri", 1.22)))
        fitted = {k: "fit" for k, f in FIT.items() if f[4] is not None}
        d = offer(Path(t) / "all", fitted)
        for vector, q in (("DC-90", 1.22), ("DC-90-ri", 1.22), ("DC-90-ri", 0.5)):
            print(line(f"{vector} q*={q} every fitted form ({len(fitted)}) at its fit", eps(d, vector, q)))


def menu_fp() -> None:
    """The form escape's FP forms: each FP kind offered beside the honest form at the lane default, at IGMMA's joint
    lines by its MACs, by its tensor time (where the s8 siblings' fits fell, at 0.8-1.0 of it) and at 0.8 of that, and
    at its fit once timed; then the untimed ones at once beside every fitted form, and each kind's break-even."""
    fitted = {k: "fit" for k, f in FIT.items() if f[4] is not None}
    with tempfile.TemporaryDirectory() as t:
        for kind, (_, _, n256, tt) in FP.items():
            scales = [("the census default", None), ("IGMMA's lines by MACs", 1.0),
                      ("IGMMA's lines by tensor time", tt / n256), ("0.8 of that", 0.8 * tt / n256)]
            if kind in fitted:
                scales.append((f"{FIT[kind][3]}'s fit", "fit"))
            for i, (label, scale) in enumerate(scales):
                d = offer(Path(t) / f"{kind}-{i}", {kind: scale})
                print(line(f"DC-90-ri q*=1.22 +{kind} at {label}", eps(d, "DC-90-ri", 1.22)))
        untimed = [k for k in FP if k not in fitted]
        for label, f in (("by tensor time", 1.0), ("at 0.8 of tensor time", 0.8)) if untimed else (("", 1.0),):
            d = offer(Path(t) / f"all-{f}", {**fitted, **{k: f * FP[k][3] / FP[k][2] for k in untimed}})
            for vector, q in (("DC-90", 1.22), ("DC-90-ri", 1.22), ("DC-90-ri", 0.5)):
                print(line(f"{vector} q*={q} {len(fitted)} fitted forms"
                           + (f" and {len(untimed)} untimed FP {label}" if untimed else ""), eps(d, vector, q)))
        base = eps(scratch(Path(t) / "base", rs=False), "DC-90-ri", 1.22)["core"]
        for kind, (_, _, n256, tt) in FP.items():
            lo, hi = 0.0, 1.0
            for i in range(8):
                mid = (lo + hi) / 2
                d = offer(Path(t) / f"{kind}-be{i}", {kind: mid * tt / n256})
                lo, hi = (mid, hi) if eps(d, "DC-90-ri", 1.22)["core"] > base + 1e-5 else (lo, mid)
            print(f"DC-90-ri q*=1.22 {kind} breaks the honest-only {base * 100:.3f}% below {hi:.3f} of its tensor time "
                  f"({hi * tt / n256:.3f} of IGMMA's lines by MACs)")


def b1_fit() -> None:
    """The form escape's b1 branch: BGMMA offered at splits 8 and 64 with its joint lines at SS's, and at its fit once
    read."""
    with tempfile.TemporaryDirectory() as t:
        for split in (8, 64):
            for scale in (1.0, "fit"):
                if scale == "fit" and FIT["BGMMA_B1_N256"][4] is None:
                    continue
                d = scratch(Path(t) / f"b1-{split}-{scale}", rs=True, forms=b1(split))
                for row, u in SS_JI.items():
                    set_line(d, "BGMMA_B1_N256", row, u * (FIT["BGMMA_B1_N256"][4][row] if scale == "fit" else scale))
                label = f"{FIT['BGMMA_B1_N256'][3]}'s fit" if scale == "fit" else f"{scale}× SS's"
                for vector, q in (("DC-90-ri", 1.22), ("DC-90-ri", 0.5)):
                    print(line(f"{vector} q*={q} b1 split {split} at {label}", eps(d, vector, q)))


# The loops that fold the wgmma's own accumulators, the plain GEMM's 122.41 over their tensor fraction: wgli_true_w2
# (0.42158), and the fastest, wgli_db_w2 (0.45574)
TRUE_LOOPS = (290.36, 268.60)
# The toml's 122.41 is the model's plain GEMM on DC-90 (lanes beside the tensor, JI90_2), not a time on the clock: a MMA
# alone takes 4 / 0.0312468 = 128.01 SMSP-cycles (census IGMMA_S8_N256 M), so a loop at fraction f takes 128.01 / f
#.
PLAIN, ALONE = 122.41, 4 / 0.0312468


def honest_true() -> None:
    """γ and ε when honest's measured time is the loop that folds the wgmma's own accumulators, each loop
    at the toml's convention (122.41 / f) and on the clock (128.01 / f)."""
    with tempfile.TemporaryDirectory() as t:
        d = scratch(Path(t), rs=False)
        p = d / "honest" / "dc90-licpw-redux-l32.toml"
        src = p.read_text()
        for plain in (204.25, *TRUE_LOOPS):
            for label, cyc in (("122.41/f", plain), ("on the clock", ALONE * plain / PLAIN)):
                p.write_text(re.sub(r"^cycles_per_mma = [0-9.]+", f"cycles_per_mma = {cyc:.4f}", src, flags=re.M))
                for vector, q in (("DS-90", 0.5), ("DC-90-ri", 1.22)):
                    r = eps(d, vector, q)
                    print(line(f"{vector} q*={q} honest {cyc:.2f} ({label})", r)
                          + f"  γ {r['gamma']:.4f}  honest {r['honest_cycles_per_mma']:.2f} ({r['honest_time']})")


def b1_gap() -> None:
    with tempfile.TemporaryDirectory() as t:
        for split in (8, 64):
            d = scratch(Path(t) / f"b1-{split}", rs=True, forms=b1(split))
            for r in (None, 0.9, 0.75, 0.5):
                fold = ("--fold-cost", str(r)) if r else ()
                print(line(f"DS-90 q*=0.5 +b1 split {split} fold {r or 'today'}", eps(d, "DS-90", 0.5, *fold)))
            print(line(f"DC-90-ri q*=1.22 +b1 split {split}", eps(d, "DC-90-ri", 1.22)))
        d = scratch(Path(t) / "base", rs=False)
        for r in (0.75, 0.5):
            print(line(f"DS-90 q*=0.5 honest form only fold {r}", eps(d, "DS-90", 0.5, "--fold-cost", str(r))))


# The fold's cost a word that FoldFanIn proves at fan-in f, 1/(f − 1): f = 3 (three registers), 5 (guarded), 10 (sm_90's
# cx[UR][imm] operand) and 11 (a guarded P2R with it), beside LI's own (FOLD-OPT's) and DS-90's cliff.
FAN_IN = {"LI, FOLD-OPT": 1.0546875, "DS-90's cliff": 0.922, "f=3": 1 / 2, "f=5": 1 / 4, "f=10": 1 / 9, "f=11": 1 / 10}


def fan_in() -> None:
    """The headline at the fold costs FoldFanIn proves, which is all the TS guarantee has without FoldOpt."""
    with tempfile.TemporaryDirectory() as t:
        d = scratch(Path(t), rs=False)
        for label, r in FAN_IN.items():
            for vector, q in (("DS-90", 0.5), ("DC-90-ri", 1.22)):
                print(line(f"{vector} q*={q} fold {r:.4g} ({label})", eps(d, vector, q, "--fold-cost", str(r))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheet", action="store_true", help="DS-90's cliffs in the fold's cost and in A2u")
    ap.add_argument("--b1", action="store_true", help="ε with BGMMA_B1_N256, which DS-90 prices on issue alone, offered")
    ap.add_argument("--rs-fit", action="store_true", help="ε with RS offered at SS's joint coefficients times the H200's ratio")
    ap.add_argument("--honest-true", action="store_true", help="γ with honest timed by the loop that folds the accumulators")
    ap.add_argument("--fan-in", action="store_true", help="ε at the fold costs FoldFanIn proves, without FoldOpt")
    ap.add_argument("--b1-fit", action="store_true", help="ε with b1 offered at its joint lines fitted from the H200")
    ap.add_argument("--menu", action="store_true", help="ε with each of the honest's s8-exact siblings offered")
    ap.add_argument("--menu-fp", action="store_true", help="ε with each FP tensor kind offered, and its break-even")
    a = ap.parse_args()
    (sheet() if a.sheet else b1_gap() if a.b1 else rs_fit() if a.rs_fit else honest_true() if a.honest_true
     else fan_in() if a.fan_in else b1_fit() if a.b1_fit else menu() if a.menu else menu_fp() if a.menu_fp
     else forms_gap())
    return 0


if __name__ == "__main__":
    sys.exit(main())
