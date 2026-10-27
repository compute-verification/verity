"""The over-credit witness for fpx_design's FP designs: each variant's credited floor ♭ against a route that writes
the same committed words without the variant's extraction.

    uv run --project benchmarks/pouw/machine_model python benchmarks/pouw/rt_fp/overcredit.py FPX_EPS OUT

FPX_EPS is fpx_design's `benchmarks/pouw/fpx_design/fpx_eps.py` (its VARIANTS and `write_variant` are read as they are,
so this prices exactly what it pushed). Each variant puts the FP32 word's extraction into the design's `[fold]`
(`fold_add`), and the LP charges that fold to every tensor form the cheater may write on (eps.py `fastest`), the int8
ones included, and to the credited floor program. data/forms.toml states the opposite convention: converting an FP
accumulator to s32 costs the cheater nothing.

The cheater against variant V runs a design that writes V's committed words with no extraction op (ROUTES, by the
variant's kind): `rt-V-x0`, V with `fold_add` removed, or a variant fpx_design wrote for that route (its int8-unit g1),
at the L that writes V's words (an FP4 k64 word is two chained k32 int8 MMAs). The bound is
ε_machine ≥ 1 − T_cheat / ♭_V a committed word, with T_cheat the lesser of the cheater's LP and its own floor program. A
variant with no such route is listed with its reason. Every bound is at the record's convention (credit ♭, operands
free) on data/headline.toml's readings of the device and on its measured vector at the proved q* = ½.
"""
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

#: why an op-free route writes the variant's words, by (device, variant prefix, base design, the cheater's design: None
#: for rt-V-x0, else a variant of FPX_EPS; the cheater's L, None for the variant's); every match is priced
G1S = ("g1s at e4m3i120 puts R + b in [−120, 121] ⊂ s8: the int8 MMA writes its words with no extraction, as FPX-DESIGN's "
       "own int8-unit g1s does")
G1S4 = ("g1s's FP4 word is (R + b)·(R' + u) over k64, and at nvi10 (integer block scales to 10, the g1s FP4 arms) "
        "|R + b| ≤ 10·12 + 1 = 121 ⊂ s8: two chained k32 int8 MMAs (C carries the s32 sum) write it with no extraction, "
        "then the int8-unit g1s's fold")
SPLIT = ("an operand x with |x| ≤ 127·m is a sum of m s8 limbs, so Σ x·y is m² s8 products with unit coefficients in one "
         "s32 accumulator: m² chained int8 MMAs a word's k32 and no extraction")
G1S4_NVI = f"at nvi (block scales to 15) |R + b| ≤ 181, m = 2: {SPLIT}"
I352 = (f"operands to ±352 (the twin and the served GEMM on e4m3i352), m = 3: {SPLIT}; charged for both GEMMs (18 a "
        "word), since the variant's fold reads the served sum")
ROUTES = [
    ("*", "fpx-c1-", "*", None, None, ("C1 is int8 Dither's transcript bit for bit: the device's int8 MMA (IMMA, wgmma "
                                       "s8, kind::i8) writes each s32 word with no extraction")),
    ("sm_90", "fpx-2sp-int5-", "*", None, None,
     "the int5 twin's operands lie in [−16, 16] ⊂ s8: wgmma s8 writes the words"),
    ("sm_120", "fpx-g1s-fp8-", "*", "fpx-g1s-fx1", None, G1S),
    ("sm_100", "fpx-g1s-fp8-", "*", "fpx-g1s-i8", None, G1S),
    ("sm_120", "fpx-g1s-fp4-", "*", "fpx-g1s-fx1", 64, G1S4),
    ("sm_100", "fpx-g1s-fp4-", "*", "fpx-g1s-i8", 64, G1S4),
    ("sm_120", "fpx-g1s-fp4-", "*", "fpx-g1s-fx1", 256, G1S4_NVI),
    ("sm_100", "fpx-g1s-fp4-", "*", "fpx-g1s-i8", 256, G1S4_NVI),
    ("*", "fpx-2sp-e4m3i352-", "*", None, 576, I352),
]

#: why a variant with no route keeps the extraction, or an op of its size, on every route
NO_ROUTE = ("its fold's entry is XOR (duo; LI-CP-W's d₀), which can't absorb an add, or fx1's IMAD, which absorbs it only "
            "with its 32 per-position addends in registers that c_fx1 (128 of 128) doesn't have, else a MOV a word "
            "(fx1_absorb.py)")

#: variants left unpriced, and why: a `fold_add` that is no extraction
UNPRICED = {"fpx-g1s-fx1": "its fold_add is fx1's keyed IMAD on Dither-2S, which every route pays"}


def load(path: Path):
    spec = importlib.util.spec_from_file_location("fpx_eps", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def routes(v: dict, names: set[str]) -> list[tuple[str | None, int | None, str]]:
    """Each (the cheater's variant, None for rt-V-x0; its L, None for V's; why) for V whose cheater is there."""
    return [(c, L, why) for d, p, b, c, L, why in ROUTES if d in ("*", v["device"]) and v["name"].startswith(p)
            and b in ("*", v["base"]) and (c is None or c in names)]


def priced(data: Path, design: str, vector: str, q: float) -> dict:
    cmd = ["machine-model", "--data", str(data), "eps", design, "--credit", "floor", "--operands", "free", "--q", f"{q:g}",
           "--vector", vector, "--json"]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def readings(fpx, dev: str) -> list[tuple[str, str, float]]:
    """data/headline.toml's readings of `dev` (the sheet's vector at q* = ½, the measured one at 1.22) and the measured
    vector at the proved ½, as FPX_EPS (cfd0d59f4c on) prices them."""
    head = tomllib.loads((Path(fpx.DATA) / "headline.toml").read_text())
    fam, q = head["family"][dev], {r: x["q"] for r, x in head["reading"].items()}
    return [(r, fam[r]["vector"], q[r]) for r in q] + [("measured-half", fam["measured"]["vector"], q["sheet"])]


def main(fpx_eps: Path, out: Path) -> int:
    fpx = load(fpx_eps)
    variants = list(fpx.VARIANTS)
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for dev, spec in fpx.DEVICES.items():
        data = out / "data" / dev
        shutil.rmtree(data, ignore_errors=True)
        src = spec["data"] if isinstance(spec, dict) else spec
        shutil.copytree(src, data, ignore=shutil.ignore_patterns("devices") if dev == "sm_120" else None)
        attacks = []
        mine = [v for v in variants if v["device"] == dev]
        names = {v["name"] for v in mine}
        for v in mine:
            fpx.write_variant(v, data)
        for v in mine:
            rs = routes(v, names)
            if not rs:
                why = UNPRICED.get(v["name"], "no extraction" if not v["fold_add"] else NO_ROUTE)
                skipped.append({"device": dev, "variant": v["name"], "why": why})
            for cheat, L, why in rs:
                if cheat is None:
                    c = copy.deepcopy(v) | {"name": f"rt-{v['name']}-x0", "fold_add": {},
                                            "title": f"{v['name']} with the extraction absorbed (RT-FP)"}
                else:
                    c = copy.deepcopy(next(x for x in mine if x["name"] == cheat))
                if L is not None:
                    c |= {"name": f"{'' if cheat is None else 'rt-'}{c['name']}-l{L}",
                          "title": f"{c['title']}, at L = {L} (RT-FP)"}
                if c["name"] not in names:
                    names.add(c["name"])
                    fpx.write_variant(c | {"summary": why}, data)
                    if L is not None:
                        f = data / "designs" / f"{c['name']}.toml"
                        f.write_text(f"L = {L}\n" + f.read_text())
                attacks.append((v["name"], c["name"], why))
        for reading, vec, q in readings(fpx, dev):
            for var, cheat, why in attacks:
                v, c = priced(data, var, vec, q), priced(data, cheat, vec, q)
                credit = v["baseline"]["cycles_per_mma"]
                # cycles a MMA times L is a committed word's cycles up to the profile's words·depth, which both share
                t = min(c["cheat"]["cycles_per_mma"], c["baseline"]["cycles_per_mma"]) * c["L"] / v["L"]
                rows.append({"device": dev, "reading": reading, "vector": vec, "q": q, "variant": var, "credit": credit,
                             "credit_binds": v["baseline"]["binding"], "model_machine": v["terms"]["machine"],
                             "model_core": v["core"], "cheater": cheat, "cheater_L": c["L"], "cheat": t,
                             "cheat_binds": c["cheat"]["binding"], "bound": 1 - t / credit, "why": why})
    (out / "overcredit.json").write_text(json.dumps({"rows": rows, "no_route": skipped}, indent=1) + "\n")
    lines = [("| device | reading: vector, q* | variant | credited ♭ (binds) | model ε_machine | model core ε | cheater "
              "| T_cheat (binds) | ε_machine ≥ |"), "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['device']} | {r['reading']}: {r['vector']}, {r['q']:g} | {r['variant']} | {r['credit']:.3f} "
              f"({', '.join(r['credit_binds'])}) | "
              f"{100 * r['model_machine']:.2f}% | {100 * r['model_core']:.3f}% | {r['cheater']} | {r['cheat']:.3f} "
              f"({', '.join(r['cheat_binds'])}) | **{100 * r['bound']:.1f}%** |" for r in rows]
    lines += ["", ("♭ and T_cheat are cycles for one of the variant's MMAs' words (T_cheat scaled by the cheater's L over "
                   "the variant's).")]
    lines += ["", "| route | cheater | why the cheater's words are the variant's |", "|---|---|---|"]
    lines += [f"| {d} {p}* on {b} | {c or 'rt-V-x0'}{f' at L = {L}' if L else ''} | {why} |"
              for d, p, b, c, L, why in ROUTES]
    lines += ["", "| device | variant not attacked | why |", "|---|---|---|"]
    lines += [f"| {s['device']} | {s['variant']} | {'the extraction stays on every route' if s['why'] == NO_ROUTE else s['why']} |"
              for s in skipped]
    lines += ["", f"The extraction stays on every route where {NO_ROUTE}."]
    (out / "overcredit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
