"""RT-FP's over-credit witness for FPX-DESIGN's FP designs: each variant's credited floor ♭ against a route that writes
the same committed words without the variant's extraction.

    uv run --project benchmarks/pouw/machine_model python benchmarks/pouw/rt_fp/overcredit.py FPX_EPS OUT

FPX_EPS is FPX-DESIGN's `benchmarks/pouw/fpx_design/fpx_eps.py` (its VARIANTS and `write_variant` are read as they are,
so this prices exactly what it pushed). Each variant puts the FP32 word's extraction into the design's `[fold]`
(`fold_add`), and the LP charges that fold to every tensor form the cheater may write on (eps.py `fastest`), the int8
ones included, and to the credited floor program. data/forms.toml states the opposite convention: converting an FP
accumulator to s32 costs the cheater nothing.

The cheater against variant V runs a design that writes V's committed words with no extraction op (ROUTES, by the
variant's kind): `rt-V-x0`, V with `fold_add` removed, or a variant FPX-DESIGN wrote for that route (its int8-unit g1).
The bound is ε_machine ≥ 1 − T_cheat / ♭_V, with T_cheat the lesser of the cheater's LP and its own floor program. A
variant with no such route is listed with its reason.
"""
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

#: why an op-free route writes the variant's words, by (device, variant prefix, base design, the cheater's design: None
#: for rt-V-x0, else a variant of FPX_EPS); the first match wins
G1S = ("g1s at e4m3i120 puts R + b in [−120, 121] ⊂ s8: the int8 MMA writes its words with no extraction, as FPX-DESIGN's "
       "own int8-unit g1s does")
ROUTES = [
    ("*", "fpx-c1-", "*", None, ("C1 is int8 Dither's transcript bit for bit: the device's int8 MMA (IMMA, wgmma s8, "
                                 "kind::i8) writes each s32 word with no extraction")),
    ("sm_90", "fpx-2sp-int5-", "*", None, "the int5 twin's operands lie in [−16, 16] ⊂ s8: wgmma s8 writes the words"),
    ("sm_120", "fpx-g1s-fp8-", "*", "fpx-g1s-fx1", G1S),
    ("sm_100", "fpx-g1s-fp8-", "*", "fpx-g1s-i8", G1S),
]

#: why the other variants keep the extraction, or an op of its size, on every route
NO_ROUTE = ("its fold's entry is XOR (duo; LI-CP-W's d₀), which can't absorb an add, or fx1's IMAD, which absorbs it only "
            "with its 32 per-position addends in registers that c_fx1 (128 of 128) doesn't have, else a MOV a word "
            "(fx1_absorb.py); and its operands (e4m3i ±352, g1's ±353) are not s8")

#: variants left unpriced, and why: a `fold_add` that is no extraction, or a route the k32 rows can't price
UNPRICED = {
    "fpx-g1s-fx1": "its fold_add is fx1's keyed IMAD on Dither-2S, which every route pays",
    "fpx-g1s-fp4-fp": ("at nvi10 (the g1s FP4 arms) R + b ≤ 121 ⊂ s8 too, so two chained int8 MMAs and one fold write each "
                       "k64 word with no extraction; not priced, since the int8 rows fold k32 words. At nvi's scales to "
                       "15 there is no int8 route"),
}


def load(path: Path):
    spec = importlib.util.spec_from_file_location("fpx_eps", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def route(v: dict, names: set[str]) -> tuple[str | None, str] | None:
    """(the cheater's variant, None for rt-V-x0; why) for V, or None when no route applies or its cheater isn't there."""
    return next(((c, why) for d, p, b, c, why in ROUTES if d in ("*", v["device"]) and v["name"].startswith(p)
                 and b in ("*", v["base"]) and (c is None or c in names)), None)


def priced(data: Path, design: str, vector: str) -> dict:
    cmd = ["machine-model", "--data", str(data), "eps", design, "--credit", "floor", "--operands", "floor", "--json"]
    return json.loads(subprocess.run(cmd + (["--vector", vector] if vector else []), capture_output=True, text=True,
                                     check=True).stdout)


def main(fpx_eps: Path, out: Path) -> int:
    fpx = load(fpx_eps)
    variants = list(fpx.VARIANTS)
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for dev, spec in fpx.DEVICES.items():
        data = out / "data" / dev
        shutil.rmtree(data, ignore_errors=True)
        shutil.copytree(spec["data"], data, ignore=shutil.ignore_patterns("devices") if dev == "sm_120" else None)
        attacks = []
        mine = [v for v in variants if v["device"] == dev]
        names = {v["name"] for v in mine}
        for v in mine:
            fpx.write_variant(v, data)
        for v in mine:
            r = route(v, names)
            if r is None:
                why = UNPRICED.get(v["name"], "no extraction" if not v["fold_add"] else NO_ROUTE)
                skipped.append({"device": dev, "variant": v["name"], "why": why})
                continue
            cheat, why = r
            if cheat is None:
                cheat = f"rt-{v['name']}-x0"
                fpx.write_variant(copy.deepcopy(v) | {"name": cheat, "fold_add": {}, "summary": why,
                                                      "title": f"{v['name']} with the extraction absorbed (RT-FP)"}, data)
            attacks.append((v["name"], cheat, why))
        for vec in spec["vectors"]:
            for var, cheat, why in attacks:
                v, c = priced(data, var, vec), priced(data, cheat, vec)
                credit = v["baseline"]["cycles_per_mma"]
                t = min(c["cheat"]["cycles_per_mma"], c["baseline"]["cycles_per_mma"])
                rows.append({"device": dev, "vector": vec or "VF3r-16", "variant": var, "credit": credit,
                             "credit_binds": v["baseline"]["binding"], "model_machine": v["terms"]["machine"],
                             "model_core": v["core"], "cheater": cheat, "cheat": t, "bound": 1 - t / credit, "why": why})
    (out / "overcredit.json").write_text(json.dumps({"rows": rows, "no_route": skipped}, indent=1) + "\n")
    lines = [("| device | vector | variant | credited ♭ (binds) | model ε_machine | model core ε | cheater | T_cheat | "
              "ε_machine ≥ |"), "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['device']} | {r['vector']} | {r['variant']} | {r['credit']:.3f} ({', '.join(r['credit_binds'])}) | "
              f"{100 * r['model_machine']:.2f}% | {100 * r['model_core']:.3f}% | {r['cheater']} | {r['cheat']:.3f} | "
              f"**{100 * r['bound']:.1f}%** |" for r in rows]
    lines += ["", "| route | cheater | why the cheater's words are the variant's |", "|---|---|---|"]
    lines += [f"| {d} {p}* on {b} | {c or 'rt-V-x0'} | {why} |" for d, p, b, c, why in ROUTES]
    lines += ["", "| device | variant not attacked | why |", "|---|---|---|"]
    lines += [f"| {s['device']} | {s['variant']} | {'the extraction stays on every route' if s['why'] == NO_ROUTE else s['why']} |"
              for s in skipped]
    lines += ["", f"The extraction stays on every route where {NO_ROUTE}."]
    (out / "overcredit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
