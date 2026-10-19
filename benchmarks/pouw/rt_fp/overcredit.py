"""RT-FP's over-credit witness for FPX-DESIGN's FP designs: each variant's credited floor ♭ against a route that writes
the same committed words without the variant's extraction.

    uv run --project benchmarks/pouw/machine_model python benchmarks/pouw/rt_fp/overcredit.py FPX_EPS OUT

FPX_EPS is FPX-DESIGN's `benchmarks/pouw/fpx_design/fpx_eps.py` (its VARIANTS and `write_variant` are read as they are,
so this prices exactly what it pushed). Each variant puts the FP32 word's extraction into the design's `[fold]`
(`fold_add`), and the LP charges that fold to every tensor form the cheater may write on (eps.py `fastest`), the int8
ones included, and to the credited floor program. data/forms.toml states the opposite convention: converting an FP
accumulator to s32 costs the cheater nothing.

The cheater against variant V runs `rt-V-x0`: V with `fold_add` removed, on a route that writes V's committed words
with no extraction op (ROUTES, by the variant's kind). The bound is ε_machine ≥ 1 − T_cheat / ♭_V, with T_cheat the
lesser of rt-V-x0's cheater LP and its own floor program. A variant with no such route is listed with its reason.
"""
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

#: why an op-free route writes the variant's words, by (device, variant prefix, base design); the first match wins
ROUTES = [
    ("*", "fpx-c1-", "*", "C1 is int8 Dither's transcript bit for bit: the device's int8 MMA (IMMA, wgmma s8, kind::i8) "
                          "writes each s32 word with no extraction"),
    ("sm_90", "fpx-2sp-int5-", "*", "the int5 twin's operands lie in [−16, 16] ⊂ s8: wgmma s8 writes the words"),
    ("sm_120", "fpx-", "dither-c-peel-fx1",
     "fx1's entry is t·m with m = key_mul(1, (4J + e)·32) a compile-time odd constant (egemm.cu fx_chain): "
     "(bits − 0x4B400000)·m = IMAD(bits, m, −0x4B400000·m), one IMAD with a per-position constant addend (a register "
     "or constant-bank operand) in place of fx1's IMUL"),
]

#: why the other variants keep the extraction on every route
NO_ROUTE = ("its fold's entry is XOR (duo; LI-CP-W's d₀), which can't absorb an add, and its operands (e4m3i ±352, "
            "g1's ±353) are not s8")

#: FPX-DESIGN af74482571 pinned the B200 extraction to IMAD; its 5b835f1e60 added the class split itself
SPLIT = dict(device="sm_100", name="fpx-2sp-e4m3i-split", base="dither-2s-licpw-redux-heavy-l32", plain=1.0,
             fold_add={"ALU": 0.5, "IMAD": 0.5}, title="FP8 2SP e4m3i, extraction split ALU/IMAD (RT-FP)",
             summary="fpx-2sp-e4m3i with the extraction split between IADD3 and the IMAD-pipe add")


def load(path: Path):
    spec = importlib.util.spec_from_file_location("fpx_eps", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def route(v: dict) -> str | None:
    return next((why for d, p, b, why in ROUTES
                 if d in ("*", v["device"]) and v["name"].startswith(p) and b in ("*", v["base"])), None)


def priced(data: Path, design: str, vector: str) -> dict:
    cmd = ["machine-model", "--data", str(data), "eps", design, "--credit", "floor", "--operands", "floor", "--json"]
    return json.loads(subprocess.run(cmd + (["--vector", vector] if vector else []), capture_output=True, text=True,
                                     check=True).stdout)


def main(fpx_eps: Path, out: Path) -> int:
    fpx = load(fpx_eps)
    variants = list(fpx.VARIANTS)
    if not any(v["name"] == SPLIT["name"] for v in variants):
        variants.append(SPLIT)
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for dev, spec in fpx.DEVICES.items():
        data = out / "data" / dev
        shutil.rmtree(data, ignore_errors=True)
        shutil.copytree(spec["data"], data, ignore=shutil.ignore_patterns("devices") if dev == "sm_120" else None)
        attacks = []
        for v in (v for v in variants if v["device"] == dev):
            fpx.write_variant(v, data)
            why = route(v)
            if why is None:
                skipped.append(dict(device=dev, variant=v["name"], why=NO_ROUTE))
                continue
            x0 = copy.deepcopy(v) | dict(name=f"rt-{v['name']}-x0", fold_add={},
                                         title=f"{v['name']} with the extraction absorbed (RT-FP)", summary=why)
            fpx.write_variant(x0, data)
            attacks.append((v["name"], x0["name"], why))
        for vec in spec["vectors"]:
            for var, cheat, why in attacks:
                v, c = priced(data, var, vec), priced(data, cheat, vec)
                credit = v["baseline"]["cycles_per_mma"]
                t = min(c["cheat"]["cycles_per_mma"], c["baseline"]["cycles_per_mma"])
                rows.append(dict(device=dev, vector=vec or "VF3r-16", variant=var, credit=credit,
                                 credit_binds=v["baseline"]["binding"], model_machine=v["terms"]["machine"],
                                 model_core=v["core"], cheater=cheat, cheat=t, bound=1 - t / credit, why=why))
    (out / "overcredit.json").write_text(json.dumps(dict(rows=rows, no_route=skipped), indent=1) + "\n")
    lines = ["| device | vector | variant | credited ♭ (binds) | model ε_machine | model core ε | T_cheat | ε_machine ≥ |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['device']} | {r['vector']} | {r['variant']} | {r['credit']:.3f} ({', '.join(r['credit_binds'])}) | "
              f"{100 * r['model_machine']:.2f}% | {100 * r['model_core']:.3f}% | {r['cheat']:.3f} | "
              f"**{100 * r['bound']:.1f}%** |" for r in rows]
    lines += ["", "| route | why the cheater's words are the variant's |", "|---|---|"]
    lines += [f"| {d} {p}* on {b} | {why} |" for d, p, b, why in ROUTES]
    lines += ["", f"No op-free route (the extraction stays on every route: {NO_ROUTE}): "
              + ", ".join(f"{s['device']} {s['variant']}" for s in skipped)]
    (out / "overcredit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
