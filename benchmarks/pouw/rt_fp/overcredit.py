"""RT-FP's over-credit witness for FPX-DESIGN's FP designs: each variant's credited floor ♭ against a route that writes
the same committed words for less.

    uv run --project benchmarks/pouw/machine_model python benchmarks/pouw/rt_fp/overcredit.py FPX_EPS OUT

FPX_EPS is FPX-DESIGN's `benchmarks/pouw/fpx_design/fpx_eps.py` (its VARIANTS and `write_variant` are read as they are,
so this prices exactly what it pushed). Each variant puts the FP32 word's extraction into the design's `[fold]`
(`fold_add`), and the LP charges that fold to every tensor form the cheater may write on (eps.py `fastest`), the int8
ones included, and to the credited floor program. data/forms.toml states the opposite convention: converting an FP
accumulator to s32 costs the cheater nothing. On every device here an int8 MMA writes the s32 word with no extraction.

For each attack the cheater runs a design whose committed words equal the variant's, and the bound is
ε_machine ≥ 1 − T_cheat / ♭_variant, with T_cheat the lesser of that design's cheater LP and its own floor program.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

#: the cheater's route against each variant: (variant, the design it runs, why its committed words are the variant's)
ATTACKS = {
    "sm_120": [
        ("fpx-c1-k4-h1", "dither-c-peel-fx1", "C1: H1 + fx1's transcript bit for bit (FPX-DESIGN's summary), run on IMMA"),
        ("fpx-2sp-e4m3i-fx1", "dither-c-peel-fx1",
         "fx1's entry is an IMAD by a compile-time key m (cgemm.cu fx_chain): (bits − 0x4B400000)·m = "
         "IMAD(bits, m, −0x4B400000·m), the addend a per-position constant held in a register; QMMA at 0.985 IMMA's "
         "rate on an issue-bound row"),
    ],
    "sm_90": [
        ("fpx-c1-k3h", "dither-2s-licpw-redux-imad-l32", "C1: the H200 LI-CP-W record's transcript bit for bit, run on wgmma s8"),
        ("fpx-2sp-int5-e4r", "dither-2s-licpw-redux-imad-l32", "the int5 twin's values lie in [−16, 16] ⊂ s8: wgmma s8 writes them"),
    ],
    "sm_100": [
        ("fpx-c1-k4t", "dither-2s-licpw-redux-heavy-l32", "C1: the B200 LI-CP-W redux record's transcript bit for bit, run on kind::i8"),
        ("fpx-c1-k9", "dither-2s-licpw-redux-heavy-l32", "C1: as fpx-c1-k4t"),
        ("fpx-c1-k4t-duo", "dither-2s-l32", "C1: Dither-2S's duo transcript bit for bit, run on kind::i8"),
        ("fpx-2sp-e4m3i", "rt-2sp-e4m3i-split",
         "the same FP route with the extraction (a two-input add) half on IADD3, half on the IMAD pipe: fpx_eps pins it "
         "to IMAD, which is what makes the IMAD row bind"),
    ],
}

#: routes the floor program may take that fpx_eps doesn't write: the extraction's class split
EXTRA = [dict(device="sm_100", name="rt-2sp-e4m3i-split", base="dither-2s-licpw-redux-heavy-l32", plain=1.0,
              fold_add={"ALU": 0.5, "IMAD": 0.5}, title="FP8 2SP e4m3i, extraction split ALU/IMAD",
              summary="fpx-2sp-e4m3i with the extraction split between IADD3 and the IMAD-pipe add")]


def load(path: Path):
    spec = importlib.util.spec_from_file_location("fpx_eps", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def priced(data: Path, design: str, vector: str) -> dict:
    cmd = ["machine-model", "--data", str(data), "eps", design, "--credit", "floor", "--operands", "floor", "--json"]
    return json.loads(subprocess.run(cmd + (["--vector", vector] if vector else []), capture_output=True, text=True,
                                     check=True).stdout)


def main(fpx_eps: Path, out: Path) -> int:
    fpx = load(fpx_eps)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dev, attacks in ATTACKS.items():
        spec = fpx.DEVICES[dev]
        data = out / "data" / dev
        shutil.rmtree(data, ignore_errors=True)
        shutil.copytree(spec["data"], data, ignore=shutil.ignore_patterns("devices") if dev == "sm_120" else None)
        for v in fpx.VARIANTS + EXTRA:
            if v["device"] == dev:
                fpx.write_variant(v, data)
        for vec in spec["vectors"]:
            for var, cheat, why in attacks:
                v, c = priced(data, var, vec), priced(data, cheat, vec)
                credit = v["baseline"]["cycles_per_mma"]
                t = min(c["cheat"]["cycles_per_mma"], c["baseline"]["cycles_per_mma"])
                rows.append(dict(device=dev, vector=vec or "VF3r-16", variant=var, credit=credit,
                                 credit_binds=v["baseline"]["binding"], model_machine=v["terms"]["machine"],
                                 model_core=v["core"], cheater=cheat, cheat=t, bound=1 - t / credit, why=why))
    (out / "overcredit.json").write_text(json.dumps(rows, indent=1) + "\n")
    lines = ["| device | vector | variant | credited ♭ (binds) | model ε_machine | model core ε | cheater's design | "
             "T_cheat | ε_machine ≥ | the cheater's words |", "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['device']} | {r['vector']} | {r['variant']} | {r['credit']:.3f} ({', '.join(r['credit_binds'])}) | "
              f"{100 * r['model_machine']:.2f}% | {100 * r['model_core']:.3f}% | {r['cheater']} | {r['cheat']:.3f} | "
              f"**{100 * r['bound']:.1f}%** | {r['why']} |" for r in rows]
    (out / "overcredit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
