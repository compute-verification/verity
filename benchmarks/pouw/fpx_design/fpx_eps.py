"""FPX-DESIGN's candidate FP designs on the machine model: each a device's int8 record with what an FP route adds.

    uv run --project benchmarks/pouw/machine_model python benchmarks/pouw/fpx_design/fpx_eps.py OUT

For each device it copies the machine model's data directory to OUT/data/<device>, writes the variants below as design
files on their int8 bases (`base`, so the skip, luck, clock and baseline are the base's), and writes OUT/<device>.txt:
`machine-model eps --table` at the record's convention (credit ♭, operands free) at each reading of data/headline.toml,
the sheet's vector at q* = ½ and the measured one at q* = 1.22, and the measured vector at the proved ½ floor
(`measured-half`). OUT/headline.txt is `machine-model headline`: the records' rows with the conditions every ε here is
stated under (the 1.22 reading rests on Theorem C, unproved). An FP route adds:

- `plain`: MMAs (FP8-slot equivalents) beside the folded one, unfolded, so they enter γ only: an FP16 GEMM's second
  slot (1), the FP4 g1 GEMM's other three block-scaled products (3), C1's extra limb products (k4 and k4t: 3; k3h: 2;
  k9: 3.5), or the 2SP peel's served GEMM beside its twin (1).
- the extraction (`fold_add`): lane instructions a word that turn FP32 accumulators into integers. The folded word's
  (one add, from a running sum or a reset at a magic base) and the served sum, read every RB words, RB the words the
  binade at 1.5 * 2^23 holds on any data: e4m3i120 (k32 words <= 32 * 121^2 = 2^18.84) RB 8, e4m3i352 (<= 2^21.9)
  RB 1. These are estimated lane counts, not measured.

Where int8 runs at the FP8 rate (sm_120, H200, B200), the int8 MMA writes the C1 rows' and the FP-unit g1 rows' words
with no extraction, so the cheater's floor is the int8 twin's and those rows' printed ε is too low there (RT-FP-1, -3,
-5, note:20261019T0546Z-draft-rt-fp). Their ε stands only where int8 is slow (B300, which has no sheet yet); on the
other three, credit the int8 twin (`fpx-g1s-*`, the record) and read the FP route's cost in γ only.

g1 (gap-one) serves the FP model's nearest levels R, in row units, moved by a fair live coin b of 1 and centred:
R + b - 1/2. The one GEMM of the integers R + b is int8 Dither-2S's on fair coins, folded and served (its rank-one
terms off in the epilogue). E4M3's integers to 120 and E2M1x2 at integer block scales to 10 (10 * 12 + 1) stay in
int8, so on int8 units g1 is its device's Dither-2S design: on B200 and H200 the record itself, on sm_120 Dither-2S
with fx1's keyed IMAD a word (`fpx-g1s-fx1`).

FP-NATIVE's FP8-A1 (`fpx-a1-*`: weights the level below plus a coin at its gap, activations the nearest level plus
its gap times b + b' - 1, note:20261019T0525Z-draft-fp-native) is 2SP with its twin and served GEMM one: each k32
word restarts from +0 and is exact in FP32 (32 * 448^2 < 2^23), so it pays 2SP's extraction a word and no second GEMM.
Its operands don't fit s8, its floor reads Lemma R2 (a paper proof), and no red team has looked for a cheaper route to
its words, so its ε is printed at its own floor. H200 can't run it (the 14-bit window).

An FP4 word at L = 64 (one kind::mxf4nvf4 k64 firing) costs what an FP8 word at L = 32 costs at B200 rates and
carries twice the MACs, so its ε and its γ against a plain FP4 GEMM are the FP8 row's; on int8 units an FP4 design's
γ against plain FP4 is twice its row's (int8 runs at half FP4's rate). The skip term is the base's, at the base's ♭: a
larger fold raises ♭, so this overstates ε_skip.
"""
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

MM = Path(__file__).resolve().parents[1] / "machine_model"
DATA = MM / "data"

DEVICES = {"sm_120": DATA, "sm_90": DATA / "devices" / "sm_90", "sm_100": DATA / "devices" / "sm_100"}

TWO_SIDED = {"linears": "two-sided", "lm_head": "two-sided", "qk": "two-sided", "pv": "peel"}

VARIANTS = [
    # sm_120 (RTX PRO 6000): H1 + fx1 is the record; int8 runs at the FP8 rate (QMMA 0.985 IMMA). mma.sync's C operand
    # resets a word at the magic base for free
    dict(device="sm_120", name="fpx-c1-k4-h1-split", base="dither-c-peel-fx1", plain=3.0,
         fold_add={"ALU": 0.5, "IMAD": 0.5},
         title="C1 k4 on H1: int8 Dither's transcript from four E4M3 limb products a word, the extraction split",
         summary="H1 + fx1's transcript bit for bit, each k32 word from 4 QMMA limb products and one add"),
    dict(device="sm_120", name="fpx-2sp-e4m3i352-fp", base="dither-c-peel-fx1", plain=1.0,
         fold_add={"ALU": 1.0, "IMAD": 1.0}, noise=TWO_SIDED,
         title="FP8 peel (2SP) on e4m3i to 352: the twin dithered on the grid's gaps, in FP8",
         summary="served RN e4m3i352; the twin's word one add, the served sum one add a word"),
    dict(device="sm_120", name="fpx-a1-e4m3i-fp", base="dither-c-peel-fx1", plain=0.0,
         fold_add={"ALU": 1.0, "IMAD": 1.0}, noise=TWO_SIDED,
         title="FP-NATIVE's FP8-A1 on e4m3i: one dithered FP8 GEMM, served and credited",
         summary="each k32 word from +0, converted (one instruction), the served sum one add a word: 2SP's extraction, "
                 "with no second GEMM"),
    dict(device="sm_120", name="fpx-g1s-fx1", base="dither-2s", plain=0.0, fold_add={"IMAD": 1.0},
         title="g1 on int8 units (RTX PRO 6000): Dither-2S on R + b with fx1's keyed IMAD a word",
         summary="E4M3 integers to 120 (FP8) or E2M1x2 at integer block scales to 10 (FP4), nearest levels moved by a "
                 "fair coin of 1; one IMMA GEMM, folded and served centred"),
    dict(device="sm_120", name="fpx-g1s-fp8-fp", base="dither-2s", plain=1.0, fold_add={"IMAD": 1.6, "ALU": 0.6},
         title="FP8 g1 on FP units: one FP16 GEMM of R + b (e4m3i120), fx1",
         summary="the word one add at a magic base, the served sum read every 8 words"),
    dict(device="sm_120", name="fpx-g1s-fp4-fp", base="dither-2s", plain=3.0, fold_add={"IMAD": 1.75, "ALU": 0.75},
         title="FP4 g1 on FP4 units: R·R + R·u + b·R + b·u in one accumulator, any integer block scale, fx1",
         summary="four block-scaled k64 FP4 products a word, u and b at unit scale"),
    # sm_90 (H200): wgmma E4M3 keeps a 14-bit adder window, so E4M3 integers past ±16 are exact only in FP16
    dict(device="sm_90", name="fpx-c1-k3h", base="dither-2s-licpw-redux-l32", plain=2.0,
         fold_add={"IMAD": 3.0, "ALU": 1.0},
         title="C1 k3h on the H200 record: int8 Dither's transcript from three E4M3 Karatsuba limb products",
         summary="the H200 LI-CP-W record's transcript bit for bit, each k32 word from 3 wgmma E4M3 into 3 "
                 "accumulators from +0 and 3 FFMA + IADD3"),
    dict(device="sm_90", name="fpx-2sp-int5-e4r", base="dither-2s-licpw-redux-l32", plain=1.0,
         fold_add={"IMAD": 1.0, "ALU": 2.0}, noise=TWO_SIDED,
         title="peel (2SP) on the ±16 integer grid (e4r), H200",
         summary="the only native grid exact under the 14-bit window: int5 served round-to-nearest, twin dithered, "
                 "restart from +0 each k32 word, FFMA + IADD, the served sum one add a word"),
    dict(device="sm_90", name="fpx-g1s-i8", base="dither-2s-licpw-redux-l32", plain=0.0, fold_add={},
         title="g1 on int8 units (H200): the record on R + b", summary="as sm_120's fpx-g1s-fx1, on the record's fold"),
    # sm_100 (B200): the LI-CP-W redux record; TMEM accumulators, rebased by tcgen05.st
    dict(device="sm_100", name="fpx-g1s-i8", base="dither-2s-licpw-redux-heavy-l32", plain=0.0, fold_add={},
         title="g1 on int8 units (B200): the record on R + b", summary="as sm_120's fpx-g1s-fx1, on the record's fold"),
    dict(device="sm_100", name="fpx-g1s-fp8-fp", base="dither-2s-licpw-redux-heavy-l32", plain=1.0,
         fold_add={"ALU": 0.625, "IMAD": 0.625},
         title="FP8 g1 on B200's FP units: one FP16 GEMM of R + b (e4m3i120)",
         summary="the word one add, the running sum rebased every 8 words"),
    dict(device="sm_100", name="fpx-g1s-fp4-fp", base="dither-2s-licpw-redux-heavy-l32", plain=3.0,
         fold_add={"ALU": 0.75, "IMAD": 0.75},
         title="FP4 g1 on B200's FP4 units: R·R + R·u + b·R + b·u in one accumulator, any integer block scale",
         summary="four block-scaled k64 FP4 products a word, u and b at unit scale, the sum rebased every other word"),
    dict(device="sm_100", name="fpx-c1-k4t-split", base="dither-2s-licpw-redux-heavy-l32", plain=3.0,
         fold_add={"ALU": 0.6, "IMAD": 0.6},
         title="C1 k4t on the B200 record: int8 Dither's transcript from four E4M3 limb products",
         summary="the B200 record's transcript bit for bit, the running sum in TMEM, one add a word and a rebase every "
                 "7 words (LOWBIT-DC's k4t)"),
    dict(device="sm_100", name="fpx-c1-k9", base="dither-2s-licpw-redux-heavy-l32", plain=3.5,
         fold_add={"ALU": 0.6, "IMAD": 0.6},
         title="C1 k9: int8 Dither's transcript from nine E2M1 digit products (kind::mxf4)",
         summary="the B200 record's transcript bit for bit, 9 unit-scale E2M1 products a k32 word at FP4's 2x rate"),
    dict(device="sm_100", name="fpx-2sp-e4m3i352-fp", base="dither-2s-licpw-redux-heavy-l32", plain=1.0,
         fold_add={"ALU": 1.5, "IMAD": 1.5}, noise=TWO_SIDED,
         title="FP8 peel (2SP) on e4m3i to 352: the twin dithered on the grid's gaps, in FP8",
         summary="served RN e4m3i352; the twin's word one add, the served sum an add and a rebase a word"),
    dict(device="sm_100", name="fpx-a1-e4m3i-fp", base="dither-2s-licpw-redux-heavy-l32", plain=0.0,
         fold_add={"ALU": 1.5, "IMAD": 1.5}, noise=TWO_SIDED,
         title="FP-NATIVE's FP8-A1 on e4m3i: one dithered FP8 GEMM, served and credited",
         summary="each k32 word from +0, 2SP's extraction a word on B200 (the word one add, the served sum an add and a "
                 "rebase a word), with no second GEMM"),
]


def resolved(name: str, data: Path) -> dict:
    """A design's table on its bases, as the machine model reads it (data/designs.toml under the first)."""
    t = tomllib.loads((data / "designs" / f"{name}.toml").read_text())
    under = resolved(t["base"], data) if "base" in t else tomllib.loads((data / "designs.toml").read_text())
    return {**{k: v for k, v in under.items() if k not in ("name", "title", "summary", "aliases", "base")}, **t}


def honest_fold(t: dict, data: Path) -> dict:
    if "fold" in t:
        return dict(t["fold"])
    return dict(tomllib.loads((data / "honest" / f"{t['honest']}.toml").read_text())["fold"])


def toml_value(v) -> str:
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {toml_value(x)}" for k, x in v.items()) + " }"
    return repr(v)


def write_variant(v: dict, data: Path) -> None:
    base = resolved(v["base"], data)
    fold = honest_fold(base, data)
    name = fold.pop("name")
    for k, x in v["fold_add"].items():
        fold[k] = fold.get(k, 0.0) + x
    added = "".join(f"+{k}{x:g}" for k, x in v["fold_add"].items())
    lines = [f"base = {toml_value(v['base'])}", f"name = {toml_value(v['name'])}",
             f"title = {toml_value(v['title'])}", f"summary = {toml_value(v['summary'])}",
             f"plain = {v['plain']!r}", "", "[fold]", f"name = {toml_value(name + ('+x' + added if added else ''))}"]
    lines += [f"{k} = {x!r}" for k, x in fold.items()]
    if "noise" in v:
        lines += ["", "[noise]"] + [f"{k} = {toml_value(x)}" for k, x in v["noise"].items()]
    (data / "designs" / f"{v['name']}.toml").write_text("\n".join(lines) + "\n")


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    head = tomllib.loads((DATA / "headline.toml").read_text())
    with (out / "headline.txt").open("w") as f:
        rc = subprocess.run(["machine-model", "headline"], stdout=f, stderr=subprocess.STDOUT).returncode
    for dev, src in DEVICES.items():
        data = out / "data" / dev
        shutil.copytree(src, data, ignore=shutil.ignore_patterns("devices") if dev == "sm_120" else None)
        mine = [v for v in VARIANTS if v["device"] == dev]
        for v in mine:
            write_variant(v, data)
        names = sorted({v["base"] for v in mine}) + [v["name"] for v in mine]
        fam, q = head["family"][dev], {r: x["q"] for r, x in head["reading"].items()}
        readings = [(r, fam[r]["vector"], q[r]) for r in q] + [("measured-half", fam["measured"]["vector"], q["sheet"])]
        with (out / f"{dev}.txt").open("w") as f:
            for reading, vec, qr in readings:
                at = ["--credit", "floor", "--operands", "free", "--q", f"{qr:g}", "--vector", vec]
                cmd = ["machine-model", "--data", str(data), "eps", "--table", *names, *at]
                f.write(f"=== {dev} {reading}: {' '.join(cmd[3:])}\n")
                f.flush()
                rc = rc or subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
                for v in mine:
                    cmd = ["machine-model", "--data", str(data), "eps", v["name"], *at, "--json"]
                    with (out / f"{dev}_{reading}_{v['name']}.json").open("w") as g:
                        rc = rc or subprocess.run(cmd, stdout=g, stderr=subprocess.DEVNULL).returncode
    return rc


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
