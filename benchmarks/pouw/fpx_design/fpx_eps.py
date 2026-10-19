"""FPX-DESIGN's candidate FP designs on the machine model: each a device's int8 record with what an FP route adds.

    uv run --project benchmarks/pouw/machine_model python benchmarks/pouw/fpx_design/fpx_eps.py OUT

For each device it copies the machine model's data directory to OUT/data/<device>, writes the variants below as design
files on their int8 bases (`base`, so the skip, luck, clock and baseline are the base's), and writes OUT/<device>.txt:
`machine-model eps --table` at the floor credit with operands at the floor and free, on each vector. An FP route adds:

- `plain`: MMAs (FP8-slot equivalents) beside the folded one, unfolded, so they enter γ only: the served GEMM (1), the
  FP16 twin's second slot (1), C1's extra limb products (k4 and k4t: 3; k3h: 2; k9: 3.5).
- the extraction (`fold_add`): lane instructions a word that turn FP32 accumulators into integers. The folded word's
  (one add, from a running sum or a reset at a magic base) and the served GEMM's exact sum, read every RB words, RB the
  words the binade at 1.5 * 2^23 holds: e4m3i120 (k32 words <= 2^18.8) RB 9, e4m3i352 (<= 2^21.9) RB 1, so its served
  sum costs an add a word and a rebase. On int8 units (g1-i8: E4M3's integers to 120, E2M1x2 at integer block scales to
  10, each moved by one, stay in int8) both GEMMs accumulate exactly in s32 and nothing is added.

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

DEVICES = {
    "sm_120": {"data": DATA, "vectors": ["", "DS-120"]},
    "sm_90": {"data": DATA / "devices" / "sm_90", "vectors": ["DS-90", "DC-90-ri"]},
    "sm_100": {"data": DATA / "devices" / "sm_100", "vectors": ["DS-100", "DC-100-ri"]},
}

TWO_SIDED = {"linears": "two-sided", "lm_head": "two-sided", "qk": "two-sided", "pv": "peel"}

VARIANTS = [
    # sm_120 (RTX PRO 6000): H1 + fx1 is the record; int8 runs at the FP8 rate (QMMA 0.985 IMMA). mma.sync's C operand
    # resets a word at the magic base for free
    dict(device="sm_120", name="fpx-g1-i8", base="dither-c-peel-fx1", plain=1.0, fold_add={}, noise=TWO_SIDED,
         title="g1 on int8 units: served E4M3 integers to 120 (or E2M1x2 at integer block scales to 10) RN, twin R + b",
         summary="the twin (R + b)(R + u), b and u fair live coins, is int8 Dither-2S on IMMA, every word folded; the "
                 "served RN GEMM a second IMMA; served = twin - peel checked in the proof"),
    dict(device="sm_120", name="fpx-g1-e4m3i120-fp", base="dither-c-peel-fx1", plain=2.0,
         fold_add={"ALU": 0.6, "IMAD": 0.6}, noise=TWO_SIDED,
         title="FP8 g1 on FP units: the twin in FP16, served e4m3i120 in FP8",
         summary="the twin's word one add, the served sum read every 9 words"),
    dict(device="sm_120", name="fpx-g1-e4m3i352-fp", base="dither-c-peel-fx1", plain=2.0,
         fold_add={"ALU": 1.0, "IMAD": 1.0}, noise=TWO_SIDED,
         title="FP8 g1 on FP units: the twin in FP16, served e4m3i352 in FP8",
         summary="the twin's word one add, the served sum one add a word"),
    dict(device="sm_120", name="fpx-c1-k4-h1-split", base="dither-c-peel-fx1", plain=3.0,
         fold_add={"ALU": 0.5, "IMAD": 0.5},
         title="C1 k4 on H1: int8 Dither's transcript from four E4M3 limb products a word, the extraction split",
         summary="H1 + fx1's transcript bit for bit, each k32 word from 4 QMMA limb products and one add"),
    dict(device="sm_120", name="fpx-2sp-e4m3i352-fp", base="dither-c-peel-fx1", plain=1.0,
         fold_add={"ALU": 1.0, "IMAD": 1.0}, noise=TWO_SIDED,
         title="FP8 peel (2SP) on e4m3i to 352: the twin dithered on the grid's gaps, in FP8",
         summary="served RN e4m3i352; the twin's word one add, the served sum one add a word"),
    dict(device="sm_120", name="fpx-g1-fp4-fp", base="dither-c-peel-fx1", plain=4.0,
         fold_add={"ALU": 0.75, "IMAD": 0.75}, noise=TWO_SIDED,
         title="FP4 g1 on FP4 units: twin R·R + R·u + b·R + b·u, served R·R, any integer block scale",
         summary="five block-scaled k64 FP4 products a word, u and b at unit scale; the twin's word one add, the "
                 "served sum (<= 2^21) read every other word"),
    # sm_90 (H200): wgmma E4M3 keeps a 14-bit adder window, so E4M3 integers past ±16 are exact only in FP16
    dict(device="sm_90", name="fpx-g1-i8", base="dither-2s-licpw-redux-imad-l32", plain=1.0, fold_add={},
         title="g1 on int8 units (H200)", summary="as sm_120's fpx-g1-i8"),
    dict(device="sm_90", name="fpx-g1-e4m3i120-f16", base="dither-2s-licpw-redux-imad-l32", plain=3.0,
         fold_add={"IMAD": 0.6, "ALU": 0.6},
         title="FP8 g1 on H200's FP units: served and twin both FP16",
         summary="the e4m3i products leave the FP8 14-bit window, so both GEMMs run in FP16 at half the FP8 rate"),
    dict(device="sm_90", name="fpx-c1-k3h", base="dither-2s-licpw-redux-imad-l32", plain=2.0,
         fold_add={"IMAD": 3.0, "ALU": 1.0},
         title="C1 k3h on the H200 record: int8 Dither's transcript from three E4M3 Karatsuba limb products",
         summary="the H200 LI-CP-W record's transcript bit for bit, each k32 word from 3 wgmma E4M3 into 3 "
                 "accumulators from +0 and 3 FFMA + IADD3"),
    dict(device="sm_90", name="fpx-2sp-int5-e4r", base="dither-2s-licpw-redux-imad-l32", plain=1.0,
         fold_add={"IMAD": 1.0, "ALU": 2.0}, noise=TWO_SIDED,
         title="peel (2SP) on the ±16 integer grid (e4r), H200",
         summary="the only native grid exact under the 14-bit window: int5 served round-to-nearest, twin dithered, "
                 "restart from +0 each k32 word, FFMA + IADD, the served sum one add a word"),
    # sm_100 (B200): the LI-CP-W redux record; TMEM accumulators, rebased by tcgen05.st
    dict(device="sm_100", name="fpx-g1-i8", base="dither-2s-licpw-redux-heavy-l32", plain=1.0, fold_add={},
         title="g1 on int8 units (B200)", summary="as sm_120's fpx-g1-i8"),
    dict(device="sm_100", name="fpx-g1-i8-duo", base="dither-2s-l32", plain=1.0, fold_add={},
         title="g1 on int8 units (B200), Dither-2S's duo fold", summary="as fpx-g1-i8 on the duo fold"),
    dict(device="sm_100", name="fpx-g1-i8-l64", base="dither-2s-l64", plain=1.0, fold_add={},
         title="g1 on int8 units (B200), duo fold at L = 64", summary="as fpx-g1-i8 on k64 words"),
    dict(device="sm_100", name="fpx-g1-e4m3i120-fp", base="dither-2s-licpw-redux-heavy-l32", plain=2.0,
         fold_add={"ALU": 0.625, "IMAD": 0.625},
         title="FP8 g1 on B200's FP units: the twin in FP16, served e4m3i120 in FP8",
         summary="the twin's word one add, each accumulator rebased every 9 words"),
    dict(device="sm_100", name="fpx-g1-e4m3i352-fp", base="dither-2s-licpw-redux-heavy-l32", plain=2.0,
         fold_add={"ALU": 1.5, "IMAD": 1.5},
         title="FP8 g1 on B200's FP units: the twin in FP16, served e4m3i352 in FP8",
         summary="the twin's word one add, the served sum an add and a rebase a word"),
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
    dict(device="sm_100", name="fpx-g1-fp4-fp", base="dither-2s-licpw-redux-heavy-l32", plain=4.0,
         fold_add={"ALU": 1.0, "IMAD": 1.0},
         title="FP4 g1 on FP4 units: twin R·R + R·u + b·R + b·u, served R·R, any integer block scale",
         summary="five block-scaled k64 FP4 products a word, u and b at unit scale; the twin's word one add, the "
                 "served sum (<= 2^21) an add and a rebase every other word"),
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
    rc = 0
    for dev, spec in DEVICES.items():
        data = out / "data" / dev
        shutil.copytree(spec["data"], data, ignore=shutil.ignore_patterns("devices") if dev == "sm_120" else None)
        mine = [v for v in VARIANTS if v["device"] == dev]
        for v in mine:
            write_variant(v, data)
        names = sorted({v["base"] for v in mine}) + [v["name"] for v in mine]
        with (out / f"{dev}.txt").open("w") as f:
            for vec in spec["vectors"]:
                for ops in ("floor", "free"):
                    cmd = ["machine-model", "--data", str(data), "eps", "--table", *names, "--credit", "floor",
                           "--operands", ops] + (["--vector", vec] if vec else [])
                    f.write(f"=== {dev} vector {vec or 'default'} operands {ops}: {' '.join(cmd[3:])}\n")
                    f.flush()
                    r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
                    rc = rc or r.returncode
            for v in mine:
                cmd = ["machine-model", "--data", str(data), "eps", v["name"], "--credit", "floor", "--operands",
                       "floor", "--vector", spec["vectors"][-1], "--json"]
                with (out / f"{dev}_{v['name']}.json").open("w") as g:
                    rc = rc or subprocess.run(cmd, stdout=g, stderr=subprocess.DEVNULL).returncode
    return rc


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
