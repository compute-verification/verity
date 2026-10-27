"""fpx_design's candidate FP designs on the machine model: each a device's int8 record with what an FP route adds.

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
- `fold_add`: lane instructions a word added to the base's fold. `keep`, a part of it every route pays (fx1's keyed
  IMAD on sm_120's Dither-2S), and the rest the extraction, which turns FP32 accumulators into integers: the folded
  word's (one add, from a running sum or a reset at a magic base) and the served sum, read every RB words, RB the words
  the binade at 1.5 * 2^23 holds on any data: e4m3i120 (k32 words <= 32 * 121^2 = 2^18.84) RB 8, e4m3i352 (<= 2^21.9)
  RB 1. These are estimated lane counts, not measured.

The extraction is the honest FP route's and no cheater's: a cheater that writes the words on lanes, or on the int8 MMA,
never extracts, but the model charges a design's fold to every route and to its floor. So V's own ε is over-credited on
every device, B300 included, where the cheater puts 81-98% of its words on lanes. Each V with an extraction is therefore
also written as `V-x0`, the same design with `keep` for `fold_add`: V's row is its honest cost (γ), and V-x0's is its
credit, the floor the words cost with no extraction, and its ε. OUT/credit.md has, per reading, V's γ, V-x0's ε, and V's
own ε beside the bound 1 - T / ♭_V, T the lesser of V-x0's cheater and floor: what crediting V's own floor would give a
cheater that doesn't extract, at V's L, marked (`CAVEATS`) where that is no witness (operands past s8, or FP4's k64
words). `rt_fp/overcredit.py` prices the routes at other L's (FP4's chained int8 pair, e4m3i352's limbs).

g1 (gap-one) serves the FP model's nearest levels R, in row units, moved by a fair live coin b of 1 and centred:
R + b - 1/2. The one GEMM of the integers R + b is int8 Dither-2S's on fair coins, folded and served (its rank-one
terms off in the epilogue). E4M3's integers to 120 and E2M1x2 at integer block scales to 10 (10 * 12 + 1) stay in
int8, so on int8 units g1 is its device's Dither-2S design: on B200 and H200 the record itself, on sm_120 Dither-2S
with fx1's keyed IMAD a word (`fpx-g1s-fx1`), which is not sm_120's record. That is H1 (co32c-u16;
`dither-c-peel-fx1`): weights to 110 with U on +-16, so g1 runs on it with E4M3 weights to 104 and block scales to 9
(`g1h_check.py`), at H1's ε only if its centred served word is the committed one; `g1h_eps.py` prices the correction
as a skip term otherwise.

FP8-A1 (`fpx-a1-*`: weights the level below plus a coin at its gap, activations the nearest level plus
its gap times b + b' - 1) is 2SP with its twin and served GEMM one: each k32
word restarts from +0 and is exact in FP32 (32 * 448^2 < 2^23), so it pays 2SP's extraction a word and no second GEMM.
Its operands don't fit s8, its floor reads Lemma R2 (a paper proof), and no red team has looked for a cheaper route to
its words; like every extraction, its own is credited to no one (`fpx-a1-*-x0`). On B200 it is also priced at
L = 64 (`fpx-a1-e4m3i-fp-l64`: two k32 MMAs a word from +0, exact since 64 * 448^2 < 2^24), which halves the
extraction a MAC. H200 can't run it (the 14-bit window).

An FP4 word at L = 64 (one kind::mxf4nvf4 k64 firing) costs what an FP8 word at L = 32 costs at B200 rates and
carries twice the MACs, so its ε and its γ against a plain FP4 GEMM are the FP8 row's; on int8 units an FP4 design's
γ against plain FP4 is twice its row's (int8 runs at half FP4's rate). That price is the sm_100 `sheet.toml`
row `UTCOMMA_MXF4NVF4_M128_N256` (512 SMSP-cycles at K 64, `UTCIMMA_M128_N256`'s at K 32; measured at 2.000x kind::i8).
DS-100 has no e2m1 peak and no `UTCOMMA` class, so there a cheater's FP4 words are
unpriced, and the B200 FP4-unit rows' ε (`fpx-g1s-fp4-fp`) stands only once that row lands. Int8 words
through `UTCOMMA` take 9 E2M1 products (`lowbit_dc`), so at that row they
cost more than the int8 MMA's. The skip term is the base's, at the base's ♭: a larger fold raises ♭, so this
overstates ε_skip.
"""
import json
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
         fold_add={"ALU": 1.0, "IMAD": 1.0}, noise=TWO_SIDED, caveat="I352",
         title="FP8 peel (2SP) on e4m3i to 352: the twin dithered on the grid's gaps, in FP8",
         summary="served RN e4m3i352; the twin's word one add, the served sum one add a word"),
    dict(device="sm_120", name="fpx-a1-e4m3i-fp", base="dither-c-peel-fx1", plain=0.0,
         fold_add={"ALU": 1.0, "IMAD": 1.0}, noise=TWO_SIDED, caveat="A1",
         title="FP-NATIVE's FP8-A1 on e4m3i: one dithered FP8 GEMM, served and credited",
         summary="each k32 word from +0, converted (one instruction), the served sum one add a word: 2SP's extraction, "
                 "with no second GEMM"),
    dict(device="sm_120", name="fpx-g1s-fx1", base="dither-2s", plain=0.0, fold_add={"IMAD": 1.0}, keep={"IMAD": 1.0},
         title="g1 on int8 units (RTX PRO 6000): Dither-2S on R + b with fx1's keyed IMAD a word",
         summary="E4M3 integers to 120 (FP8) or E2M1x2 at integer block scales to 10 (FP4), nearest levels moved by a "
                 "fair coin of 1; one IMMA GEMM, folded and served centred"),
    dict(device="sm_120", name="fpx-g1s-fp8-fp", base="dither-2s", plain=1.0, fold_add={"IMAD": 1.6, "ALU": 0.6},
         keep={"IMAD": 1.0}, title="FP8 g1 on FP units: one FP16 GEMM of R + b (e4m3i120), fx1",
         summary="the word one add at a magic base, the served sum read every 8 words"),
    dict(device="sm_120", name="fpx-g1s-fp4-fp", base="dither-2s", plain=3.0, fold_add={"IMAD": 1.75, "ALU": 0.75},
         keep={"IMAD": 1.0}, caveat="NVI", title="FP4 g1 on FP4 units: R·R + R·u + b·R + b·u in one accumulator, any integer block scale, fx1",
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
         fold_add={"ALU": 0.75, "IMAD": 0.75}, caveat="NVI",
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
         fold_add={"ALU": 1.5, "IMAD": 1.5}, noise=TWO_SIDED, caveat="I352",
         title="FP8 peel (2SP) on e4m3i to 352: the twin dithered on the grid's gaps, in FP8",
         summary="served RN e4m3i352; the twin's word one add, the served sum an add and a rebase a word"),
    dict(device="sm_100", name="fpx-a1-e4m3i-fp", base="dither-2s-licpw-redux-heavy-l32", plain=0.0,
         fold_add={"ALU": 1.5, "IMAD": 1.5}, noise=TWO_SIDED, caveat="A1",
         title="FP-NATIVE's FP8-A1 on e4m3i: one dithered FP8 GEMM, served and credited",
         summary="each k32 word from +0, 2SP's extraction a word on B200 (the word one add, the served sum an add and a "
                 "rebase a word), with no second GEMM"),
    dict(device="sm_100", name="fpx-a1-e4m3i-fp-l64", base="dither-2s-licpw-redux-heavy-l32", L=64, plain=0.0,
         fold_add={"ALU": 1.5, "IMAD": 1.5}, noise=TWO_SIDED, caveat="A1",
         title="FP-NATIVE's FP8-A1 at L = 64 on B200: each k64 word from +0 (64 * 448^2 < 2^24, exact in FP32)",
         summary="two k32 MMAs a word into one TMEM accumulator from +0, then the L = 32 row's extraction a word"),
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
             f"plain = {v['plain']!r}", *([f"L = {v['L']}"] if "L" in v else []), "", "[fold]",
             f"name = {toml_value(name + ('+x' + added if added else ''))}"]
    lines += [f"{k} = {x!r}" for k, x in fold.items()]
    if "noise" in v:
        lines += ["", "[noise]"] + [f"{k} = {toml_value(x)}" for k, x in v["noise"].items()]
    (data / "designs" / f"{v['name']}.toml").write_text("\n".join(lines) + "\n")


def x0(v: dict) -> dict | None:
    """V with no extraction (`keep` for `fold_add`), the design its credit is read from; None if V extracts nothing."""
    keep = v.get("keep", {})
    if v["fold_add"] == keep:
        return None
    return v | {"name": f"{v['name']}-x0", "fold_add": keep, "title": f"{v['title']}: its credit, with no extraction",
                "summary": f"{v['name']}'s words at the cost of a route that doesn't extract (lanes or the int8 MMA)"}


#: where V-x0's cheater at V's L, which the model lets write any word on the int8 MMA, is no route to V's own words;
#: V-x0's credited ε stands (a real cheater costs no less), but the bound on V's own floor is no witness
CAVEATS = {
    "A1": "operands to 448 are past s8, so no int8 route at V's L writes V's words: the bound is no witness; a limb "
          "route (m = 4, 16 int8 MMAs a word) is unpriced",
    "I352": "operands to 352 are past s8, so the bound at V's L is no witness; RT-FP's overcredit.py prices the limb "
            "route (m = 3, L 576)",
    "NVI": "V's FP4 words are k64, so the bound at V's L is no witness; RT-FP's overcredit.py prices two chained int8 "
           "MMAs at L 64 (block scales to 10, |R + b| <= 121) and two limbs at L 256 (scales to 15)",
}

CREDIT_HEAD = ("| device | reading | design | γ (V) | credited ε (V-x0) | V-x0's ♭ | V's ♭ | V's own ε | "
               "ε_machine at V's ♭ ≥ |\n|---|---|---|---|---|---|---|---|---|")


def credit_row(dev: str, reading: str, v: dict, j: dict, c: dict | None) -> str:
    """One design's line in credit.md: from V's JSON `j` and V-x0's `c` (None: V credits its own floor)."""
    own, flo = j["core"], j["baseline"]["cycles_per_mma"]
    if c is None:
        return f"| {dev} | {reading} | {v['name']} | {j['gamma']:.2f} | {100 * own:.3f}% | {flo:.2f} | {flo:.2f} | — | — |"
    t = min(c["cheat"]["cycles_per_mma"], c["baseline"]["cycles_per_mma"]) * c["L"] / j["L"]
    mark = f" ({v['caveat']})" if "caveat" in v else ""
    return (f"| {dev} | {reading} | {v['name']} | {j['gamma']:.2f} | {100 * c['core']:.3f}% | "
            f"{c['baseline']['cycles_per_mma']:.2f} | {flo:.2f} | {100 * own:.3f}% | "
            f"**{100 * (1 - t / flo):.1f}%**{mark} |")


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    head = tomllib.loads((DATA / "headline.toml").read_text())
    with (out / "headline.txt").open("w") as f:
        rc = subprocess.run(["machine-model", "headline"], stdout=f, stderr=subprocess.STDOUT).returncode
    credit = [CREDIT_HEAD]
    for dev, src in DEVICES.items():
        data = out / "data" / dev
        shutil.copytree(src, data, ignore=shutil.ignore_patterns("devices") if dev == "sm_120" else None)
        mine = [v for v in VARIANTS if v["device"] == dev]
        pairs = [(v, x0(v)) for v in mine]
        written = mine + [c for _, c in pairs if c is not None]
        for v in written:
            write_variant(v, data)
        names = sorted({v["base"] for v in mine}) + [v["name"] for v in written]
        fam, q = head["family"][dev], {r: x["q"] for r, x in head["reading"].items()}
        readings = [(r, fam[r]["vector"], q[r]) for r in q] + [("measured-half", fam["measured"]["vector"], q["sheet"])]
        with (out / f"{dev}.txt").open("w") as f:
            for reading, vec, qr in readings:
                at = ["--credit", "floor", "--operands", "free", "--q", f"{qr:g}", "--vector", vec]
                cmd = ["machine-model", "--data", str(data), "eps", "--table", *names, *at]
                f.write(f"=== {dev} {reading}: {' '.join(cmd[3:])}\n")
                f.flush()
                rc = rc or subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
                got = {}
                for v in written:
                    cmd = ["machine-model", "--data", str(data), "eps", v["name"], *at, "--json"]
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    (out / f"{dev}_{reading}_{v['name']}.json").write_text(p.stdout)
                    rc = rc or p.returncode
                    got[v["name"]] = json.loads(p.stdout) if p.returncode == 0 else None
                for v, c in pairs:
                    if got[v["name"]] is not None and (c is None or got[c["name"]] is not None):
                        credit.append(credit_row(dev, reading, v, got[v["name"]], c and got[c["name"]]))
    credit += ["", ("γ is V's (the honest FP route, extraction paid); the credited ε is V-x0's, at V-x0's floor ♭ "
                    "(cycles an MMA). V's own ε credits V's ♭, which the last column bounds from below with V-x0's "
                    "cheapest route at V's L; — where V extracts nothing. Each ε is core (skip, machine, luck).")]
    credit += [""] + [f"- ({k}) {why}" for k, why in CAVEATS.items()]
    (out / "credit.md").write_text("\n".join(credit) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
