"""What the tensor-form probe (dc) times on each architecture: the one table every dc script and `sass_census/rates.sh`
read, keyed by the architecture's name as its file spells it (calibration/hardware/architectures/NAME.toml).

  python3 archs.py ARCH FIELD    # one field of ARCH's entry: a list one item a line, a flag 1 or nothing

An architecture the table doesn't hold is refused by name (`REFUSAL`, exit 2). FIELD `capability` is the architecture
file's, not the table's.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, fields
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Arch:
    family: str                   # the label of the probe's outputs and census rows (analysis.json, JI90_k)
    tensor: str                   # the tensor instruction it times: "wgmma" or "tcgen05"
    source: str                   # probes/<source>.cu, the forms run.sh builds
    gates: tuple[str, ...]        # dgemm's gates, "M N K SEED" each
    honest_forms: tuple[str, ...]  # the honest form's probe rows, the first the run timed
    honest_kind: str              # the honest form's census kind
    tmem: bool                    # times tcgen05.ld, TMEM's read port: census's TL and TM rows and the two-stage model
    fp8acc: bool                  # builds and gates fp8acc.cu (tcgen05 kind::f8f6f4)
    jbreak: str                   # probes/<jbreak>.cu, DC_JBREAK's stress rows ("": none)
    s8_forms: tuple[str, ...]     # s8_mma.sh's form families
    unpriced: str                 # sass_census/rates.sh's source


#: Keyed by the architecture file's name (NAME.toml).
ARCHS = {
    "sm_90a": Arch(family="sm90", tensor="wgmma", source="sm90_wgmma",
                   gates=("128 256 256 7", "256 512 1024 11", "192 384 2048 13"),
                   honest_forms=("wg_s8_n256_a1", "wg_s8_n256_a0"), honest_kind="IGMMA_S8_N256", tmem=False,
                   fp8acc=False, jbreak="sm90_jbreak", s8_forms=("mma", "wg", "wg64", "wg128", "wg256"),
                   unpriced="unpriced90.cu"),
    "sm_100a": Arch(family="sm100", tensor="tcgen05", source="sm100_tcgen05",
                    gates=("128 256 256 7", "256 512 1024 11", "384 768 2048 13"),
                    honest_forms=("tc_i8_m128_n256",), honest_kind="UTCIMMA_M128_N256", tmem=True, fp8acc=True,
                    jbreak="", s8_forms=("mma", "tc"), unpriced="unpriced.cu"),
}
FAMILIES = tuple(a.family for a in ARCHS.values())


def refusal(arch: str) -> str:
    return (f"the card's target is {arch}, and the tensor-form probe has no instructions for it, only for "
            f"{' and '.join(ARCHS)}: nothing built")


def of(arch: str) -> Arch:
    """``arch``'s entry; refused by name where the table holds none."""
    if arch not in ARCHS:
        raise SystemExit(refusal(arch))
    return ARCHS[arch]


def of_family(family: str) -> Arch:
    """The entry whose outputs carry the label ``family``; refused where none does."""
    found = [a for a in ARCHS.values() if a.family == family]
    if not found:
        raise SystemExit(f"no architecture of the tensor-form probe has the family label {family!r} "
                         f"(it has {', '.join(FAMILIES)})")
    return found[0]


def capability(arch: str) -> str:
    """``arch``'s compute capability, from its architecture file."""
    root = HERE.parents[2]
    sys.path[:0] = [str(root), str(root / "tools" / "config")]
    from verity.device import load_architecture

    return load_architecture(arch, directory=root / "calibration" / "hardware" / "architectures")["capability"]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    names = [f.name for f in fields(Arch)] + ["capability"]
    if len(argv) != 2 or argv[1] not in names:
        print(f"usage: archs.py ARCH FIELD, FIELD one of {', '.join(names)}", file=sys.stderr)
        return 2
    arch, field = argv
    if arch not in ARCHS:
        print(refusal(arch), file=sys.stderr)
        return 2
    entry = ARCHS[arch]
    value = capability(arch) if field == "capability" else getattr(entry, field)
    if isinstance(value, tuple):
        print("\n".join(value))
    elif isinstance(value, bool):
        print("1" if value else "")
    else:
        print(value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
