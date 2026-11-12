"""Boolean gates per counted operation: the conversion from the harness's multiply-accumulates (MACs) to the gates
the training formalism counts |C|, c and w in.

    python -m training_slowdown.gates           # the table
    python -m training_slowdown.gates --check   # recompute it from the Definitions (needs verity), exit 1 on a difference

Each price is one of the circuits' own Definitions (`verity.definitions.ops.boolean`, and the TF32 step's in
`verity.definitions.registry.boolean_mhc`), at its whole gate count: AND + XOR + NOT as
`verity.circuits.evaluation.bits.compile(D).counts` gives them, the AND count being the one its
`<module>.circuit-check.json` pins (`test_gates` holds both). A counted MAC is one multiply and one accumulate:

- a GEMM MAC is a tensor-core dot product's gates over its K: bf16 (and f16, which has no Definition) at
  `HopperBF16WgmmaDot16_v2` / 16, the 26-bit-adder model sm_120's BF16 pipeline shares (`ml.circuits.targets.target`);
  FP8 at `BlackwellE4m3QmmaDot32_v2` / 32 or `BlackwellE5m2QmmaDot32_v2` / 32, NVFP4 at `BlackwellNvf4OmmaDot64_v2` /
  64, f32 at `F32Fma_v4`, and an f32 GEMM under TF32 (cuda.matmul.allow_tf32, which a row names as the dtype `tf32`) at
  `HopperTf32WgmmaDot8_v2` / 8, the k8 step cuBLAS's unsplit TF32 GEMM takes K/8 of a dot product (`GemmTf32_v1`);
- an elementwise MAC (training_slowdown.flops's `mul`, `div`, `addcmul`, a scaled `add`, a counted forward op) is a bf16 multiply
  then a bf16 add (`Bf16MulBf16_v2` + `Bf16Add_v2`), or one `F32Fma_v4` in f32;
- dense linear algebra's MAC is an elementwise MAC of its dtype (`F32Fma_v4` in f32);
- a gather's MAC (training_slowdown.flops's `gather_macs`) is one word multiplexer, y ⊕ (b ∧ (x ⊕ y)) per bit: 3 gates per bit.

An op is priced in the dtype it ran in, its result's, which its row of a record's op log names
(`flops.Counter.record`) and a trace's node has: never by a rule of its class. An optimizer's product in bf16 is a bf16
GEMM, and a forward scaling autocast leaves in f32 is an f32 multiply. A dtype no Definition prices is refused, with
the op and the dtype (`Unpriced`): nothing is priced as the cheapest. A gather moves words of its dtype's bits
(`BITS`).

A torch dtype doesn't say whether a run set cuda.matmul.allow_tf32, and the counter writes torch's, so a float32 GEMM
row is priced as the exact f32 one, and only a row whose dtype reads `tf32` as the TF32 one: whoever reads a record of
a run that set it names its f32 GEMMs' rows so. `tf32` names a GEMM alone: no elementwise op or gather has a price in
it. A record states the switch (`tf32_on`), and one that states it on while a GEMM row of its log reads float32 is
priced at neither (`exact_under_tf32`, which the grade refuses by).

A record whose op log names no dtype is priced by `ASSUMED`: GEMMs and forward or backward elementwise ops in bf16
(the lf model trains under bf16 autocast), the optimizer and linear algebra in f32, a gather's words at 16 bits;
`of_log` lists every assumption it made. The ops training_slowdown.flops doesn't count (exp, max, comparisons, casts)
are in no record, so |C| here is the counted ops' gates.
"""
from __future__ import annotations

import argparse
import json
import sys

from .opnames import GATHER_NAMES, LINALG_NAMES, gemm_like

#: (module, AND, XOR, NOT) per Definition, from `compile(D).counts`; the AND count is circuit-check's pin. A module is
#: `verity`'s, its sidecar `<module>.circuit-check.json` beside it
DEFINITIONS = {
    "HopperBF16WgmmaDot16_v2": ("definitions.ops.boolean.gemm", 7872, 15049, 761),
    "BlackwellE4m3QmmaDot32_v2": ("definitions.ops.boolean.gemm", 9128, 17402, 1025),
    "BlackwellE5m2QmmaDot32_v2": ("definitions.ops.boolean.gemm", 8931, 17231, 1251),
    "BlackwellNvf4OmmaDot64_v2": ("definitions.ops.boolean.gemm", 6864, 13949, 616),
    "HopperTf32WgmmaDot8_v2": ("definitions.registry.boolean_mhc", 6524, 12120, 439),
    "F32Fma_v4": ("definitions.ops.boolean.elementwise", 4485, 7863, 289),
    "Bf16MulBf16_v2": ("definitions.ops.boolean.elementwise", 799, 1155, 99),
    "Bf16Add_v2": ("definitions.ops.boolean.elementwise", 717, 1061, 116),
}
#: the Python names of `DEFINITIONS` in their modules
PYTHON_NAMES = {"HopperBF16WgmmaDot16_v2": "HopperBF16WgmmaDot16", "BlackwellE4m3QmmaDot32_v2": "BlackwellE4m3QmmaDot32",
                "BlackwellE5m2QmmaDot32_v2": "BlackwellE5m2QmmaDot32", "BlackwellNvf4OmmaDot64_v2": "BlackwellNvf4OmmaDot64",
                "HopperTf32WgmmaDot8_v2": "HopperTf32WgmmaDot8V2", "F32Fma_v4": "F32FmaV4", "Bf16MulBf16_v2": "Bf16MulBf16",
                "Bf16Add_v2": "Bf16Add"}
MUX_GATES_PER_BIT = 3


def gates(name: str) -> int:
    return sum(DEFINITIONS[name][1:])


#: kind -> ((Definition, divisor), ...): a counted MAC of that kind costs the sum of each Definition's gates over its
#: divisor (a dot product's K)
PRICES = {
    "gemm/bf16": (("HopperBF16WgmmaDot16_v2", 16),),
    "gemm/f16": (("HopperBF16WgmmaDot16_v2", 16),),
    "gemm/e4m3": (("BlackwellE4m3QmmaDot32_v2", 32),),
    "gemm/e5m2": (("BlackwellE5m2QmmaDot32_v2", 32),),
    "gemm/fp4": (("BlackwellNvf4OmmaDot64_v2", 64),),
    "gemm/f32": (("F32Fma_v4", 1),),
    "gemm/tf32": (("HopperTf32WgmmaDot8_v2", 8),),
    "elementwise/bf16": (("Bf16MulBf16_v2", 1), ("Bf16Add_v2", 1)),
    "elementwise/f16": (("Bf16MulBf16_v2", 1), ("Bf16Add_v2", 1)),
    "elementwise/f32": (("F32Fma_v4", 1),),
}
#: a row's dtype as `PRICES` names it; `tf32` is no torch dtype: it names an f32 GEMM the run computed under TF32
DTYPES = {"bfloat16": "bf16", "bf16": "bf16", "float16": "f16", "f16": "f16", "float32": "f32", "f32": "f32",
          "tf32": "tf32", "float8_e4m3fn": "e4m3", "float8_e5m2": "e5m2", "float4_e2m1fn_x2": "fp4"}
#: what a record without dtypes is priced as (module docstring)
ASSUMED = {"gemm": "bf16", "elementwise": "bf16", "optimizer": "f32", "linalg": "f32"}
#: the bits of the words a gather without a dtype is priced as moving
ASSUMED_BITS = 16
OPTIMIZER_PHASES = {"other", "optimizer"}
#: a word's bits by dtype: what a gather moves, a multiplexer a bit at any width
BITS = {"bool": 1, "int8": 8, "uint8": 8, "float8_e4m3fn": 8, "float8_e5m2": 8, "int16": 16, "float16": 16,
        "bfloat16": 16, "int32": 32, "float32": 32, "int64": 64, "float64": 64}


class Unpriced(ValueError):
    """An op in a dtype no Definition prices."""


def per_mac(kind: str, width: int = ASSUMED_BITS) -> float:
    """The gates of one counted MAC of `kind` (`PRICES`, or `gather` with a word of `width` bits)."""
    if kind == "gather":
        return MUX_GATES_PER_BIT * width
    return sum(gates(d) / k for d, k in PRICES[kind])


def per_and(kind: str) -> float:
    """A counted MAC of `kind`'s gates (AND + XOR + NOT) over its AND gates alone."""
    return per_mac(kind) / sum(DEFINITIONS[d][1] / k for d, k in PRICES[kind])


#: the unit of every gate count
UNIT = f"gates: AND + XOR + NOT, {per_and('gemm/bf16'):.1f}x a bf16 GEMM's AND count"


def kind_of(phase: str, role: str, op: str, dtype: str | None = None) -> tuple[str, str]:
    """(kind, assumption) of a counted op. With its `dtype` (the name of the one it ran in), its kind is a GEMM's or
    an elementwise MAC's in that dtype, whatever its phase and role, and the assumption is ""; `Unpriced` where no
    Definition prices that. Without one its dtype is `ASSUMED`'s for its class by training_slowdown.flops's name sets, and
    the assumption names it. A gather's kind is `gather`, whose width is `gather_bits`'."""
    base = op.split(".")[0]
    if base in GATHER_NAMES:
        return "gather", ""
    if base in LINALG_NAMES:
        cls = "linalg"
    elif gemm_like(base):
        cls = "gemm"
    elif phase in OPTIMIZER_PHASES or role == "param":
        cls = "optimizer"
    else:
        cls = "elementwise"
    if base == "_scaled_mm" and dtype is None:
        dtype = "float8_e4m3fn"
    group = "gemm" if cls == "gemm" else "elementwise"
    if dtype is None:
        return f"{group}/{ASSUMED[cls]}", f"{cls} in {ASSUMED[cls]} (no dtype)"
    kind = f"{group}/{DTYPES.get(dtype, dtype)}"
    if kind not in PRICES:
        have = ", ".join(sorted(d for d, short in DTYPES.items() if d != short and f"{group}/{short}" in PRICES))
        raise Unpriced(f"{op} in {dtype}: no Definition prices {'a GEMM' if group == 'gemm' else 'an elementwise'} MAC "
                       f"in {dtype}, and none is assumed for it. The priced dtypes are {have}: run the op in one, or "
                       f"give {dtype} its Definition in training_slowdown.gates' PRICES")
    return kind, ""


def gather_bits(op: str, dtype: str | None = None) -> tuple[int, str]:
    """(bits, assumption) of the words a gather moves: its `dtype`'s (`BITS`; `Unpriced` for one it doesn't hold), or
    `ASSUMED_BITS` without one, an assumption it names."""
    if dtype is None:
        return ASSUMED_BITS, f"gather of {ASSUMED_BITS}-bit words (no dtype)"
    if dtype not in BITS:
        raise Unpriced(f"{op} in {dtype}: a gather's words are priced at their dtype's bits, and {dtype} has none in "
                       f"training_slowdown.gates' BITS ({', '.join(BITS)})")
    return BITS[dtype], ""


def tf32_on(record: dict) -> bool | None:
    """Whether the record's job ran with torch's switch for TF32 matrix products on, by the two readings of it the
    record states (`tf32`, which `run.Harness.finish` writes): on where either says so. None for a record that
    states neither, one written before a record stated them."""
    stated = record.get("tf32")
    if stated is None:
        return None
    return bool(stated["cuda_matmul_allow_tf32"]) or stated["float32_matmul_precision"] != "highest"


def exact_under_tf32(record: dict) -> list[str]:
    """The GEMM ops, by phase and name, that the record's op log names float32, the exact product, while the record
    states the TF32 switch on: none where it states it off or states nothing."""
    if not tf32_on(record):
        return []
    return sorted({f"{row[0]} {row[2]}" for s in record["structures"].values() for row in s.get("log", [])
                   if row_dtype(row) == "float32" and kind_of(*row[:3], "float32")[0] == "gemm/f32"})


def row_dtype(row: list) -> str | None:
    """The dtype an op log's row names: a row is [phase, role, op, shapes, count, MACs, dtype] as
    `flops.Counter.record` writes it, or [phase, role, op, shapes, count, MACs], one written before rows named a
    dtype, for which it is None. Every reader of a row reads it here, so a row of another length, or one whose dtype
    is not a dtype's name, is refused wherever it is read and is never read as a row of the other kind."""
    if len(row) == 6:
        return None
    if len(row) == 7 and isinstance(row[6], str) and row[6] and " " not in row[6]:
        return row[6]
    named = f" and names the dtype {row[6]!r}" if len(row) == 7 else ""
    raise ValueError(f"an op log's row is [phase, role, op, shapes, count, MACs] or that and the dtype its op "
                     f"computed in; this one has {len(row)} fields{named}: {row!r}")


def op_key(row: list) -> str:
    """An op log's row (`row_dtype`) as its `macs_log` key, "phase role op shapes [dtype]" (`split_key` reads it
    back)."""
    dtype = row_dtype(row)
    return " ".join(map(str, [*row[:4], *([dtype] if dtype else [])]))


def split_key(key: str) -> tuple[str, str, str, str, str | None]:
    """A `macs_log` op key "phase role op shapes [dtype]" as (phase, role, op, shapes, dtype), dtype None where the
    key has none."""
    phase, role, op, *rest = key.split(" ", 3)
    rest = rest[0] if rest else ""
    last = rest.rsplit(" ", 1)[-1]
    if last and not last.endswith("]") and not last.endswith("}"):
        return phase, role, op, rest[:len(rest) - len(last)].rstrip(), last
    return phase, role, op, rest, None


def of_log(log: dict) -> dict:
    """A record's `macs_log` (`by` class -> MACs, `ops` "phase role op shapes [dtype]" -> [count, MACs]) in gates: the
    total, the MACs and gates per kind, and the assumptions made, each with its MACs (none where every op row names
    its dtype). MACs `by` counts that no op row holds are priced as elementwise bf16, an assumption it names."""
    kinds: dict[str, list[float]] = {}
    notes: dict[str, int] = {}
    listed = 0
    for key, (_, macs) in log.get("ops", {}).items():
        phase, role, op, _, dtype = split_key(key)
        kind, note = kind_of(phase, role, op, dtype)
        width = ASSUMED_BITS
        if kind == "gather":
            width, note = gather_bits(op, dtype)
        k = kinds.setdefault(kind, [0, 0.0])
        k[0] += macs
        k[1] += macs * per_mac(kind, width)
        listed += macs
        if note:
            notes[note] = notes.get(note, 0) + macs
    rest = sum(log.get("by", {}).values()) - listed
    if rest > 0:
        k = kinds.setdefault("elementwise/bf16", [0, 0.0])
        k[0] += rest
        k[1] += rest * per_mac("elementwise/bf16")
        notes["MACs in no op row, as elementwise bf16"] = rest
    return {"gates": sum(v[1] for v in kinds.values()), "macs": sum(v[0] for v in kinds.values()),
            "kinds": {k: {"macs": v[0], "gates": v[1]} for k, v in sorted(kinds.items())}, "assumed": notes}


def table() -> list[dict]:
    rows = [{"kind": k, "gates_per_mac": per_mac(k), "from": " + ".join(f"{d} ({gates(d)} gates)" + (f" / {n}" if n > 1
            else "") for d, n in v)} for k, v in PRICES.items()]
    return rows + [{"kind": "gather", "gates_per_mac": per_mac("gather"),
                    "from": f"one 16-bit word multiplexer, {MUX_GATES_PER_BIT} gates per bit"}]


def recompute() -> dict[str, tuple[int, int, int]]:
    """Each Definition's (AND, XOR, NOT) from verity's circuits."""
    import importlib

    from verity.circuits.evaluation.bits import compile as compile_bits

    out = {}
    for name, (module, *_) in DEFINITIONS.items():
        mod = importlib.import_module(f"verity.{module}")
        c = compile_bits(getattr(mod, PYTHON_NAMES[name])).counts
        out[name] = (c.get("and", 0), c.get("xor", 0), c.get("not", 0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="recompute every Definition's counts from verity's circuits")
    a = ap.parse_args(argv)
    if a.check:
        got = recompute()
        bad = {n: {"table": list(DEFINITIONS[n][1:]), "circuits": list(c)} for n, c in got.items()
               if tuple(DEFINITIONS[n][1:]) != c}
        print(json.dumps(bad or "every Definition's counts match", indent=1))
        return 1 if bad else 0
    for r in table():
        print(f"{r['kind']:18} {r['gates_per_mac']:12.3f}  {r['from']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
