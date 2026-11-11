"""Boolean gates per counted operation: the conversion from the harness's multiply-accumulates (MACs) to the gates
the training formalism counts |C|, c and w in.

    python -m training_slowdown.gates           # the table
    python -m training_slowdown.gates --check   # recompute it from the Definitions (needs verity), exit 1 on a difference

Each price is one of the circuits' own Definitions (`verity.definitions.ops.boolean`), at its whole gate count:
AND + XOR + NOT as `verity.circuits.evaluation.bits.compile(D).counts` gives them, the AND count being the one its
`<module>.circuit-check.json` pins (`test_gates` holds both). A counted MAC is one multiply and one accumulate:

- a GEMM MAC is a tensor-core dot product's gates over its K: bf16 (and f16, which has no Definition) at
  `HopperBF16WgmmaDot16_v2` / 16, the 26-bit-adder model sm_120's BF16 pipeline shares (`ml.circuits.targets.target`);
  FP8 at `BlackwellE4m3QmmaDot32_v2` / 32 or `BlackwellE5m2QmmaDot32_v2` / 32, NVFP4 at `BlackwellNvf4OmmaDot64_v2` /
  64, f32 at `F32Fma_v4`;
- an elementwise MAC (training_slowdown.flops's `mul`, `div`, `addcmul`, a scaled `add`, a counted forward op) is a bf16 multiply
  then a bf16 add (`Bf16MulBf16_v2` + `Bf16Add_v2`), or one `F32Fma_v4` in f32;
- an optimizer's MAC (phase `other`/`optimizer`, role `param`) and dense linear algebra's are f32 (`F32Fma_v4`);
- a gather's MAC (training_slowdown.flops's `gather_macs`) is one word multiplexer, y ⊕ (b ∧ (x ⊕ y)) per bit: 3 gates per bit.

A record names each op's dtype from training_slowdown 4.0's capture on. A record without one (v2.3's `macs_log`) is priced by
`ASSUMED`: GEMMs and forward or backward elementwise ops in bf16 (the lf model trains under bf16 autocast), the
optimizer in f32; `of_log` lists every assumption it made. The ops training_slowdown.flops doesn't count (exp, max, comparisons, casts)
are in no record, so |C| here is the counted ops' gates.
"""
from __future__ import annotations

import argparse
import json
import sys

from .opnames import GATHER_NAMES, LINALG_NAMES, gemm_like

#: (module, AND, XOR, NOT) per Definition, from `compile(D).counts`; the AND count is circuit-check's pin
DEFINITIONS = {
    "HopperBF16WgmmaDot16_v2": ("gemm", 7872, 15049, 761),
    "BlackwellE4m3QmmaDot32_v2": ("gemm", 9128, 17402, 1025),
    "BlackwellE5m2QmmaDot32_v2": ("gemm", 8931, 17231, 1251),
    "BlackwellNvf4OmmaDot64_v2": ("gemm", 6864, 13949, 616),
    "F32Fma_v4": ("elementwise", 4485, 7863, 289),
    "Bf16MulBf16_v2": ("elementwise", 799, 1155, 99),
    "Bf16Add_v2": ("elementwise", 717, 1061, 116),
}
#: the Python names of `DEFINITIONS` in their modules
PYTHON_NAMES = {"HopperBF16WgmmaDot16_v2": "HopperBF16WgmmaDot16", "BlackwellE4m3QmmaDot32_v2": "BlackwellE4m3QmmaDot32",
                "BlackwellE5m2QmmaDot32_v2": "BlackwellE5m2QmmaDot32", "BlackwellNvf4OmmaDot64_v2": "BlackwellNvf4OmmaDot64",
                "F32Fma_v4": "F32FmaV4", "Bf16MulBf16_v2": "Bf16MulBf16", "Bf16Add_v2": "Bf16Add"}
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
    "elementwise/bf16": (("Bf16MulBf16_v2", 1), ("Bf16Add_v2", 1)),
    "elementwise/f16": (("Bf16MulBf16_v2", 1), ("Bf16Add_v2", 1)),
    "elementwise/f32": (("F32Fma_v4", 1),),
}
DTYPES = {"bfloat16": "bf16", "bf16": "bf16", "float16": "f16", "f16": "f16", "float32": "f32", "f32": "f32",
          "float8_e4m3fn": "e4m3", "float8_e5m2": "e5m2", "float4_e2m1fn_x2": "fp4"}
#: what a record without dtypes is priced as (module docstring)
ASSUMED = {"gemm": "bf16", "elementwise": "bf16", "optimizer": "f32", "linalg": "f32"}
OPTIMIZER_PHASES = {"other", "optimizer"}


def per_mac(kind: str, width: int = 16) -> float:
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
    """(kind, assumption) of a counted op: its class by training_slowdown.flops's name sets, its dtype the record's or `ASSUMED`'s
    (the assumption names it; "" when the record gave the dtype)."""
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
    dt = DTYPES.get(dtype, dtype) if dtype else None
    note = "" if dt else f"{cls} in {ASSUMED[cls]} (no dtype)"
    dt = dt or ASSUMED[cls]
    group = "gemm" if cls == "gemm" else "elementwise"
    if f"{group}/{dt}" not in PRICES:
        note = f"{cls} in {dt}: no Definition, priced as bf16"
        dt = "bf16"
    return f"{group}/{dt}", note


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
    """A record's `macs_log` (`by` class -> MACs, `ops` "phase role op shapes [dtype]" -> [count, MACs]) in gates: the total, the MACs and gates per kind, and the assumptions made. MACs `by` counts that no op row holds are
    priced as elementwise bf16, an assumption it names."""
    kinds: dict[str, list[float]] = {}
    notes: dict[str, int] = {}
    listed = 0
    for key, (_, macs) in log.get("ops", {}).items():
        phase, role, op, _, dtype = split_key(key)
        kind, note = kind_of(phase, role, op, dtype)
        k = kinds.setdefault(kind, [0, 0.0])
        k[0] += macs
        k[1] += macs * per_mac(kind)
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
        mod = importlib.import_module(f"verity.definitions.ops.boolean.{module}")
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
        print(f"{r['kind']:18} {r['gates_per_mac']:10.1f}  {r['from']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
