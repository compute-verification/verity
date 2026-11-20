"""`verity.circuits.restrict`'s difftest data: blocks of registered circuits, each `compact (restrict C S)` as the block's
Program (the object, compared byte for byte) and, as a test's reference for a layout function, that Program unrolled on bits
(the listing and the map are a function's value, not the object).

    python tests/verity/circuits/restrict_difftest.py --out DIR

Writes `DIR/restrict_difftest/`: `C/<sha256>.json`, each circuit's descriptor bytes; `block/<sha256>.json`, each block's
descriptor bytes; `listing/<sha256>.txt`, each block's listing; and two files, one line a case. `cases.jsonl`, `{"name",
"c", "partition", "units"}`: the circuit by its bytes' sha256, the partition object, and the block's units as ascending
`[lo, hi)` ranges. `python.jsonl`, `{"name", "descriptor", "size", "words", "inputs", "outputs", "listing", "bits",
"input_bits", "gates", "output_bits"}`: the block's descriptor's sha256 and size; in C's word-level descriptor gate index,
the one that names things (`restrict`'s module docstring), S as gate ranges (`words`) and in(S) and out(S) as gates
(`inputs`, `outputs`), so the block's descriptor gate i is C's `(inputs + S)[i]` (`Block.to_c_gate`); and, on bits, for a
statement and a count only, the listing's sha256, C on bits' gate count, and the map from the block on bits to C on bits (its
input bits, its gates as `[lo, hi)` ranges, its output bits: `Block.to_c_bit`).

Cases: the periods' chain (eight `Gemm_v2{K=16,N=16}` units, one wire between consecutive ones, each weight a registered
read), each window member of two units (`[2i, 2i + 2)`) and each unit alone; then `test_restrict`'s toys (`TOYS`), on
one-bit templates: two readers of one writer, a registered value, a wire inside a two-unit block, and a root batch taken
whole, in part and one member.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from verity.circuits import restrict as R

ROOT = Path(__file__).resolve().parents[3]


def _chain() -> dict:
    spec = importlib.util.spec_from_file_location(
        "one_circuit_chain", ROOT / "tests/verity/protocols/proof/spot/verifier/test_one_circuit_chain.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    return M.registered()


def chain_cases() -> list[tuple[str, bytes, dict, list[tuple[int, int]]]]:
    r = _chain()
    c, part = r["C"], dict(r["partition"])
    return ([(f"chain/member{i}", c, part, [(2 * i, 2 * i + 2)]) for i in range(4)]
            + [(f"chain/unit{u}", c, part, [(u, u + 1)]) for u in range(8)])


def toy_cases() -> list[tuple[str, bytes, dict, list[tuple[int, int]]]]:
    spec = importlib.util.spec_from_file_location("restrict_toys", Path(__file__).parent / "test_restrict.py")
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    out = []
    for name, (_f, blocks) in T.TOYS.items():
        _p, c, _b = T.toy(name)
        out += [(f"toy/{name}/{'+'.join(f'{lo}-{hi}' for lo, hi in u)}", c, T.TOY_PARTITION, u) for u in blocks]
    return out


def _put(d: Path, sub: str, ext: str, data: bytes) -> str:
    h = hashlib.sha256(data).hexdigest()
    (d / sub).mkdir(parents=True, exist_ok=True)
    (d / sub / f"{h}.{ext}").write_bytes(data)
    return h


def vectors(cases, out: Path) -> None:
    d = out / "restrict_difftest"
    d.mkdir(parents=True, exist_ok=True)
    bits: dict[str, R.Bits] = {}
    with open(d / "cases.jsonl", "w") as fc, open(d / "python.jsonl", "w") as fp:
        for name, c, part, units in cases:
            ch = _put(d, "C", "json", c)
            if ch not in bits:
                bits[ch] = R.Bits(R.program(c))
            fc.write(json.dumps({"name": name, "c": ch, "partition": part, "units": units}, sort_keys=True) + "\n")
            try:
                b = R.restrict(c, part, units, bits[ch])
                listing = b.listing()
            except R.Refused as e:
                fp.write(json.dumps({"name": name, "refused": e.code}, sort_keys=True) + "\n")
                continue
            h = _put(d, "block", "json", b.descriptor)
            lh = _put(d, "listing", "txt", listing)
            fp.write(json.dumps({"name": name, "descriptor": h, "size": len(b.descriptor), "words": [list(w) for w in b.words],
                                 "inputs": list(b.inputs), "outputs": list(b.outputs), "listing": lh, "bits": b.bits.size,
                                 "input_bits": b.input_bits, "gates": [list(g) for g in b.gate_ranges],
                                 "output_bits": b.output_bits}, sort_keys=True) + "\n")
            print(name, h, len(b.descriptor), lh, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    vectors(chain_cases() + toy_cases(), ap.parse_args().out)


if __name__ == "__main__":
    main()
