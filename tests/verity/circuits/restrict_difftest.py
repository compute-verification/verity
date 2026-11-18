"""`verity.circuits.restrict`'s difftest data: blocks of registered circuits, each `compact (restrict C S)` unrolled to its
gates, as a test's reference for a layout function (the gate list and the map are a function's value, not the object).

    python tests/verity/circuits/restrict_difftest.py --out DIR

Writes `DIR/restrict_difftest/`: `C/<sha256>.json`, each circuit's descriptor bytes; `block/<sha256>.txt`, each block
circuit's text; and two files, one line a case. `cases.jsonl`, `{"name", "c", "partition", "units"}`: the circuit by its
bytes' sha256, the partition object, and the block's units as ascending `[lo, hi)` ranges. `python.jsonl`, `{"name",
"bits", "words", "inputs", "gates", "outputs", "ports", "sha256", "size"}`: C on bits' gate count, the block's gates of C
as `[lo, hi)` ranges, the file's numbering into C on bits (its input ports, its gates as `[lo, hi)` ranges, its output
ports), each port as `[gate of C, j]` (bit j of that gate's value), and the text's sha256 and size.

Cases: the periods' chain (eight `Gemm_v2{K=16,N=16}` units, one wire between consecutive ones, each weight a registered
read), each window member of two units (`[2i, 2i + 2)`) and each unit alone.
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
        "one_circuit_chain", ROOT / "tests/verity/protocols/proofs/spot/verifier/test_one_circuit_chain.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    return M.registered()


def chain_cases() -> list[tuple[str, bytes, dict, list[tuple[int, int]]]]:
    r = _chain()
    c, part = r["C"], dict(r["partition"])
    return ([(f"chain/member{i}", c, part, [(2 * i, 2 * i + 2)]) for i in range(4)]
            + [(f"chain/unit{u}", c, part, [(u, u + 1)]) for u in range(8)])


def vectors(cases, out: Path) -> None:
    d = out / "restrict_difftest"
    (d / "C").mkdir(parents=True, exist_ok=True)
    (d / "block").mkdir(parents=True, exist_ok=True)
    bits: dict[str, R.Bits] = {}
    with open(d / "cases.jsonl", "w") as fc, open(d / "python.jsonl", "w") as fp:
        for name, c, part, units in cases:
            ch = hashlib.sha256(c).hexdigest()
            if ch not in bits:
                (d / "C" / f"{ch}.json").write_bytes(c)
                bits[ch] = R.Bits(R.program(c))
            B = bits[ch]
            fc.write(json.dumps({"name": name, "c": ch, "partition": part, "units": units}, sort_keys=True) + "\n")
            try:
                b = R.restrict(c, part, units, B)
            except R.Refused as e:
                fp.write(json.dumps({"name": name, "refused": e.code}, sort_keys=True) + "\n")
                continue
            h = hashlib.sha256(b.text).hexdigest()
            (d / "block" / f"{h}.txt").write_bytes(b.text)

            def port(x: int) -> list[int]:
                g = B.gate_of(x)
                return [g, B.out[g].index(x)]

            fp.write(json.dumps({"name": name, "bits": B.size, "words": R.unit_gates(B.D, part, units),
                                 "inputs": list(b.inputs), "gates": [list(g) for g in b.gates], "outputs": list(b.outputs),
                                 "ports": {"inputs": [port(x) for x in b.inputs], "outputs": [port(x) for x in b.outputs]},
                                 "sha256": h, "size": len(b.text)}, sort_keys=True) + "\n")
            print(name, h, len(b.text), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    vectors(chain_cases(), a.out)


if __name__ == "__main__":
    main()
