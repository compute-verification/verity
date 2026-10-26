"""Conformance vectors for ``pearl-fp8-v4``, computed by Pearl's own Rust at pearl-research-labs/pearl ``284b147b``.

The inputs are derived here (SHAKE256 of fixed labels), so the pinned file ``tests/vectors/pearl_kw.json`` holds only
Pearl's outputs (large arrays as SHA-256 of their bytes).  To regenerate, with a checkout of that commit:

    python verity/core/protocols/pouw/tests/pearl_vectors.py inputs /tmp/in.json
    # add `mod verity_vectors;` and the generator module (see the pinned file's "generator") to zk-pow/src/api/fp8, then
    VERITY_VEC_IN=/tmp/in.json VERITY_VEC_OUT=/tmp/out.json cargo test --release --lib verity_vectors
    python verity/core/protocols/pouw/tests/pearl_vectors.py pin /tmp/out.json

`outputs()` computes the same outputs with `verity.core.protocols.pouw.schemes.pearl_kw`; the test compares them to the pinned file.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

PINNED = Path(__file__).with_name("vectors") / "pearl_kw.json"
PEARL_COMMIT = "284b147bf56bcc7e38c9f0bb87a56d61e9b7519d"
K = 1024


def stream(label: str, n: int) -> bytes:
    return hashlib.shake_256(f"verity/pouw/pearl-vectors/v1\0{label}".encode()).digest(n)


def words(label: str, n: int) -> list[int]:
    b = stream(label, 4 * n)
    return [int.from_bytes(b[4 * i:4 * i + 4], "little") for i in range(n)]


def bf16_of(x: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def fp8_codes(label: str, n: int) -> list[int]:
    return [c if c & 0x7F != 0x7F else c ^ 1 for c in stream(label, n)]


def side(label: str, rows: int, side_id: int, seed: bytes, f_seed: bytes, first: int, spiky: bool = False) -> dict:
    raw = stream(f"{label}/ints", rows * K)
    w = words(f"{label}/scales", rows * K // 8)
    if spiky:                                   # 60 spikes of 127 per row at one scale, zeros elsewhere: 5.9% dead > 1/64
        ints = [127 if j % 17 == 0 and j < 17 * 60 else 0 for _ in range(rows) for j in range(K)]
        scales = [0x3E00] * len(w)
    else:
        ints = [(b % 255) - 127 for b in raw]
        scales = [bf16_of(2.0 ** -(3 + x % 6) * (1 + (x >> 8) % 128 / 128)) for x in w]
    return {"rows": rows, "side": side_id, "seed": seed.hex(), "f_seed": f_seed.hex(), "lines": list(range(first, first + rows)),
            "ints": bytes(i & 0xFF for i in ints).hex(), "scales": struct.pack(f"<{len(scales)}H", *scales).hex()}


def inputs() -> dict:
    seeds = [stream(f"seed/{i}", 32) for i in range(4)]
    ops = []
    for i, x in enumerate(words("bf16", 3 * 40)):
        ops.append(x & 0x7FFF)
    ops = [[bf16_of(1 + (a % 1000) / 7.0) ^ (0x8000 * (a & 1)), bf16_of(0.01 + (b % 997) / 101.0), bf16_of((c % 1001) / 3.0 - 150)]
           for a, b, c in zip(ops[0::3], ops[1::3], ops[2::3])]
    ops += [[0x3F80, 0x4000, 0x3F80], [0x4315, 0x3F80, 0x3B95], [0x0001, 0x3F80, 0x8001], [0x7F7F, 0x3F00, 0x7F7F]]
    census = [{"device": (a >> 20) & 1, "alpha": 0x3F80 | (a % 0x7F), "x": (x % 0x7F00) | (0x8000 * (x & 1)), "l2": 0x3C00 + l % 0x800,
               "z": 2.0 ** ((z % 60) - 30), "m": 2.0 ** ((m % 60) - 30) * 1.5, "lb": 20000 + lb % 30000}
              for a, x, l, z, m, lb in zip(*(iter(words("census", 6 * 64)),) * 6)]
    matmul = [{"m": 2, "n": 3, "k": 64, "a": bytes(fp8_codes("mm/a", 128)).hex(), "b": bytes(fp8_codes("mm/b", 192)).hex()},
              {"m": 1, "n": 2, "k": 40, "a": bytes(fp8_codes("mm2/a", 40)).hex(), "b": bytes(fp8_codes("mm2/b", 80)).hex()}]
    tiles = [{"k": K, "seed_a": seeds[0].hex(), "a": side("t0/a", 16, 0, seeds[0], seeds[1], 32), "b": side("t0/b", 16, 1, seeds[1], seeds[1], 48)},
             {"k": K, "seed_a": seeds[2].hex(), "a": side("t1/a", 16, 0, seeds[2], seeds[3], 0, spiky=True), "b": side("t1/b", 16, 1, seeds[3], seeds[3], 16)}]
    return {"hash_labelled": [{"msg": stream("hl/0", 76).hex(), "label": "key-A", "key": None},
                              {"msg": stream("hl/1", 97).hex(), "label": "seed-B", "key": None},
                              {"msg": stream("hl/2", 64).hex(), "label": "jackpot", "key": seeds[0].hex()}],
            "lines": [{"seed": seeds[i % 4].hex(), "side": i % 2, "factor": (i // 2) % 2, "line": [0, 1, 7, 1023, 65535, 2 ** 24 + 3][i % 6]}
                      for i in range(12)],
            "bf16_ops": ops, "census": census, "matmul": matmul, "tiles": tiles}


def sha(hex_str: str) -> str:
    return hashlib.sha256(bytes.fromhex(hex_str)).hexdigest()


def summarize_side(s: dict) -> dict:
    return {"norms": s["norms"], "e_sha256": sha(s["e"]), "f_sha": s["f_sha"], "codes_sha256": sha(s["codes"]),
            "alpha": s["alpha"], "beta": s["beta"], "l2": s["l2"]}


def summarize(out: dict) -> dict:
    return {"hash_labelled": out["hash_labelled"], "lines": out["lines"], "bf16_ops": out["bf16_ops"], "census": out["census"],
            "matmul": out["matmul"], "lanes": out["lanes"], "pattern": out["pattern"], "budget": out["budget"],
            **{key: [{"a": summarize_side(t["a"]), "b": summarize_side(t["b"]), "acc": t["acc"], "msg": t["msg"],
                      "jackpot": t["jackpot"], "approved": t["approved"], "policy_msg": t["policy_msg"]} for t in out[key]]
               for key in ("tiles", "tiles_h100")}}


if __name__ == "__main__":
    cmd, path = sys.argv[1], Path(sys.argv[2])
    if cmd == "inputs":
        path.write_text(json.dumps(inputs()))
    elif cmd == "pin":
        rust = json.loads(path.read_text())
        PINNED.write_text(json.dumps({"schema": "verity-pouw/pearl-kw-vectors/v1", "pearl_commit": PEARL_COMMIT,
                                      "generator": "zk-pow/src/api/fp8/verity_vectors.rs (its copy: tests/pearl_vectors.rs)",
                                      "devices": ["b200", "h100"], "outputs": summarize(rust)}, indent=1) + "\n")
        print(f"wrote {PINNED}")
