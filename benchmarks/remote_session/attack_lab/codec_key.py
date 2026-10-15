"""The codec key a probe hands `kv.cu`, and the check that what `kv.cu` wrote is the registered codec's store.

    python3 codec_key.py key OUT.bin            c_0 ‖ c_1 ‖ c_2 ‖ τ ‖ K, 128 bytes each little-endian, p1024/k3/masked
    python3 codec_key.py check DUMP.bin         DUMP is records of (u64le j, payload 4096, encoded 4096): every encoded
                                                block must equal `codec.encode(payload, j, key)` of the repository

The seed is fixed (bytes 0 … 31): a compression probe needs a real store, not a secret one. The codec is loaded from its
file so no other module of the package (and nothing outside the standard library) is imported.
"""
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

SEED = bytes(range(32))
ROOT = Path(__file__).resolve().parents[3]


def codec():
    path = ROOT / "verity/core/protocols/remote_bandwidth/codec.py"
    spec = importlib.util.spec_from_file_location("rb_codec", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rb_codec"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    c = codec()
    k = c.key(SEED, c.DEFAULT)
    s = c.DEFAULT.field.word_bytes
    if sys.argv[1] == "key":
        Path(sys.argv[2]).write_bytes(b"".join(x.to_bytes(s, "little") for x in (*k.c, k.tau, k.mask)))
        return 0
    data = Path(sys.argv[2]).read_bytes()
    rec, bad = 8 + 2 * c.BLOCK_BYTES, 0
    for o in range(0, len(data), rec):
        (j,) = struct.unpack_from("<Q", data, o)
        payload, enc = data[o + 8:o + 8 + c.BLOCK_BYTES], data[o + 8 + c.BLOCK_BYTES:o + rec]
        ok = c.encode(payload, j, k) == enc
        bad += not ok
        print(f'{{"check":"codec","j":{j},"byte_exact":{str(ok).lower()}}}')
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
