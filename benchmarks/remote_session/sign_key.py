"""rows.sh's check, with no card, of the sign key and rows a built `session.cu` derives on the device: its chained
pass's messages (`rs_chain_template`, rs_chain_set's on the host) at a seed of 64 zero bytes, hashed here as
`chain_prepare_kernel` hashes them, against VECTOR at 64 rows (the wide key: K, row 0's first 16 bytes and row 63's
last 16) and against `verity.commitments.randomness`'s v2 key and rows at 16. Stdlib only, so the build's machine runs
it as it is; loading the library calls no CUDA function.

  python3 sign_key.py SESSION_SO

Prints one line a row count; exits 1 naming each of K, row 0 and row 63 (at 16 rows: the key or a row) that differs.
"""
from __future__ import annotations

import ctypes
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verity.commitments import randomness

CONTEXT = {"k": 64, "n": 4096, "tag": "sketch-sha512"}
#: the 64-row sign key and two of its rows at a source of 64 zero bytes, `sketch-signs/v1` under the wide derive
VECTOR = {"K": "b844824fb8704537ad7e865cc186b22f116ef99a0e48d8034d76cc07cf9ab10b"
               "d3e8ca87fb456891833fe7d994a5141c4b52bf3446f163a1a954024dfa5e8583",
          "row 0": "3f2ad9ff6856261e62ea25da12b64840", "row 63": "3333608eddf01f1e530f7d92d002db91"}
ROW_BYTES = 512


def template(lib: ctypes.CDLL, rows: int) -> tuple[bytes, list[int], bytes, list[int]]:
    """rs_chain_template's messages at `rows`: the derive message and (its length, the seed's offset), a row's head
    and (the row message's length, the key's offset, the row byte's, the counter's)."""
    if lib.rs_rows(rows):
        raise SystemExit(f"rs_rows({rows}) refused")
    derive, row = ctypes.create_string_buffer(256), ctypes.create_string_buffer(128)
    dn, rn = (ctypes.c_int * 2)(), (ctypes.c_int * 4)()
    if lib.rs_chain_template(bytes(32), derive, dn, row, rn):
        raise SystemExit(f"rs_chain_template at {rows} rows refused")
    return derive.raw[:dn[0]], list(dn), row.raw[:rn[3]], list(rn)


def derived(derive: bytes, dn: list[int], head: bytes, rn: list[int], rows: int, seed: bytes = bytes(64)
            ) -> tuple[bytes, list[bytes]]:
    """The key and the rows `chain_prepare_kernel` makes from these messages at pass seed `seed`."""
    m = bytearray(derive)
    m[dn[1]:dn[1] + 64] = seed
    key = hashlib.sha512(bytes(m)).digest()[:64 if rows == 64 else 32]
    out = []
    for c in range(rows):
        h = bytearray(head)
        h[rn[1]:rn[1] + len(key)] = key
        h[rn[2]] = c
        out.append(b"".join(hashlib.sha512(bytes(h) + k.to_bytes(8, "big")).digest()
                            for k in range(ROW_BYTES // 64)))
    if rn[0] != rn[3] + 8:
        raise SystemExit(f"a row's message is its head and an 8-byte counter, not {rn}")
    return key, out


def differs(key: bytes, rows: list[bytes], k: int) -> list[str]:
    """What of the device's key and rows at `k` rows is not the reference's."""
    if k == 64:
        got = {"K": key.hex(), "row 0": rows[0][:16].hex(), "row 63": rows[63][-16:].hex()}
        return [name for name in VECTOR if got[name] != VECTOR[name]]
    want = randomness.derive(bytes(64), "verity/remote-bandwidth/sketch-signs/v0", {**CONTEXT, "k": k})
    return (["the key"] if key != want.value else []) + [
        f"row {c}" for c in range(k) if rows[c] != want.stream(("row", c), ROW_BYTES)]


def main(argv: list[str] | None = None) -> int:
    lib = ctypes.CDLL(str(Path((argv or sys.argv[1:])[0]).resolve()))
    bad = False
    for k in (64, 16):
        wrong = differs(*derived(*template(lib, k), k), k)
        print(f"sign key at {k} rows: " + (f"differs in {', '.join(wrong)}" if wrong else
                                           "the vector's" if k == 64 else "randomness v2's"))
        bad = bad or bool(wrong)
    lib.rs_rows(16)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
