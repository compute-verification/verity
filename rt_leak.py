"""REDTEAM phase 19, target 4 (#3984 at bb350a9be7): can a leak in a row's tail or in a short row escape
`benchmarks/network_traces/leak_scan.py`? Run with the PR's own harness (`tests/test_leak_scan.py`'s streams, round
messages and `a8` stand-in).

    PYTHONPATH=<worktree> python rt_leak.py

A row of 16k + t bytes (0 < t < 16) packed into the format's 16-byte elements from an element boundary puts
row[16k:] and its padding in its last element. `cut`'s tail piece, row[L - 16:], then starts 16 - t bytes before that
element, so it straddles two fields (``across``) wherever the format frames 16-byte elements: `redigest.layout` reads
every element as its own field, and `records` frames a round message's elements one per record. The row's last t bytes
are then counted only if an aligned piece is, which the circuit's zeros (`public_pieces`) or a public coincidence
(`classify`) can excuse.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "benchmarks/network_traces/tests"))
sys.path.insert(0, str(ROOT / "benchmarks/network_traces"))
import leak_scan as LS  # noqa: E402
import test_leak_scan as T  # noqa: E402


def kept_of(row: bytes) -> dict[str, bytes]:
    """`public_pieces` of one witness row, the staged circuit holding sixteen zero bytes (as in the PR's test)."""
    return LS.public_pieces(T.A8, {"wit 0": row}, {"wit 0"}, b"g 1 2\n" + bytes(16).hex().encode())[0]


def in_stream(row: bytes, at: int, pad: bytes = b"", written: bytes | None = None) -> tuple:
    with tempfile.TemporaryDirectory() as d:
        vdir = T._streams(Path(d), {at: ((row + pad) if written is None else written, b"")})
        kept = kept_of(row)
        return LS.classify(T._found(vdir, kept), kept, vdir, reads=T._reads), \
            LS.placements(T._found(vdir, kept), kept, vdir, reads=T._reads)


def in_message(row: bytes) -> tuple:
    n = -(-len(row) // 16)
    with tempfile.TemporaryDirectory() as d:
        el = bytes(range(100, 116))
        vdir = T._transcript(Path(d), (T._framed((3, 2, n, row + bytes(16 * n - len(row)))), T._framed((3, 2, 1, el))))
        kept = kept_of(row)
        return LS.classify(T._found(vdir, kept), kept, vdir), LS.placements(T._found(vdir, kept), kept, vdir)


def main() -> None:
    wit = bytes(16) + b"\x07\x00\x09\x00"
    print("pieces of the PR's tail row (16 zeros the circuit holds, then 4 witness bytes):", sorted(kept_of(wit)))
    print()
    print("1. the PR's case, the tail piece itself written at an element boundary (112):")
    print("  ", in_stream(wit, 112, written=wit[4:]))
    print("2. the row written whole from an element boundary (112), its last element zero-padded:")
    print("  ", in_stream(wit, 112, bytes(12)))
    print("   the same, its last element's padding random (the stream's own bytes):")
    print("  ", in_stream(wit, 112))
    print("3. the row as a round message's two elements (op 3, n = 2), zero-padded:")
    print("  ", in_message(wit))
    print("4. every tail length t, a row of 16 zeros then t witness bytes, written whole at 112 (zero-padded):")
    for t in range(1, 16):
        row = bytes(16) + bytes(range(1, t + 1))
        (leaked, structured), _p = in_stream(row, 112, bytes(16 - t))
        (mleaked, mstructured), _q = in_message(row)
        print(f"   t = {t:2}: stream leaked {leaked or '-'} structured {structured or '-'}; "
              f"message leaked {mleaked or '-'} structured {mstructured or '-'}")
    print("5. control: a row whose aligned piece is witness too (18 bytes, no zeros), written whole at 112:")
    srv = bytes(range(1, 19))
    print("  ", in_stream(srv, 112, bytes(14))[0], in_message(srv)[0])
    print("6. a row under 16 bytes:", end=" ")
    try:
        LS.cut({"cin 0": bytes(range(12))}, 16)
        print("cut")
    except ValueError as e:
        print("refused:", str(e)[:80])


if __name__ == "__main__":
    main()
