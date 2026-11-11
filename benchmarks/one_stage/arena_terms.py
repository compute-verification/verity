"""The circuit prover's device arena at a shape, from two trees' ``cuda/arena_bytes.h``: each phase's term (a Python mirror of
``fc_arena_bytes``, checked against the header compiled by the host's C compiler) and the arena each tree reserves, in MiB.

    python3 benchmarks/one_stage/arena_terms.py BASE_HEADER NEW_HEADER [M K_LOG]

The arena is one ``cudaMalloc`` of ``fc_arena_bytes`` (``DeviceArena::begin``), so a buffer placed inside it moves the
device's peak only where it moves the largest term."""
from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "verity" / "protocols" / "flock" / "prover" / "tests"))
from arena_layout import HB, Shape

MIB = 1 << 20
WRAPPER = '#include "arena_bytes.h"\nsize_t fc_arena_bytes_c(const FcArenaShape* s) { return fc_arena_bytes(s); }\n'


class CShape(ctypes.Structure):
    _fields_ = [("m", ctypes.c_int), ("k_log", ctypes.c_int), ("sub_log", ctypes.c_int), ("initial_k", ctypes.c_int),
                ("recursive_steps", ctypes.c_int), ("log_inv_rates", ctypes.POINTER(ctypes.c_int)),
                ("recursive_ks", ctypes.POINTER(ctypes.c_int)), ("queries", ctypes.POINTER(ctypes.c_int)),
                ("hash_bytes", ctypes.c_int), ("zk", ctypes.c_int), ("t_pad", ctypes.c_longlong), ("fold_words", ctypes.c_longlong)]


def compiled(header: Path, d: Path) -> ctypes.CDLL:
    inc = d / header.stem
    inc.mkdir()
    (inc / "arena_bytes.h").write_text(header.read_text())
    (inc / "wrap.c").write_text(WRAPPER)
    subprocess.run(["cc", "-std=c99", "-O1", "-shared", "-fPIC", f"-I{inc}", "-o", str(inc / "a.so"), str(inc / "wrap.c")],
                   check=True)
    dll = ctypes.CDLL(str(inc / "a.so"))
    dll.fc_arena_bytes_c.argtypes = [ctypes.POINTER(CShape)]
    dll.fc_arena_bytes_c.restype = ctypes.c_size_t
    return dll


def arena(dll: ctypes.CDLL, s: Shape) -> int:
    ik, rates, ks, _ = s.lig()
    r, k = (ctypes.c_int * len(rates))(*rates), (ctypes.c_int * max(len(ks), 1))(*ks)
    q = (ctypes.c_int * len(s.queries))(*s.queries)
    c = CShape(s.m, s.k_log, s.sub_log, ik, len(ks), r, k, q, HB, int(s.zk), s.t_pad if s.zk else 0, sum(1 << t for t in s.types))
    return dll.fc_arena_bytes_c(ctypes.byref(c))


def terms(s: Shape) -> dict[str, int]:
    """fc_arena_bytes' phases at s, without its margin; ``salts`` is the recursive levels' salts the Ligerito term holds."""
    al = lambda b: (b + 255) & ~255  # noqa: E731
    F, log_n = 16, s.m - 7
    ik, rates, ks, _ = s.lig()
    W, K, w = F << log_n, F << s.k_log, 1 << ik
    n, lanes = 1 << (log_n - ik + rates[0]), (w + 4 if s.zk else w)
    l0 = al(n * lanes * F) + al((2 * n - 1) * HB) + ((al(lanes * s.t_pad * F) + al((F << (log_n - ik)) * 4)) if s.zk else 0)
    commit = 3 * W + l0 + ((al(n * (w >> min(ik, 2)) * F) + al(n * 4 * F)) if s.zk else 0)
    zc_out = 1 << (s.m - 6)
    packed = log_n >= 3 and zc_out // 16 > (1 << 10)
    zc_first, zc_next = F * (zc_out // 8 if packed else zc_out), F * (zc_out // 4 if packed else zc_out // 2)
    lob = log_n - 7 if log_n > 7 else 0
    zerocheck = 3 * W + l0 + al(F << (s.m - 13)) + 2 * al(zc_first) + 2 * al(zc_next) + al(F << lob) + al(F << (log_n - lob))
    fold_words = sum(1 << t for t in s.types)
    lincheck = (2 * W + l0 + 5 * al(K) + al(F << s.sub_log) + al(F << (s.m - s.k_log)) + max(al(K), al(F * fold_words))
                + (W if K > W else 0))
    queries = s.queries
    max_block, opening, prev, rows, salts = n, 0, 0, al(queries[0] * lanes * F), 0
    max_q = max(queries)
    max_al = 1 << (max_q - 1).bit_length()
    dim = log_n - ik + 1
    for i, k in enumerate(ks):
        log_cols = dim - k
        bl, cols = 1 << (log_cols + rates[i + 1]), 1 << k
        level = al(bl * cols * F) + al((2 * bl - 1) * HB)
        opening = max(opening, prev + level + rows)
        max_block = max(max_block, bl)
        salts += al(bl * 192) if HB == 64 else 0
        prev, rows, dim = level, al(queries[i + 1] * cols * F), log_cols + 1
    opening = max(opening, prev + rows)
    ligerito = 4 * W + l0 + al(max_block * F) + al(max_al * F) + al(max_q * 8) + al(F << (log_n - ik + 1)) + opening
    return {"commit": commit, "zerocheck": zerocheck, "lincheck": lincheck, "ligerito": ligerito, "salts": salts}


def main() -> None:
    base, new = Path(sys.argv[1]), Path(sys.argv[2])
    m, k_log = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (35, 25)
    with tempfile.TemporaryDirectory() as d:
        b, n = compiled(base, Path(d)), compiled(new, Path(d))
        for zk in (True, False):
            for label, s in (("served", Shape(m, k_log, zk)), ("12 types", replace(Shape(m, k_log, zk), type_logs=(k_log - 4,) * 11 + (k_log,)))):
                t = terms(s)
                phases = {k: v for k, v in t.items() if k != "salts"}
                top = max(phases, key=phases.get)
                with_salts = t["ligerito"] + t["salts"]
                print(f"m{m} k{k_log} {'Flock ZK' if zk else 'circuit prover'} {label}: " + ", ".join(f"{k} {v / MIB:,.0f}" for k, v in phases.items())
                      + f", ligerito with salts {with_salts / MIB:,.0f} (salts {t['salts'] / MIB:,.1f}); largest {top}; "
                      f"arena base {arena(b, s) / MIB:,.0f} MiB, new {arena(n, s) / MIB:,.0f} MiB")


if __name__ == "__main__":
    main()
