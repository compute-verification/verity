"""``cuda/arena_bytes.h``, the circuit prover's device arena, against ``arena_layout.py``'s replay of every allocation a
session makes: over m 21 to 37 (the prover takes 22 to 35), k_log 20 to 27, M1 and M0, keep on and off, and each shape's
statement extremes, the arena it sizes holds the session with at least half of FC_ARENA_MARGIN to spare. The header is
compiled by the host's C compiler, so the sizes checked are the prover's own."""
from __future__ import annotations

import ctypes
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from arena_layout import MIB, HB, Shape, need, rep_events, replay, session_events, shapes, worst

HEADER = Path(__file__).resolve().parents[1] / "cuda" / "arena_bytes.h"
WRAPPER = """#include "arena_bytes.h"
size_t fc_arena_bytes_c(const FcArenaShape* s) { return fc_arena_bytes(s); }
size_t fc_arena_margin_c(void) { return FC_ARENA_MARGIN; }
"""


class CShape(ctypes.Structure):
    _fields_ = [("m", ctypes.c_int), ("k_log", ctypes.c_int), ("sub_log", ctypes.c_int), ("initial_k", ctypes.c_int),
                ("recursive_steps", ctypes.c_int), ("log_inv_rates", ctypes.POINTER(ctypes.c_int)),
                ("recursive_ks", ctypes.POINTER(ctypes.c_int)), ("queries", ctypes.POINTER(ctypes.c_int)),
                ("hash_bytes", ctypes.c_int), ("zk", ctypes.c_int),
                ("t_pad", ctypes.c_longlong), ("fold_words", ctypes.c_longlong)]


@pytest.fixture(scope="module")
def lib(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    assert cc, "no C compiler: install gcc (build-essential)"
    d = tmp_path_factory.mktemp("arena")
    (d / "wrap.c").write_text(WRAPPER)
    r = subprocess.run([cc, "-std=c99", "-O1", "-shared", "-fPIC", f"-I{HEADER.parent}", "-o", str(d / "arena.so"),
                        str(d / "wrap.c")], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr[-4000:]
    dll = ctypes.CDLL(str(d / "arena.so"))
    dll.fc_arena_bytes_c.argtypes = [ctypes.POINTER(CShape)]
    dll.fc_arena_bytes_c.restype = ctypes.c_size_t
    dll.fc_arena_margin_c.restype = ctypes.c_size_t
    return dll


def arena(lib: ctypes.CDLL, s: Shape) -> int:
    ik, rates, ks, _ = s.lig()
    r, k = (ctypes.c_int * len(rates))(*rates), (ctypes.c_int * max(len(ks), 1))(*ks)
    q = (ctypes.c_int * len(s.queries))(*s.queries)
    c = CShape(s.m, s.k_log, s.sub_log, ik, len(ks), r, k, q, HB, int(s.zk), s.t_pad if s.zk else 0,
               sum(1 << t for t in s.types))
    return lib.fc_arena_bytes_c(ctypes.byref(c))


def test_every_shape_fits_with_margin(lib: ctypes.CDLL) -> None:
    spare = lib.fc_arena_margin_c() // 2
    short, cases = [], 0
    for s0 in shapes():
        for s in worst(s0):
            cap = arena(lib, s)
            for keep in (True, False):
                ev = session_events(s, keep)
                cases += 1
                if not replay(ev, cap).fits or cap - need(ev, cap // MIB) * MIB < spare:
                    short.append((s, keep, cap // MIB))
    assert cases > 1000
    assert not short, short[:10]


def test_the_cut_overflows_are_seen() -> None:
    """The two layouts the lane groups and keep-L0 removed, as the replay sees them at a full statement: before the lane
    groups, level 0's M1 commitment held every witness lane's codeword beside a kept rep 0's copy of z (8.5625 witness
    sizes); before keep-L0, a keeping rep 0 held a, b and z's copy through Ligerito (about 9.6 under M0, the recursive
    levels' salts copied beside level 0's folds included)."""
    W = (1 << 28) * 16
    m1 = Shape(35, 25, True, lane_groups=False, keep_ab=True, l0_rows=False)
    commit = list(rep_events(m1, "keep"))
    commit = commit[:[i for i, e in enumerate(commit) if e[:2] == ("r", "cw_w")][0]]
    assert replay(commit, 1 << 46).live_peak / W == pytest.approx(8.5625, abs=0.001)
    m0 = Shape(35, 25, False, keep_ab=True)
    assert 9.6 <= replay(list(rep_events(m0, "keep")), 1 << 46).live_peak / W < 9.7
    now = replay(list(rep_events(replace(m1, lane_groups=True, keep_ab=False), "keep")), 1 << 46).live_peak / W
    assert now < 7
