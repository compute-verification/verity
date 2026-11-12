"""pearl_c/hash_wide.cuh on the host (pearl_c/hash_host.cpp, built with g++): the row digests of rows past hash_h3.cuh's
RS_MAX_SEGS segments, up to RS_WIDE_MAX_SEGS, launched as pearl_c_sm90/run.py's `row_digests` launches them.  Both forms
(from a width's header states, which s512_heads_w writes, and with no states, each thread computing its header) are held to
the scheme's reference, `pearl_c.row_digest` (``sha512/row-seg/v1``), so to each other, on both sides of every bound: 1, 128,
129, 130, 501 and 1,024 segments.  Every output is poisoned (0xA5) before its launch, with a row's room past the last left
as it was; the no-write control, the poison with no rows, fails every comparison; and a launch whose k is not a positive
multiple of 256 of at most RS_WIDE_MAX_SEGS segments traps before it writes.  Then a tree of rows of 501 segments (the
width of an output whose rows are 128,256 words), hashed a chunk of rows at a time, is `audit.commit_rows`'s."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verity.commitments.merkle import tree_levels
from verity.protocols.pouw.prover.schemes import pearl_c as C
from verity.protocols.pouw.verifier import audit
from verity.protocols.pouw.verifier.audit import ROW_SCHEMA, rows_domain

HERE = Path(__file__).resolve().parent
POUW = HERE.parent


def load(name: str, path: Path):
    """The module at `path` under its own name: other trees in the suite have a run.py too."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


R = load("pearl_c_sm90_run", POUW / "pearl_c_sm90" / "run.py")
#: the host `Dev` (`Host`), its build of hash_host.cpp (`host`) and the pinned row (`h3_row`): hash_h3.cuh's host test's
HH = load("pearl_c_hash_host_test", HERE / "test_pearl_c_hash_host.py")
host = HH.host
SOURCE = (POUW / "pearl_c" / "hash_wide.cuh").read_text()
POISON = 0xA5
#: both sides of s512_rows' bound (128), an odd and an even count past it, the width of a row of 128,256 words, and the widest
SEGS = (1, 128, 129, 130, 501, 1024)


def poisoned(host, n: int) -> int:
    p = host.alloc(n)
    host.bufs[p][:] = POISON
    return p


def args(p=(), i=()):
    a = R.Args()
    for j, v in enumerate(p):
        a.p[j] = v
    for j, v in enumerate(i):
        a.i[j] = v
    return a


def states(host, k: int) -> int:
    """A width's header states as the pipeline keeps them (`rs_heads_w`): s512_heads_w's, with room for one state more."""
    s = k // 256
    p = poisoned(host, 64 * (s + 2))
    host.launch("s512_heads_w", R.cdiv(s + 1, 128), 128, p=(p,), i=(k,))
    assert host.read(p + 64 * (s + 1), 64) == bytes([POISON]) * 64          # s + 1 states, and nothing past them
    return p


def rows_of(k: int, count: int) -> list[bytes]:
    rng = np.random.default_rng(k)
    return [HH.h3_row(k)] + [rng.integers(0, 256, 4 * k, dtype=np.uint8).tobytes() for _ in range(count - 1)]


def test_the_wide_bounds_are_the_runners():
    assert R.RS_WIDE_MAX_SEGS == int(re.search(r"\bRS_WIDE_MAX_SEGS = (\d+)", SOURCE).group(1)) == 1024
    assert R.RS_MAX_SEGS == 128 and max(SEGS) == R.RS_WIDE_MAX_SEGS
    kernels = set(re.findall(r'extern "C" __global__ void (?:__launch_bounds__\([^)]*\) )?(\w+)\(', SOURCE))
    assert kernels == {"s512_heads_w", "s512_row_segs_w", "s512_folds_w"} == {n for n in R.RS_KERNELS if n.endswith("_w")}
    # every kernel takes its segments from the one function that traps
    assert SOURCE.count("= rs_wide_segs(") == len(kernels)


@pytest.mark.parametrize("form", ["states", "none"])
@pytest.mark.parametrize("segs", SEGS)
def test_wide_row_digests_are_row_seg_v1(host, segs, form):
    """s512_row_segs_w then s512_folds_w, a thread a segment and then a thread a row, are `pearl_c.row_digest` of every row
    (the pinned row and two random ones), from the header states and with none; the fold of an odd count ends in the
    digests' block, of an even one in a padding block.  A row's room past the last, in the scratch and in the output, is
    left as it was, and with no rows nothing is written at all."""
    k, count = 256 * segs, 3
    xs = rows_of(k, count)
    x, heads = host.put(b"".join(xs)), states(host, k) if form == "states" else 0
    scratch, out = poisoned(host, 64 * segs * (count + 1)), poisoned(host, 64 * (count + 1))
    want = b"".join(C.row_digest(r) for r in xs)
    poison = bytes([POISON]) * 64

    def launch(rows: int) -> None:
        host.launch("s512_row_segs_w", R.cdiv(count * segs, R.RS_FAST_NT), R.RS_FAST_NT, p=(x, scratch, heads), i=(rows, k))
        host.launch("s512_folds_w", R.cdiv(count, R.RS_FOLD_NT), R.RS_FOLD_NT, p=(scratch, out, heads), i=(rows, k))

    launch(0)                                                # the no-write control: the grids of three rows, no rows
    assert host.read(scratch, 64 * segs * (count + 1)) == poison * (segs * (count + 1))
    assert host.read(out, 64 * (count + 1)) == poison * (count + 1) and poison * count != want
    launch(count)
    assert host.read(out, 64 * count) == want
    assert host.read(out + 64 * count, 64) == poison and host.read(scratch + 64 * segs * count, 64 * segs) == poison * segs


@pytest.mark.parametrize("sha", ["fast", "ref"])
@pytest.mark.parametrize("segs", SEGS)
def test_row_digests_take_every_width_to_the_widest(host, segs, sha):
    """run.py's `row_digests` on the host: up to RS_MAX_SEGS segments today's launches (s512_rows, or s512_row_segs and
    s512_folds from hash_fast.cuh's tables), past them hash_wide.cuh's two, from the pipeline's states (sha "fast") or from
    none ("ref"): the same bytes on both sides of 128, `pearl_c.row_digest`'s."""
    k, count = 256 * segs, 2
    xs = rows_of(k, count)
    launched = []
    dev = SimpleNamespace(launch=lambda name, *a, **kw: (launched.append(name), host.launch(name, *a, **kw))[1])
    if sha == "fast":
        HH.fast_heads(host)
    b = {"rs_heads_w": states(host, k)} if sha == "fast" and segs > R.RS_MAX_SEGS else {}
    out = poisoned(host, 64 * count)
    R.Pipeline.row_digests(SimpleNamespace(dev=dev, k=k, sha=sha, b=b), host.put(b"".join(xs)), out, poisoned(host, 64 * segs * count),
                           count)
    assert host.read(out, 64 * count) == b"".join(C.row_digest(r) for r in xs)
    wide = ["s512_row_segs_w", "s512_folds_w"]
    assert launched == (wide if segs > R.RS_MAX_SEGS else ["s512_row_segs", "s512_folds"] if sha == "fast" else ["s512_rows"])


@pytest.mark.parametrize("name,slot", [("s512_heads_w", 0), ("s512_row_segs_w", 1), ("s512_folds_w", 1)])
@pytest.mark.parametrize("k", [0, -256, 255, 1000, 33000, 262400, 1 << 30])
def test_a_wide_launch_traps_on_a_width_it_cant_take(host, name, slot, k):
    """Each kernel traps, before it reads or writes, unless k is a positive multiple of 256 of at most RS_WIDE_MAX_SEGS
    segments: no whole segments, a negative k, part of a segment, and one segment and many past the widest.  At the widest
    the same launch runs."""
    widest = 256 * R.RS_WIDE_MAX_SEGS
    room = 64 * (R.RS_WIDE_MAX_SEGS + 1)
    src, out = host.alloc(max(4 * widest, room)), poisoned(host, room)
    p = (out,) if name == "s512_heads_w" else (src, out, 0)              # no states: the kernel's own headers
    for width, rc in ((k, 2), (widest, 0)):
        a = args(p=p, i=(width,) if slot == 0 else (1, width))
        assert host.lib.launch(name.encode(), 1, 128, ctypes.byref(a)) == rc
        assert (host.read(out, room) == bytes([POISON]) * room) == (rc == 2)


def test_a_tree_of_wide_rows_is_the_verifiers(host):
    """A tree over 129 rows of 128,256 words (501 segments a row), its rows hashed 50 at a time into one buffer of digests
    through one chunk's scratch, as a caller with more rows than it holds at once hashes them: the row digests by
    `row_digests` from the pipeline's header states, then `hm_level0` and the levels.  Every level is the frame-v3-sha512
    tree `audit.commit_rows` builds over the rows under their hm96 salts, its root the `MerkleTree`'s."""
    n, count, chunk, nonce = 128256, 129, 50, (1 << 61) + 5
    segs = n // 256
    rng = np.random.default_rng(n)
    rows = [rng.integers(0, 256, 4 * n, dtype=np.uint8).tobytes() for _ in range(count)]
    salt_key, binding = bytes(range(64, 96)), hashlib.sha256(b"wide rows").digest()
    dom = rows_domain(binding, 3, count, audit.HM96)
    width = R.pow2(count)
    keys = {"dom_b": host.put(dom.domain_id), "salt_key": host.put(salt_key), "hm": host.put(audit.hm96_kernel_constants())}
    HH.fast_heads(host)
    p = SimpleNamespace(dev=host, k=n, sha="fast", keys=keys,
                        b={"tree_b": poisoned(host, 64 * (2 * width - 1)), "rs_heads_w": states(host, n)})
    dig, seg = poisoned(host, 64 * count), poisoned(host, 64 * segs * chunk)
    for _, f in R.Pipeline.tree_setup(p, "b", count, "dom_b"):
        f()
    for r in range(0, count, chunk):
        part = rows[r:r + chunk]
        x = host.put(b"".join(part))
        R.Pipeline.row_digests(p, x, dig + 64 * r, seg, len(part))
        host.free(x)
    assert host.read(dig, 64 * count) == b"".join(C.row_digest(r) for r in rows)
    R.Pipeline.hm_level0(p, "b", dig, count, "dom_b", nonce=R.nonce_args(nonce))
    R.Pipeline.levels(p, "b", count, "dom_b")
    salted = [v + audit.hm96_salt(salt_key, nonce, i) for i, v in enumerate(rows)]
    want = b"".join(b"".join(level) for level in tree_levels(dom, [dom.leaf(i, i, ROW_SCHEMA, v) for i, v in enumerate(salted)]))
    assert host.read(p.b["tree_b"], 64 * (2 * width - 1)) == want
    assert want[-64:] == audit.commit_rows(binding, 3, rows, audit.HM96, salt_key, nonce).commitment.root
